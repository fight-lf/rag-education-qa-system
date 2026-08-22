# retrieval/bm25_search.py
# 导入 BM25 算法
import os.path
import sys

import jieba
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np

# 把当前包的路径也添加到sys里面
db_dir = os.path.dirname(__file__)
mysql_qa_dir = os.path.dirname(db_dir)
qa_dir = os.path.dirname(mysql_qa_dir)
sys.path.insert(0, mysql_qa_dir)
sys.path.insert(0, qa_dir)

# 导入日志
from base import logger
# 导入文本预处理
from utils.preprocess import preprocess_text

class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志
        logger.info('bm25检索初始化.....')
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()


    def _load_data(self):
        logger.info('bm25和redis开始初始化数据.......')
        '''
        :desc : 加载数据(原始问题，分词问题)到redis,初始化bm25
        :return:
        '''
        # 加载数据 到redis
        original_key = "qa_original_questions2"  # redis里面数据的key (原始数据)
        tokenized_key = "qa_tokenized_questions2"  # redis里面数据的key（分词之后的数据）
        # 从 Redis 获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        # 从 Redis 获取分词问题
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 如果 Redis 中没有数据，从 MySQL 加载
        # if original_questions: # 如果original_questions == None ,bool = False
        if not self.original_questions or not self.tokenized_questions: # 空值判断，如果能够进入代码块，说明redis里面没有数据

            # 从 MySQL 获取问题
            self.original_questions = self.mysql_client.fetch_questions()
            # 记录无问题 -> 警告
            if not self.original_questions:
                logger.info('查询无数据')
            # 对每一行问题分词
            self.tokenized_questions = [preprocess_text(doc[0]) for doc in self.original_questions]
            # 获取原始问题
            self.original_questions = [doc[0] for doc in self.original_questions]

            # 存储分词问题和原始问题到 Redis
            self.redis_client.set_data(original_key,self.original_questions) # 原始问题
            self.redis_client.set_data(tokenized_key,self.tokenized_questions) # 分词问题

        # 初始化 BM25 模型:BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized_questions)
        # 记录 BM25 初始化成功
        logger.info('BM25和Redis初始化成功.....')

    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数 -> 结果数据范围【0，1】
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        # 搜索查询 ->判断
        if not query:
            # 记录无效查询
            logger.info('无效查询')
            # 返回 None
            return None
        # 检查 Redis 缓存:
        answer = self.redis_client.get_data(f'answer:{query}')
        # 返回缓存答案
        if answer:
            logger.info(f'查询redis命中结果，并返回成功,结果如下：{answer}')
            return answer
        '''
            如果缓存没有数据，那么只有进行bm25相似度检索，用户的问题和mysql里面问题的相似度
        '''
        # 查询-> 分词
        query_tokenizer = preprocess_text(query)
        # 计算 BM25 分数
        scores = self.bm25.get_scores(query_tokenizer)
        # 计算 Softmax 分数
        softmax_scores = self._softmax(scores)
        # 获取最高分索引
        max_id = softmax_scores.argmax()
        # 获取最高分
        max_score = softmax_scores[max_id]
        # 检查是否超过阈值
        if max_score >threshold: # 用户的查询和原始问题高度相似
            # 获取原始问题
            original_question = self.original_questions[max_id] #把查询到的问题当做查询mysql答案的过滤条件
            # 获取答案:mysql
            fetch_answer = self.mysql_client.fetch_answer(original_question)
            if fetch_answer:
                # 缓存答案 key: answer:{query}
                self.redis_client.set_data(f'answer:{query}',fetch_answer)
                # 记录搜索成功
                logger.info('用户查询问题返回成功....')
                # 返回答案
                return fetch_answer
        # 记录无可靠答案
        logger.info('无可靠答案，请走本地知识库！')
        # 返回 None
        return None

if __name__ == '__main__':

    from cache.RedisClient import RedisClient
    from db.MySQLClient import MySQLClient

    client_redis = RedisClient()
    client_sql = MySQLClient()
    bm25Search = BM25Search(client_redis,client_sql)
    # result = bm25Search.search('用上下文管理器实现函数运行时间的计算?')
    # result = bm25Search.search('333lxml的tree报错')
    line = input('请输入您的问题')
    result = bm25Search.search(line)
    logger.info(result)
