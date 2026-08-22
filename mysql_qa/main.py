# 导入 MySQL 客户端
import os.path
import sys

from db.MySQLClient import MySQLClient
# 导入 Redis 客户端
from cache.RedisClient import RedisClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search

# 导入日志
from base import logger

# 导入时间库
import time

class MySQLQASystem:
    def __init__(self):
        pass
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        pass
        '''
            打印耗时，就是一次请求的响应时间
        '''
        # 查询 MySQL 系统: 开始时间
        startTime = time.time()
        # 记录查询信息
        logger.info(f'用户要查询的问题为：{query}')
        # 执行 BM25 搜索
        result = self.bm25_search.search(query)
        logger.info(f'查询结果为：{result}')
        # 结束时间
        endTime = time.time()
        # 记录处理时间
        logger.info(f'耗时为：{(endTime-startTime):.2f}')
        # 返回答案
        return result

def main():
    logger.info('前端模拟页面初始化......')
    # 初始化 MySQL 系统
    mysql_system = MySQLQASystem()
    # 打印欢迎信息
    print("\n欢迎使用 MySQL 问答系统！")
    print("输入查询进行问答，输入 'exit' 退出。")
    # 循环
    while True:
        # 获取用户输入
        line = input('王，您好，请输入您的问题：')
        # 执行查询
        answer = mysql_system.query(line)
        # 打印答案
        logger.info(f'最终的查询结果为：{answer}')
'''
    假设main方法就是前端页面
'''
if __name__ == "__main__":
    # 运行主程序
    main()
