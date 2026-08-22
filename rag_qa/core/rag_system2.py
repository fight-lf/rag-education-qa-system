# -*-coding:utf-8-*-
# core/rag_system.py 源码
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  ###  core 目录，供 prompts、query_classifier 等同目录模块导入
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from prompts2 import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base import logger, Config
from query_classifier import QueryClassifier  #   导入查询分类器
from strategy_selector import StrategySelector  #   导入策略选择器
from vector_store import VectorStore # 导入向量数据库对象

conf = Config()

#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        classifier_path = os.path.join(rag_qa_path, 'core', 'bert_query_classifier')
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()
    '''
        代码优化：
    '''
    # 1. 生成答案
    def generate_answer(self, query, source_filter=None,history=None):
        #   耗时 ： 开始时间 - 结束时间
        start_time = time.time()
        # 验证历史
        if history is not None and not isinstance(history, list):
            logger.warning(f'无效的历史格式：{type(history)},忽略历史')
            history = []
        elif history:
            history = history[-5:]  # 严格只取出最近5轮对话
        # 构造历史的上下文：
        history_context = ''
        if history:
            history_context = "\n".join([f"Q:{h['question']}\nA:{h['answer']}" for h in history])
            logger.info(f'使用对话历史：{history_context[:50]}')

        #   查询问题类型 ->意图识别
        query_category = self.query_classifier.predict_category(query)
        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == '通用知识':
            # 设置提示词,变量替换
            #  input_variables=["context", "question", "phone"],
            prompt_input = self.rag_prompt.format(context='', question=query, phone=conf.CUSTOMER_SERVICE_PHONE,history=history_context)
            # 模型调用
            answer = self.llm(prompt_input)
            return answer
        #   否则，进行 RAG 检索并生成答案
        logger.info('查询为专业咨询,执行RAG流程....')
        #   选择检索策略
        template = self.strategy_selector.strategy_prompt_template
        # 格式化模版,替换变量
        template_input = template.format(query=query)
        # LLM模型调用,由大模型帮我们选择具体的查询策略
        strategy = self.llm(template_input)
        logger.info(f'策略选择器，选择的查询策略为：{strategy}')
        # 根据检索策略,选择具体的查询策略,来执行后续的处理
        #  检索相关文档: retrieve_and_merge
        docs = self.retrieve_and_merge(query, source_filter, strategy) # 返回的是一个列表：docs

        # 遍历文档, 使用换行符拼接文档 -> 把final_docs 里面的多条语句,通过\n,拼接成一个context
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        #   构造 Prompt，调用大语言模型生成答案
        prompt_input = self.rag_prompt.format(context=context,
                                              question=query,
                                              history=history_context,
                                              phone=conf.CUSTOMER_SERVICE_PHONE)

        # 模型调用
        answer = self.llm(prompt_input)
        #   记录查询处理完成的日志
        logger.info(f'RAG本地检索完成！！！')
        # 结束时间
        end_time = time.time()
        # 耗时
        logger.info(f'一次请求耗时时间为：{end_time - start_time :.2f}')
        # 返回结果
        return answer

    # 2.策略选择,根据传输的策略,来选择具体的查询策略,并做后续的向量数据检索
    # 并基于具体的查询策略,从向量库里面返回Top-k
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):

        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            logger.info('无效查询策略')
            return []
        # 根据检索策略选择不同的检索方式
        ranked_chunks = [] # 初始化 ,查询向量数据库库的返回结果：TOP-K -> []
        # 回溯问题检索
        if strategy  == '回溯问题检索':
            # 1.获取格式化模板，并替换参数（query）
            # 2.LLM生成简化之后的提示词
            # 3.向量检索
            ranked_chunks = self._retrieve_with_backtracking(query, source_filter)
        # 子查询检索
        elif strategy == '子查询检索':
            ranked_chunks = self._retrieve_with_subqueries(query,source_filter)

        # 假设问题检索
        elif strategy == '假设问题检索':
            ranked_chunks = self._retrieve_with_hyde(query,source_filter)
        # 直接检索
        else:
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(query=query, source_filter=source_filter)

        # 返回数据
        return ranked_chunks


    # 3.1 假设文档提示词处理
    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")

        #   获取假设问题生成的 Prompt 模板
        prompt = RAGPrompts.hyde_prompt()
        prompt_input = prompt.format(query=query)
        #   调用大语言模型生成假设答案
        doc = self.llm(prompt_input)
        #   使用假设答案进行检索，并返回检索结果
        rerank = self.vector_store.hybrid_search_with_rerank(query=doc, source_filter=source_filter)
        return rerank

    # 3.2 提示词:子查询进行检索
    #   定义类似私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter):
        '''
             开发步骤：
                # 1.获取格式化模板，并替换参数（query）
                # 2.LLM生成简化之后的提示词
                # 3.向量检索:混合检索
        '''
        #   获取子查询生成的 Prompt 模板
        prompt = RAGPrompts.subquery_prompt()
        prompt_input = prompt.format(query=query)
        sub_queries_str = self.llm(prompt_input)  # -> str
        print(sub_queries_str)
        #   调用大语言模型生成子查询列表,并按照\n分割
        # 按照\n 把字符串分割成两个子查询 - > []
        sub_queries = sub_queries_str.split('\n')
        #   判断子查询是否有返回数据,如果无,返回[]
        if not sub_queries:
            return []
        #   初始化空列表，用于存储所有子查询的检索结果
        all_docs = []

        #   遍历每个子查询
        for doc in sub_queries:
            # 使用子查询混合检索,并添加入列表
            rerank = self.vector_store.hybrid_search_with_rerank(query=doc, source_filter=source_filter)
            # 把子查询的结果存储到all_docs
            all_docs.extend(rerank)
        #   遍历 -> dict , key = 文档内容
        dict_docs = {i.page_content:i for i in all_docs}  # Doucument
        #  获取字典values ,并转列表
        docs = list(dict_docs.values())
        # 返回文档
        return docs

    # 3.3 提示词:回溯问题
    #   定义类似私有方法，使用回溯问题进行检索

    def _retrieve_with_backtracking(self, query, source_filter):
        '''
        :author :jack
        :time : 2026-05-18
        :param query: 用户的查询问题
        :param source_filter: 查询文档的类型：【ai,java,c# .....】
        :return: 返回的是查询向量库里的TOP-k
        :description 此函数是使用RAG的回溯检索增强方法
        '''
        '''
            开发步骤：
                # 1.获取格式化模板，并替换参数（query）
                # 2.LLM生成简化之后的提示词
                # 3.向量检索:混合检索
        '''
        #   获取回溯问题生成的 Prompt 模板
        prompt = RAGPrompts.backtracking_prompt()
        # 格式化模板,参数替换
        prompt_input = prompt.format(query=query)
        #   调用大语言模型生成回溯问题
        question = self.llm(prompt_input)
        #   使用回溯问题进行混合检索，并返回检索结果
        docs = self.vector_store.hybrid_search_with_rerank(query=question,source_filter=source_filter)
        return docs

if __name__ == '__main__':
    vector_store = VectorStore()
    llm = StrategySelector().call_dashscope

    # print(llm(prompt="用Java写一个冒泡排序算法。"))
    rag_system = RAGSystem(vector_store,llm)
    # answer = rag_system.generate_answer(query='用Java写一个冒泡排序算法。', source_filter='ai')
    answer = rag_system.generate_answer(query='请问贵校的教学点在哪里？', source_filter='ai')
    # answer = rag_system.generate_answer(query='AI学科的课程大纲内容有什么', source_filter='ai')
    # 回溯问题
    # answer = rag_system._retrieve_with_backtracking(query='有100亿条数据，想把它存入milvus,请问可以吗？',source_filter='ai')
    # answer = rag_system._retrieve_with_subqueries(query='java和c各有什么特点？',source_filter='ai')
    # answer = rag_system._retrieve_with_hyde(query='AI课程里面的NLP技术有哪些？',source_filter='ai')
    print(answer)
