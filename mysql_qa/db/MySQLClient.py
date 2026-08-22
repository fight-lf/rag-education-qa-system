# db/mysql_client.py
# 导入 MySQL 连接库
import os.path
import sys

import pymysql  # 读写mysql
# 导入pandas
import pandas as pd  # 读取csv文件

# 把当前包的路径也添加到sys里面
db_dir = os.path.dirname(__file__)
mysql_qa_dir = os.path.dirname(db_dir)
qa_dir = os.path.dirname(mysql_qa_dir)
sys.path.insert(0, mysql_qa_dir)
sys.path.insert(0, qa_dir)

# 导入配置和日志
from base import Config, logger

class MySQLClient:
    # 初始化连接对象
    def __init__(self):
        config = Config()
        try:
            # 创建客户端连接对象
            self.connection = pymysql.Connection(
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                host=config.MYSQL_HOST,
                database=config.MYSQL_DATABASE,
                port=3306
            )
            # 获取cursor对象,对数据库进行curd ，cursor:游标
            self.cursor = self.connection.cursor()
            logger.info('mysql初始化成功。。。')
        except pymysql.MySQLError as e:
            logger.error(f'初始化mysql异常：{e}')

    # 创建表
    def create_table(self):
        logger.info('开始创建mysql表')
        try:
            sql = '''
                    create table  if not exists  jpkb2(
                        id           int auto_increment primary key,
                        subject_name varchar(20)   ,
                        question     varchar(1000) ,
                        answer       varchar(1000)
                    )
            '''
            self.cursor.execute(sql)
            logger.info('mysql创建表成功。。。')
        except pymysql.MySQLError as e:
            logger.error(f'mysql创建表异常：{e}')

    # 往mysql插入文本数据
    def insert_data(self, csv_path):

        try:
            # 读取csv文本数据
            datas = pd.read_csv(csv_path)
            # print(datas,type(datas))

            # sql插入语句
            sql = 'insert into jpkb2(subject_name, question, answer) values (%s,%s,%s)'
            # 遍历文本数据
            for id, line in datas.iterrows():
                # print(type(line))
                print(line['学科名称'], line['问题'], line['答案'])
                # 替换占位符  ：%s
                self.cursor.execute(sql, (line['学科名称'], line['问题'], line['答案']))
            # 执行sql插入操作,执行提交
            self.connection.commit()
            logger.info('mysql插入数据成功。。。')
        except pymysql.MySQLError as e:
            logger.error(f'mysql插入数据异常：{e}')

    # 查询所有的问题
    def fetch_questions(self):
        pass
        # 获取所有问题
        try:
            sql = 'select question from jpkb2'
            self.cursor.execute(sql)  # sql执行
            questions = self.cursor.fetchall()
            logger.info('获取所有的问题查询成功。。。')
            return questions
        except pymysql.MySQLError as e:
            logger.error(f'数据库问题查询异常：{e}')
            return None

    # 根据问题查询答案，查询问题就是过滤条件（where）
    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 写法1
            # sql = f"select answer from jpkb2 where question={question}"
            # self.cursor.execute(sql)

            # 写法2
            sql = 'select answer from jpkb2 where question=%s'
            self.cursor.execute(sql, question)
            row = self.cursor.fetchone()
            logger.info('数据库答案查询成功。。。')
            return row[0] if row else None
        except pymysql.MySQLError as e:
            logger.error(f'数据库答案查询异常：{e}')
            return None

    # 关闭连接
    def close(self):
        try:
            self.connection.close()
            logger.info('数据库关闭连接成功。。。')
        except pymysql.MySQLError as e:
            logger.error(f'数据库关闭连接异常：{e}')


if __name__ == '__main__':
    sql_client = MySQLClient()
    logger.info('mysql初始化。。。。')
    # sql_client.create_table() # 创建表
    file = '../data/JP学科知识问答.csv'
    # sql_client.insert_data(file)
    # 问题查询
    # questions = sql_client.fetch_questions()
    # for doc in questions:
    #     # print(doc) # ('如何打开虚拟机',)
    #     print(doc[0])
    # logger.info(questions)

    # 答案查询
    answer = sql_client.fetch_answer('lxml的tree报错')
    logger.info(answer[0])
    sql_client.close()
