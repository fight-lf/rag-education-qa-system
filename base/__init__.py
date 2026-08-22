# config.py和logger.py暴露出去，让其他文件夹、也能够调用，需要把路径添加到系统环境变量里面
import os,sys

base_dir = os.path.dirname(__file__)
qa_dir = os.path.dirname(base_dir)
sys.path.insert(0,base_dir)
sys.path.insert(0,qa_dir)

from config import Config
from logger import logger

logger.info('打印日志 。。。。。。')
logger.info(Config().LOG_FILE)
