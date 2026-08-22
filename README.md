# 基于 RAG 的智能教育问答系统

面向教育知识库的混合问答系统，结合结构化数据库检索、BM25、向量检索、重排和大语言模型生成，并提供 FastAPI、WebSocket 与前端交互接口。

## 核心能力

- PDF、Word、PPT、图片等教育文档加载与 OCR
- 中文递归切分与模型辅助切分
- 父子分块、向量检索与候选重排
- BM25 与 MySQL 结构化知识检索
- Redis 会话缓存与多轮问答
- 查询分类和动态检索策略选择
- FastAPI、WebSocket 和前端页面

## 项目结构

```text
base/                       # 配置与日志
mysql_qa/                   # MySQL、Redis、BM25 问答链路
rag_qa/core/                # 文档处理、检索、分类与 RAG 核心
rag_qa/edu_document_loaders # 多格式文档加载与 OCR
rag_qa/edu_text_spliter/    # 中文文本切分
rag_qa/static/              # Web 界面
app.py                      # 主 API 服务
new_main.py                 # 集成问答编排
```

## 快速开始

```bash
python -m venv .venv
pip install -r requirements.txt
cp config.example.ini config.ini
uvicorn app:app --reload
```

Windows PowerShell 可使用 `Copy-Item config.example.ini config.ini`。

## 数据与模型说明

为保护隐私并控制仓库体积，本仓库不包含原始/处理后数据集、知识库文件、本地模型权重、向量索引、训练检查点和运行日志。仓库保留了数据加载、清洗、切分、索引构建、检索和评估代码；运行时请自行准备数据与模型。
