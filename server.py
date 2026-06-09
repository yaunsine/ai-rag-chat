# rag_api.py
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
RAG API Service with Streaming Support
"""

import os
import time
import asyncio
import logging
import sqlite3
import uuid
from typing import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from modelscope import snapshot_download
from pathlib import Path
import argparse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Response

from config import (
    load_config, get_llm_config, get_aliyun_config, get_embedding_config,
    get_data_config, get_server_config, get_logging_config,
    get_documents_dir, get_db_path, get_modelscope_cache,
)

# 加载配置
config = load_config()

# 文档目录
documents_dir = get_documents_dir()

# 模型缓存位置
os.environ['MODELSCOPE_CACHE'] = get_modelscope_cache()

# 数据库路径
DB_PATH = get_db_path()


# 配置日志
logging.basicConfig(level=getattr(logging, get_logging_config().get("level", "INFO")))
logger = logging.getLogger(__name__)

# 全局变量存储索引
index = None

# 模板配置
templates = Jinja2Templates(directory="templates")

# 初始化模型
def init_models():
    """初始化模型配置"""
    llm_cfg = get_llm_config()
    aliyun_cfg = get_aliyun_config()
    embedding_cfg = get_embedding_config()

    # 模型下载
    model_dir = snapshot_download(embedding_cfg.get("model", "BAAI/bge-small-zh-v1.5"))

    # 设置阿里云环境变量
    os.environ["ALIYUN_ACCESS_KEY_ID"] = aliyun_cfg.get("access_key_id", "")
    os.environ["ALIYUN_ACCESS_KEY_SECRET"] = aliyun_cfg.get("access_key_secret", "")
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    llm = DeepSeek(
        model=llm_cfg.get("model", "deepseek-reasoner"),
        api_key=llm_cfg.get("api_key", "")
    )

    # 配置大模型
    Settings.llm = llm

    # 配置嵌入模型
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_dir,
        trust_remote_code=True,
        device=embedding_cfg.get("device", "cpu"),
        cache_folder=embedding_cfg.get("cache_folder", "./models")
    )


def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# 请求数据模型
class QueryRequest(BaseModel):
    query: str
    session_id: str = None

# 响应数据模型
class QueryResponse(BaseModel):
    response: str
    session_id: str

# 流式响应数据模型
class StreamQueryRequest(BaseModel):
    query: str
    session_id: str = None
    conversation_id: str = None

# 索引更新请求模型
class UpdateIndexRequest(BaseModel):
    force: bool = False

# 对话请求模型
class CreateConversationRequest(BaseModel):
    title: str = "新对话"

class RenameConversationRequest(BaseModel):
    title: str

class AddMessageRequest(BaseModel):
    role: str
    content: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理"""
    global index
    logger.info("正在初始化数据库...")
    init_db()

    logger.info("正在初始化模型...")
    init_models()

    logger.info("正在加载文档并构建索引...")
    start_time = time.time()
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    end_time = time.time()
    logger.info(f"索引构建完成，耗时: {end_time - start_time:.2f}秒")

    yield

    # 应用关闭时的清理工作
    logger.info("应用正在关闭...")


app = FastAPI(
    title="RAG API Service",
    description="基于LlamaIndex的RAG问答系统API",
    lifespan=lifespan
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加CORS中间件以允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    """处理 CORS 预检请求"""
    response_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "3600",
    }
    return Response(headers=response_headers)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "RAG API Service is running"}


@app.get("/chat")
async def chat_page(request: Request):
    """聊天界面页面"""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """普通查询接口"""
    global index
    if not index:
        raise HTTPException(status_code=503, detail="索引尚未准备好")

    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(request.query)
        return QueryResponse(response=str(response), session_id=request.session_id or "")
    except Exception as e:
        logger.error(f"查询出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-query")
async def stream_query_document(request: StreamQueryRequest):
    """流式查询接口"""
    global index
    if not index:
        raise HTTPException(status_code=503, detail="索引尚未准备好")

    async def generate_response() -> AsyncGenerator[str, None]:
        full_response = []
        try:
            query_engine = index.as_query_engine(streaming=True)
            response = query_engine.query(request.query)

            # 流式输出响应
            for token in response.response_gen:
                full_response.append(token)
                yield token
                await asyncio.sleep(0.01)

            # 查询完成后，如果提供了 conversation_id，保存消息到数据库
            if request.conversation_id:
                try:
                    now = time.time()
                    conn = get_db()
                    cursor = conn.cursor()
                    # 保存用户消息
                    cursor.execute(
                        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                        (request.conversation_id, "user", request.query, now)
                    )
                    # 保存 AI 回复
                    cursor.execute(
                        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                        (request.conversation_id, "assistant", "".join(full_response), now)
                    )
                    # 更新对话时间戳
                    cursor.execute(
                        "UPDATE conversations SET updated_at = ? WHERE id = ?",
                        (now, request.conversation_id)
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"保存消息到数据库出错: {str(e)}")
        except Exception as e:
            logger.error(f"流式查询出错: {str(e)}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/plain")


@app.post("/update-index")
async def update_index(request: UpdateIndexRequest = None):
    """更新索引接口"""
    global index

    try:
        logger.info("开始更新索引...")
        start_time = time.time()

        documents = SimpleDirectoryReader(documents_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)

        end_time = time.time()
        logger.info(f"索引更新完成，耗时: {end_time - start_time:.2f}秒")

        return {"message": "索引更新成功", "documents_count": len(documents)}
    except Exception as e:
        logger.error(f"索引更新出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 对话 API ====================

@app.get("/api/conversations")
async def get_conversations():
    """获取所有对话列表（按更新时间倒序）"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}
        for row in rows
    ]


@app.post("/api/conversations")
async def create_conversation(request: CreateConversationRequest):
    """创建新对话"""
    conv_id = str(uuid.uuid4())
    now = time.time()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, request.title, now, now)
    )
    conn.commit()
    conn.close()
    return {"id": conv_id, "title": request.title, "created_at": now, "updated_at": now}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除对话及其所有消息"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return {"message": "对话已删除"}


@app.put("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, request: RenameConversationRequest):
    """重命名对话"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")
    now = time.time()
    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (request.title, now, conv_id)
    )
    conn.commit()
    conn.close()
    return {"message": "重命名成功", "title": request.title}


@app.get("/api/conversations/{conv_id}/messages")
async def get_conversation_messages(conv_id: str):
    """获取某对话的所有消息"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
        for row in rows
    ]


@app.post("/api/conversations/{conv_id}/messages")
async def add_message_to_conversation(conv_id: str, request: AddMessageRequest):
    """手动添加消息到对话"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")
    now = time.time()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, request.role, request.content, now)
    )
    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conv_id)
    )
    conn.commit()
    conn.close()
    return {"message": "消息已添加"}


def get_row_params():
    server_cfg = get_server_config()
    parser = argparse.ArgumentParser(description='RAG API Service')
    parser.add_argument("--host", default=server_cfg.get("host", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=server_cfg.get("port", 8001), help="Port to bind to")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_row_params()
    if args.config:
        from config import reload_config
        reload_config(args.config)
    uvicorn.run(app, host=args.host, port=args.port)
