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
from modelscope import snapshot_download
from pathlib import Path
import argparse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Response

# 获取相对于脚本文件的路径
documents_dir = Path(__file__).parent / "data_dir"
# 模型缓存位置
os.environ['MODELSCOPE_CACHE'] = './llms'


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储索引
index = None

# 模板配置
templates = Jinja2Templates(directory="templates")

# 初始化模型
def init_models():
    """初始化模型配置"""
    # 模型下载
    model_dir = snapshot_download('BAAI/bge-small-zh-v1.5')
    
    # 设置环境变量
    os.environ["ALIYUN_ACCESS_KEY_ID"] = "2649898"
    os.environ["ALIYUN_ACCESS_KEY_SECRET"] = "sk-c39f58fedee34ddb8e5cd55f2df1813a"
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    # 配置大模型
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-c39f58fedee34ddb8e5cd55f2df1813a",
        is_chat_model=True
    )
    
    # 配置嵌入模型
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_dir,
        trust_remote_code=True,
        device="cpu",
        cache_folder="./models"
    )

# 请求数据模型
class QueryRequest(BaseModel):
    query: str
    session_id: str = None  # 可选的会话ID

# 响应数据模型
class QueryResponse(BaseModel):
    response: str
    session_id: str

# 流式响应数据模型
class StreamQueryRequest(BaseModel):
    query: str
    session_id: str = None

# 索引更新请求模型
class UpdateIndexRequest(BaseModel):
    force: bool = False  # 是否强制重新构建索引

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理"""
    global index
    logger.info("正在初始化模型...")
    init_models()
    
    logger.info("正在加载文档并构建索引...")
    start_time = time.time()
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    end_time = time.time()
    logger.info(f"索引构建完成，耗时: {end_time - start_time:.2f}秒")
    
    yield
    
    # 应用关闭时的清理工作（如果需要）
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
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    """处理 CORS 预检请求"""
    response_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "3600",
    }
    return Response(headers=response_headers)

@app.get("/")
async def root():
    """根路径"""
    return {"message": "RAG API Service is running"}

@app.get("/chat")
async def chat_page():
    """聊天界面页面"""
    return templates.TemplateResponse("chat.html", {"request": {}})

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
        try:
            query_engine = index.as_query_engine(streaming=True)
            response = query_engine.query(request.query)
            
            # 流式输出响应
            for token in response.response_gen:
                yield token
                # 添加短暂延迟以确保流式传输效果
                await asyncio.sleep(0.01)
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
        
        # 重新加载文档并构建索引
        documents = SimpleDirectoryReader(documents_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        
        end_time = time.time()
        logger.info(f"索引更新完成，耗时: {end_time - start_time:.2f}秒")
        
        return {"message": "索引更新成功", "documents_count": len(documents)}
    except Exception as e:
        logger.error(f"索引更新出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_row_params():
    parser = argparse.ArgumentParser(description='RAG API Service')
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--module", default="chat", help="Module to import")
    parser.add_argument("--prefix", default="/chat", help="API prefix")
    return parser.parse_args()

# 运行命令: uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
if __name__ == '__main__':
    args = get_row_params()
    uvicorn.run(app, host=args.host, port=args.port)
