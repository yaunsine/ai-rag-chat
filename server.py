# rag_api.py
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
RAG API Service with Streaming Support
"""

import os
import time
import json
import csv
import asyncio
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_index.core import Settings
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
from rag_engine import RAGEngine

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

# 全局检索引擎（Milvus 向量库 + 双路召回 + RRF 融合 + 精排 + 引用来源）
engine = RAGEngine()

# 模板配置
templates = Jinja2Templates(directory="templates")


def sse_pack(event: str, data: Dict[str, Any]) -> str:
    """SSE 事件打包：event 名 + JSON 数据"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------- 知识库浏览

KB_TEXT_EXTS = {".md", ".markdown", ".txt", ".csv", ".json", ".yaml", ".yml"}
KB_MAX_PREVIEW_BYTES = 2 * 1024 * 1024
KB_MAX_CSV_ROWS = 500


def _kb_root() -> Path:
    """知识库根目录（与 RAG 索引目录保持一致）"""
    return get_documents_dir().resolve()


def _kb_relative(path: Path) -> str:
    return path.relative_to(_kb_root()).as_posix()


def _kb_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _resolve_kb_file(rel_path: str) -> Path:
    """把前端传入的相对路径解析为知识库内的真实文件，阻止路径穿越"""
    if not rel_path or not rel_path.strip():
        raise HTTPException(status_code=400, detail="缺少文件路径")
    root = _kb_root()
    try:
        target = (root / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="非法文件路径") from exc
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="文件路径超出知识库目录")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return target


def _kb_count_files(items: List[Dict[str, Any]]) -> int:
    return sum(item["file_count"] if item["type"] == "dir" else 1 for item in items)


def _kb_tree(directory: Path, depth: int = 0) -> List[Dict[str, Any]]:
    """递归生成目录树；目录在前，文件在后，隐藏文件不展示"""
    items: List[Dict[str, Any]] = []
    if not directory.exists():
        return items
    try:
        children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
    except OSError:
        return items
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            sub_items = _kb_tree(child, depth + 1)
            if not sub_items:
                continue
            items.append({
                "type": "dir",
                "name": child.name,
                "path": _kb_relative(child),
                "file_count": _kb_count_files(sub_items),
                "children": sub_items,
            })
        elif child.is_file():
            ext = child.suffix.lower()
            try:
                stat = child.stat()
            except OSError:
                continue
            items.append({
                "type": "file",
                "name": child.name,
                "path": _kb_relative(child),
                "ext": ext.lstrip("."),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "previewable": ext in KB_TEXT_EXTS,
            })
    return items


def _kb_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:8]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


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
            metadata TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    ''')
    # 旧库升级：补 metadata 列，用于保存引用来源/检索链路，历史消息也能回显引用
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(messages)").fetchall()}
    if "metadata" not in columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
        logger.info("已为 messages 表补充 metadata 列")
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
    sources: Optional[List[Dict[str, Any]]] = None
    trace: Optional[Dict[str, Any]] = None

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
    metadata: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理"""
    logger.info("正在初始化数据库...")
    init_db()

    logger.info("正在初始化模型...")
    init_models()

    logger.info("正在连接向量库、构建双路召回与精排链路...")
    # 建索引/加载精排模型都是阻塞动作，丢到线程里，别卡住服务启动
    stats = await asyncio.to_thread(engine.setup)
    logger.info("检索链路就绪: %s", json.dumps(stats, ensure_ascii=False))

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


@app.get("/knowledge")
async def knowledge_page(request: Request):
    """知识库浏览页面"""
    return templates.TemplateResponse("knowledge.html", {"request": request})


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """普通查询接口（一次性返回答案 + 引用来源 + 检索链路）"""
    if not engine.ready:
        raise HTTPException(status_code=503, detail="索引尚未准备好")

    try:
        answer, sources, trace = await asyncio.to_thread(engine.answer, request.query)
        return QueryResponse(
            response=answer,
            session_id=request.session_id or "",
            sources=sources,
            trace=trace,
        )
    except Exception as e:
        logger.exception("查询出错")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-query")
async def stream_query_document(request: StreamQueryRequest):
    """流式查询接口（SSE）

    事件序列：
      event: retrieval  —— 双路召回/RRF/精排的链路明细（先于答案到达，前端可立即展示）
      event: token      —— 答案增量
      event: sources    —— 引用来源（文件/片段/各阶段分数/命中通道）
      event: done       —— 本轮结束（含耗时统计）
      event: error      —— 出错信息
    """
    if not engine.ready:
        raise HTTPException(status_code=503, detail="索引尚未准备好")

    async def generate_response() -> AsyncGenerator[str, None]:
        answer_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        trace: Dict[str, Any] = {}
        timings: Dict[str, Any] = {}

        try:
            async for event in engine.astream_answer(request.query):
                event_type = event.get("type")
                if event_type == "token":
                    answer_parts.append(event.get("text", ""))
                    yield sse_pack("token", {"text": event.get("text", "")})
                elif event_type == "retrieval":
                    trace = event.get("trace") or {}
                    yield sse_pack("retrieval", {"trace": trace})
                elif event_type == "done":
                    sources = event.get("sources") or []
                    trace = event.get("trace") or trace
                    timings = event.get("timings") or {}
                    yield sse_pack(
                        "sources",
                        {"sources": sources, "trace": trace, "timings": timings},
                    )
        except Exception as e:
            logger.exception("流式查询出错")
            yield sse_pack("error", {"message": str(e)})
            return

        # 落库：用户问题 + 答案 + 引用来源/链路元数据
        if request.conversation_id:
            try:
                now = time.time()
                metadata = json.dumps(
                    {"sources": sources, "trace": trace, "timings": timings},
                    ensure_ascii=False,
                )
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (request.conversation_id, "user", request.query, None, now)
                )
                cursor.execute(
                    "INSERT INTO messages (conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (request.conversation_id, "assistant", "".join(answer_parts), metadata, now)
                )
                cursor.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, request.conversation_id)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"保存消息到数据库出错: {str(e)}")

        yield sse_pack(
            "done",
            {"conversation_id": request.conversation_id, "timings": timings},
        )

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/update-index")
async def update_index(request: UpdateIndexRequest = None):
    """重建索引接口：重新切分文档、写入 Milvus、重建 BM25 索引"""
    try:
        logger.info("开始重建索引...")
        stats = await asyncio.to_thread(engine.setup, True)
        return {
            "message": "索引更新成功",
            "documents_count": stats.get("files", 0),
            "chunks": stats.get("chunks", 0),
            "vector_rows": stats.get("vector_rows", 0),
            "rebuild_seconds": stats.get("rebuild_seconds", 0),
            "retrieval": stats.get("retrieval", {}),
        }
    except Exception as e:
        logger.exception("索引更新出错")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index-status")
async def index_status():
    """检索引擎状态：向量库位置、片段数、双路召回与精排配置"""
    return {"ready": engine.ready, **engine.stats}


# ==================== 知识库浏览 API ====================

@app.get("/api/knowledge/tree")
async def knowledge_tree():
    """知识库目录树与汇总统计"""
    root = _kb_root()
    tree = _kb_tree(root)
    total_size = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                try:
                    total_size += path.stat().st_size
                except OSError:
                    pass
    return {
        "root": root.name,
        "total_files": _kb_count_files(tree),
        "total_size": total_size,
        "top_dirs": len([item for item in tree if item["type"] == "dir"]),
        "tree": tree,
    }


@app.get("/api/knowledge/file")
async def knowledge_file(path: str):
    """读取单个知识库文件，返回 Markdown/文本或 CSV 表格数据"""
    target = _resolve_kb_file(path)
    ext = target.suffix.lower()
    stat = target.stat()
    payload: Dict[str, Any] = {
        "path": _kb_relative(target),
        "name": target.name,
        "ext": ext.lstrip("."),
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "format": "unsupported",
    }

    if stat.st_size > KB_MAX_PREVIEW_BYTES:
        payload["format"] = "too_large"
        return payload

    if ext in {".md", ".markdown"}:
        content = _kb_read_text(target)
        payload.update(format="markdown", content=content, title=_kb_title(content, target.stem))
    elif ext == ".csv":
        rows: List[List[str]] = []
        truncated = False
        with target.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if index >= KB_MAX_CSV_ROWS:
                    truncated = True
                    break
                rows.append(row)
        payload.update(format="csv", rows=rows, truncated=truncated, title=target.stem)
    elif ext in KB_TEXT_EXTS:
        content = _kb_read_text(target)
        payload.update(format="text", content=content, title=_kb_title(content, target.stem))

    return payload


@app.get("/api/knowledge/search")
async def knowledge_search(q: str = "", limit: int = 50):
    """在知识库文件名和正文中做简单全文检索（不走向量召回）"""
    query = q.strip()
    if not query:
        return {"query": query, "total": 0, "results": []}

    terms = [term for term in query.lower().split() if term]
    limit = max(1, min(int(limit), 100))
    results: List[Dict[str, Any]] = []
    root = _kb_root()
    if not root.exists():
        return {"query": query, "total": 0, "results": []}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        ext = path.suffix.lower()
        if ext not in KB_TEXT_EXTS:
            continue
        try:
            if path.stat().st_size > KB_MAX_PREVIEW_BYTES:
                continue
        except OSError:
            continue
        text = _kb_read_text(path)
        lower = text.lower()
        name_lower = path.name.lower()
        if not all(term in lower or term in name_lower for term in terms):
            continue
        name_match = all(term in name_lower for term in terms)
        matches = sum(lower.count(term) for term in terms)
        positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
        start = min(positions) if positions else 0
        snippet = text[max(0, start - 60):start + 180].replace("\n", " ").strip()
        results.append({
            "path": _kb_relative(path),
            "name": path.name,
            "ext": ext.lstrip("."),
            "snippet": snippet,
            "matches": matches,
            "name_match": name_match,
        })

    results.sort(key=lambda item: (item["name_match"], item["matches"]), reverse=True)
    return {"query": query, "total": len(results), "results": results[:limit]}


@app.get("/api/knowledge/download")
async def knowledge_download(path: str):
    """下载知识库原文文件"""
    target = _resolve_kb_file(path)
    return FileResponse(target, filename=target.name, media_type="application/octet-stream")


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
    """获取某对话的所有消息（含引用来源元数据）"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, role, content, created_at, metadata FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    messages = []
    for row in rows:
        metadata = None
        if row[4]:
            try:
                metadata = json.loads(row[4])
            except Exception:
                metadata = None
        messages.append(
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "created_at": row[3],
                "metadata": metadata,
            }
        )
    return messages


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
        "INSERT INTO messages (conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            conv_id,
            request.role,
            request.content,
            json.dumps(request.metadata, ensure_ascii=False) if request.metadata else None,
            now,
        )
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
