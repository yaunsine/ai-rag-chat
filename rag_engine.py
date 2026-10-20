# -*- coding: utf-8 -*-
"""
RAG 检索引擎（Milvus 向量库 + 双路召回 + RRF 融合 + 精排 + 引用来源）
============================================================

检索链路：

    用户问题
      ├── 稠密召回：Milvus 向量库（bge 中文向量，余弦相似度）
      └── 稀疏召回：jieba 分词 + BM25（关键词精确匹配，弥补向量对型号、
                    编号、专有名词、缩写不敏感的问题）
            ↓
      RRF（Reciprocal Rank Fusion）融合：score = Σ weight_i / (k + rank_i)
            ↓
      bge-reranker 精排（Cross-Encoder 把问题与候选片段拼在一起打分）
            ↓
      取 top_n 片段 → LLM 生成答案（正文带 [编号] 引用）
            ↓
      返回引用来源（文件 / 片段 / 三路分数 / 命中通道）

向量库说明：
    Milvus 支持两种形态，代码完全一致，只换 uri：
      - 本地文件（Milvus Lite）：uri = "data_dir/milvus.db"
      - 独立部署：uri = "http://localhost:19530"（见 docker-compose.milvus.yml）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import bm25s
import jieba
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore

from config import (
    get_chunking_config,
    get_documents_dir,
    get_retrieval_config,
    get_vector_store_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 分词/停用词

# 中英文高频虚词，避免 BM25 被“的/是/怎么”之类噪音词主导
_STOPWORDS = set(
    """
    的 了 和 与 及 或 是 在 有 我 你 他 她 它 们 这 那 这个 那个 一个 我们 你们 他们
    什么 怎么 如何 为什么 哪些 哪个 可以 需要 应该 请 吗 呢 吧 啊 呀 把 被 对 从 到
    向 为 于 以 之 其 上 下 里 中 内 外 个 些 也 都 就 还 很 更 最 不 没 无 要 会 能
    该 等 并 而 但 如果 因为 所以 进行 通过 以及 相关 关于 介绍 一下 说明 告诉
    the a an of to in is are be was were what how why which and or for with on at
    by from that this these those it its as can could should would will do does
    """.split()
)

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


class _MilvusLiteNoiseFilter(logging.Filter):
    """过滤 Milvus Lite 的已知噪音日志

    Milvus Lite 是嵌入式实现，不提供 AllocTimestamp 等 RPC，pymilvus 会自行回退，
    但 grpc 内部仍把 NOT_IMPLEMENTED 记成 ERROR，本地文件模式下这类日志没有意义。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "Method not implemented" not in record.getMessage()
        except Exception:  # pragma: no cover - 拿不到消息就别过滤
            return True


def _silence_milvus_lite_noise() -> None:
    for name in ("grpc._server", "grpc._common", "grpc"):
        target = logging.getLogger(name)
        if not any(isinstance(f, _MilvusLiteNoiseFilter) for f in target.filters):
            target.addFilter(_MilvusLiteNoiseFilter())


def _exception_chain(exc: BaseException) -> List[BaseException]:
    """展开异常因果链，方便识别底层真实原因（如 Milvus Lite 的文件锁冲突）"""
    chain: List[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def tokenize_zh(text: str) -> List[str]:
    """jieba 分词 + 小停用词表过滤，供 BM25 使用（中文必须分词，按空格切会整句变一个 token）"""
    tokens: List[str] = []
    for tok in jieba.lcut(text.lower()):
        tok = tok.strip()
        if not tok or tok in _STOPWORDS or _PUNCT_RE.match(tok):
            continue
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------- 问答提示词

QA_PROMPT = """你是企业知识库问答助手。请严格依据下方【参考资料】回答问题。

回答要求：
1. 只使用【参考资料】中的信息，不要编造；资料不足以回答时，直接说明“知识库中没有找到相关信息”。
2. 凡是用到某条资料的句子，在句末标注其编号，例如 [1]、[2][3]，编号必须与资料编号一致。
3. 使用简洁的中文，条理清晰，可以用要点列表。

【参考资料】
__CONTEXT__

【用户问题】
__QUERY__

【回答】"""

NO_HIT_TEXT = "知识库中没有找到与该问题相关的资料。"


# ---------------------------------------------------------------- 数据结构


@dataclass
class RetrievedItem:
    """一个经过融合/精排后的候选片段，携带每一路的命中信息（用于引用展示与链路追踪）"""

    node: BaseNode
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    channels: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def channel_labels(self) -> List[str]:
        labels = []
        if "dense" in self.channels:
            labels.append("稠密")
        if "sparse" in self.channels:
            labels.append("稀疏")
        return labels


@dataclass
class RetrievalResult:
    items: List[RetrievedItem]
    trace: Dict[str, Any]


# ---------------------------------------------------------------- 稀疏检索器


class SparseBM25Retriever(BaseRetriever):
    """jieba + BM25 稀疏（关键词）召回，作为与向量召回并列的一路

    直接使用 bm25s 并在索引、查询两侧共用同一套 jieba 分词结果，
    避免默认 token 规则（按一个或多个空格切）在中文上失效。
    """

    def __init__(
        self,
        nodes: Sequence[BaseNode],
        similarity_top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
        show_progress: bool = False,
    ) -> None:
        self.nodes: List[BaseNode] = list(nodes)
        self.similarity_top_k = max(1, int(similarity_top_k))
        self._tokenized: List[List[str]] = []
        self._node_index: Dict[str, int] = {}

        corpus: List[List[str]] = []
        for node in self.nodes:
            corpus.append(tokenize_zh(node.get_content(metadata_mode="none")))
            self._node_index[node.node_id] = len(corpus) - 1
        self._tokenized = corpus

        self.bm25 = bm25s.BM25(k1=k1, b=b, method="lucene")
        if corpus:
            self.bm25.index(corpus, show_progress=show_progress)
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not self.nodes:
            return []
        query_tokens = tokenize_zh(query_bundle.query_str)
        if not query_tokens:
            return []
        k = min(self.similarity_top_k, len(self.nodes))
        indexes, scores = self.bm25.retrieve(
            [query_tokens], k=k, show_progress=False
        )
        out: List[NodeWithScore] = []
        for idx, score in zip(indexes[0], scores[0]):
            score = float(score)
            if score <= 0:  # BM25 全零表示确实没有关键词命中，不塞噪音给融合
                continue
            out.append(NodeWithScore(node=self.nodes[int(idx)], score=score))
        return out


# ---------------------------------------------------------------- RRF 融合


def reciprocal_rank_fusion(
    channels: Sequence[Tuple[str, Sequence[NodeWithScore], float]],
    top_k: int,
    rrf_k: float = 60.0,
) -> List[RetrievedItem]:
    """RRF（Reciprocal Rank Fusion）融合多路召回结果

    score(d) = Σ weight_c / (rrf_k + rank_c(d))
    只看排名不看原始分数，因此不需要对齐向量相似度与 BM25 分数的量纲。

    channels: [(通道名, 该通道结果(已按分数降序), 权重), ...]
    """
    fused: Dict[str, RetrievedItem] = {}
    for name, items, weight in channels:
        for rank, node_with_score in enumerate(items, start=1):
            node = node_with_score.node
            item = fused.get(node.node_id)
            if item is None:
                item = RetrievedItem(node=node)
                fused[node.node_id] = item
            item.rrf_score += weight / (rrf_k + rank)
            item.channels[name] = {
                "rank": rank,
                "score": round(float(node_with_score.score or 0.0), 6),
            }
    ranked = sorted(fused.values(), key=lambda it: it.rrf_score, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------- 检索引擎


class RAGEngine:
    """索引构建 + 双路召回 + RRF 融合 + 精排 + 引用来源"""

    def __init__(self) -> None:
        self.vector_cfg = get_vector_store_config()
        self.retrieval_cfg = get_retrieval_config()
        self.chunking_cfg = get_chunking_config()
        self.documents_dir = get_documents_dir()
        self.storage_dir = self._resolve_storage_dir()

        self.vector_store: Optional[MilvusVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.nodes: List[BaseNode] = []
        self.dense_retriever: Optional[BaseRetriever] = None
        self.sparse_retriever: Optional[SparseBM25Retriever] = None
        self.reranker: Optional[SentenceTransformerRerank] = None

        self.stats: Dict[str, Any] = {}
        self._ready = False
        self.score_mode = str((self.retrieval_cfg.get("rerank") or {}).get("score_mode", "auto"))

    @property
    def ready(self) -> bool:
        """索引与检索链路是否已就绪"""
        return self._ready

    # ------------------------------------------------------------ 配置解析

    def _resolve_storage_dir(self) -> Path:
        raw = self.vector_cfg.get("storage_dir") or "data_dir/storage"
        path = Path(raw)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        return path

    def _resolve_uri(self) -> str:
        """http(s):// 视为独立部署的 Milvus 服务，其余视为本地 Milvus Lite 数据文件"""
        uri = str(self.vector_cfg.get("uri") or "data_dir/milvus.db")
        if uri.startswith(("http://", "https://")):
            return uri
        path = Path(uri)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _embedding_dim(self) -> int:
        """优先自动探测嵌入模型维度，配置里的 dim>0 时以配置为准"""
        configured = int(self.vector_cfg.get("dim") or 0)
        if configured > 0:
            return configured
        embed_model = getattr(Settings, "embed_model", None)
        for attr in ("_model", "model"):
            model = getattr(embed_model, attr, None)
            getter = getattr(model, "get_sentence_embedding_dimension", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:  # pragma: no cover - 探测失败退回默认值
                    pass
        try:
            return int(embed_model.embed_dim)  # type: ignore[union-attr]
        except Exception:
            logger.warning("无法探测嵌入维度，回退到 512（bge-small-zh-v1.5）")
            return 512

    @property
    def file_extensions(self) -> List[str]:
        exts = self.chunking_cfg.get("file_extensions")
        if not exts:
            exts = [".txt", ".md", ".pdf", ".docx", ".doc", ".csv"]
        return [e if e.startswith(".") else "." + e for e in exts]

    # ------------------------------------------------------------ 索引构建

    def setup(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """连接 Milvus、按需重建索引、装配检索链路，返回本次状态"""
        started = time.time()
        self._connect_vector_store(overwrite=force_rebuild)

        manifest = self._load_manifest()
        files, digest = self._scan_documents()
        row_count = self._milvus_row_count()
        reusable = (
            not force_rebuild
            and manifest is not None
            and manifest.get("digest") == digest
            and row_count > 0
        )

        build_seconds = 0.0
        if reusable:
            logger.info(
                "知识库无变化（%d 个文件 / Milvus %d 行），复用已有索引与文档库",
                len(files),
                row_count,
            )
            self.nodes = self._load_persisted_nodes()
            if not self.nodes:  # 文档库丢了就老实重建，避免 BM25 一路落空
                logger.warning("本地文档库缺失，触发重建")
                reusable = False

        if not reusable:
            t0 = time.time()
            self.nodes = self._build_index(files, digest)
            build_seconds = time.time() - t0
            row_count = self._milvus_row_count()

        self._ensure_collection_loaded()
        self._build_retrievers()
        self._ready = True

        self.stats = {
            "documents_dir": str(self.documents_dir),
            "files": len(files),
            "chunks": len(self.nodes),
            "milvus_uri": self._resolve_uri(),
            "collection": self.vector_cfg.get("collection", "mom_chat_chunks"),
            "vector_rows": row_count,
            "embed_dim": self._embedding_dim(),
            "metric": self.vector_cfg.get("metric", "COSINE"),
            "reused": reusable,
            "rebuild_seconds": round(build_seconds, 2),
            "setup_seconds": round(time.time() - started, 2),
            "retrieval": self._describe_retrieval(),
        }
        logger.info("检索链路就绪: %s", json.dumps(self.stats, ensure_ascii=False))
        return self.stats

    def _connect_vector_store(self, overwrite: bool = False) -> None:
        uri = self._resolve_uri()
        collection = str(self.vector_cfg.get("collection") or "mom_chat_chunks")
        dim = self._embedding_dim()
        is_remote = uri.startswith(("http://", "https://"))
        if not is_remote and self.vector_cfg.get("filter_lite_noise", True):
            _silence_milvus_lite_noise()
        try:
            self.vector_store = MilvusVectorStore(
                uri=uri,
                token=str(self.vector_cfg.get("token") or ""),
                collection_name=collection,
                dim=dim,
                similarity_metric=str(self.vector_cfg.get("metric") or "COSINE"),
                overwrite=overwrite,
                # Milvus Lite 走本地文件，不支持异步客户端；独立服务才开启
                use_async_client=is_remote,
                consistency_level=str(
                    self.vector_cfg.get("consistency_level") or "Session"
                ),
            )
        except Exception as exc:
            detail = " | ".join(str(item) for item in _exception_chain(exc)).lower()
            if not is_remote and (
                "lock" in detail or "open local milvus failed" in detail
            ):
                raise RuntimeError(
                    f"Milvus 本地数据文件被另一个进程占用：{uri}\n"
                    "Milvus Lite（本地文件模式）同一时刻只允许一个进程访问该数据文件。\n"
                    "处理方式：① 关掉已在运行的实例再启动；"
                    "② 或改用独立部署的 Milvus（docker compose -f docker-compose.milvus.yml up -d），"
                    "并把 config.yaml 中 vector_store.uri 设为 http://localhost:19530。"
                ) from exc
            raise
        logger.info("已连接 Milvus 向量库: %s / collection=%s / dim=%d", uri, collection, dim)

    def _milvus_row_count(self) -> int:
        try:
            stats = self.vector_store.client.get_collection_stats(  # type: ignore[union-attr]
                self.vector_cfg.get("collection", "mom_chat_chunks")
            )
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def _ensure_collection_loaded(self) -> None:
        """复用已有 collection 时必须先 load，否则 search 会报 released"""
        collection = str(self.vector_cfg.get("collection") or "mom_chat_chunks")
        try:
            self.vector_store.client.load_collection(collection)  # type: ignore[union-attr]
        except Exception as exc:  # 新库还没建 collection 时忽略即可
            logger.debug("load_collection(%s) 跳过: %s", collection, exc)

    # -- 文档扫描与清单（文件内容哈希不变就不重复切分/嵌入） --------

    def _scan_documents(self) -> Tuple[List[Dict[str, Any]], str]:
        files: List[Dict[str, Any]] = []
        if self.documents_dir.exists():
            for path in sorted(self.documents_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in self.file_extensions:
                    continue
                if self.storage_dir in path.parents:  # 跳过索引自身的持久化目录
                    continue
                if path.stat().st_size == 0:  # 空文件没啥可索引的，免得解析器报错
                    logger.debug("跳过空文件: %s", path.name)
                    continue
                files.append(
                    {
                        "path": path.relative_to(self.documents_dir).as_posix(),
                        "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
                        "size": path.stat().st_size,
                    }
                )
        digest = hashlib.sha1(
            json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return files, digest

    def _manifest_path(self) -> Path:
        return self.storage_dir / "manifest.json"

    def _load_manifest(self) -> Optional[Dict[str, Any]]:
        path = self._manifest_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("读取索引清单失败: %s", exc)
            return None

    def _load_persisted_nodes(self) -> List[BaseNode]:
        if not (self.storage_dir / "docstore.json").exists():
            return []
        try:
            docstore = SimpleDocumentStore.from_persist_dir(str(self.storage_dir))
            return list(docstore.docs.values())
        except Exception as exc:
            logger.warning("恢复本地文档库失败: %s", exc)
            return []

    def _build_index(self, files: List[Dict[str, Any]], digest: str) -> List[BaseNode]:
        """读取文档 → 切分 → 嵌入并写入 Milvus → 持久化文档库与清单"""
        if not files:
            logger.warning("知识库目录 %s 下没有可解析的文档", self.documents_dir)
        documents = SimpleDirectoryReader(
            input_dir=str(self.documents_dir),
            recursive=True,
            required_exts=self.file_extensions,
            exclude_hidden=True,
        ).load_data() if files else []

        splitter = SentenceSplitter(
            chunk_size=int(self.chunking_cfg.get("chunk_size", 512)),
            chunk_overlap=int(self.chunking_cfg.get("chunk_overlap", 64)),
            paragraph_separator=str(self.chunking_cfg.get("paragraph_separator", "\n")),
        )
        nodes = splitter.get_nodes_from_documents(documents) if documents else []

        # 打上引用所需的元数据：相对路径 + 文件内片段序号
        counter: Dict[str, int] = {}
        for node in nodes:
            meta = node.metadata or {}
            file_path = str(meta.get("file_path") or "")
            rel = file_path
            try:
                rel = Path(file_path).relative_to(self.documents_dir).as_posix()
            except Exception:
                pass
            meta["relative_path"] = rel
            meta["file_name"] = meta.get("file_name") or Path(file_path).name
            idx = counter.get(rel, 0) + 1
            counter[rel] = idx
            meta["chunk_index"] = idx
            node.metadata = meta
            node.excluded_embed_metadata_keys = []
            node.excluded_llm_metadata_keys = []

        # 重建时先清空 collection，避免残留旧片段
        self._connect_vector_store(overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=False,
            # Milvus 自带文本存储，默认不会再往 docstore 写节点；这里强制双写，
            # 重启后才能直接从 docstore 恢复节点喂给 BM25，不必重新切分+嵌入
            store_nodes_override=True,
        )

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        self._manifest_path().write_text(
            json.dumps(
                {
                    "digest": digest,
                    "files": files,
                    "chunks": len(nodes),
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("索引重建完成：%d 个文件 / %d 个片段", len(files), len(nodes))
        return nodes

    # -- 检索链路装配 ------------------------------------------------

    def _build_retrievers(self) -> None:
        dense_cfg = self.retrieval_cfg.get("dense", {}) or {}
        sparse_cfg = self.retrieval_cfg.get("sparse", {}) or {}
        rerank_cfg = self.retrieval_cfg.get("rerank", {}) or {}

        # 复用已有 collection（重建分支已经产出了 index）
        if self.nodes and self.index is None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, storage_context=storage_context
            )

        self.dense_retriever = (
            self.index.as_retriever(similarity_top_k=int(dense_cfg.get("top_k", 10)))
            if self.index is not None and dense_cfg.get("enabled", True)
            else None
        )
        self.sparse_retriever = (
            SparseBM25Retriever(
                nodes=self.nodes,
                similarity_top_k=int(sparse_cfg.get("top_k", 10)),
                k1=float(sparse_cfg.get("bm25_k1", 1.5)),
                b=float(sparse_cfg.get("bm25_b", 0.75)),
            )
            if self.nodes and sparse_cfg.get("enabled", True)
            else None
        )
        self.reranker = self._build_reranker(rerank_cfg)

    def _build_reranker(
        self, rerank_cfg: Dict[str, Any]
    ) -> Optional[SentenceTransformerRerank]:
        if not rerank_cfg.get("enabled", True):
            return None
        # sentence-transformers 的 INFO 级别会打出 CrossEncoder 的进度条，日志里很吵
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        self.score_mode = str(rerank_cfg.get("score_mode") or "auto")
        model = str(rerank_cfg.get("model") or "BAAI/bge-reranker-base")
        if not Path(model).exists():
            from modelscope import snapshot_download

            model = snapshot_download(model)
        reranker = SentenceTransformerRerank(
            model=model,
            top_n=int(rerank_cfg.get("top_n", 5)),
            device=str(rerank_cfg.get("device", "cpu")),
        )
        logger.info("精排模型就绪: %s (top_n=%d)", model, reranker.top_n)
        return reranker

    def _describe_retrieval(self) -> Dict[str, Any]:
        dense_cfg = self.retrieval_cfg.get("dense", {}) or {}
        sparse_cfg = self.retrieval_cfg.get("sparse", {}) or {}
        fusion_cfg = self.retrieval_cfg.get("fusion", {}) or {}
        rerank_cfg = self.retrieval_cfg.get("rerank", {}) or {}
        return {
            "dense": {"enabled": bool(dense_cfg.get("enabled", True)), "top_k": int(dense_cfg.get("top_k", 10))},
            "sparse": {"enabled": bool(sparse_cfg.get("enabled", True)), "top_k": int(sparse_cfg.get("top_k", 10))},
            "fusion": {"method": str(fusion_cfg.get("method", "rrf")), "k": float(fusion_cfg.get("k", 60)), "top_k": int(fusion_cfg.get("top_k", 10))},
            "rerank": {"enabled": bool(rerank_cfg.get("enabled", True)), "top_n": int(rerank_cfg.get("top_n", 5)), "model": str(rerank_cfg.get("model", "BAAI/bge-reranker-base"))},
        }

    # ------------------------------------------------------------ 检索

    def retrieve(self, query: str) -> RetrievalResult:
        """双路召回 → RRF 融合 → 精排，返回最终片段与链路明细"""
        if not self._ready:
            raise RuntimeError("检索引擎尚未初始化")

        dense_cfg = self.retrieval_cfg.get("dense", {}) or {}
        sparse_cfg = self.retrieval_cfg.get("sparse", {}) or {}
        fusion_cfg = self.retrieval_cfg.get("fusion", {}) or {}
        rerank_cfg = self.retrieval_cfg.get("rerank", {}) or {}

        query_bundle = QueryBundle(query_str=query)
        trace: Dict[str, Any] = {}
        channels: List[Tuple[str, Sequence[NodeWithScore], float]] = []
        weights = list(fusion_cfg.get("weights") or [1.0, 1.0])

        # --- 稠密：Milvus 向量相似度
        if self.dense_retriever is not None:
            t0 = time.perf_counter()
            dense_nodes = self.dense_retriever.retrieve(query_bundle)
            trace["dense"] = {
                "enabled": True,
                "top_k": int(dense_cfg.get("top_k", 10)),
                "hits": len(dense_nodes),
                "best_score": round(float(dense_nodes[0].score or 0.0), 4) if dense_nodes else None,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            channels.append(("dense", dense_nodes, float(weights[0]) if weights else 1.0))
        else:
            trace["dense"] = {"enabled": False, "hits": 0}

        # --- 稀疏：jieba + BM25 关键词
        if self.sparse_retriever is not None:
            t0 = time.perf_counter()
            sparse_nodes = self.sparse_retriever.retrieve(query_bundle)
            trace["sparse"] = {
                "enabled": True,
                "top_k": int(sparse_cfg.get("top_k", 10)),
                "hits": len(sparse_nodes),
                "best_score": round(float(sparse_nodes[0].score or 0.0), 4) if sparse_nodes else None,
                "tokens": tokenize_zh(query)[:12],
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            channels.append(("sparse", sparse_nodes, float(weights[1]) if len(weights) > 1 else 1.0))
        else:
            trace["sparse"] = {"enabled": False, "hits": 0}

        # --- RRF 融合
        t0 = time.perf_counter()
        fusion_top_k = int(fusion_cfg.get("top_k", 10))
        fused = reciprocal_rank_fusion(
            channels, top_k=fusion_top_k, rrf_k=float(fusion_cfg.get("k", 60))
        )
        overlap = sum(1 for it in fused if len(it.channels) > 1)
        trace["fusion"] = {
            "method": str(fusion_cfg.get("method", "rrf")),
            "k": float(fusion_cfg.get("k", 60)),
            "candidates": len(fused),
            "hit_both": overlap,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

        items = fused

        # --- 精排
        if self.reranker is not None and fused:
            t0 = time.perf_counter()
            candidates = [
                NodeWithScore(node=it.node, score=it.rrf_score) for it in fused
            ]
            ranked = self.reranker.postprocess_nodes(candidates, query_bundle=query_bundle)
            score_by_id = {nws.node.node_id: float(nws.score or 0.0) for nws in ranked}
            order = {nws.node.node_id: rank for rank, nws in enumerate(ranked)}
            items = [
                RetrievedItem(
                    node=it.node,
                    rrf_score=it.rrf_score,
                    rerank_score=score_by_id.get(it.node.node_id),
                    channels=it.channels,
                )
                for it in fused
                if it.node.node_id in score_by_id
            ]
            items.sort(key=lambda it: order[it.node.node_id])
            # 相关性阈值：低于阈值的片段不送大模型、也不作为引用，
            # 避免“知识库外的问题”被硬凑出几条无关引用
            min_relevance = float(rerank_cfg.get("min_relevance", 0.02) or 0.0)
            kept = items
            if min_relevance > 0:
                kept = [
                    it
                    for it in items
                    if it.rerank_score is not None
                    and prob(it.rerank_score, self.score_mode) >= min_relevance
                ]
            trace["rerank"] = {
                "enabled": True,
                "model": str(rerank_cfg.get("model", "BAAI/bge-reranker-base")),
                "candidates": len(candidates),
                "kept": len(kept),
                "min_relevance": min_relevance,
                "filtered_out": len(items) - len(kept),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            items = kept
        else:
            trigger = "关闭" if self.reranker is None else "无候选"
            trace["rerank"] = {"enabled": self.reranker is not None, "note": trigger, "kept": len(items)}
        trace["final"] = len(items)
        return RetrievalResult(items=items, trace=trace)

    # ------------------------------------------------------------ 生成

    def build_prompt(self, query: str, items: Sequence[RetrievedItem]) -> str:
        blocks = []
        for i, item in enumerate(items, start=1):
            blocks.append(f"[{i}] 来源：{self.source_label(item)}\n{item.node.get_content().strip()}")
        context = "\n\n".join(blocks) if blocks else "（知识库中没有检索到相关资料）"
        return QA_PROMPT.replace("__CONTEXT__", context).replace("__QUERY__", query)

    def stream_answer(self, query: str) -> Iterator[Dict[str, Any]]:
        """同步流式生成：先回推检索链路，再逐 token 回推答案，最后回推引用来源"""
        t0 = time.perf_counter()
        result = self.retrieve(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        yield {"type": "retrieval", "trace": result.trace}

        if not result.items:
            yield {"type": "token", "text": NO_HIT_TEXT}
            yield self._done_payload([], result.trace, t0, None, retrieval_ms)
            return

        prompt = self.build_prompt(query, result.items)
        first_token_at: Optional[float] = None
        for chunk in Settings.llm.stream_complete(prompt):
            delta = getattr(chunk, "delta", None) or ""
            if not delta:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            yield {"type": "token", "text": delta}

        yield self._done_payload(
            self.citations(result.items), result.trace, t0, first_token_at, retrieval_ms
        )

    async def astream_answer(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """异步流式生成：检索放到线程里跑，token 走异步 HTTP，不阻塞事件循环"""
        t0 = time.perf_counter()
        result = await asyncio.to_thread(self.retrieve, query)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        yield {"type": "retrieval", "trace": result.trace}

        if not result.items:
            yield {"type": "token", "text": NO_HIT_TEXT}
            yield self._done_payload([], result.trace, t0, None, retrieval_ms)
            return

        prompt = self.build_prompt(query, result.items)
        first_token_at: Optional[float] = None
        stream = Settings.llm.astream_complete(prompt)
        if not hasattr(stream, "__aiter__"):  # 各实现可能返回协程，先 await 拿到异步生成器
            stream = await stream  # type: ignore[assignment]
        async for chunk in stream:
            delta = getattr(chunk, "delta", None) or ""
            if not delta:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            yield {"type": "token", "text": delta}

        yield self._done_payload(
            self.citations(result.items), result.trace, t0, first_token_at, retrieval_ms
        )

    @staticmethod
    def _done_payload(
        sources: List[Dict[str, Any]],
        trace: Dict[str, Any],
        started_at: float,
        first_token_at: Optional[float],
        retrieval_ms: float,
    ) -> Dict[str, Any]:
        return {
            "type": "done",
            "sources": sources,
            "trace": trace,
            "timings": {
                "retrieval_ms": round(retrieval_ms, 1),
                "first_token_ms": round((first_token_at - started_at) * 1000, 1)
                if first_token_at
                else None,
                "total_ms": round((time.perf_counter() - started_at) * 1000, 1),
            },
        }

    def answer(self, query: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """非流式问答，供 /query 使用"""
        result = self.retrieve(query)
        if not result.items:
            return "知识库中没有找到与该问题相关的资料。", [], result.trace
        prompt = self.build_prompt(query, result.items)
        response = Settings.llm.complete(prompt)
        return str(response), self.citations(result.items), result.trace

    # ------------------------------------------------------------ 引用来源

    def source_label(self, item: RetrievedItem) -> str:
        meta = item.node.metadata or {}
        name = str(meta.get("file_name") or "未知文件")
        chunk = meta.get("chunk_index")
        return f"{name}（第 {chunk} 片）" if chunk else name

    def citations(self, items: Sequence[RetrievedItem]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(items, start=1):
            meta = item.node.metadata or {}
            dense = item.channels.get("dense")
            sparse = item.channels.get("sparse")
            out.append(
                {
                    "index": i,
                    "file_name": str(meta.get("file_name") or "未知文件"),
                    "file_path": str(meta.get("relative_path") or meta.get("file_path") or ""),
                    "chunk": meta.get("chunk_index"),
                    "relevance": round(prob(item.rerank_score, self.score_mode), 4) if item.rerank_score is not None else None,
                    "rerank_score": round(float(item.rerank_score), 4) if item.rerank_score is not None else None,
                    "rrf_score": round(item.rrf_score, 6),
                    "channels": item.channel_labels,
                    "dense_rank": dense["rank"] if dense else None,
                    "dense_score": dense["score"] if dense else None,
                    "sparse_rank": sparse["rank"] if sparse else None,
                    "sparse_score": sparse["score"] if sparse else None,
                    "snippet": re.sub(r"\s+", " ", item.node.get_content()).strip()[:200],
                }
            )
        return out


def prob(raw: Optional[float], mode: str = "auto") -> float:
    """把精排原始分数换算成 0~1 的相关度（只影响展示，不影响排序）

    - auto：bge-reranker 系列自带 Sigmoid 激活，输出已是 0~1 概率，直接用；
      换成输出 logits 的 reranker（如 bce-reranker 的裸 logits）则自动做 sigmoid
    - probability：强制按概率处理（截断到 0~1）
    - logit：强制按 logits 处理（sigmoid 归一化）
    """
    if raw is None:
        return 0.0
    value = float(raw)
    if mode == "logit":
        return _sigmoid(value)
    if mode == "probability":
        return min(1.0, max(0.0, value))
    if 0.0 <= value <= 1.0:
        return value
    return _sigmoid(value)


def _sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0
