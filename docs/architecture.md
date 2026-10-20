# mom_chat 检索架构图（Mermaid）

> 说明：三张图分别对应「整体分层架构」「单次问答的检索链路」「离线索引与部署形态」。
> 已在 mermaid.ink 上渲染校验通过（中文标签正常），渲染产物见 `docs/diagrams/`。
> 抽象术语，可直接用于技术分享材料。

## 一、整体分层架构

```mermaid
flowchart TB
    Q["用户提问"] --> API["接入层<br/>HTTP 服务 · SSE 流式接口"]
    API --> ORCH["编排层<br/>上下文组装 · 会话与引用落库"]

    subgraph RET["检索层"]
        direction LR
        D["稠密召回<br/>嵌入模型编码 · 向量语义相似"]
        VDB[("向量库<br/>本地文件 / 独立服务")]
        S["稀疏召回<br/>中文分词 · 关键词精确匹配"]
        SPX[("稀疏索引<br/>内存 BM25")]
        F["RRF 融合<br/>按排名倒数加权合并"]
        RR["精排<br/>Cross-Encoder 逐条打分 + 相关性阈值"]
        D --- VDB
        S --- SPX
        D --> F
        S --> F
        F --> RR
    end

    ORCH -->|"原始问题"| D
    ORCH -->|"原始问题"| S
    ORCH -.->|"片段元数据 / 会话"| META[("片段元数据<br/>文档库 + 版本清单")]
    RR -->|"Top-N 片段（带引用元数据）"| GEN["生成层<br/>大模型只依据片段作答 · 句末标注引用编号"]
    GEN --> OUT["输出层<br/>流式答案 + 引用来源 + 检索链路统计"]

    classDef app fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef core fill:#eef2ff,stroke:#6366f1,color:#1e293b
    classDef gen fill:#fef3c7,stroke:#f59e0b,color:#78350f
    classDef store fill:#f1f5f9,stroke:#64748b,color:#1e293b
    class Q,API,ORCH app
    class D,S,F,RR core
    class GEN,OUT gen
    class VDB,SPX,META store
```

## 二、单次问答的检索链路

```mermaid
flowchart LR
    Q["用户问题"] --> D["稠密召回<br/>向量相似度 Top-K"]
    Q --> S["稀疏召回<br/>分词后关键词 Top-K"]

    D --> F["RRF 融合<br/>score = Σ weight / (k + rank)"]
    S --> F
    F -->|"去重后的候选池"| R["精排<br/>问题+片段 逐条打分"]

    R --> T{"是否达到<br/>相关性阈值"}
    T -->|"否"| NO["不调用大模型<br/>直接回答「知识库中无相关资料」"]
    T -->|"是"| CTX["按相关度截断 Top-N<br/>组装带编号的上下文"]
    CTX --> GEN["大模型生成答案<br/>句末标注 [1] [2]"]
    GEN --> OUT["答案 + 引用来源<br/>文件 · 片段 · 相关度 · 命中通道"]

    subgraph OBS["可观测"]
        direction TB
        TRACE["链路轨迹<br/>各阶段命中数与耗时"]
        TIME["耗时分解<br/>召回 / 融合 / 精排 / 首 token"]
    end
    F -.-> TRACE
    R -.-> TRACE
    GEN -.-> TIME

    classDef step fill:#eef2ff,stroke:#6366f1,color:#1e293b
    classDef gate fill:#ffe4e6,stroke:#e11d48,color:#881337
    classDef obs fill:#f8fafc,stroke:#94a3b8,color:#1e293b
    class Q,D,S,F,R,CTX,GEN,OUT step
    class T,NO gate
    class TRACE,TIME obs
```

## 三、离线索引与部署形态

```mermaid
flowchart LR
    subgraph OFF["离线：索引构建（增量复用）"]
        direction TB
        DOCS["知识库文档"] --> SCAN["扫描 + 内容指纹<br/>记录文件清单与哈希"]
        SCAN --> CHK{"指纹是否变化"}
        CHK -->|"无变化"| REUSE["复用已有集合与片段元数据<br/>跳过切分与向量化"]
        CHK -->|"有变化"| SPLIT["文本切分<br/>定长 + 重叠"]
        SPLIT --> EMB["向量化"]
        EMB --> UP["写入向量库<br/>重建稀疏索引 · 持久化片段元数据"]
    end

    subgraph ON["在线：一次问答"]
        direction TB
        ASK["用户提问"] --> Q["双路召回<br/>稠密 + 稀疏"]
        Q --> FUSE["RRF 融合"]
        FUSE --> RR["精排 + 阈值过滤"]
        RR --> PR["上下文组装<br/>片段带编号引用"]
        PR --> AN["大模型流式生成"]
        AN --> CI["返回引用来源与检索链路"]
    end

    subgraph DEP["部署形态：仅替换连接地址"]
        direction TB
        LITE["嵌入式向量库<br/>本地文件 · 单机开发"]
        SRV["独立向量库服务<br/>容器化 · 多实例"]
    end

    UP -.->|"索引就绪"| Q
    REUSE -.->|"索引就绪"| Q
    DEP -.->|"同一套接口"| UP

    classDef off fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef on fill:#eef2ff,stroke:#6366f1,color:#1e293b
    classDef dep fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef gate fill:#fef3c7,stroke:#f59e0b,color:#78350f
    class DOCS,SCAN,SPLIT,EMB,UP,REUSE off
    class ASK,Q,FUSE,RR,PR,AN,CI on
    class LITE,SRV dep
    class CHK gate
```

## 术语对照（抽象 → 本项目实现）

| 抽象术语 | 本项目实现 |
| --- | --- |
| 接入层 | FastAPI + SSE（`/stream-query`） |
| 编排层 | `rag_engine.RAGEngine` 提示词组装、引用抽取、会话落库 |
| 稠密召回 | Milvus 向量检索（bge-small-zh-v1.5，余弦相似度） |
| 稀疏召回 | jieba 分词 + bm25s（BM25，k1=1.5、b=0.75） |
| 融合 | RRF（k=60，两路权重可配） |
| 精排 | bge-reranker-base（Cross-Encoder）+ 相关性阈值过滤 |
| 生成层 | DeepSeek（`deepseek-reasoner`）流式生成 |
| 片段元数据 | `data_dir/storage/`（docstore + manifest 版本清单） |
