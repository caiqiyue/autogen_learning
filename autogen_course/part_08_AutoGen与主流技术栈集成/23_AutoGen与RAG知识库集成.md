---
lesson_id: lesson_23
title: AutoGen与RAG知识库集成
module: AutoGen与主流技术栈集成
---

# 第23节 AutoGen与RAG知识库集成

## 学习目标

- 掌握AutoGen与RAG系统的集成方法
- 理解检索结果作为Agent上下文注入的机制
- 能够实现AutoGen+RAG的检索增强对话系统
- 掌握混合架构：LangGraph编排+AutoGen执行+RAG检索

## 内容概述

RAG（检索增强生成）是一种将外部知识检索与LLM生成相结合的技术，能够有效解决LLM知识过时、幻觉等问题。本节将深入解析如何将AutoGen与RAG系统集成，让Agent具备知识检索能力，实现更精准、更专业的问答系统。

---

## 1. RAG系统的基本原理

### 1.1 为什么需要RAG

LLM虽然强大，但存在以下局限：

| 问题 | 说明 | RAG解决方案 |
|------|------|-------------|
| 知识过时 | 训练数据有时间截止日期 | 实时检索最新文档 |
| 幻觉 | 可能生成看似合理但错误的内容 | 基于检索证据生成 |
| 领域知识不足 | 通用模型缺乏专业领域知识 | 检索专业知识库 |
| 信息不可验证 | 输出无法追溯来源 | 提供检索来源引用 |

### 1.2 RAG系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG 系统架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   文档入库   │ -> │   分块存储   │ -> │   向量索引   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                                     │             │
│         ▼                                     ▼             │
│  ┌─────────────┐                      ┌─────────────┐     │
│  │  源文档存储   │                      │  向量数据库   │     │
│  └─────────────┘                      └─────────────┘     │
│                                              │             │
│                                              ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   用户查询   │ -> │   相似度检索 │ -> │   上下文组装 │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                              │             │
│                                              ▼             │
│                                    ┌─────────────┐         │
│                                    │   LLM 生成   │         │
│                                    └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 RAG核心组件

| 组件 | 功能 | 技术选型 |
|------|------|----------|
| 文档加载器 | 加载各类文档（PDF、HTML、TXT等） | LangChain DocumentLoaders |
| 文本分割器 | 将文档分割成小块 | RecursiveCharacterTextSplitter |
| 向量化模型 | 将文本块转换为向量 | OpenAI Embeddings、sentence-transformers |
| 向量数据库 | 存储和检索向量 | Chroma、Pinecone、Milvus |
| 检索器 | 根据查询检索相关文档 | similarity_search、mmr |
| 生成器 | 基于检索结果生成回答 | GPT-4o、Claude等 |

---

## 2. 检索结果作为Agent上下文注入

### 2.1 注入的核心机制

AutoGen的Agent通过system_message和对话上下文进行推理。将检索结果注入Agent，就是将RAG检索到的知识片段加入到Agent的上下文中，使其能够基于检索内容生成回答。

```python
# 检索结果注入的核心模式
retrieved_context = rag_retriever.query(user_question)  # 执行RAG检索

# 方式1：通过system_message注入（静态、全局）
agent = AssistantAgent(
    system_message=f"你是一个知识库问答助手。基于以下知识回答问题：\n{retrieved_context}"
)

# 方式2：通过initiate_chat的message注入（动态、每次对话）
agent.initiate_chat(
    assistant,
    message=f"基于以下知识回答：\n{retrieved_context}\n\n问题：{user_question}"
)

# 方式3：通过register_reply回调注入（高级、条件触发）
```

### 2.2 注入的时机选择

| 时机 | 方式 | 适用场景 |
|------|------|----------|
| 初始化时 | system_message | 知识相对固定、长期使用 |
| 每次查询时 | message参数 | 知识动态变化、每次查询不同 |
| 条件触发时 | register_reply | 需要根据对话内容决定是否检索 |
| 流式处理中 | 消息hook | 需要实时处理检索结果 |

### 2.3 上下文窗口管理

RAG检索结果可能很长，需要管理上下文长度：

```python
# 上下文长度管理策略
def assemble_context(query: str, max_tokens: int = 4000) -> str:
    """
    组装检索上下文，控制总长度不超过指定token数

    参数:
        query: 用户查询
        max_tokens: 最大token数（估算）

    返回:
        组装后的上下文字符串
    """
    # 执行检索
    results = vector_store.similarity_search(query, k=5)

    context_parts = []
    total_length = 0

    for doc in results:
        # 估算当前块的长度
        doc_length = len(doc.page_content) // 4  # 粗略估算token数

        if total_length + doc_length > max_tokens:
            break  # 达到上限，停止添加

        context_parts.append(doc.page_content)
        total_length += doc_length

    return "\n\n---\n\n".join(context_parts)
```

---

## 3. 知识库问答系统的架构设计

### 3.1 基础架构：AutoGen + RAG

```
┌─────────────────────────────────────────────────────────────┐
│                  基础 RAG 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌─────────────┐         ┌─────────────┐                 │
│    │   用户问题   │ ──────► │  RAG 检索器  │                 │
│    └─────────────┘         └─────────────┘                 │
│                                   │                         │
│                                   ▼                         │
│                            ┌─────────────┐                 │
│                            │  检索结果    │                 │
│                            └─────────────┘                 │
│                                   │                         │
│                                   ▼                         │
│    ┌─────────────┐         ┌─────────────┐                 │
│    │   用户       │ ──────► │  Assistant  │                 │
│    │   Proxy      │         │    Agent    │                 │
│    └─────────────┘         └─────────────┘                 │
│                                   │                         │
│                                   ▼                         │
│                            ┌─────────────┐                 │
│                            │  生成回答    │                 │
│                            └─────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 多Agent协作架构

更复杂的场景可以使用多Agent协作：

```
┌─────────────────────────────────────────────────────────────┐
│                 多Agent RAG 架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌─────────────┐                                          │
│    │   用户问题   │                                          │
│    └─────────────┘                                          │
│           │                                                  │
│           ▼                                                  │
│    ┌─────────────────────────────────────────┐              │
│    │            Router Agent                  │              │
│    │    (判断问题类型，决定路由)              │              │
│    └─────────────────────────────────────────┘              │
│           │                                                  │
│     ┌─────┴─────┐                                           │
│     ▼           ▼                                           │
│  ┌──────┐   ┌──────┐   ┌──────┐                            │
│  │检索  │   │搜索  │   │直接  │                            │
│  │Agent │   │Agent │   │生成  │                            │
│  └──┬───┘   └──┬───┘   └──────┘                            │
│     │          │                                            │
│     ▼          ▼                                            │
│  ┌──────────────┴──────────────┐                           │
│  │       Synthesizer Agent     │                           │
│  │     (综合检索结果生成回答)   │                           │
│  └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 架构选择决策树

```
问题分析
    │
    ├─► 是否需要外部知识？
    │       │
    │       ├─► 否 ──► 直接使用LLM生成
    │       │
    │       └─► 是 ──► 需要RAG
    │               │
    │               ├─► 单领域知识库 ──► 基础架构
    │               │
    │               ├─► 多领域知识库 ──► 多Agent协作架构
    │               │
    │               └─► 需要推理验证 ──► 混合架构(LangGraph+AutoGen+RAG)
    │
    └─► 回答是否需要可解释性/溯源？
            │
            ├─► 是 ──► RAG + 引用标注
            │
            └─► 否 ──► 简单RAG或直接生成
```

---

## 4. 混合架构：LangGraph编排+AutoGen执行+RAG检索

### 4.1 架构概述

在复杂的企业场景中，单一工具往往不够用。我们可以结合：
- **LangGraph**：负责工作流编排、条件分支、状态管理
- **AutoGen**：负责具体执行、多Agent协作、对话生成
- **RAG**：负责知识检索、上下文注入

```
┌─────────────────────────────────────────────────────────────┐
│               混合架构：LangGraph + AutoGen + RAG           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    LangGraph 编排层                   │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐           │ │
│  │  │  START   │───►│ Router  │───►│ Action  │           │ │
│  │  └─────────┘    └────┬────┘    └────┬────┘           │ │
│  │                      │             │                  │ │
│  │              ┌───────┴───────┐     │                  │ │
│  │              ▼               ▼     ▼                  │ │
│  │        ┌─────────┐    ┌─────────┐ ┌─────────┐        │ │
│  │        │SearchRAG│    │VectorRAG│ │DirectGen│        │ │
│  │        └────┬────┘    └────┬────┘ └─────────┘        │ │
│  │             │             │                           │ │
│  │             └──────┬──────┘                           │ │
│  │                    ▼                                   │ │
│  │              ┌─────────┐                              │ │
│  │              │Synthesize│                              │ │
│  │              └─────────┘                              │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                   AutoGen 执行层                        │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │ │
│  │  │ Router Agent│───►│RAG Agent(s) │───►│Synth Agent │ │ │
│  │  └─────────────┘    └─────────────┘    └────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                      RAG 检索层                       │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐           │ │
│  │  │向量数据库│    │搜索引擎 │    │知识图谱 │           │ │
│  │  └─────────┘    └─────────┘    └─────────┘           │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 LangGraph状态定义

```python
from typing import TypedDict, Annotated
import operator

class RAGState(TypedDict):
    """
    混合架构的状态定义

    属性:
        question: 用户问题
        intent: 分类的意图（search/vector/direct）
        retrieved_docs: 检索到的文档列表
        context: 组装后的检索上下文
        answer: 最终生成的回答
        sources: 引用的来源文档
    """
    question: str
    intent: str
    retrieved_docs: list
    context: str
    answer: str
    sources: list
```

### 4.3 节点定义

```python
from langgraph.graph import StateGraph, END

def create_rag_workflow():
    """
    创建RAG工作流图

    返回:
        编译后的LangGraph工作流
    """
    # 创建状态图
    workflow = StateGraph(RAGState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("vector_retriever", vector_retriever_node)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("direct_generator", direct_generator_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 设置入口
    workflow.set_entry_point("router")

    # 添加条件边
    workflow.add_conditional_edges(
        "router",
        route_based_on_intent,
        {
            "vector": "vector_retriever",
            "web": "web_searcher",
            "direct": "direct_generator"
        }
    )

    # 汇聚边
    workflow.add_edge("vector_retriever", "synthesizer")
    workflow.add_edge("web_searcher", "synthesizer")
    workflow.add_edge("direct_generator", "synthesizer")

    # 结束
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
```

---

## 5. 评估与优化

### 5.1 RAG系统评估指标

| 指标 | 说明 | 评估方法 |
|------|------|----------|
| 召回率 (Recall) | 检索结果中包含的相关文档比例 | 计算相关文档被检出的比例 |
| 精确率 (Precision) | 检索结果中相关文档的比例 | 计算检出文档中相关的比例 |
| MRR (Mean Reciprocal Rank) | 首个相关文档排名的倒数均值 | 评估排序质量 |
| NDCG | 归一化折损累积增益 | 评估排序相关性 |
| 上下文相关性 | 检索上下文与问题的相关程度 | 人工或LLM评估 |

### 5.2 常见优化策略

| 策略 | 说明 | 实现方式 |
|------|------|----------|
| 混合检索 | 结合向量检索和关键词检索 | Hybrid search (bm25 + vector) |
| 重排序 | 对检索结果进行二次排序 | Cross-encoder reranking |
| 查询扩展 | 扩展用户查询以提高召回 | Query expansion with LLM |
| 自适应块大小 | 根据问题类型选择块大小 | Dynamic chunk sizing |
| 元数据过滤 | 利用文档元数据过滤 | Metadata filtering |

---

## 代码案例

本节包含两个代码案例，请参考 `23_codes/` 目录：

### 案例1：RAG基础集成 (rag_basic.py)

**内容要点：**
- RAG系统的基本组件介绍
- 向量数据库初始化与文档入库
- 相似度检索实现
- 检索结果注入AutoGen Agent的几种方式
- 简单的知识库问答系统实现

**运行方式：**
```bash
python 23_codes/rag_basic.py
```

### 案例2：RAG高级集成 (rag_advanced.py)

**内容要点：**
- 混合检索：向量检索 + BM25关键词检索
- AutoGen多Agent协作与RAG结合
- LangGraph工作流编排
- 重排序机制实现
- 高级知识库问答系统架构

**运行方式：**
```bash
python 23_codes/rag_advanced.py
```

---

## 本节小结

1. **RAG基本原理**：RAG通过检索外部知识库增强LLM的生成能力，解决知识过时、幻觉等问题

2. **检索结果注入**：可以通过system_message、message参数、register_reply回调等方式将检索结果注入Agent上下文

3. **架构设计**：从简单的单Agent+RAG到复杂的多Agent协作，需要根据业务场景选择合适的架构

4. **混合架构**：LangGraph负责编排、AutoGen负责执行、RAG负责检索，三者结合实现强大的知识问答系统

5. **评估优化**：通过召回率、精确率、MRR等指标评估RAG系统，并采用混合检索、重排序等策略优化性能

---

## 延伸阅读

- [AutoGen与LangChain集成](./22_AutoGen与LangChain集成：Agent_as_Tool.md)
- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [RAG最佳实践指南](https://www.pinecone.io/learn/rag-best-practices)

---

## 下节预告

下一节我们将学习 **企业级AutoGen架构设计**，探讨如何在大规模生产环境中部署和管理AutoGen应用。