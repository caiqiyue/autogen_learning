"""
AutoGen与RAG高级集成示例
=========================

本文件展示AutoGen与RAG系统的高级集成方法，包括：
1. 混合检索：向量检索 + BM25关键词检索
2. AutoGen多Agent协作与RAG结合
3. LangGraph工作流编排
4. 重排序机制实现
5. 高级知识库问答系统架构

依赖：
    - autogen
    - chromadb / faiss
    - rank_bm25
    - langgraph

安装命令：
    pip install autogen chromadb faiss-cpu rank-bm25 langgraph
"""

import os
from typing import List, Dict, TypedDict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import Counter
import math

# ============================================================
# 第一部分：高级检索组件
# ============================================================

@dataclass
class Document:
    """
    文档数据类

    属性:
        page_content: 文档内容
        metadata: 文档元数据
        embedding: 向量嵌入（可选）
    """
    page_content: str
    metadata: Dict[str, str]
    embedding: Optional[List[float]] = None


class BM25Retriever:
    """
    BM25关键词检索器

    BM25是一种基于关键词的检索算法，相比纯向量检索，
    对精确关键词匹配更友好，可以作为向量检索的补充。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器

        参数:
            k1: 词频饱和参数，控制词频对相关性的影响
            b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}  # 词频统计
        self.idf = {}  # 逆文档频率
        self.doc_tokens = []  # 每个文档的分词结果

    def fit(self, documents: List[Document]) -> None:
        """
        构建BM25索引

        参数:
            documents: 文档列表
        """
        # 分词（简化版，实际应使用jieba等分词库）
        self.doc_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1

        # 统计词频和文档频率
        N = len(documents)
        term_freqs = Counter()

        for tokens in self.doc_tokens:
            term_freqs.update(set(tokens))  # 去重统计

        # 计算IDF
        for term, df in term_freqs.items():
            self.doc_freqs[term] = df
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词（实际应使用专业分词库）

        参数:
            text: 输入文本

        返回:
            分词列表
        """
        # 简单按空格和标点分词
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        执行BM25检索

        参数:
            query: 查询字符串
            top_k: 返回结果数量

        返回:
            元组列表：(文档索引, BM25分数)
        """
        query_tokens = self._tokenize(query.lower())
        scores = []

        for i, doc_tokens in enumerate(self.doc_tokens):
            score = self._calculate_bm25(query_tokens, doc_tokens, i)
            scores.append((i, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _calculate_bm25(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        doc_idx: int
    ) -> float:
        """
        计算单文档的BM25分数

        参数:
            query_tokens: 查询分词列表
            doc_tokens: 文档分词列表
            doc_idx: 文档索引

        返回:
            BM25分数
        """
        doc_tf = Counter(doc_tokens)
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term in doc_tf:
                tf = doc_tf[term]
                idf = self.idf.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator

        return score


class HybridRetriever:
    """
    混合检索器：结合向量检索和BM25检索

    混合检索可以同时利用：
    - 向量检索：捕捉语义相似性
    - BM25检索：精确关键词匹配
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        初始化混合检索器

        参数:
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.vector_store = None
        self.bm25_retriever = None

    def build_index(self, documents: List[Document]) -> None:
        """
        构建混合索引

        参数:
            documents: 文档列表
        """
        # 构建向量索引（使用简化的模拟）
        from rag_basic import SimpleVectorStore
        self.vector_store = SimpleVectorStore()
        self.vector_store._documents = documents
        self.vector_store._embeddings = [
            self.vector_store._mock_embedding(doc.page_content) for doc in documents
        ]

        # 构建BM25索引
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.fit(documents)

        print(f"[混合检索器] 已构建索引，文档数：{len(documents)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        执行混合检索

        参数:
            query: 查询字符串
            top_k: 返回结果数量
            vector_weight: 可覆盖的向量权重
            bm25_weight: 可覆盖的BM25权重

        返回:
            元组列表：(文档, 综合分数)
        """
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        b_weight = bm25_weight if bm25_weight is not None else self.bm25_weight

        # 向量检索
        vector_results = self.vector_store.similarity_search(query, k=top_k * 2) if self.vector_store else []
        vector_scores = {doc.metadata.get("id", id(doc)): i / len(vector_results) * v_weight
                        for i, doc in enumerate(vector_results)}

        # BM25检索
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2) if self.bm25_retriever else []
        bm25_scores = {}
        for i, (idx, score) in enumerate(bm25_results):
            if idx < len(self.vector_store._documents):
                doc = self.vector_store._documents[idx]
                doc_id = doc.metadata.get("id", str(idx))
                bm25_scores[doc_id] = (1 - i / len(bm25_results)) * b_weight

        # 合并分数
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined_scores = []

        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            total_score = v_score + b_score

            # 找到对应的文档
            for doc in self.vector_store._documents:
                if doc.metadata.get("id", id(doc)) == doc_id:
                    combined_scores.append((doc, total_score))
                    break

        # 按分数排序
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]


# ============================================================
# 第二部分：重排序机制
# ============================================================

class Reranker:
    """
    检索结果重排序器

    使用交叉编码器或其他重排序模型对初步检索结果进行二次排序，
    以提高排序质量。
    """

    def __init__(self, top_n: int = 10):
        """
        初始化重排序器

        参数:
            top_n: 重排序后返回的结果数量
        """
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重排序

        参数:
            query: 用户查询
            documents: 初步检索到的文档列表
            top_k: 返回结果数量（默认为初始化时的top_n）

        返回:
            重排序后的元组列表：(文档, 重排序分数)
        """
        if not documents:
            return []

        k = top_k if top_k is not None else self.top_n

        # 模拟交叉编码器评分（实际应使用CrossEncoder）
        # 交叉编码器会对(query, document)进行联合编码，计算相关性分数
        scores = []
        for doc in documents:
            # 模拟评分：考虑查询关键词在文档中的出现情况
            score = self._mock_cross_encoder_score(query, doc)
            scores.append(score)

        # 按分数排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"[重排序器] 已对 {len(documents)} 篇文档进行重排序")
        return doc_scores[:k]

    def _mock_cross_encoder_score(self, query: str, doc: Document) -> float:
        """
        模拟交叉编码器评分（实际应使用真实的CrossEncoder模型）

        参数:
            query: 用户查询
            doc: 文档对象

        返回:
            相关性分数
        """
        # 简单模拟：查询词在文档中出现得越多，分数越高
        query_terms = set(query.lower().split())
        doc_terms = set(doc.page_content.lower().split())

        # 计算覆盖率
        coverage = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0

        # 考虑文档长度归一化
        length_factor = min(1.0, 500 / len(doc.page_content)) if doc.page_content else 0

        # 综合评分
        score = coverage * 0.7 + length_factor * 0.3

        # 元数据加成
        if "category" in doc.metadata:
            if any(cat in doc.metadata["category"] for cat in ["核心机制", "基础概念"]):
                score *= 1.2  # 核心文档加分

        return min(score, 1.0)


# ============================================================
# 第三部分：LangGraph工作流编排
# ============================================================

class RAGState(TypedDict):
    """
    RAG工作流状态定义

    属性:
        question: 用户问题
        intent: 分类的意图类型
        retrieved_docs: 检索到的文档列表
        reranked_docs: 重排序后的文档列表
        context: 组装后的检索上下文
        answer: 最终生成的回答
        sources: 引用的来源信息
    """
    question: str
    intent: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    context: str
    answer: str
    sources: List[str]


class RAGWorkflow:
    """
    基于LangGraph的RAG工作流编排器

    工作流程：
    1. 意图分类 -> 决定检索策略
    2. 混合检索 -> 获取候选文档
    3. 重排序 -> 提高排序质量
    4. 上下文组装 -> 准备生成输入
    5. 答案生成 -> AutoGen Agent生成最终答案
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm_config: dict
    ):
        """
        初始化RAG工作流

        参数:
            retriever: 混合检索器
            reranker: 重排序器
            llm_config: LLM配置
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm_config = llm_config
        self._setup_workflow()

    def _classify_intent(self, question: str) -> str:
        """
        意图分类：根据问题类型决定处理策略

        参数:
            question: 用户问题

        返回:
            意图类型：knowledge/search/direct
        """
        # 简单关键词判断（实际应使用分类模型）
        knowledge_keywords = ["是什么", "定义", "概念", "原理", "机制", "哪些", "有什么"]
        search_keywords = ["查找", "搜索", "最新", "现在的", "当前"]

        if any(kw in question for kw in knowledge_keywords):
            return "knowledge"
        elif any(kw in question for kw in search_keywords):
            return "search"
        else:
            return "direct"

    def _assemble_context(
        self,
        documents: List[Document],
        max_tokens: int = 4000
    ) -> Tuple[str, List[str]]:
        """
        组装检索上下文

        参数:
            documents: 文档列表
            max_tokens: 最大token数

        返回:
            元组：(上下文字符串, 来源列表)
        """
        context_parts = []
        sources = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = len(doc.page_content) // 4

            if total_tokens + doc_tokens > max_tokens:
                break

            context_parts.append(f"【来源：{doc.metadata['source']}】\n{doc.page_content}")
            sources.append(f"- {doc.metadata['source']} (类别：{doc.metadata['category']})")
            total_tokens += doc_tokens

        return "\n\n---\n\n".join(context_parts), sources

    def query(self, question: str) -> Dict:
        """
        执行完整的RAG查询流程

        参数:
            question: 用户问题

        返回:
            包含答案和相关信息的字典
        """
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        # 1. 意图分类
        intent = self._classify_intent(question)
        print(f"\n[步骤1] 意图分类：{intent}")

        # 2. 检索
        print("[步骤2] 执行混合检索...")
        results = self.retriever.search(question, top_k=10)
        retrieved_docs = [doc for doc, _ in results]
        print(f"        检索到 {len(retrieved_docs)} 篇候选文档")

        # 3. 重排序
        print("[步骤3] 执行重排序...")
        reranked = self.reranker.rerank(question, retrieved_docs, top_k=5)
        reranked_docs = [doc for doc, _ in reranked]
        print(f"        重排序后保留 {len(reranked_docs)} 篇文档")

        # 4. 上下文组装
        print("[步骤4] 组装检索上下文...")
        context, sources = self._assemble_context(reranked_docs)
        print(f"        上下文长度：{len(context)} 字符")

        # 5. 答案生成
        print("[步骤5] 生成最终答案...")

        # 如果有检索结果，使用RAG增强回答
        if reranked_docs:
            prompt = f"""
请基于以下检索到的知识内容回答问题。
如果没有足够的检索信息，请基于你的知识回答，但要说明这一点。

【检索到的知识】
{context}

【用户问题】
{question}

请在回答中适当引用信息来源。
"""
        else:
            # 无检索结果，直接生成
            prompt = f"问题：{question}\n\n请直接回答。"

        # 这里简化处理，实际应调用AutoGen Agent生成答案
        # answer = autogen_agent.generate_answer(prompt)
        answer = f"[模拟答案] 基于检索内容对'{question}'的回答..."

        print(f"\n{'='*60}")
        print("生成的回答：")
        print(answer)
        print(f"{'='*60}")
        print("\n参考来源：")
        for src in sources:
            print(f"  {src}")

        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "sources": sources,
            "retrieved_docs": reranked_docs
        }


# ============================================================
# 第四部分：AutoGen多Agent协作与RAG
# ============================================================

class MultiAgentRAG:
    """
    多Agent协作的RAG系统

    架构：
    - Router Agent：分类用户问题，决定路由
    - Retriever Agent：执行知识检索
    - Synthesizer Agent：综合检索结果生成答案
    """

    def __init__(self, llm_config: dict):
        """
        初始化多Agent RAG系统

        参数:
            llm_config: LLM配置字典
        """
        self.llm_config = llm_config
        self.agents = {}
        self.retriever = None

    def setup_agents(self):
        """设置各个Agent"""
        from autogen import AssistantAgent

        # Router Agent：意图分类和路由
        self.agents["router"] = AssistantAgent(
            name="router",
            system_message="""
            你是一个智能路由助手。

            你的职责：
            1. 分析用户问题的类型
            2. 决定使用哪种检索策略

            问题类型和对应策略：
            - knowledge：知识类问题，使用知识库检索
            - search：搜索类问题，使用网络搜索
            - direct：直接回答类问题，不需要检索

            输出格式：
            STRATEGY: <strategy>
            REASON: <选择该策略的原因>
            """,
            llm_config=self.llm_config,
            max_consecutive_auto_reply=2,
        )

        # Retriever Agent：知识检索
        self.agents["retriever"] = AssistantAgent(
            name="retriever",
            system_message="""
            你是一个专业的知识检索助手。

            你的职责：
            1. 理解检索查询
            2. 评估检索结果的相关性
            3. 判断是否需要补充检索

            输出格式：
            RETRIEVAL_RESULTS:
            <列出检索到的关键信息>

            RELEVANCE: <高/中/低>
            NEED_MORE: <是/否>
            """,
            llm_config=self.llm_config,
            max_consecutive_auto_reply=3,
        )

        # Synthesizer Agent：综合生成
        self.agents["synthesizer"] = AssistantAgent(
            name="synthesizer",
            system_message="""
            你是一个答案综合助手。

            你的职责：
            1. 综合多个检索来源的信息
            2. 生成连贯、准确的回答
            3. 标注信息来源

            输出格式：
            ANSWER:
            <完整回答>

            SOURCES:
            <信息来源列表>
            """,
            llm_config=self.llm_config,
            max_consecutive_auto_reply=2,
        )

    def set_retriever(self, retriever: HybridRetriever):
        """
        设置检索器

        参数:
            retriever: 混合检索器实例
        """
        self.retriever = retriever

    def query(self, question: str) -> Dict:
        """
        执行多Agent协作查询

        参数:
            question: 用户问题

        返回:
            查询结果字典
        """
        print(f"\n{'='*60}")
        print(f"多Agent RAG 查询：{question}")
        print(f"{'='*60}")

        # 阶段1：Router分类
        print("\n[阶段1] Router Agent 意图分类...")
        router_response = self._call_agent(
            "router",
            f"请分析以下问题并决定处理策略：\n{question}"
        )
        print(f"Router响应：{router_response[:200]}...")

        # 解析策略
        strategy = "direct"
        for line in router_response.split("\n"):
            if line.startswith("STRATEGY:"):
                strategy = line.split(":")[1].strip().lower()

        # 阶段2：根据策略处理
        retrieval_context = ""
        if strategy in ["knowledge", "search"] and self.retriever:
            print("\n[阶段2] Retriever Agent 执行检索...")
            # 执行检索
            results = self.retriever.search(question, top_k=5)
            retrieved_docs = [doc for doc, _ in results]

            # 组装检索上下文
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(
                    f"【来源：{doc.metadata.get('source', 'unknown')}】\n{doc.page_content}"
                )
            retrieval_context = "\n\n---\n\n".join(context_parts)

            print(f"检索到 {len(retrieved_docs)} 篇相关文档")
        else:
            print("\n[阶段2] 无需检索，直接生成")

        # 阶段3：Synthesizer综合生成
        print("\n[阶段3] Synthesizer Agent 综合生成...")
        if retrieval_context:
            synthesis_prompt = f"""
基于以下检索上下文回答问题。

【检索上下文】
{retrieval_context}

【用户问题】
{question}

请生成完整、准确的回答，并标注信息来源。
"""
        else:
            synthesis_prompt = f"请直接回答以下问题：\n{question}"

        answer = self._call_agent("synthesizer", synthesis_prompt)

        print(f"\n{'='*60}")
        print("最终回答：")
        print(answer)
        print(f"{'='*60}")

        return {
            "question": question,
            "strategy": strategy,
            "answer": answer,
            "retrieved_count": len(retrieved_docs) if retrieval_context else 0
        }

    def _call_agent(self, agent_name: str, message: str) -> str:
        """
        调用指定Agent（简化版，实际应使用真实的Agent通信）

        参数:
            agent_name: Agent名称
            message: 发送的消息

        返回:
            Agent的响应
        """
        # 简化：直接返回模拟响应
        # 实际应使用 agent.initiate_chat() 进行真实通信
        agent = self.agents.get(agent_name)
        if not agent:
            return f"[错误] Agent {agent_name} 不存在"

        # 模拟Agent响应
        if agent_name == "router":
            if any(kw in message for kw in ["什么", "是", "原理"]):
                return "STRATEGY: knowledge\nREASON: 知识类问题，需要检索"
            else:
                return "STRATEGY: direct\nREASON: 简单问题，直接回答"

        elif agent_name == "retriever":
            return f"RETRIEVAL_RESULTS:\n检索到相关信息...\n\nRELEVANCE: 高\nNEED_MORE: 否"

        elif agent_name == "synthesizer":
            return f"根据检索内容，这是一个关于...'的回答。\n\n信息来源：知识库文档"

        return "[模拟响应]"


# ============================================================
# 第五部分：知识库管理和监控
# ============================================================

class RAGMonitor:
    """
    RAG系统监控器

    监控RAG系统的性能指标，包括：
    - 检索命中率
    - 答案质量评分
    - Token消耗统计
    """

    def __init__(self):
        """初始化监控器"""
        self.query_count = 0
        self.retrieval_stats = []
        self.answer_stats = []

    def log_query(
        self,
        question: str,
        retrieved_count: int,
        answer_length: int,
        latency: float
    ) -> None:
        """
        记录查询统计

        参数:
            question: 用户问题
            retrieved_count: 检索到的文档数
            answer_length: 答案长度
            latency: 响应延迟（秒）
        """
        self.query_count += 1
        self.retrieval_stats.append({
            "query": question,
            "retrieved": retrieved_count,
            "answer_length": answer_length,
            "latency": latency
        })

    def get_stats(self) -> Dict:
        """
        获取统计信息

        返回:
            统计字典
        """
        if not self.retrieval_stats:
            return {
                "total_queries": 0,
                "avg_retrieved": 0,
                "avg_answer_length": 0,
                "avg_latency": 0
            }

        return {
            "total_queries": self.query_count,
            "avg_retrieved": sum(s["retrieved"] for s in self.retrieval_stats) / len(self.retrieval_stats),
            "avg_answer_length": sum(s["answer_length"] for s in self.retrieval_stats) / len(self.retrieval_stats),
            "avg_latency": sum(s["latency"] for s in self.retrieval_stats) / len(self.retrieval_stats)
        }

    def print_report(self) -> None:
        """打印监控报告"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("RAG系统监控报告")
        print("=" * 50)
        print(f"总查询数：{stats['total_queries']}")
        print(f"平均检索文档数：{stats['avg_retrieved']:.1f}")
        print(f"平均答案长度：{stats['avg_answer_length']:.1f} 字符")
        print(f"平均响应延迟：{stats['avg_latency']:.2f} 秒")
        print("=" * 50)


# ============================================================
# 第六部分：主程序
# ============================================================

def create_sample_documents() -> List[Document]:
    """
    创建示例文档库

    返回:
        文档列表
    """
    docs = [
        Document(
            page_content="AutoGen是微软开发的开源多Agent框架，支持双Agent对话、GroupChat、层次化聊天等多种模式。核心类是ConversableAgent。",
            metadata={"id": "doc_1", "source": "AutoGen概述", "category": "基础概念"}
        ),
        Document(
            page_content="ConversableAgent包含四大组件：LLM配置(llm_config)、代码执行器(code_executor)、工具执行器(function_executor)、人类介入(human_input_mode)。",
            metadata={"id": "doc_2", "source": "ConversableAgent文档", "category": "核心机制"}
        ),
        Document(
            page_content="GroupChat是AutoGen的群聊机制，通过GroupChatManager管理。speaker_selection_mode控制发言者选择，有auto、manual、allow_repeat三种模式。",
            metadata={"id": "doc_3", "source": "GroupChat文档", "category": "多Agent协作"}
        ),
        Document(
            page_content="RAG(检索增强生成)结合了检索系统和生成模型，可以利用外部知识库增强LLM的生成能力，减少幻觉问题。",
            metadata={"id": "doc_4", "source": "RAG文档", "category": "RAG技术"}
        ),
        Document(
            page_content="混合检索结合向量检索和关键词检索(BM25)，可以同时利用语义相似性和精确关键词匹配，提高检索质量。",
            metadata={"id": "doc_5", "source": "混合检索文档", "category": "RAG技术"}
        ),
    ]
    return docs


def main():
    """
    主程序入口
    """
    print("=" * 60)
    print("AutoGen与RAG高级集成示例")
    print("=" * 60)

    # 检查环境
    if not os.getenv("OPENAI_API_KEY"):
        print("\n警告：未设置OPENAI_API_KEY，跳过API调用示例")
        print("将演示基础组件功能\n")

    # 创建示例文档库
    documents = create_sample_documents()
    print(f"\n已创建包含 {len(documents)} 篇文档的知识库")

    # 初始化混合检索器
    print("\n【初始化混合检索器】")
    hybrid_retriever = HybridRetriever(vector_weight=0.6, bm25_weight=0.4)
    hybrid_retriever.build_index(documents)

    # 初始化重排序器
    reranker = Reranker(top_n=5)

    # 测试混合检索
    print("\n【测试混合检索】")
    test_queries = [
        "AutoGen的核心机制是什么？",
        "GroupChat有哪些模式？",
        "RAG技术的原理"
    ]

    for query in test_queries:
        print(f"\n查询：{query}")
        results = hybrid_retriever.search(query, top_k=5)
        print(f"检索到 {len(results)} 个结果：")
        for doc, score in results:
            print(f"  - [{doc.metadata['source']}] (分数: {score:.3f})")

    # 测试重排序
    print("\n【测试重排序】")
    query = "AutoGen的ConversableAgent机制"
    vector_results = hybrid_retriever.vector_store.similarity_search(query, k=5)
    reranked = reranker.rerank(query, vector_results)
    print(f"重排序后：")
    for doc, score in reranked:
        print(f"  - [{doc.metadata['source']}] (分数: {score:.3f})")

    # 测试LangGraph工作流
    if os.getenv("OPENAI_API_KEY"):
        print("\n【测试LangGraph工作流】")
        llm_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
        }

        workflow = RAGWorkflow(hybrid_retriever, reranker, llm_config)
        result = workflow.query("请解释AutoGen的ConversableAgent")
        print(f"\n工作流结果：{result}")
    else:
        print("\n【跳过LangGraph工作流测试】（需要API Key）")

    # 测试多Agent协作
    if os.getenv("OPENAI_API_KEY"):
        print("\n【测试多Agent RAG系统】")
        multi_agent_rag = MultiAgentRAG(llm_config)
        multi_agent_rag.setup_agents()
        multi_agent_rag.set_retriever(hybrid_retriever)
        result = multi_agent_rag.query("AutoGen的核心类是什么？")
        print(f"\n多Agent结果：{result}")
    else:
        print("\n【跳过多Agent测试】（需要API Key）")

    # RAG系统监控
    print("\n【RAG系统监控】")
    monitor = RAGMonitor()
    monitor.log_query("AutoGen是什么", 5, 200, 1.5)
    monitor.log_query("GroupChat用法", 3, 150, 1.2)
    monitor.log_query("RAG原理", 4, 180, 1.3)
    monitor.print_report()

    print("\n" + "=" * 60)
    print("高级RAG示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()