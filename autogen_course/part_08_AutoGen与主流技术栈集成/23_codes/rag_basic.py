"""
AutoGen与RAG基础集成示例
=========================

本文件展示AutoGen与RAG（检索增强生成）系统的基础集成方法。

主要功能：
1. 模拟向量数据库和检索功能
2. 展示检索结果注入AutoGen Agent的多种方式
3. 实现简单的知识库问答系统

依赖：
    - autogen
    - chromadb (或模拟)
    - sentence-transformers (或模拟)

安装命令：
    pip install autogen chromadb sentence-transformers
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# ============================================================
# 第一部分：模拟RAG组件
# ============================================================

# 注意：在实际项目中，这些组件由真实的库提供
# 这里我们用简化模拟来演示集成模式

@dataclass
class Document:
    """
    文档数据类，表示知识库中的一篇文档

    属性:
        page_content: 文档内容
        metadata: 文档元数据（来源、时间等）
    """
    page_content: str
    metadata: Dict[str, str]


class SimpleVectorStore:
    """
    简化的向量存储类（模拟ChromaDB等向量数据库）

    实际生产环境中应使用：
    - ChromaDB: https://docs.trychroma.com/
    - Pinecone: https://docs.pinecone.io/
    - Milvus: https://milvus.io/docs
    """

    def __init__(self):
        """初始化向量存储"""
        # 模拟文档存储（实际中是向量）
        self._documents = []
        # 模拟嵌入向量
        self._embeddings = []

    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量存储

        参数:
            documents: Document对象列表
        """
        for doc in documents:
            self._documents.append(doc)
            # 模拟生成嵌入向量（实际使用Embedding模型）
            self._embeddings.append(self._mock_embedding(doc.page_content))

        print(f"[向量存储] 已添加 {len(documents)} 篇文档")

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.5
    ) -> List[Document]:
        """
        执行相似度检索

        参数:
            query: 用户查询
            k: 返回的最相似文档数量
            threshold: 相似度阈值

        返回:
            最相似的文档列表
        """
        # 模拟查询嵌入
        query_embedding = self._mock_embedding(query)

        # 计算相似度（简化版本，实际使用余弦相似度等）
        similarities = []
        for i, doc_emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_embedding, doc_emb)
            similarities.append((sim, i))

        # 按相似度排序
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 返回top-k结果
        results = []
        for sim, idx in similarities[:k]:
            if sim >= threshold:
                results.append(self._documents[idx])

        return results

    def _mock_embedding(self, text: str) -> List[float]:
        """
        生成模拟嵌入向量（实际应使用Embedding模型）

        参数:
            text: 输入文本

        返回:
            模拟的嵌入向量
        """
        # 简化：用文本长度和字符ASCII码生成伪向量
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        seed = hash_val % 1000 / 1000.0

        # 生成固定维度的模拟向量
        return [seed + i * 0.01 for i in range(10)]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        参数:
            vec1: 向量1
            vec2: 向量2

        返回:
            余弦相似度值
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class SimpleEmbeddings:
    """
    简化的嵌入模型类（模拟OpenAI Embeddings等）

    实际生产环境中应使用：
    - OpenAI: openai.embeddings_utils
    - HuggingFace: sentence-transformers
    """

    @staticmethod
    def embed_text(text: str) -> List[float]:
        """
        将文本转换为嵌入向量

        参数:
            text: 输入文本

        返回:
            嵌入向量
        """
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        seed = hash_val % 1000 / 1000.0
        return [seed + i * 0.01 for i in range(10)]


# ============================================================
# 第二部分：知识库初始化
# ============================================================

def initialize_knowledge_base() -> SimpleVectorStore:
    """
    初始化知识库，创建向量存储并添加示例文档

    返回:
        初始化好的向量存储对象
    """
    # 创建向量存储
    vector_store = SimpleVectorStore()

    # 定义知识库文档（AutoGen相关知识）
    documents = [
        Document(
            page_content="""
            AutoGen是微软开发的一个开源框架，用于构建多Agent对话系统。
            它支持双Agent对话、GroupChat、层次化聊天等多种对话模式。
            核心类是ConversableAgent，它封装了LLM调用、代码执行、工具调用等功能。
            """,
            metadata={"source": "AutoGen概述", "category": "基础概念"}
        ),
        Document(
            page_content="""
            ConversableAgent是AutoGen的核心类，它具有以下组件：
            1. LLM配置（llm_config）：配置大模型连接
            2. 代码执行器（code_executor）：执行Python代码
            3. 工具执行器（function_executor）：调用自定义工具
            4. 人类介入（human_input_mode）：控制人类参与方式
            """,
            metadata={"source": "ConversableAgent文档", "category": "核心机制"}
        ),
        Document(
            page_content="""
            Agent间通信通过initiate_chat方法实现：
            agent.initiate_chat(recipient, message)
            这会启动与目标Agent的对话，并传递初始消息。
            对话会一直持续直到满足终止条件（is_termination_msg）或达到最大轮次。
            """,
            metadata={"source": "Agent通信文档", "category": "核心机制"}
        ),
        Document(
            page_content="""
            GroupChat是AutoGen中实现多Agent群聊的机制。
            通过GroupChatManager管理多个Agent的对话。
            speaker_selection_mode控制下一个发言者的选择策略：
            - auto: 由LLM推荐
            - manual: 手动指定
            - allow_repeat: 允许重复发言
            """,
            metadata={"source": "GroupChat文档", "category": "多Agent协作"}
        ),
        Document(
            page_content="""
            RAG（检索增强生成）是一种将检索与生成结合的技术。
            RAG系统通常包含：文档加载器、文本分割器、向量化模型、向量数据库、检索器、生成器。
            AutoGen可以与RAG集成，让Agent具备知识检索能力。
            """,
            metadata={"source": "RAG文档", "category": "RAG技术"}
        ),
    ]

    # 添加文档到向量存储
    vector_store.add_documents(documents)

    return vector_store


# ============================================================
# 第三部分：检索功能封装
# ============================================================

class RAGRetriever:
    """
    RAG检索器封装类

    封装检索逻辑，提供统一的检索接口
    """

    def __init__(self, vector_store: SimpleVectorStore):
        """
        初始化检索器

        参数:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        max_context_tokens: int = 4000
    ) -> Tuple[List[Document], str]:
        """
        执行检索并返回组装后的上下文

        参数:
            query: 用户查询
            top_k: 检索的文档数量
            max_context_tokens: 最大上下文token数（估算）

        返回:
            元组：(文档列表, 组装后的上下文字符串)
        """
        # 执行相似度检索
        docs = self.vector_store.similarity_search(query, k=top_k)

        # 组装上下文
        context_parts = []
        total_chars = 0

        for doc in docs:
            # 估算：每4个字符约等于1个token
            doc_tokens = len(doc.page_content) // 4

            if total_chars + doc_tokens > max_context_tokens:
                break

            context_parts.append(f"【来源：{doc.metadata['source']}】\n{doc.page_content}")
            total_chars += doc_tokens

        context = "\n\n---\n\n".join(context_parts)

        return docs, context

    def format_sources(self, docs: List[Document]) -> str:
        """
        格式化文档来源信息

        参数:
            docs: 文档列表

        返回:
            格式化的来源字符串
        """
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append(
                f"{i}. [{doc.metadata['source']}] - 类别：{doc.metadata['category']}"
            )
        return "\n".join(sources)


# ============================================================
# 第四部分：AutoGen与RAG集成
# ============================================================

def create_rag_agent(llm_config: dict) -> tuple:
    """
    创建集成了RAG能力的Agent

    参数:
        llm_config: LLM配置字典

    返回:
        元组：(AssistantAgent, UserProxyAgent, RAGRetriever)
    """
    from autogen import AssistantAgent, UserProxyAgent

    # 初始化RAG组件
    vector_store = initialize_knowledge_base()
    retriever = RAGRetriever(vector_store)

    # 创建AssistantAgent
    assistant = AssistantAgent(
        name="rag_assistant",
        system_message="""
        你是一个基于知识库检索的AI助手。

        你的回答应该：
        1. 基于检索到的知识内容进行回答
        2. 明确标注信息来源
        3. 如果检索内容不足以回答问题，请诚实说明

        回答格式：
        回答内容...

        ---
        参考来源：
        [列出引用的来源]
        """,
        llm_config=llm_config,
        max_consecutive_auto_reply=3,
    )

    # 创建UserProxyAgent
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_executor="local"
    )

    return assistant, user_proxy, retriever


def query_with_rag(
    user_proxy,
    assistant,
    retriever: RAGRetriever,
    question: str
) -> None:
    """
    使用RAG检索增强的方式查询

    参数:
        user_proxy: UserProxyAgent实例
        assistant: AssistantAgent实例
        retriever: RAG检索器实例
        question: 用户问题
    """
    # 执行RAG检索
    docs, context = retriever.retrieve(question)

    print(f"\n{'='*60}")
    print(f"用户问题：{question}")
    print(f"{'='*60}")
    print(f"\n检索到 {len(docs)} 篇相关文档")

    if not docs:
        print("\n未找到相关文档，直接由LLM回答...")
        user_proxy.initiate_chat(
            assistant,
            message=f"问题：{question}"
        )
        return

    # 方式1：直接使用检索到的上下文回答
    print("\n检索上下文：")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 40)

    # 构建带上下文的提示
    prompt = f"""
请基于以下检索到的知识内容回答问题。

【检索到的知识】
{context}

【用户问题】
{question}

请在回答中标注信息来源。
"""

    user_proxy.initiate_chat(
        assistant,
        message=prompt
    )


# ============================================================
# 第五部分：注入方式对比
# ============================================================

def demonstrate_injection_methods():
    """
    演示三种不同的检索结果注入方式
    """
    from autogen import AssistantAgent

    llm_config = {
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }

    retrieved_context = """
    AutoGen框架的核心特性：
    1. 多Agent协作：支持双Agent和多Agent群聊
    2. 灵活对话模式：支持同步和异步对话
    3. 代码执行能力：内置Python代码执行器
    4. 工具集成：支持Function Calling机制
    """

    # 方式1：通过system_message注入（初始化时）
    print("\n方式1：通过system_message注入（静态方式）")
    agent1 = AssistantAgent(
        name="agent_static",
        system_message=f"""
        你是一个知识库问答助手。

        【系统知识】
        {retrieved_context}

        请基于以上知识回答用户的问题。
        """,
        llm_config=llm_config
    )
    print(f"  - 特点：知识在初始化时固定，适合长期使用的知识")
    print(f"  - 限制：每次对话都包含相同的知识，无法动态更新")

    # 方式2：通过message参数注入（动态方式）
    print("\n方式2：通过message参数注入（动态方式）")
    print(f"  - 特点：每次对话可以有不同的检索上下文")
    print(f"  - 限制：需要在每次调用时准备检索结果")
    # 使用示例：
    # agent.initiate_chat(recipient, message=f"基于以下知识：{context}\n\n问题：{question}")

    # 方式3：通过register_reply注入（条件触发方式）
    print("\n方式3：通过register_reply注入（条件触发方式）")
    def conditional_rag_reply(message, sender, config):
        """
        条件触发的RAG回复函数

        当消息中包含特定关键词时，自动触发RAG检索
        """
        content = message.get("content", "")

        # 检测是否需要触发RAG
        if any(keyword in content for keyword in ["知识库", "检索", "查询"]):
            # 执行检索
            retriever = config["retriever"]
            docs, context = retriever.retrieve(content)

            if docs:
                # 将检索结果注入回复
                return f"【基于知识库检索】\n\n{context}\n\n---\n\n{content}"
            else:
                return content
        else:
            return None  # 返回None表示不拦截，正常处理

    agent3 = AssistantAgent(
        name="agent_conditional",
        system_message="你是一个智能助手...",
        llm_config=llm_config
    )

    # 注册条件触发RAG回复
    agent3.register_reply(
        trigger=lambda msg: any(kw in msg.get("content", "") for kw in ["知识库", "检索"]),
        reply_func=conditional_rag_reply,
        config={"retriever": RAGRetriever(initialize_knowledge_base())}
    )

    print(f"  - 特点：根据条件自动触发RAG检索，灵活性高")
    print(f"  - 限制：需要合理设计触发条件")


# ============================================================
# 第六部分：主程序
# ============================================================

def main():
    """
    主程序入口
    """
    print("=" * 60)
    print("AutoGen与RAG基础集成示例")
    print("=" * 60)

    # 检查API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n警告：未设置OPENAI_API_KEY环境变量")
        print("请设置后再运行完整示例")

    # 演示注入方式
    print("\n\n【演示：检索结果注入方式】")
    demonstrate_injection_methods()

    # 如果有API Key，运行完整RAG示例
    if os.getenv("OPENAI_API_KEY"):
        llm_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
        }

        # 创建RAG Agent
        assistant, user_proxy, retriever = create_rag_agent(llm_config)

        # 示例查询
        questions = [
            "AutoGen的核心类是什么？它有哪些组件？",
            "GroupChat的speaker_selection_mode有哪些选项？",
            "RAG是什么？AutoGen如何与RAG集成？",
        ]

        print("\n\n【RAG知识库问答示例】")
        for q in questions:
            query_with_rag(user_proxy, assistant, retriever, q)
            print("\n")

    else:
        print("\n\n【跳过API调用示例】")
        print("如需运行完整示例，请设置OPENAI_API_KEY环境变量")


if __name__ == "__main__":
    main()