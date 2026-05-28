"""
AutoGen Agent 作为 LangChain Tool 的示例代码
=============================================
本文件展示如何将 AutoGen Agent 包装成 LangChain 可调用的 Tool，
实现混合 Agent 架构。

核心概念：
- AutoGen Agent: 独立的对话智能体，具有自己的系统提示和角色
- LangChain Tool: LangChain 框架中的可调用工具，遵循标准接口
- 接口映射: 将 AutoGen 的对话能力转换为 LangChain 的 tool_call 协议
"""

import os
from typing import Any, Callable, Dict, List, Optional, Type

# LangChain 相关导入
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# AutoGen 相关导入
try:
    import autogen
    from autogen import Agent, ConversableAgent, AssistantAgent
    from autogen.agentchat import AgentGram
    from autogen.code_utils import UNHANDLED_EXCEPTION_NAME, execute_code, gen_signature
except ImportError:
    raise ImportError("请先安装 autogen: pip install autogen")


# =============================================================================
# 第一部分：基础接口映射 - AutoGen Agent 包装为 LangChain Tool
# =============================================================================

class AutoGenToolInput(BaseModel):
    """
    定义 AutoGen Tool 的输入模式
    LangChain Tool 需要明确的输入模式来验证参数
    """
    message: str = Field(
        description="需要发送给 AutoGen Agent 的消息",
        default=""
    )
    session_id: Optional[str] = Field(
        description="会话ID，用于追踪多轮对话",
        default=None
    )


class AutoGenAgentTool(BaseTool):
    """
    将 AutoGen ConversableAgent 包装为 LangChain BaseTool

    包装策略：
    1. 保留 AutoGen Agent 的对话能力（系统提示、LLM配置）
    2. 实现 LangChain Tool 标准接口（_run, _arun）
    3. 维护会话状态，支持多轮对话

    使用场景：
    - 在 LangChain 链式中调用 AutoGen Agent
    - 作为 LangGraph 节点嵌入混合工作流
    - 与其他 LangChain 工具（如搜索、数据库）组合使用
    """

    # Tool 元数据
    name: str = "autogen_agent"
    description: str = "通用的 AutoGen Agent 包装工具，通过对话方式处理任务"
    args_schema: Type[BaseModel] = AutoGenToolInput

    def __init__(
        self,
        agent: ConversableAgent,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        初始化包装工具

        Args:
            agent: AutoGen ConversableAgent 实例
            session_id: 可选的会话ID，用于状态管理
            **kwargs: 传递给 BaseTool 的其他参数
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.session_id = session_id or "default_session"

        # 维护每个会话的消息历史
        # 注意：这里使用简单的字典存储，生产环境应使用数据库
        self._conversations: Dict[str, List[Dict]] = {}

    def _run(
        self,
        message: str,
        session_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """
        同步执行工具调用

        LangChain Tool 的标准执行入口。
        调用此方法时，LangChain 会将参数传递给 AutoGen Agent，
        并将 Agent 的回复作为工具结果返回。

        Args:
            message: 用户消息
            session_id: 会话标识，用于追踪对话
            run_manager: LangChain 回调管理器
            **kwargs: 其他参数（会被忽略）

        Returns:
            Agent 的回复文本
        """
        # 获取或创建会话
        sid = session_id or self.session_id
        if sid not in self._conversations:
            self._conversations[sid] = []

        # 获取 Agent 的回复
        # generate_reply 会触发 Agent 的 LLM 生成回复
        reply = self.agent.generate_reply(
            messages=self._conversations[sid],
            **kwargs
        )

        # 将用户消息和回复都记录到会话历史
        if reply is not None:
            self._conversations[sid].append({"role": "user", "content": message})
            self._conversations[sid].append({"role": "assistant", "content": reply})

        return reply or "Agent 未返回有效回复"

    async def _arun(
        self,
        message: str,
        session_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """
        异步执行工具调用

        LangChain 要求 Tool 同时实现同步和异步接口。
        对于 AutoGen Agent，我们使用 asyncio.to_thread 来处理同步调用。

        Args:
            message: 用户消息
            session_id: 会话标识
            run_manager: 回调管理器
            **kwargs: 其他参数

        Returns:
            Agent 的回复文本
        """
        import asyncio

        # 在线程池中执行同步的 _run 方法
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run,
            message,
            session_id,
            run_manager
        )
        return result


# =============================================================================
# 第二部分：使用装饰器创建结构化 Tool
# =============================================================================

def create_math_agent() -> ConversableAgent:
    """
    创建一个数学助手 Agent

    使用 AutoGen 的 AssistantAgent 配置数学计算能力
    """
    # 配置 LLM（请根据实际情况配置 API Key 和模型）
    llm_config = {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key"),
        # 可选：添加温度、上下文窗口等参数
        "temperature": 0.7,
    }

    # 创建数学助手 Agent
    math_agent = AssistantAgent(
        name="math_assistant",
        system_message="""你是一个专业的数学助手。
        当用户提出数学问题时，你应该：
        1. 理解问题的核心要求
        2. 提供清晰的解题步骤
        3. 最终给出准确的答案

        如果遇到无法计算的复杂问题，请说明原因并建议替代方案。""",
        llm_config=llm_config,
        # 禁用 Human 输入代理（因为作为 Tool 使用）
        human_input_mode="NEVER",
    )

    return math_agent


def create_code_agent() -> ConversableAgent:
    """
    创建一个代码助手 Agent

    使用 AutoGen 的 ConversableAgent 配置编程能力
    """
    llm_config = {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key"),
        "temperature": 0.7,
    }

    code_agent = ConversableAgent(
        name="code_assistant",
        system_message="""你是一个资深的 Python 程序员。
        当用户请求编写代码时，你应该：
        1. 先理解需求，明确输入和输出
        2. 编写简洁、高效、可读性强的代码
        3. 添加必要的注释说明

        你擅长处理数据分析、脚本编写、代码调试等任务。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    return code_agent


@tool
def calculate(expression: str) -> str:
    """
    数学计算工具（基于 AutoGen Agent）

    这是一个使用 AutoGen 数学 Agent 实现的计算工具。
    它演示了如何通过装饰器方式快速创建 LangChain Tool。

    注意：这个函数的实现直接调用了 Python 的 eval，
    生产环境中应使用更安全的计算方式（如 ast.literal_eval）

    Args:
        expression: 数学表达式，例如 "2 + 3 * 4" 或 "sqrt(16) + 5"

    Returns:
        计算结果的字符串表示
    """
    try:
        # 简单的安全计算实现
        # 警告：eval 在生产环境中存在安全风险，应使用 ast.literal_eval 或专用解析器
        import math

        # 将常用数学函数添加到命名空间
        allowed_names = {
            "sqrt": math.sqrt,
            "abs": abs,
            "pow": pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
        }

        # 使用有限的安全计算
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)

    except Exception as e:
        return f"计算错误: {str(e)}"


# =============================================================================
# 第三部分：创建工具集合并演示使用
# =============================================================================

def create_autogen_tools() -> List[BaseTool]:
    """
    创建包含多个 AutoGen Agent 的 LangChain 工具集合

    返回:
        List[BaseTool]: 可在 LangChain 中使用的工具列表
    """
    # 创建底层的 AutoGen Agents
    math_agent = create_math_agent()
    code_agent = create_code_agent()

    # 包装为 LangChain Tools
    math_tool = AutoGenAgentTool(
        agent=math_agent,
        name="math_assistant",
        description="数学助手工具，可以解决数学问题、计算表达式、提供解题步骤",
    )

    code_tool = AutoGenAgentTool(
        agent=code_agent,
        name="code_assistant",
        description="代码助手工具，可以编写 Python 代码、调试程序、数据分析",
    )

    return [math_tool, code_tool, calculate]


def demo_tool_usage():
    """
    演示如何在 LangChain 中使用 AutoGen 工具

    这个示例展示了完整的工具使用流程：
    1. 创建 AutoGen Agent
    2. 包装为 LangChain Tool
    3. 绑定到 LLM
    4. 执行工具调用
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import AgentType, initialize_agent

    # 初始化 LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
    )

    # 创建工具
    tools = create_autogen_tools()

    # 初始化 Agent
    # AgentType.CONVERSATIONAL_REACT_DESCRIPTION 是适合对话工具使用的类型
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )

    # 测试用例
    test_queries = [
        "请计算 125 + 378 的结果",
        "用 Python 写一个函数计算斐波那契数列第 n 项",
        "解释一下什么是梯度下降算法",
    ]

    print("=" * 60)
    print("AutoGen Agent 作为 LangChain Tool 的使用演示")
    print("=" * 60)

    for query in test_queries:
        print(f"\n[用户问题] {query}")
        try:
            response = agent.run(query)
            print(f"[Agent 回复] {response}")
        except Exception as e:
            print(f"[错误] {str(e)}")
        print("-" * 60)


# =============================================================================
# 第四部分：高级用法 - 带记忆的 Tool
# =============================================================================

class MemoryAutoGenTool(BaseTool):
    """
    带对话记忆功能的 AutoGen Tool

    继承自 BaseTool，添加了：
    1. 自动管理对话历史
    2. 支持上下文压缩（当对话过长时）
    3. 会话清理接口

    适用场景：
    - 需要多轮对话的任务
    - 对话上下文重要的场景
    - 需要维持长期会话状态的应用
    """

    name: str = "memory_autogen_tool"
    description: str = "带记忆的 AutoGen Agent 工具，支持多轮对话和上下文管理"
    args_schema: Type[BaseModel] = AutoGenToolInput

    # 最大历史消息数，超过后进行压缩
    MAX_HISTORY: int = 20

    def __init__(self, agent: ConversableAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self._conversations: Dict[str, List[Dict]] = {}
        self._metadata: Dict[str, Dict] = {}

    def _run(
        self,
        message: str,
        session_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """执行带记忆管理的工具调用"""
        sid = session_id or "default"

        # 初始化新会话
        if sid not in self._conversations:
            self._conversations[sid] = []
            self._metadata[sid] = {"created_at": None, "turns": 0}

        # 上下文压缩逻辑
        if len(self._conversations[sid]) > self.MAX_HISTORY:
            self._compress_context(sid)

        # 生成回复
        reply = self.agent.generate_reply(
            messages=self._conversations[sid],
            **kwargs
        )

        # 更新历史
        if reply is not None:
            self._conversations[sid].append({"role": "user", "content": message})
            self._conversations[sid].append({"role": "assistant", "content": reply})
            self._metadata[sid]["turns"] += 1

        return reply or "无回复"

    def _compress_context(self, session_id: str) -> None:
        """
        压缩对话上下文

        当对话历史过长时，保留最近的重要消息，
        合并早期消息为摘要，以节省上下文窗口。

        简化实现：直接保留最近的一半消息
        """
        current = self._conversations[session_id]
        # 保留最近的 N 条消息
        keep_count = self.MAX_HISTORY // 2
        self._conversations[session_id] = current[-keep_count:]

        # 添加摘要标记
        summary_msg = {
            "role": "system",
            "content": "[早期对话已压缩，保留关键上下文]"
        }
        self._conversations[session_id].insert(0, summary_msg)

    def clear_session(self, session_id: str) -> bool:
        """
        清理指定会话的历史记录

        Args:
            session_id: 要清理的会话ID

        Returns:
            是否成功清理
        """
        if session_id in self._conversations:
            self._conversations[session_id] = []
            self._metadata[session_id] = {"created_at": None, "turns": 0}
            return True
        return False

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        return self._metadata.get(session_id)


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    # 设置环境变量
    os.environ.setdefault("OPENAI_API_KEY", "your-api-key")

    print("AutoGen Agent 作为 LangChain Tool")
    print("=" * 60)
    print("本文件展示如何将 AutoGen Agent 包装为 LangChain Tool")
    print("包括：基础接口映射、装饰器创建、记忆管理等功能")
    print("=" * 60)

    # 演示基本用法（需要在环境中配置好 API Key）
    # demo_tool_usage()

    print("\n请查看代码中的注释和文档字符串了解详细用法。")
