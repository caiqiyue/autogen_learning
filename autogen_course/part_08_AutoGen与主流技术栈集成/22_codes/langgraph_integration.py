"""
AutoGen 与 LangGraph 集成示例代码
=================================
本文件展示如何将 AutoGen Agent / GroupChat 嵌入 LangGraph 状态机，
实现复杂的混合 Agent 工作流。

核心概念：
- LangGraph: 基于有向图的工作流编排框架，适合构建状态机
- StateGraph: LangGraph 的核心数据结构，用于定义图结构
- AutoGen GroupChat: 多 Agent 协作对话机制
- 条件节点: LangGraph 中根据状态决定下一步路径的节点

与 agent_as_tool.py 的区别：
- agent_as_tool.py 侧重于将 Agent 包装为 Tool
- 本文件侧重于将 Agent 嵌入图结构，作为节点参与工作流编排
"""

import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union
from typing_extensions import TypedDict

# LangGraph 核心导入
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    raise ImportError("请先安装 langgraph: pip install langgraph")

# LangChain 工具相关
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# AutoGen 相关导入
try:
    import autogen
    from autogen import Agent, ConversableAgent, AssistantAgent
    from autogen.agentchat.groupchat import GroupChat, GroupChatManager
except ImportError:
    raise ImportError("请先安装 autogen: pip install autogen")


# =============================================================================
# 第一部分：LangGraph 状态定义
# =============================================================================

class AgentState(TypedDict):
    """
    LangGraph 工作流状态定义

    LangGraph 使用 TypedDict 来定义状态的 schema。
    所有节点都可以读写状态的这些字段。

    设计原则：
    - 状态应该包含所有节点可能需要的数据
    - 使用 Annotated 添加注解（如消息历史的管理器）
    - 避免在状态中存储大对象（如 Agent 实例）
    """
    # 对话消息历史
    messages: Annotated[List[BaseMessage], "add_message"]

    # 当前活跃的 Agent 名称
    active_agent: str

    # 任务描述
    task: str

    # 执行结果或中间产物
    result: Optional[str]

    # 标志位：是否需要人类确认
    requires_human_confirmation: bool

    # 错误信息（如果有）
    error: Optional[str]

    # 路由决策（用于条件边）
    next_action: Optional[str]


# =============================================================================
# 第二部分：创建 AutoGen Agents
# =============================================================================

def create_research_agent() -> ConversableAgent:
    """
    创建研究助手 Agent

    负责信息检索、文献查找、数据收集等任务
    """
    llm_config = {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key"),
        "temperature": 0.7,
    }

    research_agent = ConversableAgent(
        name="research_assistant",
        system_message="""你是一个专业的研究助手。
        你的职责是：
        1. 理解用户的研究需求
        2. 提供相关领域的信息和背景知识
        3. 帮助用户收集和整理资料

        当无法确定某些信息时，请明确说明。
        保持回答简洁、有条理。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    return research_agent


def create_writer_agent() -> ConversableAgent:
    """
    创建写作助手 Agent

    负责文档撰写、内容创作、文案编辑等任务
    """
    llm_config = {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key"),
        "temperature": 0.7,
    }

    writer_agent = ConversableAgent(
        name="writer_assistant",
        system_message="""你是一个专业的内容写作者。
        你的职责是：
        1. 根据提供的信息撰写高质量内容
        2. 确保文章结构清晰、逻辑连贯
        3. 注意语言的准确性和可读性

        擅长撰写报告、文章、文档等各类文本内容。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    return writer_agent


def create_critic_agent() -> ConversableAgent:
    """
    创建评论家 Agent

    负责审查、批评、提出改进建议等任务
    """
    llm_config = {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key"),
        "temperature": 0.7,
    }

    critic_agent = ConversableAgent(
        name="critic_assistant",
        system_message="""你是一个严格的评论家。
        你的职责是：
        1. 审查内容的准确性、完整性和逻辑性
        2. 指出存在的问题和不足
        3. 提供具体的改进建议

        你的反馈应该直接、建设性，帮助提升内容质量。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    return critic_agent


# =============================================================================
# 第三部分：LangGraph 节点函数
# =============================================================================

def create_graph_nodes(research_agent: ConversableAgent,
                       writer_agent: ConversableAgent,
                       critic_agent: ConversableAgent) -> Dict[str, Any]:
    """
    创建 LangGraph 节点函数字典

    LangGraph 的节点可以是：
    - 普通函数：接收状态，返回更新后的状态
    - Runnables：更灵活的执行单元

    Args:
        research_agent: 研究助手 Agent
        writer_agent: 写作助手 Agent
        critic_agent: 评论家 Agent

    Returns:
        包含所有节点函数的字典
    """

    def research_node(state: AgentState) -> Dict[str, Any]:
        """
        研究节点

        调用研究助手 Agent 处理信息收集任务。
        这个节点会更新 messages 和 result 字段。
        """
        task = state.get("task", "")
        messages = state.get("messages", [])

        # 构建发送给 Agent 的消息
        user_msg = f"请帮我研究以下主题：{task}"
        full_messages = messages + [HumanMessage(content=user_msg)]

        # 调用 AutoGen Agent 获取回复
        # 注意：这里使用简化的调用方式，实际项目中可能需要更复杂的配置
        reply = research_agent.generate_reply(messages=full_messages)

        return {
            "messages": messages + [AIMessage(content=reply or "研究完成")],
            "result": reply or "研究完成",
            "active_agent": "research_assistant",
            "next_action": "decide_next",
        }

    def writer_node(state: AgentState) -> Dict[str, Any]:
        """
        写作节点

        调用写作助手 Agent 进行内容创作。
        使用前一个节点的结果作为输入。
        """
        task = state.get("task", "")
        result = state.get("result", "")
        messages = state.get("messages", [])

        user_msg = f"基于以下研究结果，撰写内容：\n\n{result}\n\n任务：{task}"
        full_messages = messages + [HumanMessage(content=user_msg)]

        reply = writer_agent.generate_reply(messages=full_messages)

        return {
            "messages": messages + [AIMessage(content=reply or "写作完成")],
            "result": reply or "写作完成",
            "active_agent": "writer_assistant",
            "next_action": "decide_next",
        }

    def critic_node(state: AgentState) -> Dict[str, Any]:
        """
        评论节点

        调用评论家 Agent 审查内容质量。
        决定是否需要返工或可以结束流程。
        """
        result = state.get("result", "")
        messages = state.get("messages", [])

        user_msg = f"请审查以下内容并给出评价：\n\n{result}"
        full_messages = messages + [HumanMessage(content=user_msg)]

        reply = critic_agent.generate_reply(messages=full_messages)

        # 检查评论是否通过（简化逻辑：检查回复中是否包含"通过"等关键词）
        approval_keywords = ["通过", "合格", "满意", "可以", "完成"]
        approved = any(kw in str(reply) for kw in approval_keywords)

        return {
            "messages": messages + [AIMessage(content=reply or "审查完成")],
            "result": reply,
            "active_agent": "critic_assistant",
            "next_action": "approved" if approved else "revision_needed",
        }

    def human_review_node(state: AgentState) -> Dict[str, Any]:
        """
        人类确认节点

        这是一个暂停点，等待人类确认后才能继续。
        在实际应用中，这里会触发某种通知机制。

        返回的状态包含 requires_human_confirmation 标志，
        外部系统可以根据这个标志暂停执行并请求人类输入。
        """
        return {
            "requires_human_confirmation": True,
            "next_action": "human_approved",
        }

    def aggregate_node(state: AgentState) -> Dict[str, Any]:
        """
        汇总节点

        收集之前所有步骤的结果，生成最终输出。
        这是工作流的终点之一。
        """
        messages = state.get("messages", [])
        result = state.get("result", "")

        # 生成最终摘要
        summary = f"任务完成。最终结果：\n\n{result}"

        return {
            "messages": messages + [AIMessage(content=summary)],
            "result": summary,
            "requires_human_confirmation": False,
        }

    return {
        "research": research_node,
        "write": writer_node,
        "critic": critic_node,
        "human_review": human_review_node,
        "aggregate": aggregate_node,
    }


# =============================================================================
# 第四部分：条件边和路由逻辑
# =============================================================================

def create_routing_functions() -> Dict[str, Any]:
    """
    创建条件边路由函数

    LangGraph 的条件边根据状态决定下一步执行哪个节点。
    这模拟了工作流中的决策逻辑。

    路由函数返回的值必须匹配图中定义的边的目标节点名称。
    """

    def should_continue(state: AgentState) -> Literal["end", "continue"]:
        """
        判断是否继续主流程

        根据 next_action 字段决定是否结束当前工作流。
        """
        next_action = state.get("next_action", "")

        if next_action in ["approved", "human_approved"]:
            return "end"
        return "continue"

    def decide_next_step(state: AgentState) -> Literal["write", "critic", "human_review", "aggregate"]:
        """
        决定下一步骤

        根据当前状态和活跃的 Agent 决定下一步：
        - 研究后 -> 进入写作
        - 写作后 -> 进入评论
        - 评论后 -> 根据结果决定（通过/需返工/需人工确认）
        - 确认后 -> 汇总
        """
        active_agent = state.get("active_agent", "")

        if active_agent == "research_assistant":
            return "write"
        elif active_agent == "writer_assistant":
            return "critic"
        elif active_agent == "critic_assistant":
            next_action = state.get("next_action", "")
            if next_action == "revision_needed":
                # 需要返工，回到研究阶段重新开始
                return "write"
            elif next_action == "approved":
                # 通过，进入人工确认或直接汇总
                return "aggregate"
            else:
                return "human_review"
        elif active_agent == "human_review":
            return "aggregate"

        return "aggregate"

    return {
        "should_continue": should_continue,
        "decide_next": decide_next_step,
    }


# =============================================================================
# 第五部分：构建完整的 LangGraph 工作流
# =============================================================================

def build_agent_workflow() -> StateGraph:
    """
    构建完整的 Agent 工作流图

    工作流结构：
    start -> research -> write -> critic -> [decision] -> aggregate -> end
                              ^           |
                              |           v
                              +--- write <+
                              |
                              +--- human_review (optional)

    这个工作流展示了典型的三阶段审查模式：
    研究(Research) -> 写作(Write) -> 评论(Critic) -> 决策(Decision)
    """

    # 创建 Agents
    research_agent = create_research_agent()
    writer_agent = create_writer_agent()
    critic_agent = create_critic_agent()

    # 创建节点函数
    nodes = create_graph_nodes(research_agent, writer_agent, critic_agent)

    # 创建路由函数
    routing = create_routing_functions()

    # 构建图
    workflow = StateGraph(AgentState)

    # 添加节点
    # Literal 类型确保只能使用预定义的节点名称
    workflow.add_node("research", nodes["research"])
    workflow.add_node("write", nodes["write"])
    workflow.add_node("critic", nodes["critic"])
    workflow.add_node("human_review", nodes["human_review"])
    workflow.add_node("aggregate", nodes["aggregate"])

    # 设置入口点
    workflow.set_entry_point("research")

    # 添加普通边（顺序执行）
    workflow.add_edge("research", "write")
    workflow.add_edge("aggregate", END)

    # 添加条件边
    # 条件边的格式：add_conditional_edges(源节点, 路由函数, {路由返回值: 目标节点})
    workflow.add_conditional_edges(
        "critic",
        routing["decide_next"],
        {
            "write": "write",           # 需要返工
            "aggregate": "aggregate",    # 通过，直接汇总
            "human_review": "human_review",  # 需要人工确认
        }
    )

    workflow.add_conditional_edges(
        "human_review",
        routing["should_continue"],
        {
            "end": END,
            "continue": "aggregate",
        }
    )

    return workflow


def create_simple_workflow() -> StateGraph:
    """
    创建一个简化版本的工作流

    适用于简单的两阶段任务：研究 + 写作
    """

    research_agent = create_research_agent()
    writer_agent = create_writer_agent()

    nodes = create_graph_nodes(research_agent, writer_agent, critic_agent=None)

    workflow = StateGraph(AgentState)

    workflow.add_node("research", nodes["research"])
    workflow.add_node("write", nodes["write"])
    workflow.add_node("aggregate", nodes["aggregate"])

    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "aggregate")
    workflow.add_edge("aggregate", END)

    return workflow


# =============================================================================
# 第六部分：使用 AutoGen GroupChat 作为 LangGraph 节点
# =============================================================================

def create_groupchat_node(groupchat: GroupChat) -> Dict[str, Any]:
    """
    将 AutoGen GroupChat 包装为 LangGraph 节点

    GroupChat 是 AutoGen 的多 Agent 协作机制。
    将其嵌入 LangGraph 可以实现与其他节点的集成。

    注意：这个实现是简化版本，实际项目可能需要：
    1. 管理 GroupChat 的发言顺序
    2. 处理 GroupChat 的终止条件
    3. 收集和汇总 GroupChat 的输出

    Args:
        groupchat: AutoGen GroupChat 实例

    Returns:
        节点函数字典
    """

    def groupchat_node(state: AgentState) -> Dict[str, Any]:
        """
        GroupChat 节点

        执行多 Agent 协作对话，并将结果注入状态。
        """
        task = state.get("task", "")
        messages = state.get("messages", [])

        # 初始化 GroupChat（如果需要）
        groupchat.messages = messages

        # 创建管理器
        manager = GroupChatManager(groupchat=groupchat)

        # 执行群聊
        # 注意：这里使用了简化的调用方式
        # 实际应用中可能需要处理更多的边界情况
        try:
            # 将任务作为第一条消息
            chat_result = manager.initiate_chat(
                manager,
                message=task,
                clear_history=True,
            )
            result = str(chat_result)
        except Exception as e:
            result = f"GroupChat 执行出错: {str(e)}"

        return {
            "messages": groupchat.messages,
            "result": result,
            "active_agent": "groupchat",
            "next_action": "decide_next",
        }

    def decide_groupchat_next(state: AgentState) -> Literal["end", "continue"]:
        """判断 GroupChat 完成后下一步"""
        next_action = state.get("next_action", "")
        if next_action == "approved":
            return "end"
        return "continue"

    return {
        "node": groupchat_node,
        "decide": decide_groupchat_next,
    }


def build_groupchat_workflow() -> StateGraph:
    """
    构建包含 GroupChat 的混合工作流

    结构：
    start -> single_agent -> groupchat -> aggregate -> end

    single_agent: 单个 Agent 做预处理
    groupchat: 多 Agent 协作讨论
    aggregate: 汇总结果
    """

    # 创建单独的 Agents
    research_agent = create_research_agent()
    writer_agent = create_writer_agent()
    critic_agent = create_critic_agent()

    # 创建 GroupChat
    # GroupChat 定义参与者和发言规则
    groupchat = GroupChat(
        agents=[research_agent, writer_agent, critic_agent],
        messages=[],
        max_round=5,  # 最多讨论 5 轮
        speaker_selection_method="round_robin",  # 轮流发言
        allow_repeat_speaker=False,  # 不允许重复发言
    )

    # 创建 GroupChat 节点
    gc_nodes = create_groupchat_node(groupchat)

    # 获取非 GroupChat 节点
    other_nodes = create_graph_nodes(
        research_agent, writer_agent, critic_agent
    )

    # 构建图
    workflow = StateGraph(AgentState)

    workflow.add_node("prepare", other_nodes["research"])
    workflow.add_node("groupchat", gc_nodes["node"])
    workflow.add_node("aggregate", other_nodes["aggregate"])

    workflow.set_entry_point("prepare")
    workflow.add_edge("prepare", "groupchat")

    # GroupChat 完成后进行汇总
    workflow.add_conditional_edges(
        "groupchat",
        gc_nodes["decide"],
        {
            "end": END,
            "continue": "aggregate",
        }
    )

    workflow.add_edge("aggregate", END)

    return workflow


# =============================================================================
# 第七部分：执行工作流
# =============================================================================

def run_workflow(workflow: StateGraph, task: str) -> Dict[str, Any]:
    """
    执行构建好的工作流

    Args:
        workflow: StateGraph 实例
        task: 任务描述

    Returns:
        最终状态
    """
    # 编译图（添加检查点以支持断点续传）
    compiled = workflow.compile(checkpointer=MemorySaver())

    # 初始化状态
    initial_state: AgentState = {
        "messages": [HumanMessage(content=task)],
        "active_agent": "",
        "task": task,
        "result": None,
        "requires_human_confirmation": False,
        "error": None,
        "next_action": None,
    }

    # 执行
    config = {"configurable": {"thread_id": "workflow-001"}}

    try:
        result = compiled.invoke(initial_state, config=config)
        return result
    except Exception as e:
        return {
            **initial_state,
            "error": str(e),
        }


def demo_workflow_execution():
    """
    演示工作流执行

    这是一个完整的示例，展示如何：
    1. 构建工作流图
    2. 准备初始状态
    3. 执行并获取结果
    """
    print("=" * 60)
    print("AutoGen 与 LangGraph 集成 - 工作流执行演示")
    print("=" * 60)

    # 构建工作流
    workflow = build_agent_workflow()

    # 准备任务
    task = "撰写一篇关于人工智能在医疗领域应用的研究报告"

    print(f"\n任务: {task}")
    print("-" * 60)

    # 执行（需要配置好 API Key）
    # result = run_workflow(workflow, task)
    # print(f"\n最终结果:\n{result.get('result', 'N/A')}")

    print("\n[注意] 请在配置好 OpenAI API Key 后取消注释执行代码")
    print("-" * 60)


# =============================================================================
# 第八部分：高级主题 - 错误处理和恢复
# =============================================================================

def create_robust_workflow() -> StateGraph:
    """
    创建带错误处理的工作流

    增强功能：
    1. 每个节点添加 try-except 错误捕获
    2. 错误状态转换
    3. 可配置的失败策略
    """

    def error_handler_node(state: AgentState) -> Dict[str, Any]:
        """
        错误处理节点

        当工作流发生错误时，将错误信息记录到状态中，
        并决定是重试还是终止。
        """
        error = state.get("error", "Unknown error")

        return {
            "messages": state["messages"] + [
                AIMessage(content=f"错误已记录: {error}")
            ],
            "next_action": "terminate",
        }

    def should_retry(state: AgentState) -> Literal["retry", "end"]:
        """
        决定是否重试

        简化的重试逻辑：
        - 如果错误计数未超过阈值，则重试
        - 否则终止并返回错误
        """
        retry_count = state.get("retry_count", 0)
        max_retries = 3

        if retry_count < max_retries:
            return "retry"
        return "end"

    # 基本工作流
    workflow = build_agent_workflow()

    # 添加错误处理节点
    workflow.add_node("error_handler", error_handler_node)

    # 添加错误条件边
    # 当 error 不为空时，进入错误处理流程
    workflow.add_conditional_edges(
        "research",
        lambda s: "error" if s.get("error") else "continue",
        {
            "error": "error_handler",
            "continue": "write",
        }
    )

    return workflow


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    # 设置环境变量
    os.environ.setdefault("OPENAI_API_KEY", "your-api-key")

    print("AutoGen 与 LangGraph 集成示例")
    print("=" * 60)
    print("本文件展示如何将 AutoGen 嵌入 LangGraph 工作流")
    print("包括：状态机构建、条件路由、GroupChat 集成等")
    print("=" * 60)

    # 演示工作流结构
    demo_workflow_execution()

    # 演示 GroupChat 工作流
    print("\n\nGroupChat 工作流示例")
    print("-" * 60)
    print("使用 GroupChat 进行多 Agent 协作讨论")
    print("请查看 build_groupchat_workflow() 函数了解详情")

    print("\n请查看代码中的注释和文档字符串了解详细用法。")
