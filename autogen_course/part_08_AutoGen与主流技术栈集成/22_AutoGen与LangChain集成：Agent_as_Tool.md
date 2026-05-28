---
lesson_id: lesson_22
title: AutoGen与LangChain集成：Agent_as_Tool
module: AutoGen与主流技术栈集成
---

# 第22节 AutoGen与LangChain集成：Agent_as_Tool

## 学习目标

- 掌握AutoGen与LangChain的集成模式
- 理解AutoGen作为LangGraph中某个节点的实现方式
- 能够设计混合架构
- 理解AutoGen与LangChain生态的互补关系

## 内容概述

LangChain是当前最流行的LLM应用开发框架之一，提供了丰富的工具链和编排能力。AutoGen则在多Agent协作和对话智能方面独具优势。本节将深入解析如何将AutoGen Agent与LangChain生态集成，实现混合Agent架构，兼顾LangChain的工具生态和AutoGen的协作能力。

---

## 1. LangChain与AutoGen的定位对比

### 1.1 框架定位差异

| 特性 | LangChain | AutoGen |
|------|-----------|---------|
| **核心定位** | LLM应用开发框架 | 多Agent协作框架 |
| **设计理念** | 链式调用、工具组合 | Agent对话、协作编排 |
| **Agent抽象** | Tool/Runnable | ConversableAgent/GroupChat |
| **对话机制** | 有限状态机 | 自由对话、多轮交互 |
| **多Agent支持** | 基础 | 原生支持GroupChat |
| **适用场景** | 工具链、RAG、简单Agent | 复杂协作、代码执行、多角色对话 |

### 1.2 互补关系分析

```
┌─────────────────────────────────────────────────────────────┐
│              AutoGen 与 LangChain 互补关系                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   LangChain 优势              AutoGen 优势                  │
│   ┌────────────────┐        ┌────────────────┐             │
│   │ 丰富的工具生态   │   +   │ 多Agent协作     │             │
│   │ (搜索、数据库)   │        │ (GroupChat)    │             │
│   ├────────────────┤        ├────────────────┤             │
│   │ 标准化的Tool接口 │        │ 对话能力        │             │
│   │ (BaseTool)      │        │ (generate_reply)│             │
│   ├────────────────┤        ├────────────────┤             │
│   │ 工作流编排      │        │ 角色定义        │             │
│   │ (LangGraph)    │        │ (system_message)│            │
│   └────────────────┘        └────────────────┘             │
│                           │                                 │
│                           ▼                                 │
│   ┌─────────────────────────────────────────┐             │
│   │            混合架构                      │             │
│   │  LangChain工具生态 + AutoGen协作能力    │             │
│   └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. AutoGen Agent包装为LangChain Tool

### 2.1 核心接口映射

LangChain的Tool接口要求实现`_run`和`_arun`方法，而AutoGen的ConversableAgent通过`generate_reply`方法生成回复。两者可以通过适配器模式进行转换。

```python
# 接口映射关系
# LangChain Tool                    AutoGen Agent
# ──────────────────────────────────────────────────
# _run(message)                  →  generate_reply(messages)
# name/description                →  agent.name / system_message
# args_schema                     →  Agent的配置参数
# Tool管理状态                     →  Agent维护对话历史
```

### 2.2 AutoGenAgentTool实现

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, List, Optional

class AutoGenToolInput(BaseModel):
    """定义AutoGen Tool的输入模式"""
    message: str = Field(description="需要发送给AutoGen Agent的消息")
    session_id: Optional[str] = Field(description="会话ID，用于追踪多轮对话", default=None)

class AutoGenAgentTool(BaseTool):
    """将AutoGen ConversableAgent包装为LangChain BaseTool"""
    name: str = "autogen_agent"
    description: str = "通用的AutoGen Agent包装工具，通过对话方式处理任务"
    args_schema: Type[BaseModel] = AutoGenToolInput

    def __init__(self, agent: ConversableAgent, session_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.session_id = session_id or "default_session"
        self._conversations: Dict[str, List[Dict]] = {}

    def _run(self, message: str, session_id: Optional[str] = None, **kwargs) -> str:
        """同步执行工具调用"""
        sid = session_id or self.session_id
        if sid not in self._conversations:
            self._conversations[sid] = []

        reply = self.agent.generate_reply(
            messages=self._conversations[sid],
            **kwargs
        )

        if reply is not None:
            self._conversations[sid].append({"role": "user", "content": message})
            self._conversations[sid].append({"role": "assistant", "content": reply})

        return reply or "Agent未返回有效回复"
```

### 2.3 创建多个工具的工厂函数

```python
def create_autogen_tools() -> List[BaseTool]:
    """
    创建包含多个AutoGen Agent的LangChain工具集合

    返回:
        List[BaseTool]: 可在LangChain中使用的工具列表
    """
    # 创建底层的AutoGen Agents
    math_agent = create_math_agent()
    code_agent = create_code_agent()

    # 包装为LangChain Tools
    math_tool = AutoGenAgentTool(
        agent=math_agent,
        name="math_assistant",
        description="数学助手工具，可以解决数学问题、计算表达式、提供解题步骤",
    )

    code_tool = AutoGenAgentTool(
        agent=code_agent,
        name="code_assistant",
        description="代码助手工具，可以编写Python代码、调试程序、数据分析",
    )

    return [math_tool, code_tool]
```

---

## 3. LangChain ConversationalAgent对比AutoGen对话模式

### 3.1 LangChain的对话模式

LangChain的`ConversationalREACTDescription` Agent通过ReAct框架循环执行：

```
用户输入 → Thought → Action → Observation → Thought → ...
```

特点：
- 基于ReAct推理模式
- Action调用Tool获取外部信息
- Observation反馈到下一轮推理
- 适合工具调用场景

### 3.2 AutoGen的对话模式

AutoGen的对话更加自由灵活：

```
UserProxy → Assistant → UserProxy → Assistant → ...
```

特点：
- 支持多轮自由对话
- 可以引入多个专业Agent
- GroupChat支持多Agent同时参与
- 支持代码执行和函数调用

### 3.3 模式对比

| 对比维度 | LangChain ConversationalAgent | AutoGen GroupChat |
|---------|-------------------------------|-------------------|
| **对话结构** | 单Agent + Tool循环 | 多Agent自由对话 |
| **状态管理** | 外部状态 | Agent内部维护 |
| **发言控制** | 固定顺序(REACT) | 可配置(轮流/自动) |
| **适用场景** | 工具调用为主 | 协作讨论为主 |
| **代码执行** | 通过Tool | 原生CodeExecutor |

---

## 4. LangGraph状态机中嵌入AutoGen GroupChat

### 4.1 LangGraph核心概念

LangGraph是基于有向图的工作流编排框架，核心概念：

- **StateGraph**：状态图，定义工作流的数据结构
- **节点(Node)**：执行单元，可以是函数或Runnable
- **边(Edge)**：节点之间的连接
- **条件边(Conditional Edge)**：根据状态决定下一个节点

```
工作流示例：
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  START  │───►│ Node A  │───►│ Node B  │───►│   END   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                      │              │
                      ▼              ▼
               ┌─────────┐    ┌─────────┐
               │Node C   │    │条件边   │
               └─────────┘    └─────────┘
```

### 4.2 状态定义

```python
from typing import Annotated, List, TypedDict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """LangGraph工作流状态定义"""
    # 对话消息历史
    messages: Annotated[List[BaseMessage], "add_message"]
    # 当前活跃的Agent名称
    active_agent: str
    # 任务描述
    task: str
    # 执行结果
    result: Optional[str]
    # 是否需要人类确认
    requires_human_confirmation: bool
    # 路由决策
    next_action: Optional[str]
```

### 4.3 创建LangGraph节点

```python
def create_graph_nodes(research_agent: ConversableAgent,
                       writer_agent: ConversableAgent,
                       critic_agent: ConversableAgent) -> Dict[str, Any]:
    """创建LangGraph节点函数字典"""

    def research_node(state: AgentState) -> Dict[str, Any]:
        """研究节点：调用研究助手Agent处理信息收集"""
        task = state.get("task", "")
        messages = state.get("messages", [])
        user_msg = f"请帮我研究以下主题：{task}"
        full_messages = messages + [HumanMessage(content=user_msg)]
        reply = research_agent.generate_reply(messages=full_messages)

        return {
            "messages": messages + [AIMessage(content=reply or "研究完成")],
            "result": reply or "研究完成",
            "active_agent": "research_assistant",
            "next_action": "decide_next",
        }

    def writer_node(state: AgentState) -> Dict[str, Any]:
        """写作节点：调用写作助手Agent进行内容创作"""
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
        """评论节点：调用评论家Agent审查内容质量"""
        result = state.get("result", "")
        messages = state.get("messages", [])
        user_msg = f"请审查以下内容并给出评价：\n\n{result}"
        full_messages = messages + [HumanMessage(content=user_msg)]
        reply = critic_agent.generate_reply(messages=full_messages)

        # 检查评论是否通过
        approval_keywords = ["通过", "合格", "满意", "可以", "完成"]
        approved = any(kw in str(reply) for kw in approval_keywords)

        return {
            "messages": messages + [AIMessage(content=reply or "审查完成")],
            "result": reply,
            "active_agent": "critic_assistant",
            "next_action": "approved" if approved else "revision_needed",
        }

    return {
        "research": research_node,
        "write": writer_node,
        "critic": critic_node,
    }
```

### 4.4 条件边路由设计

```python
def create_routing_functions() -> Dict[str, Any]:
    """创建条件边路由函数"""

    def decide_next_step(state: AgentState) -> Literal["write", "critic", "aggregate"]:
        """根据当前状态决定下一步"""
        active_agent = state.get("active_agent", "")

        if active_agent == "research_assistant":
            return "write"
        elif active_agent == "writer_assistant":
            return "critic"
        elif active_agent == "critic_assistant":
            next_action = state.get("next_action", "")
            if next_action == "revision_needed":
                return "write"  # 需要返工
            else:
                return "aggregate"  # 通过

        return "aggregate"

    return {"decide_next": decide_next_step}
```

### 4.5 构建完整工作流

```python
def build_agent_workflow() -> StateGraph:
    """构建完整的Agent工作流图"""

    # 创建Agents
    research_agent = create_research_agent()
    writer_agent = create_writer_agent()
    critic_agent = create_critic_agent()

    # 创建节点和路由
    nodes = create_graph_nodes(research_agent, writer_agent, critic_agent)
    routing = create_routing_functions()

    # 构建图
    workflow = StateGraph(AgentState)

    workflow.add_node("research", nodes["research"])
    workflow.add_node("write", nodes["write"])
    workflow.add_node("critic", nodes["critic"])
    workflow.add_node("aggregate", nodes["aggregate"])

    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("aggregate", END)

    # 添加条件边
    workflow.add_conditional_edges(
        "critic",
        routing["decide_next"],
        {
            "write": "write",
            "aggregate": "aggregate",
        }
    )

    return workflow
```

### 4.6 将GroupChat作为LangGraph节点

AutoGen的GroupChat本身就是一个多Agent协作单元，可以整体嵌入LangGraph：

```python
def create_groupchat_node(groupchat: GroupChat) -> Dict[str, Any]:
    """将AutoGen GroupChat包装为LangGraph节点"""

    def groupchat_node(state: AgentState) -> Dict[str, Any]:
        """GroupChat节点：执行多Agent协作对话"""
        task = state.get("task", "")
        messages = state.get("messages", [])

        groupchat.messages = messages
        manager = GroupChatManager(groupchat=groupchat)

        try:
            chat_result = manager.initiate_chat(
                manager,
                message=task,
                clear_history=True,
            )
            result = str(chat_result)
        except Exception as e:
            result = f"GroupChat执行出错: {str(e)}"

        return {
            "messages": groupchat.messages,
            "result": result,
            "active_agent": "groupchat",
            "next_action": "decide_next",
        }

    return {"node": groupchat_node}
```

---

## 5. AutoGen与LangChain生态的互补关系

### 5.1 集成模式总览

```
┌─────────────────────────────────────────────────────────────┐
│                AutoGen + LangChain 集成模式                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式1：AutoGen Agent作为LangChain Tool                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LangChain Agent ──► AutoGenAgentTool ──► Agent     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  模式2：LangGraph节点嵌入AutoGen                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LangGraph ──► AutoGen GroupChat ──► LangGraph      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  模式3：混合架构                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LangGraph(编排层)                                   │   │
│  │      │                                                │   │
│  │      ├──► AutoGen GroupChat(协作层)                  │   │
│  │      │                                                │   │
│  │      └──► LangChain Tools(工具层)                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 架构设计决策树

```
任务类型分析
    │
    ├─► 工具调用为主 ──► LangChain为主 + AutoGen作为Tool
    │
    ├─► 多Agent协作 ──► AutoGen GroupChat + LangGraph编排
    │
    └─► 复杂工作流 ──► 混合架构
            │
            ├─► 使用LangGraph定义主流程
            ├─► 使用AutoGen实现协作节点
            └─► 使用LangChain Tools补充工具生态
```

### 5.3 互补场景分析

| 场景 | LangChain职责 | AutoGen职责 |
|------|--------------|-------------|
| **RAG问答** | 检索器、向量数据库 | Agent生成回答 |
| **代码助手** | 代码执行环境 | 多Agent评审 |
| **数据分析** | 数据加载、可视化 | 分析Agent协作 |
| **文档处理** | PDF解析、文本分割 | 生成摘要、审查 |
| **客服系统** | 意图识别、槽位填充 | 对话生成、多角色 |

---

## 代码案例

本节包含两个代码案例，请参考 `22_codes/` 目录：

### 案例1：Agent作为Tool (agent_as_tool.py)

**内容要点：**
- AutoGen ConversableAgent包装为LangChain BaseTool的接口映射
- 使用`@tool`装饰器快速创建结构化Tool
- 带记忆管理的MemoryAutoGenTool实现
- 多Agent工具集合的创建与使用
- 与LangChain Agent的集成演示

**核心类：**
- `AutoGenAgentTool`：基础包装器
- `MemoryAutoGenTool`：带上下文压缩的记忆工具
- `create_autogen_tools()`：工具工厂函数

**运行方式：**
```bash
python 22_codes/agent_as_tool.py
```

### 案例2：LangGraph集成 (langgraph_integration.py)

**内容要点：**
- LangGraph StateGraph状态定义
- AutoGen Agent作为LangGraph节点函数的实现
- 条件边路由逻辑设计
- AutoGen GroupChat作为LangGraph节点的包装
- 完整工作流构建与执行
- 错误处理和恢复机制

**核心函数：**
- `create_graph_nodes()`：创建LangGraph节点
- `create_routing_functions()`：条件边路由
- `build_agent_workflow()`：构建工作流图
- `build_groupchat_workflow()`：GroupChat混合工作流
- `run_workflow()`：执行工作流

**运行方式：**
```bash
python 22_codes/langgraph_integration.py
```

---

## 本节小结

1. **接口映射**：通过适配器模式将AutoGen Agent包装为LangChain Tool，实现`generate_reply`到`_run`的接口转换

2. **对话模式对比**：LangChain适合工具调用场景的ReAct循环，AutoGen适合多Agent自由协作

3. **LangGraph集成**：AutoGen Agent可以作为LangGraph的节点函数，GroupChat可以作为条件节点嵌入工作流

4. **混合架构设计**：根据任务类型选择集成模式，工具调用用模式1，协作用模式2，复杂工作流用模式3

5. **生态互补**：LangChain提供丰富的工具生态，AutoGen提供强大的协作能力，两者结合实现更强的Agent系统

---

## 延伸阅读

- [AutoGen与RAG知识库集成](./23_AutoGen与RAG知识库集成.md)
- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain Tool接口文档](https://python.langchain.com/docs/concepts/tool_interface/)
- [AutoGen GroupChat文档](https://microsoft.github.io/autogen/reference/autogen_agentchat/groupchat.html)

---

## 下节预告

下一节我们将学习 **AutoGen与RAG知识库集成**，探讨如何将外部知识检索与AutoGen Agent结合，实现检索增强的对话系统。