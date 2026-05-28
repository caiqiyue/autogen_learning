---
lesson_id: lesson_09
title: GroupChat与多Agent协作模式
module: 多Agent协作与GroupChat高级机制
---

# 第9节 GroupChat与多Agent协作模式

## 学习目标

1. 理解GroupChat的整体架构
2. 掌握群聊中Agent的角色分配机制
3. 理解消息广播与私聊的区别

---

## 9.1 GroupChat整体架构

### 9.1.1 核心组件

GroupChat是AutoGen中实现多Agent协作的核心组件，由三个关键组件构成：

```
┌─────────────────────────────────────────────────────────────┐
│                      GroupChat架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│   │   Agent     │      │  GroupChat  │      │  GroupChat  │ │
│   │  (发言人)   │◄────►│  (容器)     │◄────►│  Manager    │ │
│   │             │      │             │      │  (协调器)  │ │
│   └─────────────┘      └─────────────┘      └─────────────┘ │
│         │                    │                    │         │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│   - 发送消息           - 存储消息           - 选择下一个    │
│   - 接收消息           - 管理Agent列表      - 转发消息      │
│   - 生成回复           - 维护对话历史       - 控制终止      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| 组件 | 说明 | 职责 |
|-----|------|------|
| **GroupChatManager** | 群聊管理器 | 协调整个群聊，负责选择发言者、转发消息、控制终止 |
| **GroupChat** | 群聊容器 | 存储消息历史和Agent列表，维护群聊状态 |
| **Agent** | 参与智能体 | 发送消息、接收消息、生成回复 |

### 9.1.2 消息传递流程

```
┌─────────────────────────────────────────────────────────────┐
│                    消息传递流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. [发起者] 发送初始消息到 GroupChatManager                 │
│           │                                                 │
│           ▼                                                 │
│  2. [Manager] 调用 select_speaker() 选择下一个发言者        │
│           │                                                 │
│           ▼                                                 │
│  3. [Manager] 将消息转发给选中的Agent                       │
│           │                                                 │
│           ▼                                                 │
│  4. [被选中Agent] 接收消息并生成回复                        │
│           │                                                 │
│           ▼                                                 │
│  5. [Manager] 将回复添加到消息历史                          │
│           │                                                 │
│           ▼                                                 │
│  6. 检查终止条件                                            │
│           │                                                 │
│     ├── 满足 ──► 结束，返回聊天结果                          │
│     │                                                        │
│     └── 不满足 ──► 继续步骤2                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9.2 GroupChatManager源码解析

### 9.2.1 selected_agent选择逻辑

GroupChatManager通过`select_speaker()`方法选择下一个发言者，支持多种选择策略：

```python
# speaker_selection_method 可选值
speaker_selection_method = "auto"      # 由LLM自动决定（默认）
speaker_selection_method = "manual"   # 由外部指定下一个speaker
speaker_selection_method = "round_robin"  # 轮询制，按顺序轮流
speaker_selection_method = "random"   # 随机选择
```

**auto模式（默认）**：

- LLM根据上下文和对话历史选择最合适的下一个发言者
- 灵活性高，但可能出现选择不均衡

**round_robin模式**：

- 每个Agent按顺序轮流发言
- 适用于固定流程、需要均衡参与的场景

**random模式**：

- 随机选择下一个发言者
- 适用于公平调研场景

### 9.2.2 消息广播机制

GroupChatManager默认使用广播模式：

```python
# 广播模式：消息发送给所有Agent
groupchat = GroupChat(
    agents=[agent_a, agent_b, agent_c],
    messages=[],
    max_round=10,
)
```

**广播特点**：
1. GroupChatManager将消息广播给所有Agent
2. 每个Agent都能看到完整的对话历史
3. 适合团队协作讨论场景

---

## 9.3 GroupChat的初始化与配置

### 9.3.1 基本配置参数

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# 创建Agent
agent1 = ConversableAgent(
    name="Agent1",
    system_message="你是Agent1。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

agent2 = ConversableAgent(
    name="Agent2",
    system_message="你是Agent2。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建GroupChat
groupchat = GroupChat(
    agents=[agent1, agent2],           # 参与群聊的Agent列表
    messages=[],                          # 初始消息列表
    max_round=10,                        # 最大对话轮次
    speaker_selection_method="auto",     # 选择策略
)
```

### 9.3.2 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `agents` | list | 必需 | 参与群聊的Agent列表 |
| `messages` | list | `[]` | 初始消息列表 |
| `max_round` | int | `100` | 最大对话轮次，防止无限循环 |
| `speaker_selection_method` | str | `"auto"` | 发言者选择策略 |
| `termination_msg` | func | `None` | 终止消息检测函数 |

---

## 9.4 多Agent对话的消息传递机制

### 9.4.1 广播与私聊的区别

```
┌─────────────────────────────────────────────────────────────┐
│              广播模式 vs 私聊模式                            │
├──────────────────────────────┬──────────────────────────────┤
│         广播模式             │          私聊模式            │
├──────────────────────────────┼──────────────────────────────┤
│  Manager ──► [所有Agent]     │  AgentA ──► [AgentB]        │
│                              │                              │
│  特点:                       │  特点:                       │
│  - 所有Agent都能看到消息     │  - 消息只发送给指定Agent     │
│  - 消息通过Manager转发       │  - 其他Agent不会收到消息     │
│  - 适合团队协作讨论          │  - 适合一对一协作            │
└──────────────────────────────┴──────────────────────────────┘
```

**广播模式示例**：

```python
# 通过GroupChatManager广播
result = agent1.initiate_chat(
    manager,
    message="大家好，让我们讨论一下项目架构。",
)
```

**私聊模式示例**：

```python
# 直接私聊，不经过GroupChatManager
result = agent1.initiate_chat(
    agent2,  # 直接指定接收者
    message="Bob，我有个想法想私下和你讨论...",
)
```

---

## 9.5 nested group chat模式：子群聊的创建与层级管理

### 9.5.1 嵌套GroupChat核心概念

嵌套GroupChat是一种多层级协作模式，允许在父群聊中创建子群聊处理特定子任务：

```
┌─────────────────────────────────────────────────────────────┐
│                  嵌套GroupChat层级结构                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   父群聊（项目协调）                                         │
│     │                                                        │
│     ├── 子群聊A（前端组）                                    │
│     │     ├── 前端组长                                      │
│     │     └── 前端开发                                       │
│     │                                                        │
│     └── 子群聊B（后端组）                                    │
│           ├── 后端组长                                      │
│           └── 后端开发                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**典型应用场景**：
- 大型复杂任务分解：父群聊协调，子群聊分别处理子任务
- 专家团队协作：不同子群聊包含不同领域的专家
- 并行处理：多个子群聊同时处理独立任务

### 9.5.2 创建嵌套GroupChat

**步骤1：创建各层级的Agent**

```python
# 父群聊Agent：项目协调者
coordinator = ConversableAgent(
    name="项目协调者",
    system_message="你是项目协调者，负责协调前端和后端团队完成项目。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 子群聊A（前端组）Agent
frontend_lead = ConversableAgent(
    name="前端组长",
    system_message="你是前端组长，负责协调前端开发工作。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

frontend_dev = ConversableAgent(
    name="前端开发",
    system_message="你是前端开发工程师，负责实现UI组件。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
```

**步骤2：创建子群聊**

```python
# 前端子群聊
frontend_groupchat = GroupChat(
    agents=[frontend_lead, frontend_dev],
    messages=[],
    max_round=5,
)

frontend_manager = GroupChatManager(
    groupchat=frontend_groupchat,
    llm_config=llm_config,
)
```

**步骤3：创建父群聊（包含子群聊管理器）**

```python
# 父群聊包含协调者和两个子群聊管理器
parent_groupchat = GroupChat(
    agents=[coordinator, frontend_manager, backend_manager],
    messages=[],
    max_round=10,
)

parent_manager = GroupChatManager(
    groupchat=parent_groupchat,
    llm_config=llm_config,
)
```

### 9.5.3 跨层级通信机制

嵌套GroupChat中，消息传递规则：

1. 父群聊的消息会传递给选定的Agent（可能是子群聊管理器）
2. 子群聊管理器收到消息后，会在子群聊中继续传递
3. 子群聊的回复会返回给父群聊

**重要**：Agent不能直接跨群聊通信，必须通过管理器转发

```
消息流向: 高层 -> 中层 -> 基层 -> 逐层返回
```

### 9.5.4 嵌套终止条件管理

嵌套GroupChat的终止需要特别关注：

| 层级 | 终止条件 | 配置建议 |
|-----|---------|---------|
| 子群聊 | max_round较短 | `max_round=3-5`，子任务快速完成 |
| 父群聊 | max_round较长 | `max_round=8-10`，给足协调时间 |

```python
# 子群聊配置（较短轮次，快速终止）
subtask_groupchat = GroupChat(
    agents=[subtask_leader, subtask_worker],
    messages=[],
    max_round=3,  # 子群聊轮次较少
)

# 父群聊配置（较长轮次）
parent_groupchat = GroupChat(
    agents=[main_coordinator, subtask_manager],
    messages=[],
    max_round=8,  # 父群聊轮次较多
)
```

---

## 9.6 代码案例

### 9.6.1 groupchat_basic.py

文件路径：`part_04_多Agent协作与GroupChat高级机制/09_codes/groupchat_basic.py`

本文件演示：
- GroupChatManager的创建与配置
- 多个Agent加入群聊
- 消息广播机制
- Agent角色分配与selected_agent选择逻辑
- speaker_selection_method策略对比（auto/round_robin/random）
- 广播模式vs私聊模式
- GroupChat消息传递流程详解
- GroupChatManager内部机制
- GroupChat终止条件

### 9.6.2 groupchat_nested.py

文件路径：`part_04_多Agent协作与GroupChat高级机制/09_codes/groupchat_nested.py`

本文件演示：
- 嵌套GroupChat基本模式
- 顺序执行的子群聊模式
- 并行子群聊概念
- 嵌套GroupChat终止条件管理
- 跨层级通信机制
- 层级任务分解

### 9.6.3 运行说明

**前置条件**：
1. 确保已安装AutoGen：`pip install pyautogen`
2. 确保已在`.env`文件中配置好API密钥

**运行方式**：
```bash
cd part_04_多Agent协作与GroupChat高级机制/09_codes

# 演示GroupChat基本用法
python groupchat_basic.py

# 演示嵌套GroupChat
python groupchat_nested.py
```

---

## 9.7 常见问题与解决方案

### Q1: 如何控制群聊中Agent的发言顺序？

**解决方案**：使用`speaker_selection_method="round_robin"`强制轮询

```python
groupchat = GroupChat(
    agents=[agent_a, agent_b, agent_c],
    max_round=10,
    speaker_selection_method="round_robin",  # 强制轮询
)
```

### Q2: 为什么某些Agent几乎没有发言机会？

**可能原因**：auto模式下LLM可能倾向于选择同一个Agent

**解决方案**：
1. 使用`round_robin`强制均衡
2. 自定义均衡选择函数
3. 在系统提示中强调需要均衡参与

### Q3: 如何实现一对一私聊？

**解决方案**：直接使用`initiate_chat`指定接收者，不经过GroupChatManager

```python
result = agent_a.initiate_chat(
    agent_b,  # 直接指定接收者
    message="这个消息只有你和Bob能看到",
)
```

### Q4: 子群聊完成后如何将结果返回给父群聊？

**解决方案**：子群聊的回复会自动返回给父群聊管理器，由管理器决定下一步操作

---

## 9.8 本章小结

通过本章学习，你已经：

1. **理解GroupChat整体架构**：GroupChatManager、GroupChat、Agent三组件协作
2. **掌握selected_agent选择逻辑**：auto、round_robin、random三种策略
3. **理解消息广播机制**：广播模式vs私聊模式的区别
4. **掌握嵌套GroupChat**：子群聊创建、层级管理、跨层级通信

下一章我们将学习GroupChat终止条件与常见问题处理，掌握循环终止条件的设置方法。

---

## 扩展阅读

- [AutoGen GroupChat源码](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/group_chat.py)
- [GroupChatManager源码](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/groupchat_manager.py)
- [AutoGen官方示例](https://microsoft.github.io/autogen/docs/Examples/GroupChat)