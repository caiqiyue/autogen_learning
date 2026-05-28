---
lesson_id: lesson_15
title: ConversableAgent对话模式综合实战
module: AssistantAgent与UserProxyAgent应用
---

# 第15节 ConversableAgent对话模式综合实战

## 学习目标

- 综合运用ConversableAgent的核心机制
- 掌握不同对话模式的配置方法
- 能够构建完整的多Agent协作系统

## 内容概述

本节将综合运用ConversableAgent的核心机制，深入讲解AutoGen支持的五种对话模式，并通过实际代码案例演示如何构建完整的多Agent协作系统。重点内容包括双人对话模式（Two-Agent Chat）的配置、群聊模式（GroupChat）的综合实战、以及不同协作模式的选择决策。

---

## 1. 五种对话模式综合对比

### 1.1 对话模式总览

AutoGen支持多种对话模式，每种模式有不同的适用场景和配置特点：

| 对话模式 | Agent数量 | 通信方式 | 配置复杂度 | 适用场景 |
|----------|-----------|----------|------------|----------|
| 双人对话（Two-Agent Chat） | 2个 | 直接通信 | 低 | 一对一协作、人机交互 |
| 群聊模式（GroupChat） | 2+个 | 广播通信 | 中 | 多团队协作、团队讨论 |
| 嵌套对话模式（Nested Chat） | 2+个 | 层次化通信 | 高 | 复杂任务分解、子团队协作 |
| 异步对话模式（Async Chat） | 2+个 | 并发通信 | 高 | 高性能场景、并行任务 |
| 流式对话模式（Streaming Chat） | 2+个 | 流式输出 | 中 | 长文本生成、实时反馈 |

### 1.2 各模式核心特点

**双人对话模式（Two-Agent Chat）**
- 两个Agent直接通信，无需GroupChatManager
- 使用`initiate_chat()`发起对话
- 消息直接传递给对方，简单直接
- 适合一对一协作场景

**群聊模式（GroupChat）**
- 多个Agent通过GroupChatManager协作
- 消息广播给所有Agent（由Manager转发）
- LLM自动选择下一个发言者
- 适合多Agent团队协作

**嵌套对话模式（Nested Chat）**
- Agent之间可以嵌套调用
- 支持并发对话
- 适合复杂的工作流和层次化任务分解

**异步对话模式（Async Chat）**
- 使用async/await进行异步通信
- 支持并发执行
- 适合高性能场景

**流式对话模式（Streaming Chat）**
- 支持流式输出
- 实时显示生成内容
- 适合长文本生成场景

### 1.3 模式选择决策树

```
开始
  │
  ├─ Agent数量 = 2？
  │    ├─ 是 ──→ 使用双人对话模式
  │    └─ 否 ──→ 继续判断
  │
  ├─ 需要发言均衡控制？
  │    ├─ 是 ──→ GroupChat + allow_repeat='never'
  │    └─ 否 ──→ 继续判断
  │
  ├─ 需要LLM自动选择发言者？
  │    ├─ 是 ──→ GroupChat + speaker_selection_method='auto'
  │    └─ 否 ──→ 继续判断
  │
  └─ 有层次化的任务分解？
       ├─ 是 ──→ 使用嵌套GroupChat
       └─ 否 ──→ 使用普通GroupChat
```

---

## 2. Two-Agent Chat双人对话配置

### 2.1 双人对话核心概念

双人对话是AutoGen中最基础的协作模式，涉及两个Agent之间的直接对话。

**核心特点：**
1. **一对一通信**：消息直接发送给接收方，不经过中间人
2. **简单直接**：无需GroupChatManager管理，配置简单
3. **明确的角色分工**：发起者和接收者关系清晰
4. **灵活的终止控制**：通过is_termination_msg控制对话终止

**适用场景：**
- 人机交互：UserProxyAgent与AssistantAgent协作
- 任务协作：专家一对一讨论问题
- 工具调用：Agent调用工具获取信息后返回结果
- 审查流程：一方生成内容，另一方审查修改

### 2.2 基本用法

双人对话使用`initiate_chat()`方法发起对话：

```python
from autogen import ConversableAgent

# 创建助手Agent
assistant = ConversableAgent(
    name="助手",
    system_message="你是一个有帮助的助手，擅长回答问题和提供建议。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建用户代理Agent
user_proxy = ConversableAgent(
    name="用户代理",
    system_message="你代表用户，可以发起对话并接收回复。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 用户代理发起对话
result = user_proxy.initiate_chat(
    assistant,
    message="请介绍一下人工智能的发展历史。",
)
```

### 2.3 human_input_mode对对话行为的影响

`human_input_mode`有三种模式：

| 模式 | 人工输入频率 | 适用场景 |
|------|-------------|----------|
| ALWAYS | 每次都请求 | 教学演示、严格审核 |
| TERMINATE | 智能请求 | 人机协作、迭代优化 |
| NEVER | 从不请求 | 自动化流程、代码执行 |

**ALWAYS模式特点：**
- 每次Agent生成回复前都需要人工确认
- 用户可以修改回复内容
- 适合需要严格控制的场景
- 会阻塞等待用户输入

**TERMINATE模式特点：**
- 当is_termination_msg返回False时请求人工输入
- 当正常终止时不请求输入
- 适合人机协作场景
- 是UserProxyAgent的默认模式

**NEVER模式特点：**
- 从不请求人工输入
- Agent完全自动运行
- 适合自动化流程
- 适合代码执行和工具调用

### 2.4 register_reply自定义回复逻辑

`register_reply`是ConversableAgent的核心机制之一，允许注册自定义的回复生成函数：

```python
def custom_reply_function(messages, sender, config):
    """
    自定义回复函数

    Args:
        messages: 对话消息历史列表
        sender: 发送消息的Agent
        config: 额外配置

    Returns:
        str: 自定义回复内容，如果不触发则返回None
    """
    if not messages:
        return None

    last_message = messages[-1]
    content = last_message.get("content", "").lower()

    # 检测关键字，触发自定义回复
    if "hello" in content or "你好" in content:
        return "你好！我是自定义回复，很高兴为您服务！"
    elif "time" in content or "时间" in content:
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"当前时间是: {now}"
    elif "bye" in content or "再见" in content:
        return "再见！祝您有美好的一天！"

    # 如果没有匹配的关键字，返回None使用默认LLM回复
    return None

# 注册自定义回复
assistant.register_reply(
    reply_func=custom_reply_function,
    name="custom_reply",
)
```

### 2.5 双人对话的终止控制

双人对话的终止条件由以下因素控制：

1. **is_termination_msg**：Agent级别的终止消息检测
2. **max_consecutive_auto_reply**：最大连续自动回复数
3. **手动终止**：外部代码可以强制终止对话

```python
# 使用is_termination_msg终止
assistant = ConversableAgent(
    name="助手_终止",
    system_message="""你是一个任务助手。
    如果任务完成，说'TASK_DONE'来终止对话。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TASK_DONE" in msg.get("content", ""),
)

# 使用max_consecutive_auto_reply限制
assistant = ConversableAgent(
    name="助手_最大",
    system_message="你是一个话多的助手，总是长篇大论。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,  # 最多3次连续回复
)
```

### 2.6 双人对话 vs GroupChat

| 特性 | 双人对话 | GroupChat |
|------|----------|-----------|
| Agent数量 | 2个 | 多个 |
| 消息传递 | 直接发送 | 通过Manager广播 |
| 配置复杂度 | 低 | 中 |
| 发言控制 | 无（直接通信） | speaker_selection |
| 发言均衡 | 不适用 | allow_repeat控制 |
| 适用场景 | 一对一协作 | 团队协作讨论 |

---

## 3. GroupChat综合实战

### 3.1 GroupChat核心概念

GroupChat是AutoGen中实现多Agent团队协作的核心组件，与双人对话不同，GroupChat通过GroupChatManager协调多个Agent。

**核心组件：**
1. **GroupChat** - 群聊容器，存储消息和Agent列表
2. **GroupChatManager** - 群聊管理器，负责协调整个群聊
3. **Agent** - 参与群聊的智能体

**消息传递机制：**
- 广播模式：消息发送给所有Agent（通过GroupChatManager转发）
- selected_agent：决定下一个发言者

### 3.2 speaker_selection_method四种策略

`speaker_selection_method`决定如何选择下一个发言者：

| 模式 | 选择方式 | 均衡性 | 适用场景 |
|------|----------|--------|----------|
| auto | LLM智能选择 | 中 | 复杂协作 |
| round_robin | 按顺序轮询 | 高 | 均衡发言 |
| random | 随机选择 | 低 | 随机性场景 |
| manual | 外部手动指定 | 完全可控 | 教学/调试 |

**auto模式特点：**
- 由LLM根据对话上下文选择下一个发言者
- LLM会考虑Agent的角色和能力
- 适合复杂的多Agent协作场景
- 默认选项

**round_robin模式特点：**
- 按Agent列表顺序轮流选择发言者
- 确保每个Agent都有平等的发言机会
- 不考虑对话上下文
- 适合需要均衡发言的场景

**random模式特点：**
- 随机选择下一个发言者
- 不考虑Agent顺序或上下文
- 可能导致某些Agent主导或被忽视
- 适合需要随机性的场景

**manual模式特点：**
- 由外部代码或用户手动选择发言者
- 提供最大的控制和确定性
- 需要外部逻辑干预
- 适合教学、调试或严格流程控制

```python
# 创建GroupChat并配置speaker_selection_method
groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=6,
    speaker_selection_method="auto",  # 或 "round_robin", "random", "manual"
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)
```

### 3.3 allow_repeat参数控制

`allow_repeat`参数控制同一Agent是否能连续发言：

| 值 | 说明 |
|----|------|
| "never" | 不允许连续发言，确保均衡 |
| "certain_num_turns" | 允许连续发言一定次数 |
| "always" | 允许连续发言（默认） |

**"never"模式特点：**
- 同一Agent不能连续发言
- 确保每个Agent都有发言机会
- 避免某个Agent主导对话
- 适合需要均衡发言的场景

**"certain_num_turns"模式特点：**
- 允许同一Agent连续发言一定次数
- 更灵活的控制发言模式
- 适合需要某个Agent主导讨论的场景

```python
groupchat = GroupChat(
    agents=[speaker_a, speaker_b, speaker_c],
    messages=[],
    max_round=6,
    allow_repeat="never",  # 不允许连续发言
    max_consecutive_agent_num=2,  # 允许连续发言2次（仅certain_num_turns模式）
)
```

### 3.4 终止条件配置

GroupChat支持多种终止条件：

1. **max_round** - 达到最大轮次后强制终止
2. **is_termination_msg** - Agent返回特定消息时终止
3. **speaker_count** - 特定speaker被选择次数达到阈值
4. **自定义终止条件** - 组合多个条件

这些条件可以组合使用，任一满足即终止。

```python
# 创建会发送终止消息的Agent
agent_finisher = ConversableAgent(
    name="终结者",
    system_message="""你是终结者。
    如果任务完成或讨论足够，说'TASK_COMPLETE'来终止对话。
    否则继续讨论。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TASK_COMPLETE" in msg.get("content", ""),
)

# 配置GroupChat（组合条件）
groupchat = GroupChat(
    agents=[agent_finisher, agent_helper],
    messages=[],
    max_round=8,  # 最大8轮作为保险
)
```

### 3.5 嵌套GroupChat

嵌套GroupChat是指在一个GroupChat中调用另一个GroupChat，或者让Agent参与多个群聊。这适合层次化任务分解。

**使用场景：**
- 主群聊负责总体协调
- 子群聊负责具体任务执行
- 任务完成后汇报结果给主群聊

```python
# 创建后端开发子群聊
backend_dev = ConversableAgent(
    name="后端开发",
    system_message="你是后端开发专家，负责服务器端开发。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

backend_reviewer = ConversableAgent(
    name="后端审查",
    system_message="你是后端审查专家，负责审查代码质量。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

backend_groupchat = GroupChat(
    agents=[backend_dev, backend_reviewer],
    messages=[],
    max_round=4,
)

backend_manager = GroupChatManager(
    groupchat=backend_groupchat,
    llm_config=llm_config,
)

# 创建前端开发子群聊
frontend_dev = ConversableAgent(
    name="前端开发",
    system_message="你是前端开发专家，负责用户界面开发。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

frontend_reviewer = ConversableAgent(
    name="前端审查",
    system_message="你是前端审查专家，负责审查界面代码。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

frontend_groupchat = GroupChat(
    agents=[frontend_dev, frontend_reviewer],
    messages=[],
    max_round=4,
)

frontend_manager = GroupChatManager(
    groupchat=frontend_groupchat,
    llm_config=llm_config,
)

# 主群聊协调者
coordinator = ConversableAgent(
    name="协调者",
    system_message="""你是协调者，负责协调多个小组的工作。
    你会将任务分配给不同的小组，并收集汇报。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
```

### 3.6 实际团队协作场景

完整的软件团队协作示例：

```python
# 创建团队成员
architect = ConversableAgent(
    name="架构师",
    system_message="""你是资深系统架构师。
    职责：
    - 设计系统架构和模块划分
    - 评估技术方案可行性
    - 给出架构层面的指导建议""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

coder = ConversableAgent(
    name="程序员",
    system_message="""你是经验丰富的Python程序员。
    职责：
    - 编写高质量的代码
    - 解释代码实现逻辑
    - 根据反馈优化代码""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

tester = ConversableAgent(
    name="测试工程师",
    system_message="""你是资深测试工程师。
    职责：
    - 编写和执行测试用例
    - 发现和报告问题
    - 验证修复是否有效""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

manager = ConversableAgent(
    name="项目经理",
    system_message="""你是项目经理，负责协调团队工作。
    职责：
    - 发起和组织讨论
    - 协调各方意见
    - 推进任务进展
    - 确认任务完成""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建GroupChat
groupchat = GroupChat(
    agents=[architect, coder, tester, manager],
    messages=[],
    max_round=10,
    speaker_selection_method="auto",
    allow_repeat="never",
)

manager_groupchat = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# 发起团队讨论
result = manager.initiate_chat(
    manager_groupchat,
    message="""我们需要开发一个用户认证系统。
    请按以下流程进行：
    1. 架构师先设计系统架构
    2. 程序员根据架构实现代码
    3. 测试工程师制定测试计划
    4. 最后确认方案

    完成后说'TASK_DONE'来结束。""",
)
```

---

## 4. 协作模式的选择决策

### 4.1 模式选择指南

**何时使用双人对话：**

适用场景：
- UserProxyAgent + AssistantAgent 协作
- 人机交互式问答
- 简单的请求-响应任务
- 一对一专家咨询
- 代码生成+审查的简单流程

不适用场景：
- 需要多个Agent协作
- 需要发言均衡
- 需要广播消息

**何时使用GroupChat：**

适用场景：
- 多个专家团队讨论
- 需要LLM自动选择发言者
- 需要发言均衡控制
- 复杂的团队协作流程

不适用场景：
- 只有两个Agent
- 需要精确控制发言顺序
- 配置简单即可满足需求

### 4.2 实际选择示例

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 客服问答 | 双人对话 | 一对一用户服务，简单直接 |
| 代码审查团队 | GroupChat | 多角色协作，需要发言均衡 |
| 项目管理委员会 | GroupChat | 多部门参与，需要LLM协调 |
| 微服务开发 | 嵌套GroupChat | 主团队+子团队，层次化协作 |
| 并行数据处理 | 异步GroupChat | 多个数据源并发处理 |
| 客服聊天机器人 | 双人对话 | 一对一用户服务 |
| 技术研讨会 | GroupChat | 多个专家讨论 |

### 4.3 协作模式对比总结

| 模式 | Agent数 | 通信方式 | 复杂度 | 适用场景 |
|------|---------|----------|--------|----------|
| 双人对话 | 2 | 直接通信 | 低 | 一对一协作、人机交互 |
| GroupChat | 2+ | 广播通信 | 中 | 多团队协作、团队讨论 |
| 嵌套GroupChat | 2+ | 层次化通信 | 高 | 复杂任务分解、子团队协作 |
| 异步GroupChat | 2+ | 并发通信 | 高 | 高性能场景、并行任务 |

---

## 代码案例

本节包含两个代码案例，请参考 `15_codes/` 目录：

### 案例1：Two-Agent Chat双人对话模式配置

**文件：** `15_codes/two_agent_chat.py`

**内容要点：**
- 双人对话基本用法（initiate_chat）
- human_input_mode三种模式对比
- register_reply自定义回复逻辑
- 双人对话的终止控制（is_termination_msg, max_consecutive_auto_reply）
- 双人对话与GroupChat的区别
- 实际场景：代码审查流程
- 对话模式选择决策指南

**运行方式：**
```bash
python 15_codes/two_agent_chat.py
```

### 案例2：GroupChat综合实战

**文件：** `15_codes/groupchat_comprehensive.py`

**内容要点：**
- speaker_selection_method四种策略（auto, round_robin, random, manual）
- allow_repeat参数控制（never, certain_num_turns, always）
- 终止条件配置（max_round, is_termination_msg, 组合条件）
- 嵌套GroupChat与复杂协作
- 实际团队协作场景演示
- 协作模式选择决策指南
- 综合GroupChat配置

**运行方式：**
```bash
python 15_codes/groupchat_comprehensive.py
```

---

## 企业级实践

### 实践1：构建代码审查团队

```python
# 代码审查团队配置
coder = ConversableAgent(
    name="程序员",
    system_message="""你是经验丰富的Python程序员。
    职责：
    - 编写高质量的Python代码
    - 根据审查意见修改代码
    - 解释代码实现""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = ConversableAgent(
    name="审查员",
    system_message="""你是资深代码审查员。
    职责：
    - 审查代码质量和安全性
    - 提出具体的改进建议
    - 确认修改是否满足要求""",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "LGTM" in msg.get("content", ""),
)

# 执行代码审查流程
result = coder.initiate_chat(
    reviewer,
    message="请审查以下代码:\n{initial_code}",
)
```

### 实践2：构建AI研究团队

```python
# 创建专业的AI助手团队
researcher = ConversableAgent(
    name="研究员",
    system_message="""你是AI研究专家。
    职责：
    - 研究最新的AI技术和趋势
    - 分析论文和技术报告
    - 提供技术见解和建议""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

developer = ConversableAgent(
    name="开发工程师",
    system_message="""你是软件开发工程师。
    职责：
    - 将研究成果转化为实际应用
    - 编写高质量的代码
    - 解决技术实现问题""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

consultant = ConversableAgent(
    name="技术顾问",
    system_message="""你是技术顾问。
    职责：
    - 评估技术方案的可行性
    - 提供专业的建议
    - 帮助团队做出决策""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 配置GroupChat
groupchat = GroupChat(
    agents=[researcher, developer, consultant],
    messages=[],
    max_round=8,
    speaker_selection_method="auto",
    allow_repeat="never",
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# 发起团队讨论
result = researcher.initiate_chat(
    manager,
    message="我需要评估将大语言模型部署到边缘设备的技术可行性。",
)
```

---

## 常见误区

### 误区1：所有场景都使用双人对话

**错误做法**：所有多Agent场景都使用双人对话，通过嵌套调用实现协作

**正确做法**：
- 2个Agent协作优先使用双人对话
- 3个及以上Agent协作使用GroupChat
- 需要发言均衡时必须使用GroupChat

### 误区2：忽略allow_repeat参数

**错误做法**：使用默认的allow_repeat="always"，导致某些Agent主导对话

**正确做法**：
- 需要均衡发言时设置allow_repeat="never"
- 需要某个Agent主导时使用allow_repeat="certain_num_turns"

### 误区3：speaker_selection_method选择不当

**错误做法**：
- 复杂协作场景使用round_robin，忽略上下文
- 需要上下文理解时使用random

**正确做法**：
- 复杂协作使用auto模式
- 需要严格控制时使用manual模式
- 需要均衡发言时使用round_robin模式

### 误区4：未合理设置max_round

**错误做法**：
- max_round设置过大，导致对话过长
- max_round设置过小，任务未完成就被终止

**正确做法**：
- 根据任务复杂度合理设置max_round
- 配合is_termination_msg实现智能终止
- 作为安全网防止无限循环

---

## 本节小结

1. **五种对话模式**：双人对话、GroupChat、嵌套对话、异步对话、流式对话各有适用场景

2. **双人对话配置**：
   - 使用initiate_chat()发起对话
   - 通过human_input_mode控制人工介入程度
   - 使用register_reply注册自定义回复逻辑
   - is_termination_msg和max_consecutive_auto_reply控制终止

3. **GroupChat配置**：
   - speaker_selection_method决定发言者选择方式
   - allow_repeat控制连续发言行为
   - 多种终止条件可组合使用
   - 支持嵌套实现层次化协作

4. **协作模式选择**：
   - 2个Agent优先使用双人对话
   - 多Agent协作使用GroupChat
   - 复杂任务分解使用嵌套GroupChat
   - 根据场景需求选择合适模式

---

## 延伸阅读

- [AutoGen官方文档：GroupChat](https://microsoft.github.io/autogen/)
- [AutoGen官方文档：GroupChatManager](https://microsoft.github.io/autogen/)
- [ConversableAgent核心架构解析](../part_02_ConversableAgent核心机制深度解析/03_ConversableAgent核心架构解析.md)
- [AssistantAgent基础与典型应用场景](./16_AssistantAgent基础与典型应用场景.md)

---

## 下节预告

下一节我们将学习 **UserProxyAgent三种human_input_mode与代理行为切换**，深入理解如何在AutoGen中实现灵活的人类参与机制。