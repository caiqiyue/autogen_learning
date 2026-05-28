---
lesson_id: lesson_10
title: speaker_selection_mode三种策略详解
module: 多Agent协作与GroupChat高级机制
---

# 第10节 speaker_selection_mode三种策略详解

## 学习目标

1. 掌握SpeakerSelection策略的原理
2. 理解基于LLM的智能选角机制
3. 能够自定义选角策略

---

## 10.1 核心概念解析

### 10.1.1 speaker_selection_mode 与 speaker_selection_method 的区别

在GroupChat中，有两个相关但不同的概念：

| 概念 | 说明 | 可选值 |
|-----|------|-------|
| **speaker_selection_method** | 决定选择speaker的机制 | `"auto"`, `"manual"`, `"round_robin"`, `"random"` |
| **speaker_selection_mode** | 控制是否允许重复发言 | 包含三种策略的枚举 |

```
┌─────────────────────────────────────────────────────────────┐
│                     GroupChat 配置层次                        │
├─────────────────────────────────────────────────────────────┤
│  speaker_selection_method: 决定"谁来选"                       │
│     ├── auto      → LLM根据上下文智能选择                     │
│     ├── manual    → 外部代码/用户手动指定                     │
│     ├── round_robin → 轮询强制均衡                           │
│     └── random    → 随机选择                                 │
│                                                              │
│  speaker_selection_mode: 控制"选的规则"                       │
│     └── allow_repeat 参数控制是否允许同一Agent连续发言        │
└─────────────────────────────────────────────────────────────┘
```

### 10.1.2 三种策略概述

**speaker_selection_mode** 包含三种策略：

| 策略 | 说明 | 使用场景 |
|-----|------|---------|
| **auto** | LLM根据对话上下文和Agent角色智能选择 | 复杂多Agent协作 |
| **manual** | 由外部代码或用户指定下一个发言者 | 需要精确控制对话流程 |
| **allow_repeat** | 控制同一Agent是否能连续发言 | 需要均衡发言分布 |

---

## 10.2 auto模式详解

### 10.2.1 工作原理

auto模式是默认的speaker_selection_mode，具有以下特点：

```
┌─────────────────────────────────────────────────────────────┐
│                     auto模式工作流程                          │
├─────────────────────────────────────────────────────────────┤
│  1. GroupChatManager 将当前对话上下文发送给LLM                │
│  2. LLM分析各个Agent的角色和能力                              │
│  3. LLM选择最合适的Agent作为下一个发言者                      │
│  4. 被选中的Agent接收消息并生成回复                            │
└─────────────────────────────────────────────────────────────┘
```

**auto模式的决策因素**：

1. 当前对话状态和上下文
2. 每个Agent的角色定义和能力
3. 任务的当前阶段和需求
4. 避免重复选择同一Agent（除非必要）

### 10.2.2 auto模式代码示例

参考文件：`part_04_多Agent协作与GroupChat高级机制/10_codes/speaker_selection_basic.py`

```python
# 创建具有不同角色的Agent
coder_agent = ConversableAgent(
    name="程序员",
    system_message="""你是一位经验丰富的Python程序员。
你的职责是：
- 编写高质量的Python代码
- 解释代码的实现逻辑
- 提供代码优化建议

当被选中发言时，请专注于代码相关的讨论。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer_agent = ConversableAgent(
    name="代码审查员",
    system_message="""你是一位资深的代码审查员。
你的职责是：
- 审查代码的质量和安全性
- 发现潜在的问题和bug
- 提出改进建议

当被选中发言时，请专注于代码审查和提出改进建议。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

architect_agent = ConversableAgent(
    name="架构师",
    system_message="""你是一位系统架构师。
你的职责是：
- 设计和规划系统架构
- 评估技术方案可行性
- 协调团队决策

当被选中发言时，请专注于系统设计和架构决策。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建GroupChat，默认为auto模式
groupchat = GroupChat(
    agents=[coder_agent, reviewer_agent, architect_agent],
    messages=[],
    max_round=6,
    speaker_selection_method="auto",  # 使用auto方法选择speaker
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# 启动群聊对话
result = architect_agent.initiate_chat(
    manager,
    message="我们需要设计一个用户认证系统。",
)
```

**auto模式特点**：
- 灵活性高，可适应复杂场景
- LLM根据上下文智能决策
- 可能选择不够可预测

---

## 10.3 manual模式详解

### 10.3.1 适用场景

manual模式允许外部代码或用户控制下一个发言者，适用于以下场景：

| 场景 | 说明 |
|-----|------|
| 教学演示 | 教师手动选择学生回答 |
| 流程控制 | 严格按步骤执行的任务 |
| 调试模式 | 调试群聊行为时使用 |
| 人工审核 | 重要决策需要人工确认 |

### 10.3.2 配置方式

```python
groupchat = GroupChat(
    agents=[agent_a, agent_b, agent_c],
    messages=[],
    max_round=9,
    speaker_selection_method="manual",  # 关键：使用manual方法
)
```

**manual模式使用方式**：

1. 可以通过 `groupchat.select_speaker()` 方法手动选择
2. 可以通过终止条件控制对话
3. 适合需要确定性对话流程的场景

---

## 10.4 allow_repeat参数策略

### 10.4.1 三种策略对比

`allow_repeat` 参数控制同一Agent是否能连续发言：

| 策略 | 说明 | 效果 |
|-----|------|------|
| **"never"** | 不允许同一Agent连续发言 | 确保每个Agent都有发言机会，避免某个Agent主导对话 |
| **"certain_num_turns"** | 允许连续发言一定次数 | 更灵活的控制发言模式 |
| **"always"** | 允许连续发言（默认行为） | LLM可以自由选择最合适的Agent |

### 10.4.2 策略1：allow_repeat = "never"

```python
groupchat_never = GroupChat(
    agents=[agent_x, agent_y, agent_z],
    messages=[],
    max_round=6,
    allow_repeat="never",  # 关键设置
)
```

**特点**：
- 同一Agent不能连续发言
- 确保每个Agent都有发言机会
- 避免某个Agent主导整个对话
- 适合需要均衡发言的场景

**验证示例**：

```python
# 分析发言模式
speakers = []
for msg in result.chat_history:
    if msg.get("role") == "assistant":
        name = msg.get("name", "unknown")
        speakers.append(name)

print(f"发言顺序: {' -> '.join(speakers)}")

# 检查是否有连续重复
has_consecutive = False
for i in range(len(speakers) - 1):
    if speakers[i] == speakers[i + 1]:
        has_consecutive = True
        print(f"发现连续重复: {speakers[i]} 连续发言")

if not has_consecutive:
    print("验证通过: 没有连续重复发言")
```

### 10.4.3 策略2：allow_repeat = "certain_num_turns"

```python
groupchat_certain = GroupChat(
    agents=[agent_x, agent_y, agent_z],
    messages=[],
    max_round=6,
    allow_repeat="certain_num_turns",  # 需要配合max_consecutive_agent使用
)
```

**使用场景**：
- 需要某个Agent主导讨论时
- 允许某些Agent连续发言表达深入观点
- 平衡控制与灵活性

### 10.4.4 策略3：allow_repeat = "always"（默认）

```python
groupchat_always = GroupChat(
    agents=[agent_x, agent_y, agent_z],
    messages=[],
    max_round=6,
    allow_repeat="always",  # 默认行为
)
```

**特点**：
- 允许同一Agent连续发言
- LLM可以自由选择最合适的Agent
- 适合复杂协作场景
- 可能导致发言不均衡

---

## 10.5 max_round与is_termination_msg的交互

### 10.5.1 两种终止条件

GroupChat有两个独立的终止条件：

| 终止条件 | 说明 | 配置方式 |
|---------|------|---------|
| **max_round** | 达到指定轮数后强制终止 | `GroupChat(max_round=N)` |
| **is_termination_msg** | 当Agent返回True时终止 | Agent的`is_termination_msg`参数 |

### 10.5.2 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                  终止条件检查流程                             │
├─────────────────────────────────────────────────────────────┤
│  1. Agent生成回复                                            │
│  2. 检查 is_termination_msg（Agent级别检查）                  │
│     - 如果返回True，立即终止对话                              │
│  3. 检查 max_round（GroupChat级别检查）                       │
│     - 如果当前轮次 >= max_round，终止对话                      │
│  4. 两个条件是'或'的关系（任一满足即终止）                      │
└─────────────────────────────────────────────────────────────┘
```

### 10.5.3 配置示例

```python
# Agent定义终止条件
terminator = ConversableAgent(
    name="终止者",
    system_message="""你是终止者。
当你认为讨论已经足够时，说"TASK_DONE"来结束对话。
否则继续正常讨论。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TASK_DONE" in msg.get("content", ""),
)

helper = ConversableAgent(
    name="助手",
    system_message="你是助手，帮助完成讨论。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# GroupChat配置
groupchat = GroupChat(
    agents=[terminator, helper],
    messages=[],
    max_round=10,  # 最大10轮
)

result = terminator.initiate_chat(
    manager,
    message="请分析一下云计算的发展趋势。",
)
```

**终止条件总结**：

| 优先级 | 终止条件 | 说明 |
|-------|---------|------|
| 1 | is_termination_msg | Agent级别检查，检测到立即停止 |
| 2 | max_round | GroupChat级别检查，达到最大轮次强制终止 |
| 3 | 外部终止信号 | 可以通过API发送外部终止信号 |

---

## 10.6 auto模式下的prompt优化技巧

### 10.6.1 为什么需要优化

在auto模式下，GroupChatManager会调用LLM来选择下一个发言者。**默认的提示词可能不够精确**，导致选择结果不符合预期。

**优化提示词可以**：

1. 更精确地控制speaker选择逻辑
2. 根据对话阶段动态调整选择策略
3. 实现基于条件的发言顺序
4. 提高多Agent协作的效率

### 10.6.2 默认提示词结构

默认的speaker_selection_prompt通常包含：

| 部分 | 内容 |
|-----|------|
| 当前对话状态 | 最近几条消息的内容、当前讨论的主题 |
| 可用Agent列表 | 每个Agent的名字和角色、Agent的系统提示摘要 |
| 选择指令 | 选择下一个发言者的标准、输出格式要求 |

### 10.6.3 自定义选择提示词

参考文件：`part_04_多Agent协作与GroupChat高级机制/10_codes/speaker_selection_prompt.py`

```python
custom_prompt = """你是一个对话协调者，负责选择下一个发言者。

当前对话状态：
{context}

可用发言者：
{agents}

选择规则：
1. 如果讨论刚开始或涉及研究方向，选择"研究员"
2. 如果需要实现具体功能，选择"开发者"
3. 如果需要验证或测试，选择"测试员"
4. 除非必要，避免重复选择同一个发言者

请选择一个发言者，只输出发言者的名字，不要其他内容。
输出格式：研究员 | 开发者 | 测试员
"""

groupchat = GroupChat(
    agents=[researcher, developer, tester],
    messages=[],
    max_round=9,
    speaker_selection_method="auto",
    speaker_selection_prompt=custom_prompt,  # 使用自定义提示词
)
```

### 10.6.4 基于角色描述的优化

通过在Agent的系统提示中更详细地描述角色和发言时机：

```python
planner_optimized = ConversableAgent(
    name="规划师",
    system_message="""你是项目规划师。

【角色职责】
- 分析需求，制定项目计划
- 协调团队资源分配
- 跟踪项目进度

【发言时机】
- 讨论开始时，提出项目计划
- 需要决策时，提供选项分析
- 讨论结束时，总结计划要点

【发言特点】
- 结构化表达，使用编号列表
- 简洁明了，重点突出
- 主动引导讨论方向""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
```

**增强的选择提示词**：

```python
enhanced_prompt = """你是群聊协调者，需要根据对话上下文选择最合适的下一个发言者。

【当前对话状态】
{context}

【可用Agent及其角色特征】
{agents}

【选择策略】
1. 识别当前讨论阶段：
   - 需求分析/规划阶段 → 选择"规划师"
   - 实现/开发阶段 → 选择"开发者"
   - 验证/测试阶段 → 选择"测试工程师"

2. 考虑发言均衡：
   - 除非必要，不连续选择同一Agent
   - 确保每个角色都有发言机会

3. 响应用户意图：
   - 如果用户指定了特定角色，优先考虑
   - 如果用户提问涉及特定领域，选择相关角色

【输出要求】
只输出Agent的名字，不要其他内容。
例如：规划师
"""
```

### 10.6.5 条件触发式提示词

```python
conditional_prompt = """你是一个智能对话协调者，根据条件选择下一个发言者。

【当前对话上下文】
{context}

【可用Agent及其专长】
{agents}

【条件触发规则】
1. 关键词触发：
   - 包含"数据分析"、"统计"、"可视化" → 选择"数据分析师"
   - 包含"机器学习"、"模型"、"训练" → 选择"机器学习工程师"
   - 包含"部署"、"运维"、"监控" → 选择"运维工程师"

2. 阶段触发：
   - 讨论开始阶段 → 优先"通用助手"介绍问题
   - 具体问题讨论 → 选择对应专家
   - 总结阶段 → 通用助手可以总结

3. 均衡触发：
   - 跟踪每个Agent的发言次数
   - 优先选择发言较少的Agent
   - 避免同一Agent连续发言

【输出格式】
只输出Agent名字，如：数据分析师
"""
```

### 10.6.6 高级提示词技术

**Chain of Thought + Few-shot 示例**：

```python
advanced_prompt = """你是一个智能对话协调者，需要选择下一个发言者。

【对话上下文】
{context}

【可用Agent】
{agents}

【Chain of Thought 选择过程】
请按以下步骤思考：
1. 分析当前讨论的主题和阶段
2. 确定需要什么类型的专业知识
3. 检查最近发言的Agent，避免重复
4. 选择最适合的Agent

【Few-shot 示例】
示例1：
上下文：讨论项目进度，需要了解当前状态
选择：分析师（适合提供数据概览）

示例2：
上下文：需要深入研究某个技术问题
选择：研究员（适合深入分析）

示例3：
上下文：需要制定下一步行动计划
选择：顾问（适合提供建议）

【输出要求】
首先输出选择理由（简短一行），
然后输出Agent名字。
格式：
理由：...
选择：研究员
"""
```

---

## 10.7 提示词优化常见问题与解决

### 10.7.1 问题1：提示词太模糊导致选择不稳定

**模糊提示词示例**：

```python
vague_prompt = """选择下一个发言者。
可用：Agent_A, Agent_B, Agent_C
根据感觉选择。"""
```

**问题**：LLM可能每次选择不同的Agent，结果不稳定

**解决方案**：

```python
clear_prompt = """你是一个严格的对话协调者。

【规则】
1. Agent_A 处理初始任务
2. Agent_B 处理验证
3. Agent_C 处理总结
4. 按照上述顺序轮流选择

【当前状态】
{context}

请严格按顺序选择一个Agent。
输出格式：Agent_A | Agent_B | Agent_C
"""
```

### 10.7.2 问题2：提示词偏向某些Agent

**有偏提示词示例**：

```python
biased_prompt = """优先选择Agent_A，因为它最重要。
只有Agent_A无法处理时才选择其他Agent。
可用：Agent_A, Agent_B, Agent_C"""
```

**问题**：Agent_B和Agent_C几乎不会被选中

**解决方案**：

```python
balanced_prompt = """你需要公平地选择下一个发言者。

【公平选择规则】
1. 评估每个Agent与当前任务的匹配度
2. 考虑发言均衡，优先选择发言较少的Agent
3. 除非明确需要，不连续选择同一Agent

【匹配度评估】
- Agent_A: 适合初始分析和规划
- Agent_B: 适合验证和测试
- Agent_C: 适合总结和归档

【当前状态】
{context}

请公平评估并选择。
输出格式：Agent_A | Agent_B | Agent_C
"""
```

### 10.7.3 问题3：逻辑矛盾

**矛盾提示词示例**：

```python
contradictory_prompt = """选择下一个发言者。

规则1：总是选择发言最少的Agent
规则2：总是选择Agent_A（因为它最重要）
规则3：从不连续选择同一Agent

这三条规则可能相互矛盾！
"""
```

**问题**：LLM无法同时满足所有规则，导致不可预测结果

**解决方案**：定义明确的优先级顺序

```python
prioritized_prompt = """你是一个严格的对话协调者。

【优先级规则】（按顺序应用）
1. 如果需要终止对话，选择说出"完成"的Agent
2. 如果某Agent连续发言超过2次，跳过它选择其他
3. 如果所有Agent发言次数相同，按A->B->C顺序选择
4. 其他情况根据上下文选择最合适的

【当前状态】
{context}

【发言统计】
{agent_selections}

请按优先级规则选择。
输出格式：Agent_A | Agent_B | Agent_C
"""
```

---

## 10.8 生产级别的提示词模板

```python
production_prompt = """你是群聊协调者，负责选择下一个最合适的发言者。

【当前对话状态】
{context}

【可用Agent及其专长】
{agents}

【选择策略 - 严格按此顺序执行】

第一步：检查终止条件
- 如果对话已经完成或用户要求结束，选择"完成"
- 当前Agent数量: {agent_count}
- 当前轮次: {current_round}

第二步：检查关键词触发
- 包含"需求"、"业务"、"流程" → 选择"业务分析师"
- 包含"技术"、"架构"、"代码" → 选择"技术负责人"
- 包含"进度"、"计划"、"管理" → 选择"项目经理"

第三步：检查发言均衡
- 统计当前各Agent发言次数: {agent_selections}
- 优先选择发言较少的Agent
- 除非必要，不连续选择同一Agent

第四步：基于上下文选择
- 讨论刚开始 → 选择能介绍问题的Agent
- 讨论进行中 → 根据上下文选择相关Agent
- 讨论结束前 → 可以选择能总结的Agent

【输出格式】
只输出Agent名字，不要其他内容。
例如：业务分析师
"""
```

**生产级别提示词特点**：

1. 完整的状态信息（轮次、发言统计）
2. 明确的优先级顺序
3. 清晰的输出格式
4. 处理边界情况

---

## 10.9 实际场景综合演示

### 10.9.1 场景：软件代码审查团队

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# 创建专业的代码审查团队
architect = ConversableAgent(
    name="架构师",
    system_message="""你是资深系统架构师。
职责：
- 设计系统架构和模块划分
- 评估技术方案的可行性
- 协调团队决策

在讨论中，你会根据上下文选择合适的时机发言。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

coder = ConversableAgent(
    name="程序员",
    system_message="""你是经验丰富的Python程序员。
职责：
- 编写高质量的代码
- 解释代码实现逻辑
- 根据反馈优化代码

在讨论中，你会根据上下文选择合适的时机发言。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = ConversableAgent(
    name="审查员",
    system_message="""你是资深代码审查员。
职责：
- 审查代码质量和安全性
- 发现潜在问题和bug
- 提出具体的改进建议

在讨论中，你会根据上下文选择合适的时机发言。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 配置GroupChat：auto模式 + allow_repeat="never"
groupchat = GroupChat(
    agents=[architect, coder, reviewer],
    messages=[],
    max_round=9,
    speaker_selection_method="auto",  # LLM智能选择
    allow_repeat="never",  # 不允许连续发言，确保均衡
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# 发起代码审查讨论
result = architect.initiate_chat(
    manager,
    message="""我需要审查一个新的用户认证模块。
请程序员先实现基本代码，然后审查员提出改进意见，最后架构师总结架构考虑。
完成后说'TASK_DONE'来结束讨论。""",
)
```

### 10.9.2 配置总结

| 配置项 | 值 | 说明 |
|-------|-----|------|
| speaker_selection_method | auto | LLM智能选择下一个发言者 |
| allow_repeat | never | 不允许同一Agent连续发言 |
| max_round | 9 | 最大9轮对话 |

---

## 10.10 代码案例说明

### 10.10.1 speaker_selection_basic.py

文件路径：`part_04_多Agent协作与GroupChat高级机制/10_codes/speaker_selection_basic.py`

**演示内容**：

1. **auto模式（LLM智能选择）**
   - 创建具有不同角色的Agent（程序员、审查员、架构师）
   - 演示auto模式的工作原理和选择逻辑

2. **manual模式（人类指定）**
   - 演示如何配置manual模式
   - 说明manual模式的适用场景

3. **allow_repeat参数策略**
   - never：不允许连续发言，确保均衡
   - certain_num_turns：允许连续发言一定次数
   - always：允许连续发言（默认行为）

4. **max_round与is_termination_msg的交互**
   - 场景1：只使用max_round终止
   - 场景2：使用is_termination_msg提前终止
   - 场景3：两者配合使用

5. **实际场景演示**
   - 软件代码审查团队的完整流程

### 10.10.2 speaker_selection_prompt.py

文件路径：`part_04_多Agent协作与GroupChat高级机制/10_codes/speaker_selection_prompt.py`

**演示内容**：

1. **默认speaker选择提示词**
   - 分析默认提示词的结构和内容

2. **自定义选择提示词**
   - 演示如何编写自定义的speaker_selection_prompt

3. **基于角色描述的优化**
   - 通过详细的角色描述提高选择准确性

4. **条件触发式提示词**
   - 基于关键词和对话阶段动态选择

5. **高级提示词技术**
   - Chain of Thought推理
   - Few-shot示例
   - 动态上下文

6. **提示词优化常见问题**
   - 模糊提示词、偏向提示词、逻辑矛盾

7. **生产级别的提示词模板**
   - 完整的提示词模板，包含所有最佳实践

### 10.10.3 运行说明

**前置条件**：
1. 确保已安装AutoGen：`pip install pyautogen`
2. 确保已在`.env`文件中配置好API密钥

**运行方式**：

```bash
cd part_04_多Agent协作与GroupChat高级机制/10_codes

# 演示speaker_selection_mode基本用法
python speaker_selection_basic.py

# 演示prompt优化技巧
python speaker_selection_prompt.py
```

---

## 10.11 本章小结

通过本章学习，你已经：

1. **理解speaker_selection_mode与speaker_selection_method的区别**：
   - speaker_selection_method决定"谁来选"
   - speaker_selection_mode控制"选的规则"

2. **掌握三种speaker_selection_mode策略**：
   - auto：LLM根据上下文智能选择
   - manual：外部代码/用户手动指定
   - allow_repeat：控制是否允许连续发言

3. **理解max_round与is_termination_msg的交互**：
   - 两者是"或"的关系，任一满足即终止
   - max_round作为安全网，is_termination_msg实现智能终止

4. **掌握auto模式下的prompt优化技巧**：
   - 自定义选择提示词
   - 基于角色描述的优化
   - 条件触发式提示词
   - Chain of Thought + Few-shot

5. **能够使用allow_repeat="never"避免同一Agent连续发言**：
   - 确保每个Agent都有发言机会
   - 避免某个Agent主导整个对话

---

## 扩展阅读

- [AutoGen GroupChat源码](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/group_chat.py)
- [GroupChatManager源码](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/groupchat_manager.py)
- [speaker_selection_mode官方文档](https://microsoft.github.io/autogen/docs/topics/groupchat#speaker-selection)