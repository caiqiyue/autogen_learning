---
lesson_id: lesson_18
title: AssistantAgent与UserProxyAgent协作模式
module: AssistantAgent与UserProxyAgent应用
---

# 18_AssistantAgent与UserProxyAgent协作模式

## 学习目标

1. 理解AssistantAgent与UserProxyAgent的协作机制
2. 掌握两者组合使用的标准模式
3. 能够实现复杂的人机协作工作流

---

## 1. AssistantAgent与UserProxyAgent的协作架构

### 1.1 双Agent协作原理

AutoGen框架中最核心的协作模式是AssistantAgent与UserProxyAgent的配对使用。这种模式模拟了人类工作中的"专家咨询"场景：

```
┌──────────────────┐     ┌──────────────────┐
│  UserProxyAgent   │◄───►│  AssistantAgent   │
│  (用户代理/执行器) │     │    (AI助手)       │
└──────────────────┘     └──────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│   代码执行器      │     │    LLM (GPT-4o)  │
└──────────────────┘     └──────────────────┘
```

### 1.2 角色分工

| Agent | 职责 | 核心能力 |
|-------|------|----------|
| AssistantAgent | 生成响应、编写代码、智能决策 | LLM驱动，自然语言理解与生成 |
| UserProxyAgent | 接收输入、执行代码、反馈结果 | 执行驱动，人类行为模拟 |

---

## 2. initiate_chat方法与消息传递

### 2.1 initiate_chat核心参数

`initiate_chat`是启动Agent协作的核心方法，其标准签名如下：

```python
assistant.initiate_chat(
    recipient,      # 接收消息的Agent（UserProxyAgent）
    message,        # 发送的消息内容
    clear_history,  # 是否清除对话历史（默认True）
    silent,         # 是否静默模式（默认False）
)
```

### 2.2 消息传递流程

```
1. 初始化 (initiate)
   用户/系统 --> initiate_chat() --> 创建对话上下文

2. 消息发送 (Send)
   发送方Agent --> 生成消息 --> 传递给接收方Agent

3. 接收处理 (Receive)
   接收方Agent --> 解析消息 --> 调用LLM生成回复

4. 回复返回 (Response)
   接收方Agent --> 生成回复 --> 返回给发送方

5. 状态更新 (Update)
   对话历史更新 --> 检查终止条件 --> 决定是否继续
```

### 2.3 返回值ChatResult

| 字段 | 说明 |
|------|------|
| summary | 对话摘要 |
| chat_history | 完整对话历史 |
| cost | 消耗的token和成本信息 |

---

## 3. AssistantAgent + UserProxyAgent的标准协作模式

### 3.1 基础配置模板

```python
from autogen import AssistantAgent, UserProxyAgent

# 创建AI助手
assistant = AssistantAgent(
    name="assistant",
    system_message="你是一个专业的Python编程助手...",
    llm_config={
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
)

# 创建用户代理
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 完全自动化
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)

# 启动对话
assistant.initiate_chat(
    recipient=user_proxy,
    message="请用Python写一个快速排序算法，并解释其工作原理。",
    clear_history=True,
)
```

### 3.2 双向协作示例

```python
# 创建助手Agent
assistant = AssistantAgent(
    name="分析助手",
    system_message="你是一个数据分析专家...",
    llm_config=llm_config,
)

# 创建用户代理（自动模式）
user_proxy = UserProxyAgent(
    name="数据用户",
    human_input_mode="NEVER",
)

# 启动协作
assistant.initiate_chat(
    recipient=user_proxy,
    message="我有一个销售数据集，包含产品类别、地区、月销量。请问应该如何分析？",
)
```

---

## 4. UserProxyAgent的三种工作模式

### 4.1 模式总览

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| NEVER | 完全自动，不等待人工输入 | 批量处理、无人值守任务 |
| ALWAYS | 始终等待人工确认 | 高安全性场景、关键操作审核 |
| TERMINATE | 满足终止条件时自动终止，否则持续对话 | 大多数标准任务 |

### 4.2 NEVER模式

适用于自动化任务、不需要人工干预的场景。

```python
auto_agent = UserProxyAgent(
    name="自动模式代理",
    human_input_mode="NEVER",
    code_execution_config=False,
)
```

特点：
- Agent会自动生成回复，无需等待人工输入
- 适合批量处理、无人值守的任务
- 响应速度快，但缺乏人工审核

### 4.3 ALWAYS模式

适用于需要人工确认的关键操作、安全敏感的场景。

```python
manual_agent = UserProxyAgent(
    name="人工确认模式代理",
    human_input_mode="ALWAYS",
)
```

特点：
- 每次回复前都会暂停，等待人工确认
- 可以审核、修改或拒绝AI的建议
- 适合高风险操作或需要人工监督的场景

### 4.4 TERMINATE模式

适用于需要人工介入但不需全程监督的场景。

```python
terminate_agent = UserProxyAgent(
    name="终止条件模式代理",
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)
```

---

## 5. 协作工作流的配置与优化

### 5.1 代码执行工作流

```python
# 创建具备代码执行能力的UserProxyAgent
coder_proxy = UserProxyAgent(
    name="代码执行代理",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
        "timeout": 60,
    },
)

# 创建编程助手Agent
coding_assistant = AssistantAgent(
    name="Python开发助手",
    system_message="你是一位专业的Python开发者...",
    llm_config=llm_config,
)

# 协作流程：助手生成代码 -> 用户代理执行 -> 返回结果
```

### 5.2 条件终止配置

```python
def should_terminate(message):
    """自定义终止条件判断函数"""
    content = message.get("content", "")

    # 条件1：显式终止标记
    if content.rstrip().endswith("TERMINATE"):
        return True

    # 条件2：包含完成标记
    if "任务完成" in content or "完成" in content:
        return True

    # 条件3：错误标记
    if "无法完成" in content or "失败" in content:
        return True

    return False

smart_terminator = UserProxyAgent(
    name="智能终止代理",
    human_input_mode="TERMINATE",
    is_termination_msg=should_terminate,
)
```

### 5.3 智能终止条件设计策略

1. **基于内容的终止** - 检查消息是否包含特定关键词
2. **基于状态的终止** - 检查任务是否达到目标
3. **基于轮次的终止** - 限制最大对话轮次，防止无限循环
4. **基于质量的终止** - 检查输出是否满足质量标准

---

## 6. 多Agent场景下的协作设计

### 6.1 多Agent协作模式

**模式A - 串行协作（流水线模式）**

```
需求分析师 --> 架构师 --> 项目经理
分析需求    设计架构    协调执行
```

**模式B - 并行协作（分而治之）**

```
        任务分解器
            │
    ┌──────┴──────┐
    ▼             ▼
子任务1      子任务2
```

### 6.2 使用GroupChat进行多Agent协作

```python
from autogen import GroupChat, GroupChatManager

group_chat = GroupChat(
    agents=[
        requirements_analyst,
        architect,
        pm_proxy,
    ],
    messages=[],
    max_round=10,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)
```

### 6.3 完整协作工作流

```
┌─────────────────────────────────────────────────────────┐
│                      需求输入                           │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段1: 需求分析 - 需求分析师                             │
│  • 理解业务目标                                         │
│  • 分解功能需求                                         │
│  • 输出需求文档                                         │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段2: 技术设计 - 技术设计师                            │
│  • 系统架构设计                                         │
│  • 技术选型决策                                         │
│  • 输出设计文档                                         │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段3: 代码实现 - 开发者 + 代码执行器                    │
│  • 编写实现代码                                         │
│  • 执行验证代码                                         │
│  • 输出可运行代码                                       │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段4: 测试验证 - 测试工程师                            │
│  • 单元测试                                             │
│  • 集成测试                                             │
│  • 输出测试报告                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. 高级特性

### 7.1 错误处理与恢复机制

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 重试机制 | 指数退避重试 | 网络请求、临时故障 |
| 降级处理 | 失败时返回默认值 | 非关键功能、优雅退出 |
| 超时控制 | 设置最大等待时间 | LLM调用、代码执行 |
| 状态回滚 | 恢复到之前的状态 | 数据库操作、文件修改 |
| 熔断器模式 | 连续失败达到阈值后暂停 | 外部服务调用 |

### 7.2 性能优化策略

1. **消息压缩** - 对长对话进行摘要，减少token消耗
2. **并行处理** - 独立任务并行执行，减少总等待时间
3. **缓存复用** - 相同请求使用缓存结果
4. **预热策略** - 提前初始化Agent，减少冷启动时间

### 7.3 上下文保持策略

1. **系统消息持久化** - 在system_message中定义Agent的记忆机制
2. **外部状态存储** - 使用字典、数据库等外部存储保存状态
3. **对话摘要** - 在每轮对话后生成摘要，保持核心信息

---

## 代码案例

本节包含两个代码案例，请参考 `18_codes/` 目录：

### 案例1：基础协作模式

**文件：** `18_codes/basic_collaboration.py`

**内容要点：**
- AssistantAgent与UserProxyAgent的角色定义
- initiate_chat方法的参数说明与使用
- 三种human_input_mode的工作机制
- 双向协作示例
- 消息传递机制详解

**运行方式：**
```bash
python 18_codes/basic_collaboration.py
```

### 案例2：高级协作模式

**文件：** `18_codes/advanced_collaboration.py`

**内容要点：**
- 多轮对话与上下文管理
- 代码执行工作流集成
- 条件终止与复杂工作流设计
- 多Agent协作设计（GroupChat）
- 错误处理与恢复机制
- 协作性能优化
- 完整协作工作流示例

**运行方式：**
```bash
python 18_codes/advanced_collaboration.py
```

---

## 本节小结

1. **协作架构**：AssistantAgent + UserProxyAgent是AutoGen的核心双Agent协作模式

2. **消息传递**：通过initiate_chat方法启动协作，返回ChatResult包含summary、chat_history和cost

3. **工作模式**：
   - NEVER：完全自动化
   - ALWAYS：始终等待人工确认
   - TERMINATE：条件触发终止

4. **高级特性**：
   - 多轮对话上下文管理
   - 代码执行工作流集成
   - 条件终止与智能工作流控制
   - 多Agent协作（GroupChat）
   - 错误处理与性能优化

5. **设计原则**：
   - 每个阶段有明确的输入和输出
   - Agent之间通过消息传递协作
   - UserProxyAgent作为执行和验证的桥梁
   - 支持回滚和迭代优化

---

## 延伸阅读

- [AutoGen官方文档：AssistantAgent](https://microsoft.github.io/autogen/)
- [AutoGen官方文档：UserProxyAgent](https://microsoft.github.io/autogen/)
- [ConversableAgent核心架构解析](../part_02_ConversableAgent核心机制深度解析/03_ConversableAgent核心架构解析.md)
- [GroupChat多Agent协作](../part_05_MultiAgent扩展与高级应用/11_GroupChat与群聊管理.md)

---

## 下节预告

下一节我们将学习 **自定义Agent子类化与扩展**，深入理解如何通过继承ConversableAgent来创建定制化的Agent类型。