---
lesson_id: lesson_27
title: AutoGen高频误区与调试技巧
module: AutoGen最佳实践与高频误区
---

# 第27节 AutoGen高频误区与调试技巧
| framework_ref | mod_011 |
| 代码示例 | pitfalls_demo.py, debugging_tech.py |
| 代码路径 | 27_codes/pitfalls_demo.py, 27_codes/debugging_tech.py |

---

## 学习目标

1. 识别AutoGen开发中的常见错误
2. 掌握常见问题的排查方法
3. 能够预防潜在的运行时问题

---

## 内容概述

本节通过8个真实高频误区案例，深入分析AutoGen开发中的常见错误与调试技巧。每个误区都包含根因分析、后果评估与代码级修复方案，帮助开发者形成良好的AutoGen开发习惯。

---

## 第一部分：8个高频误区案例分析

### 误区概览

| 误区编号 | 严重程度 | 发生频率 | 排查难度 |
|---------|---------|---------|---------|
| case_001 | 高 | 高 | 低 |
| case_002 | 高 | 中 | 低 |
| case_003 | 中 | 高 | 中 |
| case_004 | 高 | 中 | 高 |
| case_005 | 高 | 低 | 高 |
| case_006 | 中 | 高 | 低 |
| case_007 | 高 | 低 | 高 |
| case_008 | 中 | 中 | 低 |

---

### case_001: is_termination_msg条件设置过宽/过严

#### 问题描述

- 终止条件设置过于宽泛，如任何包含句号的消息都终止
- 导致对话过早终止，任务未完成
- 或终止条件过于严格，永远无法满足，对话无法结束

#### 根因分析

- 对消息内容的多样性估计不足
- 条件判断过于简单，未考虑边界情况
- 例如：只检查"再见"，但Agent很少说"再见"

#### 后果评估

- **严重程度**：高
- GroupChat陷入死循环或用户无法获得最终结果
- Token消耗失控

#### 代码示例

**错误示例1：条件过于宽泛**

```python
def weak_termination_too_broad(msg):
    """错误：条件过于宽泛，任何包含句号的消息都会触发终止"""
    return "。" in msg.get("content", "")

# 这会导致第一条包含句号的消息就终止对话
agent_a = ConversableAgent(
    name="Agent_A",
    system_message="你是一个有帮助的助手。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=weak_termination_too_broad,
)
```

**错误示例2：条件过于严格**

```python
def weak_termination_too_strict(msg):
    """错误：条件过于严格，需要同时包含多个不太可能同时出现的关键词"""
    content = msg.get("content", "")
    return ("任务完成" in content and
            "结论如下" in content and
            "再见" in content)
# 条件几乎不可能同时满足，对话无法终止
```

**正确示例：多条件组合的终止条件**

```python
def correct_termination(msg):
    """正确：多条件组合，平衡宽泛和严格"""
    content = msg.get("content", "").lower()
    name = msg.get("name", "")

    # 完成标记（任一即可）
    completion_markers = ["完成", "结束", "结论", "搞定", "可以了"]
    if any(marker in content for marker in completion_markers):
        return True

    # 退出指令
    exit_markers = ["不需要再讨论", "到此为止", "再见"]
    if any(marker in content for marker in exit_markers):
        return True

    # 来自特定Agent的结论性消息
    if name == "总结员" and len(content) > 50:
        return True

    return False
```

#### 修复要点

1. 提供多种完成标记，避免单一关键词依赖
2. 包含退出指令作为备用终止方式
3. 可以根据消息来源设置不同条件

---

### case_002: max_consecutive_auto_reply设置为None导致无限循环

#### 问题描述

- 未显式设置max_consecutive_auto_reply，依赖默认值
- 误以为None表示无限制，实际上None有默认值(100)
- Agent持续自说自话，无法停止

#### 根因分析

- MAX_CONSECUTIVE_AUTO_REPLY类属性有默认值(100)
- 误以为None表示无限制
- 在某些复杂场景下，默认值仍可能触发大量循环

#### 后果评估

- **严重程度**：高
- Token消耗失控
- API成本暴涨
- 可能触发LLM的重复模式

#### 代码示例

**错误示例：不设置max_consecutive_auto_reply**

```python
agent_default = ConversableAgent(
    name="Agent_Default",
    system_message="你是一个固执的助手，会不断提问。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    # 没有显式设置max_consecutive_auto_reply
    # 使用默认值100，在复杂场景下可能触发多次循环
)
```

**正确示例：显式设置合理的值**

```python
agent_correct = ConversableAgent(
    name="Agent_Correct",
    system_message="你是一个固执的助手，会不断提问。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,  # 显式设置，限制最大连续回复数
)
```

**陷阱：max_consecutive_auto_reply=0**

```python
agent_zero = ConversableAgent(
    name="Agent_Zero",
    system_message="这是一个特殊的agent。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  # 这意味着完全禁用自动回复！
)
# max_consecutive_auto_reply=0 表示完全禁用自动回复
# Agent不会自动生成任何回复，必须通过其他方式触发回复
```

#### 修复要点

1. 始终显式设置max_consecutive_auto_reply
2. 根据任务复杂度选择合适的值：
   - 简单问答: 1-2
   - 标准任务: 5-10
   - 复杂任务: 10-20
3. 设置过小可能导致任务未完成就停止，需要权衡

---

### case_003: GroupChat中忘记设置speaker_selection_mode

#### 问题描述

- 未理解auto模式的LLM推荐逻辑
- 可能推荐不合适的下一个发言者
- 对话质量不可控、发言顺序混乱

#### 根因分析

- 使用默认的auto模式
- LLM可能根据偏好选择而非任务需求
- 在复杂对话中，缺乏对发言顺序的控制

#### 后果评估

- **严重程度**：中
- 对话质量不可控
- 某些Agent可能被忽视
- 发言顺序可能不符合业务逻辑

#### 代码示例

**错误示例：使用默认的auto模式**

```python
groupchat_default = GroupChat(
    agents=[analyst, developer, tester],
    messages=[],
    max_round=10,
    # 没有显式设置speaker_selection_method
    # 默认是"auto"，LLM可能选择不合适的发言者
)
```

**正确示例1：使用round_robin强制均衡**

```python
groupchat_rr = GroupChat(
    agents=[analyst, developer, tester],
    messages=[],
    max_round=9,  # 确保每个Agent都能发言3次
    speaker_selection_method="round_robin",  # 强制轮询
)
```

**正确示例2：使用manual模式手动控制**

```python
groupchat_manual = GroupChat(
    agents=[analyst, developer, tester],
    messages=[],
    max_round=10,
    speaker_selection_method="manual",  # 手动控制
)
```

**正确示例3：自定义选择函数**

```python
speaker_history: Dict[str, int] = {}

def smart_select_speaker(groupchat: GroupChat, last_speaker=None):
    """智能选择函数：优先选择发言次数最少的Agent"""
    agents = groupchat.agents
    if len(agents) == 1:
        return agents[0]

    # 排除上一个发言者
    candidates = [a for a in agents if a != last_speaker]
    if len(candidates) == 1:
        return candidates[0]

    # 选择发言次数最少的候选者
    def get_count(agent):
        return speaker_history.get(agent.name, 0)

    return min(candidates, key=get_count)

groupchat_custom = GroupChat(
    agents=[analyst, developer, tester],
    messages=[],
    max_round=10,
    speaker_selection_method=smart_select_speaker,
)
```

#### 修复要点

1. 根据业务场景选择合适的speaker_selection_method
2. 严格交替发言使用round_robin
3. 需要人工指导使用manual
4. 需要智能选择使用auto或自定义函数

---

### case_004: Tool Call与Code Executor混用

#### 问题描述

- register_function和register_for_llm_call配置混淆
- LLM知道工具存在但无法执行
- 执行结果未反馈给LLM
- 工具调用行为不一致

#### 根因分析

- 不理解register_function vs register_for_llm_call vs register_for_exec的区别
- 混用导致LLM和实际执行不匹配
- 缺少正确的错误处理

#### 后果评估

- **严重程度**：高
- 工具调用失败但LLM不知道
- 响应不一致，用户体验差
- 调试困难，难以定位问题

#### 代码示例

**错误示例1：只注册给LLM，不注册执行**

```python
agent_llm_only = ConversableAgent(
    name="Agent_LLM_Only",
    system_message="你是一个计算器助手，可以帮助用户计算数学表达式。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 错误：只注册给LLM，没有注册执行
agent_llm_only.register_for_llm_call(name="calculator", description="计算数学表达式")
# 问题：LLM知道有这个工具，但无法调用它
```

**正确示例：明确分离注册**

```python
agent_correct = ConversableAgent(
    name="Agent_Correct",
    system_message="你是一个计算器助手，可以帮助用户计算数学表达式。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 注册给LLM（让它知道有这个工具）
agent_correct.register_for_llm_call(
    name="calculate",
    description="计算数学表达式的值，支持加减乘除和括号"
)
# 注册给执行器（让它知道如何执行）
agent_correct.register_for_execution(
    name="calculate",
    description="计算数学表达式的值"
)
```

#### 修复要点

1. 使用register_for_llm_call + register_for_execution组合
2. 确保name一致，否则会出现找不到函数的问题
3. description要清晰，帮助LLM理解何时调用
4. 或者使用register_function一次性完成注册

---

### case_005: async代码中使用同步generate_reply导致死锁

#### 问题描述

- 在异步上下文中调用同步方法
- 阻塞事件循环
- 多Agent并发场景下系统假死

#### 根因分析

- 不理解同步和异步方法的区别
- 在async函数中调用了同步的generate_reply
- 缺少正确的异步封装

#### 后果评估

- **严重程度**：高
- 系统假死，无法响应
- 多Agent并发时问题更明显
- 调试困难，死锁不易复现

#### 代码示例

**错误示例：在async函数中调用同步方法**

```python
async def wrong_async_usage():
    """错误：在async函数中调用同步方法，这会阻塞事件循环"""
    result = agent.initiate_chat(  # 错误：在async中调用同步方法
        recipient=agent,
        message="你好",
        max_consecutive_auto_reply=2
    )
    return result
```

**正确示例：使用异步方法**

```python
async def correct_async_usage():
    """正确：使用异步方法进行通信"""
    result = await agent.a_initiate_chat(  # 使用异步方法
        recipient=agent,
        message="你好",
        max_consecutive_auto_reply=2
    )
    return result
```

**正确示例：并发执行多个对话**

```python
async def concurrent_chats():
    """正确：使用asyncio.gather并发执行多个对话"""
    agent_a = ConversableAgent(...)
    agent_b = ConversableAgent(...)

    # 并发执行两个对话
    task1 = agent_a.a_initiate_chat(
        recipient=agent_b,
        message="你好",
        max_consecutive_auto_reply=1
    )

    task2 = agent_b.a_initiate_chat(
        recipient=agent_a,
        message="你好",
        max_consecutive_auto_reply=1
    )

    # 使用asyncio.gather并发执行
    results = await asyncio.gather(task1, task2)
    return results
```

#### 修复要点

1. 异步环境中统一使用异步方法（a_initiate_chat）
2. 不要在async函数中调用同步方法
3. 使用asyncio.gather并发执行多个任务
4. 注意：generate_reply有对应的a_generate_reply

---

### case_006: llm_config设置为False但仍期望Agent自动生成回复

#### 问题描述

- 将llm_config=False理解为禁用LLM
- 导致Agent无法生成回复
- Agent始终返回空回复或default_auto_reply

#### 根因分析

- 不理解llm_config=False的含义
- 误以为False表示"使用默认配置"
- 实际上False表示不使用LLM

#### 后果评估

- **严重程度**：中
- Agent无法生成有意义的回复
- 对话无法正常进行
- 调试困难，错误信息不明确

#### 代码示例

**错误示例：llm_config=False但期望自动回复**

```python
agent_wrong = ConversableAgent(
    name="Agent_Wrong",
    system_message="你是一个有帮助的助手。",
    llm_config=False,  # 这会禁用LLM！
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
)
# 问题：llm_config=False会禁用LLM，无法自动回复
# 效果：Agent会使用default_auto_reply或返回空消息
```

**正确示例：llm_config=False的正确使用场景**

```python
# 正确场景1：纯人工输入代理
agent_human = ConversableAgent(
    name="Agent_Human",
    system_message="你是人工代理，只转发人类输入。",
    llm_config=False,  # 禁用LLM，节省资源
    human_input_mode="ALWAYS",  # 等待人类输入
)

# 正确场景2：纯代码执行代理
agent_code = ConversableAgent(
    name="Agent_Code",
    system_message="你是代码执行代理。",
    llm_config=False,  # 禁用LLM，使用代码执行器
    human_input_mode="NEVER",
)
```

#### 修复要点

1. llm_config=False表示完全禁用LLM
2. 只有在纯人工输入或纯代码执行场景才使用False
3. 需要自动回复时必须提供有效的llm_config
4. 注意：如果llm_config=False但human_input_mode="NEVER"，Agent将无法响应

---

### case_007: 嵌套GroupChat中的状态污染

#### 问题描述

- 未理解nested group chat的消息隔离机制
- 子群聊消息混入父群聊
- GroupChatManager状态混乱、消息溯源失败

#### 根因分析

- 嵌套GroupChat共享状态管理不当
- 消息传递参数配置错误
- 未正确隔离不同层级的消息

#### 后果评估

- **严重程度**：高
- 消息溯源失败
- 状态混乱导致不可预测行为
- 对话内容泄露到错误的层级

#### 代码示例

**错误示例：嵌套GroupChat未正确隔离消息**

```python
# 创建子GroupChat（未隔离）
sub_groupchat = GroupChat(
    agents=[subtask_a, subtask_b],
    messages=[],  # 共享空列表？
    max_round=5,
)
# 问题：子群聊的messages可能与父群聊共享
# 后果：子群聊消息混入父群聊，状态污染
```

**正确示例：正确配置嵌套GroupChat**

```python
# 子GroupChat：使用独立配置
sub_groupchat_isolated = GroupChat(
    agents=[subtask_agent_a, subtask_agent_b],
    messages=[],  # 独立的空列表
    max_round=5,
    name="sub_groupchat",  # 独立名称
)
```

**正确的嵌套调用方式**

```python
# 通过initiate_chat启动子群聊
result = main_agent.initiate_chat(
    sub_manager_isolated,
    message="请完成子任务",
    # 关键：传递父群聊的消息历史
    chat_messages=main_agent.chat_messages.get(sub_manager_isolated, {}),
)

# 从结果中提取子群聊的输出
sub_output = result.summary or result.last_message()
```

#### 修复要点

1. 每个GroupChat使用独立的messages列表
2. 使用chat_messages参数传递消息历史
3. 正确区分不同层级的消息
4. 使用summary或last_message获取子群聊结果

---

### case_008: 未配置price字段导致成本计算不准确

#### 问题描述

- 使用非OpenAI模型时未配置price字段
- AutoGen无法计算成本
- 成本监控失效、预算超支风险

#### 根因分析

- 不理解price字段的作用
- 误以为price只是用于显示
- 未考虑成本控制的必要性

#### 后果评估

- **严重程度**：中
- 成本监控失效
- 预算超支风险
- 无法进行成本优化

#### 代码示例

**错误示例：缺少price字段**

```python
config_without_price = {
    "config_list": [{
        "model": "qwen2.5:3b",
        "api_key": "ollama-key",
        "base_url": "http://localhost:11434",
        # 缺少price字段！
    }]
}
# 问题：AutoGen无法计算成本，成本监控失效
```

**正确示例：配置price字段**

```python
config_with_price = {
    "config_list": [{
        "model": "qwen2.5:3b",
        "api_key": "ollama-key",
        "base_url": "http://localhost:11434",
        "price": [0.0001, 0.0002],  # 成本配置
    }]
}
# price字段格式: [input_price_per_1k, output_price_per_1k]
```

**价格配置说明**

```
price字段格式: [input_price_per_1k, output_price_per_1k]
  - 第一个值：每1000个输入token的价格（美元）
  - 第二个值：每1000个输出token的价格（美元）

常见模型的价格参考:
  - gpt-4o: [0.005, 0.015]
  - gpt-4o-mini: [0.00015, 0.0006]
  - qwen2.5:3b (本地): [0, 0]  # 免费
  - claude-3-opus: [0.015, 0.075]
```

**使用成本监控**

```python
from autogen import ChatCompletion

# 获取对话成本
cost = ChatCompletion.get_cost(
    model="gpt-4o-mini",
    prompt_tokens=1000,
    completion_tokens=500,
    price_config=config_with_price,
)
print(f"对话成本: ${cost}")
```

#### 修复要点

1. 为每个模型配置price字段
2. 本地模型可以设置为[0, 0]
3. 设置price后可以：
   - 监控对话成本
   - 设置预算上限
   - 优化模型选择
4. 使用get_model_cost估算对话成本

---

## 第二部分：AutoGen调试技巧

### 调试技巧概览

| 类别 | 技巧 | 难度 | 效果 |
|------|------|------|------|
| 日志配置 | verbose模式 | 低 | 高 |
| 消息分析 | chat_history | 低 | 高 |
| 状态检查 | agent属性检查 | 中 | 高 |
| GroupChat监控 | select_speaker | 中 | 高 |
| 性能分析 | token追踪 | 中 | 中 |
| 成本追踪 | price计算 | 低 | 高 |

---

### 技巧1：日志配置与调试输出

AutoGen使用Python的logging模块，可以通过配置日志级别来获取不同详细程度的调试信息。

**常用日志级别：**

- CRITICAL (50): 严重错误，导致程序无法继续
- ERROR (40): 错误，但程序可以继续
- WARNING (30): 警告，可能有问题
- INFO (20): 一般信息
- DEBUG (10): 调试信息，最详细

**代码示例：配置全局日志级别**

```python
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 获取autogen的logger
logger = logging.getLogger('autogen')
logger.setLevel(logging.DEBUG)
```

**使用verbose模式**

```python
agent = ConversableAgent(
    name='agent',
    system_message='你是一个有帮助的助手。',
    llm_config=llm_config,
    verbose=True,  # 启用详细输出
)
# 效果：每条消息都会打印详细信息
# 适用场景：开发调试时使用，生产环境建议关闭
```

**自定义调试装饰器**

```python
def debug_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[DEBUG] 调用函数: {func.__name__}")
        print(f"[DEBUG] 参数: args={args}, kwargs={kwargs}")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[DEBUG] 函数 {func.__name__} 执行完成，耗时: {elapsed:.3f}s")
        return result
    return wrapper
# 使用：@debug_decorator 装饰需要调试的函数
```

---

### 技巧2：消息历史分析

通过分析chat_history可以：
- 了解对话的完整流程
- 识别异常消息
- 检查消息内容是否符合预期
- 分析对话轮次是否合理

**打印完整消息历史**

```python
def print_chat_history(chat_history, max_content_length=100):
    """打印消息历史，每条消息最多显示指定长度"""
    print(f"\n总消息数: {len(chat_history)}")
    print("=" * 60)

    for i, msg in enumerate(chat_history):
        name = msg.get('name', 'unknown')
        content = msg.get('content', '')
        role = msg.get('role', 'unknown')

        # 截断过长的内容
        display_content = content[:max_content_length]
        if len(content) > max_content_length:
            display_content += '...'

        print(f"[{i}] {role}/{name}: {display_content}")

    print("=" * 60)
```

**统计每个Agent的发言次数**

```python
def analyze_speaker_distribution(chat_history):
    """分析发言者分布"""
    speaker_counts = {}
    for msg in chat_history:
        name = msg.get('name', 'unknown')
        speaker_counts[name] = speaker_counts.get(name, 0) + 1
    return speaker_counts
```

**分析消息长度分布**

```python
def analyze_message_lengths(chat_history):
    """分析消息长度分布"""
    lengths = [len(msg.get('content', '')) for msg in chat_history]

    return {
        'total': sum(lengths),
        'average': sum(lengths) / len(lengths) if lengths else 0,
        'min': min(lengths) if lengths else 0,
        'max': max(lengths) if lengths else 0,
    }
```

**查找特定消息**

```python
def find_messages_by_keyword(chat_history, keyword):
    """查找包含特定关键词的消息"""
    results = []

    for i, msg in enumerate(chat_history):
        content = msg.get('content', '')
        if keyword.lower() in content.lower():
            results.append({
                'index': i,
                'name': msg.get('name', 'unknown'),
                'content': content,
            })

    return results
# 使用：find_messages_by_keyword(history, '完成')
```

---

### 技巧3：Agent状态检查

通过检查Agent的各种属性，可以了解Agent的配置状态。

**打印Agent配置状态**

```python
def print_agent_state(agent: ConversableAgent):
    """打印Agent的配置状态"""
    print(f"\nAgent名称: {agent.name}")
    print("-" * 40)
    print(f"系统消息: {agent.system_message[:50]}...")
    print(f"人类输入模式: {agent.human_input_mode}")
    print(f"最大连续回复数: {agent.max_consecutive_auto_reply}")
    print(f"LLM配置: {'已设置' if agent.llm_config else '未设置'}")

    if hasattr(agent, 'chat_messages') and agent.chat_messages:
        print(f"当前对话数: {len(agent.chat_messages)}")
    else:
        print("当前对话数: 0")
```

**检查LLM配置有效性**

```python
def check_llm_config(agent: ConversableAgent) -> Tuple[bool, str]:
    """检查LLM配置是否有效"""
    if not agent.llm_config:
        return False, "LLM配置未设置"

    config_list = agent.llm_config.get('config_list', [])
    if not config_list:
        return False, "config_list为空"

    for i, config in enumerate(config_list):
        if 'model' not in config:
            return False, f"config[{i}]缺少model字段"

    return True, "配置有效"
```

---

### 技巧4：GroupChat监控

GroupChat的监控重点：
1. speaker选择过程
2. 消息广播
3. 终止条件触发
4. 发言者分布统计

**监控speaker选择过程**

```python
selection_history = []

def monitored_select_speaker(groupchat: GroupChat, last_speaker=None):
    """带监控的speaker选择函数"""
    # 执行实际的选择逻辑
    selected = groupchat.select_speaker(last_speaker)

    # 记录选择过程
    selection_history.append({
        'last_speaker': last_speaker.name if last_speaker else None,
        'selected': selected.name,
        'round': len(groupchat.messages),
    })

    # 打印日志
    print(f"[MONITOR] Round {len(groupchat.messages)}: "
          f"{last_speaker.name if last_speaker else 'None'} -> {selected.name}")

    return selected
```

**监控终止条件触发**

```python
def create_logged_termination(condition_func: Callable, name: str = "termination"):
    """创建带日志的终止条件函数"""
    def logged_condition(msg):
        result = condition_func(msg)
        content_preview = msg.get('content', '')[:30]
        print(f"[{name.upper()}] '{content_preview}...' -> {result}")
        return result

    return logged_condition
```

**分析发言者分布**

```python
def analyze_speaker_stats(messages: List[Dict]) -> Dict[str, Any]:
    """分析GroupChat中发言者的统计信息"""
    if not messages:
        return {'total_messages': 0, 'speakers': {}}

    speaker_counts = {}
    message_lengths = {}

    for msg in messages:
        name = msg.get('name', 'unknown')
        content = msg.get('content', '')

        speaker_counts[name] = speaker_counts.get(name, 0) + 1
        message_lengths[name] = message_lengths.get(name, []) + [len(content)]

    # 计算每个发言者的平均消息长度
    avg_lengths = {
        name: sum(lengths) / len(lengths) if lengths else 0
        for name, lengths in message_lengths.items()
    }

    return {
        'total_messages': len(messages),
        'speakers': speaker_counts,
        'average_lengths': avg_lengths,
    }
```

---

### 技巧5：性能分析与成本追踪

通过监控Token消耗和响应时间，可以：
- 评估系统性能
- 控制成本
- 识别性能瓶颈

**计算对话成本**

```python
def calculate_chat_cost(prompt_tokens: int, completion_tokens: int,
                       model: str = "gpt-4o-mini") -> float:
    """计算对话成本"""
    # 价格配置（每1000 token的价格）
    prices = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4": (0.03, 0.06),
        "gpt-3.5-turbo": (0.001, 0.002),
    }

    if model not in prices:
        print(f"警告：{model}不在价格表中，使用gpt-4o-mini的价格")
        model = "gpt-4o-mini"

    input_price, output_price = prices[model]

    cost = (prompt_tokens * input_price / 1000 +
            completion_tokens * output_price / 1000)

    return cost
```

**追踪响应时间**

```python
def time_llm_call(func: Callable) -> Callable:
    """装饰器：追踪LLM调用时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[TIMING] {func.__name__} 耗时: {elapsed:.3f}s")
        return result
    return wrapper
# 使用：@time_llm_call 装饰需要计时的函数
```

**性能监控器**

```python
class PerformanceMonitor:
    """性能监控器：追踪LLM调用的各项指标"""

    def __init__(self):
        self.calls = []
        self.total_tokens = 0
        self.total_cost = 0.0

    def record_call(self, prompt_tokens: int, completion_tokens: int,
                   model: str, elapsed_time: float):
        """记录一次LLM调用"""
        cost = calculate_chat_cost(prompt_tokens, completion_tokens, model)

        self.calls.append({
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'model': model,
            'cost': cost,
            'elapsed_time': elapsed_time,
        })

        self.total_tokens += prompt_tokens + completion_tokens
        self.total_cost += cost

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.calls:
            return {'total_calls': 0}

        elapsed_times = [c['elapsed_time'] for c in self.calls]

        return {
            'total_calls': len(self.calls),
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_response_time': sum(elapsed_times) / len(elapsed_times),
            'min_response_time': min(elapsed_times),
            'max_response_time': max(elapsed_times),
        }

    def print_report(self):
        """打印性能报告"""
        stats = self.get_stats()

        print("\n" + "=" * 40)
        print("性能报告")
        print("=" * 40)
        print(f"总调用次数: {stats['total_calls']}")
        print(f"总Token数: {stats['total_tokens']:,}")
        print(f"总成本: ${stats['total_cost']:.6f}")

        if 'avg_response_time' in stats:
            print(f"平均响应时间: {stats['avg_response_time']:.3f}s")
            print(f"最快响应时间: {stats['min_response_time']:.3f}s")
            print(f"最慢响应时间: {stats['max_response_time']:.3f}s")

        print("=" * 40)
```

---

### 技巧6：常见问题诊断流程

#### 诊断流程1：对话无法终止

```
诊断步骤：
1. 检查is_termination_msg是否正确配置
   - 条件是否过于严格？
   - 是否永远无法满足？

2. 检查max_consecutive_auto_reply设置
   - 是否设置为None？
   - 是否设置为过大的值？

3. 检查GroupChat的max_round
   - 是否设置为过大的值？

4. 检查Agent的系统提示
   - 是否包含可能导致无限循环的指令？

5. 检查LLM是否陷入重复模式
   - 多次对话后是否开始重复？

检查清单：
[ ] is_termination_msg返回True的条件是否合理？
[ ] max_consecutive_auto_reply是否设置？
[ ] GroupChat的max_round是否合理？
[ ] Agent系统提示是否包含循环指令？
```

#### 诊断流程2：Agent不响应

```
诊断步骤：
1. 检查llm_config配置
   - 是否设置为False？
   - config_list是否正确？

2. 检查human_input_mode
   - 是否设置为ALWAYS？（会等待人类输入）
   - 是否设置为TERMINATE？（会在特定条件停止）

3. 检查Agent类型
   - 是否是UserProxyAgent但没有正确配置？

4. 检查消息传递
   - initiate_chat是否正确调用？
   - 消息是否正确传递给Agent？

检查清单：
[ ] llm_config是否正确配置？
[ ] human_input_mode是否正确？
[ ] Agent类型是否匹配场景？
[ ] initiate_chat参数是否正确？
```

#### 诊断流程3：GroupChat发言不均

```
诊断步骤：
1. 检查speaker_selection_method
   - 是否使用auto模式？
   - LLM可能存在偏好

2. 检查Agent系统提示
   - 不同Agent的提示是否差异过大？
   - 某些Agent是否更健谈？

3. 检查Agent数量
   - Agent数量是否不均衡？

4. 检查消息内容
   - 是否某些话题只与特定Agent相关？

解决方案：
- 使用round_robin强制均衡
- 使用manual模式手动控制
- 自定义均衡选择函数

检查清单：
[ ] speaker_selection_method是否合适？
[ ] Agent系统提示是否平衡？
[ ] 是否需要使用round_robin？
```

#### 诊断流程4：成本异常高

```
诊断步骤：
1. 检查Token消耗
   - 对话轮次是否过多？
   - 每次回复的Token数是否过大？

2. 检查max_consecutive_auto_reply
   - 是否设置为None或过大的值？
   - 是否触发无限循环？

3. 检查模型选择
   - 是否使用了过大的模型？
   - 是否需要对不同任务使用不同模型？

4. 检查price配置
   - 是否为每个模型配置了price字段？
   - 价格是否正确？

解决方案：
- 设置max_consecutive_auto_reply限制
- 使用更小的模型处理简单任务
- 配置price字段进行成本监控
- 设置预算上限

检查清单：
[ ] max_consecutive_auto_reply是否设置？
[ ] 是否配置了price字段？
[ ] 模型选择是否合理？
[ ] 对话轮次是否过多？
```

**综合诊断脚本**

```python
def diagnose_autogen_issue(agent_or_groupchat):
    """AutoGen问题综合诊断函数"""
    issues = []
    warnings = []

    # 诊断Agent
    if isinstance(agent_or_groupchat, ConversableAgent):
        agent = agent_or_groupchat

        # 检查1：llm_config
        if not agent.llm_config:
            issues.append("llm_config未设置，Agent无法自动回复")

        # 检查2：max_consecutive_auto_reply
        if agent.max_consecutive_auto_reply is None:
            warnings.append("max_consecutive_auto_reply为None，使用默认值")

        # 检查3：human_input_mode
        if agent.human_input_mode == "ALWAYS":
            warnings.append("human_input_mode='ALWAYS'会等待人类输入")

        # 检查4：is_termination_msg
        if agent.is_termination_msg is None:
            warnings.append("is_termination_msg未设置")

    # 诊断GroupChat
    elif isinstance(agent_or_groupchat, GroupChat):
        groupchat = agent_or_groupchat

        # 检查1：speaker_selection_method
        if groupchat.speaker_selection_method == "auto":
            warnings.append("speaker_selection_method='auto'可能导致发言不均")

        # 检查2：max_round
        if groupchat.max_round > 50:
            warnings.append("max_round设置过大，可能导致成本过高")

        # 检查3：termination_msg
        if groupchat.termination_msg is None:
            warnings.append("termination_msg未设置，对话可能无法正常终止")

    return {
        'issues': issues,
        'warnings': warnings,
        'is_healthy': len(issues) == 0,
    }
```

---

## 第三部分：AutoGen开发调试清单

### 配置检查

```
[ ] llm_config是否正确配置？（不使用False，除非是纯代码执行）
[ ] config_list是否包含有效的模型配置？
[ ] 是否为非OpenAI模型配置了price字段？
[ ] base_url是否正确（如果使用代理）？
```

### Agent配置

```
[ ] max_consecutive_auto_reply是否显式设置？
[ ] human_input_mode是否正确（NEVER/ALWAYS/TERMINATE）？
[ ] is_termination_msg是否合理配置？
[ ] system_message是否包含可能导致问题的指令？
```

### GroupChat配置

```
[ ] speaker_selection_method是否合适？
[ ] max_round是否设置合理？
[ ] termination_msg是否正确配置？
[ ] agents列表是否正确？
```

### 调试技巧

```
[ ] 是否启用了verbose模式进行调试？
[ ] 是否打印了chat_history进行分析？
[ ] 是否监控了发言者分布？
[ ] 是否追踪了Token消耗和成本？
```

### 常见问题快速修复

```
[ ] 对话无法终止 -> 检查is_termination_msg和max_consecutive_auto_reply
[ ] Agent不响应 -> 检查llm_config和human_input_mode
[ ] 发言不均 -> 使用round_robin或调整speaker_selection_method
[ ] 成本过高 -> 设置max_consecutive_auto_reply，配置price
```

---

## 总结

### 8个高频误区

| 误区 | 核心问题 | 修复方案 |
|------|---------|---------|
| case_001 | is_termination_msg条件设置过宽/过严 | 多条件组合，设置合理的完成标记 |
| case_002 | max_consecutive_auto_reply设置为None | 显式设置合理的值(1-20) |
| case_003 | GroupChat忘记设置speaker_selection_mode | 根据场景选择round_robin/manual/auto |
| case_004 | Tool Call与Code Executor混用 | 使用register_for_llm_call + register_for_execution |
| case_005 | async代码中使用同步方法 | 使用a_initiate_chat等异步方法 |
| case_006 | llm_config=False但期望自动回复 | 只有纯人工/纯代码场景才用False |
| case_007 | 嵌套GroupChat状态污染 | 每个GroupChat使用独立的messages列表 |
| case_008 | 未配置price字段 | 为每个模型配置price字段 |

### 6大调试技巧

1. **日志配置与调试输出** - 启用verbose和DEBUG日志
2. **消息历史分析** - 打印和分析chat_history
3. **Agent状态检查** - 检查Agent配置属性
4. **GroupChat监控** - 监控speaker选择和终止条件
5. **性能分析与成本追踪** - 追踪Token和成本
6. **常见问题诊断流程** - 提供标准化诊断流程

---

## 相关代码文件

- `27_codes/pitfalls_demo.py` - 8个高频误区代码演示
- `27_codes/debugging_tech.py` - 调试技巧代码演示