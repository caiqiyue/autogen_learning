---
lesson_id: lesson_19
title: 自定义Agent子类化与扩展
module: AssistantAgent与UserProxyAgent应用
---

# 第19节 自定义Agent子类化与扩展

## 学习目标

1. 掌握自定义Agent的扩展方法
2. 理解业务逻辑封装与复用
3. 能够根据业务场景定制Agent

---

## 19.1 子类化模式概述

### 19.1.1 什么是子类化

子类化（Subclassing）是面向对象编程中的核心概念，指的是创建一个基于已有类的扩展类。在AutoGen框架中，通过继承`ConversableAgent`可以创建具有专用业务逻辑的自定义Agent。

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversableAgent（基类）                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ - generate_reply()                                   │    │
│  │ - register_reply()                                   │    │
│  │ - send_messages()                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬────────────────────────────────┘
                           │ 继承
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              CustomerServiceAgent（子类）                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ + 处理问候语                                         │    │
│  │ + 处理常见问题                                       │    │
│  │ + VIP用户识别                                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 19.1.2 子类化的优势

| 优势 | 说明 |
|------|------|
| **代码复用** | 复用ConversableAgent的核心功能 |
| **业务封装** | 将业务逻辑封装在Agent内部 |
| **可维护性** | 便于集中管理和修改业务逻辑 |
| **可测试性** | 可以独立测试业务逻辑 |
| **扩展性** | 支持多层继承和组合 |

### 19.1.3 子类化 vs register_reply

AutoGen提供两种扩展Agent的方式：

1. **子类化（Subclassing）**：创建新类继承ConversableAgent
2. **register_reply**：向已有Agent注册回复函数

| 特性 | 子类化 | register_reply |
|------|--------|----------------|
| 适用场景 | 复杂业务逻辑、需要持久状态 | 简单扩展、快速原型 |
| 代码组织 | 更好的代码组织 | 代码分散 |
| 状态管理 | 容易管理内部状态 | 状态管理困难 |
| 可复用性 | 高，可创建多个实例 | 一般，绑定特定实例 |

---

## 19.2 基础子类化方法

### 19.2.1 最简单的子类化

通过继承ConversableAgent并重写`__init__`方法，可以创建最简单的自定义Agent：

```python
class SimpleAgent(ConversableAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        # 添加自定义初始化逻辑
        self.custom_state = "initialized"
```

### 19.2.2 注册回复函数

在子类化中，最常用的扩展方式是在`__init__`中调用`register_reply`：

```python
class CustomerServiceAgent(ConversableAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # 注册自定义回复函数（高优先级）
        self.register_reply(
            trigger=self._is_greeting,      # 触发条件
            reply_func=self._greeting_reply, # 回复函数
            position=0                       # 优先级
        )

    def _is_greeting(self, messages, sender):
        """判断是否为问候语"""
        content = messages[-1].get("content", "").lower()
        return any(g in content for g in ["你好", "您好", "hello"])

    def _greeting_reply(self, messages, sender, config):
        """处理问候语回复"""
        return "您好！有什么可以帮助您的吗？"
```

### 19.2.3 trigger的四种类型

| 类型 | 示例 | 说明 |
|------|------|------|
| `str` | `"/help"` | 前缀匹配 |
| `Pattern` | `re.compile(r"1[3-9]\d{9}")` | 正则匹配 |
| `Callable` | `is_vip_user` | 函数判断 |
| `None` | `None` | 无条件触发 |

---

## 19.3 自定义reply_func详解

### 19.3.1 reply_func的函数签名

所有自定义reply_func必须遵循统一签名：

```python
def custom_reply(
    messages: List[Dict[str, Any]],  # 消息历史
    sender: str,                      # 发送者标识
    config: Optional[Dict[str, Any]]  # 配置字典
) -> str:
    """
    返回值说明：
    - 非空字符串：返回此内容作为回复，触发短路
    - 空字符串：表示不匹配此策略，继续检查下一个
    """
```

### 19.3.2 config参数的使用

`config`参数用于向reply_func传递配置信息：

```python
# 注册时传递config
agent.register_reply(
    trigger="VIP",
    reply_func=vip_handler,
    config={
        "greeting": "欢迎尊贵用户！",
        "discount": 0.8
    }
)

# reply_func中接收config
def vip_handler(messages, sender, config):
    greeting = config.get("greeting", "您好")
    discount = config.get("discount", 1.0)
    return f"{greeting} 您享有{discount*100:.0f}折优惠！"
```

### 19.3.3 position优先级

`position`参数控制执行顺序：

| position值 | 含义 | 建议场景 |
|-----------|------|----------|
| 0 | 最高优先级 | 紧急处理、特定命令 |
| 1-10 | 高优先级 | 常见问题、关键词匹配 |
| None | 最低优先级 | 默认处理、LLM后备 |

---

## 19.4 业务逻辑封装模式

### 19.4.1 关键词模式

适用于FAQ、常见问题等场景：

```python
class FAQAgent(ConversableAgent):
    def __init__(self, name, faq_data, **kwargs):
        super().__init__(name, **kwargs)
        self.faq_data = faq_data  # 常见问题数据

        self.register_reply(
            trigger=self._match_keyword,
            reply_func=self._keyword_reply,
            position=0
        )

    def _match_keyword(self, messages, sender):
        content = messages[-1].get("content", "").lower()
        return any(kw in content for kw in self.faq_data.keys())

    def _keyword_reply(self, messages, sender, config):
        content = messages[-1].get("content", "").lower()
        for keyword, answer in self.faq_data.items():
            if keyword in content:
                return answer
        return ""
```

### 19.4.2 正则匹配模式

适用于结构化数据提取：

```python
import re

class DataExtractionAgent(ConversableAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # 订单号匹配
        self.register_reply(
            trigger=re.compile(r"(订单号|order)[:：]?\s*(\d{10,})"),
            reply_func=self._order_handler,
            position=0
        )

        # 手机号匹配
        self.register_reply(
            trigger=re.compile(r"1[3-9]\d{9}"),
            reply_func=self._phone_handler,
            position=0
        )
```

### 19.4.3 条件判断模式

适用于需要复杂逻辑判断的场景：

```python
class VIPAgent(ConversableAgent):
    def __init__(self, name, vip_users, **kwargs):
        super().__init__(name, **kwargs)
        self.vip_users = set(vip_users)

        self.register_reply(
            trigger=self._is_vip,
            reply_func=self._vip_handler,
            position=0
        )

    def _is_vip(self, messages, sender):
        sender_name = getattr(sender, 'name', '')
        return any(vip in sender_name for vip in self.vip_users)
```

---

## 19.5 企业级设计模式

### 19.5.1 插件式架构

通过策略基类定义统一接口：

```python
class BaseStrategy:
    """策略基类"""
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

    def is_triggered(self, messages, sender) -> bool:
        raise NotImplementedError

    def execute(self, messages, sender) -> str:
        raise NotImplementedError

class KeywordStrategy(BaseStrategy):
    def __init__(self, keywords, responses):
        super().__init__("keyword", 0)
        self.keywords = keywords
        self.responses = responses

    def is_triggered(self, messages, sender):
        content = messages[-1].get("content", "").lower()
        return any(kw in content for kw in self.keywords)

    def execute(self, messages, sender):
        content = messages[-1].get("content", "").lower()
        for kw, resp in self.responses.items():
            if kw in content:
                return resp
        return ""
```

### 19.5.2 审计日志机制

企业级应用必须具备的操作记录：

```python
@dataclass
class AuditLog:
    timestamp: str
    operation: str
    request: str
    response: str
    processing_time_ms: float
    status: str

class AuditedAgent(ConversableAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.audit_logs: List[AuditLog] = []
        self._patch_generate_reply()

    def _patch_generate_reply(self):
        """修补generate_reply添加审计"""
        original = self.generate_reply

        def audited(messages, sender, **kwargs):
            start = time.time()
            result = original(messages, sender, **kwargs)
            self.audit_logs.append(AuditLog(
                timestamp=datetime.now().isoformat(),
                operation="generate_reply",
                request=messages[-1].get("content", ""),
                response=str(result),
                processing_time_ms=(time.time() - start) * 1000,
                status="success"
            ))
            return result

        setattr(self, 'generate_reply', audited)
```

### 19.5.3 策略管理器

统一管理策略的生命周期：

```python
class StrategyManager:
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}

    def register(self, strategy: BaseStrategy):
        self.strategies[strategy.name] = strategy
        self._sort_by_priority()

    def unregister(self, name: str):
        if name in self.strategies:
            del self.strategies[name]

    def enable(self, name: str):
        if name in self.strategies:
            self.strategies[name].enabled = True

    def disable(self, name: str):
        if name in self.strategies:
            self.strategies[name].enabled = False

    def _sort_by_priority(self):
        self.strategies = dict(
            sorted(self.strategies.items(),
                   key=lambda x: x[1].priority)
        )
```

---

## 19.6 代码案例

### 19.6.1 subclass_basic.py

文件路径：`part_06_AssistantAgent与UserProxyAgent应用/19_codes/subclass_basic.py`

本文件演示：
- 基础客服Agent的子类化
- 带配置的ConfigurableCustomerAgent
- 正则匹配OrderProcessAgent
- 函数触发的VIPAgent
- 组合触发器CombinedTriggerAgent

### 19.6.2 enterprise_agent.py

文件路径：`part_06_AssistantAgent与UserProxyAgent应用/19_codes/enterprise_agent.py`

本文件演示：
- 审计日志记录器（AuditLogger）
- 策略基类和插件式架构
- 企业级AuditedConversableAgent
- EnterpriseCustomerServiceAgent完整示例
- 策略管理器（StrategyManager）

### 19.6.3 运行说明

**前置条件**：
1. 确保已安装AutoGen：`pip install pyautogen`
2. 确保已在`.env`文件中配置好API密钥

**运行方式**：
```bash
cd "part_06_AssistantAgent与UserProxyAgent应用/19_codes"

# 运行基础子类化演示
python subclass_basic.py

# 运行企业级设计模式演示
python enterprise_agent.py
```

**预期输出**：
两个脚本都会输出详细的执行过程日志，展示各类自定义Agent的行为特点。

---

## 19.7 常见问题与解决方案

### Q1: 子类化后Agent没有正确响应？

**检查点**：
1. 是否调用了`super().__init__()`
2. register_reply是否使用了正确的position
3. reply_func是否返回了空字符串（表示不匹配）

**解决方案**：
```python
class MyAgent(ConversableAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)  # 必须调用父类构造

        self.register_reply(
            trigger="keyword",
            reply_func=self._handler,
            position=0  # 使用高优先级
        )
```

### Q2: 如何实现多个策略按顺序执行？

**问题**：希望多个策略都能执行，而不是短路

**解决方案**：将多个逻辑合并到一个策略中：
```python
def combined_strategy(messages, sender, config):
    results = []

    # 执行策略1
    if condition1:
        results.append(handle1(messages))

    # 执行策略2
    if condition2:
        results.append(handle2(messages))

    return "\n".join(results) if results else ""
```

### Q3: 如何让自定义Agent支持LLM后备？

**解决方案**：保留低优先级的register_reply用于规则处理，在最后让LLM处理：
```python
self.register_reply(
    trigger=None,  # 无条件触发
    reply_func=self._fallback_handler,
    position=100  # 低优先级
)
# 如果上面返回空字符串，generate_reply会调用LLM
```

---

## 19.8 本章小结

通过本章学习，你已经：

1. **理解子类化模式**：继承ConversableAgent创建专用Agent
2. **掌握四种trigger类型**：前缀匹配、正则匹配、函数判断、无条件触发
3. **学会reply_func编写**：正确编写函数签名和返回值处理
4. **了解企业级设计模式**：
   - 插件式架构（Strategy Pattern）
   - 审计日志机制
   - 策略管理器
5. **能够根据业务场景定制Agent**

下一节我们将学习AutoGen框架的实战技巧与最佳实践。

---

## 扩展阅读

- [AutoGen ConversableAgent源码](https://github.com/microsoft/autogen/blob/main/autogen/agentchat/conversable_agent.py)
- [设计模式：策略模式](https://refactoringguru.cn/design-patterns/strategy)
- [Python dataclass文档](https://docs.python.org/3/library/dataclasses.html)