---
lesson_id: lesson_07
title: Function Calling与register_function用法
module: 代码执行器与工具执行器
---

# 第7节 Function Calling与register_function用法

## 学习目标

1. 理解register_function()的核心原理
2. 掌握函数注册与调用的完整流程
3. 能够自定义Agent工具函数

## 内容概述

Function Calling是LLM与外部工具交互的核心机制，而register_function()则是AutoGen中将Python函数注册为Agent工具的主要入口。本节将深入解析函数注册到LLM function calling的映射过程，对比register_for_llm_call与register_for_exec的差异，并探讨Tool Call与Code Executor的选择决策框架。

---

## 1. register_function核心原理

### 1.1 方法签名与参数详解

```python
def register_function(
    func: Callable,                    # 要注册的函数
    name: Optional[str] = None,        # 可选：显式指定函数名（默认使用函数本身的名字）
    description: Optional[str] = None, # 可选：描述函数用途（用于 LLM 理解何时调用）
    signature: Optional[str] = None,   # 可选：显式指定函数签名
):
```

**核心概念解析：**

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `func` | 要注册的函数对象 | 必需 |
| `name` | 注册到Agent后的工具名称 | 函数原名 |
| `description` | 告诉LLM这个函数做什么 | 函数的docstring |
| `signature` | 显式指定函数签名（覆盖自动提取） | None |

### 1.2 函数签名到LLM Function Calling的映射过程

Python函数签名通过类型注解映射为LLM可理解的工具描述格式（OpenAI Function Calling Schema）：

```python
# Python 函数
def calculate_bmi(height: float, weight: float) -> Dict[str, Any]:
    """
    计算BMI指数

    Args:
        height: 身高（米），例如 1.75
        weight: 体重（公斤），例如 70.0

    Returns:
        包含BMI值和健康建议的字典
    """
    bmi = weight / (height ** 2)
    # ...
```

**映射为JSON Schema：**

```json
{
    "name": "calculate_bmi",
    "description": "计算BMI指数\n\nArgs:\n    height: 身高（米），例如 1.75\n    weight: 体重（公斤），例如 70.0\n\nReturns:\n    包含BMI值和健康建议的字典",
    "parameters": {
        "type": "object",
        "properties": {
            "height": {
                "type": "number",
                "description": "参数 height，类型 float"
            },
            "weight": {
                "type": "number",
                "description": "参数 weight，类型 float"
            }
        },
        "required": ["height", "weight"]
    }
}
```

### 1.3 类型映射规则

| Python类型 | JSON Schema类型 |
|------------|-----------------|
| `str` | string |
| `int` | integer |
| `float` | number |
| `bool` | boolean |
| `list` / `List` | array |
| `dict` / `Dict` | object |

---

## 2. register_function完整流程演示

### 2.1 从函数定义到LLM调用的完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 步骤1：定义业务函数（带类型注解）                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤2：使用 register_function 注册到 Agent                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤3：Agent 将函数 schema 发送给 LLM                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤4：LLM 决定调用工具并返回 function_call                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤5：Agent 执行函数并返回结果                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 实际代码示例

参考 `07_codes/register_function_demo.py`：

```python
from typing import Dict, Any, Optional

# 步骤1：定义业务函数
def order_query(order_id: str, user_id: str) -> Dict[str, Any]:
    """
    查询订单状态

    Args:
        order_id: 订单ID，格式为 8位数字
        user_id: 用户ID

    Returns:
        订单信息字典，包含订单状态、金额、地址等
    """
    # 模拟订单数据
    orders = {
        "12345678": {
            "order_id": "12345678",
            "status": "已发货",
            "amount": 299.00,
            "address": "北京市朝阳区某某路1号"
        }
    }
    return orders.get(order_id, {"error": f"未找到订单 {order_id}"})

# 步骤2：注册函数
registry.register_function(
    order_query,
    name="query_order",
    description="查询用户订单的物流状态和详细信息"
)

# 步骤4：模拟 LLM Function Call
llm_call_request = {
    "name": "query_order",
    "arguments": {
        "order_id": "12345678",
        "user_id": "u_10001"
    }
}

# 步骤5：执行函数
func = registry.get_function(llm_call_request["name"])
result = func(**llm_call_request["arguments"])
```

---

## 3. register_for_llm_call vs register_for_exec

### 3.1 三种注册方式的本质区别

```
┌─────────────────────────────────────────────────────────────────┐
│                    register_for_llm_call                       │
│  作用：将函数注入 LLM 的 tool_calls，让 LLM "知道" 这个工具存在  │
│  效果：LLM 生成 function_call，但不能直接执行                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    register_for_exec                            │
│  作用：让 Agent 能够实际执行这个函数                              │
│  效果：Agent 收到 function_call 后，执行函数并返回结果           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    register_function                            │
│  作用：同时完成上述两件事                                        │
│  效果：LLM 知道工具存在 + Agent 可执行                          │
│  备注：等于 register_for_llm_call + register_for_exec            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 重要区分

| 概念 | register_for_llm_call | register_for_exec |
|------|----------------------|-------------------|
| LLM知道工具存在 | 是 | 否 |
| Agent能执行工具 | 否 | 是 |
| 单独使用是否有意义 | 否（LLM会调用但无法执行） | 否（LLM不会调用） |
| 典型使用场景 | 精细控制工具暴露 | 配合register_for_llm_call使用 |

### 3.3 代码对比

参考 `07_codes/llm_call_vs_exec.py`：

```python
# 方式1：register_function（推荐）
# 一次性完成 LLM 知道 + Agent 可执行
agent.register_function(
    func=query_order,
    name="query_order",
    description="查询订单信息"
)

# 方式2：分离注册（高级场景）
# 先注册给 LLM
agent.register_for_llm_call(name="query_order", func=query_order)

# 再注册给 Agent，但包装执行逻辑
def wrapped_query_order(order_id, user_id):
    print(f"[执行] 查询订单: {order_id}")
    return query_order(order_id, user_id)

agent.register_for_exec(name="query_order", func=wrapped_query_order)
```

---

## 4. function_map参数与register_function的异同

### 4.1 对比表

| 特性 | register_function() | function_map |
|------|---------------------|--------------|
| 注册时机 | 创建Agent后动态注册 | 创建Agent时批量注册 |
| 注册数量 | 单个函数 | 多个函数 |
| 名称自定义 | 支持 | 支持（字典key即名称） |
| 典型用法 | 运行时动态添加工具 | 初始化时批量配置 |

### 4.2 代码对比

```python
# register_function 方式
agent = ConversableAgent("assistant", llm_config)
agent.register_function(my_func)

# function_map 方式
agent = ConversableAgent(
    "assistant",
    llm_config,
    function_map={
        "custom_name": my_func,  # 可以自定义名称
        "another_name": another_func
    }
)
```

### 4.3 内部实现关联

实际上，`function_map` 在初始化时会遍历字典，对每个函数调用内部的 `register_function()` 机制：

```python
# 内部实现示意
def __init__(self, function_map=None, ...):
    if function_map:
        for name, func in function_map.items():
            self.register_function(func, name=name)
```

---

## 5. Tool Call与Code Executor的选择决策框架

### 5.1 两种执行器的本质区别

```
┌─────────────────────────────────────────────────────────────┐
│                      Tool Executor                          │
├─────────────────────────────────────────────────────────────┤
│  用途：执行已定义的 Python 函数（业务逻辑封装）             │
│  注册：register_function() / register_for_llm_call()        │
│  参数：通过 function_map 传入                              │
│  执行：LLM 生成 function_call → Agent 路由到 executor      │
│  示例：天气查询、订单处理、数据转换                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Code Executor                            │
├─────────────────────────────────────────────────────────────┤
│  用途：动态执行代码字符串（支持代码生成与执行）             │
│  配置：CodeExecutor 类 + code_execution_config               │
│  执行：LLM 生成代码 → Agent 路由到 executor → 执行 → 返回   │
│  示例：数据计算、文件处理、动态编程                         │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 决策维度

| 维度 | 选择Tool Call | 选择Code Executor |
|------|---------------|-------------------|
| 确定性 | 业务操作结果可预期 | 代码执行结果不确定 |
| 安全性 | 可测试、可审计 | 需要隔离环境(Docker) |
| 复杂性 | 简单调用 | 复杂逻辑 |
| 动态性 | 固定业务逻辑 | 需要动态生成代码 |

### 5.3 决策树

```
开始决策
    │
    ├── Q1: 这个操作需要 LLM 决定调用时机吗？
    │         │
    │         ├── No → 自动执行（不暴露给LLM）
    │         │
    │         └── Yes → Q2: 是预定义的业务函数还是动态逻辑？
    │                      │
    │                      ├── 业务函数 → Tool Call (register_function)
    │                      │
    │                      └── 动态代码 → Code Executor
```

### 5.4 典型用例对照表

| 场景 | 推荐方式 | 说明 |
|------|----------|------|
| 天气预报查询 | Tool Call | 业务函数，参数固定 |
| 数据库CRUD操作 | Tool Call | SQL执行器封装 |
| 文件格式转换 | Tool Call | 调用ffmpeg/PIL等 |
| 数据分析计算 | Code Executor | 动态生成pandas代码 |
| 正则提取处理 | Code Executor | 代码字符串执行 |
| API批量调用 | Tool Call | 业务逻辑封装 |
| 动态页面生成 | Code Executor | HTML/JS动态生成 |
| 复杂业务工作流 | Tool Call | 状态机封装 |

---

## 代码案例

本节包含两个代码案例，请参考 `07_codes/` 目录：

### 案例1：register_function核心用法

**文件：** `07_codes/register_function_demo.py`

**运行方式：**
```bash
python 07_codes/register_function_demo.py
```

**内容要点：**
- 函数签名到LLM function calling的映射过程
- FunctionRegistry模拟实现
- 业务工具注册示例
- function_map参数用法

### 案例2：register_for_llm_call与register_for_exec对比

**文件：** `07_codes/llm_call_vs_exec.py`

**运行方式：**
```bash
python 07_codes/llm_call_vs_exec.py
```

**内容要点：**
- 两种注册方式的本质区别
- 执行器架构解析
- 组合使用场景
- Tool Call与Code Executor决策框架

---

## 企业级实践

### 实践1：业务工具批量注册

```python
# 使用 function_map 批量注册业务工具
business_tools = {
    "bmi_calculator": calculate_bmi,
    "currency_converter": convert_currency,
    "weather_checker": get_weather,
}

agent = ConversableAgent(
    name="assistant",
    system_message="你是一个业务助手，可以使用各种工具",
    function_map=business_tools  # 批量注册
)
```

### 实践2：敏感操作包装执行

```python
# 对于敏感操作，在执行层添加额外验证
def wrapped_sensitive_operation(user_id: str, action: str):
    # 执行前验证
    if action == "delete" and not user_id.startswith("admin_"):
        return {"error": "权限不足：删除操作需要管理员权限"}

    # 执行实际操作
    return sensitive_operation(user_id, action)

# 分离注册以添加包装逻辑
agent.register_for_llm_call(name="sensitive_operation", func=sensitive_operation)
agent.register_for_exec(name="sensitive_operation", func=wrapped_sensitive_operation)
```

---

## 常见误区

### 误区1：混淆register_for_llm_call和register_for_exec

**错误做法：** 只使用register_for_llm_call，期望LLM调用后自动执行

**正确做法：** 需要同时注册或使用register_function一次性完成

```python
# 错误：只注册给LLM，Agent无法执行
agent.register_for_llm_call(name="tool", func=tool_func)

# 正确：使用register_function或组合使用
agent.register_function(func=tool_func, name="tool")
# 或
agent.register_for_llm_call(name="tool", func=tool_func)
agent.register_for_exec(name="tool", func=tool_func)
```

### 误区2：忽略类型注解的重要性

**错误做法：** 函数不包含类型注解，导致LLM无法正确理解参数

**正确做法：** 为所有参数添加类型注解

```python
# 错误：无类型注解
def query_order(order_id, user_id):
    return {}

# 正确：完整类型注解
def query_order(order_id: str, user_id: str) -> Dict[str, Any]:
    return {}
```

### 误区3：description过于简单

**错误做法：** 描述只写函数名，不说明用途

**正确做法：** 详细描述函数功能、参数含义和返回值

```python
# 错误：描述太简单
agent.register_function(func=calculate_bmi, description="BMI计算")

# 正确：详细描述
agent.register_function(
    func=calculate_bmi,
    description="根据身高体重计算BMI指数，评估健康状况。返回BMI值、分类（偏瘦/正常/偏胖/肥胖）和健康建议"
)
```

---

## 本节小结

1. **register_function核心原理**：将Python函数注册为Agent工具，通过类型注解映射为LLM Function Calling格式

2. **三种注册方式**：
   - `register_function()` = LLM知道 + Agent可执行
   - `register_for_llm_call()` = 仅LLM知道
   - `register_for_exec()` = 仅Agent可执行

3. **function_map与register_function**：功能等价，时机不同（初始化vs运行时）

4. **选择决策框架**：
   - 预定义业务函数 → Tool Call (register_function)
   - 动态代码执行 → Code Executor

---

## 延伸阅读

- [AutoGen官方文档：register_function](https://microsoft.github.io/autogen/)
- [OpenAI Function Calling规范](https://platform.openai.com/docs/api-reference/chat/create#functions)
- [Tool Call与Code Executor对比](../part_06_代码执行器与工具执行器/06_Function_Calling与代码执行器.md)

---

## 下节预告

下一节我们将学习 **代码执行器高级配置**，了解如何配置Docker隔离环境、超时处理机制以及自定义执行器实现。