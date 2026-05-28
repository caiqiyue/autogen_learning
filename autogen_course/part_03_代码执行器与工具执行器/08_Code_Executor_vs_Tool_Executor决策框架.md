# 08_Code Executor vs Tool Executor决策框架

---
lesson_id: lesson_08
title: Code Executor vs Tool Executor决策框架
module: 代码执行器与工具执行器
---

## 学习目标

1. 理解 Code Executor vs Tool Executor 的适用场景边界
2. 能够根据任务类型选择 Code Executor 或 Tool Executor
3. 能够处理 Tool Call 与 Code Executor 的混合使用场景

---

## 1. Code Executor vs Tool Executor 决策框架

### 1.1 核心决策原则

| 任务类型 | 特征描述 | 推荐执行器 |
|---------|---------|-----------|
| **计算密集型** | 数学计算、数据处理、算法执行、统计分析 | Code Executor |
| **业务操作型** | API调用、数据库操作、文件处理、消息发送 | Tool Executor |
| **混合型** | 同时包含计算和业务操作 | 优先 Tool Executor，fallback 到 Code Executor |

### 1.2 任务类型识别

Code Executor 与 Tool Executor 的选择首先需要对任务类型进行准确识别：

```python
# 引用文件：08_codes/executor_decision.py

class TaskType(Enum):
    """
    任务类型枚举，用于决策框架

    - CALCULATION: 计算密集型 - 需要大量计算、数据处理、算法执行
    - BUSINESS_OPERATION: 业务操作型 - API调用、数据库操作、文件处理
    - HYBRID: 混合型 - 同时包含计算和业务操作
    """
    CALCULATION = "calculation"
    BUSINESS_OPERATION = "business_operation"
    HYBRID = "hybrid"
```

### 1.3 决策框架实现

```python
# 引用文件：08_codes/executor_decision.py

def make_executor_decision(task_description: str, task_code: Optional[str] = None) -> ExecutorDecision:
    """
    执行器决策框架 - 根据任务类型推荐合适的执行器

    决策逻辑：
    1. 计算密集型任务 → Code Executor（代码执行器）
    2. 业务操作型任务 → Tool Executor（工具执行器）
    3. 混合型任务 → 优先 Tool Executor，fallback 到 Code Executor
    """
    task_type = analyze_task_type(task_description, task_code)

    if task_type == TaskType.CALCULATION:
        return ExecutorDecision(
            recommended_executor="code_executor",
            task_type=task_type,
            confidence=0.9,
            reasoning="任务涉及计算密集型操作，Code Executor 可以直接执行代码并返回计算结果",
            fallback_executor="tool_executor"
        )
    elif task_type == TaskType.BUSINESS_OPERATION:
        return ExecutorDecision(
            recommended_executor="tool_executor",
            task_type=task_type,
            confidence=0.85,
            reasoning="任务涉及业务操作，Tool Executor 提供更好的错误处理和状态管理",
            fallback_executor="code_executor"
        )
    else:  # HYBRID
        return ExecutorDecision(
            recommended_executor="tool_executor",
            task_type=task_type,
            confidence=0.7,
            reasoning="混合型任务优先使用 Tool Executor，fallback 到 Code Executor",
            fallback_executor="code_executor"
        )
```

### 1.4 适用场景对比

#### Code Executor 适用场景

- **数学计算**：矩阵运算、统计分析、数值优化
- **数据处理**：数据清洗、格式转换、批量处理
- **算法实现**：排序算法、搜索算法、机器学习算法
- **代码生成**：动态生成并执行代码

#### Tool Executor 适用场景

- **API 调用**：HTTP 请求、第三方服务集成
- **数据库操作**：增删改查、事务处理
- **文件处理**：上传下载、文件读写
- **业务逻辑**：预定义的业务流程和操作

---

## 2. last_n_messages='auto' 动态回溯机制解析

### 2.1 机制原理

AutoGen 支持通过设置 `last_n_messages='auto'` 实现动态消息回溯，其核心原理是根据上下文窗口的剩余空间自动调整回溯的消息数量。

```python
# 引用文件：08_codes/executor_decision.py

class DynamicContextWindow:
    """
    动态上下文窗口管理器

    last_n_messages='auto' 机制解析：
    - AutoGen 根据当前上下文窗口大小（模型支持的 max token）
    - 动态调整回溯的消息数量，确保重要信息不被截断
    - 当上下文窗口接近满时，自动减少回溯消息数
    """

    def calculate_auto_last_n(self, current_messages_count: int) -> int:
        """
        计算 auto 模式下的 last_n_messages 值
        """
        # 计算可用 token 数
        available_tokens = self.max_tokens - self.reserved_tokens
        # 计算可用空间能容纳的消息数
        max_messages = available_tokens // self.avg_message_tokens
        # 取较小值，确保不超过实际消息数
        return min(current_messages_count, max_messages)
```

### 2.2 上下文状态监控

```python
# 引用文件：08_codes/executor_decision.py

def get_context_status(self, current_messages_count: int) -> dict:
    """
    获取当前上下文状态
    """
    recommended_last_n = self.calculate_auto_last_n(current_messages_count)
    usage_ratio = current_messages_count / (self.max_tokens / self.avg_message_tokens)

    return {
        "model": self.model_name,
        "max_tokens": self.max_tokens,
        "current_messages": current_messages_count,
        "recommended_last_n": recommended_last_n,
        "usage_ratio": f"{usage_ratio:.1%}",
        "status": "normal" if usage_ratio < 0.7 else "high" if usage_ratio < 0.9 else "critical"
    }
```

### 2.3 不同模型的动态调整示例

| 模型 | 最大 Token | 消息数 | 推荐 last_n | 使用率 | 状态 |
|------|-----------|-------|-------------|-------|------|
| gpt-4 | 8192 | 50 | 动态计算 | 动态计算 | normal/high/critical |
| gpt-3.5-turbo | 4096 | 20 | 动态计算 | 动态计算 | normal/high/critical |
| claude-3 | 200000 | 100 | 动态计算 | 动态计算 | normal/high/critical |

---

## 3. Code Executor 执行失败时的 Fallback 策略

### 3.1 错误类型与处理策略映射

```python
# 引用文件：08_codes/executor_decision.py

class ExecutorFallbackManager:
    """
    执行器失败时的 Fallback 策略管理器

    当 Code Executor 执行失败时的处理策略：
    1. 语法错误 → 尝试修复代码后重试
    2. 超时 → 减少计算量或分段执行
    3. 运行时错误 → 记录错误信息，降级到 Tool Executor
    4. 资源限制 → 减少内存/CPU 使用
    """

    ERROR_STRATEGIES = {
        "SyntaxError": "fix_and_retry",        # 语法错误：修复后重试
        "TimeoutError": "reduce_and_retry",     # 超时：减少计算量
        "RuntimeError": "fallback_to_tool",    # 运行时错误：降级到工具执行器
        "MemoryError": "reduce_memory",        # 内存错误：减少内存使用
        "ImportError": "install_and_retry",     # 导入错误：安装依赖后重试
    }
```

### 3.2 Fallback 处理流程

```python
# 引用文件：08_codes/executor_decision.py

def handle_execution_failure(
    self,
    error: Exception,
    original_code: str,
    executor_type: str
) -> dict:
    """
    处理执行失败
    """
    error_type = type(error).__name__

    # 获取处理策略
    strategy = self.ERROR_STRATEGIES.get(error_type, "log_and_fallback")

    result = {
        "error_type": error_type,
        "strategy": strategy,
        "fallback_executor": None,
        "recovery_action": None
    }

    if strategy == "fix_and_retry":
        result["recovery_action"] = self._generate_code_fix_suggestion(original_code, error_message)
        result["fallback_executor"] = "code_executor"

    elif strategy == "reduce_and_retry":
        result["recovery_action"] = "代码执行超时，建议减少计算量或分段执行"
        result["fallback_executor"] = "code_executor"

    elif strategy == "fallback_to_tool":
        result["recovery_action"] = "代码执行失败，降级到 Tool Executor 实现相同功能"
        result["fallback_executor"] = "tool_executor"

    # ... 其他策略
    return result
```

### 3.3 Fallback 策略分类

| 错误类型 | 处理策略 | Fallback 执行器 | 恢复动作 |
|---------|---------|---------------|---------|
| SyntaxError | fix_and_retry | code_executor | 修复语法后重试 |
| TimeoutError | reduce_and_retry | code_executor | 减少计算量 |
| RuntimeError | fallback_to_tool | tool_executor | 降级到工具执行器 |
| MemoryError | reduce_memory | tool_executor | 减少内存使用 |
| ImportError | install_and_retry | code_executor | 安装依赖后重试 |

---

## 4. 企业级工具插件开发规范

### 4.1 工具函数标准模板

```python
# 引用文件：08_codes/mixed_executor.py

@create_standard_tool
def fetch_stock_price(symbol: str) -> dict:
    """
    获取股票价格

    Args:
        symbol: 股票代码

    Returns:
        dict: 包含价格信息的字典
    """
    # 模拟 API 调用
    time.sleep(0.1)

    mock_prices = {
        "AAPL": 175.43,
        "GOOGL": 142.65,
        "MSFT": 378.91,
        "TSLA": 248.50
    }

    price = mock_prices.get(symbol, 100.00)

    return {
        "symbol": symbol,
        "price": price,
        "currency": "USD",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
```

### 4.2 工具元数据规范

```python
# 引用文件：08_codes/mixed_executor.py

class ToolPluginStandard:
    """
    企业级工具插件开发规范

    规范要点：
    1. 工具函数必须有清晰的文档字符串
    2. 参数必须有类型注解和默认值
    3. 返回值必须是标准格式的字典
    4. 必须包含错误处理和日志记录
    5. 工具必须有版本信息和作者信息
    """

    @staticmethod
    def create_tool_metadata(
        name: str,
        version: str,
        author: str,
        description: str,
        tags: list = None
    ) -> dict:
        """
        创建工具元数据
        """
        return {
            "name": name,
            "version": version,
            "author": author,
            "description": description,
            "tags": tags or [],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_version": "v1"
        }
```

### 4.3 工具函数验证规范

```python
# 引用文件：08_codes/mixed_executor.py

@staticmethod
def validate_tool_function(func: Callable) -> tuple:
    """
    验证工具函数是否符合规范

    验证要点：
    - 必须有文档字符串
    - 必须有类型注解
    - 必须有返回类型注解
    """
    errors = []

    if not func.__doc__:
        errors.append("缺少文档字符串")

    annotations = func.__annotations__
    if not annotations:
        errors.append("缺少类型注解")

    if 'return' not in annotations:
        errors.append("缺少返回类型注解")

    return (len(errors) == 0, errors)
```

### 4.4 企业级工具开发 Checklist

- [ ] 工具函数有完整的文档字符串
- [ ] 所有参数都有类型注解
- [ ] 返回值有明确的类型注解
- [ ] 返回标准格式的字典（包含 success、result 等字段）
- [ ] 包含错误处理逻辑
- [ ] 有版本信息和作者信息
- [ ] 包含执行日志记录
- [ ] 通过工具验证器的检查

---

## 5. 混合使用场景实战

### 5.1 混合执行器架构

```python
# 引用文件：08_codes/mixed_executor.py

class MixedExecutorOrchestrator:
    """
    混合执行器编排器

    协调 Code Executor 和 Tool Executor 的混合使用

    使用场景：
    1. 数据获取（Tool）→ 数据处理（Code）→ 结果存储（Tool）
    2. API 调用（Tool）→ 数据分析（Code）→ 可视化（Code）
    3. 文件读取（Tool）→ 数据计算（Code）→ 数据库写入（Tool）
    """
```

### 5.2 混合执行管道示例

```python
# 引用文件：08_codes/mixed_executor.py

# 定义执行管道
pipeline = [
    # 步骤1：获取股票价格（Tool）
    {"type": "tool", "name": "fetch_stock_price", "params": {"symbol": "AAPL"}},

    # 步骤2：计算股价上涨10%后的价值（Code）
    {
        "type": "code",
        "code": """
original_price = 175.43  # 从上一步获取
new_price = original_price * 1.1
shares = 100
result = {'original_price': original_price, 'new_price': new_price, 'total_value': new_price * shares}
"""
    },

    # 步骤3：保存到数据库（Tool）
    {
        "type": "tool",
        "name": "save_to_database",
        "params": {
            "data": {"symbol": "AAPL", "adjusted_price": 192.97, "shares": 100},
            "table": "stock_analysis"
        }
    }
]

# 执行管道
result = orchestrator.execute_pipeline(pipeline)
```

### 5.3 数据流图

```
┌─────────────────┐
│  Tool Executor  │
│  (fetch_stock)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Code Executor  │
│  (calculate)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Executor  │
│  (save_to_db)   │
└─────────────────┘
```

### 5.4 典型混合场景

| 场景 | 步骤1 | 步骤2 | 步骤3 |
|-----|-------|-------|-------|
| 数据分析流水线 | Tool: API获取数据 | Code: 数据清洗分析 | Tool: 结果入库 |
| 投资组合计算 | Tool: 获取多只股票价格 | Code: 计算组合价值 | Tool: 生成报告 |
| 文档处理 | Tool: 读取文件 | Code: 内容提取处理 | Tool: 保存结果 |

---

## 6. 框架参考

本节内容涉及以下模块的引用：

- **mod_004**: Code Executor 与 Tool Executor 决策框架核心模块

---

## 7. 代码文件索引

| 文件名 | 说明 |
|-------|------|
| `08_codes/executor_decision.py` | 执行器决策框架实现，包含任务类型分析、动态上下文窗口、Fallback策略 |
| `08_codes/mixed_executor.py` | 混合执行器实现，包含ToolExecutor、CodeExecutor、MixedExecutorOrchestrator |

---

## 总结

本节介绍了 Code Executor 与 Tool Executor 的决策框架，主要内容包括：

1. **决策框架**：根据任务类型（计算密集型/业务操作型/混合型）选择合适的执行器
2. **动态回溯机制**：通过 `last_n_messages='auto'` 实现上下文窗口的动态管理
3. **Fallback 策略**：当 Code Executor 执行失败时的降级处理方案
4. **企业级规范**：工具插件的开发标准和验证机制
5. **混合使用**：Tool Executor 与 Code Executor 的协同工作模式

掌握这些内容后，您将能够根据实际任务需求灵活选择和组合使用两种执行器，构建更强大、更可靠的多智能体系统。