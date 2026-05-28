# executor_decision.py
# 第8节 Code Executor vs Tool Executor决策框架 - 决策框架实际应用
#
# 本文件演示如何根据任务类型选择合适的执行器
#
# 决策框架核心原则：
# - 计算密集型任务（如数学计算、数据处理、算法实现）→ 选择 Code Executor
# - 业务操作型任务（如API调用、数据库操作、文件处理）→ 选择 Tool Executor
#
# 要运行此代码，你需要：
# 1. 安装 AutoGen: pip install pyautogen
# 2. 配置环境变量 OPENAI_API_KEY, OPENAI_MODEL 等

# ============================================================
# AutoGen 导入说明
# ============================================================
# 以下导入语句仅在配置好真实环境后使用：
# ```python
# try:
#     import os
#     from autogen import ConversableAgent, UserProxyAgent
#     # 如需使用真实API，取消下面的注释并确保.env配置正确
#     # load_env()  # 加载环境变量
# except ImportError as e:
#     print(f"AutoGen 未安装或导入失败: {e}")
#     print("演示模式：代码逻辑仍可正常展示")
# ```
#
# 本文件中的代码为模拟演示逻辑，展示了 AutoGen 的核心机制
# 实际运行时需要配置真实的 API 密钥和环境
# ============================================================

import os
import json
from pathlib import Path
from typing import Literal, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================
# 第一部分：环境配置加载
# ============================================================

def load_env(env_path: str = ".env") -> None:
    """
    从 .env 文件加载环境变量

    Args:
        env_path: .env 文件路径，默认为当前目录下的 .env
    """
    path = Path(env_path)
    if not path.exists():
        print(f"警告：未找到 {env_path} 文件，请确保环境变量已正确设置")
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_required_env(name: str) -> str:
    """
    获取必需的环境变量，如果不存在则抛出异常

    Args:
        name: 环境变量名称

    Returns:
        环境变量的值

    Raises:
        RuntimeError: 当环境变量未设置时
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少必需的环境变量: {name}，请在 .env 文件中配置")
    return value


# ============================================================
# 第二部分：任务类型定义与执行器决策
# ============================================================

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


@dataclass
class ExecutorDecision:
    """
    执行器决策结果数据类

    Attributes:
        recommended_executor: 推荐的执行器类型 ('code_executor' 或 'tool_executor')
        task_type: 识别的任务类型
        confidence: 决策置信度 (0.0 ~ 1.0)
        reasoning: 决策理由说明
        fallback_executor: 备用执行器（当主执行器失败时使用）
    """
    recommended_executor: Literal["code_executor", "tool_executor"]
    task_type: TaskType
    confidence: float
    reasoning: str
    fallback_executor: Optional[Literal["code_executor", "tool_executor"]] = None


def analyze_task_type(task_description: str, task_code: Optional[str] = None) -> TaskType:
    """
    分析任务类型

    根据任务描述和可选的代码内容，判断任务属于哪种类型

    Args:
        task_description: 任务的文字描述
        task_code: 可选的任务代码示例

    Returns:
        TaskType: 识别出的任务类型
    """
    # 计算密集型任务的关键词
    calc_keywords = [
        "计算", "算法", "数学", "统计", "分析", "处理",
        "数据", "矩阵", "数值", "优化", "回归", "聚类",
        "排序", "搜索", "加密", "压缩", "编码", "解码"
    ]

    # 业务操作型任务的关键词
    biz_keywords = [
        "调用", "请求", "获取", "发送", "上传", "下载",
        "保存", "读取", "查询", "插入", "更新", "删除",
        "数据库", "API", "HTTP", "文件", "邮件", "短信"
    ]

    task_text = task_description.lower()

    # 如果提供了代码，进一步分析
    if task_code:
        # 检查代码中是否有循环、计算操作
        has_calculation = any(k in task_code for k in ["for ", "while ", "=", "*", "/", "+", "-", "**", "//"])
        has_io_operation = any(k in task_code for k in ["open(", "requests", "http", "cursor", "session"])

        if has_calculation and not has_io_operation:
            return TaskType.CALCULATION
        if has_io_operation and not has_calculation:
            return TaskType.BUSINESS_OPERATION
        if has_calculation and has_io_operation:
            return TaskType.HYBRID

    # 基于关键词判断
    calc_score = sum(1 for k in calc_keywords if k in task_text)
    biz_score = sum(1 for k in biz_keywords if k in task_text)

    if calc_score > biz_score:
        return TaskType.CALCULATION
    elif biz_score > calc_score:
        return TaskType.BUSINESS_OPERATION
    else:
        return TaskType.HYBRID


def make_executor_decision(task_description: str, task_code: Optional[str] = None) -> ExecutorDecision:
    """
    执行器决策框架 - 根据任务类型推荐合适的执行器

    决策逻辑：
    1. 计算密集型任务 → Code Executor（代码执行器）
    2. 业务操作型任务 → Tool Executor（工具执行器）
    3. 混合型任务 → 优先 Tool Executor，fallback 到 Code Executor

    Args:
        task_description: 任务的文字描述
        task_code: 可选的任务代码示例

    Returns:
        ExecutorDecision: 包含决策结果和建议的数据类
    """
    task_type = analyze_task_type(task_description, task_code)

    if task_type == TaskType.CALCULATION:
        return ExecutorDecision(
            recommended_executor="code_executor",
            task_type=task_type,
            confidence=0.9,
            reasoning="任务涉及计算密集型操作（如数学计算、数据处理、算法执行），"
                     "Code Executor 可以直接执行代码并返回计算结果，效率更高",
            fallback_executor="tool_executor"
        )
    elif task_type == TaskType.BUSINESS_OPERATION:
        return ExecutorDecision(
            recommended_executor="tool_executor",
            task_type=task_type,
            confidence=0.85,
            reasoning="任务涉及业务操作（如API调用、数据库操作），"
                     "Tool Executor 提供更好的错误处理和状态管理",
            fallback_executor="code_executor"
        )
    else:  # HYBRID
        return ExecutorDecision(
            recommended_executor="tool_executor",
            task_type=task_type,
            confidence=0.7,
            reasoning="任务为混合型，包含计算和业务操作两部分。"
                     "优先使用 Tool Executor 处理业务操作，计算部分通过工具封装实现。"
                     "当 Tool Executor 不支持时，fallback 到 Code Executor。",
            fallback_executor="code_executor"
        )


def print_decision_report(decision: ExecutorDecision) -> None:
    """
    打印决策报告

    Args:
        decision: 执行器决策结果
    """
    print("\n" + "=" * 60)
    print("执行器决策报告")
    print("=" * 60)
    print(f"推荐执行器: {decision.recommended_executor}")
    print(f"任务类型: {decision.task_type.value}")
    print(f"置信度: {decision.confidence:.0%}")
    print(f"决策理由: {decision.reasoning}")
    if decision.fallback_executor:
        print(f"备用执行器: {decision.fallback_executor}")
    print("=" * 60 + "\n")


# ============================================================
# 第三部分：Code Executor 配置
# ============================================================

def create_code_executor(
    timeout: int = 30,
    max_compute_units: Optional[int] = None,
    work_dir: Optional[str] = None
):
    """
    创建 Code Executor 配置

    Code Executor 特点：
    - 直接执行 Python/JS 代码
    - 适合计算密集型任务
    - 支持代码执行超时控制
    - 可以限制资源使用

    Args:
        timeout: 代码执行超时时间（秒）
        max_compute_units: 最大计算单元限制
        work_dir: 工作目录，代码将在此目录下执行

    Returns:
        dict: Code Executor 配置字典
    """
    config = {
        "executor_type": "code_executor",
        "timeout": timeout,
        "work_dir": work_dir or "./code_execution",
    }

    if max_compute_units:
        config["max_compute_units"] = max_compute_units

    return config


def create_tool_executor(
    tools: list,
    default_timeout: int = 60,
    retry_count: int = 3
):
    """
    创建 Tool Executor 配置

    Tool Executor 特点：
    - 通过函数调用执行预定义工具
    - 适合业务操作型任务
    - 内置重试机制和错误处理
    - 支持工具调用日志

    Args:
        tools: 工具函数列表
        default_timeout: 默认超时时间（秒）
        retry_count: 失败重试次数

    Returns:
        dict: Tool Executor 配置字典
    """
    return {
        "executor_type": "tool_executor",
        "tools": [t.__name__ if callable(t) else t for t in tools],
        "default_timeout": default_timeout,
        "retry_count": retry_count,
    }


# ============================================================
# 第四部分：last_n_messages='auto' 动态回溯机制
# ============================================================

class DynamicContextWindow:
    """
    动态上下文窗口管理器

    last_n_messages='auto' 机制解析：
    - AutoGen 根据当前上下文窗口大小（模型支持的 max token）
    - 动态调整回溯的消息数量，确保重要信息不被截断
    - 当上下文窗口接近满时，自动减少回溯消息数
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        avg_message_tokens: int = 150
    ):
        """
        初始化动态上下文窗口

        Args:
            model_name: 模型名称
            max_tokens: 模型支持的最大 token 数
            avg_message_tokens: 平均每条消息的 token 数（估算）
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.avg_message_tokens = avg_message_tokens
        # 保留一定空间给系统提示和响应
        self.reserved_tokens = max_tokens // 4

    def calculate_auto_last_n(self, current_messages_count: int) -> int:
        """
        计算 auto 模式下的 last_n_messages 值

        Args:
            current_messages_count: 当前消息总数

        Returns:
            int: 应该回溯的消息数量
        """
        # 计算可用 token 数
        available_tokens = self.max_tokens - self.reserved_tokens

        # 计算可用空间能容纳的消息数
        max_messages = available_tokens // self.avg_message_tokens

        # 取较小值，确保不超过实际消息数
        return min(current_messages_count, max_messages)

    def get_context_status(self, current_messages_count: int) -> dict:
        """
        获取当前上下文状态

        Args:
            current_messages_count: 当前消息总数

        Returns:
            dict: 上下文状态信息
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


# ============================================================
# 第五部分：Code Executor Fallback 策略
# ============================================================

class ExecutorFallbackManager:
    """
    执行器失败时的 Fallback 策略管理器

    当 Code Executor 执行失败时的处理策略：
    1. 语法错误 → 尝试修复代码后重试
    2. 超时 → 减少计算量或分段执行
    3. 运行时错误 → 记录错误信息，降级到 Tool Executor
    4. 资源限制 → 减少内存/CPU 使用
    """

    # 错误类型到处理策略的映射
    ERROR_STRATEGIES = {
        "SyntaxError": "fix_and_retry",        # 语法错误：修复后重试
        "TimeoutError": "reduce_and_retry",     # 超时：减少计算量
        "RuntimeError": "fallback_to_tool",    # 运行时错误：降级到工具执行器
        "MemoryError": "reduce_memory",        # 内存错误：减少内存使用
        "ImportError": "install_and_retry",     # 导入错误：安装依赖后重试
    }

    def __init__(self):
        self.error_log = []

    def handle_execution_failure(
        self,
        error: Exception,
        original_code: str,
        executor_type: str
    ) -> dict:
        """
        处理执行失败

        Args:
            error: 捕获的异常
            original_code: 原始代码
            executor_type: 当前执行器类型

        Returns:
            dict: 包含处理结果和建议的字典
        """
        error_type = type(error).__name__
        error_message = str(error)

        # 记录错误
        self.error_log.append({
            "error_type": error_type,
            "message": error_message,
            "executor": executor_type,
            "code": original_code
        })

        # 获取处理策略
        strategy = self.ERROR_STRATEGIES.get(error_type, "log_and_fallback")

        result = {
            "error_type": error_type,
            "error_message": error_message,
            "strategy": strategy,
            "fallback_executor": None,
            "recovery_action": None
        }

        # 根据策略生成恢复建议
        if strategy == "fix_and_retry":
            result["recovery_action"] = self._generate_code_fix_suggestion(original_code, error_message)
            result["fallback_executor"] = "code_executor"

        elif strategy == "reduce_and_retry":
            result["recovery_action"] = "代码执行超时，建议减少计算量或分段执行"
            result["fallback_executor"] = "code_executor"

        elif strategy == "fallback_to_tool":
            result["recovery_action"] = "代码执行失败，降级到 Tool Executor 实现相同功能"
            result["fallback_executor"] = "tool_executor"

        elif strategy == "reduce_memory":
            result["recovery_action"] = "内存不足，建议减少数据规模或优化算法"
            result["fallback_executor"] = "tool_executor"

        elif strategy == "install_and_retry":
            result["recovery_action"] = f"缺少依赖库: {self._extract_missing_module(error_message)}"
            result["fallback_executor"] = "code_executor"

        else:  # log_and_fallback
            result["recovery_action"] = "未知错误，记录并降级到 Tool Executor"
            result["fallback_executor"] = "tool_executor"

        return result

    def _generate_code_fix_suggestion(self, code: str, error_message: str) -> str:
        """
        生成代码修复建议（简单的启发式方法）

        Args:
            code: 原始代码
            error_message: 错误信息

        Returns:
            str: 修复建议
        """
        # 这里可以集成更复杂的代码修复逻辑
        return "代码存在语法错误，请检查并修复后重试"

    def _extract_missing_module(self, error_message: str) -> str:
        """
        从错误信息中提取缺失的模块名

        Args:
            error_message: 错误信息

        Returns:
            str: 缺失的模块名
        """
        if "ModuleNotFoundError" in error_message or "ImportError" in error_message:
            parts = error_message.split("'")
            if len(parts) >= 2:
                return parts[1]
        return "未知模块"


# ============================================================
# 第六部分：演示函数
# ============================================================

def demo_task_analysis():
    """
    演示任务分析功能
    """
    print("\n" + "=" * 60)
    print("演示1：任务类型分析")
    print("=" * 60)

    tasks = [
        ("计算前1000个斐波那契数列的和", None),
        ("调用API获取用户信息并存入数据库", "requests.get('https://api.example.com/user')"),
        ("对数据集进行回归分析并预测", "from sklearn import linear_model\nmodel.fit(X, y)"),
    ]

    for desc, code in tasks:
        print(f"\n任务: {desc}")
        task_type = analyze_task_type(desc, code)
        print(f"识别类型: {task_type.value}")

        decision = make_executor_decision(desc, code)
        print_decision_report(decision)


def demo_context_window():
    """
    演示动态上下文窗口
    """
    print("\n" + "=" * 60)
    print("演示2：last_n_messages='auto' 动态回溯")
    print("=" * 60)

    # 模拟不同模型的上下文窗口
    models = [
        ("gpt-4", 8192),
        ("gpt-3.5-turbo", 4096),
        ("claude-3", 200000),
    ]

    for model_name, max_tokens in models:
        window = DynamicContextWindow(model_name, max_tokens)

        for msg_count in [5, 20, 50, 100]:
            status = window.get_context_status(msg_count)
            print(f"\n模型: {status['model']} (max_tokens={status['max_tokens']})")
            print(f"  消息数: {status['current_messages']}")
            print(f"  推荐 last_n: {status['recommended_last_n']}")
            print(f"  使用率: {status['usage_ratio']}")
            print(f"  状态: {status['status']}")


def demo_fallback_strategy():
    """
    演示 Fallback 策略
    """
    print("\n" + "=" * 60)
    print("演示3：Code Executor Fallback 策略")
    print("=" * 60)

    manager = ExecutorFallbackManager()

    # 模拟不同的错误场景
    errors = [
        (SyntaxError("invalid syntax"), "print('hello"),
        (TimeoutError("execution timeout"), "for i in range(10**8): pass"),
        (RuntimeError("division by zero"), "1/0"),
    ]

    for error, code in errors:
        print(f"\n错误类型: {type(error).__name__}")
        result = manager.handle_execution_failure(error, code, "code_executor")
        print(f"  处理策略: {result['strategy']}")
        print(f"  恢复操作: {result['recovery_action']}")
        print(f"  Fallback: {result['fallback_executor']}")


def main():
    """
    主函数：运行决策框架演示
    """
    print("=" * 60)
    print("第8节 Code Executor vs Tool Executor 决策框架")
    print("=" * 60)

    # 加载环境变量
    load_env()

    # 演示任务分析
    demo_task_analysis()

    # 演示动态上下文窗口
    demo_context_window()

    # 演示 Fallback 策略
    demo_fallback_strategy()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()