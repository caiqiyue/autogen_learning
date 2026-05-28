# mixed_executor.py
# 第8节 Code Executor vs Tool Executor决策框架 - 混合使用场景
#
# 本文件演示如何在实际应用中混合使用 Code Executor 和 Tool Executor
#
# 混合使用场景：
# 1. 复杂任务分解：计算部分用 Code Executor，业务部分用 Tool Executor
# 2. 级联调用：Tool Executor 调用外部 API 获取数据，Code Executor 处理数据
# 3. Fallback 机制：主执行器失败时切换到备用执行器
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
#     from autogen import ConversableAgent, UserProxyAgent
#     from autogen.code_executor import CodeExecutor
#     from autogen.tool_executor import ToolExecutor
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
import time
from pathlib import Path
from typing import Optional, Any, Callable, Union
from dataclasses import dataclass, field
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
# 第二部分：Tool Executor 工具定义
# ============================================================

class ToolExecutor:
    """
    Tool Executor 实现类

    负责执行预定义的工具函数，适合业务操作型任务
    """

    def __init__(self, tools: dict, timeout: int = 60):
        """
        初始化 Tool Executor

        Args:
            tools: 工具函数字典，格式为 {tool_name: callable}
            timeout: 超时时间（秒）
        """
        self.tools = tools
        self.timeout = timeout
        self.execution_log = []

    def execute(self, tool_name: str, **kwargs) -> dict:
        """
        执行工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            dict: 包含执行结果的字典
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"工具 '{tool_name}' 不存在",
                "available_tools": list(self.tools.keys())
            }

        start_time = time.time()
        try:
            result = self.tools[tool_name](**kwargs)
            elapsed = time.time() - start_time

            execution_record = {
                "tool": tool_name,
                "params": kwargs,
                "success": True,
                "result": result,
                "elapsed": elapsed
            }
            self.execution_log.append(execution_record)

            return {
                "success": True,
                "result": result,
                "elapsed": elapsed
            }

        except Exception as e:
            elapsed = time.time() - start_time
            execution_record = {
                "tool": tool_name,
                "params": kwargs,
                "success": False,
                "error": str(e),
                "elapsed": elapsed
            }
            self.execution_log.append(execution_record)

            return {
                "success": False,
                "error": str(e),
                "elapsed": elapsed
            }

    def get_available_tools(self) -> list:
        """
        获取可用工具列表

        Returns:
            list: 工具名称列表
        """
        return list(self.tools.keys())


# ============================================================
# 第三部分：Code Executor 实现
# ============================================================

class CodeExecutor:
    """
    Code Executor 实现类

    负责执行代码片段，适合计算密集型任务
    """

    def __init__(self, work_dir: str = "./code_execution", timeout: int = 30):
        """
        初始化 Code Executor

        Args:
            work_dir: 工作目录
            timeout: 超时时间（秒）
        """
        self.work_dir = Path(work_dir)
        self.timeout = timeout
        self.execution_log = []

        # 确保工作目录存在
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, language: str = "python") -> dict:
        """
        执行代码

        Args:
            code: 要执行的代码
            language: 代码语言（目前仅支持 python）

        Returns:
            dict: 包含执行结果的字典
        """
        if language != "python":
            return {
                "success": False,
                "error": f"不支持的编程语言: {language}"
            }

        start_time = time.time()
        try:
            # 使用 exec 执行代码，创建独立的命名空间
            local_namespace = {}
            exec(code, {}, local_namespace)

            # 尝试获取结果（如果有 result 变量）
            result = local_namespace.get("result", None)

            elapsed = time.time() - start_time

            execution_record = {
                "code": code,
                "language": language,
                "success": True,
                "result": result,
                "elapsed": elapsed
            }
            self.execution_log.append(execution_record)

            return {
                "success": True,
                "result": result,
                "elapsed": elapsed
            }

        except Exception as e:
            elapsed = time.time() - start_time

            execution_record = {
                "code": code,
                "language": language,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "elapsed": elapsed
            }
            self.execution_log.append(execution_record)

            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "elapsed": elapsed
            }

    def get_execution_history(self) -> list:
        """
        获取执行历史

        Returns:
            list: 执行记录列表
        """
        return self.execution_log


# ============================================================
# 第四部分：混合执行器编排器
# ============================================================

class MixedExecutorOrchestrator:
    """
    混合执行器编排器

    协调 Code Executor 和 Tool Executor 的混合使用

    使用场景：
    1. 数据获取（Tool）→ 数据处理（Code）→ 结果存储（Tool）
    2. API 调用（Tool）→ 数据分析（Code）→ 可视化（Code）
    3. 文件读取（Tool）→ 数据计算（Code）→ 数据库写入（Tool）
    """

    def __init__(self):
        self.code_executor = CodeExecutor()
        self.tool_executor = None  # 稍后初始化
        self.execution_pipeline = []
        self.fallback_enabled = True

    def set_tools(self, tools: dict):
        """
        设置工具

        Args:
            tools: 工具函数字典
        """
        self.tool_executor = ToolExecutor(tools)

    def enable_fallback(self, enabled: bool = True):
        """
        启用/禁用 Fallback 机制

        Args:
            enabled: 是否启用 fallback
        """
        self.fallback_enabled = enabled

    def execute_pipeline(self, pipeline: list) -> dict:
        """
        执行混合执行管道

        Args:
            pipeline: 执行管道，格式为
                [
                    {"type": "tool", "name": "get_stock_price", "params": {...}},
                    {"type": "code", "code": "result = data * 1.1"},
                    {"type": "tool", "name": "save_to_db", "params": {...}},
                ]

        Returns:
            dict: 管道执行结果
        """
        results = []
        final_result = None

        for i, step in enumerate(pipeline):
            step_type = step["type"]
            step_name = step.get("name", "code_block") if step_type == "tool" else f"code_{i}"

            print(f"\n[步骤 {i+1}] 执行 {step_type}: {step_name}")

            if step_type == "code":
                result = self.code_executor.execute(step["code"])
            elif step_type == "tool":
                result = self.tool_executor.execute(step["name"], **step.get("params", {}))
            else:
                result = {"success": False, "error": f"未知类型: {step_type}"}

            results.append({
                "step": i + 1,
                "type": step_type,
                "name": step_name,
                "result": result
            })

            if result["success"]:
                final_result = result.get("result")
                print(f"  ✓ 成功，结果: {final_result}")
            else:
                print(f"  ✗ 失败: {result.get('error')}")

                # Fallback 处理
                if self.fallback_enabled:
                    fallback_result = self._try_fallback(step, result)
                    if fallback_result:
                        results[-1]["fallback"] = fallback_result
                        final_result = fallback_result.get("result")
                        print(f"  → Fallback 成功: {final_result}")

        return {
            "success": all(r.get("result", {}).get("success", False) for r in results),
            "steps": results,
            "final_result": final_result
        }

    def _try_fallback(self, failed_step: dict, error_result: dict) -> Optional[dict]:
        """
        尝试 Fallback

        当执行失败时，尝试使用另一种执行器

        Args:
            failed_step: 失败的步骤信息
            error_result: 错误结果

        Returns:
            dict or None: Fallback 结果（如果有）
        """
        step_type = failed_step["type"]

        print(f"  → 尝试 Fallback ({'code' if step_type == 'tool' else 'tool'} executor)...")

        if step_type == "code":
            # Code Executor 失败，尝试用更简单的代码
            original_code = failed_step["code"]
            simplified_code = f"""
# 简化版代码
try:
{chr(10).join('    ' + line for line in original_code.splitlines())}
except Exception as e:
    result = f'Error: {{e}}'
"""
            return self.code_executor.execute(simplified_code)

        else:
            # Tool Executor 失败，尝试用代码模拟
            tool_name = failed_step["name"]
            print(f"  → 工具 '{tool_name}' 执行失败，尝试用 Code Executor 模拟...")

            # 简单的模拟逻辑
            return {
                "success": True,
                "result": f"[模拟] {tool_name} 执行成功（Tool Executor Fallback）",
                "fallback": True
            }


# ============================================================
# 第五部分：企业级工具插件开发规范
# ============================================================

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

        Args:
            name: 工具名称
            version: 版本号
            author: 作者
            description: 功能描述
            tags: 标签列表

        Returns:
            dict: 工具元数据
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

    @staticmethod
    def validate_tool_function(func: Callable) -> tuple:
        """
        验证工具函数是否符合规范

        Args:
            func: 工具函数

        Returns:
            tuple: (是否通过验证, 错误信息列表)
        """
        errors = []

        # 检查是否有文档字符串
        if not func.__doc__:
            errors.append("缺少文档字符串")

        # 检查是否有类型注解
        annotations = func.__annotations__
        if not annotations:
            errors.append("缺少类型注解")

        # 检查返回类型注解
        if 'return' not in annotations:
            errors.append("缺少返回类型注解")

        return (len(errors) == 0, errors)


def create_standard_tool(
    func: Callable,
    metadata: dict = None
) -> Callable:
    """
    创建符合企业级标准的工具装饰器

    Args:
        func: 工具函数
        metadata: 工具元数据

    Returns:
        Callable: 装饰后的工具函数
    """
    def wrapper(*args, **kwargs):
        # 添加执行前检查
        is_valid, errors = ToolPluginStandard.validate_tool_function(func)
        if not is_valid:
            return {
                "success": False,
                "error": f"工具验证失败: {', '.join(errors)}"
            }

        # 执行工具
        try:
            result = func(*args, **kwargs)
            return {
                "success": True,
                "result": result,
                "metadata": metadata
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"执行失败: {str(e)}",
                "metadata": metadata
            }

    # 保留原始函数信息
    wrapper._is_tool = True
    wrapper._metadata = metadata
    wrapper._original_func = func

    return wrapper


# ============================================================
# 第六部分：示例工具定义
# ============================================================

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
    time.sleep(0.1)  # 模拟网络延迟

    # 模拟数据
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


@create_standard_tool
def save_to_database(data: dict, table: str = "stock_data") -> dict:
    """
    保存数据到数据库

    Args:
        data: 要保存的数据
        table: 表名

    Returns:
        dict: 包含保存结果的字典
    """
    # 模拟数据库操作
    time.sleep(0.05)  # 模拟数据库延迟

    return {
        "success": True,
        "table": table,
        "rows_affected": 1,
        "data": data
    }


@create_standard_tool
def calculate_portfolio_value(positions: list, prices: dict) -> float:
    """
    计算投资组合价值

    Args:
        positions: 持仓列表，格式为 [{"symbol": "AAPL", "shares": 100}, ...]
        prices: 价格字典，格式为 {"AAPL": 175.43, ...}

    Returns:
        float: 投资组合总价值
    """
    total_value = 0.0
    for pos in positions:
        symbol = pos["symbol"]
        shares = pos["shares"]
        price = prices.get(symbol, 0)
        total_value += shares * price

    return total_value


# ============================================================
# 第七部分：演示函数
# ============================================================

def demo_mixed_execution():
    """
    演示混合执行场景
    """
    print("\n" + "=" * 60)
    print("演示1：混合执行器 - 数据获取→处理→存储")
    print("=" * 60)

    # 创建编排器
    orchestrator = MixedExecutorOrchestrator()

    # 设置工具
    tools = {
        "fetch_stock_price": fetch_stock_price,
        "save_to_database": save_to_database,
        "calculate_portfolio_value": calculate_portfolio_value
    }
    orchestrator.set_tools(tools)

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

    print("\n管道执行完成！")
    print(f"总体成功: {result['success']}")
    print(f"最终结果: {result['final_result']}")


def demo_tool_plugin_validation():
    """
    演示工具插件验证
    """
    print("\n" + "=" * 60)
    print("演示2：企业级工具插件开发规范验证")
    print("=" * 60)

    # 创建元数据
    metadata = ToolPluginStandard.create_tool_metadata(
        name="fetch_stock_price",
        version="1.0.0",
        author="Enterprise Team",
        description="获取股票当前价格",
        tags=["finance", "stock", "api"]
    )

    print("\n工具元数据:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    # 验证工具函数
    is_valid, errors = ToolPluginStandard.validate_tool_function(fetch_stock_price)

    print(f"\n工具验证结果: {'通过' if is_valid else '失败'}")
    if errors:
        print(f"错误列表: {errors}")


def demo_portfolio_calculation():
    """
    演示投资组合计算
    """
    print("\n" + "=" * 60)
    print("演示3：投资组合价值计算（完整流程）")
    print("=" * 60)

    # 创建编排器
    orchestrator = MixedExecutorOrchestrator()
    orchestrator.set_tools({
        "fetch_stock_price": fetch_stock_price,
        "calculate_portfolio_value": calculate_portfolio_value
    })

    # 定义持仓
    positions = [
        {"symbol": "AAPL", "shares": 100},
        {"symbol": "GOOGL", "shares": 50},
        {"symbol": "MSFT", "shares": 75}
    ]

    # 执行管道：获取多个股票价格 → 计算总价值
    pipeline = [
        # 获取所有股票价格
        {"type": "tool", "name": "fetch_stock_price", "params": {"symbol": "AAPL"}},
        {"type": "tool", "name": "fetch_stock_price", "params": {"symbol": "GOOGL"}},
        {"type": "tool", "name": "fetch_stock_price", "params": {"symbol": "MSFT"}},

        # 计算投资组合价值
        {
            "type": "code",
            "code": """
positions = [
    {"symbol": "AAPL", "shares": 100},
    {"symbol": "GOOGL", "shares": 50},
    {"symbol": "MSFT", "shares": 75}
]
prices = {"AAPL": 175.43, "GOOGL": 142.65, "MSFT": 378.91}

total = 0
for pos in positions:
    symbol = pos["symbol"]
    shares = pos["shares"]
    price = prices[symbol]
    total += shares * price

result = {'total_value': total, 'currency': 'USD', 'positions': len(positions)}
"""
        }
    ]

    result = orchestrator.execute_pipeline(pipeline)

    print("\n" + "=" * 60)
    print("投资组合计算完成！")
    print("=" * 60)
    print(f"最终结果: {result['final_result']}")


def demo_fallback_scenario():
    """
    演示 Fallback 场景
    """
    print("\n" + "=" * 60)
    print("演示4：Fallback 降级场景")
    print("=" * 60)

    orchestrator = MixedExecutorOrchestrator()
    orchestrator.set_tools({"some_tool": lambda x: x})
    orchestrator.enable_fallback(True)

    # 故意使用会失败的代码
    pipeline = [
        {
            "type": "code",
            "code": "this_will_cause_syntax_error"  # 故意写成会导致错误的代码
        }
    ]

    print("\n执行一个会失败的代码...")
    result = orchestrator.execute_pipeline(pipeline)

    print("\nFallback 执行后:")
    print(f"  步骤数: {len(result['steps'])}")
    if result['steps'][0].get('fallback'):
        print("  Fallback 成功执行！")


def main():
    """
    主函数：运行混合执行器演示
    """
    print("=" * 60)
    print("第8节 Code Executor vs Tool Executor - 混合使用场景")
    print("=" * 60)

    # 加载环境变量
    load_env()

    # 演示混合执行
    demo_mixed_execution()

    # 演示工具插件验证
    demo_tool_plugin_validation()

    # 演示投资组合计算
    demo_portfolio_calculation()

    # 演示 Fallback 场景
    demo_fallback_scenario()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()