# llm_call_vs_exec.py
# 第7节 Function Calling与register_function用法
# 演示 register_for_llm_call 与 register_for_exec 的区别
#
# 本文件展示两种函数注册方式的对比：
# 1. register_for_llm_call - 仅让 LLM 知道有这个工具
# 2. register_for_exec - 完整注册执行链（LLM 知道 + Agent 可执行）
# 3. 两种方式的使用场景与选择决策
#
# 要运行此代码，你需要：
# 1. 安装 AutoGen: pip install pyautogen
# 2. 配置环境变量 OPENAI_API_KEY

# ============================================================
# AutoGen 导入说明
# ============================================================
# 以下导入语句仅在配置好真实环境后使用：
# ```python
# try:
#     from autogen import ConversableAgent
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
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

# ============================================================
# 第一部分：环境配置与辅助函数
# ============================================================

def load_env(env_path: str = ".env") -> None:
    """
    从 .env 文件加载环境变量

    Args:
        env_path: .env 文件路径
    """
    path = Path(env_path)
    if not path.exists():
        print(f"警告：未找到 {env_path} 文件")
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# ============================================================
# 第二部分：两种注册方式的本质区别
# ============================================================

@dataclass
class LLMCallVsExecDemo:
    """
    register_for_llm_call 与 register_for_exec 对比演示

    核心概念：
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
    │                    register_function (= register_for_llm_call   │
    │                               + register_for_exec)             │
    │  作用：同时完成上述两件事                                        │
    │  效果：LLM 知道工具存在 + Agent 可执行                          │
    └─────────────────────────────────────────────────────────────────┘

    重要区分：
    1. LLM 知道的工具 ≠ Agent 能执行的工具
    2. register_for_llm_call 只影响 LLM 的工具列表
    3. register_for_exec 只影响 Agent 的执行能力
    """

    func: Callable = field(default=None)
    name: Optional[str] = None


# ============================================================
# 第三部分：执行器架构解析
# ============================================================

class ExecutorArchitecture:
    """
    解析 AutoGen 的执行器架构

    AutoGen 中存在两套独立的执行机制：

    1. Tool Executor（工具执行器）
       - 通过 register_function / register_for_llm_call / register_for_exec 注册
       - 执行 Python 函数
       - 受 function_map 参数控制

    2. Code Executor（代码执行器）
       - 通过 CodeExecutor 类实现
       - 执行代码字符串（如 execute_code）
       - 受 code_execution_config 参数控制

    两者相互独立，可以单独使用或组合使用。
    """

    @staticmethod
    def explain_executor_difference():
        """
        详细解释两种执行器的区别
        """
        print("\n" + "=" * 60)
        print("Tool Executor vs Code Executor 架构解析")
        print("=" * 60)

        print("""
        ┌─────────────────────────────────────────────────────────────┐
        │                      Tool Executor                          │
        ├─────────────────────────────────────────────────────────────┤
        │  用途：执行已定义的 Python 函数（业务逻辑封装）             │
        │  注册：register_function() / register_for_llm_call()       │
        │  参数：通过 function_map 传入                              │
        │  执行：LLM 生成 function_call → Agent 路由到 executor      │
        │  示例：天气查询、订单处理、数据转换                        │
        ├─────────────────────────────────────────────────────────────┤
        │  流程：                                                    │
        │    LLM 决策调用工具                                         │
        │       ↓                                                     │
        │    Agent 收到 function_call {name, arguments}             │
        │       ↓                                                     │
        │    executor.execute_function(name, arguments)              │
        │       ↓                                                     │
        │    返回执行结果给 LLM                                       │
        └─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │                     Code Executor                            │
        ├─────────────────────────────────────────────────────────────┤
        │  用途：动态执行代码字符串（支持代码生成与执行）             │
        │  配置：CodeExecutor 类 + code_execution_config               │
        │  执行：LLM 生成代码 → Agent 路由到 executor → 执行 → 返回   │
        │  示例：数据计算、文件处理、动态编程                         │
        ├─────────────────────────────────────────────────────────────┤
        │  流程：                                                    │
        │    LLM 生成代码字符串                                       │
        │       ↓                                                     │
        │    Agent 发送代码给 CodeExecutor                            │
        │       ↓                                                     │
        │    executor.execute_code(code)                              │
        │       ↓                                                     │
        │    返回执行结果（stdout/stderr）给 LLM                     │
        └─────────────────────────────────────────────────────────────┘
        """)


# ============================================================
# 第四部分：register_for_llm_call 详解
# ============================================================

class RegisterForLLMCallDemo:
    """
    register_for_llm_call 用法演示

    方法签名：
    ```python
    def register_for_llm_call(
        name: str,           # 注册到 LLM 的工具名称
        func: Callable,      # 实际执行的函数
    ):
    ```

    作用：
    - 仅将函数注入 LLM 的 tool_calls 列表
    - LLM 会知道有这个工具，并可能生成 function_call
    - 但 Agent 收到 call 后需要配合 register_for_exec 才能执行

    典型使用场景：
    1. 需要精细控制哪些工具对 LLM 可见
    2. 工具执行需要特殊处理（如异步执行、批量执行）
    3. 需要在执行前做额外验证或转换
    """

    @staticmethod
    def demo_register_for_llm_call():
        """
        演示 register_for_llm_call 的效果
        """
        print("\n" + "-" * 60)
        print("register_for_llm_call 演示")
        print("-" * 60)

        # 模拟 LLM 可用的工具列表
        llm_available_tools = []

        # 模拟 Agent 的可执行函数映射
        agent_executable_functions = {}

        # 业务函数
        def sensitive_api_call(user_id: str, action: str) -> Dict[str, Any]:
            """
            敏感操作 API（需要特殊权限检查）

            Args:
                user_id: 用户ID
                action: 操作类型

            Returns:
                操作结果
            """
            # 模拟权限检查
            if action == "delete" and not user_id.startswith("admin_"):
                return {"error": "权限不足：删除操作需要管理员权限"}
            return {"success": True, "user_id": user_id, "action": action}

        # 场景1：只注册给 LLM（LLM 知道，但不直接执行）
        tool_name = "sensitive_operation"

        # 模拟 register_for_llm_call
        # 这会将函数添加到 LLM 的 tool_calls 列表
        llm_available_tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": sensitive_api_call.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "action": {"type": "string"}
                    },
                    "required": ["user_id", "action"]
                }
            }
        })

        print(f"[注册到 LLM] 工具名称: {tool_name}")
        print(f"  LLM 现在知道这个工具，可以生成 function_call")

        # 注意：这里没有添加到 agent_executable_functions
        # 所以即使 LLM 生成了 call，Agent 也无法执行

        print(f"\n  LLM 可用工具列表长度: {len(llm_available_tools)}")
        print(f"  Agent 可执行函数数量: {len(agent_executable_functions)}")
        print("  → Agent 无法执行！LLM 会收到执行失败的通知")


# ============================================================
# 第五部分：register_for_exec 详解
# ============================================================

class RegisterForExecDemo:
    """
    register_for_exec 用法演示

    方法签名：
    ```python
    def register_for_exec(
        name: str,           # 执行时的函数名称
        func: Callable,      # 实际执行的函数
    ):
    ```

    作用：
    - 仅将函数添加到 Agent 的可执行列表
    - Agent 收到 function_call 后可以执行
    - 但 LLM 不知道有这个工具存在

    重要：单独使用 register_for_exec 是无意义的，
    因为 LLM 不会生成对这个工具的调用。

    典型使用场景：
    1. 作为 register_function 的内部实现
    2. 需要先注册给 LLM，再选择性执行
    """

    @staticmethod
    def demo_register_for_exec():
        """
        演示 register_for_exec 的效果
        """
        print("\n" + "-" * 60)
        print("register_for_exec 演示")
        print("-" * 60)

        # 模拟 Agent 的可执行函数映射
        agent_executable_functions = {}

        # 业务函数
        def internal_helper(data: str) -> str:
            """内部辅助函数（不需要 LLM 知道）"""
            return f"[已处理] {data}"

        # 只注册给 Agent 执行（LLM 不知道）
        tool_name = "internal_process"

        agent_executable_functions[tool_name] = internal_helper

        print(f"[注册到 Agent] 工具名称: {tool_name}")
        print(f"  Agent 可以执行这个函数")

        print(f"\n  LLM 可用工具列表长度: 0（LLM 不知道有这个工具）")
        print(f"  Agent 可执行函数数量: {len(agent_executable_functions)}")
        print("  → 问题：LLM 永远不会调用它，因为不知道存在")


# ============================================================
# 第六部分：组合使用场景
# ============================================================

class CombinedUsageDemo:
    """
    演示 register_for_llm_call + register_for_exec 组合使用

    某些高级场景下，可能需要：
    1. LLM 知道工具 A，但通过不同的方式执行
    2. LLM 知道工具 B，但需要执行前验证
    3. 多个工具注册，但选择性暴露给 LLM
    """

    @staticmethod
    def demo_combined_usage():
        """
        演示组合使用的典型场景
        """
        print("\n" + "=" * 60)
        print("组合使用场景演示")
        print("=" * 60)

        # 模拟 LLM 可用工具和 Agent 可执行函数
        llm_tools = []
        agent_executors = {}

        # 业务函数定义
        def direct_call(amount: float) -> Dict[str, Any]:
            """直接调用外部 API"""
            return {"status": "success", "amount": amount}

        def validated_call(user_id: str, action: str) -> Dict[str, Any]:
            """带验证的调用"""
            # 执行前的验证逻辑
            if len(user_id) < 5:
                return {"error": "user_id 无效"}
            return {"status": "success", "user_id": user_id, "action": action}

        def batch_process(items: List[str]) -> Dict[str, Any]:
            """批量处理（需要特殊执行器）"""
            return {"processed": len(items), "items": items}

        # 场景1：普通工具 - 使用 register_function 一次性完成
        print("\n场景1：普通工具")
        print("-" * 40)
        print("使用 register_function() 同时注册到 LLM 和 Agent")

        # 模拟 register_function 效果
        llm_tools.append({"name": "direct_call", "function": direct_call})
        agent_executors["direct_call"] = direct_call
        print("  ✓ LLM 知道工具存在")
        print("  ✓ Agent 可以执行")

        # 场景2：需要验证的工具 - 分离注册
        print("\n场景2：需要执行前验证")
        print("-" * 40)
        print("使用 register_for_llm_call + 自定义执行包装")

        # 注册到 LLM
        llm_tools.append({"name": "validated_call", "function": validated_call})

        # 注册到 Agent，但用包装函数
        def validated_wrapper(user_id: str, action: str) -> Dict:
            """执行前包装验证逻辑"""
            print(f"    [执行前验证] user_id={user_id}, action={action}")
            return validated_call(user_id, action)

        agent_executors["validated_call"] = validated_wrapper
        print("  ✓ LLM 知道工具存在")
        print("  ✓ Agent 执行时会先验证")
        print("  ✓ 执行器收到调用，打印日志")

        # 场景3：批量工具 - LLM 知道但执行不同
        print("\n场景3：批量处理（执行器不同）")
        print("-" * 40)
        print("LLM 看到的是简单接口，实际执行是批量逻辑")

        # LLM 看到的接口
        llm_tools.append({
            "name": "batch_item",
            "function": lambda items: {"status": "ok"}
        })

        # Agent 实际执行的是批量版本
        agent_executors["batch_item"] = batch_process
        print("  ✓ LLM 以为它在调用单个 item")
        print("  ✓ 实际执行是批量处理")


# ============================================================
# 第七部分：决策框架
# ============================================================

class DecisionFramework:
    """
    Tool Call 与 Code Executor 的选择决策框架

    根据任务特性选择合适的执行方式：

    ┌─────────────────────────────────────────────────────────────┐
    │                    决策流程                                  │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   任务类型判断 ──────────────────┐                          │
    │                                │                          │
    │          ↓                     ↓                          │
    │   ┌─────────────┐       ┌─────────────┐                   │
    │   │  业务操作型  │       │  计算密集型  │                   │
    │   │  (查库/调用) │       │  (数据处理) │                   │
    │   └─────────────┘       └─────────────┘                   │
    │          ↓                     ↓                          │
    │   ┌─────────────┐       ┌─────────────┐                   │
    │   │ Tool Call   │       │ Code Exec   │                   │
    │   │ (register_  │       │ (动态代码   │                   │
    │   │  function)  │       │  执行)       │                   │
    │   └─────────────┘       └─────────────┘                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    决策维度：
    1. 确定性：业务操作结果可预期，代码执行结果不确定
    2. 安全性：业务操作可测试，代码执行需隔离
    3. 复杂性：简单调用选 Tool，复杂逻辑选 Code
    """

    @staticmethod
    def show_decision_tree():
        """
        展示完整的决策树
        """
        print("\n" + "=" * 60)
        print("Tool Call vs Code Executor 决策框架")
        print("=" * 60)

        decision_tree = """
        ┌─────────────────────────────────────────────────────────────────┐
        │                    开始决策                                      │
        └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  Q1: 这个操作需要 LLM 决定调用时机吗？                             │
        └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                   Yes                              No
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │  Q2: 是预定义的业务    │       │  自动执行，不暴露给   │
        │  函数还是动态逻辑？    │       │  LLM（用 register_    │
        └───────────────────────┘       │  for_exec 或内部使用）│
                    │                     └───────────────────────┘
        ┌───────────┴───────────┐                   │
        ▼                       ▼                   ▼
       业务函数              动态代码          结束（内部处理）
        │                       │
        ▼                       ▼
    ┌─────────────┐       ┌─────────────┐
    │ Tool Call   │       │ Code Executor
    │ register_   │       │ (代码字符串 │
    │ function()  │       │  执行)      │
    └─────────────┘       └─────────────┘

        ┌─────────────────────────────────────────────────────────────────┐
        │                      决策维度说明                                 │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │  选择 Tool Call (register_function) 当：                       │
        │  ✓ 业务逻辑固定，结果可预期                                      │
        │  ✓ 需要参数校验和类型转换                                        │
        │  ✓ 涉及外部 API 调用                                            │
        │  ✓ 需要审计日志和访问控制                                       │
        │                                                                 │
        │  选择 Code Executor 当：                                        │
        │ ✓ 需要动态生成代码                                               │
        │ ✓ 数据处理逻辑复杂（PD.read_csv 等）                            │
        │ ✓ 需要实时编译执行                                              │
        │ ✓ 执行环境需要隔离（Docker）                                    │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
        """
        print(decision_tree)

    @staticmethod
    def show_use_case_table():
        """
        展示典型用例对照表
        """
        print("\n" + "=" * 60)
        print("典型用例对照表")
        print("=" * 60)

        use_cases = """
        ┌────────────────────┬────────────────────────┬────────────────────────┐
        │       场景         │     推荐方式            │        说明            │
        ├────────────────────┼────────────────────────┼────────────────────────┤
        │ 天气预报查询       │ Tool Call              │ 业务函数，参数固定      │
        │ 数据库 CRUD 操作   │ Tool Call              │ SQL 执行器封装         │
        │ 文件格式转换       │ Tool Call              │ 调用 ffmpeg/PIL 等     │
        │ 数据分析计算       │ Code Executor          │ 动态生成 pandas 代码   │
        │ 正则提取处理       │ Code Executor          │ 代码字符串执行         │
        │ API 批量调用       │ Tool Call              │ 业务逻辑封装           │
        │ 动态页面生成       │ Code Executor          │ HTML/JS 动态生成       │
        │ 复杂业务工作流     │ Tool Call              │ 状态机封装             │
        └────────────────────┴────────────────────────┴────────────────────────┘
        """
        print(use_cases)


# ============================================================
# 第八部分：实际代码对比
# ============================================================

class CodeComparisonDemo:
    """
    展示三种注册方式的实际代码对比
    """

    @staticmethod
    def show_code_comparison():
        """
        展示三种注册方式的代码对比
        """
        print("\n" + "=" * 60)
        print("三种注册方式代码对比")
        print("=" * 60)

        comparisons = [
            {
                "method": "register_function()",
                "desc": "一次性完成 LLM 知道 + Agent 可执行",
                "code": '''
# 直接注册，LLM 和 Agent 都能使用
agent.register_function(
    func=query_order,
    name="query_order",
    description="查询订单信息"
)
'''
            },
            {
                "method": "register_for_llm_call()",
                "desc": "仅让 LLM 知道，Agent 不一定能执行",
                "code": '''
# 只注入 LLM 的 tool_calls
# Agent 收到 call 后需要其他机制执行
agent.register_for_llm_call(
    name="query_order",
    func=query_order  # 这里只是记录，实际执行可能走其他路径
)
'''
            },
            {
                "method": "register_for_exec()",
                "desc": "仅让 Agent 可执行，LLM 不知道（单独使用无意义）",
                "code": '''
# 单独使用无意义，因为 LLM 不会生成调用
# 通常配合 register_for_llm_call 使用
agent.register_for_exec(
    name="query_order",
    func=query_order
)
'''
            },
            {
                "method": "组合使用",
                "desc": "精细控制注册流程",
                "code": '''
# 先注册给 LLM
agent.register_for_llm_call(name="query_order", func=query_order)

# 再注册给 Agent，但包装执行逻辑
def wrapped_query_order(order_id, user_id):
    print(f"[执行] 查询订单: {order_id}")
    return query_order(order_id, user_id)

agent.register_for_exec(name="query_order", func=wrapped_query_order)
'''
            }
        ]

        for i, comp in enumerate(comparisons, 1):
            print(f"\n{'─' * 60}")
            print(f"[{i}] {comp['method']}")
            print(f"    描述：{comp['desc']}")
            print(f"    代码：{comp['code']}")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 加载环境变量
    load_env()

    print("\n" + "#" * 60)
    print("# 第一部分：两种注册方式的本质区别")
    print("#" * 60)

    demo = LLMCallVsExecDemo()
    print(f"核心概念：{demo}")

    print("\n" + "#" * 60)
    print("# 第二部分：执行器架构解析")
    print("#" * 60)

    ExecutorArchitecture.explain_executor_difference()

    print("\n" + "#" * 60)
    print("# 第三部分：register_for_llm_call 详解")
    print("#" * 60)

    RegisterForLLMCallDemo.demo_register_for_llm_call()

    print("\n" + "#" * 60)
    print("# 第四部分：register_for_exec 详解")
    print("#" * 60)

    RegisterForExecDemo.demo_register_for_exec()

    print("\n" + "#" * 60)
    print("# 第五部分：组合使用场景")
    print("#" * 60)

    CombinedUsageDemo.demo_combined_usage()

    print("\n" + "#" * 60)
    print("# 第六部分：决策框架")
    print("#" * 60)

    DecisionFramework.show_decision_tree()
    DecisionFramework.show_use_case_table()

    print("\n" + "#" * 60)
    print("# 第七部分：代码对比")
    print("#" * 60)

    CodeComparisonDemo.show_code_comparison()

    print("\n" + "=" * 60)
    print("llm_call_vs_exec 用法演示结束")
    print("=" * 60)
    print("""
    学习要点总结：
    1. register_for_llm_call：仅让 LLM 知道工具存在
    2. register_for_exec：仅让 Agent 可执行（单独使用无意义）
    3. register_function：同时完成上述两件事（等于组合使用）
    4. 选择决策：业务操作选 Tool Call，动态代码选 Code Executor

    下一步：
    - 对比 lesson_06 的 Code Executor 配置
    - 了解 function_map 与 register_function 的关系
    """)