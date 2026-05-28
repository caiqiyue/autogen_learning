# register_function_demo.py
# 第7节 Function Calling与register_function用法
# 演示 register_function 的核心用法与函数注册流程
#
# 本文件展示 register_function 的完整用法：
# 1. register_function 的基本注册流程
# 2. 函数签名到 LLM function calling 的映射过程
# 3. function_map 参数的用法
# 4. 实际业务工具注册示例
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
import re
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
# 第二部分：register_function 核心概念
# ============================================================

@dataclass
class RegisterFunctionDemo:
    """
    register_function 用法演示类

    register_function 是 AutoGen 中用于将 Python 函数注册为 Agent 工具的核心方法。
    它建立了从 Python 函数到 LLM Function Calling 的映射关系。

    方法签名：
    ```python
    def register_function(
        func: Callable,                    # 要注册的函数
        name: Optional[str] = None,        # 可选：显式指定函数名（默认使用函数本身的名字）
        description: Optional[str] = None, # 可选：描述函数用途（用于 LLM 理解何时调用）
        signature: Optional[str] = None,   # 可选：显式指定函数签名
    ):
    ```

    重要概念：
    1. 函数签名（signature）：Python 函数的参数类型注解，会转换为 LLM 能理解的工具描述
    2. 函数描述（description）：告诉 LLM 这个函数做什么，影响 LLM 的调用决策
    3. name：注册到 Agent 后的工具名称，默认使用函数原名
    """
    func: Callable = field(default=None)
    name: Optional[str] = None
    description: Optional[str] = None


# ============================================================
# 第三部分：函数签名到 Tool Call 的映射原理
# ============================================================

class SignatureMapper:
    """
    演示 Python 函数签名如何映射为 LLM Function Calling 格式

    LLM 需要的工具描述格式（OpenAI Function Calling schema）：
    ```json
    {
        "name": "函数名",
        "description": "函数描述",
        "parameters": {
            "type": "object",
            "properties": {
                "参数名": {
                    "type": "类型",
                    "description": "参数描述"
                }
            },
            "required": ["必填参数"]
        }
    }
    ```
    """

    @staticmethod
    def python_type_to_json_type(py_type: str) -> str:
        """
        将 Python 类型映射为 JSON Schema 类型

        Args:
            py_type: Python 类型字符串（如 "str", "int", "List[int]"）

        Returns:
            JSON Schema 类型字符串
        """
        # 基本类型映射表
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "List": "array",
            "Dict": "object",
        }

        # 处理泛型，如 List[int], Dict[str, Any]
        for py_type_key, json_type in type_mapping.items():
            if py_type.startswith(py_type_key):
                return json_type

        return "string"  # 默认值

    @staticmethod
    def extract_signature(func: Callable) -> Dict[str, Any]:
        """
        从 Python 函数提取签名信息，用于生成 LLM 工具描述

        Args:
            func: Python 函数对象

        Returns:
            包含函数元信息的字典
        """
        func_name = func.__name__
        func_doc = func.__doc__ or "无描述"

        # 提取类型注解
        annotations = func.__annotations__

        # 构建 parameters 字段
        properties = {}
        required = []

        for param_name, param_type in annotations.items():
            if param_name == "return":
                continue  # 跳过返回类型注解

            # 将类型转为字符串
            type_str = (
                str(param_type)
                .replace("typing.", "")
                .replace("<class '", "")
                .replace("'>", "")
            )

            properties[param_name] = {
                "type": SignatureMapper.python_type_to_json_type(type_str),
                "description": f"参数 {param_name}，类型 {type_str}"
            }
            required.append(param_name)

        return {
            "name": func_name,
            "description": func_doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# ============================================================
# 第四部分：register_function 模拟实现
# ============================================================

class FunctionRegistry:
    """
    模拟 ConversableAgent 的函数注册表

    实际 AutoGen 中：
    - _function_map: 存储注册的工具函数
    - register_function(): 向 Agent 注册工具函数，使其能被 LLM 调用
    """

    def __init__(self):
        self._function_map: Dict[str, Callable] = {}
        self._function_schemas: Dict[str, Dict] = {}

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        注册函数的模拟实现

        参数详解：
        1. func - 要注册的函数
           必须是可调用的 Python 函数
           建议包含类型注解，使 LLM 能理解参数类型

        2. name - 工具名称
           - None: 使用 func.__name__
           - str: 使用指定名称
           注意：name 会成为 LLM function calling 中的工具标识符

        3. description - 函数描述
           - None: 使用函数的 docstring
           - str: 使用指定描述
           重要：描述质量直接影响 LLM 调用准确性

        4. **kwargs - 其他参数（如 signature），会被传递给函数架构构建

        用法示例：
        ```python
        # 基本用法
        def add(a: int, b: int) -> int:
            '''加法计算器'''
            return a + b

        agent.register_function(add)

        # 指定名称和描述
        agent.register_function(
            add,
            name="calculator_add",
            description="执行两个数字的加法运算"
        )

        # 使用 function_map 批量注册
        agent.register_function(
            func=my_function,
            name="custom_name",
            **kwargs
        )
        ```
        """
        # 确定函数名
        func_name = name if name else func.__name__

        # 确定描述
        func_desc = (
            description
            if description
            else (func.__doc__ or "无描述").strip()
        )

        # 提取函数签名
        schema = SignatureMapper.extract_signature(func)
        schema["name"] = func_name
        schema["description"] = func_desc

        # 存储到注册表
        self._function_map[func_name] = func
        self._function_schemas[func_name] = schema

        print(f"[注册函数] 名称: {func_name}")
        print(f"           描述: {func_desc}")
        print(f"           签名: {json.dumps(schema['parameters'], ensure_ascii=False, indent=2)}")

    def get_function(self, name: str) -> Optional[Callable]:
        """
        根据名称获取已注册的函数

        Args:
            name: 函数名称

        Returns:
            函数对象，如果未找到返回 None
        """
        return self._function_map.get(name)

    def get_all_schemas(self) -> List[Dict]:
        """
        获取所有函数的 schema 列表（用于 LLM 工具调用）

        Returns:
            函数 schema 列表
        """
        return list(self._function_schemas.values())

    def list_registered(self) -> None:
        """列出所有已注册的函数"""
        print("\n" + "=" * 60)
        print("已注册的函数列表：")
        print("=" * 60)

        if not self._function_map:
            print("  (空)")
            return

        for name, func in self._function_map.items():
            schema = self._function_schemas[name]
            print(f"\n  [{name}]")
            print(f"      原函数: {func.__name__}")
            print(f"      描述: {schema['description']}")
            print(f"      参数: {list(schema['parameters']['properties'].keys())}")


# ============================================================
# 第五部分：实际业务工具注册示例
# ============================================================

class BusinessTools:
    """
    演示如何在实际业务中注册工具函数
    """

    @staticmethod
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
        if bmi < 18.5:
            category = "偏瘦"
            advice = "建议适当增加营养摄入"
        elif bmi < 24:
            category = "正常"
            advice = "继续保持健康生活方式"
        elif bmi < 28:
            category = "偏胖"
            advice = "建议适当增加运动量"
        else:
            category = "肥胖"
            advice = "建议咨询专业医生制定减肥计划"

        return {
            "bmi": round(bmi, 2),
            "category": category,
            "advice": advice
        }

    @staticmethod
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        货币换算（模拟）

        Args:
            amount: 金额
            from_currency: 源货币代码（如 "CNY", "USD"）
            to_currency: 目标货币代码

        Returns:
            换算结果字典
        """
        # 简化的汇率表（实际应用中应调用真实汇率API）
        rates_to_usd = {
            "CNY": 0.14,
            "USD": 1.0,
            "EUR": 1.08,
            "JPY": 0.0067,
            "GBP": 1.27
        }

        if from_currency not in rates_to_usd or to_currency not in rates_to_usd:
            return {"error": f"不支持的货币代码: {from_currency} 或 {to_currency}"}

        # 转换为 USD 再转目标货币
        usd_amount = amount * rates_to_usd[from_currency]
        result = usd_amount / rates_to_usd[to_currency]

        return {
            "original": f"{amount} {from_currency}",
            "result": f"{result:.2f} {to_currency}",
            "rate": f"1 {to_currency} = {1/rates_to_usd[to_currency]:.2f} {from_currency}"
        }

    @staticmethod
    def get_weather(city: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取天气预报（模拟）

        Args:
            city: 城市名称
            date: 日期（可选），格式 YYYY-MM-DD，默认返回今天

        Returns:
            天气信息字典
        """
        # 模拟天气数据
        weather_data = {
            "北京": {"temp": 22, "condition": "晴", "humidity": 45},
            "上海": {"temp": 25, "condition": "多云", "humidity": 60},
            "广州": {"temp": 28, "condition": "雷阵雨", "humidity": 75},
            "深圳": {"temp": 29, "condition": "晴", "humidity": 55},
        }

        if city not in weather_data:
            return {"error": f"未找到城市 {city} 的天气数据"}

        data = weather_data[city]
        return {
            "city": city,
            "date": date or "今天",
            "temperature": f"{data['temp']}°C",
            "condition": data['condition'],
            "humidity": f"{data['humidity']}%"
        }


# ============================================================
# 第六部分：function_map 参数详解
# ============================================================

class FunctionMapDemo:
    """
    演示 function_map 参数与 register_function 的关系

    function_map 是 ConversableAgent.__init__ 中的一个参数，
    用于在创建 Agent 时批量注册函数。

    对比：
    - register_function(): 创建 Agent 后动态注册单个函数
    - function_map: 创建 Agent 时批量注册多个函数

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
    """

    @staticmethod
    def demo_function_map_usage():
        """
        演示 function_map 的用法
        """
        print("\n" + "-" * 60)
        print("function_map 参数演示")
        print("-" * 60)

        registry = FunctionRegistry()

        # 模拟 function_map 批量注册
        function_map = {
            "bmi_calculator": BusinessTools.calculate_bmi,
            "currency_converter": BusinessTools.convert_currency,
            "weather_checker": BusinessTools.get_weather,
        }

        # 批量注册
        for name, func in function_map.items():
            registry.register_function(func, name=name)

        # 显示结果
        registry.list_registered()

        # 模拟 LLM 获取工具列表
        print("\n" + "-" * 60)
        print("LLM 看到的工具描述（Function Calling Schema）：")
        print("-" * 60)
        schemas = registry.get_all_schemas()
        for schema in schemas:
            print(f"\n{json.dumps(schema, ensure_ascii=False, indent=2)}")


# ============================================================
# 第七部分：register_function 完整流程演示
# ============================================================

class RegisterFlowDemo:
    """
    演示从函数定义到 LLM 调用工具的完整流程
    """

    @staticmethod
    def demo_complete_flow():
        """
        完整流程：
        1. 定义业务函数（带类型注解）
        2. 使用 register_function 注册到 Agent
        3. Agent 将函数 schema 发送给 LLM
        4. LLM 决定调用工具并返回 function_call
        5. Agent 执行函数并返回结果
        """
        print("\n" + "=" * 60)
        print("register_function 完整流程演示")
        print("=" * 60)

        registry = FunctionRegistry()

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
                    "address": "北京市朝阳区某某路1号",
                    "create_time": "2024-01-15 10:30:00"
                },
                "87654321": {
                    "order_id": "87654321",
                    "status": "待支付",
                    "amount": 1599.00,
                    "address": "上海市浦东新区某某街100号",
                    "create_time": "2024-01-20 15:45:00"
                }
            }

            if order_id not in orders:
                return {"error": f"未找到订单 {order_id}"}

            return orders[order_id]

        # 步骤2：注册函数
        print("\n步骤2：注册函数")
        print("-" * 40)
        registry.register_function(
            order_query,
            name="query_order",
            description="查询用户订单的物流状态和详细信息"
        )

        # 步骤3：获取 LLM 可用的工具列表
        print("\n步骤3：获取 LLM 工具描述")
        print("-" * 40)
        schemas = registry.get_all_schemas()
        print(f"共 {len(schemas)} 个工具可用")
        print(f"\n工具 Schema:\n{json.dumps(schemas[0], ensure_ascii=False, indent=2)}")

        # 步骤4：模拟 LLM 调用
        print("\n步骤4：模拟 LLM Function Call")
        print("-" * 40)

        # LLM 生成的调用请求
        llm_call_request = {
            "name": "query_order",
            "arguments": {
                "order_id": "12345678",
                "user_id": "u_10001"
            }
        }

        print(f"LLM 请求调用: {llm_call_request['name']}")
        print(f"参数: {json.dumps(llm_call_request['arguments'], ensure_ascii=False)}")

        # 步骤5：执行函数
        print("\n步骤5：Agent 执行函数")
        print("-" * 40)

        func = registry.get_function(llm_call_request["name"])
        if func:
            result = func(**llm_call_request["arguments"])
            print(f"执行结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print(f"错误：未找到函数 {llm_call_request['name']}")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 加载环境变量
    load_env()

    print("\n" + "#" * 60)
    print("# 第一部分：register_function 核心概念")
    print("#" * 60)

    demo = RegisterFunctionDemo()
    print(f"方法签名演示：{demo}")

    print("\n" + "#" * 60)
    print("# 第二部分：函数签名映射原理")
    print("#" * 60)

    def sample_func(a: int, b: str, c: List[float]) -> Dict:
        """示例函数用于演示签名提取"""
        return {"a": a, "b": b, "c": c}

    schema = SignatureMapper.extract_signature(sample_func)
    print(f"Python 函数签名提取结果：")
    print(json.dumps(schema, ensure_ascii=False, indent=2))

    print("\n" + "#" * 60)
    print("# 第三部分：业务工具注册示例")
    print("#" * 60)

    biz_registry = FunctionRegistry()
    biz_registry.register_function(
        BusinessTools.calculate_bmi,
        name="calc_bmi",
        description="根据身高体重计算BMI指数，评估健康状况"
    )
    biz_registry.register_function(
        BusinessTools.get_weather,
        name="check_weather",
        description="查询指定城市的天气预报信息"
    )
    biz_registry.list_registered()

    print("\n" + "#" * 60)
    print("# 第四部分：function_map 参数演示")
    print("#" * 60)

    FunctionMapDemo.demo_function_map_usage()

    print("\n" + "#" * 60)
    print("# 第五部分：完整流程演示")
    print("#" * 60)

    RegisterFlowDemo.demo_complete_flow()

    print("\n" + "=" * 60)
    print("register_function 用法演示结束")
    print("=" * 60)
    print("""
    学习要点总结：
    1. register_function 将 Python 函数注册为 Agent 工具
    2. 函数签名通过类型注解映射为 LLM Function Calling 格式
    3. description 描述影响 LLM 的调用决策
    4. function_map 用于创建 Agent 时批量注册函数

    下一步：
    - 查看 llm_call_vs_exec.py 了解 register_for_llm_call 与 register_for_exec 的区别
    - 阅读 AutoGen 源码验证 register_function 实现细节
    """)