# register_reply_demo.py
# 第4节 generate_reply策略链与register_reply深度掌握
# 演示 register_reply 的各种用法
#
# 本文件展示 register_reply 方法的完整用法：
# 1. trigger 类型详解（str、tuple、regex、callable、None）
# 2. config 传参模式
# 3. 链式注册与优先级控制
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
# 第二部分：register_reply 方法签名解析
# ============================================================

@dataclass
class RegisterReplyConfig:
    """
    register_reply 的配置结构（模拟 AutoGen 的实际参数）

    register_reply 方法签名：
    ```python
    def register_reply(
        trigger: Union[str, Pattern, Callable, None],  # 触发条件
        reply_func: Callable,                           # 回调函数
        position: Optional[int] = None,                 # 注册位置
        config: Optional[Dict[str, Any]] = None,        # 配置字典
        invalidation_message: Optional[str] = None,     # 失效消息
    ):
    ```

    参数详解：
    - trigger: 触发条件，支持多种类型
    - reply_func: 回调函数，接收 (messages, sender, config) 并返回回复
    - position: 在 _reply_func_list 中的位置，None 表示追加到末尾
    - config: 传递给 reply_func 的配置字典
    - invalidation_message: 可选的失效消息（用于清除某些回复条件）
    """
    trigger: Union[str, tuple, re.Pattern, Callable, None]
    reply_func: Callable
    position: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    invalidation_message: Optional[str] = None


# ============================================================
# 第三部分：四种 trigger 类型详解
# ============================================================

class TriggerTypeDemo:
    """
    演示四种 trigger 类型的用法
    """

    # --------------------------------------------------
    # 类型1：字符串前缀匹配 (str trigger)
    # --------------------------------------------------

    @staticmethod
    def str_trigger_handler(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
        """
        处理以特定前缀开头的消息

        Args:
            messages: 消息列表
            sender: 发送者标识
            config: 配置字典

        Returns:
            str: 回复内容

        示例：
            register_reply("/help", help_handler)
            - 触发条件：消息 content 以 "/help" 开头
            - "帮我看看" → 不触发
            - "/help" → 触发
            - "/help me" → 触发
        """
        latest = messages[-1]
        content = latest.get("content", "")

        # 在实际应用中，trigger 会自动处理，这里手动演示
        trigger = "/help" if not isinstance(latest.get("_trigger"), str) else latest.get("_trigger")

        if content.startswith(trigger):
            return f"[前缀匹配] 您请求了帮助信息。配置: {config}"
        return ""


    # --------------------------------------------------
    # 类型2：tuple 前缀匹配 (tuple trigger)
    # --------------------------------------------------

    @staticmethod
    def tuple_trigger_handler(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
        """
        处理以多个可能前缀开头的消息

        Args:
            messages: 消息列表
            sender: 发送者标识
            config: 配置字典

        示例：
            register_reply(("/status", "/stat", "/s"), status_handler)
            - 触发条件：消息 content 以 tuple 中任一元素开头
            - "/status" → 触发
            - "/stat" → 触发
            - "/s" → 触发
            - "status" → 不触发（没有 /）
        """
        latest = messages[-1]
        content = latest.get("content", "")

        triggers = ("/status", "/stat", "/s")

        if any(content.startswith(p) for p in triggers):
            return f"[tuple匹配] 系统状态查询。配置: {config}"
        return ""


    # --------------------------------------------------
    # 类型3：正则表达式匹配 (Pattern trigger)
    # --------------------------------------------------

    @staticmethod
    def regex_trigger_handler(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
        """
        使用正则表达式匹配消息内容

        Args:
            messages: 消息列表
            sender: 发送者标识
            config: 配置字典

        示例：
            pattern = re.compile(r"\\b\\d{11}\\b")  # 匹配11位数字（手机号）
            register_reply(pattern, phone_handler)

            - "我的电话是13812345678" → 触发，提取到 13812345678
            - "你好" → 不触发
        """
        latest = messages[-1]
        content = latest.get("content", "")

        # 匹配手机号的正则表达式
        phone_pattern = re.compile(r'\b(\d{11})\b')

        match = phone_pattern.search(content)
        if match:
            return f"[正则匹配] 检测到手机号: {match.group(1)}"
        return ""


    # --------------------------------------------------
    # 类型4：函数条件判断 (callable trigger)
    # --------------------------------------------------

    @staticmethod
    def callable_trigger_handler(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
        """
        使用自定义函数判断是否触发

        Args:
            messages: 消息列表
            sender: 发送者标识
            config: 配置字典

        函数签名：
            trigger_function(message: Dict, sender: str) -> bool

        示例：
            def is_vip_user(msg, sender):
                return sender.startswith("vip_")

            register_reply(is_vip_user, vip_handler)
            - 触发条件：is_vip_user() 返回 True
        """
        latest = messages[-1]
        sender = latest.get("role", "")

        # 简化的 VIP 判断
        is_vip = sender.lower().startswith("vip") or sender == "admin"

        if is_vip:
            return f"[函数匹配] VIP/管理员用户专属回复"
        return ""


    # --------------------------------------------------
    # 类型5：None（无条件触发）
    # --------------------------------------------------

    @staticmethod
    def none_trigger_handler(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
        """
        无条件触发（通常用于处理代码执行结果）

        Args:
            messages: 消息列表
            sender: 发送者标识
            config: 配置字典

        示例：
            register_reply(None, code_result_handler)
            - 触发条件：始终触发（除非前面的策略已返回回复）
            - 通常与其他 trigger 配合使用，position 放在最后
        """
        latest = messages[-1]
        content = latest.get("content", "")

        # 通常用于处理特殊类型的内容
        if "```output" in content or "Execution" in content:
            return f"[无条件触发] 检测到执行结果"
        return ""


# ============================================================
# 第四部分：register_reply 模拟实现
# ============================================================

class ReplyFunctionRegistry:
    """
    模拟 ConversableAgent 的回复函数注册表

    实际 AutoGen 中：
    - _reply_func_list: 存储所有注册的回复函数
    - register_reply(): 向列表中添加新的回复函数
    """

    def __init__(self):
        self._reply_func_list: List[Dict[str, Any]] = []
        self._invalidation_rules: Dict[str, str] = {}


    def register_reply(
        self,
        trigger: Union[str, tuple, re.Pattern, Callable, None],
        reply_func: Callable,
        position: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        invalidation_message: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        注册回复函数的模拟实现

        参数详解：
        1. trigger - 触发条件
           - str: 前缀匹配
           - tuple: 多个前缀匹配
           - Pattern: 正则表达式匹配
           - Callable: 函数判断
           - None: 无条件触发

        2. reply_func - 回调函数
           签名: (messages, sender, config) -> str
           - messages: 消息历史列表
           - sender: 发送者标识
           - config: 配置字典（来自 register_reply 的 config 参数）

        3. position - 插入位置
           - None: 追加到列表末尾（最低优先级）
           - int: 插入到指定位置（数值越小优先级越高）
           - 0: 插入到开头（最高优先级）

        4. config - 配置字典
           这个字典会原样传递给 reply_func
           用于传递业务配置（如 VIP 用户列表等）

        5. invalidation_message - 失效消息
           当收到此消息时，移除对应的回复函数

        用法示例：
        ```python
        # 基本用法：前缀匹配
        agent.register_reply("/help", help_handler)

        # 指定位置：高优先级
        agent.register_reply("/debug", debug_handler, position=0)

        # 传递配置
        agent.register_reply(
            "VIP",
            vip_handler,
            config={"vip_greeting": "欢迎尊贵用户！"}
        )

        # 使用正则表达式
        agent.register_reply(
            re.compile(r"\\b\\d{11}\\b"),
            phone_handler
        )

        # 使用函数条件
        agent.register_reply(
            lambda msg, sender: sender == "admin",
            admin_handler
        )

        # 清除失效规则
        agent.register_reply(
            trigger="CLEAR",
            reply_func=lambda *args: "",
            invalidation_message="reset"
        )
        ```
        """
        record = {
            "trigger": trigger,
            "reply_func": reply_func,
            "config": config,
            "position": position,
            "invalidation_message": invalidation_message,
            **kwargs
        }

        # 根据 position 插入到合适的位置
        if position is None:
            # 追加到末尾
            self._reply_func_list.append(record)
        elif position == 0:
            # 插入到开头
            self._reply_func_list.insert(0, record)
        else:
            # 插入到指定位置
            self._reply_func_list.insert(position, record)

        print(f"[注册] 回复函数: {reply_func.__name__}")
        print(f"       触发类型: {self._get_trigger_type_name(trigger)}")
        print(f"       位置: {position if position is not None else '末尾(最低优先级)'}")
        print(f"       配置: {config}")


    def _get_trigger_type_name(self, trigger) -> str:
        """获取 trigger 类型的友好名称"""
        if trigger is None:
            return "None (无条件)"
        elif isinstance(trigger, str):
            return f"str (前缀匹配: '{trigger}')"
        elif isinstance(trigger, tuple):
            return f"tuple (多前缀匹配: {trigger})"
        elif isinstance(trigger, type(re.compile(r""))):
            return f"Pattern (正则: '{trigger.pattern}')"
        elif callable(trigger):
            return f"Callable (函数: {trigger.__name__})"
        else:
            return f"unknown ({type(trigger).__name__})"


    def unregister_reply(self, trigger, reply_func=None) -> None:
        """
        取消注册回复函数

        Args:
            trigger: 触发条件（用于匹配要移除的记录）
            reply_func: 可选的回调函数（精确匹配）

        如果提供了 reply_func，则同时匹配 trigger 和 reply_func
        如果只提供了 trigger，则移除所有匹配 trigger 的记录
        """
        original_count = len(self._reply_func_list)

        if reply_func is not None:
            self._reply_func_list = [
                r for r in self._reply_func_list
                if not (r["trigger"] == trigger and r["reply_func"] == reply_func)
            ]
        else:
            self._reply_func_list = [
                r for r in self._reply_func_list
                if r["trigger"] != trigger
            ]

        removed = original_count - len(self._reply_func_list)
        print(f"[移除] 移除了 {removed} 个回复函数记录")


    def list_registered(self) -> None:
        """列出所有已注册的回复函数"""
        print("\n" + "=" * 60)
        print("已注册的回复函数列表：")
        print("=" * 60)

        if not self._reply_func_list:
            print("  (空)")
            return

        for idx, record in enumerate(self._reply_func_list):
            print(f"\n  [{idx}] {record['reply_func'].__name__}")
            print(f"      触发: {self._get_trigger_type_name(record['trigger'])}")
            print(f"      位置: {record['position']}")
            print(f"      配置: {record['config']}")


# ============================================================
# 第五部分：config 传参模式详解
# ============================================================

class ConfigDemo:
    """
    演示 config 参数的多种传参模式
    """

    @staticmethod
    def config_as_context():
        """
        演示1：config 作为上下文传递

        config 是 register_reply 时传递的字典，会原样传递给 reply_func
        这允许你在不修改 reply_func 的情况下定制其行为
        """
        print("\n" + "-" * 60)
        print("config 用法演示1：作为上下文传递")
        print("-" * 60)

        def greeting_handler(messages, sender, config):
            """根据配置返回不同的问候语"""
            greeting = config.get("greeting", "你好")
            name = config.get("name", "用户")
            return f"{greeting}，{name}！"

        registry = ReplyFunctionRegistry()

        # 注册时传递配置
        registry.register_reply(
            trigger="hello",
            reply_func=greeting_handler,
            config={
                "greeting": "欢迎",
                "name": "VIP会员"
            }
        )

        # 模拟调用
        messages = [{"role": "user", "content": "hello"}]
        for record in registry._reply_func_list:
            result = record["reply_func"](messages, "user", record["config"])
            print(f"结果: {result}")


    @staticmethod
    def config_as_feature_flags():
        """
        演示2：config 作为功能开关

        使用 config 控制_reply_func 是否启用某些功能
        """
        print("\n" + "-" * 60)
        print("config 用法演示2：作为功能开关")
        print("-" * 60)

        def feature_handler(messages, sender, config):
            """根据配置决定是否启用某功能"""
            enabled = config.get("enabled", False)
            threshold = config.get("threshold", 10)

            if not enabled:
                return ""  # 功能未启用

            latest = messages[-1]
            content = latest.get("content", "")

            if len(content) > threshold:
                return f"[功能开启] 内容长度 {len(content)} 超过阈值 {threshold}"
            return ""

        registry = ReplyFunctionRegistry()

        # 未启用配置
        registry.register_reply(
            trigger="check",
            reply_func=feature_handler,
            config={"enabled": False, "threshold": 10}
        )

        # 启用配置
        registry.register_reply(
            trigger="check",
            reply_func=feature_handler,
            config={"enabled": True, "threshold": 5}
        )


    @staticmethod
    def config_as_vip_list():
        """
        演示3：config 传递业务数据（如VIP列表）
        """
        print("\n" + "-" * 60)
        print("config 用法演示3：传递业务数据")
        print("-" * 60)

        def vip_handler(messages, sender, config):
            """根据VIP列表判断用户是否VIP"""
            vip_users = config.get("vip_users", [])

            # 这里简化了判断逻辑
            if sender in vip_users:
                return f"[VIP] {sender} 是尊贵的VIP用户"
            return ""

        registry = ReplyFunctionRegistry()

        registry.register_reply(
            trigger="vip_check",
            reply_func=vip_handler,
            config={
                "vip_users": ["alice", "bob", "charlie"],
                "vip_level": "gold"
            }
        )


# ============================================================
# 第六部分：position 优先级演示
# ============================================================

class PositionDemo:
    """
    演示 position 参数对注册顺序的影响
    """

    @staticmethod
    def demo_priority_order():
        """
        演示优先级顺序对回复处理的影响

        注册顺序和 position 决定了策略链的遍历顺序：
        - position=0: 最高优先级，最先被检查
        - position=None: 追加到末尾，最低优先级

        重要：前面的函数返回非空会短路后面的函数
        """
        print("\n" + "-" * 60)
        print("position 优先级演示")
        print("-" * 60)

        registry = ReplyFunctionRegistry()

        def low_priority_handler(messages, sender, config):
            print("  [low_priority_handler] 被调用")
            return "[低优先级] 这个应该最后处理"

        def medium_priority_handler(messages, sender, config):
            print("  [medium_priority_handler] 被调用")
            return "[中优先级]"

        def high_priority_handler(messages, sender, config):
            print("  [high_priority_handler] 被调用")
            return "[高优先级]"

        # 1. 先注册低优先级
        registry.register_reply(
            trigger="priority_test",
            reply_func=low_priority_handler,
            position=None  # 末尾
        )

        # 2. 注册中优先级
        registry.register_reply(
            trigger="priority_test",
            reply_func=medium_priority_handler,
            position=1  # 中间位置
        )

        # 3. 最后注册高优先级
        registry.register_reply(
            trigger="priority_test",
            reply_func=high_priority_handler,
            position=0  # 开头（最高优先级）
        )

        # 列出当前注册顺序
        registry.list_registered()

        print("\n实际注册顺序（从高优先级到低优先级）：")
        for idx, record in enumerate(registry._reply_func_list):
            print(f"  {idx}: {record['reply_func'].__name__} (position={record['position']})")


    @staticmethod
    def demo_short_circuit():
        """
        演示短路机制

        当高优先级的函数返回非空时，低优先级的函数不会被调用
        """
        print("\n" + "-" * 60)
        print("短路机制演示")
        print("-" * 60)

        registry = ReplyFunctionRegistry()

        def first_handler(messages, sender, config):
            return "[first_handler] 我返回了，不执行后面的"

        def second_handler(messages, sender, config):
            print("  [second_handler] 被调用了吗？")
            return "[second_handler]"

        def third_handler(messages, sender, config):
            print("  [third_handler] 被调用了吗？")
            return "[third_handler]"

        # 按顺序注册
        registry.register_reply("short_circuit", first_handler, position=0)
        registry.register_reply("short_circuit", second_handler, position=1)
        registry.register_reply("short_circuit", third_handler, position=2)

        # 模拟触发
        print("\n模拟触发 short_circuit 消息：")
        messages = [{"role": "user", "content": "short_circuit test"}]

        for record in registry._reply_func_list:
            trigger = record["trigger"]
            content = messages[-1].get("content", "")

            # 检查是否触发
            should_trigger = (
                (isinstance(trigger, str) and content.startswith(trigger)) or
                (isinstance(trigger, tuple) and any(content.startswith(t) for t in trigger)) or
                (isinstance(trigger, type(re.compile(r""))) and trigger.search(content)) or
                (callable(trigger) and trigger(messages[-1], messages[-1].get("role", ""))) or
                (trigger is None)
            )

            if should_trigger:
                result = record["reply_func"](messages, "user", record["config"])
                if result:
                    print(f"  短路！收到回复: {result}")
                    break


# ============================================================
# 第七部分：完整使用示例
# ============================================================

def complete_example():
    """
    完整示例：构建一个有多重回复策略的 ConversableAgent
    """
    print("\n" + "=" * 60)
    print("完整示例：多重回复策略的 Agent")
    print("=" * 60)

    # 创建注册表（模拟 ConversableAgent）
    registry = ReplyFunctionRegistry()

    # 策略1：VIP用户优先
    def vip_priority_handler(messages, sender, config):
        if sender.startswith("vip_") or sender == "admin":
            return "[VIP优先] 您好，尊贵的用户！"
        return ""

    # 策略2：命令处理
    def command_handler(messages, sender, config):
        content = messages[-1].get("content", "")

        if content.startswith("/hello"):
            return "你好！有什么可以帮助你的？"
        elif content.startswith("/help"):
            return "可用命令：/hello, /help, /status"
        elif content.startswith("/status"):
            return "系统状态正常"

        return ""

    # 策略3：邮箱提取
    def email_extractor_handler(messages, sender, config):
        content = messages[-1].get("content", "")

        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        match = email_pattern.search(content)

        if match:
            return f"检测到邮箱: {match.group(0)}，已记录"
        return ""

    # 策略4：关键词触发
    def keyword_handler(messages, sender, config):
        content = messages[-1].get("content", "").lower()

        keywords = config.get("keywords", [])
        responses = config.get("responses", {})

        for keyword in keywords:
            if keyword.lower() in content:
                return responses.get(keyword, "")

        return ""

    # 按优先级注册
    print("\n注册回复策略：")
    registry.register_reply("vip_", vip_priority_handler, position=0)  # 最高优先级
    registry.register_reply(("/hello", "/help", "/status"), command_handler, position=1)
    registry.register_reply(re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), email_extractor_handler, position=2)
    registry.register_reply(
        None,  # 无条件触发（最低优先级，作为后备）
        keyword_handler,
        position=3,
        config={
            "keywords": ["你好", "再见", "谢谢"],
            "responses": {
                "你好": "你好！很高兴见到你！",
                "再见": "再见！下次见！",
                "谢谢": "不客气！有什么需要随时叫我！"
            }
        }
    )

    registry.list_registered()

    # 测试各种场景
    print("\n" + "-" * 60)
    print("测试场景：")
    print("-" * 60)

    test_cases = [
        ({"role": "vip_user", "content": "你好"}, "VIP用户"),
        ({"role": "user", "content": "/help"}, "help命令"),
        ({"role": "user", "content": "请发到 test@example.com"}, "邮箱"),
        ({"role": "user", "content": "你好，谢谢"}, "关键词"),
        ({"role": "user", "content": "普通消息"}, "无匹配"),
    ]

    for msg, desc in test_cases:
        messages = [msg]
        sender = msg.get("role", "")

        print(f"\n>>> 测试: {desc}")
        print(f"    消息: {msg['content']}")

        reply = ""
        for record in registry._reply_func_list:
            trigger = record["trigger"]
            content = msg.get("content", "")

            # 简化触发判断
            triggered = False
            if isinstance(trigger, str):
                triggered = content.startswith(trigger)
            elif isinstance(trigger, tuple):
                triggered = any(content.startswith(t) for t in trigger)
            elif isinstance(trigger, type(re.compile(r""))):
                triggered = bool(trigger.search(content))
            elif callable(trigger):
                try:
                    triggered = trigger(msg, sender)
                except:
                    triggered = False
            elif trigger is None:
                triggered = True

            if triggered:
                result = record["reply_func"](messages, sender, record["config"])
                if result:
                    reply = result
                    break

        print(f"    回复: {reply if reply else '(无匹配，使用默认处理)'}")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 加载环境变量
    load_env()

    # 运行所有演示
    print("\n" + "#" * 60)
    print("# 第一部分：trigger 类型详解")
    print("#" * 60)

    TriggerTypeDemo.str_trigger_handler(
        [{"role": "user", "content": "/help me"}], "user", None
    )
    print("  str trigger 示例已执行\n")

    print("\n" + "#" * 60)
    print("# 第二部分：config 传参模式")
    print("#" * 60)

    ConfigDemo.config_as_context()
    ConfigDemo.config_as_feature_flags()
    ConfigDemo.config_as_vip_list()

    print("\n" + "#" * 60)
    print("# 第三部分：position 优先级")
    print("#" * 60)

    PositionDemo.demo_priority_order()
    PositionDemo.demo_short_circuit()

    print("\n" + "#" * 60)
    print("# 第四部分：完整示例")
    print("#" * 60)

    complete_example()

    print("\n" + "=" * 60)
    print("register_reply 用法演示结束")
    print("=" * 60)
    print("""
    学习要点总结：
    1. trigger 类型：str、tuple、Pattern、Callable、None
    2. config 传参：作为上下文、功能开关、业务数据
    3. position 优先级：数值越小优先级越高
    4. 短路机制：高优先级返回非空时停止遍历

    下一步：
    - 查看 generate_reply_flow.py 了解执行流程
    - 阅读 AutoGen 源码验证实现细节
    """)