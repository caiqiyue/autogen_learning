"""
第19节 代码案例：subclass_basic.py
自定义Agent子类化基础方法演示

本文件展示如何通过继承ConversableAgent来创建自定义Agent，
实现业务逻辑的封装与复用。

核心概念：
1. 子类化：通过继承ConversableAgent创建专用Agent
2. 重写方法：覆写generate_reply等核心方法
3. register_reply：注册自定义回复函数
"""

# ============================================================
# 导入必要的模块
# ============================================================
import os
import re
from typing import Any, Callable, Dict, List, Optional, Union, Pattern

# 导入AutoGen核心组件
from autogen import ConversableAgent, Agent
from autogen.agentchat.conversable_agent import KnownIssue


# ============================================================
# 案例1：基础客服Agent
# ============================================================

class CustomerServiceAgent(ConversableAgent):
    """
    基础客服Agent - 演示最简单直接的子类化方法

    这个Agent会自动处理常见的问候语和常见问题，
    无需每次都调用LLM，提升响应速度。

    使用场景：
    - 常见问题快速回复
    - 降低API调用成本
    - 提升响应速度
    """

    def __init__(self, name: str, **kwargs):
        """
        初始化客服Agent

        Args:
            name: Agent名称
            **kwargs: 传递给父类的其他参数
        """
        # 调用父类构造函数
        super().__init__(name, **kwargs)

        # 注册基础回复函数（使用position=0提高优先级）
        self.register_reply(
            trigger=self._is_greeting,  # 触发条件
            reply_func=self._greeting_reply,  # 回复函数
            position=0  # 高优先级
        )

        # 注册关键词回复
        self.register_reply(
            trigger=self._is_keywords,
            reply_func=self._keyword_reply,
            position=1
        )

    def _is_greeting(self, messages: List[Dict], sender: Agent) -> bool:
        """
        判断是否为问候语

        Args:
            messages: 消息历史列表
            sender: 发送者Agent

        Returns:
            bool: 如果是问候语返回True
        """
        if not messages:
            return False

        latest_message = messages[-1]
        content = latest_message.get("content", "").lower()

        # 定义问候语关键词列表
        greeting_keywords = ["你好", "您好", "hello", "hi", "早上好", "晚上好"]

        # 检查消息是否以问候语开头
        return any(content.startswith(keyword) for keyword in greeting_keywords)

    def _greeting_reply(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理问候语回复

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典（未使用，保留接口一致性）

        Returns:
            str: 回复内容
        """
        return "您好！我是智能客服，有什么可以帮助您的吗？"

    def _is_keywords(self, messages: List[Dict], sender: Agent) -> bool:
        """
        判断是否包含常见问题关键词

        Args:
            messages: 消息历史列表
            sender: 发送者Agent

        Returns:
            bool: 如果匹配关键词返回True
        """
        if not messages:
            return False

        latest_message = messages[-1]
        content = latest_message.get("content", "").lower()

        # 定义关键词列表
        keywords = ["价格", "多少钱", "怎么买", "联系", "电话", "地址"]

        # 检查是否包含任何关键词
        return any(keyword in content for keyword in keywords)

    def _keyword_reply(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理关键词回复

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 回复内容
        """
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "").lower()

        # 根据关键词返回对应回复
        if "价格" in content or "多少钱" in content:
            return "我们的产品价格根据配置不同，从99元到999元不等。您可以告诉我您的具体需求，我帮您推荐。"

        if "怎么买" in content:
            return "您可以通过我们的官网在线购买，也可以联系客服人工下单。"

        if "联系" in content or "电话" in content:
            return "我们的客服电话是 400-123-4567，工作时间是周一到周五 9:00-18:00。"

        if "地址" in content:
            return "我们公司的地址是：北京市朝阳区科技园区A座10层。"

        return ""  # 空字符串表示不处理，继续检查其他策略


# ============================================================
# 案例2：带配置的客服Agent（使用config参数）
# ============================================================

class ConfigurableCustomerAgent(ConversableAgent):
    """
    可配置客服Agent - 演示如何使用config参数

    通过config参数，可以实现：
    - 动态配置欢迎语
    - 自定义知识库
    - 业务规则配置
    """

    def __init__(self, name: str, greeting: str = "您好！", **kwargs):
        """
        初始化可配置客服Agent

        Args:
            name: Agent名称
            greeting: 自定义欢迎语
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(name, **kwargs)

        # 将配置存储在实例中
        self.greeting = greeting

        # 注册带有配置的回复函数
        self.register_reply(
            trigger=None,  # 无条件触发，作为后备处理
            reply_func=self._default_handler,
            position=2,  # 低优先级，在其他策略之后
            config={  # 配置字典，会传递给reply_func
                "greeting": greeting,
                "farewell": "再见，欢迎下次光临！",
                "unknown": "抱歉，我暂时无法回答这个问题，请联系人工客服。"
            }
        )

    def _default_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        默认处理器 - 演示config参数的使用

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典，来自register_reply的config参数

        Returns:
            str: 回复内容
        """
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "").lower()

        # 检查再见意图
        farewell_words = ["再见", "拜拜", "bye", "下次见"]
        if any(word in content for word in farewell_words):
            return config.get("farewell", "再见！")

        # 检查是否为空消息
        if not content.strip():
            return config.get("greeting", "您好！")

        # 其他情况返回未知回复
        # 注意：这个实现会捕获所有未被处理的消息
        # 实际应用中可能需要更复杂的逻辑
        return ""


# ============================================================
# 案例3：带正则匹配的专业Agent
# ============================================================

class OrderProcessAgent(ConversableAgent):
    """
    订单处理Agent - 演示正则表达式匹配

    使用正则表达式来匹配和解析：
    - 订单号查询
    - 物流状态查询
    - 退换货请求
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        # 注册订单号查询处理
        # 匹配格式：订单号、order、查单 等关键词 + 数字
        self.register_reply(
            trigger=re.compile(r"(订单号|order|查单)[:：]?\s*(\d{10,})", re.IGNORECASE),
            reply_func=self._order_query_handler,
            position=0
        )

        # 注册物流查询处理
        # 匹配格式：物流、快递、发货 等关键词
        self.register_reply(
            trigger=re.compile(r"(物流|快递|发货| shipment)[:：]?\s*([A-Z0-9]{10,})", re.IGNORECASE),
            reply_func=self._logistics_handler,
            position=0
        )

        # 注册退换货处理
        self.register_reply(
            trigger=re.compile(r"(退货|换货|退款|return|refund)", re.IGNORECASE),
            reply_func=self._return_handler,
            position=0
        )

    def _order_query_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理订单号查询

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 订单状态信息
        """
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")

        # 从消息中提取订单号
        match = re.search(r"\d{10,}", content)
        if match:
            order_id = match.group()
            # 实际应用中这里会调用订单系统API
            return f"您的订单 {order_id} 状态为：已发货，预计2-3天送达。"

        return ""

    def _logistics_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理物流查询

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 物流信息
        """
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")

        # 从消息中提取快递号
        match = re.search(r"[A-Z0-9]{10,}", content, re.IGNORECASE)
        if match:
            tracking_number = match.group()
            # 实际应用中这里会调用物流API
            return f"快递号 {tracking_number} 的物流状态：包裹已到达北京分拨中心"

        return ""

    def _return_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理退换货请求

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 退换货流程说明
        """
        return "您好，关于退换货服务，请提供订单号，我帮您申请。退换货需在签收后7天内完成，感谢理解！"


# ============================================================
# 案例4：带函数触发的条件判断Agent
# ============================================================

class VIPAgent(ConversableAgent):
    """
    VIP会员Agent - 演示函数触发条件

    根据发送者身份自动识别VIP用户并提供差异化服务：
    - VIP专属欢迎语
    - VIP优先处理
    - 专属优惠
    """

    def __init__(self, name: str, vip_users: List[str] = None, **kwargs):
        """
        初始化VIP Agent

        Args:
            name: Agent名称
            vip_users: VIP用户ID列表
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(name, **kwargs)

        # VIP用户列表
        self.vip_users = vip_users or ["vip_user_001", "vip_user_002", "vip_gold_001"]

        # 注册VIP识别触发器（使用函数条件判断）
        self.register_reply(
            trigger=self._is_vip_user,  # 触发函数
            reply_func=self._vip_handler,
            position=0  # 高优先级
        )

        # 注册普通用户处理
        self.register_reply(
            trigger=None,  # 无条件触发
            reply_func=self._normal_handler,
            position=1
        )

    def _is_vip_user(self, messages: List[Dict], sender: Agent) -> bool:
        """
        判断发送者是否为VIP用户

        Args:
            messages: 消息历史列表
            sender: 发送者Agent（关键！用于判断用户身份）

        Returns:
            bool: 如果是VIP返回True
        """
        if not sender:
            return False

        # 获取发送者名称（通常包含用户ID信息）
        sender_name = sender.name if hasattr(sender, 'name') else str(sender)

        # 检查是否在VIP列表中
        return any(vip_id in sender_name for vip_id in self.vip_users)

    def _vip_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理VIP用户请求

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: VIP专属回复
        """
        return "尊敬VIP会员，感谢您一直以来对我们的支持！您享有优先处理权和专属折扣。"

    def _normal_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        处理普通用户请求

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 普通用户回复
        """
        return "您好！请问有什么可以帮助您的？" if messages else ""


# ============================================================
# 案例5：演示如何组合使用多种触发器
# ============================================================

class CombinedTriggerAgent(ConversableAgent):
    """
    组合触发器Agent - 演示同时使用多种触发方式

    这个Agent展示如何组合：
    - 前缀匹配（str）
    - 正则匹配（Pattern）
    - 函数判断（Callable）
    - 无条件触发（None）
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        # 方式1：字符串前缀匹配 - 处理命令
        self.register_reply(
            trigger="/help",
            reply_func=self._help_handler,
            position=0
        )

        self.register_reply(
            trigger="/status",
            reply_func=self._status_handler,
            position=0
        )

        self.register_reply(
            trigger="/reset",
            reply_func=self._reset_handler,
            position=0
        )

        # 方式2：正则表达式匹配 - 处理手机号
        self.register_reply(
            trigger=re.compile(r"1[3-9]\d{9}"),  # 匹配11位手机号
            reply_func=self._phone_handler,
            position=0
        )

        # 方式3：函数条件判断 - 检查特定条件
        self.register_reply(
            trigger=self._is_long_message,
            reply_func=self._long_message_handler,
            position=1
        )

        # 方式4：默认处理（无条件触发）
        self.register_reply(
            trigger=None,
            reply_func=self._default_reply,
            position=2
        )

    def _help_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """处理help命令"""
        return """
可用命令：
/help - 显示帮助信息
/status - 查看系统状态
/reset - 重置会话
手机号 - 自动识别并保存联系方式
"""

    def _status_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """处理status命令"""
        return "系统状态：正常运行中 | 在线用户：128人 | 处理消息：1,234条"

    def _reset_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """处理reset命令"""
        return "会话已重置，有什么需要帮助的吗？"

    def _phone_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """处理手机号识别"""
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")

        # 提取手机号
        match = re.search(r"1[3-9]\d{9}", content)
        if match:
            phone = match.group()
            # 实际应用中会保存到数据库
            return f"已识别您的手机号：{phone}，我们将以此联系您。"

        return ""

    def _is_long_message(self, messages: List[Dict], sender: Agent) -> bool:
        """判断是否为长消息（超过100字符）"""
        if not messages:
            return False

        latest_message = messages[-1]
        content = latest_message.get("content", "")
        return len(content) > 100

    def _long_message_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """处理长消息"""
        return "您的消息较长，我们已记录。如果需要帮助，请简述您的问题，我们会尽快回复。"

    def _default_reply(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """默认回复"""
        if messages:
            return "收到您的消息，有什么可以帮助您的吗？"
        return ""


# ============================================================
# 演示函数：展示各类Agent的使用方法
# ============================================================

def demo_basic_subclass():
    """
    演示基础子类化Agent的使用

    创建一个客服Agent并模拟交互
    """
    print("=" * 60)
    print("演示1：基础客服Agent")
    print("=" * 60)

    # 创建Agent实例
    agent = CustomerServiceAgent(
        name="在线客服",
        llm_config=False  # 不使用LLM，纯规则引擎
    )

    # 模拟用户消息
    test_messages = [
        {"content": "你好", "role": "user"},
        {"content": "价格是多少", "role": "user"},
        {"content": "我想退货", "role": "user"}
    ]

    print("\n--- 测试结果 ---")
    for msg in test_messages:
        # 直接调用generate_reply（不通过Agent协作）
        response = agent.generate_reply(
            messages=[msg],
            sender=None
        )
        print(f"用户: {msg['content']}")
        print(f"客服: {response}")
        print()


def demo_configurable_agent():
    """
    演示可配置Agent的使用
    """
    print("=" * 60)
    print("演示2：可配置客服Agent")
    print("=" * 60)

    # 创建带有自定义欢迎语的Agent
    agent = ConfigurableCustomerAgent(
        name="定制客服",
        greeting="欢迎光临！请问有什么可以帮您？",
        llm_config=False
    )

    test_messages = [
        {"content": "你好", "role": "user"},
        {"content": "再见", "role": "user"}
    ]

    print("\n--- 测试结果 ---")
    for msg in test_messages:
        response = agent.generate_reply(
            messages=[msg],
            sender=None
        )
        print(f"用户: {msg['content']}")
        print(f"客服: {response}")
        print()


def demo_order_agent():
    """
    演示订单处理Agent的正则匹配
    """
    print("=" * 60)
    print("演示3：订单处理Agent（正则匹配）")
    print("=" * 60)

    agent = OrderProcessAgent(
        name="订单助手",
        llm_config=False
    )

    test_messages = [
        {"content": "订单号: 1234567890123456", "role": "user"},
        {"content": "快递号: SF1234567890", "role": "user"},
        {"content": "我想申请退货", "role": "user"}
    ]

    print("\n--- 测试结果 ---")
    for msg in test_messages:
        response = agent.generate_reply(
            messages=[msg],
            sender=None
        )
        print(f"用户: {msg['content']}")
        print(f"助手: {response}")
        print()


def demo_vip_agent():
    """
    演示VIP Agent的函数触发
    """
    print("=" * 60)
    print("演示4：VIP Agent（函数触发条件）")
    print("=" * 60)

    # 模拟一个VIP用户（通过sender名称标识）
    class MockVIPUser:
        name = "vip_user_001"

    class MockNormalUser:
        name = "normal_user_123"

    agent = VIPAgent(
        name="VIP服务",
        vip_users=["vip_user_001", "vip_gold_001"],
        llm_config=False
    )

    print("\n--- 测试结果 ---")

    # VIP用户测试
    response = agent.generate_reply(
        messages=[{"content": "你好", "role": "user"}],
        sender=MockVIPUser()
    )
    print(f"VIP用户: 你好")
    print(f"Agent: {response}")

    # 普通用户测试
    response = agent.generate_reply(
        messages=[{"content": "你好", "role": "user"}],
        sender=MockNormalUser()
    )
    print(f"普通用户: 你好")
    print(f"Agent: {response}")


def demo_combined_agent():
    """
    演示组合触发器Agent
    """
    print("=" * 60)
    print("演示5：组合触发器Agent")
    print("=" * 60)

    agent = CombinedTriggerAgent(
        name="综合助手",
        llm_config=False
    )

    test_messages = [
        {"content": "/help", "role": "user"},
        {"content": "我的手机号是13812345678", "role": "user"},
        {"content": "a" * 150, "role": "user"},  # 长消息
        {"content": "今天天气不错", "role": "user"}  # 普通消息
    ]

    print("\n--- 测试结果 ---")
    for msg in test_messages:
        response = agent.generate_reply(
            messages=[msg],
            sender=None
        )
        content_display = msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
        print(f"用户: {content_display}")
        print(f"助手: {response}")
        print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AutoGen自定义Agent子类化 - 基础演示")
    print("=" * 60)

    # 运行所有演示
    demo_basic_subclass()
    demo_configurable_agent()
    demo_order_agent()
    demo_vip_agent()
    demo_combined_agent()

    print("\n演示完成！")
    print("=" * 60)