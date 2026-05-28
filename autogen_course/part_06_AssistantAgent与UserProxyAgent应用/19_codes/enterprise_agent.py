"""
第19节 代码案例：enterprise_agent.py
企业级自定义Agent设计模式演示

本文件展示企业级场景下的自定义Agent设计模式，包括：
1. 插件式架构：功能模块化，可动态加载
2. 多策略组合：复杂业务逻辑的策略编排
3. 状态管理：会话状态持久化与恢复
4. 审计日志：操作记录与追溯
5. 错误处理与降级：容错机制设计
"""

# ============================================================
# 导入必要的模块
# ============================================================
import os
import re
import json
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# 导入AutoGen核心组件
from autogen import ConversableAgent, Agent
from autogen.agentchat.conversable_agent import KnownIssue


# ============================================================
# 辅助类和函数
# ============================================================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class AuditLog:
    """
    审计日志数据类

    用于记录所有Agent交互操作，支持：
    - 操作类型分类
    - 时间戳记录
    - 请求/响应记录
    - 性能指标追踪
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    operation: str = ""
    trigger_type: str = ""  # prefix/regex/func/none
    request_content: str = ""
    response_content: str = ""
    processing_time_ms: float = 0.0
    status: str = "success"  # success/error
    error_message: str = ""


class AuditLogger:
    """
    审计日志记录器

    企业级应用必须具备的操作记录能力，
    用于：
    - 合规审计
    - 问题排查
    - 性能分析
    - 用户行为分析
    """

    def __init__(self, log_file: str = "audit.log"):
        """
        初始化审计日志记录器

        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        self.logs: List[AuditLog] = []

        # 配置标准日志记录器
        self.logger = logging.getLogger("AuditLogger")
        self.logger.setLevel(logging.INFO)

        # 如果还没有handler，添加一个
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_operation(self, audit_log: AuditLog):
        """
        记录操作日志

        Args:
            audit_log: 审计日志对象
        """
        self.logs.append(audit_log)

        # 同时写入标准日志
        log_message = (
            f"[{audit_log.timestamp}] {audit_log.operation} | "
            f"触发类型: {audit_log.trigger_type} | "
            f"状态: {audit_log.status} | "
            f"耗时: {audit_log.processing_time_ms:.2f}ms"
        )

        if audit_log.status == "success":
            self.logger.info(log_message)
        else:
            self.logger.error(f"{log_message} | 错误: {audit_log.error_message}")

    def get_logs(self, operation: str = None, status: str = None) -> List[AuditLog]:
        """
        查询日志

        Args:
            operation: 操作类型过滤
            status: 状态过滤

        Returns:
            过滤后的日志列表
        """
        result = self.logs

        if operation:
            result = [log for log in result if log.operation == operation]

        if status:
            result = [log for log in result if log.status == status]

        return result

    def export_to_json(self, filepath: str):
        """
        导出日志为JSON格式

        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [vars(log) for log in self.logs],
                f,
                ensure_ascii=False,
                indent=2
            )


def timer_decorator(func):
    """
    计时装饰器

    用于测量函数执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[计时] {func.__name__} 执行耗时: {(end_time - start_time) * 1000:.2f}ms")
        return result
    return wrapper


# ============================================================
# 策略类定义（企业级插件式架构）
# ============================================================

@dataclass
class StrategyConfig:
    """
    策略配置类

    标准化策略的配置结构，
    支持动态策略管理和配置更新
    """
    name: str
    enabled: bool = True
    priority: int = 100  # 越小越优先
    timeout_ms: float = 1000.0  # 超时时间
    retry_count: int = 0  # 重试次数
    config: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """
    策略基类

    所有业务策略都应继承此类，
    保证策略接口一致性，便于：
    - 动态加载
    - 统一管理
    - 单元测试
    """

    def __init__(self, config: StrategyConfig):
        """
        初始化策略

        Args:
            config: 策略配置
        """
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.priority = config.priority

    def is_triggered(self, messages: List[Dict], sender: Agent) -> bool:
        """
        判断是否触发此策略

        Args:
            messages: 消息列表
            sender: 发送者

        Returns:
            bool: 是否触发
        """
        raise NotImplementedError("子类必须实现is_triggered方法")

    def execute(self, messages: List[Dict], sender: Agent) -> str:
        """
        执行策略

        Args:
            messages: 消息列表
            sender: 发送者

        Returns:
            str: 响应内容
        """
        raise NotImplementedError("子类必须实现execute方法")

    def should_fallback(self, exception: Exception) -> bool:
        """
        判断是否需要降级处理

        Args:
            exception: 发生的异常

        Returns:
            bool: 是否需要降级
        """
        return True  # 默认都需要降级


class KeywordStrategy(BaseStrategy):
    """
    关键词匹配策略

    基于关键词的快速响应策略，
    适用于FAQ、常见问题等场景
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.keywords = config.config.get("keywords", [])
        self.responses = config.config.get("responses", {})

    def is_triggered(self, messages: List[Dict], sender: Agent) -> bool:
        """检查是否匹配关键词"""
        if not messages:
            return False

        content = messages[-1].get("content", "").lower()
        return any(kw in content for kw in self.keywords)

    def execute(self, messages: List[Dict], sender: Agent) -> str:
        """返回匹配的响应"""
        content = messages[-1].get("content", "").lower()

        for keyword, response in self.responses.items():
            if keyword in content:
                return response

        return ""


class RegexStrategy(BaseStrategy):
    """
    正则匹配策略

    基于正则表达式的模式匹配策略，
    适用于结构化数据提取，如订单号、手机号等
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.pattern = re.compile(
            config.config.get("pattern", ""),
            config.config.get("flags", 0)
        )
        self.response_template = config.config.get("response_template", "{}")

    def is_triggered(self, messages: List[Dict], sender: Agent) -> bool:
        """检查是否匹配正则"""
        if not messages:
            return False

        content = messages[-1].get("content", "")
        return bool(self.pattern.search(content))

    def execute(self, messages: List[Dict], sender: Agent) -> str:
        """提取匹配内容并返回响应"""
        content = messages[-1].get("content", "")
        match = self.pattern.search(content)

        if match:
            matched_value = match.group()
            return self.response_template.format(matched_value)

        return ""


# ============================================================
# 企业级审计Agent
# ============================================================

class AuditedConversableAgent(ConversableAgent):
    """
    带审计功能的ConversableAgent

    企业级应用必备特性：
    1. 操作审计：记录所有交互
    2. 性能监控：追踪响应时间
    3. 错误记录：异常信息保存
    4. 报表生成：支持数据分析

    使用场景：
    - 金融、医疗等合规要求严格的行业
    - 客服系统的问题追踪和质量分析
    - 用户行为分析和个性化推荐
    """

    def __init__(self, name: str, audit_logger: AuditLogger = None, **kwargs):
        """
        初始化审计Agent

        Args:
            name: Agent名称
            audit_logger: 审计日志记录器（如果为None则创建默认）
            **kwargs: 传递给父类的参数
        """
        super().__init__(name, **kwargs)

        # 审计日志记录器
        self.audit_logger = audit_logger or AuditLogger()

        # 策略列表（企业级插件架构）
        self.strategies: List[BaseStrategy] = []

        # 注册默认的generate_reply hook用于统计
        # 注意：这里通过猴子补丁的方式在父类方法前后添加逻辑
        self._patch_generate_reply()

    def _patch_generate_reply(self):
        """
        修补generate_reply方法以添加审计功能

        通过保存原方法、创建包装函数、替换原方法的方式，
        在不修改父类源码的情况下添加审计能力
        """
        original_method = self.generate_reply

        def audited_generate_reply(messages, sender, **kwargs):
            # 记录开始时间
            start_time = time.time()

            # 调用原始方法
            try:
                result = original_method(messages, sender, **kwargs)
                return result
            finally:
                # 计算耗时
                elapsed_ms = (time.time() - start_time) * 1000

                # 记录审计日志
                if messages:
                    audit_log = AuditLog(
                        operation="generate_reply",
                        trigger_type="auto",
                        request_content=messages[-1].get("content", "")[:200],
                        response_content=str(result)[:200] if result else "",
                        processing_time_ms=elapsed_ms,
                        status="success"
                    )
                    self.audit_logger.log_operation(audit_log)

        # 替换原方法
        setattr(self, 'generate_reply', audited_generate_reply)

    def register_strategy(self, strategy: BaseStrategy):
        """
        注册策略（企业级插件架构）

        Args:
            strategy: 策略实例
        """
        self.strategies.append(strategy)
        # 按优先级排序
        self.strategies.sort(key=lambda s: s.priority)

    def unregister_strategy(self, strategy_name: str):
        """
        注销策略

        Args:
            strategy_name: 策略名称
        """
        self.strategies = [
            s for s in self.strategies
            if s.name != strategy_name
        ]

    def get_audit_report(self) -> Dict[str, Any]:
        """
        获取审计报告

        Returns:
            包含统计信息的字典
        """
        logs = self.audit_logger.logs

        if not logs:
            return {"total_operations": 0}

        # 计算统计数据
        total = len(logs)
        success = len([l for l in logs if l.status == "success"])
        errors = len([l for l in logs if l.status == "error"])
        avg_time = sum(l.processing_time_ms for l in logs) / total if total > 0 else 0

        # 按操作类型统计
        by_operation: Dict[str, int] = {}
        for log in logs:
            by_operation[log.operation] = by_operation.get(log.operation, 0) + 1

        return {
            "total_operations": total,
            "success_count": success,
            "error_count": errors,
            "success_rate": f"{success/total*100:.1f}%" if total > 0 else "N/A",
            "average_processing_time_ms": f"{avg_time:.2f}",
            "by_operation": by_operation,
            "logs_count": len(logs)
        }


# ============================================================
# 企业级客服Agent（完整示例）
# ============================================================

class EnterpriseCustomerServiceAgent(AuditedConversableAgent):
    """
    企业级客服Agent

    结合多种企业级设计模式：
    1. 插件式策略架构
    2. 完整的审计日志
    3. 多级降级机制
    4. 状态管理
    5. 性能监控

    适用场景：
    - 大型企业客服系统
    - 需要合规审计的系统
    - 需要数据分析的客服系统
    """

    def __init__(
        self,
        name: str,
        department: str = "general",
        session_timeout_minutes: int = 30,
        **kwargs
    ):
        """
        初始化企业级客服Agent

        Args:
            name: Agent名称
            department: 部门（用于路由）
            session_timeout_minutes: 会话超时时间
            **kwargs: 传递给父类的参数
        """
        super().__init__(name, **kwargs)

        # 业务配置
        self.department = department
        self.session_timeout_minutes = session_timeout_minutes

        # 会话状态管理
        self.session_data: Dict[str, Any] = {}
        self.last_interaction_time: float = time.time()

        # 初始化内置策略
        self._init_builtin_strategies()

        # 注册消息处理器
        self._register_handlers()

    def _init_builtin_strategies(self):
        """初始化内置策略"""

        # 1. 问候语策略
        greeting_strategy = KeywordStrategy(StrategyConfig(
            name="greeting",
            priority=0,
            config={
                "keywords": ["你好", "您好", "hello", "hi"],
                "responses": {
                    "你好": f"您好！欢迎来到{self.department}部门，请问有什么可以帮助您？",
                    "您好": f"您好！欢迎来到{self.department}部门，请问有什么可以帮助您？",
                    "hello": f"Hello! Welcome to {self.department} department, how can I help you?",
                    "hi": f"Hi! Welcome to {self.department} department, how can I help you?"
                }
            }
        ))
        self.register_strategy(greeting_strategy)

        # 2. 订单查询策略
        order_strategy = RegexStrategy(StrategyConfig(
            name="order_query",
            priority=10,
            config={
                "pattern": r"(订单号|order)[:：]?\s*(\d{10,})",
                "response_template": "您的订单 {} 状态查询中，请稍候..."
            }
        ))
        self.register_strategy(order_strategy)

        # 3. 手机号识别策略
        phone_strategy = RegexStrategy(StrategyConfig(
            name="phone_capture",
            priority=10,
            config={
                "pattern": r"1[3-9]\d{9}",
                "response_template": "已识别您的手机号：{}，我们将以此联系您。"
            }
        ))
        self.register_strategy(phone_strategy)

    def _register_handlers(self):
        """注册消息处理器"""

        # 注册策略执行处理器
        self.register_reply(
            trigger=None,
            reply_func=self._strategy_handler,
            position=0
        )

        # 注册会话超时检查
        self.register_reply(
            trigger=self._check_session_timeout,
            reply_func=self._session_timeout_handler,
            position=100
        )

    def _strategy_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        策略处理器 - 执行注册的策略

        Args:
            messages: 消息列表
            sender: 发送者
            config: 配置

        Returns:
            str: 策略执行结果
        """
        # 按优先级遍历策略
        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            try:
                if strategy.is_triggered(messages, sender):
                    result = strategy.execute(messages, sender)
                    if result:
                        # 更新会话状态
                        self.last_interaction_time = time.time()
                        return result
            except Exception as e:
                # 策略执行出错，记录日志并继续
                audit_log = AuditLog(
                    operation=f"strategy_{strategy.name}",
                    trigger_type="strategy",
                    request_content=messages[-1].get("content", "")[:200] if messages else "",
                    response_content="",
                    processing_time_ms=0,
                    status="error",
                    error_message=str(e)
                )
                self.audit_logger.log_operation(audit_log)

                # 如果策略需要降级则继续，否则返回错误
                if strategy.should_fallback(e):
                    continue

        # 没有策略匹配，返回空字符串让LLM处理
        return ""

    def _check_session_timeout(self, messages: List[Dict], sender: Agent) -> bool:
        """检查会话是否超时"""
        elapsed = time.time() - self.last_interaction_time
        return elapsed > (self.session_timeout_minutes * 60)

    def _session_timeout_handler(self, messages: List[Dict], sender: Agent, config: Dict) -> str:
        """会话超时处理器"""
        return f"您的会话已超时（超过{self.session_timeout_minutes}分钟），是否需要重新开始？"

    def set_session_data(self, key: str, value: Any):
        """
        设置会话数据

        Args:
            key: 数据键
            value: 数据值
        """
        self.session_data[key] = value
        self.last_interaction_time = time.time()

    def get_session_data(self, key: str, default: Any = None) -> Any:
        """
        获取会话数据

        Args:
            key: 数据键
            default: 默认值

        Returns:
            数据值
        """
        return self.session_data.get(key, default)

    def clear_session(self):
        """
        清除会话数据
        """
        self.session_data.clear()


# ============================================================
# 策略管理器（企业级策略管理模式）
# ============================================================

class StrategyManager:
    """
    策略管理器

    负责策略的：
    - 动态加载
    - 配置更新
    - 运行时切换
    - 优先级调整

    这是企业级Agent的重要基础设施
    """

    def __init__(self):
        """初始化策略管理器"""
        self.strategies: Dict[str, StrategyConfig] = {}
        self._strategy_instances: Dict[str, BaseStrategy] = {}

    def register_strategy_config(self, config: StrategyConfig):
        """
        注册策略配置

        Args:
            config: 策略配置
        """
        self.strategies[config.name] = config

    def update_strategy_config(self, name: str, **kwargs):
        """
        更新策略配置

        Args:
            name: 策略名称
            **kwargs: 要更新的配置项
        """
        if name in self.strategies:
            config = self.strategies[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    def enable_strategy(self, name: str):
        """启用策略"""
        if name in self.strategies:
            self.strategies[name].enabled = True

    def disable_strategy(self, name: str):
        """禁用策略"""
        if name in self.strategies:
            self.strategies[name].enabled = False

    def get_strategy_config(self, name: str) -> Optional[StrategyConfig]:
        """获取策略配置"""
        return self.strategies.get(name)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """列出所有策略"""
        return [
            {
                "name": name,
                "enabled": config.enabled,
                "priority": config.priority
            }
            for name, config in self.strategies.items()
        ]


# ============================================================
# 演示函数
# ============================================================

def demo_audit_logger():
    """
    演示审计日志记录器
    """
    print("=" * 60)
    print("演示1：审计日志记录器")
    print("=" * 60)

    # 创建审计日志记录器
    audit_logger = AuditLogger("demo_audit.log")

    # 模拟记录一些操作
    operations = [
        AuditLog(
            operation="generate_reply",
            trigger_type="keyword",
            request_content="你好",
            response_content="您好！欢迎光临",
            processing_time_ms=5.23,
            status="success"
        ),
        AuditLog(
            operation="generate_reply",
            trigger_type="regex",
            request_content="订单号: 1234567890123",
            response_content="订单查询中...",
            processing_time_ms=12.45,
            status="success"
        ),
        AuditLog(
            operation="generate_reply",
            trigger_type="func",
            request_content="无效消息",
            response_content="",
            processing_time_ms=0.5,
            status="error",
            error_message="策略执行异常"
        )
    ]

    for op in operations:
        audit_logger.log_operation(op)

    print(f"\n已记录 {len(audit_logger.logs)} 条操作日志")

    # 获取审计报告
    print("\n审计报告:")
    print(f"- 总操作数: {len(audit_logger.logs)}")
    print(f"- 成功: {len([l for l in audit_logger.logs if l.status == 'success'])}")
    print(f"- 失败: {len([l for l in audit_logger.logs if l.status == 'error'])}")

    # 查询特定日志
    error_logs = audit_logger.get_logs(status="error")
    print(f"\n错误日志: {len(error_logs)} 条")


def demo_strategy_manager():
    """
    演示策略管理器
    """
    print("\n" + "=" * 60)
    print("演示2：策略管理器")
    print("=" * 60)

    manager = StrategyManager()

    # 注册策略配置
    manager.register_strategy_config(StrategyConfig(
        name="greeting",
        priority=0,
        config={
            "keywords": ["你好", "hello"],
            "responses": {"你好": "您好！"}
        }
    ))

    manager.register_strategy_config(StrategyConfig(
        name="order_query",
        priority=10,
        config={
            "pattern": r"订单号:\d{10,}",
            "response_template": "订单查询结果"
        }
    ))

    print("\n已注册策略:")
    for s in manager.list_strategies():
        print(f"  - {s['name']}: 优先级={s['priority']}, 启用={s['enabled']}")

    # 禁用某个策略
    manager.disable_strategy("order_query")
    print("\n禁用order_query后的状态:")
    for s in manager.list_strategies():
        print(f"  - {s['name']}: 启用={s['enabled']}")

    # 更新优先级
    manager.update_strategy_config("greeting", priority=5)
    print("\n更新greeting优先级后:")
    for s in manager.list_strategies():
        print(f"  - {s['name']}: 优先级={s['priority']}")


def demo_enterprise_agent():
    """
    演示企业级客服Agent
    """
    print("\n" + "=" * 60)
    print("演示3：企业级客服Agent")
    print("=" * 60)

    # 创建企业级客服Agent
    agent = EnterpriseCustomerServiceAgent(
        name="企业客服",
        department="客户服务中心",
        llm_config=False  # 不使用LLM进行演示
    )

    # 模拟用户交互
    test_messages = [
        {"content": "你好", "role": "user"},
        {"content": "订单号: 1234567890123456", "role": "user"},
        {"content": "我的手机号是13812345678", "role": "user"},
        {"content": "你们公司在哪里", "role": "user"}
    ]

    print("\n--- 交互测试 ---")
    for msg in test_messages:
        response = agent.generate_reply(
            messages=[msg],
            sender=None
        )
        print(f"用户: {msg['content']}")
        print(f"Agent: {response}")
        print()

    # 演示会话数据管理
    print("\n--- 会话数据管理 ---")
    agent.set_session_data("user_id", "U12345")
    agent.set_session_data("user_name", "张三")
    print(f"用户ID: {agent.get_session_data('user_id')}")
    print(f"用户名: {agent.get_session_data('user_name')}")

    # 获取审计报告
    print("\n--- 审计报告 ---")
    report = agent.get_audit_report()
    for key, value in report.items():
        print(f"  {key}: {value}")


def demo_plugin_architecture():
    """
    演示插件式架构
    """
    print("\n" + "=" * 60)
    print("演示4：插件式架构")
    print("=" * 60)

    # 创建Agent
    agent = AuditedConversableAgent(
        name="插件测试",
        llm_config=False
    )

    # 动态注册自定义策略
    class CustomStrategy(BaseStrategy):
        """自定义策略示例"""

        def __init__(self):
            super().__init__(StrategyConfig(
                name="custom",
                priority=50,
                config={"trigger_word": "自定义"}
            ))
            self.trigger_word = self.config.get("trigger_word", "自定义")

        def is_triggered(self, messages: List[Dict], sender: Agent) -> bool:
            if not messages:
                return False
            content = messages[-1].get("content", "")
            return self.trigger_word in content

        def execute(self, messages: List[Dict], sender: Agent) -> str:
            return "这是自定义策略的响应！"

    # 注册插件
    custom_strategy = CustomStrategy()
    agent.register_strategy(custom_strategy)
    print(f"已注册策略: {custom_strategy.name}")

    # 测试插件
    response = agent.generate_reply(
        messages=[{"content": "测试自定义关键词", "role": "user"}],
        sender=None
    )
    print(f"自定义策略触发: {response}")

    # 注销插件
    agent.unregister_strategy("custom")
    print("已注销自定义策略")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AutoGen企业级自定义Agent设计模式 - 完整演示")
    print("=" * 60)

    # 运行所有演示
    demo_audit_logger()
    demo_strategy_manager()
    demo_enterprise_agent()
    demo_plugin_architecture()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n企业级设计模式总结:")
    print("1. 插件式架构 - StrategyManager管理策略的生命周期")
    print("2. 审计日志 - 完整的操作记录和报表生成")
    print("3. 策略基类 - 统一的策略接口，便于扩展")
    print("4. 配置驱动 - StrategyConfig标准化配置结构")
    print("5. 状态管理 - 会话数据的存储和恢复")