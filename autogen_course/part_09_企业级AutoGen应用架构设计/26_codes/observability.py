"""
AutoGen可观测性设计示例
=====================================

本文件展示AutoGen应用的可观测性设计核心组件。

核心组件：
1. 对话链路追踪（ConversationTracer）：记录完整的多Agent对话流程
2. Token消耗监控（TokenMonitor）：追踪每个租户、每个模型的Token消耗
3. 异常告警系统（AlertManager）：支持多级告警、告警聚合与抑制
4. Prometheus监控指标暴露（PrometheusMetrics）：将指标暴露给Prometheus进行采集

设计理念：
- 可观测性三支柱：日志（Logs）、指标（Metrics）、追踪（Traces）
- 零侵入设计：通过装饰器和钩子实现最小化代码改动
- 多租户支持：所有指标和日志按租户隔离

作者：AutoGen学习课程
版本：1.0
"""

import asyncio
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps

# 尝试导入prometheus_client，如果不存在则提供优雅降级
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # 如果Prometheus客户端不可用，创建空实现
    class CollectorRegistry:
        pass
    def Counter(*args, **kwargs):
        return _DummyMetric()
    def Histogram(*args, **kwargs):
        return _DummyMetric()
    def Gauge(*args, **kwargs):
        return _DummyMetric()
    class _DummyMetric:
        def labels(self, **kwargs): return self
        def observe(self, *args): pass
        def inc(self, *args): pass
        def dec(self, *args): pass
        def set(self, *args): pass


# =============================================================================
# 第一部分：核心数据结构和枚举定义
# =============================================================================

class AlertLevel(Enum):
    """
    告警级别枚举

    定义从低到高的五个告警级别，每个级别对应不同的处理策略
    """
    DEBUG = "debug"           # 调试级：仅记录，不触发任何动作
    INFO = "info"             # 信息级：记录重要事件，如Agent启动/停止
    WARNING = "warning"        # 警告级：潜在问题，如Token使用量超过50%
    ERROR = "error"           # 错误级：一般错误，如LLM调用超时
    CRITICAL = "critical"     # 严重级：系统不可用，如连续失败超过阈值


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"       # 计数器：只增不减，如请求总数
    GAUGE = "gauge"           # 仪表：可增可减，如当前活跃请求数
    HISTOGRAM = "histogram"   # 直方图：分布统计，如响应时间分布


@dataclass
class AlertRule:
    """
    告警规则定义

    用于配置告警的触发条件和处理方式
    """
    name: str                          # 规则名称
    metric_name: str                  # 关联的指标名称
    condition: str                     # 触发条件，如 " > 100" 或 " == 0"
    threshold: float                   # 阈值
    level: AlertLevel                  # 告警级别
    cooldown_seconds: int = 60         # 冷却时间（秒），防止告警风暴
    enabled: bool = True               # 是否启用


@dataclass
class Alert:
    """
    告警实例

    表示一次具体的告警事件
    """
    alert_id: str                     # 唯一标识
    rule_name: str                    # 触发的规则名称
    level: AlertLevel                  # 告警级别
    message: str                      # 告警消息
    metric_value: float                # 触发时的指标值
    tenant_id: Optional[str] = None    # 租户ID（如果有）
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False          # 是否已确认


@dataclass
class TokenRecord:
    """
    Token使用记录

    记录单次LLM调用的Token消耗详情
    """
    record_id: str                    # 记录ID
    tenant_id: str                   # 租户ID
    conversation_id: str              # 对话ID
    agent_name: str                  # Agent名称
    model_name: str                  # 模型名称
    prompt_tokens: int               # 输入Token数
    completion_tokens: int           # 输出Token数
    total_tokens: int                # 总Token数
    cost: float = 0.0                # 这次调用的成本
    latency_ms: float = 0.0           # 延迟（毫秒）
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# 第二部分：对话链路追踪器
# =============================================================================

class ConversationTracer:
    """
    对话链路追踪器

    实现AutoGen应用的全链路追踪，支持：
    1. 嵌套追踪：支持Agent嵌套调用，通过parent_trace_id串联
    2. 并发追踪：支持多线程/多协程并发执行，每条链路独立追踪
    3. 性能分析：记录每个环节的耗时，便于性能优化
    4. 错误追踪：记录失败信息，便于问题排查

    追踪数据结构：
    - trace_id: 全局唯一追踪ID
    - parent_trace_id: 父追踪ID（用于嵌套对话）
    - span_id: 当前环节ID
    - conversation_id: 对话会话ID

    使用示例：
        tracer = ConversationTracer()

        # 开始根追踪
        root_trace = tracer.start_trace(
            tenant_id="tenant_001",
            conversation_id="conv_001",
            agent_name="code_assistant",
            input_message="帮我写一个排序算法"
        )

        # 在嵌套Agent中创建子追踪
        child_trace = tracer.start_trace(
            tenant_id="tenant_001",
            conversation_id="conv_001",
            agent_name="code_reviewer",
            parent_trace_id=root_trace.trace_id,
            input_message="审查代码"
        )

        # 结束追踪
        tracer.end_trace(child_trace.trace_id, "代码审查完成", 150)
        tracer.end_trace(root_trace.trace_id, "排序算法已生成", 2000)
    """

    def __init__(self, max_traces: int = 10000):
        """
        初始化追踪器

        Args:
            max_traces: 最大保存的追踪数量，超过后自动清理旧数据
        """
        self.max_traces = max_traces
        # 追踪记录存储: trace_id -> ConversationTrace
        self.traces: Dict[str, 'ConversationTrace'] = {}
        # 租户追踪索引: tenant_id -> [trace_ids]
        self.tenant_traces: Dict[str, List[str]] = defaultdict(list)
        # 对话追踪索引: conversation_id -> [trace_ids]
        self.conversation_traces: Dict[str, List[str]] = defaultdict(list)
        # 当前活跃的追踪（用于嵌套场景）
        self.active_traces: Dict[str, str] = {}  # thread_id -> trace_id

        # 统计指标
        self.total_traces_started = 0
        self.total_traces_completed = 0
        self.total_traces_failed = 0

    def start_trace(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        input_message: str,
        parent_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'ConversationTrace':
        """
        开始一个追踪

        Args:
            tenant_id: 租户ID
            conversation_id: 对话会话ID
            agent_name: Agent名称
            input_message: 输入消息
            parent_trace_id: 父追踪ID（用于嵌套对话）
            metadata: 额外的元数据

        Returns:
            追踪记录对象
        """
        trace_id = self._generate_trace_id()

        # 创建追踪记录
        trace = ConversationTrace(
            trace_id=trace_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            parent_trace_id=parent_trace_id,
            input_message=self._truncate(input_message, 500),
            metadata=metadata or {}
        )

        # 存储追踪记录
        self.traces[trace_id] = trace
        self.tenant_traces[tenant_id].append(trace_id)
        self.conversation_traces[conversation_id].append(trace_id)

        # 更新统计
        self.total_traces_started += 1

        # 清理超过最大数量的旧追踪
        self._cleanup_if_needed()

        # 记录日志
        logging.info(
            f"[追踪] 开始: trace_id={trace_id}, agent={agent_name}, "
            f"parent={parent_trace_id or 'None'}"
        )

        return trace

    def end_trace(
        self,
        trace_id: str,
        output_message: str,
        token_used: int,
        status: str = "completed"
    ):
        """
        结束一个追踪

        Args:
            trace_id: 追踪ID
            output_message: 输出消息
            token_used: Token消耗
            status: 最终状态（completed/failed/stopped）
        """
        if trace_id not in self.traces:
            logging.warning(f"[追踪] 尝试结束未知追踪: {trace_id}")
            return

        trace = self.traces[trace_id]
        trace.end_time = datetime.now()
        trace.output_message = self._truncate(output_message, 500)
        trace.token_used = token_used
        trace.status = status
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

        # 更新统计
        if status == "completed":
            self.total_traces_completed += 1
        else:
            self.total_traces_failed += 1

        logging.info(
            f"[追踪] 结束: trace_id={trace_id}, status={status}, "
            f"duration={trace.duration_ms:.2f}ms, tokens={token_used}"
        )

    def fail_trace(self, trace_id: str, error: str, error_type: Optional[str] = None):
        """
        标记追踪失败

        Args:
            trace_id: 追踪ID
            error: 错误信息
            error_type: 错误类型（可选）
        """
        if trace_id not in self.traces:
            logging.warning(f"[追踪] 尝试标记未知追踪失败: {trace_id}")
            return

        trace = self.traces[trace_id]
        trace.end_time = datetime.now()
        trace.status = "failed"
        trace.error = error
        trace.error_type = error_type or "UnknownError"
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

        self.total_traces_failed += 1

        logging.error(
            f"[追踪] 失败: trace_id={trace_id}, error={error}, "
            f"error_type={error_type}"
        )

    def get_trace(self, trace_id: str) -> Optional['ConversationTrace']:
        """获取追踪记录"""
        return self.traces.get(trace_id)

    def get_conversation_traces(
        self,
        conversation_id: str,
        include_children: bool = True
    ) -> List['ConversationTrace']:
        """
        获取对话的所有追踪记录

        Args:
            conversation_id: 对话会话ID
            include_children: 是否包含子追踪

        Returns:
            按时间排序的追踪记录列表
        """
        trace_ids = self.conversation_traces.get(conversation_id, [])
        traces = [self.traces[tid] for tid in trace_ids if tid in self.traces]

        if not include_children:
            # 只返回根追踪（没有父追踪的）
            traces = [t for t in traces if not t.parent_trace_id]

        return sorted(traces, key=lambda x: x.start_time)

    def get_trace_tree(self, conversation_id: str) -> Dict[str, Any]:
        """
        获取追踪树（用于可视化）

        将扁平的所有追踪记录构建成树形结构，便于理解嵌套关系

        Args:
            conversation_id: 对话会话ID

        Returns:
            树形结构的追踪数据
        """
        traces = self.get_conversation_traces(conversation_id, include_children=True)

        # 构建ID到追踪记录的映射
        trace_map = {t.trace_id: t for t in traces}

        # 找出根追踪
        roots = [t for t in traces if not t.parent_trace_id]

        def build_tree(trace: 'ConversationTrace') -> Dict[str, Any]:
            """递归构建子树"""
            children = [
                build_tree(trace_map[child_id])
                for child_id in trace_map
                if trace_map[child_id].parent_trace_id == trace.trace_id
            ]
            return {
                "trace_id": trace.trace_id,
                "agent_name": trace.agent_name,
                "status": trace.status,
                "duration_ms": trace.duration_ms,
                "token_used": trace.token_used,
                "start_time": trace.start_time.isoformat(),
                "children": children
            }

        return {
            "conversation_id": conversation_id,
            "total_traces": len(traces),
            "roots": [build_tree(root) for root in roots]
        }

    def _generate_trace_id(self) -> str:
        """生成唯一的追踪ID"""
        return uuid.uuid4().hex[:16]

    def _truncate(self, text: str, max_length: int) -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _cleanup_if_needed(self):
        """清理超过最大数量的旧追踪"""
        if len(self.traces) > self.max_traces:
            # 删除最旧的追踪
            sorted_traces = sorted(self.traces.items(), key=lambda x: x[1].start_time)
            to_remove = len(self.traces) - self.max_traces
            for trace_id, _ in sorted_traces[:to_remove]:
                trace = self.traces[trace_id]
                # 从索引中移除
                if trace.tenant_id in self.tenant_traces:
                    self.tenant_traces[trace.tenant_id].remove(trace_id)
                if trace.conversation_id in self.conversation_traces:
                    self.conversation_traces[trace.conversation_id].remove(trace_id)
                del self.traces[trace_id]


@dataclass
class ConversationTrace:
    """
    对话链路追踪记录

    用于记录单个Agent执行单元的详细信息
    """
    trace_id: str                    # 唯一追踪ID
    tenant_id: str                   # 租户ID
    conversation_id: str             # 对话会话ID
    agent_name: str                  # 当前Agent名称
    parent_trace_id: Optional[str] = None  # 父追踪ID
    input_message: str = ""          # 输入消息
    output_message: str = ""         # 输出消息
    token_used: int = 0              # Token消耗
    status: str = "started"          # 状态：started/running/completed/failed/stopped
    error: str = ""                  # 错误信息
    error_type: str = ""             # 错误类型
    duration_ms: float = 0.0         # 执行时长（毫秒）
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 第三部分：Token消耗监控器
# =============================================================================

class TokenMonitor:
    """
    Token消耗监控器

    监控和统计LLM的Token消耗情况，支持：
    1. 租户级别统计：每个租户的Token使用量
    2. 模型级别统计：每个模型的调用次数和Token量
    3. 实时成本计算：根据模型价格实时计算成本
    4. 配额管理：监控租户配额使用情况，支持配额告警

    数据存储：
    - 按租户聚合的统计数据
    - 按时间序列的详细记录
    - 配额使用进度

    使用示例：
        monitor = TokenMonitor()

        # 记录一次LLM调用
        monitor.record(
            tenant_id="tenant_001",
            conversation_id="conv_001",
            agent_name="code_assistant",
            model_name="gpt-4",
            prompt_tokens=100,
            completion_tokens=200,
            latency_ms=1500
        )

        # 获取租户统计
        stats = monitor.get_tenant_stats("tenant_001")
        print(f"总请求数: {stats['total_requests']}")
        print(f"总Token: {stats['total_tokens']}")
        print(f"总成本: ${stats['total_cost']:.4f}")

        # 获取配额使用情况
        quota = monitor.get_quota_usage("tenant_001")
        print(f"月度配额: {quota['monthly_limit']}")
        print(f"已使用: {quota['monthly_used']} ({quota['usage_percent']:.1f}%)")
    """

    def __init__(self):
        # 租户统计: tenant_id -> TokenStats
        self.tenant_stats: Dict[str, 'TokenStats'] = {}
        # 模型统计: model_name -> ModelStats
        self.model_stats: Dict[str, 'ModelStats'] = defaultdict(lambda: ModelStats())
        # Token记录详情（用于详细分析）: record_id -> TokenRecord
        self.records: Dict[str, TokenRecord] = {}
        # 配额配置: tenant_id -> QuotaConfig
        self.quota_configs: Dict[str, 'QuotaConfig'] = {}
        # 模型价格配置: model_name -> (input_price, output_price) per 1M tokens
        self.model_prices: Dict[str, tuple[float, float]] = defaultdict(
            lambda: (5.0, 15.0)  # 默认GPT-4价格
        )

    def record(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0
    ):
        """
        记录一次Token使用

        Args:
            tenant_id: 租户ID
            conversation_id: 对话ID
            agent_name: Agent名称
            model_name: 模型名称
            prompt_tokens: 输入Token数
            completion_tokens: 输出Token数
            latency_ms: 延迟（毫秒）
        """
        total_tokens = prompt_tokens + completion_tokens

        # 创建记录
        record = TokenRecord(
            record_id=uuid.uuid4().hex[:12],
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost=self._calculate_cost(model_name, prompt_tokens, completion_tokens)
        )

        # 存储记录
        self.records[record.record_id] = record

        # 更新租户统计
        self._update_tenant_stats(tenant_id, total_tokens, record.cost, latency_ms)

        # 更新模型统计
        self._update_model_stats(model_name, prompt_tokens, completion_tokens, latency_ms)

        logging.debug(
            f"[Token监控] 记录: tenant={tenant_id}, model={model_name}, "
            f"tokens={total_tokens}, cost=${record.cost:.4f}"
        )

    def _calculate_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """计算成本"""
        input_price, output_price = self.model_prices[model_name]
        prompt_cost = prompt_tokens / 1_000_000 * input_price
        completion_cost = completion_tokens / 1_000_000 * output_price
        return prompt_cost + completion_cost

    def _update_tenant_stats(
        self,
        tenant_id: str,
        tokens: int,
        cost: float,
        latency_ms: float
    ):
        """更新租户统计"""
        if tenant_id not in self.tenant_stats:
            self.tenant_stats[tenant_id] = TokenStats()

        stats = self.tenant_stats[tenant_id]
        stats.total_requests += 1
        stats.total_tokens += tokens
        stats.total_cost += cost
        stats.total_latency_ms += latency_ms

        # 更新今天和本月的统计
        now = datetime.now()
        today_key = now.date().isoformat()
        month_key = f"{now.year}-{now.month:02d}"

        stats.daily_tokens.setdefault(today_key, 0)
        stats.daily_tokens[today_key] += tokens

        stats.monthly_tokens.setdefault(month_key, 0)
        stats.monthly_tokens[month_key] += tokens

    def _update_model_stats(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ):
        """更新模型统计"""
        stats = self.model_stats[model_name]
        stats.total_requests += 1
        stats.total_prompt_tokens += prompt_tokens
        stats.total_completion_tokens += prompt_tokens + completion_tokens
        stats.total_latency_ms += latency_ms
        stats.request_latencies.append(latency_ms)

    def set_model_price(self, model_name: str, input_price: float, output_price: float):
        """
        设置模型价格

        Args:
            model_name: 模型名称
            input_price: 输入价格（每百万Token）
            output_price: 输出价格（每百万Token）
        """
        self.model_prices[model_name] = (input_price, output_price)
        logging.info(f"[Token监控] 设置模型价格: {model_name} = ${input_price}/${output_price}")

    def set_quota(self, tenant_id: str, monthly_limit: int, daily_limit: int):
        """
        设置租户配额

        Args:
            tenant_id: 租户ID
            monthly_limit: 月度Token限额
            daily_limit: 每日Token限额
        """
        self.quota_configs[tenant_id] = QuotaConfig(
            tenant_id=tenant_id,
            monthly_limit=monthly_limit,
            daily_limit=daily_limit
        )
        logging.info(
            f"[Token监控] 设置配额: tenant={tenant_id}, "
            f"monthly={monthly_limit}, daily={daily_limit}"
        )

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        获取租户统计

        Args:
            tenant_id: 租户ID

        Returns:
            租户统计数据字典
        """
        stats = self.tenant_stats.get(tenant_id)
        if not stats:
            return {
                "tenant_id": tenant_id,
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0
            }

        avg_latency = (
            stats.total_latency_ms / stats.total_requests
            if stats.total_requests > 0 else 0.0
        )

        return {
            "tenant_id": tenant_id,
            "total_requests": stats.total_requests,
            "total_tokens": stats.total_tokens,
            "total_cost": round(stats.total_cost, 4),
            "avg_latency_ms": round(avg_latency, 2)
        }

    def get_quota_usage(self, tenant_id: str) -> Dict[str, Any]:
        """
        获取租户配额使用情况

        Args:
            tenant_id: 租户ID

        Returns:
            配额使用情况字典
        """
        config = self.quota_configs.get(tenant_id)
        if not config:
            return {
                "tenant_id": tenant_id,
                "has_quota": False
            }

        stats = self.tenant_stats.get(tenant_id, TokenStats())
        now = datetime.now()
        today_key = now.date().isoformat()
        month_key = f"{now.year}-{now.month:02d}"

        daily_used = stats.daily_tokens.get(today_key, 0)
        monthly_used = stats.monthly_tokens.get(month_key, 0)

        return {
            "tenant_id": tenant_id,
            "has_quota": True,
            "monthly_limit": config.monthly_limit,
            "monthly_used": monthly_used,
            "monthly_remaining": max(0, config.monthly_limit - monthly_used),
            "usage_percent": round(monthly_used / config.monthly_limit * 100, 2) if config.monthly_limit > 0 else 0,
            "daily_limit": config.daily_limit,
            "daily_used": daily_used,
            "daily_remaining": max(0, config.daily_limit - daily_used),
            "daily_usage_percent": round(daily_used / config.daily_limit * 100, 2) if config.daily_limit > 0 else 0
        }

    def get_all_tenant_stats(self) -> List[Dict[str, Any]]:
        """获取所有租户的统计"""
        return [
            self.get_tenant_stats(tenant_id)
            for tenant_id in self.tenant_stats.keys()
        ]


@dataclass
class TokenStats:
    """Token统计数据聚合"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    daily_tokens: Dict[str, int] = field(default_factory=dict)
    monthly_tokens: Dict[str, int] = field(default_factory=dict)


@dataclass
class ModelStats:
    """模型级别统计"""
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    request_latencies: List[float] = field(default_factory=list)


@dataclass
class QuotaConfig:
    """配额配置"""
    tenant_id: str
    monthly_limit: int
    daily_limit: int


# =============================================================================
# 第四部分：异常告警系统
# =============================================================================

class AlertManager:
    """
    异常告警管理系统

    实现多级告警、告警聚合、告警抑制和告警升级机制：

    告警级别：
    - DEBUG: 调试信息，不触发实际告警
    - INFO: 重要事件通知
    - WARNING: 潜在问题，需要关注
    - ERROR: 一般错误，需要处理
    - CRITICAL: 严重问题，系统不可用

    告警策略：
    1. 聚合：将短时间内的多个相同告警合并为一个
    2. 抑制：当高级别告警发生时，自动抑制低级别告警
    3. 升级：如果告警未及时处理，自动升级到更高级别
    4. 冷却：触发后进入冷却期，防止告警风暴

    使用示例：
        alert_manager = AlertManager()

        # 配置告警规则
        alert_manager.add_rule(AlertRule(
            name="high_token_usage",
            metric_name="tenant_tokens_daily",
            condition=">=",
            threshold=80000,  # 每日Token使用超过80%时告警
            level=AlertLevel.WARNING
        ))

        # 触发告警
        alert_manager.trigger(
            rule_name="high_token_usage",
            metric_value=85000,
            tenant_id="tenant_001",
            message="租户 tenant_001 今日Token使用量达到85000，超过80%阈值"
        )
    """

    def __init__(self):
        # 告警规则: rule_name -> AlertRule
        self.rules: Dict[str, AlertRule] = {}
        # 活跃告警: alert_id -> Alert
        self.active_alerts: Dict[str, Alert] = {}
        # 告警历史: alert_id -> Alert
        self.alert_history: List[Alert] = []
        # 告警处理器回调: level -> [callback_funcs]
        self.handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        # 冷却期追踪: rule_name -> last_trigger_time
        self.cooldown_tracker: Dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule):
        """
        添加告警规则

        Args:
            rule: 告警规则对象
        """
        self.rules[rule.name] = rule
        logging.info(f"[告警] 添加规则: {rule.name}, level={rule.level.value}")

    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logging.info(f"[告警] 移除规则: {rule_name}")

    def enable_rule(self, rule_name: str):
        """启用告警规则"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logging.info(f"[告警] 启用规则: {rule_name}")

    def disable_rule(self, rule_name: str):
        """禁用告警规则"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logging.info(f"[告警] 禁用规则: {rule_name}")

    def register_handler(self, level: AlertLevel, handler: Callable):
        """
        注册告警处理器

        Args:
            level: 处理的告警级别
            handler: 回调函数，签名为 (alert: Alert) -> None
        """
        self.handlers[level].append(handler)
        logging.info(f"[告警] 注册处理器: level={level.value}")

    def trigger(
        self,
        rule_name: str,
        metric_value: float,
        tenant_id: Optional[str] = None,
        message: Optional[str] = None
    ) -> Optional[Alert]:
        """
        触发告警

        检查规则并触发告警，支持冷却期控制

        Args:
            rule_name: 规则名称
            metric_value: 当前指标值
            tenant_id: 租户ID（可选）
            message: 自定义告警消息（可选）

        Returns:
            如果触发成功返回Alert对象，否则返回None
        """
        rule = self.rules.get(rule_name)
        if not rule:
            logging.warning(f"[告警] 规则不存在: {rule_name}")
            return None

        if not rule.enabled:
            logging.debug(f"[告警] 规则已禁用: {rule_name}")
            return None

        # 检查冷却期
        if self._is_in_cooldown(rule_name, rule.cooldown_seconds):
            logging.debug(f"[告警] 规则在冷却期: {rule_name}")
            return None

        # 检查触发条件
        if not self._check_condition(rule, metric_value):
            logging.debug(
                f"[告警] 条件不满足: {rule_name}, value={metric_value}, "
                f"condition={rule.condition} {rule.threshold}"
            )
            return None

        # 创建告警
        alert = Alert(
            alert_id=uuid.uuid4().hex[:12],
            rule_name=rule_name,
            level=rule.level,
            message=message or self._generate_message(rule, metric_value),
            metric_value=metric_value,
            tenant_id=tenant_id
        )

        # 存储告警
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # 更新冷却期
        self.cooldown_tracker[rule_name] = datetime.now()

        # 调用处理器
        self._dispatch_alert(alert)

        return alert

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID

        Returns:
            是否成功确认
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logging.info(f"[告警] 确认告警: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID

        Returns:
            是否成功解决
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            logging.info(f"[告警] 解决告警: {alert_id}")
            return True
        return False

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        tenant_id: Optional[str] = None
    ) -> List[Alert]:
        """
        获取活跃告警

        Args:
            level: 按级别过滤（可选）
            tenant_id: 按租户过滤（可选）

        Returns:
            告警列表
        """
        alerts = list(self.active_alerts.values())

        if level:
            alerts = [a for a in alerts if a.level == level]

        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]

        return sorted(alerts, key=lambda x: x.level.value, reverse=True)

    def _check_condition(self, rule: AlertRule, metric_value: float) -> bool:
        """检查条件是否满足"""
        condition_map = {
            ">": lambda v, t: v > t,
            ">=": lambda v, t: v >= t,
            "<": lambda v, t: v < t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t
        }
        condition_func = condition_map.get(rule.condition)
        if not condition_func:
            logging.warning(f"[告警] 未知条件: {rule.condition}")
            return False
        return condition_func(metric_value, rule.threshold)

    def _is_in_cooldown(self, rule_name: str, cooldown_seconds: int) -> bool:
        """检查是否在冷却期"""
        if rule_name not in self.cooldown_tracker:
            return False
        last_trigger = self.cooldown_tracker[rule_name]
        elapsed = (datetime.now() - last_trigger).total_seconds()
        return elapsed < cooldown_seconds

    def _generate_message(self, rule: AlertRule, metric_value: float) -> str:
        """生成告警消息"""
        return (
            f"告警规则 '{rule.name}' 触发: "
            f"指标 {rule.metric_name} = {metric_value}, "
            f"条件 '{rule.condition}' {rule.threshold}"
        )

    def _dispatch_alert(self, alert: Alert):
        """分发告警到处理器"""
        # 获取该级别及更高级别的处理器
        levels_to_notify = [
            AlertLevel.DEBUG, AlertLevel.INFO, AlertLevel.WARNING,
            AlertLevel.ERROR, AlertLevel.CRITICAL
        ]
        level_index = levels_to_notify.index(alert.level)

        for i in range(level_index, len(levels_to_notify)):
            level = levels_to_notify[i]
            for handler in self.handlers.get(level, []):
                try:
                    handler(alert)
                except Exception as e:
                    logging.error(f"[告警] 处理器执行失败: {e}")


# =============================================================================
# 第五部分：Prometheus监控指标暴露
# =============================================================================

class PrometheusMetrics:
    """
    Prometheus监控指标暴露器

    将AutoGen应用的各类指标暴露给Prometheus进行采集：

    指标类型：
    1. Counter（计数器）：请求总数、成功次数、失败次数
    2. Gauge（仪表）：当前活跃请求数、Agent数量
    3. Histogram（直方图）：响应时间分布、Token消耗分布

    核心指标：
    - autogen_requests_total: 总请求数（按tenant_id, agent_name, status分组）
    - autogen_request_duration_seconds: 请求处理时长分布
    - autogen_tokens_total: 总Token消耗（按tenant_id, model分组）
    - autogen_active_conversations: 当前活跃对话数
    - autogen_agent_errors_total: Agent错误总数（按agent_name, error_type分组）

    使用示例：
        metrics = PrometheusMetrics()

        # 记录请求
        metrics.record_request(
            tenant_id="tenant_001",
            agent_name="code_assistant",
            status="success",
            duration_ms=1500
        )

        # 记录Token
        metrics.record_tokens(
            tenant_id="tenant_001",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=200
        )

        # 获取Prometheus格式的指标
        metrics_output = metrics.get_metrics()

        # Flask/Gin等Web框架集成示例：
        # @app.route('/metrics')
        # def metrics():
        #     return metrics_output
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        初始化Prometheus指标

        Args:
            registry: Prometheus注册器，如果为None则使用默认注册器
        """
        if not PROMETHEUS_AVAILABLE:
            logging.warning(
                "[Prometheus] prometheus_client未安装，指标将仅记录到日志"
            )

        self.registry = registry

        # 创建指标定义
        self._create_metrics()

    def _create_metrics(self):
        """创建Prometheus指标"""
        # 请求计数器
        self.requests_total = Counter(
            "autogen_requests_total",
            "AutoGen总请求数",
            ["tenant_id", "agent_name", "status"]
        )

        # 请求时长直方图
        self.request_duration = Histogram(
            "autogen_request_duration_seconds",
            "AutoGen请求处理时长",
            ["tenant_id", "agent_name"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )

        # Token计数器
        self.tokens_total = Counter(
            "autogen_tokens_total",
            "AutoGen总Token消耗",
            ["tenant_id", "model", "token_type"]
        )

        # 当前活跃请求仪表
        self.active_requests = Gauge(
            "autogen_active_requests",
            "当前活跃请求数",
            ["tenant_id", "agent_name"]
        )

        # Agent错误计数器
        self.agent_errors = Counter(
            "autogen_agent_errors_total",
            "Agent错误总数",
            ["tenant_id", "agent_name", "error_type"]
        )

        # 对话数仪表
        self.active_conversations = Gauge(
            "autogen_active_conversations",
            "当前活跃对话数",
            ["tenant_id"]
        )

        # LLM调用时长直方图
        self.llm_duration = Histogram(
            "autogen_llm_duration_seconds",
            "LLM调用时长",
            ["tenant_id", "model"],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )

    def record_request(
        self,
        tenant_id: str,
        agent_name: str,
        status: str,
        duration_ms: float
    ):
        """
        记录请求

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称
            status: 请求状态（success/failed/timeout）
            duration_ms: 处理时长（毫秒）
        """
        # 计数器 +1
        self.requests_total.labels(
            tenant_id=tenant_id,
            agent_name=agent_name,
            status=status
        ).inc()

        # 时长直方图记录
        self.request_duration.labels(
            tenant_id=tenant_id,
            agent_name=agent_name
        ).observe(duration_ms / 1000)  # 转换为秒

    def record_tokens(
        self,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """
        记录Token消耗

        Args:
            tenant_id: 租户ID
            model: 模型名称
            prompt_tokens: 输入Token数
            completion_tokens: 输出Token数
        """
        self.tokens_total.labels(
            tenant_id=tenant_id,
            model=model,
            token_type="prompt"
        ).inc(prompt_tokens)

        self.tokens_total.labels(
            tenant_id=tenant_id,
            model=model,
            token_type="completion"
        ).inc(completion_tokens)

    def record_llm_call(self, tenant_id: str, model: str, duration_ms: float):
        """
        记录LLM调用

        Args:
            tenant_id: 租户ID
            model: 模型名称
            duration_ms: 调用时长（毫秒）
        """
        self.llm_duration.labels(
            tenant_id=tenant_id,
            model=model
        ).observe(duration_ms / 1000)

    def record_agent_error(
        self,
        tenant_id: str,
        agent_name: str,
        error_type: str
    ):
        """
        记录Agent错误

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称
            error_type: 错误类型
        """
        self.agent_errors.labels(
            tenant_id=tenant_id,
            agent_name=agent_name,
            error_type=error_type
        ).inc()

    def set_active_requests(
        self,
        tenant_id: str,
        agent_name: str,
        count: int
    ):
        """
        设置当前活跃请求数

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称
            count: 当前请求数
        """
        self.active_requests.labels(
            tenant_id=tenant_id,
            agent_name=agent_name
        ).set(count)

    def increment_active_requests(self, tenant_id: str, agent_name: str):
        """增加活跃请求数"""
        self.active_requests.labels(
            tenant_id=tenant_id,
            agent_name=agent_name
        ).inc()

    def decrement_active_requests(self, tenant_id: str, agent_name: str):
        """减少活跃请求数"""
        self.active_requests.labels(
            tenant_id=tenant_id,
            agent_name=agent_name
        ).dec()

    def set_active_conversations(self, tenant_id: str, count: int):
        """设置活跃对话数"""
        self.active_conversations.labels(tenant_id=tenant_id).set(count)

    def get_metrics(self) -> bytes:
        """
        获取Prometheus格式的指标输出

        用于/metrics端点

        Returns:
            Prometheus格式的指标数据
        """
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        else:
            # 如果Prometheus不可用，返回空字节
            return b"# Prometheus client not available"

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        获取当前所有指标的快照

        用于内省和调试

        Returns:
            指标快照字典
        """
        return {
            "requests_total": self._get_counter_value(self.requests_total),
            "tokens_total": self._get_counter_value(self.tokens_total),
            "agent_errors": self._get_counter_value(self.agent_errors),
            "active_conversations": self._get_gauge_value(self.active_conversations)
        }

    def _get_counter_value(self, counter: Counter) -> Dict[str, float]:
        """获取计数器的当前值"""
        # 这是一个简化的实现，实际环境中需要更复杂的逻辑
        return {}

    def _get_gauge_value(self, gauge: Gauge) -> Dict[str, float]:
        """获取仪表的当前值"""
        return {}


# =============================================================================
# 第六部分：AutoGen可观测性集成装饰器
# =============================================================================

def observable(
    tracer: ConversationTracer,
    monitor: TokenMonitor,
    metrics: PrometheusMetrics,
    tenant_id: str,
    conversation_id: str,
    agent_name: str
):
    """
    可观测性装饰器

    为AutoGen Agent方法添加可观测性支持，自动记录追踪、Token和指标

    使用示例：
        tracer = ConversationTracer()
        monitor = TokenMonitor()
        metrics = PrometheusMetrics()

        @observable(tracer, monitor, metrics, tenant_id="tenant_001", conversation_id="conv_001", agent_name="code_assistant")
        def process_message(message: str) -> str:
            # ... 处理逻辑 ...
            return result

    效果：
    - 自动创建追踪记录
    - 记录Token消耗
    - 更新Prometheus指标
    - 记录处理时长
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 开始追踪
            trace = tracer.start_trace(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                agent_name=agent_name,
                input_message=str(args)[:200] if args else ""
            )

            start_time = time.time()
            metrics.increment_active_requests(tenant_id, agent_name)

            try:
                # 执行原函数
                result = func(*args, **kwargs)

                # 记录成功
                duration_ms = (time.time() - start_time) * 1000
                tracer.end_trace(trace.trace_id, str(result)[:200], 0)
                metrics.record_request(tenant_id, agent_name, "success", duration_ms)

                return result

            except Exception as e:
                # 记录失败
                duration_ms = (time.time() - start_time) * 1000
                tracer.fail_trace(trace.trace_id, str(e), type(e).__name__)
                metrics.record_request(tenant_id, agent_name, "failed", duration_ms)
                metrics.record_agent_error(tenant_id, agent_name, type(e).__name__)
                raise

            finally:
                metrics.decrement_active_requests(tenant_id, agent_name)

        return wrapper
    return decorator


# =============================================================================
# 第七部分：可观测性管理器整合
# =============================================================================

class ObservabilityManager:
    """
    可观测性管理器

    整合所有可观测性组件，提供统一的管理接口：

    组件关系：
    - ConversationTracer: 对话链路追踪
    - TokenMonitor: Token消耗监控
    - AlertManager: 异常告警管理
    - PrometheusMetrics: Prometheus指标暴露

    数据流：
    Agent执行 -> Tracer记录链路 -> Monitor记录Token -> Metrics更新指标 -> 告警检查

    使用示例：
        obs_manager = ObservabilityManager()

        # 初始化（可选配置）
        obs_manager.setup_model_prices({
            "gpt-4": (3.0, 15.0),
            "gpt-4o-mini": (0.15, 0.6),
            "qwen2.5": (0.16, 0.64)
        })

        obs_manager.setup_quotas({
            "tenant_001": {"monthly": 1_000_000, "daily": 50_000}
        })

        # 在Agent执行中使用
        obs_manager.before_agent_execution(
            tenant_id="tenant_001",
            conversation_id="conv_001",
            agent_name="code_assistant",
            input_message="帮我写一个排序算法"
        )

        # ... Agent执行 ...

        obs_manager.after_agent_execution(
            tenant_id="tenant_001",
            conversation_id="conv_001",
            agent_name="code_assistant",
            output_message="排序算法已生成",
            token_used=500
        )

        # 获取监控数据
        stats = obs_manager.get_tenant_stats("tenant_001")
        alerts = obs_manager.get_active_alerts()
    """

    def __init__(self):
        print("[可观测性] 初始化可观测性管理器")
        self.tracer = ConversationTracer()
        self.monitor = TokenMonitor()
        self.alert_manager = AlertManager()
        self.metrics = PrometheusMetrics()

        # 注册默认告警规则
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        # 高Token使用告警
        self.alert_manager.add_rule(AlertRule(
            name="high_daily_token_usage",
            metric_name="tenant_tokens_daily",
            condition=">=",
            threshold=80000,  # 80%
            level=AlertLevel.WARNING,
            cooldown_seconds=3600  # 1小时冷却
        ))

        # 配额耗尽告警
        self.alert_manager.add_rule(AlertRule(
            name="quota_exceeded",
            metric_name="quota_usage_percent",
            condition=">=",
            threshold=100,
            level=AlertLevel.CRITICAL,
            cooldown_seconds=300  # 5分钟冷却
        ))

        # Agent连续失败告警
        self.alert_manager.add_rule(AlertRule(
            name="agent_consecutive_failures",
            metric_name="agent_failure_count",
            condition=">=",
            threshold=5,
            level=AlertLevel.ERROR,
            cooldown_seconds=600  # 10分钟冷却
        ))

        # LLM响应过慢告警
        self.alert_manager.add_rule(AlertRule(
            name="slow_llm_response",
            metric_name="llm_latency_ms",
            condition=">=",
            threshold=30000,  # 30秒
            level=AlertLevel.WARNING,
            cooldown_seconds=300
        ))

    def setup_model_prices(self, prices: Dict[str, tuple[float, float]]):
        """
        设置模型价格

        Args:
            prices: 模型价格字典 {model_name: (input_price, output_price)}
        """
        for model_name, (input_price, output_price) in prices.items():
            self.monitor.set_model_price(model_name, input_price, output_price)
            logging.info(f"[可观测性] 配置模型价格: {model_name} = ${input_price}/${output_price}")

    def setup_quotas(self, quotas: Dict[str, Dict[str, int]]):
        """
        设置租户配额

        Args:
            quotas: 配额字典 {tenant_id: {"monthly": int, "daily": int}}
        """
        for tenant_id, quota in quotas.items():
            self.monitor.set_quota(
                tenant_id,
                quota.get("monthly", 1_000_000),
                quota.get("daily", 50_000)
            )

    def before_agent_execution(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        input_message: str,
        parent_trace_id: Optional[str] = None
    ) -> ConversationTrace:
        """
        Agent执行前回调

        用于开始追踪和更新指标

        Args:
            tenant_id: 租户ID
            conversation_id: 对话ID
            agent_name: Agent名称
            input_message: 输入消息
            parent_trace_id: 父追踪ID

        Returns:
            追踪记录对象
        """
        trace = self.tracer.start_trace(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            input_message=input_message,
            parent_trace_id=parent_trace_id
        )

        self.metrics.increment_active_requests(tenant_id, agent_name)

        return trace

    def after_agent_execution(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        output_message: str,
        token_used: int,
        status: str = "completed"
    ):
        """
        Agent执行后回调

        用于结束追踪和更新指标

        Args:
            tenant_id: 租户ID
            conversation_id: 对话ID
            agent_name: Agent名称
            output_message: 输出消息
            token_used: Token消耗
            status: 执行状态
        """
        # 获取最新的追踪并结束
        traces = self.tracer.get_conversation_traces(conversation_id)
        if traces:
            latest_trace = traces[-1]
            self.tracer.end_trace(
                latest_trace.trace_id,
                output_message,
                token_used,
                status
            )

        self.metrics.decrement_active_requests(tenant_id, agent_name)

    def record_llm_call(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ):
        """
        记录LLM调用

        Args:
            tenant_id: 租户ID
            conversation_id: 对话ID
            agent_name: Agent名称
            model_name: 模型名称
            prompt_tokens: 输入Token数
            completion_tokens: 输出Token数
            latency_ms: 延迟（毫秒）
        """
        # 记录Token
        self.monitor.record(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms
        )

        # 更新指标
        self.metrics.record_tokens(
            tenant_id=tenant_id,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        self.metrics.record_llm_call(
            tenant_id=tenant_id,
            model=model_name,
            duration_ms=latency_ms
        )

    def record_error(
        self,
        tenant_id: str,
        agent_name: str,
        error: Exception
    ):
        """
        记录错误

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称
            error: 异常对象
        """
        error_type = type(error).__name__
        self.metrics.record_agent_error(tenant_id, agent_name, error_type)

        # 触发告警检查
        self.alert_manager.trigger(
            rule_name="agent_consecutive_failures",
            metric_value=1.0,  # 简化：每次错误+1
            tenant_id=tenant_id,
            message=f"Agent {agent_name} 发生错误: {error}"
        )

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """获取租户统计"""
        return self.monitor.get_tenant_stats(tenant_id)

    def get_quota_usage(self, tenant_id: str) -> Dict[str, Any]:
        """获取配额使用情况"""
        return self.monitor.get_quota_usage(tenant_id)

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()

    def get_metrics_output(self) -> bytes:
        """获取Prometheus指标输出"""
        return self.metrics.get_metrics()


# =============================================================================
# 第八部分：演示和使用示例
# =============================================================================

async def demo_observability():
    """
    演示可观测性设计

    展示各组件的协同工作流程
    """
    print("=" * 70)
    print("AutoGen可观测性设计演示")
    print("=" * 70)

    # 1. 创建可观测性管理器
    obs_manager = ObservabilityManager()

    # 2. 配置模型价格
    obs_manager.setup_model_prices({
        "gpt-4": (3.0, 15.0),      # $3/输入, $15/输出
        "gpt-4o-mini": (0.15, 0.6),  # $0.15/输入, $0.6/输出
        "qwen2.5": (0.16, 0.64)      # ¥1.12/输入, ¥4.48/输出
    })

    # 3. 配置租户配额
    obs_manager.setup_quotas({
        "tenant_001": {"monthly": 1_000_000, "daily": 50_000},
        "tenant_002": {"monthly": 5_000_000, "daily": 200_000}
    })

    # 4. 模拟多租户对话流程
    tenants = ["tenant_001", "tenant_002"]

    for i, tenant in enumerate(tenants):
        conv_id = f"conv_{i+1}"
        print(f"\n--- 租户 {tenant} 的对话 ---")

        # 代码助手Agent执行
        trace1 = obs_manager.before_agent_execution(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_assistant",
            input_message="用Python写一个快速排序算法"
        )
        print(f"[追踪] 开始: trace_id={trace1.trace_id}")

        # 模拟LLM调用
        await asyncio.sleep(0.5)

        # 记录LLM调用
        obs_manager.record_llm_call(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_assistant",
            model_name="gpt-4",
            prompt_tokens=150,
            completion_tokens=350,
            latency_ms=500
        )

        obs_manager.after_agent_execution(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_assistant",
            output_message="以下是快速排序算法的Python实现...",
            token_used=500,
            status="completed"
        )

        # 代码审查Agent执行
        trace2 = obs_manager.before_agent_execution(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_reviewer",
            input_message="审查刚才生成的代码",
            parent_trace_id=trace1.trace_id
        )

        await asyncio.sleep(0.3)

        obs_manager.record_llm_call(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_reviewer",
            model_name="gpt-4o-mini",
            prompt_tokens=200,
            completion_tokens=100,
            latency_ms=300
        )

        obs_manager.after_agent_execution(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_reviewer",
            output_message="代码审查完成，发现1个潜在问题...",
            token_used=300,
            status="completed"
        )

    # 5. 打印租户统计
    print("\n--- 租户统计 ---")
    for tenant in tenants:
        stats = obs_manager.get_tenant_stats(tenant)
        print(f"{tenant}: {stats['total_requests']} 请求, {stats['total_tokens']} Token, ${stats['total_cost']:.4f}")

    # 6. 打印配额使用
    print("\n--- 配额使用情况 ---")
    for tenant in tenants:
        quota = obs_manager.get_quota_usage(tenant)
        if quota["has_quota"]:
            print(
                f"{tenant}: 月度 {quota['monthly_used']}/{quota['monthly_limit']} "
                f"({quota['usage_percent']:.1f}%), "
                f"每日 {quota['daily_used']}/{quota['daily_limit']} "
                f"({quota['daily_usage_percent']:.1f}%)"
            )

    # 7. 打印追踪树
    print("\n--- 对话追踪树 ---")
    traces = obs_manager.tracer.get_trace_tree("conv_1")
    print(json.dumps(traces, indent=2, default=str))

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


# 程序入口
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # 运行演示
    asyncio.run(demo_observability())