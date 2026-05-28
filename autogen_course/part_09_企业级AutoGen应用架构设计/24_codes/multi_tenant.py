"""
多租户隔离配置方案
=====================================

本文件展示多租户场景下AutoGen Agent的资源隔离与配置管理。

核心功能：
1. 租户配置管理：独立的LLM配置、速率限制、费用配额
2. 资源隔离：租户间的Agent实例、数据、日志完全隔离
3. 优先级调度：根据租户等级分配不同的计算资源
4. 配额控制：基于Token数量的使用限额管理

设计原则：
- 租户间互不干扰：一个租户的问题不会影响其他租户
- 资源公平分配：根据等级合理分配系统资源
- 可审计可追溯：所有操作都有记录，支持计费和审计
- 灵活可配置：支持不同租户的不同需求

作者：AutoGen学习课程
版本：1.0
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# 第一部分：租户等级与配额定义
# =============================================================================

class TenantTier(Enum):
    """
    租户等级枚举

    不同等级对应不同的资源配额和服务质量
    """
    BASIC = "basic"                  # 基础版：限流较严，共享资源
    PROFESSIONAL = "professional"    # 专业版：中等配额，专有资源池
    ENTERPRISE = "enterprise"        # 企业版：高配额，优先调度


@dataclass
class RateLimitConfig:
    """
    速率限制配置

    定义租户的API调用频率限制
    """
    requests_per_minute: int = 30     # 每分钟最大请求数
    tokens_per_minute: int = 50000   # 每分钟最大Token数
    concurrent_chats: int = 2        # 最大并发对话数

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "RateLimitConfig":
        """根据租户等级获取默认的速率限制配置"""
        configs = {
            TenantTier.BASIC: cls(requests_per_minute=30, tokens_per_minute=50000, concurrent_chats=2),
            TenantTier.PROFESSIONAL: cls(requests_per_minute=100, tokens_per_minute=200000, concurrent_chats=10),
            TenantTier.ENTERPRISE: cls(requests_per_minute=500, tokens_per_minute=1000000, concurrent_chats=50)
        }
        return configs.get(tier, configs[TenantTier.BASIC])


@dataclass
class QuotaConfig:
    """
    配额配置

    定义租户的使用限额，支持按周期重置
    """
    monthly_token_limit: int = 1_000_000    # 每月Token限额
    daily_token_limit: int = 100_000        # 每日Token限额
    budget_limit: float = 100.0             # 月度预算上限（美元）

    def check_limit(self, used: int) -> bool:
        """检查是否超过限额"""
        return used < self.monthly_token_limit


@dataclass
class TenantProfile:
    """
    租户档案

    完整存储租户的配置信息、资源使用情况
    """
    tenant_id: str                          # 租户唯一标识
    name: str                               # 租户名称
    tier: TenantTier                        # 租户等级
    rate_limit: RateLimitConfig            # 速率限制配置
    quota: QuotaConfig                      # 配额配置

    # LLM配置
    allowed_models: List[str] = field(default_factory=list)  # 允许使用的模型列表
    preferred_model: Optional[str] = None   # 首选模型

    # 特性开关
    features: Dict[str, bool] = field(default_factory=lambda: {
        "async_enabled": False,             # 是否启用异步
        "multi_agent_enabled": True,        # 是否启用多Agent
        "custom_tools_enabled": True,       # 是否允许自定义工具
        "analytics_enabled": True           # 是否启用分析功能
    })

    # 关联的Agent池ID
    agent_pool_id: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_feature_enabled(self, feature: str) -> bool:
        """检查特性是否启用"""
        return self.features.get(feature, False)

    def can_use_model(self, model: str) -> bool:
        """检查是否可以使用某模型"""
        if not self.allowed_models:
            return True  # 空列表表示不限制
        return model in self.allowed_models


# =============================================================================
# 第二部分：资源监控与配额管理
# =============================================================================

@dataclass
class TokenUsageRecord:
    """
    Token使用记录

    记录单次API调用的Token消耗详情
    """
    timestamp: datetime
    model: str                          # 使用的模型
    prompt_tokens: int                  # 输入Token数
    completion_tokens: int             # 输出Token数
    cost: float                         # 本次费用
    conversation_id: str               # 关联的对话ID
    agent_name: str                     # 调用的Agent名称


class QuotaManager:
    """
    配额管理器

    负责监控和限制租户的Token使用量

    核心功能：
    1. 实时追踪：记录每个租户的实时Token使用量
    2. 配额检查：调用前检查是否超过配额
    3. 预算控制：监控月度费用支出
    4. 告警通知：当使用量达到阈值时发送告警

    使用场景：
    - 防止租户超过月度配额
    - 控制日间使用峰值
    - 生成月度账单
    """

    def __init__(self):
        # 租户配额配置
        self.quotas: Dict[str, QuotaConfig] = {}

        # Token使用记录（按时间排序）
        self.usage_records: Dict[str, List[TokenUsageRecord]] = defaultdict(list)

        # 日使用量统计（用于日内配额控制）
        self.daily_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 告警阈值（百分比）
        self.alert_thresholds = [0.5, 0.75, 0.9, 1.0]  # 50%, 75%, 90%, 100%

    def register_quota(self, tenant_id: str, quota: QuotaConfig):
        """注册租户配额"""
        self.quotas[tenant_id] = quota
        print(f"[配额管理] 为租户 {tenant_id} 注册配额: 月限{quota.monthly_token_limit}, 日限{quota.daily_token_limit}")

    def check_quota(self, tenant_id: str, required_tokens: int) -> bool:
        """
        检查配额是否允许请求

        Args:
            tenant_id: 租户ID
            required_tokens: 需要的Token数量

        Returns:
            True表示允许，False表示超过配额
        """
        if tenant_id not in self.quotas:
            # 未注册的租户使用默认配额
            return True

        quota = self.quotas[tenant_id]

        # 检查月度配额
        monthly_usage = self.get_monthly_usage(tenant_id)
        if monthly_usage + required_tokens > quota.monthly_token_limit:
            print(f"[配额管理] 租户 {tenant_id} 超过月度配额: {monthly_usage} + {required_tokens} > {quota.monthly_token_limit}")
            return False

        # 检查日配额
        daily_usage = self.get_daily_usage(tenant_id)
        if daily_usage + required_tokens > quota.daily_token_limit:
            print(f"[配额管理] 租户 {tenant_id} 超过日配额: {daily_usage} + {required_tokens} > {quota.daily_token_limit}")
            return False

        return True

    def record_usage(self, tenant_id: str, record: TokenUsageRecord):
        """
        记录Token使用量

        Args:
            tenant_id: 租户ID
            record: 使用记录
        """
        self.usage_records[tenant_id].append(record)

        # 更新日使用量
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_usage[tenant_id][today] += record.prompt_tokens + record.completion_tokens

        # 检查是否需要告警
        self._check_alerts(tenant_id)

    def get_monthly_usage(self, tenant_id: str) -> int:
        """获取本月使用量"""
        records = self.usage_records.get(tenant_id, [])
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return sum(
            r.prompt_tokens + r.completion_tokens
            for r in records
            if r.timestamp >= month_start
        )

    def get_daily_usage(self, tenant_id: str) -> int:
        """获取今日使用量"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_usage.get(tenant_id, {}).get(today, 0)

    def get_monthly_cost(self, tenant_id: str) -> float:
        """获取本月费用"""
        records = self.usage_records.get(tenant_id, [])
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return sum(
            r.cost for r in records
            if r.timestamp >= month_start
        )

    def _check_alerts(self, tenant_id: str):
        """检查是否需要发送告警"""
        if tenant_id not in self.quotas:
            return

        quota = self.quotas[tenant_id]
        monthly_usage = self.get_monthly_usage(tenant_id)

        for threshold in self.alert_thresholds:
            usage_ratio = monthly_usage / quota.monthly_token_limit
            if usage_ratio >= threshold:
                alert_key = f"{tenant_id}_{threshold}"
                if not hasattr(self, '_alerted_thresholds'):
                    self._alerted_thresholds = set()

                if alert_key not in self._alerted_thresholds:
                    print(f"[配额告警] 租户 {tenant_id} 使用量达到 {int(threshold*100)}%: {monthly_usage}/{quota.monthly_token_limit}")
                    self._alerted_thresholds.add(alert_key)
                break

    def get_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """获取使用摘要"""
        monthly_usage = self.get_monthly_usage(tenant_id)
        daily_usage = self.get_daily_usage(tenant_id)
        monthly_cost = self.get_monthly_cost(tenant_id)

        quota = self.quotas.get(tenant_id)

        return {
            "tenant_id": tenant_id,
            "monthly_usage": monthly_usage,
            "monthly_limit": quota.monthly_token_limit if quota else None,
            "monthly_usage_percent": (monthly_usage / quota.monthly_token_limit * 100) if quota else 0,
            "daily_usage": daily_usage,
            "daily_limit": quota.daily_token_limit if quota else None,
            "monthly_cost": round(monthly_cost, 2),
            "budget_limit": quota.budget_limit if quota else None,
            "budget_usage_percent": (monthly_cost / quota.budget_limit * 100) if quota and quota.budget_limit > 0 else 0
        }


class RateLimitManager:
    """
    速率限制管理器

    实现滑动窗口限流算法，支持：
    1. 请求频率限制：每分钟最大请求数
    2. Token频率限制：每分钟最大Token数
    3. 并发会话限制：最大并发对话数

    限流算法：滑动窗口
    - 将时间划分为小窗口（如1秒）
    - 统计滑动窗口内的请求数/Tensor数
    - 超过阈值则拒绝请求
    """

    def __init__(self):
        # 租户速率限制配置
        self.configs: Dict[str, RateLimitConfig] = {}

        # 请求时间戳记录（用于滑动窗口）
        self.request_timestamps: Dict[str, List[datetime]] = defaultdict(list)

        # Token使用时间戳记录
        self.token_timestamps: Dict[str, List[tuple[datetime, int]]] = defaultdict(list)

        # 当前活跃对话数
        self.active_conversations: Dict[str, int] = defaultdict(int)

    def register_config(self, tenant_id: str, config: RateLimitConfig):
        """注册速率限制配置"""
        self.configs[tenant_id] = config
        print(f"[限流管理] 为租户 {tenant_id} 注册限流配置: {config.requests_per_minute}请求/分钟")

    async def check_rate_limit(
        self,
        tenant_id: str,
        estimated_tokens: int = 0
    ) -> tuple[bool, str]:
        """
        检查是否允许请求

        Args:
            tenant_id: 租户ID
            estimated_tokens: 预估的Token数（用于Token限流）

        Returns:
            (是否允许, 拒绝原因)
        """
        config = self.configs.get(tenant_id)

        # 未注册租户使用默认限制
        if not config:
            config = RateLimitConfig.for_tier(TenantTier.BASIC)

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        # 1. 检查请求频率
        recent_requests = [
            ts for ts in self.request_timestamps[tenant_id]
            if ts > one_minute_ago
        ]

        if len(recent_requests) >= config.requests_per_minute:
            return False, f"请求频率超限: 每分钟最多{config.requests_per_minute}请求"

        # 2. 检查Token频率
        if estimated_tokens > 0:
            recent_tokens = [
                (ts, tokens) for ts, tokens in self.token_timestamps[tenant_id]
                if ts > one_minute_ago
            ]
            total_recent_tokens = sum(tokens for _, tokens in recent_tokens)

            if total_recent_tokens + estimated_tokens > config.tokens_per_minute:
                return False, f"Token频率超限: 每分钟最多{config.tokens_per_minute}Token"

        # 3. 检查并发会话数
        if self.active_conversations[tenant_id] >= config.concurrent_chats:
            return False, f"并发会话超限: 最多{config.concurrent_chats}个并发会话"

        # 通过检查，记录本次请求
        self.request_timestamps[tenant_id].append(now)
        if estimated_tokens > 0:
            self.token_timestamps[tenant_id].append((now, estimated_tokens))

        return True, ""

    def start_conversation(self, tenant_id: str):
        """开始一个对话（增加活跃计数）"""
        self.active_conversations[tenant_id] += 1

    def end_conversation(self, tenant_id: str):
        """结束一个对话（减少活跃计数）"""
        if self.active_conversations[tenant_id] > 0:
            self.active_conversations[tenant_id] -= 1

    def cleanup_old_records(self, max_age_minutes: int = 10):
        """清理过期的记录，防止内存泄漏"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=max_age_minutes)

        for tenant_id in list(self.request_timestamps.keys()):
            self.request_timestamps[tenant_id] = [
                ts for ts in self.request_timestamps[tenant_id]
                if ts > cutoff
            ]

        for tenant_id in list(self.token_timestamps.keys()):
            self.token_timestamps[tenant_id] = [
                (ts, tokens) for ts, tokens in self.token_timestamps[tenant_id]
                if ts > cutoff
            ]

    def get_rate_limit_status(self, tenant_id: str) -> Dict[str, Any]:
        """获取速率限制状态"""
        config = self.configs.get(tenant_id) or RateLimitConfig.for_tier(TenantTier.BASIC)

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_requests = len([
            ts for ts in self.request_timestamps.get(tenant_id, [])
            if ts > one_minute_ago
        ])

        recent_tokens = sum(
            tokens for ts, tokens in self.token_timestamps.get(tenant_id, [])
            if ts > one_minute_ago
        )

        return {
            "tenant_id": tenant_id,
            "requests_this_minute": recent_requests,
            "requests_limit": config.requests_per_minute,
            "tokens_this_minute": recent_tokens,
            "tokens_limit": config.tokens_per_minute,
            "active_conversations": self.active_conversations.get(tenant_id, 0),
            "conversations_limit": config.concurrent_chats
        }


# =============================================================================
# 第三部分：租户隔离的Agent配置
# =============================================================================

class TenantIsolationManager:
    """
    租户隔离管理器

    实现租户间的完全隔离，包括：
    1. Agent实例隔离：每个租户有独立的Agent实例池
    2. 配置隔离：租户的配置互不影响
    3. 数据隔离：租户的数据不会泄露给其他租户
    4. 日志隔离：日志按照租户ID分组

    核心机制：
    - 租户ID作为隔离标识符
    - 所有资源都打上租户标签
    - 跨租户访问被显式禁止
    """

    def __init__(self):
        # 租户配置
        self.tenants: Dict[str, TenantProfile] = {}

        # 租户专属Agent实例
        self.tenant_agents: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # 租户日志
        self.tenant_logs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # 租户元数据
        self.tenant_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def register_tenant(self, profile: TenantProfile):
        """
        注册租户

        Args:
            profile: 租户档案
        """
        self.tenants[profile.tenant_id] = profile
        print(f"[隔离管理] 注册租户: {profile.tenant_id} ({profile.tier.value})")

    def get_tenant(self, tenant_id: str) -> Optional[TenantProfile]:
        """获取租户档案"""
        return self.tenants.get(tenant_id)

    def create_tenant_agent(
        self,
        tenant_id: str,
        agent_type: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为租户创建隔离的Agent实例

        每个租户的Agent都有独立的：
        - 实例ID（包含租户ID前缀）
        - 配置副本（不与其他租户共享）
        - 系统消息（包含租户信息）

        Args:
            tenant_id: 租户ID
            agent_type: Agent类型
            base_config: 基础配置

        Returns:
            Agent实例
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"租户不存在: {tenant_id}")

        # 生成租户专属的Agent ID
        instance_id = f"{tenant_id}_{agent_type}_{uuid.uuid4().hex[:8]}"

        # 创建隔离的配置副本
        isolated_config = {
            "instance_id": instance_id,
            "tenant_id": tenant_id,
            "agent_type": agent_type,
            # 系统消息中加入租户标识
            "system_message": base_config.get("system_message", "").format(
                tenant_id=tenant_id,
                tenant_name=tenant.name
            ),
            "llm_config": self._create_tenant_llm_config(tenant, base_config.get("llm_config")),
            "max_consecutive_auto_reply": base_config.get("max_consecutive_auto_reply", 10),
            "tools": base_config.get("tools", []),
            "created_at": datetime.now()
        }

        self.tenant_agents[tenant_id][instance_id] = isolated_config

        # 记录日志
        self._log_tenant_event(tenant_id, "agent_created", {
            "instance_id": instance_id,
            "agent_type": agent_type
        })

        print(f"[隔离管理] 为租户 {tenant_id} 创建Agent: {instance_id}")
        return isolated_config

    def _create_tenant_llm_config(
        self,
        tenant: TenantProfile,
        base_llm_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        创建租户专属的LLM配置

        根据租户等级和权限，配置不同的模型访问策略

        Args:
            tenant: 租户档案
            base_llm_config: 基础LLM配置

        Returns:
            隔离后的LLM配置
        """
        if not base_llm_config:
            base_llm_config = {}

        # 合并租户配置
        config = base_llm_config.copy()

        # 设置允许的模型列表
        if tenant.allowed_models:
            config["allowed_models"] = tenant.allowed_models

        # 设置首选模型
        if tenant.preferred_model:
            config["model"] = tenant.preferred_model

        # 根据等级设置温度默认值
        if tenant.tier == TenantTier.BASIC:
            config["temperature"] = min(config.get("temperature", 0.7), 0.5)
        elif tenant.tier == TenantTier.ENTERPRISE:
            config["temperature"] = config.get("temperature", 0.9)

        return config

    def get_tenant_agents(self, tenant_id: str) -> List[Dict[str, Any]]:
        """获取租户的所有Agent实例"""
        return list(self.tenant_agents.get(tenant_id, {}).values())

    def release_tenant_agent(self, tenant_id: str, instance_id: str):
        """释放租户的Agent实例"""
        if tenant_id in self.tenant_agents and instance_id in self.tenant_agents[tenant_id]:
            del self.tenant_agents[tenant_id][instance_id]
            self._log_tenant_event(tenant_id, "agent_released", {"instance_id": instance_id})
            print(f"[隔离管理] 释放租户 {tenant_id} 的Agent: {instance_id}")

    def _log_tenant_event(self, tenant_id: str, event_type: str, data: Dict[str, Any]):
        """记录租户事件日志"""
        log_entry = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "data": data
        }
        self.tenant_logs[tenant_id].append(log_entry)

    def get_tenant_logs(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """获取租户日志"""
        logs = self.tenant_logs.get(tenant_id, [])

        if event_type:
            logs = [l for l in logs if l["event_type"] == event_type]

        if start_time:
            logs = [l for l in logs if l["timestamp"] >= start_time]

        return logs

    def cleanup_tenant(self, tenant_id: str):
        """
        清理租户的所有资源

        用于租户注销或长期不活跃场景

        Args:
            tenant_id: 租户ID
        """
        # 清理Agent实例
        agent_count = len(self.tenant_agents.get(tenant_id, {}))
        if tenant_id in self.tenant_agents:
            del self.tenant_agents[tenant_id]

        # 清理日志
        if tenant_id in self.tenant_logs:
            del self.tenant_logs[tenant_id]

        # 清理配额和限流记录（由各自的Manager处理）

        print(f"[隔离管理] 清理租户 {tenant_id}: 释放{agent_count}个Agent实例")


# =============================================================================
# 第四部分：多租户场景下的AutoGen集成
# =============================================================================

class MultiTenantAutoGenManager:
    """
    多租户AutoGen管理器

    整合上述所有组件，形成完整的多租户AutoGen解决方案

    核心能力：
    1. 租户生命周期管理：注册、配置、更新、注销
    2. 资源隔离与配额控制
    3. 请求路由与负载均衡
    4. 可观测性：日志、监控、告警

    使用示例：
        manager = MultiTenantAutoGenManager()

        # 注册租户
        manager.register_tenant(
            tenant_id="tenant_001",
            name="示例公司",
            tier=TenantTier.PROFESSIONAL
        )

        # 处理请求
        result = await manager.process_request(
            tenant_id="tenant_001",
            message="请帮我写一个排序算法"
        )
    """

    def __init__(self):
        # 租户隔离管理器
        self.isolation_manager = TenantIsolationManager()

        # 配额管理器
        self.quota_manager = QuotaManager()

        # 速率限制管理器
        self.rate_limit_manager = RateLimitManager()

        # Agent配置模板
        self.agent_templates: Dict[str, Dict[str, Any]] = {}

        print("[多租户管理] 初始化多租户AutoGen管理器")

    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier,
        allowed_models: Optional[List[str]] = None,
        preferred_model: Optional[str] = None,
        async_enabled: bool = False
    ):
        """
        注册新租户

        Args:
            tenant_id: 租户唯一标识
            name: 租户名称
            tier: 租户等级
            allowed_models: 允许使用的模型列表
            preferred_model: 首选模型
            async_enabled: 是否启用异步功能
        """
        # 创建租户档案
        profile = TenantProfile(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            rate_limit=RateLimitConfig.for_tier(tier),
            quota=QuotaConfig() if tier == TenantTier.BASIC else QuotaConfig(
                monthly_token_limit=10_000_000 if tier == TenantTier.ENTERPRISE else 5_000_000,
                daily_token_limit=500_000 if tier == TenantTier.ENTERPRISE else 200_000,
                budget_limit=1000.0 if tier == TenantTier.ENTERPRISE else 500.0
            ),
            allowed_models=allowed_models or [],
            preferred_model=preferred_model,
            features={
                "async_enabled": async_enabled,
                "multi_agent_enabled": tier != TenantTier.BASIC,
                "custom_tools_enabled": tier == TenantTier.ENTERPRISE,
                "analytics_enabled": True
            }
        )

        # 注册到隔离管理器
        self.isolation_manager.register_tenant(profile)

        # 注册配额
        self.quota_manager.register_quota(tenant_id, profile.quota)

        # 注册限流配置
        self.rate_limit_manager.register_config(tenant_id, profile.rate_limit)

        print(f"[多租户管理] 注册租户完成: {tenant_id} ({tier.value})")

    def register_agent_template(
        self,
        template_id: str,
        system_message: str,
        max_consecutive_auto_reply: int = 10,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        注册Agent模板

        Args:
            template_id: 模板ID
            system_message: 系统消息（支持占位符 {tenant_id}, {tenant_name}）
            max_consecutive_auto_reply: 最大连续回复数
            llm_config: LLM配置
        """
        self.agent_templates[template_id] = {
            "system_message": system_message,
            "max_consecutive_auto_reply": max_consecutive_auto_reply,
            "llm_config": llm_config or {}
        }
        print(f"[多租户管理] 注册Agent模板: {template_id}")

    async def process_request(
        self,
        tenant_id: str,
        message: str,
        agent_template: str = "default",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理租户请求

        完整流程：
        1. 验证租户存在
        2. 检查速率限制
        3. 检查配额
        4. 获取或创建Agent
        5. 执行请求
        6. 记录使用量

        Args:
            tenant_id: 租户ID
            message: 用户消息
            agent_template: Agent模板ID
            conversation_id: 对话ID（可选）

        Returns:
            处理结果
        """
        tenant = self.isolation_manager.get_tenant(tenant_id)
        if not tenant:
            return {
                "success": False,
                "error": "tenant_not_found",
                "message": f"租户不存在: {tenant_id}"
            }

        # 生成对话ID
        if not conversation_id:
            conversation_id = uuid.uuid4().hex[:16]

        # 1. 速率限制检查
        rate_allowed, rate_reason = await self.rate_limit_manager.check_rate_limit(
            tenant_id,
            estimated_tokens=len(message) * 2  # 粗略估算
        )

        if not rate_allowed:
            return {
                "success": False,
                "error": "rate_limit_exceeded",
                "message": rate_reason,
                "tenant_id": tenant_id
            }

        # 2. 配额检查（预估需要1000 tokens）
        if not self.quota_manager.check_quota(tenant_id, required_tokens=1000):
            return {
                "success": False,
                "error": "quota_exceeded",
                "message": "配额已用尽，请升级或等待下个周期重置",
                "tenant_id": tenant_id
            }

        # 开始对话
        self.rate_limit_manager.start_conversation(tenant_id)

        try:
            # 3. 获取或创建Agent
            agents = self.isolation_manager.get_tenant_agents(tenant_id)
            agent = None

            if agents:
                # 复用已有Agent
                agent = agents[0]
            else:
                # 创建新Agent
                template = self.agent_templates.get(agent_template, self.agent_templates.get("default", {
                    "system_message": "你是一个AI助手。",
                    "max_consecutive_auto_reply": 10,
                    "llm_config": {}
                }))

                agent = self.isolation_manager.create_tenant_agent(
                    tenant_id=tenant_id,
                    agent_type=agent_template,
                    base_config=template
                )

            # 4. 模拟LLM调用
            await asyncio.sleep(0.1)  # 模拟API延迟

            response = f"[AutoGen回复] 租户 {tenant_id} 的请求已处理: {message[:50]}..."

            # 5. 记录使用量
            usage_record = TokenUsageRecord(
                timestamp=datetime.now(),
                model=tenant.preferred_model or "gpt-4",
                prompt_tokens=len(message),
                completion_tokens=len(response),
                cost=0.001,  # 简化计算
                conversation_id=conversation_id,
                agent_name=agent["instance_id"]
            )
            self.quota_manager.record_usage(tenant_id, usage_record)

            return {
                "success": True,
                "response": response,
                "tenant_id": tenant_id,
                "agent_id": agent["instance_id"],
                "conversation_id": conversation_id,
                "token_usage": {
                    "prompt": usage_record.prompt_tokens,
                    "completion": usage_record.completion_tokens,
                    "cost": usage_record.cost
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": "processing_failed",
                "message": str(e),
                "tenant_id": tenant_id
            }

        finally:
            # 结束对话
            self.rate_limit_manager.end_conversation(tenant_id)

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """获取租户统计信息"""
        tenant = self.isolation_manager.get_tenant(tenant_id)
        if not tenant:
            return {"error": "tenant_not_found"}

        usage_summary = self.quota_manager.get_usage_summary(tenant_id)
        rate_limit_status = self.rate_limit_manager.get_rate_limit_status(tenant_id)
        agents = self.isolation_manager.get_tenant_agents(tenant_id)

        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "tier": tenant.tier.value,
            "quota_usage": usage_summary,
            "rate_limit": rate_limit_status,
            "active_agents": len(agents),
            "features": tenant.features
        }

    def delete_tenant(self, tenant_id: str):
        """删除租户及其所有资源"""
        # 清理隔离资源
        self.isolation_manager.cleanup_tenant(tenant_id)

        # 注意：QuotaManager和RateLimitManager的记录会保留一段时间用于计费

        print(f"[多租户管理] 删除租户: {tenant_id}")


# =============================================================================
# 第五部分：演示代码
# =============================================================================

async def demo_multi_tenant():
    """
    多租户隔离方案演示

    展示以下场景：
    1. 注册不同等级的租户
    2. 模拟请求处理
    3. 配额和限流验证
    4. 租户资源隔离验证
    """
    print("=" * 70)
    print("多租户隔离配置方案演示")
    print("=" * 70)

    # 创建管理器
    manager = MultiTenantAutoGenManager()

    # 1. 注册Agent模板
    manager.register_agent_template(
        template_id="default",
        system_message="你是一个专业的AI助手，服务于租户 {tenant_id} ({tenant_name})。",
        max_consecutive_auto_reply=10
    )
    manager.register_agent_template(
        template_id="code_assistant",
        system_message="你是一个代码生成助手，帮助租户 {tenant_id} ({tenant_name}) 生成高质量代码。",
        max_consecutive_auto_reply=5
    )

    # 2. 注册多个租户（不同等级）
    print("\n--- 注册租户 ---")
    manager.register_tenant(
        tenant_id="tenant_basic",
        name="基础租户",
        tier=TenantTier.BASIC,
        preferred_model="gpt-4o-mini"
    )
    manager.register_tenant(
        tenant_id="tenant_pro",
        name="专业租户",
        tier=TenantTier.PROFESSIONAL,
        allowed_models=["gpt-4", "gpt-4o-mini", "claude-3"],
        preferred_model="gpt-4"
    )
    manager.register_tenant(
        tenant_id="tenant_ent",
        name="企业租户",
        tier=TenantTier.ENTERPRISE,
        async_enabled=True,
        preferred_model="claude-3"
    )

    # 3. 模拟请求处理
    print("\n--- 模拟请求处理 ---")
    test_cases = [
        {"tenant": "tenant_basic", "msg": "请帮我写一个Hello World程序"},
        {"tenant": "tenant_pro", "msg": "请实现快速排序算法"},
        {"tenant": "tenant_ent", "msg": "设计一个高可用的微服务架构"}
    ]

    for case in test_cases:
        print(f"\n请求: {case['tenant']} -> {case['msg'][:30]}...")
        result = await manager.process_request(
            tenant_id=case["tenant"],
            message=case["msg"],
            agent_template="default"
        )
        print(f"结果: {'成功' if result.get('success') else '失败'}")
        if result.get("success"):
            print(f"  响应: {result['response'][:50]}...")
            print(f"  Token: {result['token_usage']}")
        else:
            print(f"  错误: {result.get('message')}")

    # 4. 获取租户统计
    print("\n--- 租户统计 ---")
    for tenant_id in ["tenant_basic", "tenant_pro", "tenant_ent"]:
        stats = manager.get_tenant_stats(tenant_id)
        print(f"\n{tenant_id} ({stats['tier']}):")
        print(f"  活跃Agent: {stats['active_agents']}")
        print(f"  月度使用: {stats['quota_usage']['monthly_usage']} / {stats['quota_usage'].get('monthly_limit', 'N/A')}")
        print(f"  本月费用: ${stats['quota_usage']['monthly_cost']}")

    # 5. 验证隔离性
    print("\n--- 验证资源隔离 ---")
    basic_agents = manager.isolation_manager.get_tenant_agents("tenant_basic")
    pro_agents = manager.isolation_manager.get_tenant_agents("tenant_pro")
    print(f"基础租户Agent数: {len(basic_agents)}")
    print(f"专业租户Agent数: {len(pro_agents)}")
    print(f"Agent实例ID不同: {basic_agents[0]['instance_id'] != pro_agents[0]['instance_id'] if basic_agents and pro_agents else 'N/A'}")

    # 6. 测试限流
    print("\n--- 测试速率限制 ---")
    print("模拟短时间内多次请求...")
    for i in range(5):
        result = await manager.process_request(
            tenant_id="tenant_basic",
            message=f"测试请求 #{i+1}"
        )
        status = "成功" if result.get("success") else result.get("error", "未知错误")
        print(f"  请求#{i+1}: {status}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


# 程序入口
if __name__ == "__main__":
    asyncio.run(demo_multi_tenant())