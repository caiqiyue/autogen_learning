"""
企业级AutoGen架构设计示例
=====================================

本文件展示企业内网AI部署架构设计与AutoGen的集成方案。

核心组件：
1. LLM网关（LLMGateway）：统一的模型调用入口，支持多模型负载均衡与故障转移
2. 模型调度器（ModelScheduler）：根据任务类型智能调度到合适的模型
3. Agent池（AgentPool）：管理多个AutoGen Agent实例，实现资源复用与隔离
4. 可观测性模块（Observability）：对话链路追踪、Token消耗监控

架构设计要点：
- 高可用：多副本部署，自动故障转移
- 多租户隔离：租户间的配置、资源、数据的完全隔离
- 可扩展：水平扩展Agent池应对高并发
- 可观测：完整的日志、监控、追踪体系

作者：AutoGen学习课程
版本：1.0
"""

import asyncio
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# 第一部分：核心数据结构和枚举定义
# =============================================================================

class ModelType(Enum):
    """
    模型类型枚举

    用于区分不同类型的LLM模型，便于模型调度器进行智能路由
    """
    GPT4 = "gpt-4"           # OpenAI GPT-4系列，适合复杂推理任务
    GPT4_MINI = "gpt-4o-mini"  # OpenAI GPT-4o mini，高性价比
    CLAUDE = "claude-3"      # Anthropic Claude系列，适合长文本处理
    QWEN = "qwen2.5"         # 阿里通义千问，适合中文场景
    INTERNLM = "internlm2.5"  # 上海AI Lab书生大模型，适合代码任务
    LOCAL_OLLAMA = "ollama"   # 本地Ollama部署，开源模型


class TenantTier(Enum):
    """
    租户等级枚举

    用于多租户场景下的资源配额管理
    """
    BASIC = "basic"          # 基础版：少量请求，低优先级
    PROFESSIONAL = "professional"  # 专业版：中等请求量，保证可用性
    ENTERPRISE = "enterprise"  # 企业版：高请求量，专属资源


@dataclass
class TokenUsage:
    """
    Token使用量记录

    用于追踪每个租户、每个模型的Token消耗情况
    """
    tenant_id: str                    # 租户ID
    model_type: ModelType            # 使用的模型类型
    prompt_tokens: int = 0           # 输入Token数量
    completion_tokens: int = 0       # 输出Token数量
    total_tokens: int = 0            # 总Token数量
    request_count: int = 0           # 请求次数
    timestamp: datetime = field(default_factory=datetime.now)

    def add_usage(self, prompt: int, completion: int):
        """
        累加Token使用量

        Args:
            prompt: 新增的输入Token数
            completion: 新增的输出Token数
        """
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.request_count += 1


@dataclass
class ConversationTrace:
    """
    对话链路追踪记录

    用于记录完整的多Agent对话流程，便于问题排查和性能分析
    """
    trace_id: str                    # 唯一追踪ID
    tenant_id: str                   # 租户ID
    conversation_id: str             # 对话会话ID
    agent_name: str                  # 当前Agent名称
    agent_role: str                  # Agent角色
    parent_trace_id: Optional[str]   # 父追踪ID（用于嵌套对话）
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    input_message: str = ""          # 输入消息
    output_message: str = ""         # 输出消息
    token_used: int = 0             # 本次Token消耗
    status: str = "started"         # 状态：started/processing/completed/failed
    error_message: Optional[str] = None  # 错误信息（如有）

    def complete(self, output: str, tokens: int):
        """标记对话完成"""
        self.end_time = datetime.now()
        self.output_message = output
        self.token_used = tokens
        self.status = "completed"

    def fail(self, error: str):
        """标记对话失败"""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error

    def duration_ms(self) -> float:
        """计算执行时长（毫秒）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


# =============================================================================
# 第二部分：LLM网关实现
# =============================================================================

class LLMConfig:
    """
    LLM配置类

    封装单个模型的配置信息，包括API端点、密钥、价格等
    """

    def __init__(
        self,
        model_type: ModelType,
        api_key: str,
        base_url: str,
        price: tuple[float, float],  # (input_price, output_price) per 1M tokens
        max_tokens: int = 4096,
        timeout: float = 60.0
    ):
        self.model_type = model_type
        self.api_key = api_key
        self.base_url = base_url
        self.price = price  # 元组：(输入价格/百万Token, 输出价格/百万Token)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.is_healthy = True  # 健康状态标记
        self.error_count = 0     # 连续错误计数

    def __repr__(self):
        return f"LLMConfig(model={self.model_type.value}, healthy={self.is_healthy})"


class LLMGateway:
    """
    LLM网关

    企业级LLM统一调用入口，提供以下能力：
    1. 多模型接入：支持OpenAI、Anthropic、阿里、Ollama等多种模型
    2. 负载均衡：根据响应时间自动选择最佳模型
    3. 故障转移：当主模型失败时自动切换到备选模型
    4. 成本控制：根据价格选择高性价比模型
    5. 限流保护：防止单个租户消耗过多资源

    使用示例：
        gateway = LLMGateway()
        gateway.register_model(ModelType.GPT4, api_key="xxx", base_url="https://api.openai.com")
        gateway.register_model(ModelType.QWEN, api_key="yyy", base_url="https://dashscope.aliyuncs.com")

        # 调用模型
        response = await gateway.generate("tenant_123", "你好，请介绍一下自己")
    """

    def __init__(self):
        # 模型配置列表，支持多模型注册
        self.models: Dict[ModelType, List[LLMConfig]] = defaultdict(list)
        # 模型健康状态权重（用于负载均衡）
        self.model_weights: Dict[str, float] = {}
        # Token使用量记录
        self.token_usage: Dict[str, List[TokenUsage]] = defaultdict(list)
        # 租户请求计数（用于限流）
        self.tenant_request_count: Dict[str, List[datetime]] = defaultdict(list)
        # 限流配置：每分钟最大请求数
        self.rate_limit = 100

    def register_model(
        self,
        model_type: ModelType,
        api_key: str,
        base_url: str,
        price: tuple[float, float],
        max_tokens: int = 4096,
        priority: int = 1
    ):
        """
        注册LLM模型

        Args:
            model_type: 模型类型枚举
            api_key: API密钥
            base_url: API基础URL
            price: 价格元组 (输入价格, 输出价格) 每百万Token
            max_tokens: 最大Token数
            priority: 优先级（数字越大优先级越高）
        """
        config = LLMConfig(model_type, api_key, base_url, price, max_tokens)
        # 按优先级插入到列表中（优先级高的排在前面）
        self.models[model_type].append(config)
        self.models[model_type].sort(key=lambda x: priority, reverse=True)
        print(f"[LLM网关] 注册模型: {model_type.value}, 优先级: {priority}")

    async def generate(
        self,
        tenant_id: str,
        prompt: str,
        model_preference: Optional[ModelType] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        生成文本回复

        这是LLM网关的核心方法，负责：
        1. 限流检查
        2. 模型选择
        3. 调用重试
        4. 成本计算

        Args:
            tenant_id: 租户ID
            prompt: 输入提示
            model_preference: 首选模型类型（可选）
            temperature: 温度参数
            max_tokens: 最大生成Token数

        Returns:
            包含 response、model_used、token_usage 的字典
        """
        # 1. 限流检查
        if not self._check_rate_limit(tenant_id):
            return {
                "success": False,
                "error": "rate_limit_exceeded",
                "message": "请求过于频繁，请稍后再试"
            }

        # 2. 选择模型
        model_config = self._select_model(model_preference)
        if not model_config:
            return {
                "success": False,
                "error": "no_available_model",
                "message": "暂无可用模型"
            }

        # 3. 调用模型（模拟）
        start_time = time.time()
        try:
            response = await self._call_model(model_config, prompt, temperature, max_tokens)
            elapsed = time.time() - start_time

            # 4. 更新模型权重（根据响应时间）
            self._update_model_weight(model_config, elapsed)

            # 5. 记录Token使用量
            self._record_usage(tenant_id, model_config, len(prompt), len(response))

            return {
                "success": True,
                "response": response,
                "model_used": model_config.model_type.value,
                "elapsed_ms": int(elapsed * 1000),
                "token_used": len(prompt) + len(response)
            }
        except Exception as e:
            # 模型调用失败，标记为不健康并尝试故障转移
            model_config.error_count += 1
            if model_config.error_count >= 3:
                model_config.is_healthy = False
            print(f"[LLM网关] 模型调用失败: {model_config.model_type.value}, 错误: {e}")
            return {
                "success": False,
                "error": "model_call_failed",
                "message": str(e)
            }

    def _check_rate_limit(self, tenant_id: str) -> bool:
        """
        检查限流

        实现简单的滑动窗口限流：每分钟最多N个请求

        Args:
            tenant_id: 租户ID

        Returns:
            是否允许请求
        """
        now = datetime.now()
        # 移除1分钟之前的请求记录
        self.tenant_request_count[tenant_id] = [
            ts for ts in self.tenant_request_count[tenant_id]
            if (now - ts).total_seconds() < 60
        ]
        # 检查是否超过限制
        if len(self.tenant_request_count[tenant_id]) >= self.rate_limit:
            return False
        self.tenant_request_count[tenant_id].append(now)
        return True

    def _select_model(self, preference: Optional[ModelType]) -> Optional[LLMConfig]:
        """
        选择最佳模型

        选择策略：
        1. 优先使用首选模型
        2. 其次选择健康的、高权重的模型
        3. 忽略连续失败的模型

        Args:
            preference: 首选模型类型

        Returns:
            选中的模型配置
        """
        candidate_models = []

        # 优先使用首选模型
        if preference and self.models.get(preference):
            for config in self.models[preference]:
                if config.is_healthy:
                    candidate_models.append(config)

        # 如果首选模型不可用，选择其他可用模型
        if not candidate_models:
            for model_type, configs in self.models.items():
                if model_type != preference:
                    for config in configs:
                        if config.is_healthy:
                            candidate_models.append(config)

        if not candidate_models:
            return None

        # 按权重排序，选择权重最高的
        candidate_models.sort(
            key=lambda x: self.model_weights.get(x.model_type.value, 1.0),
            reverse=True
        )
        return candidate_models[0]

    def _update_model_weight(self, config: LLMConfig, elapsed: float):
        """
        更新模型权重

        响应时间越短，权重越高

        Args:
            config: 模型配置
            elapsed: 响应时间（秒）
        """
        model_key = config.model_type.value
        current_weight = self.model_weights.get(model_key, 1.0)
        # 新权重 = 旧权重 * 0.9 + 新响应速度因子 * 0.1
        # 响应速度因子：1.0表示1秒以内，越快越大
        speed_factor = max(0.1, 1.0 / (elapsed + 0.1))
        new_weight = current_weight * 0.9 + speed_factor * 0.1
        self.model_weights[model_key] = new_weight

    def _record_usage(self, tenant_id: str, config: LLMConfig, prompt_len: int, response_len: int):
        """记录Token使用量"""
        usage = TokenUsage(
            tenant_id=tenant_id,
            model_type=config.model_type,
            prompt_tokens=prompt_len,
            completion_tokens=response_len,
            total_tokens=prompt_len + response_len
        )
        self.token_usage[tenant_id].append(usage)

    async def _call_model(
        self,
        config: LLMConfig,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        调用模型（模拟实现）

        在实际环境中，这里会调用真实的LLM API
        """
        # 模拟API调用延迟
        await asyncio.sleep(0.1)
        # 模拟返回
        return f"[{config.model_type.value}] 模拟回复: 你好，这是来自{config.model_type.value}的回复"

    def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """
        获取租户Token使用统计

        Args:
            tenant_id: 租户ID

        Returns:
            使用统计字典
        """
        usages = self.token_usage.get(tenant_id, [])
        total_prompt = sum(u.prompt_tokens for u in usages)
        total_completion = sum(u.completion_tokens for u in usages)
        total_cost = sum(
            u.prompt_tokens / 1_000_000 * p + u.completion_tokens / 1_000_000 * c
            for u in usages
            for p, c in [self._get_model_price(u.model_type)]
        )
        return {
            "tenant_id": tenant_id,
            "total_requests": len(usages),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cost": round(total_cost, 4)
        }

    def _get_model_price(self, model_type: ModelType) -> tuple[float, float]:
        """获取模型价格"""
        for configs in self.models.values():
            for config in configs:
                if config.model_type == model_type:
                    return config.price
        return (0.01, 0.03)  # 默认价格


# =============================================================================
# 第三部分：Agent池管理
# =============================================================================

@dataclass
class AgentConfig:
    """
    Agent配置

    定义单个AutoGen Agent的行为参数
    """
    name: str                        # Agent名称
    role: str                        # Agent角色描述
    system_message: str              # 系统提示词
    max_consecutive_auto_reply: int = 10  # 最大连续自动回复数
    llm_config: Optional[Dict[str, Any]] = None  # LLM配置
    tools: Optional[List[Callable]] = None  # 工具列表
    is_termination_msg: Optional[Callable] = None  # 终止条件判断函数


class AgentPool:
    """
    Agent池

    管理多个AutoGen Agent实例，提供以下能力：
    1. 租户隔离：每个租户有独立的Agent实例
    2. 资源复用：复用Agent实例减少创建开销
    3. 弹性扩缩：根据负载动态调整Agent数量
    4. 故障隔离：单个Agent失败不影响其他租户

    核心设计理念：
    - 租户隔离：通过tenant_id区分不同租户的资源
    - 懒加载：直到首次使用时才创建Agent实例
    - 缓存复用：已创建的Agent实例放入缓存供复用
    """

    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway  # LLM网关引用
        # 租户 -> Agent实例映射
        self.agents: Dict[str, Dict[str, Any]] = defaultdict(dict)
        # Agent配置模板
        self.agent_templates: Dict[str, AgentConfig] = {}
        # Agent实例计数器
        self.agent_counter: Dict[str, int] = defaultdict(int)

    def register_agent_template(self, template: AgentConfig):
        """
        注册Agent配置模板

        Args:
            template: Agent配置模板
        """
        self.agent_templates[template.name] = template
        print(f"[Agent池] 注册Agent模板: {template.name}")

    def get_agent(self, tenant_id: str, agent_name: str) -> Dict[str, Any]:
        """
        获取租户的Agent实例

        如果不存在，则创建新的Agent实例

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称

        Returns:
            Agent实例字典
        """
        # 检查租户是否有该Agent
        if agent_name not in self.agents[tenant_id]:
            # 创建新的Agent实例
            agent = self._create_agent(tenant_id, agent_name)
            self.agents[tenant_id][agent_name] = agent
            print(f"[Agent池] 为租户 {tenant_id} 创建Agent: {agent_name}")

        return self.agents[tenant_id][agent_name]

    def _create_agent(self, tenant_id: str, agent_name: str) -> Dict[str, Any]:
        """
        创建Agent实例

        根据模板配置和租户信息创建Agent

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称

        Returns:
            Agent实例字典
        """
        template = self.agent_templates.get(agent_name)
        if not template:
            raise ValueError(f"Agent模板不存在: {agent_name}")

        self.agent_counter[agent_name] += 1

        # 构建Agent实例（模拟AutoGen的ConversableAgent结构）
        return {
            "name": f"{tenant_id}_{template.name}_{self.agent_counter[agent_name]}",
            "role": template.role,
            "system_message": template.system_message.format(tenant_id=tenant_id),
            "max_consecutive_auto_reply": template.max_consecutive_auto_reply,
            "llm_config": template.llm_config,
            "tools": template.tools or [],
            "is_termination_msg": template.is_termination_msg,
            "tenant_id": tenant_id,  # 绑定租户ID，用于隔离
            "instance_id": uuid.uuid4().hex[:8],  # 实例唯一ID
            "created_at": datetime.now()
        }

    def release_tenant_agents(self, tenant_id: str):
        """
        释放租户的所有Agent实例

        用于租户注销或资源回收场景

        Args:
            tenant_id: 租户ID
        """
        if tenant_id in self.agents:
            agent_count = len(self.agents[tenant_id])
            del self.agents[tenant_id]
            print(f"[Agent池] 释放租户 {tenant_id} 的 {agent_count} 个Agent实例")

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取Agent池统计信息"""
        total_agents = sum(len(agents) for agents in self.agents.values())
        return {
            "total_tenants": len(self.agents),
            "total_agents": total_agents,
            "agent_counts_by_tenant": {
                tenant: len(agents) for tenant, agents in self.agents.items()
            }
        }


# =============================================================================
# 第四部分：对话链路追踪器
# =============================================================================

class ConversationTracer:
    """
    对话链路追踪器

    用于记录完整的多Agent对话流程，支持：
    1. 分布式追踪：跨多个Agent的对话可以串联成完整链路
    2. 性能分析：记录每个环节的耗时
    3. 问题排查：快速定位哪个环节出现问题
    4. 成本分析：追踪每个租户的Token消耗

    使用示例：
        tracer = ConversationTracer()
        trace = tracer.start_trace("tenant_123", "assistant_agent", "代码生成")
        # ... 执行任务 ...
        tracer.end_trace(trace.trace_id, response, tokens_used)
    """

    def __init__(self):
        # 追踪记录存储
        self.traces: Dict[str, ConversationTrace] = {}
        # 租户追踪ID列表（用于快速查询某租户的所有追踪）
        self.tenant_traces: Dict[str, List[str]] = defaultdict(list)

    def start_trace(
        self,
        tenant_id: str,
        agent_name: str,
        agent_role: str,
        conversation_id: str,
        input_message: str,
        parent_trace_id: Optional[str] = None
    ) -> ConversationTrace:
        """
        开始一个追踪

        Args:
            tenant_id: 租户ID
            agent_name: Agent名称
            agent_role: Agent角色
            conversation_id: 对话会话ID
            input_message: 输入消息
            parent_trace_id: 父追踪ID（用于嵌套对话）

        Returns:
            追踪记录对象
        """
        trace_id = uuid.uuid4().hex[:16]
        trace = ConversationTrace(
            trace_id=trace_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            agent_role=agent_role,
            parent_trace_id=parent_trace_id,
            input_message=input_message[:200]  # 截断以节省存储
        )
        self.traces[trace_id] = trace
        self.tenant_traces[tenant_id].append(trace_id)
        print(f"[追踪] 开始追踪: trace_id={trace_id}, agent={agent_name}")
        return trace

    def end_trace(self, trace_id: str, output: str, token_used: int):
        """
        结束一个追踪

        Args:
            trace_id: 追踪ID
            output: 输出消息
            token_used: Token消耗
        """
        if trace_id in self.traces:
            self.traces[trace_id].complete(output[:200], token_used)
            print(f"[追踪] 结束追踪: trace_id={trace_id}, duration={self.traces[trace_id].duration_ms():.2f}ms")

    def fail_trace(self, trace_id: str, error: str):
        """
        标记追踪失败

        Args:
            trace_id: 追踪ID
            error: 错误信息
        """
        if trace_id in self.traces:
            self.traces[trace_id].fail(error)
            print(f"[追踪] 追踪失败: trace_id={trace_id}, error={error}")

    def get_conversation_traces(self, conversation_id: str) -> List[ConversationTrace]:
        """
        获取会话的所有追踪记录

        用于分析完整的多Agent对话流程

        Args:
            conversation_id: 对话会话ID

        Returns:
            按时间排序的追踪记录列表
        """
        return sorted(
            [t for t in self.traces.values() if t.conversation_id == conversation_id],
            key=lambda x: x.start_time
        )

    def get_tenant_traces(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ConversationTrace]:
        """
        获取租户的追踪记录

        支持时间范围过滤

        Args:
            tenant_id: 租户ID
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）

        Returns:
            符合条件的追踪记录列表
        """
        trace_ids = self.tenant_traces.get(tenant_id, [])
        traces = [self.traces[tid] for tid in trace_ids if tid in self.traces]

        if start_time:
            traces = [t for t in traces if t.start_time >= start_time]
        if end_time:
            traces = [t for t in traces if t.start_time <= end_time]

        return traces


# =============================================================================
# 第五部分：容错与恢复机制
# =============================================================================

class CircuitBreaker:
    """
    熔断器

    实现熔断模式，防止故障在系统中扩散

    工作原理：
    1. 统计连续失败次数
    2. 超过阈值时触发熔断（开启）
    3. 熔断期间所有请求直接失败
    4. 熔断时间过后，进入半开状态，允许一个请求尝试
    5. 如果成功则关闭熔断器，否则重新开启

    适用场景：
    - LLM API调用不稳定
    - 第三方服务可能出现临时故障
    - 需要防止级联失败
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1
    ):
        self.name = name
        self.failure_threshold = failure_threshold  # 触发熔断的连续失败次数
        self.recovery_timeout = recovery_timeout    # 熔断恢复超时（秒）
        self.half_open_max_calls = half_open_max_calls  # 半开状态下允许的尝试次数

        self.failure_count = 0          # 当前连续失败次数
        self.state = "closed"           # 状态：closed（正常）/ open（熔断）/ half_open（半开）
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0         # 半开状态下的尝试次数

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数

        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数执行结果

        Raises:
            CircuitOpenException: 熔断器开启时抛出
        """
        # 检查熔断器状态
        self._check_state()

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _check_state(self):
        """检查熔断器状态"""
        if self.state == "closed":
            return

        if self.state == "open":
            # 检查是否超时可以进入半开状态
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half_open"
                    self.half_open_calls = 0
                    print(f"[熔断器] {self.name} 进入半开状态")
            else:
                raise CircuitOpenException(f"熔断器 {self.name} 处于开启状态")

        elif self.state == "half_open":
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenException(f"熔断器 {self.name} 处于半开状态，已达最大尝试次数")

    def _on_success(self):
        """处理成功调用"""
        if self.state == "half_open":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                # 所有尝试都成功，关闭熔断器
                self._reset()
                print(f"[熔断器] {self.name} 关闭，恢复正常")

    def _on_failure(self):
        """处理失败调用"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == "half_open":
            # 半开状态下失败，重新开启熔断器
            self.state = "open"
            print(f"[熔断器] {self.name} 重新开启")

        elif self.failure_count >= self.failure_threshold:
            # 达到阈值，开启熔断器
            self.state = "open"
            print(f"[熔断器] {self.name} 开启（连续失败 {self.failure_count} 次）")

    def _reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
        self.half_open_calls = 0

    def get_status(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitOpenException(Exception):
    """熔断器开启异常"""
    pass


class AgentFailureHandler:
    """
    Agent失败处理器

    负责处理Agent执行过程中的各种失败场景：
    1. 单Agent失败：隔离失败Agent，尝试恢复
    2. 多Agent协作失败：回滚整个协作流程
    3. 超时处理：防止Agent长时间阻塞
    4. 重试策略：指数退避重试机制

    设计理念：
    - 快速失败：不要让系统长时间处于不确定状态
    - 优雅降级：部分功能可用时不要完全停机
    - 可观测：所有失败都要有清晰的日志记录
    """

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_count_by_agent: Dict[str, int] = defaultdict(int)
        self.max_failures_before_isolate = 3  # 隔离前的最大失败次数

    def register_circuit_breaker(self, name: str, **kwargs):
        """注册熔断器"""
        self.circuit_breakers[name] = CircuitBreaker(name, **kwargs)

    def handle_failure(
        self,
        agent_name: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理Agent失败

        根据失败类型和上下文，返回处理结果：
        - isolation: 隔离该Agent
        - retry: 重试
        - fallback: 使用备用方案
        - escalate: 升级处理

        Args:
            agent_name: Agent名称
            error: 异常对象
            context: 失败上下文信息

        Returns:
            处理结果字典
        """
        self.failure_count_by_agent[agent_name] += 1
        failure_count = self.failure_count_by_agent[agent_name]

        print(f"[失败处理] Agent {agent_name} 失败 #{failure_count}: {error}")

        # 检查是否有熔断器
        if agent_name in self.circuit_breakers:
            cb = self.circuit_breakers[agent_name]
            try:
                return cb.call(self._do_handle_failure, agent_name, error, context)
            except CircuitOpenException:
                return {
                    "action": "isolate",
                    "reason": "circuit_breaker_open",
                    "agent": agent_name,
                    "message": "Agent已被熔断器隔离"
                }

        return self._do_handle_failure(agent_name, error, context)

    def _do_handle_failure(
        self,
        agent_name: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        实际执行失败处理逻辑

        根据失败次数决定处理策略：
        - 1-2次：记录警告，尝试重试
        - 3次以上：隔离Agent，等待人工介入
        """
        failure_count = self.failure_count_by_agent[agent_name]

        if failure_count <= 2:
            # 前两次失败：记录警告，指数退避重试
            return {
                "action": "retry",
                "agent": agent_name,
                "retry_count": failure_count,
                "backoff_seconds": 2 ** failure_count,  # 退避时间：4, 8, 16...
                "message": f"Agent失败，第{failure_count}次重试"
            }
        else:
            # 超过3次：隔离Agent
            return {
                "action": "isolate",
                "agent": agent_name,
                "failure_count": failure_count,
                "message": f"Agent连续失败{failure_count}次，已隔离"
            }

    def recover_agent(self, agent_name: str):
        """
        恢复被隔离的Agent

        用于人工确认问题解决后，手动恢复Agent

        Args:
            agent_name: Agent名称
        """
        self.failure_count_by_agent[agent_name] = 0
        if agent_name in self.circuit_breakers:
            self.circuit_breakers[agent_name]._reset()
        print(f"[失败处理] Agent {agent_name} 已恢复")


# =============================================================================
# 第六部分：企业架构整合示例
# =============================================================================

class EnterpriseAutoGenArchitecture:
    """
    企业级AutoGen架构整合类

    整合所有组件，形成完整的企业级AI Agent架构：

    组件关系：
    - LLMGateway: 所有LLM调用的统一入口
    - AgentPool: 管理AutoGen Agent的生命周期
    - ConversationTracer: 记录所有对话的完整链路
    - AgentFailureHandler: 处理各种失败场景

    数据流：
    用户请求 -> 限流检查 -> Agent选择 -> LLM调用 -> 追踪记录 -> 响应返回
    """

    def __init__(self):
        print("[企业架构] 初始化企业级AutoGen架构")
        self.llm_gateway = LLMGateway()
        self.agent_pool = AgentPool(self.llm_gateway)
        self.tracer = ConversationTracer()
        self.failure_handler = AgentFailureHandler()
        self._setup_default_agents()

    def _setup_default_agents(self):
        """设置默认的Agent模板"""
        # 代码助手Agent
        self.agent_pool.register_agent_template(AgentConfig(
            name="code_assistant",
            role="代码生成助手",
            system_message="你是一个专业的代码生成助手，帮助租户 {tenant_id} 生成高质量代码。",
            max_consecutive_auto_reply=5
        ))

        # 代码审查Agent
        self.agent_pool.register_agent_template(AgentConfig(
            name="code_reviewer",
            role="代码审查助手",
            system_message="你是一个严格的代码审查助手，帮助租户 {tenant_id} 审查代码质量。",
            max_consecutive_auto_reply=3
        ))

        # 文档助手Agent
        self.agent_pool.register_agent_template(AgentConfig(
            name="doc_assistant",
            role="文档生成助手",
            system_message="你是一个专业的文档助手，帮助租户 {tenant_id} 生成技术文档。",
            max_consecutive_auto_reply=5
        ))

    async def process_request(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        user_message: str
    ) -> Dict[str, Any]:
        """
        处理用户请求

        完整流程：
        1. 开始追踪
        2. 获取Agent
        3. 调用LLM
        4. 记录结果
        5. 返回响应

        Args:
            tenant_id: 租户ID
            conversation_id: 对话会话ID
            agent_name: 使用的Agent名称
            user_message: 用户消息

        Returns:
            处理结果字典
        """
        print(f"\n[企业架构] 处理请求: tenant={tenant_id}, agent={agent_name}")
        print(f"[企业架构] 消息: {user_message[:50]}...")

        # 开始追踪
        trace = self.tracer.start_trace(
            tenant_id=tenant_id,
            agent_name=agent_name,
            agent_role="",  # 会在获取Agent时填充
            conversation_id=conversation_id,
            input_message=user_message
        )

        try:
            # 获取Agent实例
            agent = self.agent_pool.get_agent(tenant_id, agent_name)

            # 调用LLM（通过网关）
            response = await self.llm_gateway.generate(
                tenant_id=tenant_id,
                prompt=user_message,
                model_preference=None
            )

            if response.get("success"):
                # 成功：结束追踪并返回
                self.tracer.end_trace(trace.trace_id, response["response"], response["token_used"])
                return {
                    "success": True,
                    "response": response["response"],
                    "agent": agent["name"],
                    "trace_id": trace.trace_id
                }
            else:
                # LLM调用失败
                self.tracer.fail_trace(trace.trace_id, response.get("message", "unknown error"))
                raise Exception(response.get("message", "LLM调用失败"))

        except Exception as e:
            # Agent执行失败
            self.tracer.fail_trace(trace.trace_id, str(e))

            # 处理失败
            result = self.failure_handler.handle_failure(
                agent_name=agent_name,
                error=e,
                context={"tenant_id": tenant_id, "conversation_id": conversation_id}
            )

            return {
                "success": False,
                "error": str(e),
                "handling": result
            }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取诊断信息

        用于系统监控和问题排查
        """
        return {
            "agent_pool": self.agent_pool.get_pool_stats(),
            "llm_gateway": {
                "registered_models": list(self.llm_gateway.models.keys()),
                "model_weights": self.llm_gateway.model_weights
            },
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self.failure_handler.circuit_breakers.items()
            }
        }


# =============================================================================
# 第七部分：使用示例和测试
# =============================================================================

async def demo_enterprise_architecture():
    """
    演示企业级AutoGen架构

    展示各组件的协同工作流程
    """
    print("=" * 60)
    print("企业级AutoGen架构演示")
    print("=" * 60)

    # 1. 创建架构实例
    arch = EnterpriseAutoGenArchitecture()

    # 2. 注册多个模型
    arch.llm_gateway.register_model(
        ModelType.GPT4,
        api_key="demo-key-1",
        base_url="https://api.openai.com/v1",
        price=(3.0, 15.0),  # GPT-4: $3/输入, $15/输出
        priority=2
    )
    arch.llm_gateway.register_model(
        ModelType.GPT4_MINI,
        api_key="demo-key-2",
        base_url="https://api.openai.com/v1",
        price=(0.15, 0.6),  # GPT-4o-mini: $0.15/输入, $0.6/输出
        priority=1
    )

    # 3. 模拟多租户请求
    tenants = ["tenant_001", "tenant_002"]
    for i, tenant in enumerate(tenants):
        conv_id = f"conv_{i+1}"
        print(f"\n--- 租户 {tenant} 的请求 ---")

        # 调用代码助手
        result1 = await arch.process_request(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="code_assistant",
            user_message="请用Python写一个快速排序算法"
        )
        print(f"结果: {result1.get('success')}, 响应: {result1.get('response', result1.get('error'))[:80]}...")

        # 调用文档助手
        result2 = await arch.process_request(
            tenant_id=tenant,
            conversation_id=conv_id,
            agent_name="doc_assistant",
            user_message="请为快速排序算法生成技术文档"
        )
        print(f"结果: {result2.get('success')}, 响应: {result2.get('response', result2.get('error'))[:80]}...")

    # 4. 获取诊断信息
    print("\n--- 系统诊断信息 ---")
    diag = arch.get_diagnostics()
    print(f"Agent池统计: {diag['agent_pool']}")
    print(f"模型权重: {diag['llm_gateway']['model_weights']}")

    # 5. 获取租户使用统计
    print("\n--- 租户使用统计 ---")
    for tenant in tenants:
        usage = arch.llm_gateway.get_tenant_usage(tenant)
        print(f"{tenant}: {usage['total_requests']} 请求, {usage['total_cost']:.4f} 美元")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


# 程序入口
if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_enterprise_architecture())