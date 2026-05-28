---
lesson_id: lesson_26
title: AutoGen可观测性与自动化测试
module: 企业级AutoGen应用架构设计
---

# 第26节 AutoGen可观测性与自动化测试

## 学习目标

- 掌握AutoGen应用的监控与可观测性设计
- 理解对话链路追踪、Token消耗监控、异常告警的实现方法
- 能够实现pytest自动化测试
- 掌握CI/CD集成的自动化测试策略

## 内容概述

企业级AutoGen应用需要完善的可观测性设计来监控运行状态、追踪问题和优化性能。本节将深入解析可观测性三支柱（日志、指标、追踪）的实现，以及如何使用pytest实现自动化测试。

---

## 1. 可观测性三支柱

### 1.1 可观测性概念

可观测性（Observability）是指系统能够通过外部输出推断内部状态的能力。对于AutoGen应用，主要包含三个层面：

| 层面 | 说明 | 核心作用 |
|------|------|----------|
| **日志（Logs）** | 记录系统运行事件 | 问题排查、审计追踪 |
| **指标（Metrics）** | 量化系统运行状态 | 性能监控、容量规划 |
| **追踪（Traces）** | 串联请求完整路径 | 链路分析、性能优化 |

### 1.2 AutoGen可观测性设计目标

```
Agent执行 -> Tracer记录链路 -> Monitor记录Token -> Metrics更新指标 -> 告警检查
```

设计目标：
- **零侵入**：通过装饰器和钩子实现最小化代码改动
- **多租户支持**：所有指标和日志按租户隔离
- **实时性**：支持实时监控和告警
- **可扩展**：支持对接Prometheus、Grafana等外部系统

---

## 2. 对话链路追踪 (ConversationTracer)

### 2.1 追踪数据结构

```python
@dataclass
class ConversationTrace:
    """对话链路追踪记录"""
    trace_id: str                    # 唯一追踪ID
    tenant_id: str                   # 租户ID
    conversation_id: str             # 对话会话ID
    agent_name: str                  # 当前Agent名称
    parent_trace_id: Optional[str] = None  # 父追踪ID（用于嵌套对话）
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
```

### 2.2 追踪器核心功能

```python
class ConversationTracer:
    """
    对话链路追踪器

    实现AutoGen应用的全链路追踪，支持：
    1. 嵌套追踪：支持Agent嵌套调用，通过parent_trace_id串联
    2. 并发追踪：支持多线程/多协程并发执行
    3. 性能分析：记录每个环节的耗时
    4. 错误追踪：记录失败信息
    """

    def start_trace(
        self,
        tenant_id: str,
        conversation_id: str,
        agent_name: str,
        input_message: str,
        parent_trace_id: Optional[str] = None,
    ) -> 'ConversationTrace':
        """开始一个追踪"""
        trace_id = self._generate_trace_id()

        trace = ConversationTrace(
            trace_id=trace_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            parent_trace_id=parent_trace_id,
            input_message=self._truncate(input_message, 500),
        )

        self.traces[trace_id] = trace
        return trace

    def end_trace(self, trace_id: str, output_message: str, token_used: int, status: str = "completed"):
        """结束一个追踪"""
        trace = self.traces[trace_id]
        trace.end_time = datetime.now()
        trace.output_message = self._truncate(output_message, 500)
        trace.token_used = token_used
        trace.status = status
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

    def get_trace_tree(self, conversation_id: str) -> Dict[str, Any]:
        """
        获取追踪树（用于可视化）

        将扁平的所有追踪记录构建成树形结构
        """
        traces = self.get_conversation_traces(conversation_id, include_children=True)
        trace_map = {t.trace_id: t for t in traces}
        roots = [t for t in traces if not t.parent_trace_id]

        def build_tree(trace: 'ConversationTrace') -> Dict[str, Any]:
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
```

---

## 3. Token消耗监控 (TokenMonitor)

### 3.1 Token记录结构

```python
@dataclass
class TokenRecord:
    """Token使用记录"""
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
```

### 3.2 监控器核心功能

```python
class TokenMonitor:
    """
    Token消耗监控器

    监控和统计LLM的Token消耗情况，支持：
    1. 租户级别统计：每个租户的Token使用量
    2. 模型级别统计：每个模型的调用次数和Token量
    3. 实时成本计算：根据模型价格实时计算成本
    4. 配额管理：监控租户配额使用情况
    """

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
        """记录一次Token使用"""
        total_tokens = prompt_tokens + completion_tokens

        record = TokenRecord(
            record_id=uuid.uuid4().hex[:12],
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=self._calculate_cost(model_name, prompt_tokens, completion_tokens)
        )

        self.records[record.record_id] = record
        self._update_tenant_stats(tenant_id, total_tokens, record.cost, latency_ms)
        self._update_model_stats(model_name, prompt_tokens, completion_tokens, latency_ms)

    def set_model_price(self, model_name: str, input_price: float, output_price: float):
        """设置模型价格（每百万Token）"""
        self.model_prices[model_name] = (input_price, output_price)

    def set_quota(self, tenant_id: str, monthly_limit: int, daily_limit: int):
        """设置租户配额"""
        self.quota_configs[tenant_id] = QuotaConfig(
            tenant_id=tenant_id,
            monthly_limit=monthly_limit,
            daily_limit=daily_limit
        )

    def get_quota_usage(self, tenant_id: str) -> Dict[str, Any]:
        """获取租户配额使用情况"""
        config = self.quota_configs.get(tenant_id)
        stats = self.tenant_stats.get(tenant_id, TokenStats())

        return {
            "tenant_id": tenant_id,
            "monthly_limit": config.monthly_limit,
            "monthly_used": stats.monthly_tokens.get(month_key, 0),
            "monthly_remaining": max(0, config.monthly_limit - monthly_used),
            "usage_percent": round(monthly_used / config.monthly_limit * 100, 2),
            "daily_limit": config.daily_limit,
            "daily_used": stats.daily_tokens.get(today_key, 0),
            "daily_usage_percent": round(daily_used / config.daily_limit * 100, 2)
        }
```

### 3.3 模型价格配置示例

```python
# 配置模型价格（每百万Token）
monitor.setup_model_prices({
    "gpt-4": (3.0, 15.0),           # $3/输入, $15/输出
    "gpt-4o-mini": (0.15, 0.6),     # $0.15/输入, $0.6/输出
    "qwen2.5": (0.16, 0.64)          # ¥1.12/输入, ¥4.48/输出
})

# 配置租户配额
monitor.setup_quotas({
    "tenant_001": {"monthly": 1_000_000, "daily": 50_000},
    "tenant_002": {"monthly": 5_000_000, "daily": 200_000}
})
```

---

## 4. 异常告警系统 (AlertManager)

### 4.1 告警级别定义

```python
class AlertLevel(Enum):
    """告警级别枚举"""
    DEBUG = "debug"           # 调试级：仅记录
    INFO = "info"             # 信息级：重要事件通知
    WARNING = "warning"       # 警告级：潜在问题
    ERROR = "error"           # 错误级：一般错误
    CRITICAL = "critical"     # 严重级：系统不可用
```

### 4.2 告警规则配置

```python
@dataclass
class AlertRule:
    """告警规则定义"""
    name: str                          # 规则名称
    metric_name: str                  # 关联的指标名称
    condition: str                     # 触发条件，如 " > " 或 " >= "
    threshold: float                   # 阈值
    level: AlertLevel                  # 告警级别
    cooldown_seconds: int = 60         # 冷却时间（秒）
    enabled: bool = True               # 是否启用
```

### 4.3 告警管理器功能

```python
class AlertManager:
    """
    异常告警管理系统

    实现：
    1. 聚合：将短时间内的多个相同告警合并
    2. 抑制：高级别告警发生时自动抑制低级别告警
    3. 升级：告警未及时处理时自动升级
    4. 冷却：触发后进入冷却期，防止告警风暴
    """

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""

    def trigger(self, rule_name: str, metric_value: float,
                tenant_id: Optional[str] = None, message: Optional[str] = None) -> Optional[Alert]:
        """
        触发告警

        检查规则并触发告警，支持冷却期控制
        """
        rule = self.rules.get(rule_name)
        if not rule or not rule.enabled:
            return None

        # 检查冷却期
        if self._is_in_cooldown(rule_name, rule.cooldown_seconds):
            return None

        # 检查触发条件
        if not self._check_condition(rule, metric_value):
            return None

        # 创建告警
        alert = Alert(
            alert_id=uuid.uuid4().hex[:12],
            rule_name=rule_name,
            level=rule.level,
            message=message,
            metric_value=metric_value,
            tenant_id=tenant_id
        )

        self.active_alerts[alert.alert_id] = alert
        self._dispatch_alert(alert)

        return alert

    def get_active_alerts(self, level: Optional[AlertLevel] = None,
                          tenant_id: Optional[str] = None) -> List[Alert]:
        """获取活跃告警"""
```

### 4.4 默认告警规则

```python
def _setup_default_alert_rules(self):
    """设置默认告警规则"""
    # 高Token使用告警（每日超过80%）
    self.alert_manager.add_rule(AlertRule(
        name="high_daily_token_usage",
        metric_name="tenant_tokens_daily",
        condition=">=",
        threshold=80000,
        level=AlertLevel.WARNING,
        cooldown_seconds=3600
    ))

    # 配额耗尽告警
    self.alert_manager.add_rule(AlertRule(
        name="quota_exceeded",
        metric_name="quota_usage_percent",
        condition=">=",
        threshold=100,
        level=AlertLevel.CRITICAL,
        cooldown_seconds=300
    ))

    # Agent连续失败告警
    self.alert_manager.add_rule(AlertRule(
        name="agent_consecutive_failures",
        metric_name="agent_failure_count",
        condition=">=",
        threshold=5,
        level=AlertLevel.ERROR,
        cooldown_seconds=600
    ))
```

---

## 5. Prometheus监控指标暴露

### 5.1 核心指标定义

```python
class PrometheusMetrics:
    """
    Prometheus监控指标暴露器

    指标类型：
    1. Counter（计数器）：请求总数、成功次数、失败次数
    2. Gauge（仪表）：当前活跃请求数、Agent数量
    3. Histogram（直方图）：响应时间分布、Token消耗分布

    核心指标：
    - autogen_requests_total: 总请求数
    - autogen_request_duration_seconds: 请求处理时长分布
    - autogen_tokens_total: 总Token消耗
    - autogen_active_conversations: 当前活跃对话数
    - autogen_agent_errors_total: Agent错误总数
    """

    def _create_metrics(self):
        """创建Prometheus指标"""
        self.requests_total = Counter(
            "autogen_requests_total",
            "AutoGen总请求数",
            ["tenant_id", "agent_name", "status"]
        )

        self.request_duration = Histogram(
            "autogen_request_duration_seconds",
            "AutoGen请求处理时长",
            ["tenant_id", "agent_name"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )

        self.tokens_total = Counter(
            "autogen_tokens_total",
            "AutoGen总Token消耗",
            ["tenant_id", "model", "token_type"]
        )

        self.active_requests = Gauge(
            "autogen_active_requests",
            "当前活跃请求数",
            ["tenant_id", "agent_name"]
        )
```

### 5.2 指标记录示例

```python
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

# Flask/Gin等Web框架集成
# @app.route('/metrics')
# def metrics():
#     return metrics.get_metrics()
```

---

## 6. pytest自动化测试

### 6.1 测试场景定义

```python
@dataclass
class TestScenario:
    """测试场景定义"""
    scenario_id: str                 # 场景ID
    description: str                 # 场景描述
    input_message: str              # 输入消息
    expected_response_contains: List[str] = field(default_factory=list)
    expected_agent_count: int = 1
    max_execution_time_ms: float = 30000
    expected_tokens_min: int = 0
    expected_tokens_max: int = 10000
    should_fail: bool = False
```

### 6.2 Agent模拟器

```python
class MockAutoGenAgent:
    """
    AutoGen Agent模拟器

    在测试环境中模拟AutoGen Agent的行为，无需真实的LLM调用
    """

    def __init__(self, name: str, agent_type: AgentType,
                 system_message: str, max_consecutive_auto_reply: int = 10):
        self.name = name
        self.agent_type = agent_type
        self.system_message = system_message
        self.message_history: List[Dict[str, Any]] = []
        self.reply_count = 0
        self.response_patterns: Dict[str, str] = {}

    async def generate_reply(self, message: str) -> str:
        """生成回复"""
        self.reply_count += 1
        self.message_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })

        response = self._generate_response(message)

        self.message_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })

        return response

    def set_response_pattern(self, pattern: str, response: str):
        """设置响应模式"""
        self.response_patterns[pattern.lower()] = response
```

### 6.3 测试运行器

```python
class AutoGenTestRunner:
    """AutoGen测试运行器"""

    async def run_smoke_tests(self) -> List[TestResult]:
        """
        运行冒烟测试

        冒烟测试特点：
        1. 执行速度快，通常在秒级完成
        2. 验证核心功能可用
        3. 失败即表示系统不可用
        """
        results = []
        for scenario in self.scenarios:
            result = await self._run_single_test(scenario)
            results.append(result)
        return results

    async def run_integration_test(self, scenario: TestScenario,
                                   agent_configs: List[AgentTestConfig]) -> TestResult:
        """运行集成测试，测试多Agent协作场景"""
        conversation = MockConversation(f"test_conv_{scenario.scenario_id}")

        # 创建Agent
        agents = {}
        for config in agent_configs:
            agent = MockAutoGenAgent(
                name=config.name,
                agent_type=config.agent_type,
                system_message=config.system_message,
            )
            agents[config.name] = agent
            conversation.add_agent(agent)

        # 模拟多Agent协作
        current_message = scenario.input_message
        message_count = 0

        while message_count < scenario.expected_agent_count * 2:
            response = await conversation.send_message(
                from_agent="test_user",
                to_agent=agent_name,
                message=current_message
            )
            current_message = response
            message_count += 1

        return TestResult(
            scenario_id=scenario.scenario_id,
            success=True,
            actual_response=current_message,
            execution_time_ms=execution_time,
            token_used=sum(len(m["content"]) for m in conversation.messages)
        )
```

### 6.4 pytest测试用例

```python
class TestAutoGenAgent:
    """AutoGen Agent单元测试"""

    @pytest.fixture
    def code_assistant(self):
        """创建代码助手Agent fixture"""
        return MockAutoGenAgent(
            name="code_assistant",
            agent_type=AgentType.ASSISTANT,
            system_message="你是一个专业的代码助手"
        )

    @pytest.mark.asyncio
    async def test_agent_generate_reply(self, code_assistant):
        """测试Agent生成回复"""
        response = await code_assistant.generate_reply("写一个快速排序算法")
        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_agent_response_patterns(self, code_assistant):
        """测试Agent响应模式"""
        code_assistant.set_response_pattern("排序", "这是一个关于排序的回答")
        response = await code_assistant.generate_reply("请解释排序算法")
        assert "排序" in response


class TestAutoGenSmokeTests:
    """AutoGen冒烟测试"""

    @pytest.fixture
    def test_runner(self):
        return AutoGenTestRunner()

    @pytest.mark.asyncio
    async def test_smoke_test_single_agent(self, test_runner):
        """测试单Agent冒烟测试"""
        scenario = TestScenario(
            scenario_id="smoke_001",
            description="单Agent冒烟测试",
            input_message="测试消息",
            max_execution_time_ms=5000
        )
        test_runner.add_scenario(scenario)
        results = await test_runner.run_smoke_tests()
        assert len(results) == 1
        assert results[0].success
```

---

## 7. CI/CD集成

### 7.1 CI/CD集成类

```python
class CICDIntegration:
    """
    CI/CD集成支持类

    支持的CI/CD系统：
    1. GitHub Actions
    2. GitLab CI
    3. Jenkins
    4. CircleCI

    集成方式：
    1. JUnit XML格式的测试报告
    2. 测试结果JSON导出
    3. 性能指标Prometheus格式导出
    """

    def export_junit_xml(self, results: List[TestResult], output_path: str):
        """导出JUnit XML格式的测试报告"""

    def export_json(self, results: List[TestResult], output_path: str):
        """导出JSON格式的测试结果"""

    def should_block_merge(self, results: List[TestResult]) -> bool:
        """
        检查是否应该阻止代码合并

        阻止合并条件：
        1. 测试通过率低于80%
        2. 有任何关键测试失败
        3. 平均执行时间超过阈值
        """
        passed = sum(1 for r in results if r.success)
        success_rate = passed / len(results)

        if success_rate < 0.8:
            return True
        if passed < len(results):
            return True

        return False

    def get_performance_metrics(self, results: List[TestResult]) -> Dict[str, float]:
        """获取性能指标"""
        return {
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "min_execution_time_ms": min(execution_times),
            "max_execution_time_ms": max(execution_times),
            "p95_execution_time_ms": sorted(execution_times)[int(len(execution_times) * 0.95)],
            "total_tokens": sum(r.token_used for r in results),
            "success_rate": sum(1 for r in results if r.success) / len(results) * 100
        }
```

### 7.2 CI/CD工作流示例

```bash
# .github/workflows/autogen-test.yml
name: AutoGen Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install autogen pytest pytest-asyncio

      - name: Run smoke tests
        run: |
          python -m pytest tests/smoke_tests.py -v

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test-results.xml
```

---

## 代码案例

本节包含两个代码案例，请参考 `26_codes/` 目录：

### 案例1：observability.py

**内容要点：**
- 对话链路追踪器（ConversationTracer）
- Token消耗监控器（TokenMonitor）
- 异常告警系统（AlertManager）
- Prometheus指标暴露（PrometheusMetrics）
- 可观测性装饰器（@observable）

**运行方式：**
```bash
cd part_09_企业级AutoGen应用架构设计/26_codes
python observability.py
```

**输出示例：**
```
============================================================
AutoGen可观测性设计演示
============================================================
[可观测性] 初始化可观测性管理器
[可观测性] 配置模型价格: gpt-4 = $3.0/$15.0

--- 租户 tenant_001 的对话 ---
[追踪] 开始: trace_id=a1b2c3d4e5f6
[Token监控] 记录: tenant=tenant_001, model=gpt-4, tokens=500, cost=$2.2500

--- 租户统计 ---
tenant_001: 2 请求, 1000 Token, $4.5000

--- 配额使用情况 ---
tenant_001: 月度 1000000/1000000 (0.1%), 每日 50000/50000 (0.2%)
```

### 案例2：autogen_testing.py

**内容要点：**
- pytest测试框架设计
- Agent模拟器（MockAutoGenAgent）
- 测试运行器（AutoGenTestRunner）
- 冒烟测试设计
- CI/CD集成（JUnit XML、JSON导出）

**运行方式：**
```bash
cd part_09_企业级AutoGen应用架构设计/26_codes
python autogen_testing.py
```

**输出示例：**
```
============================================================
AutoGen自动化测试演示
============================================================
[测试运行器] 添加场景: test_01 - 测试代码助手生成排序算法
[测试运行器] 添加场景: test_02 - 测试助手响应超时
[测试运行器] 添加场景: test_03 - 测试终止条件

--- 运行冒烟测试 ---
[测试运行器] 执行场景: test_01
[测试运行器] 执行场景: test_02
[测试运行器] 执行场景: test_03

============================================================
AutoGen自动化测试报告
============================================================
总测试数: 3
通过: 3
失败: 0
通过率: 100.0%
平均执行时间: 15.23ms
------------------------------------------------------------
详细结果:
  [PASS] test_01 - 12.34ms
  [PASS] test_02 - 18.45ms
  [PASS] test_03 - 14.90ms
============================================================

--- CI/CD集成 ---
已导出JSON结果: demo-test-results.json
代码合并检查: 允许
```

---

## 本节小结

1. **可观测性三支柱**：日志（问题排查）、指标（性能监控）、追踪（链路分析）

2. **对话链路追踪**：
   - 支持嵌套追踪，通过parent_trace_id串联
   - 支持追踪树可视化
   - 用于问题排查和性能分析

3. **Token消耗监控**：
   - 租户级别和模型级别统计
   - 实时成本计算
   - 配额管理和告警

4. **异常告警系统**：
   - 多级告警（DEBUG/INFO/WARNING/ERROR/CRITICAL）
   - 告警聚合、抑制、冷却机制
   - 支持自定义告警规则

5. **Prometheus集成**：
   - Counter/Gauge/Histogram指标类型
   - 支持外部监控系统集成

6. **pytest自动化测试**：
   - MockAutoGenAgent用于无真实LLM调用测试
   - 冒烟测试快速验证核心功能
   - 集成测试验证多Agent协作

7. **CI/CD集成**：
   - JUnit XML格式测试报告
   - JSON格式测试结果导出
   - 自动阻止合并条件检查

---

## 延伸阅读

- [企业级AutoGen架构设计](./24_企业级AutoGen架构设计.md)
- [Vibe Coding工作流与AI辅助编程](./25_Vibe_Coding工作流与AI辅助编程.md)
- [Pytest官方文档](https://docs.pytest.org/)
- [Prometheus监控指南](https://prometheus.io/docs/introduction/overview/)

---

## 下节预告

完成本节学习后，建议继续深入学习：
- AutoGen与Kubernetes集成实现弹性扩缩容
- AutoGen安全策略与权限管理
- AutoGen生产环境最佳实践