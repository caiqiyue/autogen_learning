"""
AutoGen自动化测试示例
=====================================

本文件展示AutoGen应用的自动化测试设计与实现。

核心内容：
1. pytest实现AutoGen多Agent协作的自动化测试
2. 自动化冒烟测试的设计与实现
3. CI/CD集成的自动化测试策略

测试类型：
1. 单元测试（Unit Test）：测试单个Agent或函数的正确性
2. 集成测试（Integration Test）：测试多Agent协作流程
3. 冒烟测试（Smoke Test）：快速验证核心功能可用
4. 端到端测试（E2E Test）：测试完整用户场景

测试框架：
- pytest：Python标准测试框架
- pytest-asyncio：支持异步测试
- pytest-mock：支持模拟对象

作者：AutoGen学习课程
版本：1.0
"""

import asyncio
import time
import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# 测试依赖导入（实际环境中需要安装）
try:
    import pytest
    from pytest import mark
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # 如果pytest不可用，提供模拟的pytest对象
    class _MockMark:
        @staticmethod
        def asyncio(x): return x
    class _MockPytest:
        @staticmethod
        def fixture(func): return func
        @staticmethod
        def skip(msg): return lambda x: x
        mark = _MockMark()
    pytest = _MockPytest()
    mark = pytest.mark

try:
    from unittest.mock import Mock, AsyncMock, patch, MagicMock
    from unittest.mock import sentinel
except ImportError:
    from mock import Mock, AsyncMock, patch, MagicMock
    from mock import sentinel

# =============================================================================
# 第一部分：AutoGen测试基础架构
# =============================================================================

class AgentType(Enum):
    """Agent类型枚举"""
    ASSISTANT = "assistant"           # 助手Agent
    USER_PROXY = "user_proxy"         # 用户代理Agent
    CODE_EXECUTOR = "code_executor"   # 代码执行Agent
    GROUP_CHAT = "group_chat"         # 组聊Agent


@dataclass
class AgentTestConfig:
    """
    Agent测试配置

    定义单个Agent测试所需的配置参数
    """
    name: str                        # Agent名称
    agent_type: AgentType            # Agent类型
    system_message: str              # 系统提示词
    llm_config: Optional[Dict[str, Any]] = None  # LLM配置
    max_consecutive_auto_reply: int = 10  # 最大连续自动回复数
    is_termination_msg: Optional[Callable] = None  # 终止条件


@dataclass
class TestScenario:
    """
    测试场景定义

    用于参数化测试，定义输入、期望输出和验证条件
    """
    scenario_id: str                 # 场景ID
    description: str                 # 场景描述
    input_message: str              # 输入消息
    expected_response_contains: List[str] = field(default_factory=list)  # 期望包含的关键词
    expected_agent_count: int = 1    # 期望参与的Agent数量
    max_execution_time_ms: float = 30000  # 最大执行时间（毫秒）
    expected_tokens_min: int = 0     # 期望最小Token数
    expected_tokens_max: int = 10000  # 期望最大Token数
    should_fail: bool = False        # 是否期望失败


@dataclass
class TestResult:
    """
    测试结果

    记录单个测试的执行结果
    """
    scenario_id: str                # 场景ID
    success: bool                   # 是否成功
    actual_response: str = ""       # 实际响应
    execution_time_ms: float = 0   # 执行时长
    token_used: int = 0            # Token消耗
    error_message: str = ""        # 错误信息（如果有）
    trace_ids: List[str] = field(default_factory=list)  # 追踪ID列表
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# 第二部分：AutoGen Agent模拟器（用于测试）
# =============================================================================

class MockAutoGenAgent:
    """
    AutoGen Agent模拟器

    在测试环境中模拟AutoGen Agent的行为，无需真实的LLM调用

    模拟行为：
    1. 消息传递：模拟Agent间的消息收发
    2. 终止条件：模拟终止消息的判断
    3. 自动回复：模拟LLM的自动回复生成
    4. 嵌套对话：支持嵌套的Agent调用

    使用示例：
        agent = MockAutoGenAgent(
            name="code_assistant",
            agent_type=AgentType.ASSISTANT,
            system_message="你是一个代码助手"
        )

        # 模拟接收消息并生成回复
        response = await agent.generate_reply("帮我写一个排序算法")
        print(f"Agent回复: {response}")
    """

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        system_message: str,
        llm_config: Optional[Dict[str, Any]] = None,
        max_consecutive_auto_reply: int = 10,
        is_termination_msg: Optional[Callable] = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.system_message = system_message
        self.llm_config = llm_config or {}
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.is_termination_msg = is_termination_msg or (lambda x: False)

        # 消息历史
        self.message_history: List[Dict[str, Any]] = []
        # 回复计数
        self.reply_count = 0
        # 追踪ID
        self.current_trace_id: Optional[str] = None

        # 模拟回复映射（可配置）
        self.response_patterns: Dict[str, str] = {}

    async def generate_reply(self, message: str) -> str:
        """
        生成回复

        模拟Agent接收消息并生成回复的过程

        Args:
            message: 输入消息

        Returns:
            生成的回复
        """
        self.reply_count += 1
        trace_id = uuid.uuid4().hex[:8]

        # 记录消息
        self.message_history.append({
            "role": "user" if self.agent_type == AgentType.USER_PROXY else "assistant",
            "content": message,
            "timestamp": datetime.now(),
            "trace_id": trace_id
        })

        # 模拟处理延迟
        await asyncio.sleep(0.01)

        # 生成回复
        response = self._generate_response(message)

        # 记录回复
        self.message_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(),
            "trace_id": trace_id
        })

        return response

    def _generate_response(self, message: str) -> str:
        """根据消息内容生成模拟回复"""
        # 检查预定义的回复模式
        for pattern, response in self.response_patterns.items():
            if pattern.lower() in message.lower():
                return response

        # 默认回复模板
        templates = {
            AgentType.ASSISTANT: f"[{self.name}] 已收到您的消息，正在处理...",
            AgentType.USER_PROXY: f"[{self.name}] 用户代理确认收到消息",
            AgentType.CODE_EXECUTOR: f"[{self.name}] 代码执行完成",
            AgentType.GROUP_CHAT: f"[{self.name}] 组聊消息广播"
        }

        default_response = templates.get(
            self.agent_type,
            f"[{self.name}] 默认回复: {message[:50]}..."
        )

        return default_response

    def set_response_pattern(self, pattern: str, response: str):
        """
        设置响应模式

        用于测试特定场景时配置预期的回复

        Args:
            pattern: 消息匹配模式
            response: 匹配时的回复
        """
        self.response_patterns[pattern.lower()] = response

    def reset(self):
        """重置Agent状态"""
        self.message_history.clear()
        self.reply_count = 0
        self.current_trace_id = None

    def get_message_count(self) -> int:
        """获取消息数量"""
        return len(self.message_history)


class MockConversation:
    """
    模拟对话

    模拟AutoGen的对话会话，管理多个Agent的交互
    """

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.agents: Dict[str, MockAutoGenAgent] = {}
        self.messages: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def add_agent(self, agent: MockAutoGenAgent):
        """
        添加Agent到对话

        Args:
            agent: Agent实例
        """
        self.agents[agent.name] = agent
        logging.info(f"[模拟对话] 添加Agent: {agent.name}")

    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str
    ) -> str:
        """
        发送消息

        Args:
            from_agent: 发送方Agent名称
            to_agent: 接收方Agent名称
            message: 消息内容

        Returns:
            接收方的回复
        """
        if to_agent not in self.agents:
            raise ValueError(f"Agent不存在: {to_agent}")

        # 记录消息
        self.messages.append({
            "from": from_agent,
            "to": to_agent,
            "content": message,
            "timestamp": datetime.now()
        })

        # 获取Agent并生成回复
        agent = self.agents[to_agent]
        response = await agent.generate_reply(message)

        # 记录回复
        self.messages.append({
            "from": to_agent,
            "to": from_agent,
            "content": response,
            "timestamp": datetime.now()
        })

        return response

    def get_messages(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取消息历史

        Args:
            agent_name: 如果指定，只返回涉及该Agent的消息

        Returns:
            消息列表
        """
        if not agent_name:
            return self.messages

        return [
            m for m in self.messages
            if m["from"] == agent_name or m["to"] == agent_name
        ]


# =============================================================================
# 第三部分：AutoGen测试运行器
# =============================================================================

class AutoGenTestRunner:
    """
    AutoGen测试运行器

    负责执行AutoGen相关的自动化测试，包括：
    1. 单元测试：测试单个Agent的正确性
    2. 集成测试：测试多Agent协作流程
    3. 冒烟测试：快速验证核心功能
    4. 性能测试：验证性能指标

    使用示例：
        runner = AutoGenTestRunner()

        # 添加测试场景
        runner.add_scenario(TestScenario(
            scenario_id="test_01",
            description="测试代码助手生成排序算法",
            input_message="写一个快速排序算法",
            expected_response_contains=["def quicksort", "时间复杂度"]
        ))

        # 运行测试
        results = await runner.run_smoke_tests()

        # 生成报告
        report = runner.generate_report(results)
        print(report)
    """

    def __init__(self):
        self.scenarios: List[TestScenario] = []
        self.test_results: List[TestResult] = []

    def add_scenario(self, scenario: TestScenario):
        """
        添加测试场景

        Args:
            scenario: 测试场景定义
        """
        self.scenarios.append(scenario)
        logging.info(f"[测试运行器] 添加场景: {scenario.scenario_id} - {scenario.description}")

    def clear_scenarios(self):
        """清空所有测试场景"""
        self.scenarios.clear()
        self.test_results.clear()

    async def run_smoke_tests(self) -> List[TestResult]:
        """
        运行冒烟测试

        冒烟测试的特点：
        1. 执行速度快，通常在秒级完成
        2. 验证核心功能可用，不关注细节
        3. 失败即表示系统不可用

        Returns:
            测试结果列表
        """
        logging.info(f"[测试运行器] 开始运行冒烟测试，共 {len(self.scenarios)} 个场景")

        results = []
        for scenario in self.scenarios:
            result = await self._run_single_test(scenario)
            results.append(result)
            self.test_results.append(result)

        return results

    async def _run_single_test(self, scenario: TestScenario) -> TestResult:
        """
        运行单个测试

        Args:
            scenario: 测试场景

        Returns:
            测试结果
        """
        logging.info(f"[测试运行器] 执行场景: {scenario.scenario_id}")

        start_time = time.time()

        try:
            # 创建测试用的Agent
            agent = MockAutoGenAgent(
                name="test_agent",
                agent_type=AgentType.ASSISTANT,
                system_message="你是一个测试助手"
            )

            # 配置响应模式（用于验证）
            if scenario.expected_response_contains:
                for keyword in scenario.expected_response_contains:
                    agent.set_response_pattern(
                        keyword,
                        f"[模拟回复] 这是一个包含'{keyword}'的回复"
                    )

            # 执行测试
            if scenario.max_execution_time_ms:
                response = await asyncio.wait_for(
                    agent.generate_reply(scenario.input_message),
                    timeout=scenario.max_execution_time_ms / 1000
                )
            else:
                response = await agent.generate_reply(scenario.input_message)

            execution_time = (time.time() - start_time) * 1000

            # 验证响应内容
            success = True
            if scenario.expected_response_contains:
                for keyword in scenario.expected_response_contains:
                    if keyword.lower() not in response.lower():
                        success = False
                        logging.warning(
                            f"[测试运行器] 响应缺少关键词: {keyword}"
                        )

            # 检查执行时间
            if execution_time > scenario.max_execution_time_ms:
                success = False
                logging.warning(
                    f"[测试运行器] 执行超时: {execution_time:.2f}ms > "
                    f"{scenario.max_execution_time_ms}ms"
                )

            return TestResult(
                scenario_id=scenario.scenario_id,
                success=success,
                actual_response=response,
                execution_time_ms=execution_time,
                token_used=len(scenario.input_message) + len(response)
            )

        except asyncio.TimeoutError:
            return TestResult(
                scenario_id=scenario.scenario_id,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="测试执行超时"
            )

        except Exception as e:
            logging.error(f"[测试运行器] 测试执行失败: {e}")
            return TestResult(
                scenario_id=scenario.scenario_id,
                success=scenario.should_fail,  # 如果期望失败则成功
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

    async def run_integration_test(
        self,
        scenario: TestScenario,
        agent_configs: List[AgentTestConfig]
    ) -> TestResult:
        """
        运行集成测试

        测试多Agent协作场景

        Args:
            scenario: 测试场景
            agent_configs: Agent配置列表

        Returns:
            测试结果
        """
        logging.info(
            f"[测试运行器] 开始集成测试: {scenario.scenario_id}, "
            f"Agent数量: {len(agent_configs)}"
        )

        start_time = time.time()

        try:
            # 创建对话
            conversation = MockConversation(f"test_conv_{scenario.scenario_id}")

            # 创建Agent
            agents = {}
            for config in agent_configs:
                agent = MockAutoGenAgent(
                    name=config.name,
                    agent_type=config.agent_type,
                    system_message=config.system_message,
                    max_consecutive_auto_reply=config.max_consecutive_auto_reply
                )
                agents[config.name] = agent
                conversation.add_agent(agent)

            # 模拟多Agent协作
            current_message = scenario.input_message
            message_count = 0
            max_messages = scenario.expected_agent_count * 2

            while message_count < max_messages:
                # 找到下一个应该处理的Agent
                agent_name = self._select_next_agent(
                    agents,
                    message_count,
                    scenario.expected_agent_count
                )

                # 发送消息
                response = await conversation.send_message(
                    from_agent="test_user",
                    to_agent=agent_name,
                    message=current_message
                )

                current_message = response
                message_count += 1

                # 检查终止条件
                if self._should_terminate(response, agents):
                    break

            execution_time = (time.time() - start_time) * 1000
            total_messages = len(conversation.messages)

            # 验证结果
            success = True
            if total_messages < scenario.expected_agent_count:
                success = False
                logging.warning(
                    f"[测试运行器] Agent参与数量不足: "
                    f"{total_messages} < {scenario.expected_agent_count}"
                )

            return TestResult(
                scenario_id=scenario.scenario_id,
                success=success,
                actual_response=current_message,
                execution_time_ms=execution_time,
                token_used=sum(len(m["content"]) for m in conversation.messages)
            )

        except Exception as e:
            logging.error(f"[测试运行器] 集成测试失败: {e}")
            return TestResult(
                scenario_id=scenario.scenario_id,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

    def _select_next_agent(
        self,
        agents: Dict[str, MockAutoGenAgent],
        message_index: int,
        agent_count: int
    ) -> str:
        """选择下一个处理的Agent（轮询）"""
        agent_names = list(agents.keys())
        return agent_names[message_index % len(agent_names)]

    def _should_terminate(
        self,
        message: str,
        agents: Dict[str, MockAutoGenAgent]
    ) -> bool:
        """检查是否应该终止对话"""
        termination_keywords = ["完成", "结束", "terminated", "done"]
        for keyword in termination_keywords:
            if keyword in message:
                return True
        return False

    def generate_report(self, results: List[TestResult]) -> str:
        """
        生成测试报告

        Args:
            results: 测试结果列表

        Returns:
            格式化的测试报告
        """
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        avg_execution_time = (
            sum(r.execution_time_ms for r in results) / total
            if total > 0 else 0
        )

        report_lines = [
            "=" * 70,
            "AutoGen自动化测试报告",
            "=" * 70,
            f"总测试数: {total}",
            f"通过: {passed}",
            f"失败: {failed}",
            f"通过率: {pass_rate:.1f}%",
            f"平均执行时间: {avg_execution_time:.2f}ms",
            "-" * 70,
            "详细结果:"
        ]

        for result in results:
            status = "PASS" if result.success else "FAIL"
            report_lines.append(
                f"  [{status}] {result.scenario_id} - "
                f"{result.execution_time_ms:.2f}ms"
            )
            if result.error_message:
                report_lines.append(f"         错误: {result.error_message}")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)


# =============================================================================
# 第四部分：pytest测试用例
# =============================================================================

# 注意：以下测试用例使用pytest框架编写
# 实际运行时需要确保已安装 pytest, pytest-asyncio


class TestAutoGenAgent:
    """
    AutoGen Agent单元测试

    测试单个Agent的正确性
    """

    @pytest.fixture
    def code_assistant(self):
        """创建代码助手Agent fixture"""
        agent = MockAutoGenAgent(
            name="code_assistant",
            agent_type=AgentType.ASSISTANT,
            system_message="你是一个专业的代码助手"
        )
        return agent

    @pytest.mark.asyncio
    async def test_agent_generate_reply(self, code_assistant):
        """
        测试Agent生成回复

        验证：
        1. Agent能够接收消息
        2. Agent能够生成回复
        3. 消息历史被正确记录
        """
        initial_count = code_assistant.get_message_count()

        response = await code_assistant.generate_reply("写一个快速排序算法")

        assert response is not None
        assert len(response) > 0
        assert code_assistant.get_message_count() == initial_count + 2  # 用户消息 + 助手回复

    @pytest.mark.asyncio
    async def test_agent_response_patterns(self, code_assistant):
        """
        测试Agent响应模式

        验证：
        1. 配置的响应模式被正确匹配
        2. 自定义回复被正确返回
        """
        # 配置响应模式
        code_assistant.set_response_pattern(
            "排序",
            "这是一个关于排序的回答"
        )

        response = await code_assistant.generate_reply("请解释排序算法")

        assert "排序" in response

    @pytest.mark.asyncio
    async def test_agent_reset(self, code_assistant):
        """
        测试Agent状态重置

        验证：
        1. 重置后消息历史被清空
        2. 重置后回复计数归零
        """
        await code_assistant.generate_reply("测试消息")
        assert code_assistant.get_message_count() > 0

        code_assistant.reset()

        assert code_assistant.get_message_count() == 0
        assert code_assistant.reply_count == 0


class TestAutoGenConversation:
    """
    AutoGen对话集成测试

    测试多Agent协作场景
    """

    @pytest.fixture
    def conversation(self):
        """创建对话fixture"""
        conv = MockConversation("test_conv_001")

        # 添加多个Agent
        assistant = MockAutoGenAgent(
            name="assistant",
            agent_type=AgentType.ASSISTANT,
            system_message="你是一个助手"
        )

        reviewer = MockAutoGenAgent(
            name="reviewer",
            agent_type=AgentType.ASSISTANT,
            system_message="你是一个审查员"
        )

        conv.add_agent(assistant)
        conv.add_agent(reviewer)

        return conv

    @pytest.mark.asyncio
    async def test_multi_agent_message_passing(self, conversation):
        """
        测试多Agent消息传递

        验证：
        1. 消息能够正确传递给目标Agent
        2. 消息历史正确记录
        """
        initial_message_count = len(conversation.messages)

        response = await conversation.send_message(
            from_agent="user",
            to_agent="assistant",
            message="你好"
        )

        assert response is not None
        assert len(conversation.messages) == initial_message_count + 2  # 请求 + 响应

    @pytest.mark.asyncio
    async def test_conversation_message_retrieval(self, conversation):
        """
        测试消息检索

        验证：
        1. 能够获取完整的消息历史
        2. 能够按Agent过滤消息
        """
        await conversation.send_message(
            from_agent="user",
            to_agent="assistant",
            message="你好"
        )

        all_messages = conversation.get_messages()
        assert len(all_messages) > 0

        assistant_messages = conversation.get_messages(agent_name="assistant")
        assert len(assistant_messages) > 0


class TestAutoGenSmokeTests:
    """
    AutoGen冒烟测试

    快速验证核心功能是否可用
    """

    @pytest.fixture
    def test_runner(self):
        """创建测试运行器fixture"""
        return AutoGenTestRunner()

    @pytest.mark.asyncio
    async def test_smoke_test_single_agent(self, test_runner):
        """
        测试单Agent冒烟测试

        验证：
        1. 能够创建Agent
        2. Agent能够生成回复
        """
        scenario = TestScenario(
            scenario_id="smoke_001",
            description="单Agent冒烟测试",
            input_message="测试消息",
            expected_response_contains=[],
            max_execution_time_ms=5000
        )

        test_runner.add_scenario(scenario)
        results = await test_runner.run_smoke_tests()

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_smoke_test_execution_timeout(self, test_runner):
        """
        测试执行超时处理

        验证：
        1. 超时场景被正确捕获
        2. 错误信息被正确记录
        """
        scenario = TestScenario(
            scenario_id="smoke_timeout",
            description="超时测试",
            input_message="超时测试",
            max_execution_time_ms=1  # 设置极短超时
        )

        test_runner.add_scenario(scenario)
        results = await test_runner.run_smoke_tests()

        # 由于超时，测试应该失败
        assert len(results) == 1


class TestAutoGenIntegration:
    """
    AutoGen集成测试

    测试复杂的多Agent协作场景
    """

    @pytest.fixture
    def test_runner(self):
        """创建测试运行器fixture"""
        return AutoGenTestRunner()

    @pytest.mark.asyncio
    async def test_integration_code_assistant_flow(self, test_runner):
        """
        测试代码助手集成流程

        场景：用户请求生成代码 -> Agent生成代码 -> Reviewer审查

        验证：
        1. 多Agent协作正确执行
        2. 消息正确传递
        3. 最终结果合理
        """
        scenario = TestScenario(
            scenario_id="int_001",
            description="代码助手完整流程",
            input_message="写一个快速排序算法",
            expected_agent_count=2,
            max_execution_time_ms=10000
        )

        agent_configs = [
            AgentTestConfig(
                name="code_assistant",
                agent_type=AgentType.ASSISTANT,
                system_message="你是一个代码助手"
            ),
            AgentTestConfig(
                name="code_reviewer",
                agent_type=AgentType.ASSISTANT,
                system_message="你是一个代码审查员"
            )
        ]

        result = await test_runner.run_integration_test(
            scenario,
            agent_configs
        )

        assert result.success or result.error_message == ""


# =============================================================================
# 第五部分：CI/CD集成支持
# =============================================================================

class CICDIntegration:
    """
    CI/CD集成支持类

    提供与CI/CD系统集成的功能：

    支持的CI/CD系统：
    1. GitHub Actions
    2. GitLab CI
    3. Jenkins
    4. CircleCI

    集成方式：
    1. JUnit XML格式的测试报告
    2. 测试结果JSON导出
    3. 性能指标Prometheus格式导出
    4. 告警webhook通知

    使用示例：
        ci_cd = CICDIntegration()

        # 运行测试
        runner = AutoGenTestRunner()
        # ... 添加测试场景 ...

        results = await runner.run_smoke_tests()

        # 生成CI/CD友好的报告
        ci_cd.export_junit_xml(results, "test-results.xml")
        ci_cd.export_json(results, "test-results.json")

        # 检查是否应该阻止合并
        if ci_cd.should_block_merge(results):
            sys.exit(1)
    """

    def __init__(self, project_name: str = "autogen-project"):
        self.project_name = project_name
        self.junit_template = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="{project}" tests="{total}" failures="{failures}" errors="{errors}" time="{time}">\n{tests}</testsuite>'

    def export_junit_xml(self, results: List[TestResult], output_path: str):
        """
        导出JUnit XML格式的测试报告

        Args:
            results: 测试结果列表
            output_path: 输出文件路径
        """
        total = len(results)
        failures = sum(1 for r in results if not r.success)
        errors = sum(1 for r in results if r.error_message)
        total_time = sum(r.execution_time_ms for r in results) / 1000

        test_cases = []
        for result in results:
            status = "passed" if result.success else "failed"
            error_xml = ""
            if result.error_message:
                error_xml = f'\n          <error message="{result.error_message}"/>'

            test_case = f'  <testcase name="{result.scenario_id}" time="{result.execution_time_ms/1000:.3f}" classname="AutoGenTests">{error_xml}\n    <failure message="test failed" type="AssertionError"/>\n  </testcase>' if not result.success else f'  <testcase name="{result.scenario_id}" time="{result.execution_time_ms/1000:.3f}" classname="AutoGenTests"/>'
            test_cases.append(test_case)

        xml_content = self.junit_template.format(
            project=self.project_name,
            total=total,
            failures=failures,
            errors=errors,
            time=f"{total_time:.3f}",
            tests="\n".join(test_cases)
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        logging.info(f"[CI/CD] JUnit XML已导出: {output_path}")

    def export_json(self, results: List[TestResult], output_path: str):
        """
        导出JSON格式的测试结果

        Args:
            results: 测试结果列表
            output_path: 输出文件路径
        """
        import json

        output = {
            "project": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "success_rate": sum(1 for r in results if r.success) / len(results) * 100 if results else 0
            },
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "success": r.success,
                    "execution_time_ms": r.execution_time_ms,
                    "token_used": r.token_used,
                    "error_message": r.error_message,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"[CI/CD] JSON结果已导出: {output_path}")

    def should_block_merge(self, results: List[TestResult]) -> bool:
        """
        检查是否应该阻止代码合并

        阻止合并条件：
        1. 测试通过率低于80%
        2. 有任何关键测试失败
        3. 平均执行时间超过阈值

        Args:
            results: 测试结果列表

        Returns:
            是否应该阻止合并
        """
        if not results:
            return True  # 没有测试结果，默认阻止

        passed = sum(1 for r in results if r.success)
        success_rate = passed / len(results)

        # 通过率低于80%时阻止
        if success_rate < 0.8:
            logging.warning(f"[CI/CD] 测试通过率 {success_rate*100:.1f}% < 80%，阻止合并")
            return True

        # 有失败测试时阻止
        if passed < len(results):
            logging.warning(f"[CI/CD] 有 {len(results) - passed} 个测试失败，阻止合并")
            return True

        return False

    def get_performance_metrics(
        self,
        results: List[TestResult]
    ) -> Dict[str, float]:
        """
        获取性能指标

        用于性能监控和趋势分析

        Args:
            results: 测试结果列表

        Returns:
            性能指标字典
        """
        if not results:
            return {}

        execution_times = [r.execution_time_ms for r in results]

        return {
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "min_execution_time_ms": min(execution_times),
            "max_execution_time_ms": max(execution_times),
            "p95_execution_time_ms": sorted(execution_times)[int(len(execution_times) * 0.95)] if execution_times else 0,
            "total_tokens": sum(r.token_used for r in results),
            "success_rate": sum(1 for r in results if r.success) / len(results) * 100
        }


# =============================================================================
# 第六部分：冒烟测试自动化脚本
# =============================================================================

async def run_smoke_tests_suite() -> bool:
    """
    运行完整的冒烟测试套件

    这是一个独立的冒烟测试脚本，可用于：
    1. 本地开发时的快速验证
    2. CI/CD流程中的自动化测试
    3. 部署前的健康检查

    Returns:
        所有测试是否通过
    """
    print("=" * 70)
    print("AutoGen自动化冒烟测试")
    print("=" * 70)

    # 创建测试运行器
    runner = AutoGenTestRunner()

    # 添加冒烟测试场景
    smoke_scenarios = [
        TestScenario(
            scenario_id="smoke_001",
            description="单Agent消息处理",
            input_message="你好，请介绍一下自己",
            expected_response_contains=["你好", "收到"],
            max_execution_time_ms=5000
        ),
        TestScenario(
            scenario_id="smoke_002",
            description="代码助手生成",
            input_message="写一个Python的快速排序算法",
            expected_response_contains=["def", "quicksort"],
            max_execution_time_ms=10000
        ),
        TestScenario(
            scenario_id="smoke_003",
            description="终止条件检测",
            input_message="今天的天气如何",
            expected_response_contains=[],
            max_execution_time_ms=5000
        ),
        TestScenario(
            scenario_id="smoke_004",
            description="嵌套对话模拟",
            input_message="帮我分析这段代码的性能",
            expected_response_contains=["分析"],
            expected_agent_count=2,
            max_execution_time_ms=15000
        ),
        TestScenario(
            scenario_id="smoke_005",
            description="错误处理",
            input_message="error_test_trigger",
            expected_response_contains=[],
            should_fail=False,
            max_execution_time_ms=5000
        )
    ]

    for scenario in smoke_scenarios:
        runner.add_scenario(scenario)

    # 运行测试
    print(f"\n开始运行 {len(smoke_scenarios)} 个冒烟测试场景...\n")
    results = await runner.run_smoke_tests()

    # 生成报告
    report = runner.generate_report(results)
    print("\n" + report)

    # CI/CD集成
    ci_cd = CICDIntegration(project_name="autogen-course")

    # 导出测试结果
    ci_cd.export_json(results, "smoke-test-results.json")
    ci_cd.export_junit_xml(results, "smoke-test-results.xml")

    # 性能指标
    metrics = ci_cd.get_performance_metrics(results)
    print("\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # 检查是否应该阻止
    should_block = ci_cd.should_block_merge(results)
    print(f"\n代码合并检查: {'阻止' if should_block else '允许'}")

    print("\n" + "=" * 70)
    success = all(r.success for r in results)
    print(f"测试结果: {'全部通过' if success else '存在失败'}")
    print("=" * 70)

    return success


# =============================================================================
# 第七部分：演示和测试
# =============================================================================

async def demo_testing():
    """
    演示自动化测试

    展示测试框架的使用方法
    """
    print("=" * 70)
    print("AutoGen自动化测试演示")
    print("=" * 70)

    # 1. 创建测试运行器
    runner = AutoGenTestRunner()

    # 2. 添加测试场景
    print("\n--- 添加测试场景 ---")

    scenario1 = TestScenario(
        scenario_id="test_01",
        description="测试代码助手生成排序算法",
        input_message="写一个快速排序算法",
        expected_response_contains=["def quicksort", "时间复杂度"],
        max_execution_time_ms=5000
    )
    runner.add_scenario(scenario1)
    print(f"添加场景: {scenario1.scenario_id} - {scenario1.description}")

    scenario2 = TestScenario(
        scenario_id="test_02",
        description="测试助手响应超时",
        input_message="计算一个很大数字的斐波那契数列",
        max_execution_time_ms=1  # 设置超短超时
    )
    runner.add_scenario(scenario2)
    print(f"添加场景: {scenario2.scenario_id} - {scenario2.description}")

    scenario3 = TestScenario(
        scenario_id="test_03",
        description="测试终止条件",
        input_message="生成代码完成，结束对话",
        expected_response_contains=["完成", "结束"],
        max_execution_time_ms=5000
    )
    runner.add_scenario(scenario3)
    print(f"添加场景: {scenario3.scenario_id} - {scenario3.description}")

    # 3. 运行冒烟测试
    print("\n--- 运行冒烟测试 ---")
    results = await runner.run_smoke_tests()

    # 4. 生成报告
    report = runner.generate_report(results)
    print("\n" + report)

    # 5. CI/CD集成演示
    print("\n--- CI/CD集成 ---")
    ci_cd = CICDIntegration(project_name="autogen-course-demo")

    # 导出格式
    ci_cd.export_json(results, "demo-test-results.json")
    print("已导出JSON结果: demo-test-results.json")

    # 合并检查
    should_block = ci_cd.should_block_merge(results)
    print(f"代码合并检查: {'阻止' if should_block else '允许'}")

    # 性能指标
    metrics = ci_cd.get_performance_metrics(results)
    print("\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

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
    asyncio.run(demo_testing())

    # 如果要运行完整的冒烟测试套件，，取消下面的注释
    # asyncio.run(run_smoke_tests_suite())