# async_advanced.py
# 第20节 异步Agent高级用法演示
#
# 本文件展示异步Agent的高级用法，包括：
# 1. asyncio.gather实现多Agent并发对话
# 2. 嵌套对话中的异步消息传递
# 3. 异步GroupChat的实现与超时异常处理
#
# ============================================================
# 高级异步编程概念
# ============================================================
#
# 1. 并发 vs 并行：
#    - 并发（Concurrency）：多个任务交替执行，通过事件循环切换
#    - 并行（Parallelism）：多个任务同时执行，需要多核CPU
#    - asyncio.gather 实现协程级别的并发
#
# 2. asyncio.gather：
#    - 接收多个协程，同时启动它们
#    - 等待所有协程完成
#    - 返回所有结果的列表
#    - 适用于I/O密集型任务（如LLM API调用）
#
# 3. 嵌套对话（Nested Chat）：
#    - 一个Agent可以在处理消息时发起另一个对话
#    - 父对话与子对话可以独立运行
#    - 异步环境下，嵌套对话可以并发执行
#
# ============================================================

import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import time

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager, Agent
except ImportError:
    print("错误：请先安装autogen库：pip install autogen")
    sys.exit(1)


# ============================================================
# 第一部分：环境配置（与async_basic.py保持一致）
# ============================================================

def load_env(env_path: str = ".env") -> None:
    """从.env文件加载环境变量"""
    path = Path(env_path)
    if not path.exists():
        print(f"警告：未找到{env_path}文件，请确保环境变量已正确设置")
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_llm_config():
    """构建AutoGen的LLM配置"""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
    base_url = os.getenv("OPENAI_BASE_URL")

    config = {
        "config_list": [{
            "model": model,
            "api_key": api_key,
        }]
    }

    if base_url:
        config["config_list"][0]["base_url"] = base_url

    return config


# ============================================================
# 第二部分：asyncio.gather并发执行
# ============================================================

def demo_asyncio_gather_basic():
    """
    演示asyncio.gather的基本用法

    asyncio.gather是实现并发执行的核心函数：
    - 接收多个协程作为参数
    - 同时启动所有协程（不按顺序等待）
    - 返回所有结果的有序列表

    与串行执行相比，gather可以大幅减少总执行时间。
    """
    print("\n" + "=" * 60)
    print("演示1: asyncio.gather并发执行基础")
    print("=" * 60)

    async def task_1():
        """模拟LLM API调用（耗时1秒）"""
        print("  [任务1] 开始处理...")
        await asyncio.sleep(1)
        print("  [任务1] 处理完成")
        return "任务1结果"

    async def task_2():
        """模拟LLM API调用（耗时0.5秒）"""
        print("  [任务2] 开始处理...")
        await asyncio.sleep(0.5)
        print("  [任务2] 处理完成")
        return "任务2结果"

    async def task_3():
        """模拟LLM API调用（耗时1.5秒）"""
        print("  [任务3] 开始处理...")
        await asyncio.sleep(1.5)
        print("  [任务3] 处理完成")
        return "任务3结果"

    # 串行执行：总时间 = 1 + 0.5 + 1.5 = 3秒
    print("\n1. 串行执行（不推荐）:")
    print("   总时间 = 1s + 0.5s + 1.5s = 3s")

    async def run_serial():
        start = time.time()
        result1 = await task_1()
        result2 = await task_2()
        result3 = await task_3()
        elapsed = time.time() - start
        return [result1, result2, result3], elapsed

    serial_results, serial_time = asyncio.run(run_serial())
    print(f"   实际耗时: {serial_time:.2f}秒")
    print(f"   结果: {serial_results}")

    # 并发执行：总时间 = max(1, 0.5, 1.5) = 1.5秒
    print("\n2. 并发执行（使用asyncio.gather，推荐）:")
    print("   总时间 = max(1s, 0.5s, 1.5s) = 1.5s")

    async def run_concurrent():
        start = time.time()
        # gather同时启动所有协程
        results = await asyncio.gather(
            task_1(),
            task_2(),
            task_3()
        )
        elapsed = time.time() - start
        return results, elapsed

    concurrent_results, concurrent_time = asyncio.run(run_concurrent())
    print(f"   实际耗时: {concurrent_time:.2f}秒")
    print(f"   结果: {list(concurrent_results)}")

    # 时间对比
    print(f"\n性能对比:")
    print(f"   串行执行: {serial_time:.2f}秒")
    print(f"   并发执行: {concurrent_time:.2f}秒")
    print(f"   加速比: {serial_time/concurrent_time:.2f}x")


def demo_multi_agent_concurrent_chat():
    """
    演示多Agent并发对话

    使用asyncio.gather实现多个Agent同时与用户对话，
    收集所有Agent的响应后进行汇总处理。

    适用场景：
    - 并行咨询：多个专家Agent同时提供意见
    - 投票机制：收集多个Agent的投票结果
    - 竞态分析：让多个Agent从不同角度分析问题
    """
    print("\n" + "=" * 60)
    print("演示2: 多Agent并发对话")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建多个专家Agent（不同角色）
    analyst_agent = ConversableAgent(
        name="分析师",
        system_message="你是一位数据分析师，擅长从数据角度分析问题。回答尽量简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    engineer_agent = ConversableAgent(
        name="工程师",
        system_message="你是一位软件工程师，擅长从技术实现角度分析问题。回答尽量简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    business_agent = ConversableAgent(
        name="商务专家",
        system_message="你是一位商务专家，擅长从业务价值角度分析问题。回答尽量简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 用户代理（发起请求）
    user_proxy = ConversableAgent(
        name="用户代理",
        system_message="你代表用户提问。",
        llm_config=False,
        human_input_mode="NEVER",
    )

    async def query_single_agent(agent: ConversableAgent, query: str) -> str:
        """
        向单个Agent发送查询并获取响应

        Args:
            agent: 目标Agent
            query: 查询内容

        Returns:
            Agent的响应内容
        """
        # 调用Agent的异步回复方法
        response = await agent.a_generate_reply(
            messages=[{"content": query, "role": "user"}],
            sender=user_proxy
        )
        return response

    async def run_concurrent_consultation():
        """
        并发咨询多个Agent

        场景：用户提出一个问题，多个专家同时从各自角度提供分析。
        """
        query = "请分析Python和JavaScript在前端开发中的各自优劣势"

        print(f"\n查询内容: {query}")
        print("\n并发咨询3个专家...")

        start = time.time()

        # 并发查询所有Agent
        responses = await asyncio.gather(
            query_single_agent(analyst_agent, query),
            query_single_agent(engineer_agent, query),
            query_single_agent(business_agent, query),
        )

        elapsed = time.time() - start

        return responses, elapsed

    # 执行并发咨询
    print("\n" + "-" * 40)
    responses, elapsed = asyncio.run(run_concurrent_consultation())
    print("-" * 40)

    print(f"\n总耗时: {elapsed:.2f}秒")
    print("\n专家观点汇总:")

    expert_names = ["分析师", "工程师", "商务专家"]
    for name, response in zip(expert_names, responses):
        print(f"\n【{name}】")
        # 截断过长输出
        display = response[:200] + "..." if len(response) > 200 else response
        print(f"  {display}")


def demo_concurrent_with_error_handling():
    """
    演示带错误处理的并发执行

    asyncio.gather在遇到异常时的行为可以通过参数控制：
    - return_exceptions=False（默认）：任何一个失败，整体失败
    - return_exceptions=True：捕获异常作为结果返回，不影响其他任务

    推荐做法：使用return_exceptions=True，然后检查每个结果是否为异常。
    """
    print("\n" + "=" * 60)
    print("演示3: 带错误处理的并发执行")
    print("=" * 60)

    async def successful_task(task_id: int):
        """模拟成功执行的任务"""
        await asyncio.sleep(0.5)
        return f"任务{task_id}完成"

    async def failing_task(task_id: int):
        """模拟失败的任务"""
        await asyncio.sleep(0.3)
        raise ValueError(f"任务{task_id}执行失败！")

    async def run_with_error_handling():
        # 场景1：默认行为（遇到异常立即失败）
        print("\n场景1: return_exceptions=False（默认行为）")
        print("  任何一个任务失败，整体立即失败")

        try:
            results = await asyncio.gather(
                successful_task(1),
                failing_task(2),
                successful_task(3),
            )
            print(f"  结果: {results}")
        except ValueError as e:
            print(f"  捕获异常: {e}")

        # 场景2：捕获异常（不中断其他任务）
        print("\n场景2: return_exceptions=True（推荐做法）")
        print("  异常作为结果返回，不中断其他任务")

        results = await asyncio.gather(
            successful_task(1),
            failing_task(2),
            successful_task(3),
            return_exceptions=True  # 关键参数：捕获异常
        )

        print(f"  结果数量: {len(results)}")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  任务{i+1}: 异常 - {result}")
            else:
                print(f"  任务{i+1}: 成功 - {result}")

    asyncio.run(run_with_error_handling())

    # 在AutoGen Agent调用中的错误处理示例
    print("\n场景3: AutoGen Agent并发调用的错误处理")
    print("  使用try-except捕获每个Agent的异常")

    llm_config = build_llm_config()

    async def call_agent_safely(agent, message):
        """安全调用Agent，捕获异常"""
        try:
            response = await agent.a_generate_reply(
                messages=[{"content": message, "role": "user"}],
                sender=None
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # 创建测试Agent
    test_agent = ConversableAgent(
        name="测试Agent",
        system_message="你是一个测试Agent。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    async def safe_concurrent_calls():
        """并发安全调用多个Agent"""
        results = await asyncio.gather(
            call_agent_safely(test_agent, "你好"),
            call_agent_safely(test_agent, "今天天气如何"),  # 正常消息
            # 模拟一个会失败的情况：传入无效参数
        )

        print("\n  并发调用结果:")
        for i, result in enumerate(results, 1):
            if result["success"]:
                print(f"    Agent{i}: 成功")
            else:
                print(f"    Agent{i}: 失败 - {result['error']}")

    asyncio.run(safe_concurrent_calls())


# ============================================================
# 第三部分：异步GroupChat
# ============================================================

def demo_async_groupchat():
    """
    演示异步GroupChat的使用

    GroupChat在异步环境下的行为：
    - 使用GroupChatManager管理群聊
    - 消息通过Manager广播给所有Agent
    - 可以使用asyncio.gather实现多GroupChat并发

    注意：GroupChatManager本身已经支持异步操作，
    无需额外配置即可在异步环境中使用。
    """
    print("\n" + "=" * 60)
    print("演示4: 异步GroupChat")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建多个Agent
    researcher = ConversableAgent(
        name="研究员",
        system_message="你是一位研究员，负责收集和分析信息。回答简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    critic = ConversableAgent(
        name="评论员",
        system_message="你是一位评论员，负责提出批评意见和改进建议。回答简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    synthesizer = ConversableAgent(
        name="综合员",
        system_message="你是一位综合员，负责整合各方观点形成结论。回答简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 创建GroupChat
    groupchat = GroupChat(
        agents=[researcher, critic, synthesizer],
        messages=[],
        max_round=5,
    )

    # 创建GroupChatManager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print("\n创建异步GroupChat:")
    print(f"  - 参与Agent: {len(groupchat.agents)}个")
    print(f"  - 最大轮次: {groupchat.max_round}")
    print(f"  - speaker_selection_method: {groupchat.speaker_selection_method}")

    async def run_async_groupchat():
        """异步执行GroupChat"""

        print("\n启动异步群聊...")
        print("-" * 40)

        # 使用initiate_chat启动群聊
        # 在异步环境中调用，底层会自动处理
        chat_result = await researcher.a_initiate_chat(
            manager,
            message="请分析人工智能对就业市场的影响。",
            max_turns=5,
        )

        return chat_result

    # 执行异步GroupChat
    result = asyncio.run(run_async_groupchat())

    print("-" * 40)
    print(f"群聊完成:")
    print(f"  - 消息数: {len(result.chat_history)}")
    print(f"  - 最后一条消息: {result.chat_history[-1]['content'][:50]}...")


def demo_multiple_groupchat_concurrent():
    """
    演示多个GroupChat并发执行

    这是高级用法：同时运行多个独立的GroupChat，
    每个GroupChat处理不同的主题。

    适用场景：
    - 并行研究：多个团队同时研究不同主题
    - 多角度分析：同一问题多个GroupChat从不同角度分析
    """
    print("\n" + "=" * 60)
    print("演示5: 多个GroupChat并发执行")
    print("=" * 60)

    llm_config = build_llm_config()

    def create_groupchat(name_prefix: str, topic: str) -> tuple:
        """
        创建一个简单的GroupChat

        Returns:
            (agents, groupchat, manager, topic)
        """
        # 创建两个简单Agent
        agent1 = ConversableAgent(
            name=f"{name_prefix}_专家A",
            system_message=f"你是{name_prefix}专家A。专注讨论：{topic}",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        agent2 = ConversableAgent(
            name=f"{name_prefix}_专家B",
            system_message=f"你是{name_prefix}专家B。专注讨论：{topic}",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        # 创建GroupChat
        groupchat = GroupChat(
            agents=[agent1, agent2],
            messages=[],
            max_round=3,
        )

        # 创建Manager
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
        )

        return agent1, groupchat, manager, topic

    # 创建多个GroupChat
    gc1_agent, gc1, gc1_mgr, topic1 = create_groupchat("经济", "AI对经济的影响")
    gc2_agent, gc2, gc2_mgr, topic2 = create_groupchat("教育", "AI对教育的影响")
    gc3_agent, gc3, gc3_mgr, topic3 = create_groupchat("医疗", "AI对医疗的影响")

    print("\n创建3个并发GroupChat:")
    print(f"  1. 经济组 - 主题: {topic1}")
    print(f"  2. 教育组 - 主题: {topic2}")
    print(f"  3. 医疗组 - 主题: {topic3}")

    async def run_concurrent_groupchats():
        """并发运行多个GroupChat"""

        print("\n并发启动3个GroupChat...")
        print("-" * 40)

        start = time.time()

        # 使用gather并发启动所有GroupChat
        results = await asyncio.gather(
            gc1_agent.a_initiate_chat(gc1_mgr, message=f"讨论主题：{topic1}", max_turns=3),
            gc2_agent.a_initiate_chat(gc2_mgr, message=f"讨论主题：{topic2}", max_turns=3),
            gc3_agent.a_initiate_chat(gc3_mgr, message=f"讨论主题：{topic3}", max_turns=3),
        )

        elapsed = time.time() - start

        return results, elapsed

    # 执行并发GroupChat
    results, elapsed = asyncio.run(run_concurrent_groupchats())

    print("-" * 40)
    print(f"\n并发执行完成:")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  如果串行执行，预计需要3倍以上时间")

    print("\n各组讨论结果:")
    for i, (name, topic) in enumerate([("经济组", topic1), ("教育组", topic2), ("医疗组", topic3)]):
        result = results[i]
        last_msg = result.chat_history[-1]['content'][:80] if result.chat_history else "无"
        print(f"\n  【{name}】主题: {topic}")
        print(f"    消息数: {len(result.chat_history)}")
        print(f"    总结: {last_msg}...")


# ============================================================
# 第四部分：异步超时与异常处理
# ============================================================

def demo_async_timeout_exception_handling():
    """
    演示异步环境下的超时与异常处理

    异步环境中的异常处理需要注意：
    1. asyncio.TimeoutError 是 asyncio 内置的超时异常
    2. 需要区分正常超时和系统错误
    3. 超时后应该妥善清理资源

    推荐的超时处理模式：
    - 使用 asyncio.wait_for 添加超时
    - 捕获 asyncio.TimeoutError 处理超时情况
    - 提供有意义的错误信息
    """
    print("\n" + "=" * 60)
    print("演示6: 异步超时与异常处理")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建一个"慢"Agent用于测试超时
    slow_agent = ConversableAgent(
        name="慢速Agent",
        system_message="你是一个响应很慢的助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 注册一个模拟慢速响应的reply_func
    async def slow_reply(messages, sender, config):
        """模拟慢速处理（可通过config配置延迟）"""
        delay = config.get("delay", 2.0) if config else 2.0
        print(f"    [慢速Agent] 开始处理，预计延迟{delay}秒...")
        await asyncio.sleep(delay)
        return f"处理完成，延迟了{delay}秒"

    slow_agent.register_reply(
        trigger=None,
        reply_func=slow_reply,
        position=0,
        config={"delay": 3.0}  # 默认3秒延迟
    )

    print("\n场景1: 正常完成的调用")
    print("  处理时间: 0.5秒，超时: 2秒")

    async def test_normal_completion():
        """测试正常完成的场景"""
        try:
            # 临时修改延迟为0.5秒
            slow_agent.register_reply(
                trigger=None,
                reply_func=slow_reply,
                position=0,
                config={"delay": 0.5}
            )

            result = await asyncio.wait_for(
                slow_agent.a_generate_reply(
                    messages=[{"content": "快速响应", "role": "user"}],
                    sender=None
                ),
                timeout=2.0
            )
            return f"成功: {result}"
        except asyncio.TimeoutError:
            return "超时"

    result = asyncio.run(test_normal_completion())
    print(f"  结果: {result}")

    print("\n场景2: 超时未完成的调用")
    print("  处理时间: 3秒，超时: 1秒")
    print("  预期: asyncio.TimeoutError异常")

    async def test_timeout():
        """测试超时场景"""
        try:
            # 恢复3秒延迟
            slow_agent.register_reply(
                trigger=None,
                reply_func=slow_reply,
                position=0,
                config={"delay": 3.0}
            )

            result = await asyncio.wait_for(
                slow_agent.a_generate_reply(
                    messages=[{"content": "需要超时", "role": "user"}],
                    sender=None
                ),
                timeout=1.0  # 1秒超时
            )
            return f"成功: {result}"
        except asyncio.TimeoutError:
            return "超时：操作在规定时间内未完成"

    result = asyncio.run(test_timeout())
    print(f"  结果: {result}")

    print("\n场景3: 使用asyncio.timeout上下文管理器（Python 3.11+）")
    print("  这是更现代的超时控制API")

    async def test_timeout_context():
        """使用上下文管理器的超时控制"""
        try:
            async with asyncio.timeout(1.0) as cm:
                # 模拟一个需要2秒的操作
                await asyncio.sleep(2)
                return "操作完成"
        except asyncio.TimeoutError:
            # 判断是否是我们的超时（cm.deadline）
            return f"超时：操作未在{cm.deadline:.1f}秒内完成"

    result = asyncio.run(test_timeout_context())
    print(f"  结果: {result}")

    print("\n场景4: 带重试的超时处理")
    print("  当超时时，自动重试一次（使用更短的超时时间）")

    async def call_with_retry(agent, message, max_retries=2):
        """
        带重试的调用

        策略：
        1. 首次调用使用正常超时
        2. 如果超时，重试一次（可设置更短超时）
        3. 两次都超时则返回错误信息
        """
        for attempt in range(max_retries):
            try:
                timeout = 2.0 if attempt == 0 else 1.0  # 首次2秒，重试1秒
                result = await asyncio.wait_for(
                    agent.a_generate_reply(
                        messages=[{"content": message, "role": "user"}],
                        sender=None
                    ),
                    timeout=timeout
                )
                return f"成功(第{attempt+1}次): {result}"
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    print(f"    第{attempt+1}次超时，准备重试...")
                    continue
                else:
                    return f"失败：连续{max_retries}次超时"

        return "未知错误"

    # 恢复3秒延迟
    slow_agent.register_reply(
        trigger=None,
        reply_func=slow_reply,
        position=0,
        config={"delay": 3.0}
    )

    result = asyncio.run(call_with_retry(slow_agent, "测试重试"))
    print(f"  结果: {result}")


def demo_async_context_variables():
    """
    演示异步上下文变量（Context Variables）

    在异步编程中，上下文变量允许在协程之间传递请求级数据。
    这对于：
    - 请求追踪（trace_id）
    - 日志上下文
    - 用户会话信息

    Python 3.7+ 引入了 contextvars 模块。
    """
    print("\n" + "=" * 60)
    print("演示7: 异步上下文变量")
    print("=" * 60)

    from contextvars import ContextVar

    # 定义上下文变量
    request_id: ContextVar[str] = ContextVar("request_id")
    user_session: ContextVar[dict] = ContextVar("user_session")

    async def child_task(task_name: str):
        """子任务：可以访问父任务的上下文变量"""
        # 获取上下文变量
        req_id = request_id.get("未设置")
        session = user_session.get({})

        print(f"  [{task_name}] request_id={req_id}, user={session.get('name', '未知')}")

        # 模拟处理
        await asyncio.sleep(0.1)

        return f"{task_name}完成"

    async def parent_task(task_name: str):
        """父任务：设置上下文变量并启动子任务"""
        print(f"[{task_name}] 设置上下文...")

        # 设置上下文变量（只在这个协程及其子协程中可见）
        request_id.set(f"req-{task_name}-001")
        user_session.set({"name": f"用户{task_name}", "tier": "vip"})

        # 启动子任务
        result = await child_task(f"{task_name}_child")

        return result

    async def run_with_context():
        """演示上下文变量的隔离性"""
        print("\n并发运行多个任务，观察上下文变量的隔离:")

        # 并发运行两个父任务
        results = await asyncio.gather(
            parent_task("任务A"),
            parent_task("任务B"),
        )

        return results

    results = asyncio.run(run_with_context())
    print(f"\n结果: {results}")


# ============================================================
# 第五部分：综合示例
# ============================================================

async def demo_async_agent_pool():
    """
    演示异步Agent池的概念

    Agent池是一种设计模式：
    - 预先创建一组Agent实例
    - 根据负载动态分配任务
    - 减少Agent创建销毁的开销

    在异步环境下，Agent池可以高效利用事件循环。
    """
    print("\n" + "=" * 60)
    print("演示8: 异步Agent池概念")
    print("=" * 60)

    llm_config = build_llm_config()

    class AsyncAgentPool:
        """
        简单的异步Agent池实现

        特性：
        - 预先创建固定数量的Agent
        - 使用asyncio.Queue管理可用Agent
        - 获取Agent时从队列取，用完后放回
        """

        def __init__(self, agent_class, agent_configs, pool_size=3):
            """
            初始化Agent池

            Args:
                agent_class: Agent类（如ConversableAgent）
                agent_configs: Agent配置列表
                pool_size: 池大小
            """
            self.pool_size = pool_size
            self.agent_class = agent_class
            self.agent_configs = agent_configs
            self.available_agents = asyncio.Queue()
            self.agents = []

        async def initialize(self):
            """异步初始化Agent池中的Agent实例"""
            for i, config in enumerate(self.agent_configs[:self.pool_size]):
                agent = self.agent_class(**config)
                self.agents.append(agent)
                await self.available_agents.put(agent)
            print(f"  Agent池初始化完成: {self.pool_size}个Agent")

        async def get_agent(self):
            """从池中获取一个可用的Agent"""
            agent = await self.available_agents.get()
            return agent

        async def release_agent(self, agent):
            """将Agent归还到池中"""
            await self.available_agents.put(agent)

        async def execute_task(self, task_fn, *args, **kwargs):
            """
            在池中执行任务

            Args:
                task_fn: 任务函数，接收agent作为第一个参数
            """
            agent = await self.get_agent()
            try:
                result = await task_fn(agent, *args, **kwargs)
                return result
            finally:
                # 确保Agent被归还到池中
                await self.release_agent(agent)

    # 创建Agent配置
    agent_configs = [
        {
            "name": f"池Agent_{i}",
            "system_message": "你是一个高效的助手。",
            "llm_config": llm_config,
            "human_input_mode": "NEVER",
        }
        for i in range(3)
    ]

    print("\n创建异步Agent池...")
    pool = AsyncAgentPool(ConversableAgent, agent_configs, pool_size=3)
    await pool.initialize()
    print(f"  Agent池初始化完成: {pool.pool_size}个Agent")

    async def sample_task(agent, task_id):
        """示例任务：让Agent处理一条消息"""
        response = await agent.a_generate_reply(
            messages=[{"content": f"任务{task_id}：简短回复", "role": "user"}],
            sender=None
        )
        return f"任务{task_id} -> {response[:30]}..."

    print("\n提交5个并发任务到池（池大小为3）...")
    print("  注意：虽然池只有3个Agent，但可以高效处理5个并发任务")

    async def run_pool_demo():
        """运行Agent池演示"""
        tasks = [sample_task(None, i) for i in range(5)]

        # 使用gather并发执行（Agent会被复用）
        results = await asyncio.gather(*[
            pool.execute_task(sample_task, i) for i in range(5)
        ])

        return results

    results = asyncio.run(run_pool_demo())

    print("\n任务结果:")
    for result in results:
        print(f"  {result}")

    print("\nAgent池优势:")
    print("  1. 减少Agent创建销毁开销")
    print("  2. 限制并发数，防止资源耗尽")
    print("  3. 适用于高负载场景")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：运行所有异步高级用法演示
    """
    print("=" * 60)
    print("AutoGen异步Agent高级用法演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_asyncio_gather_basic()
    demo_multi_agent_concurrent_chat()
    demo_concurrent_with_error_handling()
    demo_async_groupchat()
    demo_multiple_groupchat_concurrent()
    demo_async_timeout_exception_handling()
    demo_async_context_variables()
    demo_async_agent_pool()

    print("\n" + "=" * 60)
    print("高级演示完成")
    print("=" * 60)
    print("\n高级知识点总结:")
    print("  1. asyncio.gather: 并发执行多个协程，提高吞吐量")
    print("  2. return_exceptions=True: 捕获异常，不中断其他任务")
    print("  3. 异步GroupChat: 多Agent协作的高级模式")
    print("  4. asyncio.wait_for/timeout: 超时控制")
    print("  5. 上下文变量: 在异步任务间传递请求级数据")
    print("  6. Agent池: 高效管理Agent实例的设计模式")


if __name__ == "__main__":
    main()