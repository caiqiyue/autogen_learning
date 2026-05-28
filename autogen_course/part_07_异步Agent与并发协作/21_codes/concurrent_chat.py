# concurrent_chat.py
# 第21节 嵌套对话与并发协作机制 - 并发协作配置演示
#
# 本文件演示并发协作的配置与使用，包括：
# 1. asyncio.gather实现多Agent并发对话收集结果
# 2. 异步环境下的超时控制
# 3. 超时异常处理机制
# 4. 并发任务的结果汇总
#
# ============================================================
# 并发协作（Concurrent Collaboration）核心概念
# ============================================================
#
# 并发协作是指多个Agent或群聊同时工作，以加快任务处理速度。
# AutoGen支持通过asyncio实现并发对话。
#
# 关键机制：
# 1. asyncio.gather：同时启动多个协程，收集所有结果
# 2. 超时控制：设置任务的最大执行时间
# 3. 异常处理：处理并发执行中的超时和错误
#
# 使用场景：
# - 并行咨询多个专家团队
# - 同时处理多个独立子任务
# - 加快复杂任务的处理速度
#
# ============================================================

import os
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# ============================================================
# 第一部分：环境配置加载
# ============================================================

def load_env(env_path: str = ".env") -> None:
    """
    从 .env 文件加载环境变量

    Args:
        env_path: .env 文件路径，默认为当前目录下的 .env
    """
    path = Path(env_path)
    if not path.exists():
        print(f"警告：未找到 {env_path} 文件，请确保环境变量已正确设置")
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_required_env(name: str) -> str:
    """
    获取必需的环境变量，如果不存在则抛出异常

    Args:
        name: 环境变量名称

    Returns:
        环境变量的值

    Raises:
        RuntimeError: 当环境变量未设置时
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少必需的环境变量: {name}，请在 .env 文件中配置")
    return value


# ============================================================
# 第二部分：LLM 配置构建
# ============================================================

def build_llm_config():
    """
    构建 AutoGen 的 LLM 配置

    Returns:
        dict: 包含模型配置的字典
    """
    model = get_required_env("OPENAI_MODEL")
    api_key = get_required_env("OPENAI_API_KEY")
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
# 第三部分：asyncio.gather并发协作
# ============================================================

async def run_single_chat(
    agent: 'ConversableAgent',
    recipient: 'ConversableAgent',
    message: str,
    max_round: int = 5
) -> 'ChatResult':
    """
    运行单个对话（异步）

    Args:
        agent: 发起对话的Agent
        recipient: 接收消息的Agent
        message: 初始消息
        max_round: 最大轮次

    Returns:
        ChatResult: 对话结果
    """
    # 使用ainitiate_chat进行异步对话
    result = await agent.a_initiate_chat(
        recipient,
        message=message,
        max_round=max_round,
    )
    return result


async def demo_asyncio_gather_basic():
    """
    演示使用asyncio.gather实现多Agent并发对话

    asyncio.gather的核心用法：
    1. 收集多个协程的返回值
    2. 所有协程同时执行
    3. 等待所有结果返回

    适用场景：
    - 并行咨询多个专家
    - 同时执行多个独立任务
    - 加快处理速度
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示1: asyncio.gather并发协作基础")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建多个Agent（并行工作）
    # ---------------------------------------------

    # 分析师Agent
    analyst = ConversableAgent(
        name="分析师",
        system_message="你是市场分析师，提供市场分析报告。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 设计师Agent
    designer = ConversableAgent(
        name="设计师",
        system_message="你是UI设计师，提供界面设计方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 开发者Agent
    developer = ConversableAgent(
        name="开发者",
        system_message="你是后端开发者，提供技术实现方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个并行Agent: 分析师、设计师、开发者")

    # ---------------------------------------------
    # 并行执行多个对话
    # ---------------------------------------------

    print("\n使用asyncio.gather并行执行3个对话...")

    start_time = time.time()

    # 创建3个对话任务
    tasks = [
        run_single_chat(analyst, analyst, "请分析当前市场趋势", max_round=3),
        run_single_chat(designer, designer, "请设计用户登录界面", max_round=3),
        run_single_chat(developer, developer, "请设计用户认证API架构", max_round=3),
    ]

    # 并行执行所有任务
    results = await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time

    # ---------------------------------------------
    # 汇总结果
    # ---------------------------------------------

    print(f"\n并发执行完成:")
    print(f"  - 总耗时: {elapsed_time:.2f}秒")
    print(f"  - 完成任务数: {len(results)}")

    for i, result in enumerate(results):
        agent_name = [analyst.name, designer.name, developer.name][i]
        print(f"  - {agent_name}: {len(result.chat_history)}条消息")

    return results


async def demo_parallel_groupchats():
    """
    演示并行执行多个GroupChat

    场景：同时启动多个群聊，每个群聊处理一个独立任务
    所有群聊并行执行，加快整体处理速度
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: 并行执行多个GroupChat")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建多个独立的GroupChat
    # ---------------------------------------------

    # 群聊A：数据处理组
    data_lead = ConversableAgent(
        name="数据组长",
        system_message="你是数据组长，协调数据处理工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    data_dev = ConversableAgent(
        name="数据工程师",
        system_message="你是数据工程师，负责数据清洗和分析。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    data_groupchat = GroupChat(
        agents=[data_lead, data_dev],
        messages=[],
        max_round=4,
    )
    data_manager = GroupChatManager(
        groupchat=data_groupchat,
        llm_config=llm_config,
    )

    # 群聊B：UI设计组
    ui_lead = ConversableAgent(
        name="UI组长",
        system_message="你是UI组长，协调界面设计工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    ui_dev = ConversableAgent(
        name="UI设计师",
        system_message="你是UI设计师，负责界面原型设计。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    ui_groupchat = GroupChat(
        agents=[ui_lead, ui_dev],
        messages=[],
        max_round=4,
    )
    ui_manager = GroupChatManager(
        groupchat=ui_groupchat,
        llm_config=llm_config,
    )

    # 群聊C：架构设计组
    arch_lead = ConversableAgent(
        name="架构组长",
        system_message="你是架构组长，协调系统架构设计。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    arch_dev = ConversableAgent(
        name="架构师",
        system_message="你是系统架构师，负责技术架构设计。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    arch_groupchat = GroupChat(
        agents=[arch_lead, arch_dev],
        messages=[],
        max_round=4,
    )
    arch_manager = GroupChatManager(
        groupchat=arch_groupchat,
        llm_config=llm_config,
    )

    print("\n已创建3个独立GroupChat:")
    print("  - 数据处理组: 数据组长 + 数据工程师")
    print("  - UI设计组: UI组长 + UI设计师")
    print("  - 架构设计组: 架构组长 + 架构师")

    # ---------------------------------------------
    # 并行执行多个群聊
    # ---------------------------------------------

    print("\n并行执行3个GroupChat...")

    start_time = time.time()

    # 创建并行任务
    tasks = [
        run_single_chat(data_lead, data_manager, "请分析用户行为数据", max_round=4),
        run_single_chat(ui_lead, ui_manager, "请设计新功能的界面", max_round=4),
        run_single_chat(arch_lead, arch_manager, "请设计系统架构方案", max_round=4),
    ]

    # 并行执行
    results = await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time

    # ---------------------------------------------
    # 汇总结果
    # ---------------------------------------------

    print(f"\n并行GroupChat完成:")
    print(f"  - 总耗时: {elapsed_time:.2f}秒")
    print(f"  - 数据组: {len(results[0].chat_history)}条消息")
    print(f"  - UI组: {len(results[1].chat_history)}条消息")
    print(f"  - 架构组: {len(results[2].chat_history)}条消息")


# ============================================================
# 第四部分：超时控制与异常处理
# ============================================================

async def run_chat_with_timeout(
    agent: 'ConversableAgent',
    recipient: 'ConversableAgent',
    message: str,
    timeout: float,
    max_round: int = 5
) -> Dict[str, Any]:
    """
    运行带超时控制的对话

    Args:
        agent: 发起对话的Agent
        recipient: 接收消息的Agent
        message: 初始消息
        timeout: 超时时间（秒）
        max_round: 最大轮次

    Returns:
        dict: 包含结果或错误信息的字典
    """
    try:
        # 使用asyncio.wait_for设置超时
        result = await asyncio.wait_for(
            agent.a_initiate_chat(
                recipient,
                message=message,
                max_round=max_round,
            ),
            timeout=timeout
        )
        return {
            "success": True,
            "result": result,
            "error": None,
        }
    except asyncio.TimeoutError:
        # 超时处理
        return {
            "success": False,
            "result": None,
            "error": f"Timeout after {timeout} seconds",
        }
    except Exception as e:
        # 其他异常处理
        return {
            "success": False,
            "result": None,
            "error": str(e),
        }


def demo_timeout_control():
    """
    演示超时控制机制

    超时控制的用途：
    1. 防止任务无限执行
    2. 控制资源使用
    3. 提供更好的用户体验

    实现方式：
    - asyncio.wait_for：设置协程的超时时间
    - TimeoutError：超时后抛出的异常
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示3: 超时控制机制")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    slow_agent = ConversableAgent(
        name="慢速Agent",
        system_message="你是慢速Agent，处理每个请求需要较长时间。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    fast_agent = ConversableAgent(
        name="快速Agent",
        system_message="你是快速Agent，迅速给出简洁回答。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建2个Agent:")
    print("  - 慢速Agent: 处理速度较慢")
    print("  - 快速Agent: 处理速度较快")

    # ---------------------------------------------
    # 测试快速Agent（不会超时）
    # ---------------------------------------------

    print("\n测试1: 快速Agent (超时时间: 60秒)")
    print("-" * 40)

    async def test_fast():
        return await run_chat_with_timeout(
            fast_agent, fast_agent,
            "请简洁地介绍你自己。",
            timeout=60,
            max_round=2
        )

    result_fast = asyncio.run(test_fast())

    if result_fast["success"]:
        print("  状态: 成功")
        print(f"  消息数: {len(result_fast['result'].chat_history)}")
    else:
        print(f"  状态: 失败 - {result_fast['error']}")

    # ---------------------------------------------
    # 测试慢速Agent（可能超时）
    # ---------------------------------------------

    print("\n测试2: 慢速Agent (超时时间: 30秒)")
    print("-" * 40)

    async def test_slow():
        return await run_chat_with_timeout(
            slow_agent, slow_agent,
            "请详细描述一个复杂的技术架构，包括所有细节。",
            timeout=30,
            max_round=5
        )

    result_slow = asyncio.run(test_slow())

    if result_slow["success"]:
        print("  状态: 成功")
        print(f"  消息数: {len(result_slow['result'].chat_history)}")
    else:
        print(f"  状态: 失败 - {result_slow['error']}")


async def demo_timeout_exception_handling():
    """
    演示超时异常处理

    超时异常处理流程：
    1. 设置任务超时时间
    2. 执行协程
    3. 如果超时，抛出TimeoutError
    4. 捕获异常并进行相应处理
    5. 记录日志或返回错误信息

    注意事项：
    - 超时后任务不会自动停止，只是抛出异常
    - 需要在代码中妥善处理超时情况
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示4: 超时异常处理")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    worker = ConversableAgent(
        name="Worker",
        system_message="你是Worker，处理任务并报告进度。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n异常处理流程:")
    print("  1. 设置超时时间")
    print("  2. 执行异步任务")
    print("  3. 任务超时时捕获TimeoutError")
    print("  4. 返回错误信息")
    print("  5. 继续执行其他任务")

    # ---------------------------------------------
    # 模拟超时场景
    # ---------------------------------------------

    print("\n模拟超时场景...")

    async def long_running_task():
        """模拟一个长时间运行的任务"""
        try:
            result = await asyncio.wait_for(
                worker.a_initiate_chat(
                    worker,
                    message="执行一个耗时的分析任务，包含大量数据处理。",
                    max_round=3,
                ),
                timeout=0.1  # 设置很短的超时时间，确保超时
            )
            return {"success": True, "result": result}
        except asyncio.TimeoutError:
            return {"success": False, "error": "任务执行超时"}

    result = await long_running_task()

    print("\n结果:")
    if result["success"]:
        print(f"  - 状态: 成功")
        print(f"  - 消息数: {len(result['result'].chat_history)}")
    else:
        print(f"  - 状态: 失败")
        print(f"  - 错误: {result['error']}")


# ============================================================
# 第五部分：综合并发场景
# ============================================================

async def demo_concurrent_results_aggregation():
    """
    演示并发结果汇总

    并发执行多个任务后，需要汇总各任务的结果：
    1. 收集所有任务的结果
    2. 分析结果的质量和完整性
    3. 汇总成最终报告
    4. 提供统一的输出格式
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: 并发结果汇总")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建各领域的专家Agent
    # ---------------------------------------------

    # 技术专家
    tech_expert = ConversableAgent(
        name="技术专家",
        system_message="你是技术专家，提供技术方案和建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 市场专家
    market_expert = ConversableAgent(
        name="市场专家",
        system_message="你是市场专家，提供市场分析和预测。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 财务专家
    finance_expert = ConversableAgent(
        name="财务专家",
        system_message="你是财务专家，提供财务分析和建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个专家Agent: 技术专家、市场专家、财务专家")

    # ---------------------------------------------
    # 并行执行咨询任务
    # ---------------------------------------------

    print("\n并行咨询3个专家...")

    tasks = [
        run_single_chat(tech_expert, tech_expert,
                       "请评估新技术的可行性", max_round=3),
        run_single_chat(market_expert, market_expert,
                       "请分析市场趋势", max_round=3),
        run_single_chat(finance_expert, finance_expert,
                       "请评估项目成本", max_round=3),
    ]

    results = await asyncio.gather(*tasks)

    # ---------------------------------------------
    # 汇总结果
    # ---------------------------------------------

    print("\n" + "=" * 40)
    print("汇总专家意见:")
    print("=" * 40)

    for i, result in enumerate(results):
        expert_name = [tech_expert.name, market_expert.name, finance_expert.name][i]
        last_msg = result.chat_history[-1] if result.chat_history else {}
        content = last_msg.get("content", "无内容")[:100]

        print(f"\n[{expert_name}]")
        print(f"  消息数: {len(result.chat_history)}")
        print(f"  最终建议: {content}...")


async def demo_fault_tolerant_concurrent():
    """
    演示容错性并发处理

    在并发执行中，部分任务可能失败，需要容错处理：
    1. 单个任务失败不影响其他任务
    2. 收集失败和成功的任务
    3. 提供部分结果或重试机制
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示6: 容错性并发处理")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建多个Agent
    # ---------------------------------------------

    agents = [
        ConversableAgent(name=f"Agent{i}", system_message=f"你是Agent{i}。",
                       llm_config=llm_config, human_input_mode="NEVER")
        for i in range(1, 5)
    ]

    print("\n已创建4个Agent")

    # ---------------------------------------------
    # 带异常处理的任务执行
    # ---------------------------------------------

    async def safe_chat(agent, message, timeout=60):
        """安全执行对话，捕获所有异常"""
        try:
            result = await asyncio.wait_for(
                agent.a_initiate_chat(agent, message=message, max_round=2),
                timeout=timeout
            )
            return {"success": True, "agent": agent.name, "result": result}
        except asyncio.TimeoutError:
            return {"success": False, "agent": agent.name, "error": "超时"}
        except Exception as e:
            return {"success": False, "agent": agent.name, "error": str(e)}

    print("\n并发执行（部分可能失败）...")

    # 模拟部分失败：某些任务使用较短的超时
    tasks = [
        safe_chat(agents[0], "正常任务1", timeout=60),
        safe_chat(agents[1], "正常任务2", timeout=60),
        safe_chat(agents[2], "超时任务", timeout=0.01),  # 会超时
        safe_chat(agents[3], "正常任务3", timeout=60),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ---------------------------------------------
    # 分析结果
    # ---------------------------------------------

    print("\n" + "=" * 40)
    print("执行结果分析:")
    print("=" * 40)

    success_count = 0
    fail_count = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"  异常: {result}")
            fail_count += 1
        elif result["success"]:
            print(f"  [{result['agent']}] 成功")
            success_count += 1
        else:
            print(f"  [{result['agent']}] 失败: {result['error']}")
            fail_count += 1

    print(f"\n统计: 成功={success_count}, 失败={fail_count}")


# ============================================================
# 主函数
# ============================================================

async def async_main():
    """
    异步主函数：运行所有并发协作演示
    """
    print("=" * 60)
    print("并发协作配置演示 - 异步执行")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    await demo_asyncio_gather_basic()
    await demo_parallel_groupchats()
    await demo_timeout_exception_handling()
    await demo_concurrent_results_aggregation()
    await demo_fault_tolerant_concurrent()

    print("\n" + "=" * 60)
    print("异步演示完成")
    print("=" * 60)


def main():
    """
    主函数：运行所有演示
    """
    print("=" * 60)
    print("嵌套对话与并发协作机制 - 并发协作配置演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行同步演示
    demo_timeout_control()

    # 运行异步演示
    asyncio.run(async_main())

    print("\n" + "=" * 60)
    print("所有演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
