# groupchat_basic.py
# 第9节 GroupChat与多Agent协作模式 - 基本用法演示
#
# 本文件演示 GroupChat 的基本用法，包括：
# 1. GroupChatManager 的创建与配置
# 2. 多个Agent加入群聊
# 3. 消息广播机制
# 4. Agent角色分配与selected_agent选择逻辑
#
# ============================================================
# GroupChat 核心概念
# ============================================================
#
# GroupChat 是 AutoGen 中实现多Agent协作的核心组件。
# 它通过 GroupChatManager 管理一群Agent之间的对话。
#
# 关键组件：
# 1. GroupChatManager - 群聊管理器，负责协调整个群聊
# 2. GroupChat - 群聊容器，存储消息和Agent列表
# 3. Agent - 参与群聊的智能体
#
# 消息传递机制：
# - 广播模式：消息发送给所有Agent（通过GroupChatManager转发）
# - 私聊模式：消息只发送给特定Agent（GroupChat支持但需要特殊配置）
#
# selected_agent 选择逻辑：
# - 由 speaker_selection_method 指定，默认为 "auto"
# - 可选值："auto", "manual", "round_robin", "random"
#
# ============================================================

import os
from pathlib import Path

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
# 第三部分：GroupChat 基本用法演示
# ============================================================

def demo_basic_groupchat():
    """
    演示最基本的 GroupChat 用法

    场景：创建一个包含3个Agent的群聊，让它们讨论一个问题
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: 基本的 GroupChat 用法")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 步骤1：创建多个 Agent
    # ---------------------------------------------

    # 创建三个助手Agent，每个有不同的角色和系统提示
    coder_agent = ConversableAgent(
        name="程序员",  # 负责编写代码
        system_message="你是一位经验丰富的Python程序员，擅长编写高质量的代码。",
        llm_config=llm_config,
        human_input_mode="NEVER",  # 不需要人工输入
    )

    reviewer_agent = ConversableAgent(
        name="代码审查员",  # 负责审查代码
        system_message="你是一位资深的代码审查员，擅长发现代码中的问题和改进建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    planner_agent = ConversableAgent(
        name="架构师",  # 负责系统设计
        system_message="你是一位系统架构师，擅长设计和规划复杂系统的结构。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent:")
    print(f"  - {coder_agent.name}")
    print(f"  - {reviewer_agent.name}")
    print(f"  - {planner_agent.name}")

    # ---------------------------------------------
    # 步骤2：创建 GroupChat（群聊容器）
    # ---------------------------------------------

    # GroupChat 是群聊消息和Agent列表的容器
    groupchat = GroupChat(
        agents=[coder_agent, reviewer_agent, planner_agent],  # 参与群聊的Agent列表
        messages=[],  # 初始消息列表为空
        max_round=10,  # 最大对话轮次，防止无限循环
    )

    print(f"\n已创建 GroupChat:")
    print(f"  - 参与Agent数: {len(groupchat.agents)}")
    print(f"  - 最大轮次: {groupchat.max_round}")
    print(f"  - speaker_selection_method: {groupchat.speaker_selection_method}")

    # ---------------------------------------------
    # 步骤3：创建 GroupChatManager（群聊管理器）
    # ---------------------------------------------

    # GroupChatManager 负责协调群聊，管理消息传递和Agent选择
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,  # Manager也需要LLM配置来选择下一个speaker
    )

    print(f"\n已创建 GroupChatManager:")
    print(f"  - 管理的GroupChat: {manager.groupchat is not None}")
    print(f"  - 是否为Agent: {isinstance(manager, ConversableAgent)}")

    # ---------------------------------------------
    # 步骤4：启动群聊对话
    # ---------------------------------------------

    # 通过initiate_chat方法启动群聊，指定一个初始消息
    print("\n启动群聊对话...")
    print("-" * 40)

    # 使用架构师发起讨论，提出一个架构问题
    chat_result = planner_agent.initiate_chat(
        manager,
        message="我我们需要设计一个可扩展的Web应用架构。请程序员先给出初步方案，然后请审查员提出改进建议。",
    )

    print("-" * 40)
    print("群聊对话结束")
    print(f"总消息数: {len(chat_result.chat_history)}")

    return chat_result


def demo_speaker_selection_methods():
    """
    演示不同的 speaker_selection_method 对群聊行为的影响

    AutoGen 支持多种选择下一个发言者的策略：
    - "auto": 由LLM自动决定（默认）
    - "manual": 由外部指定下一个speaker
    - "round_robin": 轮询制，每个Agent按顺序轮流发言
    - "random": 随机选择下一个speaker
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: speaker_selection_method 策略对比")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建两个简单的Agent
    agent_a = ConversableAgent(
        name="Agent_A",
        system_message="你叫做Agent_A，是团队中的成员。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="Agent_B",
        system_message="你叫做Agent_B，是团队中的成员。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 测试 round_robin 模式
    # ---------------------------------------------

    print("\n--- round_robin 模式 ---")

    # round_robin: 按顺序轮流选择下一个speaker
    groupchat_rr = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=4,  # 4轮足够展示轮询效果
        speaker_selection_method="round_robin",  # 关键参数
    )

    manager_rr = GroupChatManager(
        groupchat=groupchat_rr,
        llm_config=llm_config,
    )

    print(f"speaker_selection_method: {groupchat_rr.speaker_selection_method}")
    print("预期行为: Agent_A -> Agent_B -> Agent_A -> Agent_B")

    # ---------------------------------------------
    # 测试 random 模式
    # ---------------------------------------------

    print("\n--- random 模式 ---")

    groupchat_random = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=4,
        speaker_selection_method="random",  # 关键参数：随机选择
    )

    manager_random = GroupChatManager(
        groupchat=groupchat_random,
        llm_config=llm_config,
    )

    print(f"speaker_selection_method: {groupchat_random.speaker_selection_method}")
    print("预期行为: 随机选择下一个speaker（不保证轮流）")

    # ---------------------------------------------
    # 测试 auto 模式（默认）
    # ---------------------------------------------

    print("\n--- auto 模式（默认） ---")

    groupchat_auto = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=4,
        speaker_selection_method="auto",  # LLM自动决定（默认）
    )

    manager_auto = GroupChatManager(
        groupchat=groupchat_auto,
        llm_config=llm_config,
    )

    print(f"speaker_selection_method: {groupchat_auto.speaker_selection_method}")
    print("预期行为: 由LLM根据上下文选择下一个speaker")


def demo_broadcast_vs_private():
    """
    演示消息广播与私聊的区别

    GroupChat 默认使用广播模式：
    - 消息通过GroupChatManager转发给所有Agent
    - 每个Agent收到的是完整的对话历史

    私聊模式（需要特殊配置）：
    - 使用 send_messages_to_agent 而非群聊广播
    - 消息只发送给特定Agent
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: 广播模式 vs 私聊模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建两个Agent
    alice = ConversableAgent(
        name="Alice",
        system_message="你是Alice，与Bob一起工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    bob = ConversableAgent(
        name="Bob",
        system_message="你是Bob，与Alice一起工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 广播模式（默认）
    # ---------------------------------------------

    print("\n--- 广播模式（GroupChat默认） ---")
    print("特点:")
    print("  1. GroupChatManager 将消息广播给所有Agent")
    print("  2. 每个Agent都能看到完整的对话历史")
    print("  3. 适合团队协作讨论场景")

    groupchat = GroupChat(
        agents=[alice, bob],
        messages=[],
        max_round=5,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print("\n执行广播:")
    result = alice.initiate_chat(
        manager,
        message="大家好，让我们讨论一下项目架构。",
    )
    print(f"  广播完成，消息数: {len(result.chat_history)}")

    # ---------------------------------------------
    # 私聊模式（使用 initiate_chat 指定接收者）
    # ---------------------------------------------

    print("\n--- 私聊模式 ---")
    print("特点:")
    print("  1. 使用 recipient 参数指定私聊对象")
    print("  2. 消息只发送给特定Agent")
    print("  3. 其他Agent不会收到这条消息")
    print("  4. 适合需要保密的讨论或一对一协作")

    # 直接私聊，不经过GroupChatManager
    print("\n执行私聊 (Alice -> Bob):")
    result = alice.initiate_chat(
        bob,  # 直接指定接收者为Bob，不经过manager
        message="Bob，我有个想法想私下和你讨论...",
    )
    print(f"  私聊完成，消息数: {len(result.chat_history)}")


def demo_message_flow():
    """
    演示 GroupChat 中的消息传递流程

    消息传递流程：
    1. 用户或Agent发送消息到 GroupChatManager
    2. GroupChatManager 调用 select_speaker() 选择下一个发言者
    3. 被选中的Agent接收消息并生成回复
    4. 回复被添加到消息历史
    5. 重复步骤2-4直到达到终止条件
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: GroupChat 消息传递流程详解")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建三个简单Agent
    agent1 = ConversableAgent(
        name="发起者",
        system_message="你是讨论的发起者，负责提出问题。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent2 = ConversableAgent(
        name="分析员",
        system_message="你是分析员，负责分析问题并提供见解。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent3 = ConversableAgent(
        name="总结员",
        system_message="你是总结员，负责总结讨论结果。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    groupchat = GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=6,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 展示消息流程
    # ---------------------------------------------

    print("\n消息传递流程:")
    print("  1. [发起者] 发送消息到 GroupChatManager")
    print("  2. [Manager] 调用 select_speaker() 选择下一个发言者")
    print("  3. [Manager] 将消息转发给选中的Agent")
    print("  4. [被选中Agent] 生成回复")
    print("  5. [Manager] 将回复添加到消息历史")
    print("  6. 检查终止条件，不满足则继续步骤2")
    print("  7. 达到终止条件后，返回聊天结果")

    print("\n执行群聊:")
    result = agent1.initiate_chat(
        manager,
        message="请分析一下云计算的发展趋势，最后请总结员做总结。",
    )

    print("\n消息历史记录:")
    for i, msg in enumerate(result.chat_history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:60]
        print(f"  [{i}] {role}: {content}...")


def demo_groupchat_manager_internal():
    """
    演示 GroupChatManager 的内部机制

    GroupChatManager 的核心职责：
    1. 维护 GroupChat 实例（群聊状态）
    2. select_speaker() - 选择下一个发言者
    3. reset() - 重置群聊状态
    4. 消息转发和路由
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: GroupChatManager 内部机制")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建Agent
    agent1 = ConversableAgent(
        name="Agent1",
        system_message="你是Agent1。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent2 = ConversableAgent(
        name="Agent2",
        system_message="你是Agent2。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 创建群聊
    groupchat = GroupChat(
        agents=[agent1, agent2],
        messages=[],
        max_round=3,
    )

    # 创建管理器
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 检查 Manager 的内部状态
    # ---------------------------------------------

    print("\nGroupChatManager 内部状态:")
    print(f"  - groupchat 属性: {manager.groupchat is not None}")
    print(f"  - agents 数量: {len(manager.groupchat.agents)}")
    print(f"  - 当前消息数: {len(manager.groupchat.messages)}")
    print(f"  - max_round: {manager.groupchat.max_round}")

    # ---------------------------------------------
    # 演示 reset 功能
    # ---------------------------------------------

    print("\n执行一轮对话后:")
    _ = agent1.initiate_chat(
        manager,
        message="你好，Agent2。",
    )
    print(f"  - 消息数: {len(manager.groupchat.messages)}")

    print("\n调用 reset() 重置群聊:")
    manager.groupchat.reset()
    print(f"  - 消息数: {len(manager.groupchat.messages)}")
    print("  - 群聊状态已重置，可用于下一轮讨论")


def demo_termination_conditions():
    """
    演示 GroupChat 的终止条件

    GroupChat 支持多种终止条件：
    1. max_round - 达到最大轮次
    2. speaker_count - 特定speaker被选择的次数
    3. 消息内容终止（通过 is_termination_msg 判断）

    重要：termination_msg_config 控制终止逻辑
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示6: GroupChat 终止条件")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建两个Agent
    agent1 = ConversableAgent(
        name="Agent1",
        system_message="你是Agent1。如果对话完成，说'TASK_COMPLETE'来结束。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent2 = ConversableAgent(
        name="Agent2",
        system_message="你是Agent2。如果任务完成，说'TASK_COMPLETE'来结束。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 配置终止条件：基于 max_round
    # ---------------------------------------------

    print("\n方式1: 基于 max_round")
    print("  - 设置 max_round=5，当对话达到5轮时自动终止")
    print("  - 这是最简单的终止方式")

    groupchat1 = GroupChat(
        agents=[agent1, agent2],
        messages=[],
        max_round=5,  # 5轮后自动终止
    )

    manager1 = GroupChatManager(
        groupchat=groupchat1,
        llm_config=llm_config,
    )

    result1 = agent1.initiate_chat(
        manager1,
        message="我们来讨论一下天气。",
    )
    print(f"  实际轮次: {len(result1.chat_history)}")

    # ---------------------------------------------
    # 配置终止条件：基于消息内容
    # ---------------------------------------------

    print("\n方式2: 基于消息内容终止")
    print("  - 设置 enabled=True 启用消息内容检查")
    print("  - 当消息包含特定关键词时终止")

    # 使用自定义的终止消息检测
    def custom_termination(msg):
        """自定义终止条件：检测消息是否包含 TASK_COMPLETE"""
        if hasattr(msg, "content"):
            return "TASK_COMPLETE" in msg.content
        return False

    groupchat2 = GroupChat(
        agents=[agent1, agent2],
        messages=[],
        max_round=10,
        termination_msg=None,  # 使用默认终止检测
    )

    manager2 = GroupChatManager(
        groupchat=groupchat2,
        llm_config=llm_config,
    )

    print("  - 当Agent回复包含'TASK_COMPLETE'时，对话终止")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有 GroupChat 基本用法演示
    """
    print("=" * 60)
    print("GroupChat 与多Agent协作模式 - 基本用法演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_basic_groupchat()
    demo_speaker_selection_methods()
    demo_broadcast_vs_private()
    demo_message_flow()
    demo_groupchat_manager_internal()
    demo_termination_conditions()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()