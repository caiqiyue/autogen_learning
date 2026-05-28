# speaker_selection_basic.py
# 第10节 speaker_selection_mode 三种策略详解 - 基本用法演示
#
# 本文件演示 GroupChat 中 speaker_selection_mode 的三种策略：
# 1. auto（LLM推荐）- 由大模型根据上下文智能选择下一个发言者
# 2. manual（人类指定）- 由人工指定下一个发言者
# 3. allow_repeat（允许重复）- 控制同一Agent是否能连续发言
#
# ============================================================
# speaker_selection_mode 核心概念
# ============================================================
#
# speaker_selection_mode 控制群聊中如何选择下一个发言者。
# 这与 speaker_selection_method（选择方法）是两个不同的概念：
#
# - speaker_selection_method: "auto", "manual", "round_robin", "random"
#   决定选择speaker的机制
#
# - speaker_selection_mode: 控制是否允许重复发言
#   包含三种策略: "auto", "manual", "allow_repeat"
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
# 第三部分：策略详解
# ============================================================
#
# speaker_selection_mode 有三种值：
#
# 1. "auto" (默认)
#    - LLM根据对话上下文和各个Agent的角色定义，智能选择最合适的下一个发言者
#    - 需要LLM能够理解Agent的角色和当前对话状态
#    - 适合复杂的多Agent协作场景
#
# 2. "manual"
#    - 由外部代码或用户指定下一个发言者
#    - 通过 speaker_selection_method 配合实现
#    - 适合需要精确控制对话流程的场景
#
# 3. "allow_repeat" (这是一个枚举值，控制重复发言行为)
#    - allow_repeat 参数有三个选项：
#      * "never" - 不允许同一Agent连续发言（避免重复）
#      * "certain_num_turns" - 允许连续发言一定次数
#      * "always" - 允许连续发言（默认行为）
#
# ============================================================


def demo_auto_mode():
    """
    演示 auto 模式（LLM智能选择）

    auto模式是默认的speaker_selection_mode，具有以下特点：
    - 由LLM根据对话上下文选择下一个发言者
    - LLM会考虑每个Agent的角色定义和当前对话状态
    - 适合复杂的多Agent协作场景
    - 灵活性高，但可能选择不够可预测

    auto模式的工作原理：
    1. GroupChatManager 将当前对话上下文发送给LLM
    2. LLM分析各个Agent的角色和能力
    3. LLM选择最合适的Agent作为下一个发言者
    4. 被选中的Agent接收消息并生成回复
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: auto 模式（LLM智能选择）")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建具有不同角色的Agent
    # ---------------------------------------------

    # 程序员Agent - 负责编写代码
    coder_agent = ConversableAgent(
        name="程序员",
        system_message="""你是一位经验丰富的Python程序员。
你的职责是：
- 编写高质量的Python代码
- 解释代码的实现逻辑
- 提供代码优化建议

当被选中发言时，请专注于代码相关的讨论。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 审查员Agent - 负责审查代码
    reviewer_agent = ConversableAgent(
        name="代码审查员",
        system_message="""你是一位资深的代码审查员。
你的职责是：
- 审查代码的质量和安全性
- 发现潜在的问题和bug
- 提出改进建议

当被选中发言时，请专注于代码审查和提出改进建议。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 架构师Agent - 负责系统设计
    architect_agent = ConversableAgent(
        name="架构师",
        system_message="""你是一位系统架构师。
你的职责是：
- 设计和规划系统架构
- 评估技术方案可行性
- 协调团队决策

当被选中发言时，请专注于系统设计和架构决策。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个具有不同角色的Agent:")
    print("  - 程序员: 负责编写代码")
    print("  - 代码审查员: 负责审查代码")
    print("  - 架构师: 负责系统设计")

    # ---------------------------------------------
    # 创建 GroupChat，默认为 auto 模式
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[coder_agent, reviewer_agent, architect_agent],
        messages=[],
        max_round=6,  # 限制最大轮次
        speaker_selection_method="auto",  # 使用auto方法选择speaker
        # speaker_selection_mode 默认为 "auto"
    )

    print(f"\nGroupChat 配置:")
    print(f"  - speaker_selection_method: {groupchat.speaker_selection_method}")
    print(f"  - speaker_selection_mode: {groupchat.speaker_selection_mode}")
    print("  - 模式说明: 由LLM根据上下文和Agent角色智能选择下一个发言者")

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 启动群聊对话
    # ---------------------------------------------

    print("\n启动群聊对话...")
    print("-" * 40)

    result = architect_agent.initiate_chat(
        manager,
        message="我我们需要设计一个用户认证系统。请程序员先给出初步的代码实现方案，然后审查员评估，最后架构师给出架构建议。",
    )

    print("-" * 40)
    print(f"auto模式对话结束，总消息数: {len(result.chat_history)}")

    # ---------------------------------------------
    # 分析speaker选择结果
    # ---------------------------------------------

    print("\n=== auto模式分析 ===")
    print("LLM会根据以下因素选择下一个发言者:")
    print("  1. 当前对话状态和上下文")
    print("  2. 每个Agent的角色定义和能力")
    print("  3. 任务的当前阶段和需求")
    print("  4. 避免重复选择同一Agent（除非必要）")

    # 统计每个Agent被选中的次数
    agent_selections = {}
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            name = msg.get("name", "unknown")
            agent_selections[name] = agent_selections.get(name, 0) + 1

    print("\n各Agent被选中次数:")
    for name, count in agent_selections.items():
        print(f"  - {name}: {count}次")

    return result


def demo_manual_mode():
    """
    演示 manual 模式（人类指定）

    manual模式允许外部代码或用户控制下一个发言者。
    这在以下场景特别有用：
    - 需要精确控制对话流程
    - 用户希望手动选择下一个发言者
    - 与外部系统集成需要确定性

    manual模式的工作原理：
    1. GroupChatManager 暂停，等待外部指定speaker
    2. 外部代码调用 select_speaker 方法指定下一个发言者
    3. 被指定的Agent接收消息并生成回复
    4. 重复直到对话结束

    注意：manual模式需要用户或代码主动干预，不适合完全自动化的场景
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: manual 模式（人类指定）")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    agent_a = ConversableAgent(
        name="Agent_A",
        system_message="你是Agent_A，负责处理任务A。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="Agent_B",
        system_message="你是Agent_B，负责处理任务B。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_c = ConversableAgent(
        name="Agent_C",
        system_message="你是Agent_C，负责处理任务C。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: Agent_A, Agent_B, Agent_C")

    # ---------------------------------------------
    # 创建 GroupChat，使用 manual 方法
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        messages=[],
        max_round=9,  # 足够展示手动选择效果
        speaker_selection_method="manual",  # 关键：使用manual方法
    )

    print(f"\nGroupChat 配置:")
    print(f"  - speaker_selection_method: {groupchat.speaker_selection_method}")
    print("  - 模式说明: 由外部代码/用户手动指定下一个发言者")

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 演示如何手动指定speaker
    # ---------------------------------------------

    print("\n=== manual模式使用方式 ===")
    print("在manual模式下，speaker选择由外部控制：")
    print("  1. 可以通过 groupchat.select_speaker() 方法手动选择")
    print("  2. 可以通过 termination_condition 控制终止")
    print("  3. 适合需要确定性对话流程的场景")

    # 注意：manual模式需要外部干预，这里演示配置方式
    # 实际使用时需要配合外部逻辑来选择speaker

    print("\nmanual模式适用场景:")
    print("  - 教学演示：教师手动选择学生回答")
    print("  - 流程控制：严格按步骤执行的任务")
    print("  - 调试模式：调试群聊行为时使用")
    print("  - 人工审核：重要决策需要人工确认")

    # 演示一个简化的自动执行示例（使用auto做演示）
    # 因为manual模式在实际运行中需要外部干预
    print("\n由于manual模式需要外部干预，这里改用auto模式演示实际运行效果:")

    # 创建用于实际演示的groupchat
    demo_groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        messages=[],
        max_round=6,
        speaker_selection_method="auto",  # 实际使用auto
    )

    demo_manager = GroupChatManager(
        groupchat=demo_groupchat,
        llm_config=llm_config,
    )

    result = agent_a.initiate_chat(
        demo_manager,
        message="请依次由Agent_A、Agent_B、Agent_C分别发表意见。",
    )

    print(f"\n实际运行结果: 总消息数 {len(result.chat_history)}")


def demo_allow_repeat_mode():
    """
    演示 allow_repeat 参数的三种策略

    allow_repeat 参数控制同一Agent是否能连续发言：
    - "never" - 不允许同一Agent连续发言，避免重复
    - "certain_num_turns" - 允许连续发言一定次数
    - "always" - 允许连续发言（默认行为）

    这个参数对于控制群聊的发言分布很重要：
    - "never" 确保每个Agent都有机会发言
    - "certain_num_turns" 允许一定的连续发言但有限制
    - "always" 让LLM自由选择，可能导致某些Agent主导对话

    与max_round的交互：
    - max_round 控制总对话轮数
    - allow_repeat 控制发言模式
    - 两者配合可以达到不同的对话效果
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: allow_repeat 参数策略详解")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建简单的Agent用于演示
    # ---------------------------------------------

    agent_x = ConversableAgent(
        name="Agent_X",
        system_message="你是Agent_X，请简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_y = ConversableAgent(
        name="Agent_Y",
        system_message="你是Agent_Y，请简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_z = ConversableAgent(
        name="Agent_Z",
        system_message="你是Agent_Z，请简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: Agent_X, Agent_Y, Agent_Z")

    # ---------------------------------------------
    # 策略1: allow_repeat = "never"
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略1: allow_repeat = 'never'")
    print("-" * 40)
    print("特点:")
    print("  - 同一Agent不能连续发言")
    print("  - 确保每个Agent都有发言机会")
    print("  - 避免某个Agent主导整个对话")
    print("  - 适合需要均衡发言的场景")

    groupchat_never = GroupChat(
        agents=[agent_x, agent_y, agent_z],
        messages=[],
        max_round=6,
        allow_repeat="never",  # 关键设置
    )

    print(f"\n配置: allow_repeat = '{groupchat_never.allow_repeat}'")
    print("预期效果: 每个Agent轮流发言，不会连续两次选中同一个Agent")

    manager_never = GroupChatManager(
        groupchat=groupchat_never,
        llm_config=llm_config,
    )

    result_never = agent_x.initiate_chat(
        manager_never,
        message="讨论主题：人工智能的未来。请每位成员发表看法。",
    )

    # 分析发言模式
    print("\n发言模式分析:")
    speakers = []
    for msg in result_never.chat_history:
        if msg.get("role") == "assistant":
            name = msg.get("name", "unknown")
            speakers.append(name)

    print(f"  发言顺序: {' -> '.join(speakers)}")

    # 检查是否有连续重复
    has_consecutive = False
    for i in range(len(speakers) - 1):
        if speakers[i] == speakers[i + 1]:
            has_consecutive = True
            print(f"  发现连续重复: {speakers[i]} 连续发言")

    if not has_consecutive:
        print("  验证通过: 没有连续重复发言")

    # ---------------------------------------------
    # 策略2: allow_repeat = "certain_num_turns"
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略2: allow_repeat = 'certain_num_turns'")
    print("-" * 40)
    print("特点:")
    print("  - 允许同一Agent连续发言一定次数")
    print("  - 更灵活的控制发言模式")
    print("  - 可以通过 max_consecutive_agent 设置次数限制")

    # 注意：certain_num_turns 需要配合其他参数使用
    # 这里是配置说明，实际效果取决于具体实现

    groupchat_certain = GroupChat(
        agents=[agent_x, agent_y, agent_z],
        messages=[],
        max_round=6,
        allow_repeat="certain_num_turns",  # 关键设置
    )

    print(f"\n配置: allow_repeat = '{groupchat_certain.allow_repeat}'")
    print("说明: 需要配合max_consecutive_agent参数限制连续发言次数")
    print("使用场景:")
    print("  - 需要某个Agent主导讨论时")
    print("  - 允许某些Agent连续发言表达深入观点")
    print("  - 平衡控制与灵活性")

    # ---------------------------------------------
    # 策略3: allow_repeat = "always" (默认)
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略3: allow_repeat = 'always' (默认行为)")
    print("-" * 40)
    print("特点:")
    print("  - 允许同一Agent连续发言")
    print("  - LLM可以自由选择最合适的Agent")
    print("  - 适合复杂协作场景")
    print("  - 可能导致发言不均衡")

    groupchat_always = GroupChat(
        agents=[agent_x, agent_y, agent_z],
        messages=[],
        max_round=6,
        allow_repeat="always",  # 默认行为
    )

    print(f"\n配置: allow_repeat = '{groupchat_always.allow_repeat}'")
    print("预期效果: LLM根据上下文自由选择发言者")

    manager_always = GroupChatManager(
        groupchat=groupchat_always,
        llm_config=llm_config,
    )

    result_always = agent_x.initiate_chat(
        manager_always,
        message="讨论主题：人工智能的未来。请每位成员发表看法。",
    )

    # 分析发言模式
    print("\n发言模式分析:")
    speakers = []
    for msg in result_always.chat_history:
        if msg.get("role") == "assistant":
            name = msg.get("name", "unknown")
            speakers.append(name)

    print(f"  发言顺序: {' -> '.join(speakers)}")

    # 统计每个Agent发言次数
    from collections import Counter
    counts = Counter(speakers)
    print("\n各Agent发言次数:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")


def demo_max_round_interaction():
    """
    演示 max_round 与 is_termination_msg 的交互

    max_round 和 is_termination_msg 是两个独立的终止条件：
    - max_round: 达到指定轮数后强制终止
    - is_termination_msg: 当Agent返回True时终止

    两者可以同时使用，任一满足即终止：
    - 对话可能在 max_round 之前结束（如果某Agent返回 termination message）
    - 对话一定会到达 max_round（除非被更早终止）

    max_round 控制的是"轮"数，每轮包括：
    1. 选择speaker
    2. speaker生成回复
    3. 检查终止条件
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: max_round 与 is_termination_msg 的交互")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建会在特定条件下终止的Agent
    # ---------------------------------------------

    # Agent1: 当收到特定关键词时终止
    agent_early_terminate = ConversableAgent(
        name="Agent_早终止",
        system_message="""你 是 Agent_早终止。
当用户消息包含"终止讨论"时，返回"TALKING_END"来结束对话。
否则正常回复。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TALKING_END" in msg.get("content", ""),
    )

    # Agent2: 普通Agent
    agent_normal = ConversableAgent(
        name="Agent_正常",
        system_message="你是Agent_正常，负责正常讨论。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建两个Agent:")
    print("  - Agent_早终止: 收到'终止讨论'时终止")
    print("  - Agent_正常: 正常参与讨论")

    # ---------------------------------------------
    # 场景1: 只使用 max_round
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("场景1: 只使用 max_round 终止")
    print("-" * 40)

    groupchat1 = GroupChat(
        agents=[agent_early_terminate, agent_normal],
        messages=[],
        max_round=10,  # 足够多的轮次
        # 不设置 is_termination_msg，使用默认值
    )

    manager1 = GroupChatManager(
        groupchat=groupchat1,
        llm_config=llm_config,
    )

    print(f"  max_round: {groupchat1.max_round}")
    print("  预期: 对话持续到10轮后终止")

    result1 = agent_early_terminate.initiate_chat(
        manager1,
        message="我们来讨论一下项目计划。",
    )

    print(f"  实际消息数: {len(result1.chat_history)}")
    print(f"  终止原因: 达到 max_round={groupchat1.max_round}")

    # ---------------------------------------------
    # 场景2: 使用 is_termination_msg 提前终止
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("场景2: 使用 is_termination_msg 提前终止")
    print("-" * 40)

    groupchat2 = GroupChat(
        agents=[agent_early_terminate, agent_normal],
        messages=[],
        max_round=10,  # 较大的max_round作为保险
        # is_termination_msg 由Agent的 human_input_mode 控制
    )

    manager2 = GroupChatManager(
        groupchat=groupchat2,
        llm_config=llm_config,
    )

    print(f"  max_round: {groupchat2.max_round}")
    print("  终止条件: Agent返回包含'TALKING_END'的消息")

    # 直接调用，不通过initiate_chat的message触发终止
    # 这里演示配置，实际终止由消息内容触发

    print("\n  说明:")
    print("    - is_termination_msg 在每次Agent生成回复后检查")
    print("    - 如果返回True，对话立即终止，不等待max_round")
    print("    - 两个条件是'或'的关系（任一满足即终止）")

    # ---------------------------------------------
    # 场景3: 两者配合的完整示例
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("场景3: max_round + is_termination_msg 配合使用")
    print("-" * 40)

    # 创建会主动终止的Agent
    terminator = ConversableAgent(
        name="终止者",
        system_message="""你是终止者。
当你认为讨论已经足够时，说"TASK_DONE"来结束对话。
否则继续正常讨论。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TASK_DONE" in msg.get("content", ""),
    )

    helper = ConversableAgent(
        name="助手",
        system_message="你是助手，帮助完成讨论。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    groupchat3 = GroupChat(
        agents=[terminator, helper],
        messages=[],
        max_round=10,  # 最大10轮
    )

    manager3 = GroupChatManager(
        groupchat=groupchat3,
        llm_config=llm_config,
    )

    print(f"  max_round: {groupchat3.max_round}")
    print("  终止条件: 'TASK_DONE'出现 或 达到10轮")

    result3 = terminator.initiate_chat(
        manager3,
        message="请分析一下云计算的发展趋势。",
    )

    print(f"  实际消息数: {len(result3.chat_history)}")
    if len(result3.chat_history) < 10:
        print("  终止原因: Agent主动说'TASK_DONE'提前终止")
    else:
        print("  终止原因: 达到 max_round=10")

    # ---------------------------------------------
    # 总结：终止条件的工作原理
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("终止条件总结")
    print("=" * 60)
    print("\nGroupChat 终止条件检查流程:")
    print("  1. Agent生成回复")
    print("  2. 检查 is_termination_msg（Agent级别检查）")
    print("     - 如果返回True，立即终止对话")
    print("  3. 检查 max_round（GroupChat级别检查）")
    print("     - 如果当前轮次 >= max_round，终止对话")
    print("  4. 两个条件是'或'的关系")
    print("\n配置建议:")
    print("  - 设置合理的 max_round 作为安全网")
    print("  - 使用 is_termination_msg 实现智能终止")
    print("  - 确保终止消息格式在Agent系统提示中说明")


def demo_termination_msg_config():
    """
    演示 termination_msg_config 的详细配置

    termination_msg_config 是 GroupChat 的终止消息配置，
    用于控制如何检测和触发对话终止。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: termination_msg_config 详解")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建会发送终止消息的Agent
    agent_with_term = ConversableAgent(
        name="终止Agent",
        system_message="""你是终止Agent。
你会评估对话是否完成。如果完成，说"FINISH"来终止对话。
否则继续正常回复。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "FINISH" in msg.get("content", ""),
    )

    agent_normal = ConversableAgent(
        name="普通Agent",
        system_message="你是普通Agent，参与讨论。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建两个Agent:")
    print("  - 终止Agent: 会评估是否需要终止")
    print("  - 普通Agent: 正常参与讨论")

    # ---------------------------------------------
    # 默认终止配置
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("默认终止配置")
    print("-" * 40)

    groupchat_default = GroupChat(
        agents=[agent_with_term, agent_normal],
        messages=[],
        max_round=10,
    )

    print("  GroupChat 默认终止配置:")
    print("    - 使用 Agent 的 is_termination_msg 方法")
    print("    - 检查消息内容是否包含终止标记")
    print("    - max_round 作为最终保险")

    # ---------------------------------------------
    # 自定义终止消息检测
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("自定义终止消息检测")
    print("-" * 40)

    def custom_termination_detection(message):
        """
        自定义终止消息检测函数

        Args:
            message: 消息字典

        Returns:
            bool: 是否应该终止
        """
        content = message.get("content", "").lower()
        # 检测多个终止关键词
        termination_keywords = ["finish", "done", "complete", "end"]
        return any(kw in content for kw in termination_keywords)

    groupchat_custom = GroupChat(
        agents=[agent_with_term, agent_normal],
        messages=[],
        max_round=10,
        # 可以传入自定义的终止检测函数
    )

    print("  自定义终止检测:")
    print("    - 可以自定义检测函数")
    print("    - 检测多个关键词: finish, done, complete, end")
    print("    - 更灵活的终止条件控制")

    # ---------------------------------------------
    # 终止消息的优先级
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("终止消息的优先级")
    print("-" * 40)
    print("  1. is_termination_msg (Agent级别) - 最高优先级")
    print("     - 每个Agent可以定义自己的终止条件")
    print("     - 检测到终止条件时立即停止")
    print("  2. max_round (GroupChat级别) - 次高优先级")
    print("     - 达到最大轮次后强制终止")
    print("     - 作为安全网防止无限循环")
    print("  3. 外部终止信号 - 最低优先级")
    print("     - 可以通过API发送外部终止信号")


def demo_practical_scenario():
    """
    演示一个实际的综合场景

    场景：软件代码审查团队
    - 架构师发起任务
    - 程序员编写代码
    - 审查员审查代码
    - 使用 auto 模式和 allow_repeat="never" 确保均衡发言
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示6: 实际场景 - 软件代码审查团队")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建专业的代码审查团队
    # ---------------------------------------------

    architect = ConversableAgent(
        name="架构师",
        system_message="""你是资深系统架构师。
职责：
- 设计系统架构和模块划分
- 评估技术方案的可行性
- 协调团队决策

在讨论中，你会根据上下文选择合适的时机发言。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    coder = ConversableAgent(
        name="程序员",
        system_message="""你是经验丰富的Python程序员。
职责：
- 编写高质量的代码
- 解释代码实现逻辑
- 根据反馈优化代码

在讨论中，你会根据上下文选择合适的时机发言。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    reviewer = ConversableAgent(
        name="审查员",
        system_message="""你是资深代码审查员。
职责：
- 审查代码质量和安全性
- 发现潜在问题和bug
- 提出具体的改进建议

在讨论中，你会根据上下文选择合适的时机发言。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n代码审查团队:")
    print("  - 架构师: 负责系统设计")
    print("  - 程序员: 负责代码实现")
    print("  - 审查员: 负责质量审查")

    # ---------------------------------------------
    # 配置 GroupChat
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[architect, coder, reviewer],
        messages=[],
        max_round=9,
        speaker_selection_method="auto",  # LLM智能选择
        allow_repeat="never",  # 不允许连续发言，确保均衡
    )

    print(f"\nGroupChat 配置:")
    print(f"  - speaker_selection_method: auto (LLM智能选择)")
    print(f"  - allow_repeat: never (不允许连续发言)")
    print(f"  - max_round: 9 (最大9轮)")

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 发起代码审查讨论
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("发起代码审查讨论")
    print("-" * 40)

    result = architect.initiate_chat(
        manager,
        message="""我需要审查一个新的用户认证模块。
请程序员先实现基本代码，然后审查员提出改进意见，最后架构师总结架构考虑。
完成后说'TASK_DONE'来结束讨论。""",
    )

    print(f"\n讨论完成，总消息数: {len(result.chat_history)}")

    # ---------------------------------------------
    # 分析讨论结果
    # ---------------------------------------------

    print("\n=== 讨论分析 ===")

    # 统计发言
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            name = msg.get("name", "unknown")
            speakers.append(name)

    from collections import Counter
    counts = Counter(speakers)

    print("\n各成员发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")

    # 检查是否均衡（allow_repeat="never"的效果）
    has_consecutive = False
    for i in range(len(speakers) - 1):
        if speakers[i] == speakers[i + 1]:
            has_consecutive = True
            break

    print(f"\nallow_repeat='never'验证:")
    if has_consecutive:
        print("  警告: 发现连续重复发言")
    else:
        print("  通过: 没有连续重复发言，发言分布均衡")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有 speaker_selection_mode 演示
    """
    print("=" * 60)
    print("speaker_selection_mode 三种策略详解 - 基本用法演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_auto_mode()
    demo_manual_mode()
    demo_allow_repeat_mode()
    demo_max_round_interaction()
    demo_termination_msg_config()
    demo_practical_scenario()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()