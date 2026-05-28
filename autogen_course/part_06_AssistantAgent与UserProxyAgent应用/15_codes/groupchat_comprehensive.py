# groupchat_comprehensive.py
# 第15节 ConversableAgent对话模式综合实战 - GroupChat综合实战
#
# 本文件演示GroupChat的综合实战用法，包括：
# 1. GroupChatManager的创建与配置
# 2. speaker_selection_method四种策略详解
# 3. allow_repeat参数对发言分布的影响
# 4. 终止条件的多种配置方式
# 5. 嵌套GroupChat与复杂协作
# 6. 实际团队协作场景
#
# ============================================================
# GroupChat 核心概念回顾
# ============================================================
#
# GroupChat 是AutoGen中实现多Agent团队协作的核心组件。
# 与双人对话不同，GroupChat通过GroupChatManager协调多个Agent。
#
# 核心组件：
# 1. GroupChat - 群聊容器，存储消息和Agent列表
# 2. GroupChatManager - 群聊管理器，负责协调整个群聊
# 3. Agent - 参与群聊的智能体
#
# 消息传递机制：
# - 广播模式：消息发送给所有Agent（通过GroupChatManager转发）
# - selected_agent：决定下一个发言者
#
# ============================================================
# 五种对话模式综合对比
# ============================================================
#
# 1. 双人对话模式（Two-Agent Chat）
#    - 两个Agent直接通信
#    - 简单直接，无需Manager
#    - 适合一对一协作
#
# 2. 群聊模式（GroupChat）
#    - 多个Agent通过Manager协作
#    - LLM自动选择下一个发言者
#    - 适合团队协作
#
# 3. 嵌套对话模式（Nested Chat）
#    - Agent之间可以嵌套调用
#    - 支持复杂的任务分解
#    - 适合层次化工作流
#
# 4. 异步对话模式（Async Chat）
#    - 使用async/await异步通信
#    - 支持并发执行
#    - 适合高性能场景
#
# 5. 流式对话模式（Streaming Chat）
#    - 支持流式输出
#    - 实时显示生成内容
#    - 适合长文本生成
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
# 第三部分：speaker_selection_method 四种策略
# ============================================================

def demo_speaker_selection_methods():
    """
    演示 speaker_selection_method 的四种策略

    speaker_selection_method 决定如何选择下一个发言者：
    1. "auto" - 由LLM根据上下文智能选择（默认）
    2. "manual" - 由外部代码/用户手动指定
    3. "round_robin" - 按顺序轮询选择
    4. "random" - 随机选择

    每种策略有不同适用场景：
    - auto: 复杂协作，需要LLM判断最佳发言者
    - manual: 教学/调试，需要精确控制发言顺序
    - round_robin: 均衡发言，确保每个Agent都有机会
    - random: 随机协作，增加多样性
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: speaker_selection_method 四种策略")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建三个简单的Agent用于演示
    agent1 = ConversableAgent(
        name="Agent_1",
        system_message="你是Agent_1，团队中的发起者。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent2 = ConversableAgent(
        name="Agent_2",
        system_message="你是Agent_2，团队中的分析师。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent3 = ConversableAgent(
        name="Agent_3",
        system_message="你是Agent_3，团队中的总结者。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: Agent_1, Agent_2, Agent_3")

    # ---------------------------------------------
    # 策略1: auto 模式（LLM智能选择）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略1: speaker_selection_method = 'auto'")
    print("-" * 40)
    print("特点:")
    print("  - 由LLM根据对话上下文选择下一个发言者")
    print("  - LLM会考虑Agent的角色和能力")
    print("  - 适合复杂的多Agent协作场景")
    print("  - 默认选项")

    groupchat_auto = GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=6,
        speaker_selection_method="auto",  # 默认选项
    )

    manager_auto = GroupChatManager(
        groupchat=groupchat_auto,
        llm_config=llm_config,
    )

    print(f"\n配置: speaker_selection_method='{groupchat_auto.speaker_selection_method}'")
    print("执行群聊...")

    result_auto = agent1.initiate_chat(
        manager_auto,
        message="我们讨论一下如何提高代码质量。请Agent_2先分析问题，然后Agent_3总结建议。",
    )

    print(f"\n结果:")
    print(f"  - 消息数: {len(result_auto.chat_history)}")
    print("  - LLM根据上下文选择最合适的发言者")

    # ---------------------------------------------
    # 策略2: round_robin 模式（轮询）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略2: speaker_selection_method = 'round_robin'")
    print("-" * 40)
    print("特点:")
    print("  - 按Agent列表顺序轮流选择发言者")
    print("  - 确保每个Agent都有平等的发言机会")
    print("  - 不考虑对话上下文")
    print("  - 适合需要均衡发言的场景")

    groupchat_rr = GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",  # 轮询模式
    )

    manager_rr = GroupChatManager(
        groupchat=groupchat_rr,
        llm_config=llm_config,
    )

    print(f"\n配置: speaker_selection_method='{groupchat_rr.speaker_selection_method}'")
    print("执行群聊...")

    result_rr = agent1.initiate_chat(
        manager_rr,
        message="请各位成员依次发表对项目架构的看法。",
    )

    print(f"\n结果:")
    print(f"  - 消息数: {len(result_rr.chat_history)}")

    # 统计发言顺序
    speakers_rr = []
    for msg in result_rr.chat_history:
        if msg.get("role") == "assistant":
            speakers_rr.append(msg.get("name", "unknown"))

    print(f"  - 发言顺序: {' -> '.join(speakers_rr)}")
    print("  - 预期: 轮流发言（忽略内容上下文）")

    # ---------------------------------------------
    # 策略3: random 模式（随机）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略3: speaker_selection_method = 'random'")
    print("-" * 40)
    print("特点:")
    print("  - 随机选择下一个发言者")
    print("  - 不考虑Agent顺序或上下文")
    print("  - 可能导致某些Agent主导或被忽视")
    print("  - 适合需要随机性的场景")

    groupchat_random = GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=6,
        speaker_selection_method="random",  # 随机模式
    )

    manager_random = GroupChatManager(
        groupchat=groupchat_random,
        llm_config=llm_config,
    )

    print(f"\n配置: speaker_selection_method='{groupchat_random.speaker_selection_method}'")
    print("执行群聊...")

    result_random = agent1.initiate_chat(
        manager_random,
        message="请各位成员讨论一下人工智能的发展趋势。",
    )

    print(f"\n结果:")
    print(f"  - 消息数: {len(result_random.chat_history)}")

    speakers_random = []
    for msg in result_random.chat_history:
        if msg.get("role") == "assistant":
            speakers_random.append(msg.get("name", "unknown"))

    print(f"  - 发言顺序: {' -> '.join(speakers_random)}")
    print("  - 注意: 发言顺序完全随机")

    # ---------------------------------------------
    # 策略4: manual 模式（手动）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略4: speaker_selection_method = 'manual'")
    print("-" * 40)
    print("特点:")
    print("  - 由外部代码或用户手动选择发言者")
    print("  - 提供最大的控制和确定性")
    print("  - 需要外部逻辑干预")
    print("  - 适合教学、调试或严格流程控制")

    groupchat_manual = GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=6,
        speaker_selection_method="manual",  # 手动模式
    )

    print(f"\n配置: speaker_selection_method='{groupchat_manual.speaker_selection_method}'")
    print("\n说明: manual模式需要外部干预选择下一个发言者")
    print("实际使用中，可以通过GroupChat.select_speaker()方法手动选择")
    print("由于manual模式需要外部输入，这里演示配置而非实际运行")

    # ---------------------------------------------
    # 总结对比
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("speaker_selection_method 对比总结")
    print("=" * 60)
    print("\n模式          | 选择方式        | 均衡性  | 适用场景")
    print("-" * 60)
    print("auto          | LLM智能选择     | 中      | 复杂协作")
    print("round_robin   | 按顺序轮询      | 高      | 均衡发言")
    print("random        | 随机选择        | 低      | 随机性场景")
    print("manual        | 外部手动指定    | 完全可控 | 教学/调试")


# ============================================================
# 第四部分：allow_repeat 参数控制
# ============================================================

def demo_allow_repeat_control():
    """
    演示 allow_repeat 参数对发言分布的影响

    allow_repeat 参数控制同一Agent是否能连续发言：
    1. "never" - 不允许连续发言，确保均衡
    2. "certain_num_turns" - 允许连续发言一定次数
    3. "always" - 允许连续发言（默认）

    这个参数与speaker_selection_method配合使用，
    可以实现不同的发言模式。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: allow_repeat 参数控制")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建三个Agent
    speaker_a = ConversableAgent(
        name="Speaker_A",
        system_message="你是Speaker_A，请简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    speaker_b = ConversableAgent(
        name="Speaker_B",
        system_message="你是Speaker_B，请简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    speaker_c = ConversableAgent(
        name="Speaker_C",
        system_message="你是Speaker_C，请简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: Speaker_A, Speaker_B, Speaker_C")

    # ---------------------------------------------
    # 策略1: allow_repeat = "never"
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略1: allow_repeat = 'never'")
    print("-" * 40)
    print("特点:")
    print("  - 同一Agent不能连续发言")
    print("  - 确保每个Agent都有发言机会")
    print("  - 避免某个Agent主导对话")
    print("  - 适合需要均衡发言的场景")

    groupchat_never = GroupChat(
        agents=[speaker_a, speaker_b, speaker_c],
        messages=[],
        max_round=6,
        allow_repeat="never",  # 不允许连续发言
    )

    manager_never = GroupChatManager(
        groupchat=groupchat_never,
        llm_config=llm_config,
    )

    print(f"\n配置: allow_repeat='{groupchat_never.allow_repeat}'")

    result_never = speaker_a.initiate_chat(
        manager_never,
        message="讨论主题：如何提高团队效率。请每位成员发表看法。",
    )

    # 分析发言模式
    speakers_never = []
    for msg in result_never.chat_history:
        if msg.get("role") == "assistant":
            speakers_never.append(msg.get("name", "unknown"))

    print(f"\n发言顺序: {' -> '.join(speakers_never)}")

    # 检查是否有连续重复
    has_consecutive = False
    for i in range(len(speakers_never) - 1):
        if speakers_never[i] == speakers_never[i + 1]:
            has_consecutive = True
            print(f"  警告: 发现连续重复 - {speakers_never[i]}")

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
    print("  - 适合需要某个Agent主导讨论的场景")

    groupchat_certain = GroupChat(
        agents=[speaker_a, speaker_b, speaker_c],
        messages=[],
        max_round=6,
        allow_repeat="certain_num_turns",
        max_consecutive_agent_num=2,  # 允许连续发言2次
    )

    print(f"\n配置: allow_repeat='{groupchat_certain.allow_repeat}'")
    print(f"         max_consecutive_agent_num={groupchat_certain.max_consecutive_agent_num}")

    # ---------------------------------------------
    # 策略3: allow_repeat = "always" (默认)
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("策略3: allow_repeat = 'always' (默认)")
    print("-" * 40)
    print("特点:")
    print("  - 允许同一Agent连续发言")
    print("  - LLM可以自由选择最合适的Agent")
    print("  - 可能导致发言不均衡")
    print("  - 适合复杂协作场景")

    groupchat_always = GroupChat(
        agents=[speaker_a, speaker_b, speaker_c],
        messages=[],
        max_round=6,
        allow_repeat="always",  # 默认行为
    )

    print(f"\n配置: allow_repeat='{groupchat_always.allow_repeat}'")

    result_always = speaker_a.initiate_chat(
        manager_always,
        message="讨论主题：如何提高团队效率。请每位成员发表看法。",
    )

    # 统计发言分布
    speakers_always = []
    for msg in result_always.chat_history:
        if msg.get("role") == "assistant":
            speakers_always.append(msg.get("name", "unknown"))

    print(f"\n发言顺序: {' -> '.join(speakers_always)}")

    from collections import Counter
    counts_always = Counter(speakers_always)
    print("\n各Agent发言次数:")
    for name, count in counts_always.items():
        print(f"  - {name}: {count}次")

    print("\n注意: 某些Agent可能发言较多或较少")


# ============================================================
# 第五部分：终止条件配置
# ============================================================

def demo_termination_conditions():
    """
    演示GroupChat的多种终止条件配置

    GroupChat支持多种终止条件：
    1. max_round - 达到最大轮次后强制终止
    2. is_termination_msg - Agent返回特定消息时终止
    3. speaker_count - 特定speaker被选择次数达到阈值
    4. 自定义终止条件 - 组合多个条件

    这些条件可以组合使用，任一满足即终止。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: GroupChat 终止条件配置")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建会发送终止消息的Agent
    agent_finisher = ConversableAgent(
        name="终结者",
        system_message="""你是终结者。
如果任务完成或讨论足够，说'TASK_COMPLETE'来终止对话。
否则继续讨论。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TASK_COMPLETE" in msg.get("content", ""),
    )

    agent_helper = ConversableAgent(
        name="助手",
        system_message="你是助手，帮助完成讨论。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建两个Agent: 终结者, 助手")

    # ---------------------------------------------
    # 终止条件1: max_round
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("终止条件1: max_round")
    print("-" * 40)
    print("特点:")
    print("  - 设置最大对话轮次")
    print("  - 达到轮次后强制终止")
    print("  - 作为安全网防止无限循环")
    print("  - 最简单的终止方式")

    groupchat_max = GroupChat(
        agents=[agent_finisher, agent_helper],
        messages=[],
        max_round=5,  # 关键参数：5轮后终止
    )

    manager_max = GroupChatManager(
        groupchat=groupchat_max,
        llm_config=llm_config,
    )

    print(f"\n配置: max_round={groupchat_max.max_round}")

    result_max = agent_finisher.initiate_chat(
        manager_max,
        message="我们来讨论一下云计算的发展趋势。",
    )

    print(f"\n结果:")
    print(f"  - 设置的max_round: {groupchat_max.max_round}")
    print(f"  - 实际消息数: {len(result_max.chat_history)}")
    print(f"  - 终止原因: {'达到max_round' if len(result_max.chat_history) >= 5 * 2 else '自然结束'}")

    # ---------------------------------------------
    # 终止条件2: is_termination_msg
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("终止条件2: is_termination_msg")
    print("-" * 40)
    print("特点:")
    print("  - 检查Agent回复是否包含终止标记")
    print("  - 由Agent自己判断任务是否完成")
    print("  - 可以实现智能终止")
    print("  - 需要Agent配合发送特定消息")

    groupchat_term = GroupChat(
        agents=[agent_finisher, agent_helper],
        messages=[],
        max_round=10,  # 较大的max_round作为保险
    )

    manager_term = GroupChatManager(
        groupchat=groupchat_term,
        llm_config=llm_config,
    )

    print(f"\n配置: max_round={groupchat_term.max_round}")
    print("终止条件: Agent回复包含'TASK_COMPLETE'")

    result_term = agent_finisher.initiate_chat(
        manager_term,
        message="请简要说明数据分析的基本流程。",
    )

    print(f"\n结果:")
    print(f"  - 实际消息数: {len(result_term.chat_history)}")

    if any("TASK_COMPLETE" in msg.get("content", "") for msg in result_term.chat_history):
        print("  - 终止原因: Agent发送'TASK_COMPLETE'提前终止")
    else:
        print("  - 终止原因: 自然结束")

    # ---------------------------------------------
    # 终止条件3: 组合条件
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("终止条件3: 组合条件")
    print("-" * 40)
    print("特点:")
    print("  - max_round + is_termination_msg 同时生效")
    print("  - 任一条件满足即终止")
    print("  - 灵活性最高的配置方式")
    print("  - 推荐使用这种方式")

    groupchat_combo = GroupChat(
        agents=[agent_finisher, agent_helper],
        messages=[],
        max_round=8,  # 最大8轮作为保险
        # is_termination_msg 由Agent的 is_termination_msg 属性控制
    )

    manager_combo = GroupChatManager(
        groupchat=groupchat_combo,
        llm_config=llm_config,
    )

    print(f"\n配置: max_round={groupchat_combo.max_round}")
    print("        + Agent的is_termination_msg检测")

    result_combo = agent_finisher.initiate_chat(
        manager_combo,
        message="分析一下大数据技术的主要应用场景。",
    )

    print(f"\n结果:")
    print(f"  - 实际消息数: {len(result_combo.chat_history)}")

    term_by_msg = any("TASK_COMPLETE" in msg.get("content", "") for msg in result_combo.chat_history)
    term_by_round = len(result_combo.chat_history) >= 8 * 2

    if term_by_msg and term_by_round:
        print("  - 终止原因: 同时满足两个条件")
    elif term_by_msg:
        print("  - 终止原因: Agent发送'TASK_COMPLETE'")
    elif term_by_round:
        print("  - 终止原因: 达到max_round")
    else:
        print("  - 终止原因: 自然结束")


# ============================================================
# 第六部分：嵌套GroupChat
# ============================================================

def demo_nested_groupchat():
    """
    演示嵌套GroupChat的用法

    嵌套GroupChat是指在一个GroupChat中调用另一个GroupChat，
    或者让Agent参与多个群聊。这适合层次化任务分解。

    使用场景：
    - 主群聊负责总体协调
    - 子群聊负责具体任务执行
    - 任务完成后汇报结果给主群聊
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: 嵌套GroupChat")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建主群聊的Agent
    # ---------------------------------------------

    coordinator = ConversableAgent(
        name="协调者",
        system_message="""你是协调者，负责协调多个小组的工作。
你会将任务分配给不同的小组，并收集汇报。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建子群聊1：后端开发组
    # ---------------------------------------------

    backend_dev = ConversableAgent(
        name="后端开发",
        system_message="你是后端开发专家，负责服务器端开发。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    backend_reviewer = ConversableAgent(
        name="后端审查",
        system_message="你是后端审查专家，负责审查代码质量。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    backend_groupchat = GroupChat(
        agents=[backend_dev, backend_reviewer],
        messages=[],
        max_round=4,
    )

    backend_manager = GroupChatManager(
        groupchat=backend_groupchat,
        llm_config=llm_config,
    )

    print("\n已创建后端开发子群聊:")
    print("  - 成员: 后端开发, 后端审查")
    print("  - 最大轮次: 4")

    # ---------------------------------------------
    # 创建子群聊2：前端开发组
    # ---------------------------------------------

    frontend_dev = ConversableAgent(
        name="前端开发",
        system_message="你是前端开发专家，负责用户界面开发。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    frontend_reviewer = ConversableAgent(
        name="前端审查",
        system_message="你是前端审查专家，负责审查界面代码。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    frontend_groupchat = GroupChat(
        agents=[frontend_dev, frontend_reviewer],
        messages=[],
        max_round=4,
    )

    frontend_manager = GroupChatManager(
        groupchat=frontend_groupchat,
        llm_config=llm_config,
    )

    print("\n已创建前端开发子群聊:")
    print("  - 成员: 前端开发, 前端审查")
    print("  - 最大轮次: 4")

    # ---------------------------------------------
    # 主群聊：项目协调
    # ---------------------------------------------

    project_manager = ConversableAgent(
        name="项目经理",
        system_message="你是项目经理，负责协调前后端开发。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建主群聊协调者: 项目经理")

    # ---------------------------------------------
    # 演示嵌套调用
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("嵌套GroupChat工作流程")
    print("-" * 40)
    print("  1. 协调者发起主群聊讨论")
    print("  2. 任务分配给后端组和前端组")
    print("  3. 各子组独立讨论并完成子任务")
    print("  4. 子组汇报结果给主群聊")
    print("  5. 主群聊汇总并给出最终方案")

    print("\n注意: 嵌套GroupChat在实际使用中需要精心设计消息传递协议")
    print("本演示展示配置方式，实际运行时需要考虑:")
    print("  - 如何传递任务给子群聊")
    print("  - 如何收集子群聊的结果")
    print("  - 如何处理并发执行")


# ============================================================
# 第七部分：实际团队协作场景
# ============================================================

def demo_practical_team_collaboration():
    """
    实际场景演示：完整的软件团队协作

    团队组成：
    - 架构师：负责系统设计和架构决策
    - 程序员：负责代码实现
    - 测试工程师：负责测试和验证
    - 项目经理：负责协调和进度管理

    工作流程：
    1. 项目经理发起任务
    2. 架构师设计架构
    3. 程序员实现代码
    4. 测试工程师验证
    5. 循环迭代直到完成
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: 实际场景 - 软件团队协作")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建团队成员
    # ---------------------------------------------

    architect = ConversableAgent(
        name="架构师",
        system_message="""你是资深系统架构师。
职责：
- 设计系统架构和模块划分
- 评估技术方案可行性
- 给出架构层面的指导建议

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

    tester = ConversableAgent(
        name="测试工程师",
        system_message="""你是资深测试工程师。
职责：
- 编写和执行测试用例
- 发现和报告问题
- 验证修复是否有效

在讨论中，你会根据上下文选择合适的时机发言。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    manager = ConversableAgent(
        name="项目经理",
        system_message="""你是项目经理，负责协调团队工作。
职责：
- 发起和组织讨论
- 协调各方意见
- 推进任务进展
- 确认任务完成

你会积极推动讨论有序进行。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n软件团队:")
    print("  - 架构师: 系统设计")
    print("  - 程序员: 代码实现")
    print("  - 测试工程师: 测试验证")
    print("  - 项目经理: 协调管理")

    # ---------------------------------------------
    # 创建GroupChat
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[architect, coder, tester, manager],
        messages=[],
        max_round=10,
        speaker_selection_method="auto",  # LLM智能选择
        allow_repeat="never",  # 不允许连续发言
    )

    manager_groupchat = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print("\nGroupChat配置:")
    print(f"  - speaker_selection_method: auto")
    print(f"  - allow_repeat: never")
    print(f"  - max_round: 10")

    # ---------------------------------------------
    # 发起团队讨论
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("发起用户认证系统开发讨论")
    print("-" * 40)

    result = manager.initiate_chat(
        manager_groupchat,
        message="""我们需要开发一个用户认证系统。
请按以下流程进行：
1. 架构师先设计系统架构
2. 程序员根据架构实现代码
3. 测试工程师制定测试计划
4. 最后确认方案

完成后说'TASK_DONE'来结束。""",
    )

    print(f"\n讨论完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")

    # ---------------------------------------------
    # 分析讨论结果
    # ---------------------------------------------

    print("\n=== 讨论分析 ===")

    # 统计发言
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name", "unknown"))

    from collections import Counter
    counts = Counter(speakers)

    print("\n各成员发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")

    # 检查发言均衡性
    print("\n发言均衡性检查:")
    has_consecutive = False
    for i in range(len(speakers) - 1):
        if speakers[i] == speakers[i + 1]:
            has_consecutive = True
            print(f"  - 警告: {speakers[i]}连续发言")

    if not has_consecutive:
        print("  - 通过: 没有连续重复发言")

    # 检查是否有人主导
    max_count = max(counts.values())
    min_count = min(counts.values())
    if max_count > min_count * 2:
        print(f"  - 警告: 发言不均衡，最多的{max_count}次，最少的{min_count}次")
    else:
        print("  - 通过: 发言分布相对均衡")


# ============================================================
# 第八部分：协作模式选择决策
# ============================================================

def demo_collaboration_mode_selection():
    """
    协作模式选择决策指南

    根据不同场景选择合适的协作模式：
    1. 双人对话：简单的一对一协作
    2. GroupChat：多Agent团队协作
    3. 嵌套对话：层次化任务分解
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示6: 协作模式选择决策指南")
    print("=" * 60)

    llm_config = build_llm_config()

    print("\n=== 选择决策流程 ===")
    print("\n1. Agent数量是否只有2个？")
    print("   -> 是: 使用双人对话模式")
    print("   -> 否: 继续下一步")

    print("\n2. 是否需要发言均衡控制？")
    print("   -> 是: 使用GroupChat + allow_repeat='never'")
    print("   -> 否: 继续下一步")

    print("\n3. 是否需要LLM自动选择发言者？")
    print("   -> 是: 使用GroupChat + speaker_selection_method='auto'")
    print("   -> 否: 考虑manual模式或其他")

    print("\n4. 是否有层次化的任务分解？")
    print("   -> 是: 使用嵌套GroupChat")
    print("   -> 否: 使用普通GroupChat")

    print("\n=== 模式对比总结 ===")

    modes = [
        ("双人对话", "2", "直接通信", "低", "一对一协作、人机交互"),
        ("GroupChat", "2+", "广播通信", "中", "多团队协作、团队讨论"),
        ("嵌套GroupChat", "2+", "层次化通信", "高", "复杂任务分解、子团队协作"),
        ("异步GroupChat", "2+", "并发通信", "高", "高性能场景、并行任务"),
    ]

    print("\n模式            | Agent数 | 通信方式    | 复杂度 | 适用场景")
    print("-" * 70)
    for mode, agents, comm, complexity, scenarios in modes:
        print(f"{mode:<15} | {agents:<6} | {comm:<11} | {complexity:<6} | {scenarios}")

    print("\n=== 实际选择示例 ===")

    scenarios = [
        ("客服问答", "双人对话", "一对一用户服务，简单直接"),
        ("代码审查团队", "GroupChat", "多角色协作，需要发言均衡"),
        ("项目管理委员会", "GroupChat", "多部门参与，需要LLM协调"),
        ("微服务开发", "嵌套GroupChat", "主团队+子团队，层次化协作"),
        ("并行数据处理", "异步GroupChat", "多个数据源并发处理"),
    ]

    print("\n场景              | 推荐模式        | 原因")
    print("-" * 65)
    for scenario, mode, reason in scenarios:
        print(f"{scenario:<16} | {mode:<15} | {reason}")


# ============================================================
# 第九部分：综合演示
# ============================================================

def demo_comprehensive_groupchat():
    """
    综合演示：GroupChat的完整配置

    展示如何配置一个完整的GroupChat系统，
    包括团队创建、配置选择、对话执行和结果分析。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示7: GroupChat 完整配置")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建专业的AI助手团队
    # ---------------------------------------------

    researcher = ConversableAgent(
        name="研究员",
        system_message="""你是AI研究专家。
职责：
- 研究最新的AI技术和趋势
- 分析论文和技术报告
- 提供技术见解和建议

请始终保持对最新技术发展的关注。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    developer = ConversableAgent(
        name="开发工程师",
        system_message="""你是软件开发工程师。
职责：
- 将研究成果转化为实际应用
- 编写高质量的代码
- 解决技术实现问题

你专注于将想法变成可用的产品。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    consultant = ConversableAgent(
        name="技术顾问",
        system_message="""你是技术顾问。
职责：
- 评估技术方案的可行性
- 提供专业的建议
- 帮助团队做出决策

你有多年的技术咨询经验。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\nAI助手团队:")
    print("  - 研究员: AI研究和趋势分析")
    print("  - 开发工程师: 技术实现")
    print("  - 技术顾问: 方案评估和建议")

    # ---------------------------------------------
    # 配置GroupChat
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[researcher, developer, consultant],
        messages=[],
        max_round=8,
        speaker_selection_method="auto",
        allow_repeat="never",
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print("\nGroupChat配置:")
    print(f"  - speaker_selection_method: auto")
    print(f"  - allow_repeat: never")
    print(f"  - max_round: 8")

    # ---------------------------------------------
    # 发起团队讨论
    # ---------------------------------------------

    print("\n执行团队讨论:")
    print("-" * 40)

    result = researcher.initiate_chat(
        manager,
        message="""我需要评估将大语言模型部署到边缘设备的技术可行性。
请按照以下流程讨论：
1. 研究员分析当前大语言模型的发展现状
2. 开发工程师评估部署到边缘设备的技术挑战
3. 技术顾问综合给出建议
4. 最后给出综合评估报告

完成后说'TASK_COMPLETE'来结束。""",
    )

    print("-" * 40)
    print(f"\n讨论完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")

    # ---------------------------------------------
    # 分析讨论结果
    # ---------------------------------------------

    print("\n=== 讨论结果分析 ===")

    # 统计发言
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name", "unknown"))

    from collections import Counter
    counts = Counter(speakers)

    print("\n各成员发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")

    print("\n发言顺序:")
    print(f"  {' -> '.join(speakers)}")

    # 检查终止原因
    if any("TASK_COMPLETE" in msg.get("content", "") for msg in result.chat_history):
        print("\n终止原因: Agent发送'TASK_COMPLETE'正常终止")
    elif len(result.chat_history) >= 8 * 2:
        print("\n终止原因: 达到max_round限制")
    else:
        print("\n终止原因: 自然结束")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：运行所有GroupChat综合实战演示
    """
    print("=" * 60)
    print("ConversableAgent对话模式综合实战 - GroupChat综合实战")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_speaker_selection_methods()
    demo_allow_repeat_control()
    demo_termination_conditions()
    demo_nested_groupchat()
    demo_practical_team_collaboration()
    demo_collaboration_mode_selection()
    demo_comprehensive_groupchat()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()