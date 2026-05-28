# termination_config.py
# 第11节 GroupChat终止条件与常见问题处理 - 终止条件配置演示
#
# 本文件演示 GroupChat 中各种终止条件的配置方法，包括：
# 1. max_round 基础配置
# 2. is_termination_msg 在 GroupChat 中的特殊行为
# 3. 终止条件组合策略
# 4. speaker_selection_method 与终止的交互
#
# ============================================================
# GroupChat 终止条件概述
# ============================================================
#
# GroupChat 有三种主要的终止机制：
# 1. max_round - 最大轮次限制，防止无限循环
# 2. 消息内容终止 - 通过 is_termination_msg 检测特定内容
# 3. speaker_count - 限制某个 speaker 被选择的次数
#
# 重要：在 GroupChat 场景下，终止条件的处理比单个 Agent 复杂，
# 因为需要协调多个 Agent 的对话流程。
#
# ============================================================

import os
import re
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
# 第三部分：终止条件配置演示
# ============================================================

def demo_max_round_termination():
    """
    演示 max_round 终止条件的基础配置

    max_round 是 GroupChat 最基本的终止条件：
    - 当对话轮次达到 max_round 时，强制终止
    - 轮次计算：每选择一次 speaker 算一轮
    - 适用于：已知任务复杂度，可以预估最大轮数的场景
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: max_round 基础配置")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建两个简单的 Agent
    agent_a = ConversableAgent(
        name="规划师",
        system_message="你是一位规划师，负责提出方案。简短回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="执行者",
        system_message="你是一位执行者，负责评估方案。简短回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 配置1：max_round=3（短对话）
    # ---------------------------------------------
    print("\n--- max_round=3（短对话） ---")

    groupchat_short = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=3,  # 只进行3轮对话
    )

    manager_short = GroupChatManager(
        groupchat=groupchat_short,
        llm_config=llm_config,
    )

    print(f"max_round: {groupchat_short.max_round}")
    print("预期：对话在3轮后强制终止")

    result = agent_a.initiate_chat(
        manager_short,
        message="我们需要设计一个用户登录系统。",
    )

    print(f"实际消息数: {len(result.chat_history)}")
    print(f"终止原因: 达到 max_round={groupchat_short.max_round}")

    # ---------------------------------------------
    # 配置2：max_round=10（中等对话）
    # ---------------------------------------------
    print("\n--- max_round=10（中等对话） ---")

    groupchat_medium = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=10,  # 进行10轮对话
    )

    manager_medium = GroupChatManager(
        groupchat=groupchat_medium,
        llm_config=llm_config,
    )

    print(f"max_round: {groupchat_medium.max_round}")
    print("适用场景：需要多轮讨论的复杂任务")

    # ---------------------------------------------
    # 配置3：max_round=50（长对话）
    # ---------------------------------------------
    print("\n--- max_round=50（长对话） ---")

    groupchat_long = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=50,  # 进行50轮对话
    )

    print(f"max_round: {groupchat_long.max_round}")
    print("适用场景：需要深入讨论的研究型任务")
    print("注意：过大的 max_round 可能导致对话时间过长")

    print("\nmax_round 选择指南:")
    print("  - 3-5: 简单问答、快速讨论")
    print("  - 10-15: 标准协作任务（如代码审查）")
    print("  - 20-30: 复杂分析、多角度讨论")
    print("  - 50+: 研究型任务（慎用）")


def demo_termination_msg_in_groupchat():
    """
    演示 is_termination_msg 在 GroupChat 场景下的特殊行为

    重要：GroupChat 中的 is_termination_msg 行为与单个 Agent 不同！
    - GroupChatManager 会自动为每个 Agent 设置 is_termination_msg 检查
    - 当任何 Agent 的回复满足终止条件时，整个群聊终止
    - 终止检查发生在每次消息后，而不是轮次结束后
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: is_termination_msg 在 GroupChat 中的特殊行为")
    print("=" * 60)

    llm_config = build_llm_config()

    # 定义终止条件：包含"完成"或"结束"时终止
    def groupchat_termination(msg):
        """
        GroupChat 终止条件函数

        Args:
            msg: 消息字典，包含 content 等字段

        Returns:
            bool: True 表示终止，False 表示继续
        """
        content = msg.get("content", "")
        termination_keywords = ["完成", "结束", "再见", "TERMINATE"]

        for keyword in termination_keywords:
            if keyword in content:
                return True
        return False

    # 创建两个 Agent
    planner = ConversableAgent(
        name="规划师",
        system_message="你是一位规划师，负责提出方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    executor = ConversableAgent(
        name="执行者",
        system_message="你是一位执行者，负责实施和反馈。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建带终止条件的 GroupChat
    # ---------------------------------------------
    print("\n配置 GroupChat 终止条件:")

    groupchat = GroupChat(
        agents=[planner, executor],
        messages=[],
        max_round=20,
        # 关键：在这里设置终止条件函数
        termination_msg=groupchat_termination,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print(f"  - termination_msg: 已设置")
    print(f"  - 终止关键词: {['完成', '结束', '再见', 'TERMINATE']}")
    print(f"  - max_round: {groupchat.max_round}")

    print("\n执行群聊（规划师发起）:")
    print("-" * 40)

    result = planner.initiate_chat(
        manager,
        message="我们来讨论一个项目规划。请先提出一个方案。",
    )

    print("-" * 40)
    print(f"对话结束:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 终止原因: 消息包含终止关键词 或 达到 max_round")

    # ---------------------------------------------
    # 验证终止条件的传播
    # ---------------------------------------------
    print("\n验证终止条件传播:")
    print("  - GroupChatManager 会将 termination_msg 传递给每个 Agent")
    print("  - 当任何 Agent 回复包含终止关键词时，群聊终止")
    print("  - 这是 GroupChat 与单独 Agent 对话的重要区别")


def demo_termination_msg_vs_max_round():
    """
    演示终止条件组合：is_termination_msg + max_round

    两种终止条件可以组合使用：
    - is_termination_msg: 基于内容（更智能，但依赖模型输出）
    - max_round: 基于轮次（更可靠，作为保底机制）

    推荐策略：同时设置两者，形成"智能+保底"的双重保护
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: 终止条件组合策略")
    print("=" * 60)

    llm_config = build_llm_config()

    # 定义终止条件
    def smart_termination(msg):
        """
        智能终止条件：检查消息是否表示任务完成

        终止条件：
        1. 包含"完成"、"结束"等完成标记
        2. 包含"再见"等退出标记
        3. 包含"谢谢"且消息较短（正常结束语）
        """
        content = msg.get("content", "")

        # 完成标记
        if any(kw in content for kw in ["完成", "结束", "搞定"]):
            return True

        # 退出标记
        if any(kw in content for kw in ["再见", "拜拜", "exit"]):
            return True

        # 简短结束语
        if "谢谢" in content and len(content) < 50:
            return True

        return False

    # 创建 Agent
    coordinator = ConversableAgent(
        name="协调者",
        system_message="你是团队协调者，负责推进讨论。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    specialist = ConversableAgent(
        name="专家",
        system_message="你是领域专家，提供专业意见。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 配置：双重保护策略
    # ---------------------------------------------
    print("\n配置双重保护策略:")
    print("  保护1: is_termination_msg（智能，基于内容）")
    print("  保护2: max_round=10（保底，基于轮次）")

    groupchat = GroupChat(
        agents=[coordinator, specialist],
        messages=[],
        max_round=10,  # 保底：最多10轮
        termination_msg=smart_termination,  # 智能终止
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print(f"\n终止条件配置:")
    print(f"  - max_round: {groupchat.max_round}（保底上限）")
    print(f"  - termination_msg: 已设置（智能检测）")
    print(f"  - 实际行为: 先到先止（任一条件满足即终止）")

    # 执行群聊
    result = coordinator.initiate_chat(
        manager,
        message="我们来讨论AI的发展趋势。",
    )

    print(f"\n执行结果:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 是否达到上限: {'是' if len(result.chat_history) >= 10 else '否'}")

    print("\n双重保护的优势:")
    print("  1. 内容触发：对话可能在任意轮次提前结束（更自然）")
    print("  2. 轮次保底：确保对话不会超过预期时长（更安全）")
    print("  3. 组合使用：平衡智能与可靠性")


def demo_speaker_selection_and_termination():
    """
    演示 speaker_selection_method 与终止条件的交互

    不同 speaker_selection_method 可能会影响终止行为：
    - auto: LLM选择下一个speaker，可能提前终止或继续
    - round_robin: 固定轮换，更容易预测终止时机
    - random: 随机选择，终止时机不确定
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: speaker_selection_method 与终止条件交互")
    print("=" * 60)

    llm_config = build_llm_config()

    agent1 = ConversableAgent(
        name="Agent_A",
        system_message="你是Agent A。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent2 = ConversableAgent(
        name="Agent_B",
        system_message="你是Agent B。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 场景1：round_robin + max_round（可预测终止）
    # ---------------------------------------------
    print("\n--- 场景1: round_robin + max_round ---")
    print("特点：可预测的轮换 + 可控的终止时机")

    groupchat_rr = GroupChat(
        agents=[agent1, agent2],
        messages=[],
        max_round=4,  # 4轮终止
        speaker_selection_method="round_robin",  # 轮询选择
    )

    manager_rr = GroupChatManager(
        groupchat=groupchat_rr,
        llm_config=llm_config,
    )

    print(f"  - speaker_selection_method: {groupchat_rr.speaker_selection_method}")
    print(f"  - max_round: {groupchat_rr.max_round}")
    print("  - 预期行为: Agent_A -> Agent_B -> Agent_A -> Agent_B（4轮后终止）")
    print("  - 优势：终止时机高度可预测，便于测试和调试")

    # ---------------------------------------------
    # 场景2：auto + max_round（LLM驱动）
    # ---------------------------------------------
    print("\n--- 场景2: auto + max_round ---")
    print("特点：智能speaker选择 + 轮次限制")

    groupchat_auto = GroupChat(
        agents=[agent1, agent2],
        messages=[],
        max_round=10,
        speaker_selection_method="auto",  # LLM自动选择
    )

    manager_auto = GroupChatManager(
        groupchat=groupchat_auto,
        llm_config=llm_config,
    )

    print(f"  - speaker_selection_method: {groupchat_auto.speaker_selection_method}")
    print(f"  - max_round: {groupchat_auto.max_round}")
    print("  - 优势：灵活度高，LLM根据上下文选择合适的speaker")
    print("  - 注意：终止时机取决于对话内容，不如round_robin可预测")

    # ---------------------------------------------
    # 决策框架
    # ---------------------------------------------
    print("\n发言策略选择决策框架:")
    print("  | 场景                    | 推荐策略       | 原因           |")
    print("  |-------------------------|----------------|----------------|")
    print("  | 研讨会/头脑风暴          | auto           | 需要灵活性     |")
    print("  | 代码审查/固定流程        | round_robin    | 需要可预测性   |")
    print("  | 快速投票/随机调研        | random         | 需要公平性     |")
    print("  |-------------------------|----------------|----------------|")


def demo_termination_by_speaker_count():
    """
    演示 speaker_count 终止条件

    speaker_count 限制某个 speaker 被选择的次数。
    当某个 Agent 被选择达到指定次数时，强制终止对话。

    适用场景：
    - 确保每个 Agent 都有发言机会
    - 防止某个 Agent 主导整个讨论
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: speaker_count 终止条件")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建三个 Agent
    agent_a = ConversableAgent(
        name="设计师",
        system_message="你是设计师，负责提出设计方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="开发者",
        system_message="你是开发者，负责实现方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_c = ConversableAgent(
        name="测试员",
        system_message="你是测试员，负责验证方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 配置：限制特定 Agent 的发言次数
    # ---------------------------------------------
    print("\n配置 speaker_count 限制:")
    print("  - 设计师: 限制发言3次")
    print("  - 开发者: 限制发言5次")
    print("  - 测试员: 限制发言2次")

    # 注意：speaker_count 是通过 max_round 间接控制的
    # 如果需要严格限制某个 Agent 的发言次数，可以使用 custom selection function

    groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",  # 使用轮询确保均衡发言
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print(f"\n实际配置:")
    print(f"  - 使用 round_robin 确保均衡发言")
    print(f"  - max_round: {groupchat.max_round}（总轮次限制）")
    print("  - 注意：AutoGen 原生不支持精确的 speaker_count 控制")
    print("  - 如需精确控制，需要自定义 select_speaker 函数")

    print("\nspeaker_count 使用建议:")
    print("  1. 通过 max_round 间接控制总发言次数")
    print("  2. 使用 round_robin 确保均衡发言")
    print("  3. 如需精确控制某个 Agent 发言次数，需要自定义逻辑")


def demo_advanced_termination_patterns():
    """
    演示高级终止模式：基于消息内容的智能终止

    高级技巧：
    1. 多阶段终止：不同阶段使用不同的终止条件
    2. 角色感知终止：根据回复者的角色决定是否终止
    3. 上下文感知终止：根据对话历史决定是否终止
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示6: 高级终止模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 模式1：角色感知终止
    # ---------------------------------------------
    print("\n--- 模式1: 角色感知终止 ---")

    def role_aware_termination(msg):
        """
        角色感知终止：只有特定角色的回复才能触发终止

        例如：只有"总结员"说"完成"才终止，其他人说"完成"不终止
        """
        content = msg.get("content", "")
        name = msg.get("name", "")

        # 只有"总结员"说"完成"才终止
        if name == "总结员" and "完成" in content:
            return True

        return False

    summarizer = ConversableAgent(
        name="总结员",
        system_message="你是总结员，负责总结讨论结果。回复时以'总结员'开头。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    analyst = ConversableAgent(
        name="分析员",
        system_message="你是分析员，负责分析问题。回复时以'分析员'开头。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("  - 规则：只有'总结员'说'完成'才终止")
    print("  - 优势：确保只有特定角色可以结束讨论")

    # ---------------------------------------------
    # 模式2：轮次感知终止
    # ---------------------------------------------
    print("\n--- 模式2: 轮次感知终止 ---")

    def round_aware_termination(msg):
        """
        轮次感知终止：根据当前轮次调整终止条件

        例如：前3轮不终止，之后检查终止条件
        """
        # 这种逻辑需要在 GroupChatManager 中实现
        # 这里仅展示概念
        return False  # 示例代码，实际需要访问群聊状态

    print("  - 规则：根据轮次动态调整终止条件")
    print("  - 优势：适应对话的不同阶段")
    print("  - 注意：需要在 Manager 中维护轮次状态")

    print("\n高级模式使用建议:")
    print("  1. 角色感知适用于需要明确主持人的场景")
    print("  2. 轮次感知适用于多阶段任务")
    print("  3. 上下文感知适用于复杂对话逻辑")


def demo_termination_config_checklist():
    """
    演示终止条件配置检查清单

    创建一个实用的检查清单，帮助配置正确的终止条件
    """
    print("\n" + "=" * 60)
    print("演示7: 终止条件配置检查清单")
    print("=" * 60)

    print("\n配置 GroupChat 终止条件时的检查项目:")
    print("")

    checklist = [
        ("1. max_round 设置", [
            "是否设置了合理的 max_round？",
            "max_round 是否足够完成预期任务？",
            "是否设置了保底机制防止无限循环？",
        ]),
        ("2. termination_msg 设置", [
            "是否定义了清晰的终止条件？",
            "终止关键词是否覆盖了主要场景？",
            "是否测试过各种边界情况？",
        ]),
        ("3. speaker_selection_method", [
            "选择的策略是否符合任务需求？",
            "是否需要均衡发言（round_robin）？",
            "是否需要灵活性（auto）？",
        ]),
        ("4. 双重保护", [
            "是否同时设置了内容终止和轮次终止？",
            "两种终止条件的优先级是否正确？",
        ]),
    ]

    for category, items in checklist:
        print(f"  {category}")
        for item in items:
            print(f"    [ ] {item}")
        print("")

    print("常见配置模式:")
    print("")
    print("  模式A: 快速任务（代码审查）")
    print("    max_round=5, speaker_selection=round_robin")
    print("")
    print("  模式B: 标准任务（问题分析）")
    print("    max_round=10, termination_msg=自定义, speaker_selection=auto")
    print("")
    print("  模式C: 复杂任务（架构设计）")
    print("    max_round=20, termination_msg=自定义, speaker_selection=auto")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有终止条件配置演示
    """
    print("=" * 60)
    print("GroupChat 终止条件配置演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_max_round_termination()
    demo_termination_msg_in_groupchat()
    demo_termination_msg_vs_max_round()
    demo_speaker_selection_and_termination()
    demo_termination_by_speaker_count()
    demo_advanced_termination_patterns()
    demo_termination_config_checklist()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()