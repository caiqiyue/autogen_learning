# groupchat_problems.py
# 第11节 GroupChat终止条件与常见问题处理 - 常见问题及解决方案
#
# 本文件演示 GroupChat 中的三种典型问题及解决方案：
# 1. 死循环 - 对话无法正常终止
# 2. 发言不均 - 某些 Agent 过度参与或参与不足
# 3. 过早终止 - 对话在任务完成前就结束了
#
# 每个问题都包含问题描述、原因分析、解决方案和代码示例。
#
# ============================================================
# GroupChat 常见问题概览
# ============================================================
#
# 问题类型          发生频率    严重程度    解决难度
# ---------------------------------------------------------
# 死循环            中          高          中
# 发言不均          高          中          高
# 过早终止          中          中          低
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
# 第三部分：问题1 - 死循环及解决方案
# ============================================================

def demo_problem_dead_loop():
    """
    问题描述：死循环

    现象：
    - 对话无法正常终止，一直进行下去
    - 达到 max_round 后仍然继续
    - Agent 重复发送类似的消息

    原因分析：
    1. is_termination_msg 未正确设置或过于宽松
    2. max_round 设置过大
    3. Agent 的系统提示导致无限循环（如"不断提问"）
    4. LLM 陷入重复模式（hallucination loop）
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("问题1: 死循环 - 现象与原因分析")
    print("=" * 60)

    print("\n【现象】")
    print("  - 对话持续进行，无法终止")
    print("  - Agent 反复讨论相同话题")
    print("  - 达到 max_round 后仍然继续（如果配置不当）")

    print("\n【原因分析】")
    print("  原因1: is_termination_msg 过于宽松")
    print("         例如：只检查'再见'，但 Agent 从不说'再见'")
    print("")
    print("  原因2: max_round 设置过大")
    print("         例如：设置为 100，实际只需要 5 轮")
    print("")
    print("  原因3: Agent 系统提示设计不当")
    print("         例如：提示包含'不断追问'等指令")
    print("")
    print("  原因4: LLM 陷入重复模式")
    print("         模型反复生成相似的回复")


def demo_solution_dead_loop():
    """
    解决方案：防止死循环

    策略1：设置合理的 max_round
    策略2：配置严格的终止条件
    策略3：使用 round_robin 确保均衡发言
    策略4：添加超时机制（异步场景）
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("解决方案: 防止死循环")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建 Agent
    agent_a = ConversableAgent(
        name="规划师",
        system_message="你是一位规划师，负责提出方案。回复要简洁，不要重复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="执行者",
        system_message="你是一位执行者，负责评估方案并给出结论。回复要简洁。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 解决方案1：设置合理的 max_round
    # ---------------------------------------------
    print("\n【策略1】设置合理的 max_round")

    max_round_recommended = 10  # 对于大多数场景，10轮足够

    print(f"  推荐值: {max_round_recommended}")
    print("  原则：设置比预期稍大的值，但不要过大")
    print("  示例：")
    print("    - 简单问答: 3-5")
    print("    - 标准任务: 10-15")
    print("    - 复杂任务: 20-30")

    # ---------------------------------------------
    # 解决方案2：配置严格的终止条件
    # ---------------------------------------------
    print("\n【策略2】配置严格的终止条件")

    def strict_termination(msg):
        """
        严格的终止条件：多种情况都会触发终止

        终止条件：
        1. 包含"完成"、"结束"、"结论"等完成标记
        2. 包含"不需要再讨论"等明确退出指令
        3. 消息来自特定角色（如"总结员"）
        """
        content = msg.get("content", "").lower()

        # 完成标记
        completion_keywords = ["完成", "结束", "结论", "搞定", "可以了"]
        if any(kw in content for kw in completion_keywords):
            return True

        # 退出指令
        exit_keywords = ["不需要再讨论", "到此为止", "就这样", "再见"]
        if any(kw in content for kw in exit_keywords):
            return True

        return False

    print("  终止条件设计要点:")
    print("    - 覆盖多种完成场景")
    print("    - 包含明确的退出指令")
    print("    - 不依赖单一关键词")

    # ---------------------------------------------
    # 解决方案3：使用 round_robin 确保均衡
    # ---------------------------------------------
    print("\n【策略3】使用 round_robin 确保均衡发言")

    groupchat = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",  # 轮询选择，防止某方过度发言
        termination_msg=strict_termination,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print(f"  配置:")
    print(f"    - speaker_selection_method: round_robin")
    print(f"    - max_round: {groupchat.max_round}")
    print(f"    - termination_msg: 已设置")
    print("  优势：每个 Agent 都有均等的发言机会，避免单方垄断")

    # ---------------------------------------------
    # 解决方案4：超时机制（异步场景）
    # ---------------------------------------------
    print("\n【策略4】超时机制（异步场景）")

    print("  import asyncio")
    print("  async def run_with_timeout(groupchat, timeout=60):")
    print("      try:")
    print("          result = await asyncio.wait_for(")
    print("              groupchat_manager.run(),")
    print("              timeout=timeout")
    print("          )")
    print("          return result")
    print("      except asyncio.TimeoutError:")
    print("          # 超时后强制终止")
    print("          groupchat.reset()")
    print("          return None")

    print("\n防止死循环的配置模板:")
    print("-" * 40)
    print("""
    groupchat = GroupChat(
        agents=[agent_a, agent_b],
        messages=[],
        max_round=10,                    # 设置保底上限
        speaker_selection_method="round_robin",  # 均衡发言
        termination_msg=strict_termination,     # 严格的终止条件
    )
    """)
    print("-" * 40)


# ============================================================
# 第四部分：问题2 - 发言不均及解决方案
# ============================================================

def demo_problem_speaker_imbalance():
    """
    问题描述：发言不均

    现象：
    - 某些 Agent 过度参与，其他 Agent 几乎没有发言机会
    - 使用 auto 模式时，LLM 可能倾向于选择同一个 Agent
    - 对话被某个 Agent 主导

    原因分析：
    1. speaker_selection_method="auto" 时，LLM 可能偏袒某个 Agent
    2. Agent 系统提示设计导致某些 Agent 更健谈
    3. Agent 数量不均衡（如2个 vs 5个）
    4. 话题与某些 Agent 领域不相关
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("问题2: 发言不均 - 现象与原因分析")
    print("=" * 60)

    print("\n【现象】")
    print("  - 某个 Agent 发言次数远超其他 Agent")
    print("  - 某些 Agent 几乎没有发言机会")
    print("  - 使用 auto 模式时问题更明显")

    print("\n【原因分析】")
    print("  原因1: auto 模式的 LLM 偏好")
    print("         LLM 可能倾向于选择某个'更健谈'的 Agent")
    print("")
    print("  原因2: Agent 系统提示差异")
    print("         某些 Agent 的提示更详细，导致更容易被选中")
    print("")
    print("  原因3: Agent 能力差异")
    print("         某些 Agent 的回复更有'价值'，被优先选择")
    print("")
    print("  原因4: 话题相关性")
    print("         如果话题与某个 Agent 领域高度相关，它可能一直被选中")


def demo_solution_speaker_imbalance():
    """
    解决方案：解决发言不均问题

    策略1：使用 round_robin 强制均衡
    策略2：自定义 select_speaker 函数实现均衡策略
    策略3：监控发言次数并动态调整
    策略4：调整 Agent 系统提示使其行为一致
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("解决方案: 解决发言不均")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建三个 Agent
    agent_a = ConversableAgent(
        name="设计师",
        system_message="你是设计师，负责提出创意方案。简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="开发者",
        system_message="你是开发者，负责实现方案。简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_c = ConversableAgent(
        name="测试员",
        system_message="你是测试员，负责验证方案。简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 解决方案1：使用 round_robin 强制均衡
    # ---------------------------------------------
    print("\n【策略1】使用 round_robin 强制均衡")

    groupchat_rr = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        messages=[],
        max_round=9,  # 确保每个 Agent 都能发言3次
        speaker_selection_method="round_robin",  # 强制轮询
    )

    print("  配置:")
    print(f"    - speaker_selection_method: round_robin")
    print(f"    - 预期发言分布: 设计师 -> 开发者 -> 测试员 -> (循环)")
    print("  优势：完全均衡，可预测")
    print("  劣势：缺乏灵活性")

    # ---------------------------------------------
    # 解决方案2：自定义均衡选择函数
    # ---------------------------------------------
    print("\n【策略2】自定义均衡选择函数")

    # 记录发言次数的字典（实际使用时需要更复杂的机制）
    speaker_counts = {"设计师": 0, "开发者": 0, "测试员": 0}

    def balanced_select_speaker(groupchat: GroupChat, last_speaker=None):
        """
        均衡选择函数：优先选择发言次数最少的 Agent

        Args:
            groupchat: GroupChat 实例
            last_speaker: 上一个发言的 Agent

        Returns:
            下一个发言的 Agent
        """
        agents = groupchat.agents

        # 如果只有一个 Agent，直接返回
        if len(agents) == 1:
            return agents[0]

        # 排除上一个发言的 Agent（除非只有两个）
        candidates = [a for a in agents if a != last_speaker]

        # 如果只有一个候选者，返回它
        if len(candidates) == 1:
            return candidates[0]

        # 选择发言次数最少的候选者
        def get_count(agent):
            return speaker_counts.get(agent.name, 0)

        return min(candidates, key=get_count)

    print("  自定义选择函数示例:")
    print("    def balanced_select_speaker(groupchat, last_speaker):")
    print("        # 优先选择发言次数最少的 Agent")
    print("        candidates = [a for a in groupchat.agents if a != last_speaker]")
    print("        return min(candidates, key=lambda a: speaker_counts[a.name])")
    print("  优势：兼顾均衡与灵活")
    print("  注意：需要维护发言计数状态")

    # ---------------------------------------------
    # 解决方案3：监控发言次数
    # ---------------------------------------------
    print("\n【策略3】监控发言次数并动态调整")

    def create_speaker_monitoring_groupchat(agents, max_round=10):
        """
        创建带发言监控的 GroupChat

        每次选择 speaker 后记录其发言次数
        """
        groupchat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round,
            speaker_selection_method="auto",
        )

        # 添加监控逻辑（需要在选择后更新计数）
        return groupchat

    print("  监控要点:")
    print("    - 记录每个 Agent 的发言次数")
    print("    - 当某个 Agent 发言过多时，调整选择概率")
    print("    - 在对话结束时输出统计信息")

    print("\n发言均衡的配置模板:")
    print("-" * 40)
    print("""
    # 方案1: round_robin（简单可靠）
    groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        max_round=9,
        speaker_selection_method="round_robin",
    )

    # 方案2: 自定义均衡选择
    def balanced_select_speaker(groupchat, last_speaker):
        candidates = [a for a in groupchat.agents if a != last_speaker]
        return min(candidates, key=lambda a: speaker_counts[a.name])

    groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        max_round=10,
        speaker_selection_method=balanced_select_speaker,
    )
    """)
    print("-" * 40)


# ============================================================
# 第五部分：问题3 - 过早终止及解决方案
# ============================================================

def demo_problem_early_termination():
    """
    问题描述：过早终止

    现象：
    - 对话在任务完成前就结束了
    - Agent 说"完成"但实际任务未完成
    - 达到终止条件但结果不完整

    原因分析：
    1. is_termination_msg 条件过于宽松（如任何包含"。"的消息都终止）
    2. Agent 误解任务，提前说"完成"
    3. max_round 设置过小
    4. 终止条件设计未考虑边界情况
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("问题3: 过早终止 - 现象与原因分析")
    print("=" * 60)

    print("\n【现象】")
    print("  - 对话在任务完成前结束")
    print("  - 结果不完整，缺少关键步骤")
    print("  - Agent 说'完成'但实际只是部分完成")

    print("\n【原因分析】")
    print("  原因1: is_termination_msg 过于宽松")
    print("         例如：任何包含'。'的消息都终止")
    print("")
    print("  原因2: Agent 误解任务")
    print("         LLM 可能过早判断任务完成")
    print("")
    print("  原因3: max_round 设置过小")
    print("         没有给足对话轮次")
    print("")
    print("  原因4: 终止条件设计不当")
    print("         没有考虑任务的具体要求")


def demo_solution_early_termination():
    """
    解决方案：防止过早终止

    策略1：使用保守的终止条件
    策略2：增加 max_round 预留空间
    策略3：使用多条件组合的终止判断
    策略4：添加确认机制（human_input_mode='TERMINATE'）
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("解决方案: 防止过早终止")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建 Agent
    researcher = ConversableAgent(
        name="研究员",
        system_message="你是一位研究员，负责深入分析问题。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    reviewer = ConversableAgent(
        name="评审员",
        system_message="你是一位评审员，负责评估分析质量。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 解决方案1：使用保守的终止条件
    # ---------------------------------------------
    print("\n【策略1】使用保守的终止条件")

    def conservative_termination(msg):
        """
        保守的终止条件：只有明确的任务完成标记才终止

        与宽松条件相反，要求多个条件同时满足
        """
        content = msg.get("content", "")

        # 需要同时满足多个条件才终止
        conditions = []

        # 条件1: 包含"完成"
        conditions.append("完成" in content)

        # 条件2: 内容足够长（不是简单敷衍）
        conditions.append(len(content) > 100)

        # 条件3: 包含具体结论
        conclusion_keywords = ["结论", "因此", "所以", "总结"]
        conditions.append(any(kw in content for kw in conclusion_keywords))

        # 所有条件都满足才终止
        return all(conditions)

    print("  保守条件示例:")
    print("    - 需要同时满足：完成 + 内容长 + 有结论")
    print("    - 避免误判：只有'完成了'这样的短消息不会终止")
    print("  优势：大幅降低过早终止的概率")

    # ---------------------------------------------
    # 解决方案2：增加 max_round 预留空间
    # ---------------------------------------------
    print("\n【策略2】增加 max_round 预留空间")

    expected_rounds = 5  # 预期轮次
    buffer_multiplier = 2  # 缓冲倍数

    max_round_safe = expected_rounds * buffer_multiplier

    print(f"  预期轮次: {expected_rounds}")
    print(f"  缓冲倍数: {buffer_multiplier}")
    print(f"  推荐的 max_round: {max_round_safe}")
    print("  原则：宁可多几轮，也不要过早终止")

    # ---------------------------------------------
    # 解决方案3：多条件组合的终止判断
    # ---------------------------------------------
    print("\n【策略3】多条件组合的终止判断")

    def multi_condition_termination(msg):
        """
        多条件终止：需要多种完成信号同时满足

        终止条件（需要同时满足）：
        1. 消息来自评审员
        2. 包含"通过"或"完成"
        3. 不包含"需要"或"建议"（未完成的工作）
        """
        content = msg.get("content", "")
        name = msg.get("name", "")

        # 条件1: 只有评审员可以触发终止
        condition1 = name == "评审员"

        # 条件2: 包含完成标记
        condition2 = "通过" in content or "完成" in content

        # 条件3: 不包含未完成标记
        unfinished_markers = ["需要", "建议", "还要", "应该"]
        condition3 = not any(marker in content for marker in unfinished_markers)

        return condition1 and condition2 and condition3

    print("  多条件组合的优势:")
    print("    - 更精确：只有满足所有条件才终止")
    print("    - 更可靠：避免了单一条件误判")
    print("    - 更灵活：可以根据任务需求调整条件")

    # ---------------------------------------------
    # 解决方案4：添加确认机制
    # ---------------------------------------------
    print("\n【策略4】添加确认机制（TERMINATE 模式）")

    agent_confirm = ConversableAgent(
        name="确认助手",
        system_message="你是确认助手，当任务完成时说'完成'。",
        llm_config=llm_config,
        human_input_mode="TERMINATE",  # 关键：终止前请求确认
        is_termination_msg=lambda msg: "完成" in msg.get("content", ""),
    )

    print("  human_input_mode='TERMINATE' 的行为:")
    print("    1. 当 is_termination_msg 返回 True 时")
    print("    2. 系统会请求人类确认是否真正终止")
    print("    3. 人类输入 'y' 确认终止，输入 'n' 继续对话")
    print("  适用场景：关键任务需要人工确认完成状态")

    print("\n防止过早终止的配置模板:")
    print("-" * 40)
    print("""
    # 方案1: 保守的终止条件
    def conservative_termination(msg):
        return ("完成" in msg["content"] and
                len(msg["content"]) > 100 and
                any(kw in msg["content"] for kw in ["结论", "因此"]))

    # 方案2: 多条件组合
    def multi_condition_termination(msg):
        return (msg["name"] == "评审员" and
                "通过" in msg["content"] and
                not any(m in msg["content"] for m in ["需要", "建议"]))

    # 方案3: 预留足够的 max_round
    max_round = expected_rounds * 2  # 缓冲倍数

    groupchat = GroupChat(
        agents=[researcher, reviewer],
        max_round=max_round,
        termination_msg=multi_condition_termination,
    )
    """)
    print("-" * 40)


# ============================================================
# 第六部分：综合诊断与调试
# ============================================================

def demo_diagnosis_and_debug():
    """
    演示 GroupChat 问题诊断与调试技巧

    技巧1：打印消息历史分析问题
    技巧2：监控 speaker 选择过程
    技巧3：记录终止条件触发情况
    """
    print("\n" + "=" * 60)
    print("综合诊断与调试技巧")
    print("=" * 60)

    # ---------------------------------------------
    # 技巧1：打印消息历史分析问题
    # ---------------------------------------------
    print("\n【技巧1】打印消息历史分析问题")

    print("""
    # 在对话结束后打印详细的消息历史
    for i, msg in enumerate(result.chat_history):
        print(f"[{i}] {msg.get('name', 'unknown')}: {msg.get('content', '')[:50]}...")
    """)

    print("  分析要点:")
    print("    - 检查消息数量是否符合预期")
    print("    - 检查发言顺序是否合理")
    print("    - 检查是否有异常模式（如重复消息）")

    # ---------------------------------------------
    # 技巧2：监控 speaker 选择过程
    # ---------------------------------------------
    print("\n【技巧2】监控 speaker 选择过程")

    print("""
    # 自定义选择函数中添加日志
    def debug_select_speaker(groupchat, last_speaker):
        print(f"[DEBUG] last_speaker={last_speaker}")
        print(f"[DEBUG] available_agents={groupchat.agents}")

        selected = auto_select(groupchat, last_speaker)
        print(f"[DEBUG] selected={selected}")

        return selected
    """)

    print("  分析要点:")
    print("    - 检查选择逻辑是否符合预期")
    print("    - 识别是否有偏好某个 Agent 的问题")

    # ---------------------------------------------
    # 技巧3：记录终止条件触发情况
    # ---------------------------------------------
    print("\n【技巧3】记录终止条件触发情况")

    print("""
    # 在终止条件函数中添加日志
    def logged_termination(msg):
        content = msg.get("content", "")
        result = "完成" in content or "结束" in content
        print(f"[TERMINATION CHECK] content='{content[:30]}...' -> {result}")
        return result
    """)

    print("  分析要点:")
    print("    - 确认终止条件是否正常工作")
    print("    - 识别过早或过晚终止的原因")

    print("\n调试检查清单:")
    print("  [ ] 打印消息历史，检查轮次是否符合预期")
    print("  [ ] 添加选择函数日志，检查 speaker 选择逻辑")
    print("  [ ] 添加终止条件日志，检查触发时机")
    print("  [ ] 检查 max_round 是否足够完成任务")
    print("  [ ] 验证 termination_msg 是否正确设置")


# ============================================================
# 第七部分：决策框架
# ============================================================

def demo_decision_framework():
    """
    演示发言策略选择的决策框架

    根据不同场景选择合适的 speaker_selection_method
    """
    print("\n" + "=" * 60)
    print("发言策略选择决策框架")
    print("=" * 60)

    print("\n决策矩阵:")
    print("-" * 60)
    print("| 场景              | 推荐策略        | 终止配置              |")
    print("|-------------------|-----------------|----------------------|")
    print("| 代码审查          | round_robin     | max_round=5-10       |")
    print("| 头脑风暴          | auto            | max_round=15-20      |")
    print("| 快速投票          | random          | max_round=3-5        |")
    print("| 架构设计          | auto            | max_round=20-30      |")
    print("| 教学辅导          | round_robin     | max_round=10-15      |")
    print("| 模拟面试          | auto            | max_round=20-30      |")
    print("|-------------------|-----------------|----------------------|")

    print("\n场景详解:")
    print("")
    print("  1. 代码审查")
    print("     - 策略: round_robin（作者 -> 审查者 -> 作者 -> ...）")
    print("     - 终止: max_round=5-10，确保每次审查有明确结论")
    print("")
    print("  2. 头脑风暴")
    print("     - 策略: auto（LLM 根据话题选择最合适的发言者）")
    print("     - 终止: max_round=15-20，给足创意碰撞的时间")
    print("")
    print("  3. 快速投票")
    print("     - 策略: random（公平选择，避免偏好）")
    print("     - 终止: max_round=3-5，快速收敛")
    print("")
    print("  4. 架构设计")
    print("     - 策略: auto（不同阶段需要不同专家）")
    print("     - 终止: max_round=20-30，设计需要多轮迭代")
    print("")
    print("  5. 教学辅导")
    print("     - 策略: round_robin（教师 -> 学生 -> 教师）")
    print("     - 终止: max_round=10-15，确保知识传递完整")
    print("")
    print("  6. 模拟面试")
    print("     - 策略: auto（根据面试者回答选择下一个问题）")
    print("     - 终止: max_round=20-30，全面评估候选人")


# ============================================================
# 第八部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有问题及解决方案演示
    """
    print("=" * 60)
    print("GroupChat 常见问题及解决方案")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_problem_dead_loop()
    demo_solution_dead_loop()
    demo_problem_speaker_imbalance()
    demo_solution_speaker_imbalance()
    demo_problem_early_termination()
    demo_solution_early_termination()
    demo_diagnosis_and_debug()
    demo_decision_framework()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n总结：GroupChat 问题处理要点")
    print("  1. 死循环 -> 设置合理的 max_round + 严格的终止条件")
    print("  2. 发言不均 -> 使用 round_robin 或自定义均衡选择函数")
    print("  3. 过早终止 -> 使用保守的终止条件 + 预留足够的轮次")
    print("  4. 综合调试 -> 打印消息历史 + 监控选择过程 + 记录终止触发")


if __name__ == "__main__":
    main()