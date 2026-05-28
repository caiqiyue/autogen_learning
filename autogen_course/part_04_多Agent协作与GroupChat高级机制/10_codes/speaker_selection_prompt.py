# speaker_selection_prompt.py
# 第10节 speaker_selection_mode 三种策略详解 - Prompt优化演示
#
# 本文件演示如何在 auto 模式下优化提示词，以提高speaker选择的准确性：
# 1. 默认的speaker选择提示词
# 2. 自定义speaker选择提示词
# 3. 基于角色描述的优化
# 4. 条件触发式提示词优化
#
# ============================================================
# 为什么需要优化 auto 模式的提示词
# ============================================================
#
# 在 auto 模式下，GroupChatManager 会调用 LLM 来选择下一个发言者。
# 默认的提示词可能不够精确，导致选择结果不符合预期。
# 通过优化提示词，可以：
# 1. 更精确地控制speaker选择逻辑
# 2. 根据对话阶段动态调整选择策略
# 3. 实现基于条件的发言顺序
# 4. 提高多Agent协作的效率
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
# 第三部分：Prompt优化技术详解
# ============================================================
#
# auto 模式的提示词优化主要涉及以下几个方面：
#
# 1. speaker_prompt（选择提示词）
#    - 定义如何选择下一个发言者
#    - 包含选择逻辑和优先级
#    - 可以根据上下文动态调整
#
# 2. 角色描述优化
#    - 在Agent的system_message中清晰定义角色
#    - 描述角色的职责和发言时机
#    - 帮助LLM理解谁应该发言
#
# 3. 条件触发式发言
#    - 基于对话状态触发特定Agent发言
#    - 实现流程控制
#    - 确保关键步骤不被跳过
#
# ============================================================


def demo_default_speaker_prompt():
    """
    演示默认的 speaker 选择提示词

    AutoGen 使用默认的提示词来让LLM选择下一个发言者。
    默认提示词会包含：
    - 当前的对话历史摘要
    - 所有可用Agent的角色信息
    - 选择下一个发言者的指令

    了解默认提示词有助于我们理解如何优化它。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: 默认 speaker 选择提示词")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建简单的Agent团队
    # ---------------------------------------------

    planner = ConversableAgent(
        name="规划师",
        system_message="你是规划师，负责制定计划。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    executor = ConversableAgent(
        name="执行者",
        system_message="你是执行者，负责执行计划。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    reviewer = ConversableAgent(
        name="审查员",
        system_message="你是审查员，负责审查结果。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: 规划师, 执行者, 审查员")

    # ---------------------------------------------
    # 创建 GroupChat，查看默认配置
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[planner, executor, reviewer],
        messages=[],
        max_round=6,
        speaker_selection_method="auto",
    )

    print("\nGroupChat 默认配置:")
    print(f"  - speaker_selection_method: {groupchat.speaker_selection_method}")
    print(f"  - speaker_selection_prompt: {groupchat.speaker_selection_prompt[:100]}...")

    # ---------------------------------------------
    # 分析默认提示词结构
    # ---------------------------------------------

    print("\n=== 默认提示词结构分析 ===")
    print("\n默认的 speaker_selection_prompt 通常包含:")
    print("  1. 当前对话状态描述")
    print("     - 最近几条消息的内容")
    print("     - 当前讨论的主题")
    print("  2. 可用Agent列表")
    print("     - 每个Agent的名字和角色")
    print("     - Agent的系统提示摘要")
    print("  3. 选择指令")
    print("     - 选择下一个发言者的标准")
    print("     - 输出格式要求")

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 演示默认行为
    print("\n执行群聊:")
    result = planner.initiate_chat(
        manager,
        message="请完成一个任务：先规划，再执行，最后审查。",
    )

    print(f"总消息数: {len(result.chat_history)}")

    # 分析发言模式
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")


def demo_custom_speaker_prompt():
    """
    演示自定义 speaker 选择提示词

    通过设置 speaker_selection_prompt，可以自定义选择逻辑。
    自定义提示词应该包含：
    - 选择标准和优先级
    - 特定场景下的选择规则
    - 输出格式要求
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: 自定义 speaker 选择提示词")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    researcher = ConversableAgent(
        name="研究员",
        system_message="""你是研究员，负责进行研究和分析。
你会根据讨论的进展，被要求提供研究报告。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    developer = ConversableAgent(
        name="开发者",
        system_message="""你是开发者，负责实现具体功能。
你会根据需求实现代码，并在被要求时解释技术细节。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    tester = ConversableAgent(
        name="测试员",
        system_message="""你是测试员，负责测试和验证。
你会检查功能是否正确实现，并报告发现的问题。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: 研究员, 开发者, 测试员")

    # ---------------------------------------------
    # 自定义选择提示词
    # ---------------------------------------------

    custom_prompt = """你是一个对话协调者，负责选择下一个发言者。

当前对话状态：
{context}

可用发言者：
{agents}

选择规则：
1. 如果讨论刚开始或涉及研究方向，选择"研究员"
2. 如果需要实现具体功能，选择"开发者"
3. 如果需要验证或测试，选择"测试员"
4. 除非必要，避免重复选择同一个发言者

请选择一个发言者，只输出发言者的名字，不要其他内容。
输出格式：研究员 | 开发者 | 测试员
"""

    print("\n自定义提示词:")
    print("-" * 40)
    print("关键要素:")
    print("  1. 明确的选择规则（基于任务类型）")
    print("  2. 考虑发言均衡（避免重复）")
    print("  3. 清晰的输出格式要求")
    print("  4. 基于上下文状态做出决策")

    groupchat = GroupChat(
        agents=[researcher, developer, tester],
        messages=[],
        max_round=9,
        speaker_selection_method="auto",
        speaker_selection_prompt=custom_prompt,  # 使用自定义提示词
    )

    print(f"\n配置: 使用自定义 speaker_selection_prompt")
    print(f"  - 提示词长度: {len(custom_prompt)} 字符")

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 演示自定义提示词效果
    print("\n执行群聊:")
    result = researcher.initiate_chat(
        manager,
        message="我们需要研究一个新功能，然后实现它，最后测试它。",
    )

    print(f"\n总消息数: {len(result.chat_history)}")

    # 分析发言模式
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")


def demo_role_based_prompt_optimization():
    """
    演示基于角色描述的提示词优化

    通过在Agent的系统提示中更详细地描述角色和发言时机，
    可以帮助LLM更准确地选择下一个发言者。

    优化策略：
    1. 清晰定义角色的职责范围
    2. 描述角色应该何时发言
    3. 说明角色发言的特点和风格
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: 基于角色描述的提示词优化")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建具有详细角色描述的Agent
    # ---------------------------------------------

    # 优化后的规划师 - 包含详细的发言时机描述
    planner_optimized = ConversableAgent(
        name="规划师",
        system_message="""你是项目规划师。

【角色职责】
- 分析需求，制定项目计划
- 协调团队资源分配
- 跟踪项目进度

【发言时机】
- 讨论开始时，提出项目计划
- 需要决策时，提供选项分析
- 讨论结束时，总结计划要点

【发言特点】
- 结构化表达，使用编号列表
- 简洁明了，重点突出
- 主动引导讨论方向""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 优化后的开发者
    developer_optimized = ConversableAgent(
        name="开发者",
        system_message="""你是资深开发者。

【角色职责】
- 根据计划实现代码
- 解释技术实现细节
- 解决技术难题

【发言时机】
- 被要求实现功能时
- 需要解释技术方案时
- 讨论技术可行性时

【发言特点】
- 代码优先，用代码说明问题
- 技术细节准确
- 乐于分享实现经验""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 优化后的测试工程师
    tester_optimized = ConversableAgent(
        name="测试工程师",
        system_message="""你是测试工程师。

【角色职责】
- 设计测试用例
- 执行功能测试
- 报告缺陷和改进建议

【发言时机】
- 代码实现后，要求进行测试
- 发现问题时，提出缺陷报告
- 验证完成后，确认功能正常

【发言特点】
- 注重细节，观察敏锐
- 测试用例结构化
- 报告清晰，有优先级""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个具有详细角色描述的Agent:")
    print("  - 规划师: 包含职责、发言时机、发言特点")
    print("  - 开发者: 包含职责、发言时机、发言特点")
    print("  - 测试工程师: 包含职责、发言时机、发言特点")

    # ---------------------------------------------
    # 增强的选择提示词
    # ---------------------------------------------

    enhanced_prompt = """你是群聊协调者，需要根据对话上下文选择最合适的下一个发言者。

【当前对话状态】
{context}

【可用Agent及其角色特征】
{agents}

【选择策略】
1. 识别当前讨论阶段：
   - 需求分析/规划阶段 → 选择"规划师"
   - 实现/开发阶段 → 选择"开发者"
   - 验证/测试阶段 → 选择"测试工程师"

2. 考虑发言均衡：
   - 除非必要，不连续选择同一Agent
   - 确保每个角色都有发言机会

3. 响应用户意图：
   - 如果用户指定了特定角色，优先考虑
   - 如果用户提问涉及特定领域，选择相关角色

【输出要求】
只输出Agent的名字，不要其他内容。
例如：规划师
"""

    print("\n增强的选择提示词特点:")
    print("  1. 三段式结构：状态 -> Agent -> 策略")
    print("  2. 明确的阶段对应关系")
    print("  3. 考虑发言均衡")
    print("  4. 响应用户意图")

    groupchat = GroupChat(
        agents=[planner_optimized, developer_optimized, tester_optimized],
        messages=[],
        max_round=9,
        speaker_selection_method="auto",
        speaker_selection_prompt=enhanced_prompt,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 演示效果
    print("\n执行群聊:")
    result = planner_optimized.initiate_chat(
        manager,
        message="我们需要开发一个新的用户认证功能。",
    )

    print(f"\n总消息数: {len(result.chat_history)}")

    # 分析发言模式
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")

    # 统计各Agent发言次数
    from collections import Counter
    counts = Counter(speakers)
    print("\n各Agent发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")


def demo_conditional_trigger_prompt():
    """
    演示条件触发式提示词优化

    条件触发式提示词允许基于特定条件选择发言者，
    实现更精确的流程控制。

    场景示例：
    - 用户明确要求某人发言时，强制选择该Agent
    - 检测到特定关键词时，触发特定Agent
    - 对话达到特定阶段时，自动选择对应Agent
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: 条件触发式提示词优化")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建针对不同任务的Agent
    # ---------------------------------------------

    data_analyst = ConversableAgent(
        name="数据分析师",
        system_message="""你是数据分析师。
专长：数据分析、统计建模、数据可视化。
当讨论涉及数据分析时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    ml_engineer = ConversableAgent(
        name="机器学习工程师",
        system_message="""你是机器学习工程师。
专长：机器学习算法、模型训练、特征工程。
当讨论涉及机器学习时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    devops_engineer = ConversableAgent(
        name="运维工程师",
        system_message="""你是运维工程师。
专长：系统部署、监控、自动化运维。
当讨论涉及部署或运维时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    general_assistant = ConversableAgent(
        name="通用助手",
        system_message="""你是通用助手。
处理一般性问题，当没有特定专家被需要时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建4个Agent:")
    print("  - 数据分析师: 专门处理数据分析相关问题")
    print("  - 机器学习工程师: 专门处理机器学习相关问题")
    print("  - 运维工程师: 专门处理部署运维相关问题")
    print("  - 通用助手: 处理一般性问题")

    # ---------------------------------------------
    # 条件触发式提示词
    # ---------------------------------------------

    conditional_prompt = """你是一个智能对话协调者，根据条件选择下一个发言者。

【当前对话上下文】
{context}

【可用Agent及其专长】
{agents}

【条件触发规则】
1. 关键词触发：
   - 包含"数据分析"、"统计"、"可视化" → 选择"数据分析师"
   - 包含"机器学习"、"模型"、"训练" → 选择"机器学习工程师"
   - 包含"部署"、"运维"、"监控" → 选择"运维工程师"

2. 阶段触发：
   - 讨论开始阶段 → 优先"通用助手"介绍问题
   - 具体问题讨论 → 选择对应专家
   - 总结阶段 → 通用助手可以总结

3. 均衡触发：
   - 跟踪每个Agent的发言次数
   - 优先选择发言较少的Agent
   - 避免同一Agent连续发言

【输出格式】
只输出Agent名字，如：数据分析师
"""

    print("\n条件触发式提示词特点:")
    print("  1. 明确的关键词对应关系")
    print("  2. 阶段敏感的触发机制")
    print("  3. 发言均衡控制")

    groupchat = GroupChat(
        agents=[data_analyst, ml_engineer, devops_engineer, general_assistant],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        speaker_selection_prompt=conditional_prompt,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 演示条件触发效果
    print("\n执行群聊 - 场景：讨论数据科学项目")

    conversation_topic = """我们有一个数据科学项目需要讨论：
1. 首先分析数据特征
2. 然后训练机器学习模型
3. 最后部署到生产环境
请各位专家依次发表意见。"""

    result = general_assistant.initiate_chat(
        manager,
        message=conversation_topic,
    )

    print(f"\n总消息数: {len(result.chat_history)}")

    # 分析发言模式
    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")

    # 统计
    from collections import Counter
    counts = Counter(speakers)
    print("\n各Agent发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")


def demo_advanced_prompt_techniques():
    """
    演示高级提示词技术

    高级技术包括：
    1. 链式思考（Chain of Thought）- 让LLM解释选择理由
    2.few-shot示例 - 提供选择示例
    3. 动态上下文 - 根据对话状态调整提示词
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: 高级提示词技术")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建专业团队
    # ---------------------------------------------

    researcher = ConversableAgent(
        name="研究员",
        system_message="你是研究员，负责进行深入研究和分析。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    analyst = ConversableAgent(
        name="分析师",
        system_message="你是分析师，负责分析数据并提供见解。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    consultant = ConversableAgent(
        name="顾问",
        system_message="你是顾问，负责提供专业建议和解决方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: 研究员, 分析师, 顾问")

    # ---------------------------------------------
    # 高级提示词：包含few-shot示例
    # ---------------------------------------------

    advanced_prompt = """你是一个智能对话协调者，需要选择下一个发言者。

【对话上下文】
{context}

【可用Agent】
{agents}

【Chain of Thought 选择过程】
请按以下步骤思考：
1. 分析当前讨论的主题和阶段
2. 确定需要什么类型的专业知识
3. 检查最近发言的Agent，避免重复
4. 选择最适合的Agent

【Few-shot 示例】
示例1：
上下文：讨论项目进度，需要了解当前状态
选择：分析师（适合提供数据概览）

示例2：
上下文：需要深入研究某个技术问题
选择：研究员（适合深入分析）

示例3：
上下文：需要制定下一步行动计划
选择：顾问（适合提供建议）

【输出要求】
首先输出选择理由（简短一行），
然后输出Agent名字。
格式：
理由：...
选择：研究员
"""

    print("\n高级提示词技术:")
    print("  1. Chain of Thought - 明确选择推理过程")
    print("  2. Few-shot 示例 - 提供选择参考")
    print("  3. 分离理由和结果 - 便于调试")

    groupchat = GroupChat(
        agents=[researcher, analyst, consultant],
        messages=[],
        max_round=9,
        speaker_selection_method="auto",
        speaker_selection_prompt=advanced_prompt,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 演示
    print("\n执行群聊:")
    result = researcher.initiate_chat(
        manager,
        message="我们来讨论一下公司的战略规划问题。",
    )

    print(f"\n总消息数: {len(result.chat_history)}")

    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")


def demo_prompt_troubleshooting():
    """
    演示提示词优化中的常见问题和解决方案

    常见问题：
    1. 选择结果不稳定 - 提示词太模糊
    2. 某些Agent从未被选中 - 提示词偏向特定Agent
    3. 选择逻辑不合理 - 提示词逻辑有矛盾
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示6: 提示词优化常见问题与解决")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建测试Agent
    agent_a = ConversableAgent(
        name="Agent_A",
        system_message="你是Agent_A，擅长处理A类任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="Agent_B",
        system_message="你是Agent_B，擅长处理B类任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_c = ConversableAgent(
        name="Agent_C",
        system_message="你是Agent_C，擅长处理C类任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个Agent: Agent_A, Agent_B, Agent_C")

    # ---------------------------------------------
    # 问题1: 提示词太模糊导致选择不稳定
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("问题1: 提示词太模糊")
    print("-" * 40)

    vague_prompt = """选择下一个发言者。
可用：Agent_A, Agent_B, Agent_C
根据感觉选择。"""

    print("\n模糊提示词示例:")
    print(f"  '{vague_prompt}'")
    print("\n问题: LLM可能每次选择不同的Agent，结果不稳定")

    # ---------------------------------------------
    # 解决方案1: 明确的选择标准
    # ---------------------------------------------

    print("\n解决方案1: 明确的选择标准")

    clear_prompt = """你是一个严格的对话协调者。

【规则】
1. Agent_A 处理初始任务
2. Agent_B 处理验证
3. Agent_C 处理总结
4. 按照上述顺序轮流选择

【当前状态】
{context}

请严格按顺序选择一个Agent。
输出格式：Agent_A | Agent_B | Agent_C
"""

    print("明确的提示词:")
    print("  - 定义清晰的角色分工")
    print("  - 指定严格的顺序规则")
    print("  - 明确的输出格式")

    # ---------------------------------------------
    # 问题2: 提示词偏向某些Agent
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("问题2: 提示词偏向某些Agent")
    print("-" * 40)

    biased_prompt = """优先选择Agent_A，因为它最重要。
只有Agent_A无法处理时才选择其他Agent。
可用：Agent_A, Agent_B, Agent_C"""

    print("\n有偏提示词示例:")
    print(f"  '{biased_prompt}'")
    print("\n问题: Agent_B和Agent_C几乎不会被选中")

    # ---------------------------------------------
    # 解决方案2: 均衡加权
    # ---------------------------------------------

    print("\n解决方案2: 均衡加权")

    balanced_prompt = """你需要公平地选择下一个发言者。

【公平选择规则】
1. 评估每个Agent与当前任务的匹配度
2. 考虑发言均衡，优先选择发言较少的Agent
3. 除非明确需要，不连续选择同一Agent

【匹配度评估】
- Agent_A: 适合初始分析和规划
- Agent_B: 适合验证和测试
- Agent_C: 适合总结和归档

【当前状态】
{context}

请公平评估并选择。
输出格式：Agent_A | Agent_B | Agent_C
"""

    print("均衡提示词:")
    print("  - 强调公平选择")
    print("  - 考虑匹配度和均衡")
    print("  - 避免机械顺序")

    # ---------------------------------------------
    # 问题3: 逻辑矛盾
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("问题3: 逻辑矛盾")
    print("-" * 40)

    contradictory_prompt = """选择下一个发言者。

规则1：总是选择发言最少的Agent
规则2：总是选择Agent_A（因为它最重要）
规则3：从不连续选择同一Agent

这三条规则可能相互矛盾！
"""

    print("矛盾提示词示例:")
    print(f"  '{contradictory_prompt}'")
    print("\n问题: LLM无法同时满足所有规则，导致不可预测结果")

    # ---------------------------------------------
    # 解决方案3: 优先级排序
    # ---------------------------------------------

    print("\n解决方案3: 优先级排序")

    prioritized_prompt = """你是一个严格的对话协调者。

【优先级规则】（按顺序应用）
1. 如果需要终止对话，选择说出"完成"的Agent
2. 如果某Agent连续发言超过2次，跳过它选择其他
3. 如果所有Agent发言次数相同，按A->B->C顺序选择
4. 其他情况根据上下文选择最合适的

【当前状态】
{context}

【发言统计】
{agent_selections}

请按优先级规则选择。
输出格式：Agent_A | Agent_B | Agent_C
"""

    print("有优先级的提示词:")
    print("  - 明确定义规则优先级")
    print("  - 避免规则冲突")
    print("  - 提供决策框架")

    # 实际演示
    print("\n" + "-" * 40)
    print("实际演示：使用均衡提示词")
    print("-" * 40)

    groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        messages=[],
        max_round=9,
        speaker_selection_method="auto",
        speaker_selection_prompt=balanced_prompt,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    result = agent_a.initiate_chat(
        manager,
        message="请讨论：如何提高团队效率？",
    )

    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")

    from collections import Counter
    counts = Counter(speakers)
    print("\n发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")

    if max(counts.values()) - min(counts.values()) <= 2:
        print("验证通过: 发言分布相对均衡")
    else:
        print("提示: 可以进一步调整提示词以优化均衡")


def demo_production_ready_prompt():
    """
    演示生产级别的提示词模板

    提供一个完整的、可用于生产的提示词模板，
    包含所有最佳实践。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示7: 生产级别的提示词模板")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建专业团队
    business_analyst = ConversableAgent(
        name="业务分析师",
        system_message="""你是业务分析师。
专长：理解业务需求、流程优化、需求分析。
当讨论涉及业务需求时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    technical_lead = ConversableAgent(
        name="技术负责人",
        system_message="""你是技术负责人。
专长：架构设计、技术决策、代码审查。
当讨论涉及技术方案时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    project_manager = ConversableAgent(
        name="项目经理",
        system_message="""你是项目经理。
专长：进度跟踪、风险管理、沟通协调。
当讨论涉及项目进度时，你应该被选中。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个专业角色:")
    print("  - 业务分析师: 处理业务需求相关讨论")
    print("  - 技术负责人: 处理技术方案相关讨论")
    print("  - 项目经理: 处理项目进度相关讨论")

    # ---------------------------------------------
    # 生产级别提示词模板
    # ---------------------------------------------

    production_prompt = """你是群聊协调者，负责选择下一个最合适的发言者。

【当前对话状态】
{context}

【可用Agent及其专长】
{agents}

【选择策略 - 严格按此顺序执行】

第一步：检查终止条件
- 如果对话已经完成或用户要求结束，选择"完成"
- 当前Agent数量: {agent_count}
- 当前轮次: {current_round}

第二步：检查关键词触发
- 包含"需求"、"业务"、"流程" → 选择"业务分析师"
- 包含"技术"、"架构"、"代码" → 选择"技术负责人"
- 包含"进度"、"计划"、"管理" → 选择"项目经理"

第三步：检查发言均衡
- 统计当前各Agent发言次数: {agent_selections}
- 优先选择发言较少的Agent
- 除非必要，不连续选择同一Agent

第四步：基于上下文选择
- 讨论刚开始 → 选择能介绍问题的Agent
- 讨论进行中 → 根据上下文选择相关Agent
- 讨论结束前 → 可以选择能总结的Agent

【输出格式】
只输出Agent名字，不要其他内容。
例如：业务分析师
"""

    print("\n生产级别提示词特点:")
    print("  1. 完整的状态信息（轮次、发言统计）")
    print("  2. 明确的优先级顺序")
    print("  3. 清晰的输出格式")
    print("  4. 处理边界情况")

    groupchat = GroupChat(
        agents=[business_analyst, technical_lead, project_manager],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        speaker_selection_prompt=production_prompt,
    )

    # 注入动态变量
    groupchat._agent_selection_prompt = production_prompt

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 演示
    print("\n执行群聊:")
    result = business_analyst.initiate_chat(
        manager,
        message="我们讨论一个新项目的启动：业务部门需要一个新的客户管理系统。",
    )

    print(f"\n总消息数: {len(result.chat_history)}")

    speakers = []
    for msg in result.chat_history:
        if msg.get("role") == "assistant":
            speakers.append(msg.get("name"))

    print(f"发言顺序: {' -> '.join(speakers)}")

    from collections import Counter
    counts = Counter(speakers)
    print("\n发言统计:")
    for name, count in counts.items():
        print(f"  - {name}: {count}次")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有提示词优化演示
    """
    print("=" * 60)
    print("speaker_selection_mode - Prompt优化演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_default_speaker_prompt()
    demo_custom_speaker_prompt()
    demo_role_based_prompt_optimization()
    demo_conditional_trigger_prompt()
    demo_advanced_prompt_techniques()
    demo_prompt_troubleshooting()
    demo_production_ready_prompt()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()