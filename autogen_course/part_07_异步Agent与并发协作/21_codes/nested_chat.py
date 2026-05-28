# nested_chat.py
# 第21节 嵌套对话与并发协作机制 - 嵌套对话配置演示
#
# 本文件演示嵌套对话（Nested Chat）的配置与使用，包括：
# 1. 嵌套对话的基本配置
# 2. sender属性与消息溯源机制
# 3. 嵌套GroupChat中的异步消息传递
# 4. 嵌套对话的层级管理与终止条件
#
# ============================================================
# 嵌套对话（Nested Chat）核心概念
# ============================================================
#
# 嵌套对话是一种在对话中启动另一个对话的机制，常见于：
# 1. 子任务的委托处理：当主Agent需要处理复杂任务时，可以委托子Agent处理
# 2. 专家咨询场景：一个Agent向专家群聊咨询获取专业意见
# 3. 分层协作：不同层级的Agent处理不同抽象级别的任务
#
# 关键特性：
# - sender属性：标识消息的发送者，用于消息溯源
# - 消息溯源：通过sender追踪消息的发起者，构建完整的对话链
# - 异步传递：嵌套对话中的消息异步传递给子Agent
#
# ============================================================

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

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
# 第三部分：嵌套对话基础配置
# ============================================================

def demo_nested_chat_basic():
    """
    演示最基本的嵌套对话配置

    嵌套对话的基本结构：
    1. 用户发起主对话
    2. 主Agent在处理过程中启动子对话
    3. 子对话完成后，结果返回给主Agent
    4. 主Agent继续处理并返回最终结果

    关键配置：
    - is_termination_msg: 控制何时终止对话
    - max_round: 控制最大对话轮次
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: 嵌套对话基本配置")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 步骤1：创建主Agent（协调者）
    # ---------------------------------------------

    coordinator = ConversableAgent(
        name="协调者",
        system_message="""你是项目协调者。
当需要专业帮助时，你会将任务委托给专家群聊。
完成所有咨询后，汇总结果并给出最终建议。
如果任务完成，说 'FINAL_RESULT' 来结束对话。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建协调者Agent")
    print(f"  - 名称: {coordinator.name}")
    print(f"  - human_input_mode: {coordinator.human_input_mode}")

    # ---------------------------------------------
    # 步骤2：创建子群聊（专家团队）
    # ---------------------------------------------

    # 专家Agent
    tech_expert = ConversableAgent(
        name="技术专家",
        system_message="你是一名技术专家，提供技术方面的专业建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    business_expert = ConversableAgent(
        name="业务专家",
        system_message="你是一名业务专家，提供业务方面的专业建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 创建子群聊
    expert_groupchat = GroupChat(
        agents=[tech_expert, business_expert],
        messages=[],
        max_round=6,  # 子群聊的最大轮次
    )

    expert_manager = GroupChatManager(
        groupchat=expert_groupchat,
        llm_config=llm_config,
    )

    print("\n已创建专家群聊:")
    print(f"  - 参与Agent: {[a.name for a in expert_groupchat.agents]}")
    print(f"  - 最大轮次: {expert_groupchat.max_round}")

    # ---------------------------------------------
    # 步骤3：创建父群聊（包含协调者和专家管理器）
    # ---------------------------------------------

    parent_groupchat = GroupChat(
        agents=[coordinator, expert_manager],
        messages=[],
        max_round=10,  # 父群聊的最大轮次
    )

    parent_manager = GroupChatManager(
        groupchat=parent_groupchat,
        llm_config=llm_config,
    )

    print("\n已创建父群聊:")
    print(f"  - 参与Agent: {[a.name for a in parent_groupchat.agents]}")
    print(f"  - 最大轮次: {parent_groupchat.max_round}")

    # ---------------------------------------------
    # 步骤4：启动嵌套对话
    # ---------------------------------------------

    print("\n启动嵌套对话...")
    print("  - 协调者收到用户请求")
    print("  - 协调者委托专家群聊咨询")
    print("  - 专家群聊讨论并给出建议")
    print("  - 协调者汇总结果")

    result = coordinator.initiate_chat(
        parent_manager,
        message="我们需要开发一个新的移动应用。请咨询技术专家和业务专家的意见。",
    )

    print("\n嵌套对话完成")
    print(f"  - 总消息数: {len(result.chat_history)}")


# ============================================================
# 第四部分：sender属性与消息溯源机制
# ============================================================

def demo_sender_traceability():
    """
    演示sender属性与消息溯源机制

    消息溯源的核心要素：
    1. sender属性：每个消息都包含sender字段，标识发送者
    2. 消息追溯：通过sender追踪消息的发起者
    3. 对话链构建：构建完整的对话链路图

    在嵌套对话中：
    - 顶层sender是发起者
    - 子层sender是执行者
    - 通过sender可以追踪消息的完整路径
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: sender属性与消息溯源机制")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    manager_agent = ConversableAgent(
        name="经理",
        system_message="你是经理，向员工分配任务并汇总结果。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    worker_agent = ConversableAgent(
        name="员工",
        system_message="你是员工，完成经理分配的任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 创建群聊
    groupchat = GroupChat(
        agents=[manager_agent, worker_agent],
        messages=[],
        max_round=5,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 启动对话并分析消息结构
    # ---------------------------------------------

    print("\n启动对话...")
    result = manager_agent.initiate_chat(
        manager,
        message="请完成这份报告的撰写。",
    )

    # ---------------------------------------------
    # 分析消息结构中的sender
    # ---------------------------------------------

    print("\n消息结构分析:")
    print("-" * 40)

    for i, msg in enumerate(result.chat_history):
        # 提取消息的元数据
        role = msg.get("role", "unknown")
        sender = msg.get("sender", None)
        content_preview = msg.get("content", "")[:50]

        sender_name = sender.name if sender else "Unknown"

        print(f"[{i}] role={role}, sender={sender_name}")
        print(f"    content: {content_preview}...")

    print("-" * 40)
    print("\n溯源机制说明:")
    print("  1. 每条消息都包含sender字段，标识发送者Agent")
    print("  2. 通过sender可以追踪消息的发起者")
    print("  3. 在嵌套对话中，sender标识消息的实际发送者")
    print("  4. GroupChatManager的sender是群聊本身，不是实际发言的Agent")


def demo_nested_sender_chain():
    """
    演示嵌套对话中的sender链

    嵌套对话中的消息传递链：
    1. 用户 -> 主Agent（sender=用户）
    2. 主Agent -> 子群聊管理器（sender=主Agent）
    3. 子Agent在子群聊中发言（sender=子Agent）
    4. 子群聊回复 -> 子群聊管理器 -> 主Agent

    sender链的作用：
    - 追踪消息的完整路径
    - 分析对话的发起者
    - 理解信息的传递方向
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: 嵌套对话中的sender链")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建层级Agent
    # ---------------------------------------------

    # 顶层Agent
    director = ConversableAgent(
        name="总监",
        system_message="你是总监，协调各部门完成项目。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 中层Agent
    manager = ConversableAgent(
        name="经理",
        system_message="你是经理，接受总监指示并管理团队。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 底层Agent
    developer = ConversableAgent(
        name="开发人员",
        system_message="你是开发人员，执行具体的开发任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 构建嵌套结构
    # ---------------------------------------------

    # 底层群聊
    dev_groupchat = GroupChat(
        agents=[manager, developer],  # 经理和开发人员
        messages=[],
        max_round=4,
    )
    dev_manager = GroupChatManager(
        groupchat=dev_groupchat,
        llm_config=llm_config,
    )

    # 顶层群聊
    top_groupchat = GroupChat(
        agents=[director, dev_manager],  # 总监和底层群聊管理器
        messages=[],
        max_round=6,
    )
    top_manager = GroupChatManager(
        groupchat=top_groupchat,
        llm_config=llm_config,
    )

    print("\n嵌套结构:")
    print("  顶层: 总监")
    print("    └── 底层群聊管理器")
    print("          ├── 经理")
    print("          └── 开发人员")

    # ---------------------------------------------
    # 启动嵌套对话
    # ---------------------------------------------

    print("\n启动嵌套对话...")
    result = director.initiate_chat(
        top_manager,
        message="请完成这个功能模块的开发。",
    )

    # ---------------------------------------------
    # 分析sender链
    # ---------------------------------------------

    print("\nSender链分析:")
    print("-" * 40)

    for i, msg in enumerate(result.chat_history):
        sender = msg.get("sender", None)
        sender_name = sender.name if sender else "Unknown"
        content_preview = msg.get("content", "")[:40]

        print(f"[{i}] sender={sender_name}: {content_preview}...")

    print("-" * 40)

    print("\nSender链说明:")
    print("  - 消息从sender流向接收者")
    print("  - 在嵌套对话中，消息经过多个层级")
    print("  - sender标识消息的实际发起者")
    print("  - 通过分析sender链可以理解对话结构")


# ============================================================
# 第五部分：异步消息传递
# ============================================================

def demo_async_message_passsing():
    """
    演示嵌套GroupChat中的异步消息传递

    异步消息传递的特点：
    1. 非阻塞：发送消息后不等待立即响应
    2. 并发处理：多个消息可以同时处理
    3. 消息队列：消息进入队列，异步处理

    在AutoGen中：
    - initiate_chat是异步的，可以通过await等待
    -GroupChat内部使用asyncio处理消息传递
    - 消息传递是非阻塞的
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: 异步消息传递机制")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    agent_a = ConversableAgent(
        name="Agent_A",
        system_message="你是Agent_A，负责发起异步任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="Agent_B",
        system_message="你是Agent_B，负责处理异步任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_c = ConversableAgent(
        name="Agent_C",
        system_message="你是Agent_C，负责处理异步任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建群聊
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[agent_a, agent_b, agent_c],
        messages=[],
        max_round=8,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print("\n异步消息传递特点:")
    print("  1. 消息发送后立即返回，不阻塞等待响应")
    print("  2. GroupChat内部使用asyncio处理消息队列")
    print("  3. 多个Agent可以并发接收和处理消息")
    print("  4. 消息传递顺序由GroupChat管理器决定")

    # ---------------------------------------------
    # 启动异步对话
    # ---------------------------------------------

    print("\n启动异步对话...")
    result = agent_a.initiate_chat(
        manager,
        message="我们需要同时处理三个独立任务。",
    )

    print(f"\n异步对话完成: {len(result.chat_history)} 条消息")


def demo_nested_async_chat():
    """
    演示嵌套异步对话

    嵌套异步对话的特点：
    1. 外层对话：主Agent与主群聊的对话
    2. 内层对话：子Agent与子群聊的对话
    3. 异步传递：内层对话结果异步返回给外层

    使用场景：
    - 需要同时咨询多个专家团队
    - 子任务可以并行处理
    - 需要分层决策的场景
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: 嵌套异步对话")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建主Agent
    # ---------------------------------------------

    main_agent = ConversableAgent(
        name="主Agent",
        system_message="你是主Agent，协调多个子群聊完成任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建子群聊A
    # ---------------------------------------------

    expert_a1 = ConversableAgent(
        name="专家A1",
        system_message="你是专家A1，提供A领域的专业意见。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    expert_a2 = ConversableAgent(
        name="专家A2",
        system_message="你是专家A2，提供A领域的专业意见。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    groupchat_a = GroupChat(
        agents=[expert_a1, expert_a2],
        messages=[],
        max_round=4,
    )
    manager_a = GroupChatManager(
        groupchat=groupchat_a,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 创建子群聊B
    # ---------------------------------------------

    expert_b1 = ConversableAgent(
        name="专家B1",
        system_message="你是专家B1，提供B领域的专业意见。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    expert_b2 = ConversableAgent(
        name="专家B2",
        system_message="你是专家B2，提供B领域的专业意见。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    groupchat_b = GroupChat(
        agents=[expert_b1, expert_b2],
        messages=[],
        max_round=4,
    )
    manager_b = GroupChatManager(
        groupchat=groupchat_b,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 创建主群聊
    # ---------------------------------------------

    main_groupchat = GroupChat(
        agents=[main_agent, manager_a, manager_b],
        messages=[],
        max_round=10,
    )

    main_manager = GroupChatManager(
        groupchat=main_groupchat,
        llm_config=llm_config,
    )

    print("\n嵌套异步结构:")
    print("  主Agent")
    print("    ├── 子群聊A (专家A1, 专家A2)")
    print("    └── 子群聊B (专家B1, 专家B2)")

    # ---------------------------------------------
    # 启动嵌套异步对话
    # ---------------------------------------------

    print("\n启动嵌套异步对话...")
    result = main_agent.initiate_chat(
        main_manager,
        message="请分别咨询A领域和B领域的专家意见。",
    )

    print(f"\n嵌套异步对话完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 子群聊A消息数: {len(groupchat_a.messages)}")
    print(f"  - 子群聊B消息数: {len(groupchat_b.messages)}")


# ============================================================
# 第六部分：嵌套对话终止条件
# ============================================================

def demo_nested_termination():
    """
    演示嵌套对话的终止条件

    嵌套对话终止条件类型：
    1. max_round：达到最大轮次后自动终止
    2. 消息内容终止：通过is_termination_msg判断
    3. 显式终止：Agent发送特定消息终止对话

    层级终止策略：
    - 子群聊先终止：子任务完成
    - 父群聊后终止：汇总子结果后结束
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示6: 嵌套对话终止条件")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    orchestrator = ConversableAgent(
        name="编排器",
        system_message="""你是任务编排器。
你将任务委托给子群聊，然后汇总子结果。
当所有子任务完成后，说 'ALL_TASKS_COMPLETE' 结束。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    worker1 = ConversableAgent(
        name="工作者1",
        system_message="你是工作者1，完成分配的任务后说 'TASK_1_DONE'。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    worker2 = ConversableAgent(
        name="工作者2",
        system_message="你是工作者2，完成分配的任务后说 'TASK_2_DONE'。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建子群聊
    # ---------------------------------------------

    worker_groupchat = GroupChat(
        agents=[orchestrator, worker1, worker2],
        messages=[],
        max_round=6,
    )

    worker_manager = GroupChatManager(
        groupchat=worker_groupchat,
        llm_config=llm_config,
    )

    print("\n终止条件配置:")
    print("  - max_round: 6")
    print("  - 消息内容终止: 检测 'ALL_TASKS_COMPLETE'")

    # ---------------------------------------------
    # 启动对话并观察终止
    # ---------------------------------------------

    print("\n启动对话...")
    result = worker1.initiate_chat(
        worker_manager,
        message="请完成两个并行任务。",
    )

    print(f"\n对话终止:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 终止原因: 达到max_round或检测到终止消息")


def demo_three_level_nesting():
    """
    演示三级嵌套结构

    三级嵌套结构：
    1. 第一级（战略层）：高层决策和任务分配
    2. 第二级（战术层）：协调中层管理和资源
    3. 第三级（执行层）：具体任务执行

    消息流向：
    - 向下：任务分配
    - 向上：结果汇报
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示7: 三级嵌套结构")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 第一级：战略层
    # ---------------------------------------------

    ceo = ConversableAgent(
        name="CEO",
        system_message="你是CEO，负责公司战略决策。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 第二级：战术层
    # ---------------------------------------------

    vp_engineering = ConversableAgent(
        name="工程副总裁",
        system_message="你是工程副总裁，负责技术团队管理。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    vp_sales = ConversableAgent(
        name="销售副总裁",
        system_message="你是销售副总裁，负责销售团队管理。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 第三级：执行层
    # ---------------------------------------------

    senior_dev = ConversableAgent(
        name="高级工程师",
        system_message="你是高级工程师，负责技术实现。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    junior_dev = ConversableAgent(
        name="初级工程师",
        system_message="你是初级工程师，协助高级工程师工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    sales_rep = ConversableAgent(
        name="销售代表",
        system_message="你是销售代表，负责客户对接。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 构建第三级群聊（工程团队）
    # ---------------------------------------------

    eng_groupchat = GroupChat(
        agents=[vp_engineering, senior_dev, junior_dev],
        messages=[],
        max_round=4,
    )
    eng_manager = GroupChatManager(
        groupchat=eng_groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 构建第三级群聊（销售团队）
    # ---------------------------------------------

    sales_groupchat = GroupChat(
        agents=[vp_sales, sales_rep],
        messages=[],
        max_round=4,
    )
    sales_manager = GroupChatManager(
        groupchat=sales_groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 构建第二级群聊
    # ---------------------------------------------

    director_groupchat = GroupChat(
        agents=[ceo, eng_manager, sales_manager],
        messages=[],
        max_round=8,
    )
    director_manager = GroupChatManager(
        groupchat=director_groupchat,
        llm_config=llm_config,
    )

    print("\n三级嵌套结构:")
    print("  第1级(CEO)")
    print("    ├── 第2级(工程VP) -> [高级工程师, 初级工程师]")
    print("    └── 第2级(销售VP) -> [销售代表]")

    # ---------------------------------------------
    # 启动三级嵌套对话
    # ---------------------------------------------

    print("\n启动三级嵌套对话...")
    result = ceo.initiate_chat(
        director_manager,
        message="请执行本季度的技术开发和销售目标。",
    )

    print(f"\n三级嵌套完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 工程团队消息数: {len(eng_groupchat.messages)}")
    print(f"  - 销售团队消息数: {len(sales_groupchat.messages)}")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：运行所有嵌套对话演示
    """
    print("=" * 60)
    print("嵌套对话与并发协作机制 - 嵌套对话配置演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_nested_chat_basic()
    demo_sender_traceability()
    demo_nested_sender_chain()
    demo_async_message_passsing()
    demo_nested_async_chat()
    demo_nested_termination()
    demo_three_level_nesting()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
