# groupchat_nested.py
# 第9节 GroupChat与多Agent协作模式 - 嵌套GroupChat演示
#
# 本文件演示嵌套GroupChat（nested group chat）的用法，包括：
# 1. 子群聊的创建与配置
# 2. 层级管理机制
# 3. 嵌套对话的终止条件
# 4. 跨群聊消息传递
#
# ============================================================
# 嵌套 GroupChat 核心概念
# ============================================================
#
# 嵌套GroupChat是一种多层级协作模式，它允许：
# 1. 在父群聊中创建子群聊
# 2. 子群聊独立运行，处理特定子任务
# 3. 子群聊完成后，结果返回给父群聊
# 4. 父群聊基于子群聊结果继续协调
#
# 典型应用场景：
# - 大型复杂任务分解：父群聊协调，子群聊分别处理子任务
# - 专家团队协作：不同子群聊包含不同领域的专家
# - 并行处理：多个子群聊同时处理独立任务
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
# 第三部分：嵌套GroupChat基本模式
# ============================================================

def demo_nested_groupchat_basic():
    """
    演示最基本的嵌套GroupChat模式

    场景：一个项目协调者创建两个子群聊，分别处理前端和后端任务

    层级结构：
    - 父群聊（项目协调）：协调整个项目
      - 子群聊A（前端组）：处理前端任务
      - 子群聊B（后端组）：处理后端任务
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示1: 嵌套GroupChat基本模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 步骤1：创建各层级的Agent
    # ---------------------------------------------

    # 父群聊Agent：项目协调者
    coordinator = ConversableAgent(
        name="项目协调者",
        system_message="你是项目协调者，负责协调前端和后端团队完成项目。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 子群聊A（前端组）Agent
    frontend_lead = ConversableAgent(
        name="前端组长",
        system_message="你是前端组长，负责协调前端开发工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    frontend_dev = ConversableAgent(
        name="前端开发",
        system_message="你是前端开发工程师，负责实现UI组件。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 子群聊B（后端组）Agent
    backend_lead = ConversableAgent(
        name="后端组长",
        system_message="你是后端组长，负责协调后端开发工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    backend_dev = ConversableAgent(
        name="后端开发",
        system_message="你是后端开发工程师，负责实现API服务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建Agent:")
    print("  父群聊: 项目协调者")
    print("  子群聊A(前端组): 前端组长, 前端开发")
    print("  子群聊B(后端组): 后端组长, 后端开发")

    # ---------------------------------------------
    # 步骤2：创建子群聊
    # ---------------------------------------------

    # 前端子群聊
    frontend_groupchat = GroupChat(
        agents=[frontend_lead, frontend_dev],
        messages=[],
        max_round=5,
    )

    frontend_manager = GroupChatManager(
        groupchat=frontend_groupchat,
        llm_config=llm_config,
        name="前端群聊管理器",
    )

    print("\n已创建前端子群聊:")
    print(f"  - 参与Agent: {[a.name for a in frontend_groupchat.agents]}")
    print(f"  - 最大轮次: {frontend_groupchat.max_round}")

    # 后端子群聊
    backend_groupchat = GroupChat(
        agents=[backend_lead, backend_dev],
        messages=[],
        max_round=5,
    )

    backend_manager = GroupChatManager(
        groupchat=backend_groupchat,
        llm_config=llm_config,
        name="后端群聊管理器",
    )

    print("\n已创建后端子群聊:")
    print(f"  - 参与Agent: {[a.name for a in backend_groupchat.agents]}")
    print(f"  - 最大轮次: {backend_groupchat.max_round}")

    # ---------------------------------------------
    # 步骤3：创建父群聊（包含子群聊管理器）
    # ---------------------------------------------

    # 父群聊包含协调者和两个子群聊管理器
    parent_groupchat = GroupChat(
        agents=[coordinator, frontend_manager, backend_manager],
        messages=[],
        max_round=10,
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
    print("-" * 40)

    result = coordinator.initiate_chat(
        parent_manager,
        message="我们需要开发一个电商网站。请前端组负责UI开发，后端组负责API开发。",
    )

    print("-" * 40)
    print("嵌套对话完成")
    print(f"总消息数: {len(result.chat_history)}")


def demo_sequential_subgroups():
    """
    演示顺序执行的子群聊模式

    场景：父群聊协调下，多个子群聊按顺序执行任务
    子群聊A完成后，子群聊B才开始

    这种模式适用于：
    - 任务有先后依赖关系
    - 需要先完成一个任务再做下一个
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: 顺序执行的子群聊模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建子群聊Agent
    # ---------------------------------------------

    # 调研组
    researcher = ConversableAgent(
        name="调研员",
        system_message="你是调研员，负责收集和分析信息。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    analyst = ConversableAgent(
        name="分析师",
        system_message="你是分析师，负责解读调研数据。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 规划组
    planner = ConversableAgent(
        name="规划师",
        system_message="你是规划师，负责制定实施方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    executor = ConversableAgent(
        name="实施员",
        system_message="你是实施员，负责执行计划。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建子群聊
    # ---------------------------------------------

    # 调研子群聊
    research_groupchat = GroupChat(
        agents=[researcher, analyst],
        messages=[],
        max_round=4,
    )
    research_manager = GroupChatManager(
        groupchat=research_groupchat,
        llm_config=llm_config,
    )

    # 规划子群聊
    planning_groupchat = GroupChat(
        agents=[planner, executor],
        messages=[],
        max_round=4,
    )
    planning_manager = GroupChatManager(
        groupchat=planning_groupchat,
        llm_config=llm_config,
    )

    print("已创建两个子群聊:")
    print("  调研组: 调研员 + 分析师")
    print("  规划组: 规划师 + 实施员")

    # ---------------------------------------------
    # 创建父群聊
    # ---------------------------------------------

    manager_of_managers = ConversableAgent(
        name="总协调员",
        system_message="你是总协调员，负责协调调研组和规划组按顺序工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    parent_groupchat = GroupChat(
        agents=[manager_of_managers, research_manager, planning_manager],
        messages=[],
        max_round=10,
    )

    parent_manager = GroupChatManager(
        groupchat=parent_groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 执行顺序工作流程
    # ---------------------------------------------

    print("\n顺序工作流程:")
    print("  1. 总协调员指派任务给调研组")
    print("  2. 调研组完成工作（调研员+分析师）")
    print("  3. 总协调员汇总调研结果")
    print("  4. 总协调员指派任务给规划组")
    print("  5. 规划组完成工作（规划师+实施员）")
    print("  6. 总协调员汇总最终结果")

    print("\n执行顺序工作流程...")
    result = manager_of_managers.initiate_chat(
        parent_manager,
        message="请先进行市场调研，调研完成后制定实施计划。",
    )

    print(f"\n完成，总消息数: {len(result.chat_history)}")


def demo_parallel_subgroups():
    """
    演示并行执行的子群聊模式

    场景：多个子群聊同时工作，处理独立的任务
    适用于任务之间没有依赖，可以并行处理的情况

    注意：AutoGen的GroupChat本身是串行的，
    但可以通过模拟方式展示并行概念
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示3: 并行子群聊概念")
    print("=" * 60)

    print("\n说明: AutoGen的GroupChat本身是串行执行的，")
    print("但我们可以通过多个独立的GroupChat来模拟并行效果。")
    print("每个子群聊独立运行，处理各自的任务。")

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建三个独立工作的子群聊
    # ---------------------------------------------

    # 数据分析组
    data_lead = ConversableAgent(
        name="数据组长",
        system_message="你是数据组长，负责协调数据分析工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    data_dev = ConversableAgent(
        name="数据分析师",
        system_message="你是数据分析师，负责处理数据。",
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

    # UI设计组
    ui_lead = ConversableAgent(
        name="UI组长",
        system_message="你是UI组长，负责协调界面设计工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    ui_dev = ConversableAgent(
        name="UI设计师",
        system_message="你是UI设计师，负责设计界面。",
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

    # 基础架构组
    infra_lead = ConversableAgent(
        name="架构组长",
        system_message="你是架构组长，负责协调系统架构工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    infra_dev = ConversableAgent(
        name="架构工程师",
        system_message="你是架构工程师，负责设计系统架构。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    infra_groupchat = GroupChat(
        agents=[infra_lead, infra_dev],
        messages=[],
        max_round=4,
    )
    infra_manager = GroupChatManager(
        groupchat=infra_groupchat,
        llm_config=llm_config,
    )

    print("\n已创建三个并行子群聊:")
    print("  数据分析组: 数据组长 + 数据分析师")
    print("  UI设计组: UI组长 + UI设计师")
    print("  基础架构组: 架构组长 + 架构工程师")

    # ---------------------------------------------
    # 依次启动三个子群聊（模拟并行）
    # ---------------------------------------------

    print("\n模拟并行执行（依次启动各子群聊）...")

    # 数据分析组
    print("\n[数据组] 开始工作...")
    result_data = data_lead.initiate_chat(
        data_manager,
        message="请分析用户增长数据，并给出报告。",
    )
    print(f"[数据组] 完成，消息数: {len(result_data.chat_history)}")

    # UI设计组
    print("\n[UI组] 开始工作...")
    result_ui = ui_lead.initiate_chat(
        ui_manager,
        message="请设计新功能的用户界面。",
    )
    print(f"[UI组] 完成，消息数: {len(result_ui.chat_history)}")

    # 基础架构组
    print("\n[架构组] 开始工作...")
    result_infra = infra_lead.initiate_chat(
        infra_manager,
        message="请设计系统的技术架构。",
    )
    print(f"[架构组] 完成，消息数: {len(result_infra.chat_history)}")

    # ---------------------------------------------
    # 汇总结果
    # ---------------------------------------------

    print("\n" + "=" * 40)
    print("并行任务汇总:")
    print("=" * 40)
    print(f"  数据分析组完成 {len(result_data.chat_history)} 条消息")
    print(f"  UI设计组完成 {len(result_ui.chat_history)} 条消息")
    print(f"  基础架构组完成 {len(result_infra.chat_history)} 条消息")


def demo_nested_termination():
    """
    演示嵌套GroupChat的终止条件管理

    嵌套GroupChat的终止需要特别关注：
    1. 子群聊的终止条件
    2. 父群聊的终止条件
    3. 跨层级的终止信号传递

    常见模式：
    - 子群聊达到max_round自动终止
    - 父群聊需要根据子群聊结果决定是否继续
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示4: 嵌套GroupChat终止条件管理")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建Agent
    # ---------------------------------------------

    # 主协调者
    main_coordinator = ConversableAgent(
        name="主协调者",
        system_message="你是主协调者，负责协调子群聊完成复杂任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 子群聊Agent
    subtask_leader = ConversableAgent(
        name="子任务负责人",
        system_message="你是子任务负责人，负责领导团队完成任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    subtask_worker = ConversableAgent(
        name="子任务执行者",
        system_message="你是子任务执行者，负责具体执行任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 配置子群聊（较短轮次，快速终止）
    # ---------------------------------------------

    subtask_groupchat = GroupChat(
        agents=[subtask_leader, subtask_worker],
        messages=[],
        max_round=3,  # 子群聊轮次较少
    )

    subtask_manager = GroupChatManager(
        groupchat=subtask_groupchat,
        llm_config=llm_config,
    )

    print("\n子群聊配置:")
    print(f"  - max_round: {subtask_groupchat.max_round}")
    print("  - 预期行为: 3轮后自动终止")

    # ---------------------------------------------
    # 配置父群聊（较长轮次）
    # ---------------------------------------------

    parent_groupchat = GroupChat(
        agents=[main_coordinator, subtask_manager],
        messages=[],
        max_round=8,  # 父群聊轮次较多
    )

    parent_manager = GroupChatManager(
        groupchat=parent_groupchat,
        llm_config=llm_config,
    )

    print("\n父群聊配置:")
    print(f"  - max_round: {parent_groupchat.max_round}")
    print("  - 预期行为: 8轮后或子群聊完成后终止")

    # ---------------------------------------------
    # 执行并观察终止行为
    # ---------------------------------------------

    print("\n执行嵌套对话...")
    result = main_coordinator.initiate_chat(
        parent_manager,
        message="请子群聊完成数据处理任务。",
    )

    print(f"\n对话结束:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 子群聊实际轮次: {len(subtask_groupchat.messages)}")


def demo_cross_level_communication():
    """
    演示跨层级通信

    嵌套GroupChat中，消息传递规则：
    1. 父群聊的消息会传递给选定的Agent（可能是子群聊管理器）
    2. 子群聊管理器收到消息后，会在子群聊中继续传递
    3. 子群聊的回复会返回给父群聊

    重要：Agent不能直接跨群聊通信，必须通过管理器转发
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: 跨层级通信机制")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建父层级Agent
    # ---------------------------------------------

    top_level_manager = ConversableAgent(
        name="高层管理者",
        system_message="你是高层管理者，向下传达战略决策。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建中间层级Agent
    # ---------------------------------------------

    middle_level_manager = ConversableAgent(
        name="中层管理者",
        system_message="你是中层管理者，接收高层决策并传达给基层。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建基层Agent
    # ---------------------------------------------

    ground_level_worker1 = ConversableAgent(
        name="基层员工A",
        system_message="你是基层员工A，执行具体任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    ground_level_worker2 = ConversableAgent(
        name="基层员工B",
        system_message="你是基层员工B，执行具体任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 构建三层嵌套结构
    # ---------------------------------------------

    # 基层群聊
    ground_groupchat = GroupChat(
        agents=[ground_level_worker1, ground_level_worker2],
        messages=[],
        max_round=5,
    )
    ground_manager = GroupChatManager(
        groupchat=ground_groupchat,
        llm_config=llm_config,
    )

    # 中层管理的群聊（包含基层管理器）
    middle_groupchat = GroupChat(
        agents=[middle_level_manager, ground_manager],
        messages=[],
        max_round=6,
    )
    middle_manager = GroupChatManager(
        groupchat=middle_groupchat,
        llm_config=llm_config,
    )

    # 高层管理的群聊（包含中层管理器）
    top_groupchat = GroupChat(
        agents=[top_level_manager, middle_manager],
        messages=[],
        max_round=7,
    )
    top_manager = GroupChatManager(
        groupchat=top_groupchat,
        llm_config=llm_config,
    )

    print("三层嵌套结构:")
    print("  第1层(高层): 高层管理者")
    print("    └── 第2层(中层): 中层管理者")
    print("          └── 第3层(基层): 基层员工A, 基层员工B")

    # ---------------------------------------------
    # 执行跨层级通信
    # ---------------------------------------------

    print("\n执行跨层级通信...")
    print("消息流向: 高层 -> 中层 -> 基层 -> 逐层返回")

    result = top_level_manager.initiate_chat(
        top_manager,
        message="请执行今年的运营计划。",
    )

    print(f"\n通信完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 基层群聊消息数: {len(ground_groupchat.messages)}")


def demo_hierarchical_task_decomposition():
    """
    演示层级任务分解

    这是一个典型的嵌套GroupChat应用场景：
    1. 顶层：任务分解与协调
    2. 中间层：子任务执行
    3. 底层：具体操作

    示例场景：开发一个电商网站
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示6: 层级任务分解 - 电商网站开发")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 第一层：项目总负责人
    # ---------------------------------------------

    project_director = ConversableAgent(
        name="项目总监",
        system_message="你是项目总监，负责将项目分解给各专业团队。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 第二层：各专业团队负责人
    # ---------------------------------------------

    frontend_lead = ConversableAgent(
        name="前端负责人",
        system_message="你是前端负责人，负责协调前端团队开发网站界面。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    backend_lead = ConversableAgent(
        name="后端负责人",
        system_message="你是后端负责人，负责协调后端团队开发API服务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    devops_lead = ConversableAgent(
        name="运维负责人",
        system_message="你是运维负责人，负责协调部署和运维工作。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 第三层：各团队成员
    # ---------------------------------------------

    # 前端团队成员
    frontend_dev1 = ConversableAgent(
        name="前端开发A",
        system_message="你是前端开发工程师，负责开发首页和商品列表页。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    frontend_dev2 = ConversableAgent(
        name="前端开发B",
        system_message="你是前端开发工程师，负责开发购物车和结算页。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 后端团队成员
    backend_dev1 = ConversableAgent(
        name="后端开发A",
        system_message="你是后端开发工程师，负责开发用户认证和商品管理API。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    backend_dev2 = ConversableAgent(
        name="后端开发B",
        system_message="你是后端开发工程师，负责开发订单和支付API。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 运维团队成员
    devops_dev1 = ConversableAgent(
        name="运维开发A",
        system_message="你是运维工程师，负责配置CI/CD流水线。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    devops_dev2 = ConversableAgent(
        name="运维开发B",
        system_message="你是运维工程师，负责配置服务器和网络。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 创建第三层群聊（各团队内部）
    # ---------------------------------------------

    frontend_groupchat = GroupChat(
        agents=[frontend_lead, frontend_dev1, frontend_dev2],
        messages=[],
        max_round=5,
    )
    frontend_manager = GroupChatManager(
        groupchat=frontend_groupchat,
        llm_config=llm_config,
    )

    backend_groupchat = GroupChat(
        agents=[backend_lead, backend_dev1, backend_dev2],
        messages=[],
        max_round=5,
    )
    backend_manager = GroupChatManager(
        groupchat=backend_groupchat,
        llm_config=llm_config,
    )

    devops_groupchat = GroupChat(
        agents=[devops_lead, devops_dev1, devops_dev2],
        messages=[],
        max_round=5,
    )
    devops_manager = GroupChatManager(
        groupchat=devops_groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 创建第二层群聊（团队间协调）
    # ---------------------------------------------

    team_coordination_groupchat = GroupChat(
        agents=[project_director, frontend_manager, backend_manager, devops_manager],
        messages=[],
        max_round=10,
    )
    team_coordination_manager = GroupChatManager(
        groupchat=team_coordination_groupchat,
        llm_config=llm_config,
    )

    # ---------------------------------------------
    # 执行层级任务分解
    # ---------------------------------------------

    print("\n层级结构:")
    print("  第1层: 项目总监")
    print("  ├── 第2层: 前端负责人 -> [前端开发A, 前端开发B]")
    print("  ├── 第2层: 后端负责人 -> [后端开发A, 后端开发B]")
    print("  └── 第2层: 运维负责人 -> [运维开发A, 运维开发B]")

    print("\n执行电商网站开发项目...")
    result = project_director.initiate_chat(
        team_coordination_manager,
        message="我们需要开发一个完整的电商网站，包括前端界面、后端API和部署运维。",
    )

    print(f"\n项目协调完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 前端群聊消息数: {len(frontend_groupchat.messages)}")
    print(f"  - 后端群聊消息数: {len(backend_groupchat.messages)}")
    print(f"  - 运维群聊消息数: {len(devops_groupchat.messages)}")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有嵌套GroupChat演示
    """
    print("=" * 60)
    print("嵌套GroupChat与层级管理 - 综合演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_nested_groupchat_basic()
    demo_sequential_subgroups()
    demo_parallel_subgroups()
    demo_nested_termination()
    demo_cross_level_communication()
    demo_hierarchical_task_decomposition()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()