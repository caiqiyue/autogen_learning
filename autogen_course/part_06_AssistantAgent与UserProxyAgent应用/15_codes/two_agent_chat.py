# two_agent_chat.py
# 第15节 ConversableAgent对话模式综合实战 - 双人对话模式配置
#
# 本文件演示双人对话模式（Two-Agent Chat）的配置方法和核心机制：
# 1. initiate_chat - 主动发起对话
# 2. generate_reply - 回复生成机制
# 3. register_reply - 注册自定义回复
# 4. 不同human_input_mode对对话行为的影响
# 5. 双人对话与GroupChat的区别
#
# ============================================================
# 双人对话模式（Two-Agent Chat）核心概念
# ============================================================
#
# 双人对话是AutoGen中最基础的协作模式，涉及两个Agent之间的直接对话。
# 与GroupChat相比，双人对话更加简单直接，适合一对一协作场景。
#
# 核心特点：
# 1. 一对一通信：消息直接发送给接收方，不经过中间人
# 2. 简单直接：无需GroupChatManager管理，配置简单
# 3. 明确的角色分工：发起者和接收者关系清晰
# 4. 灵活的终止控制：通过is_termination_msg控制对话终止
#
# 适用场景：
# - 人机交互：UserProxyAgent与AssistantAgent协作
# - 任务协作：专家一对一讨论问题
# - 工具调用：Agent调用工具获取信息后返回结果
# - 审查流程：一方生成内容，另一方审查修改
#
# ============================================================
# 对话模式对比
# ============================================================
#
# AutoGen支持多种对话模式，本节重点对比：
#
# 1. 双人对话模式（Two-Agent Chat）
#    - 两个Agent直接通信
#    - 使用 initiate_chat() 发起对话
#    - 消息直接传递给对方
#    - 适合简单的一对一协作
#
# 2. 群聊模式（GroupChat）
#    - 多个Agent通过GroupChatManager协作
#    - 消息广播给所有Agent
#    - LLM自动选择下一个发言者
#    - 适合多Agent团队协作
#
# 3. 嵌套对话模式（Nested Chat）
#    - Agent之间可以嵌套调用
#    - 支持并发对话
#    - 适合复杂的工作流
#
# 4. 异步对话模式（Async Chat）
#    - 使用async/await进行异步通信
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
# 第三部分：双人对话基本用法
# ============================================================

def demo_basic_two_agent_chat():
    """
    演示最基本的双人对话用法

    双人对话是最简单的多Agent协作模式：
    1. 创建一个AssistantAgent（助手）
    2. 创建一个UserProxyAgent（用户代理）
    3. 使用initiate_chat发起对话
    4. 观察回复和对话历史
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示1: 双人对话基本用法")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建两个Agent：一个助手和一个用户代理
    # ---------------------------------------------

    # 助手Agent - 负责回答问题和生成内容
    assistant = ConversableAgent(
        name="助手",
        system_message="你是一个有帮助的助手，擅长回答问题和提供建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",  # 助手不需要人工输入
    )

    # 用户代理Agent - 代表用户参与对话
    user_proxy = ConversableAgent(
        name="用户代理",
        system_message="你代表用户，可以发起对话并接收回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",  # 也不需要人工输入（演示用）
    )

    print("\n已创建两个Agent:")
    print(f"  - {assistant.name}: 负责生成回复")
    print(f"  - {user_proxy.name}: 代表用户发起对话")

    # ---------------------------------------------
    # 使用 initiate_chat 发起对话
    # ---------------------------------------------

    print("\n执行双人对话:")
    print("-" * 40)

    # 用户代理发起对话，向助手提问
    result = user_proxy.initiate_chat(
        assistant,  # 指定对话对象
        message="请介绍一下人工智能的发展历史。",
    )

    print("-" * 40)
    print(f"\n对话完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 最后一条消息长度: {len(result.last_message.get('content', ''))} 字符")

    return result


# ============================================================
# 第四部分：human_input_mode 对话行为的影响
# ============================================================

def demo_human_input_modes():
    """
    演示不同的 human_input_mode 对双人对话行为的影响

    human_input_mode 有三种模式：
    1. "ALWAYS" - 每次回复前都需要人工确认
    2. "TERMINATE" - 只有当is_termination_msg返回False时才请求人工输入
    3. "NEVER" - 从不请求人工输入，完全自动

    这个参数决定了UserProxyAgent在生成回复前的行为：
    - ALWAYS: 每一步都需要人工批准，适合需要严格控制的场景
    - TERMINATE: 默认终止时请求输入，适合人机协作场景
    - NEVER: 完全自动，适合自动化流程
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示2: human_input_mode 对话行为影响")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 模式1: ALWAYS - 每次都需要人工确认
    # ---------------------------------------------

    print("\n--- human_input_mode = 'ALWAYS' ---")
    print("特点:")
    print("  - 每次Agent生成回复前都需要人工确认")
    print("  - 用户可以修改回复内容")
    print("  - 适合需要严格控制的场景")
    print("  - 会阻塞等待用户输入")

    assistant_always = ConversableAgent(
        name="助手_ALWAYS",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,
        human_input_mode="ALWAYS",  # 关键参数：每次都需要人工确认
    )

    user_proxy_always = ConversableAgent(
        name="用户代理_ALWAYS",
        system_message="你代表用户参与对话。",
        llm_config=llm_config,
        human_input_mode="ALWAYS",
    )

    print("  配置: human_input_mode='ALWAYS'")
    print("  说明: 由于需要人工输入，演示中改为NEVER模式实际运行")

    # 注意：ALWAYS模式在实际运行中需要用户提供输入
    # 这里演示配置方式，实际执行会阻塞

    # ---------------------------------------------
    # 模式2: TERMINATE - 智能请求输入
    # ---------------------------------------------

    print("\n--- human_input_mode = 'TERMINATE' ---")
    print("特点:")
    print("  - 当is_termination_msg返回False时请求人工输入")
    print("  - 当正常终止时不请求输入")
    print("  - 适合人机协作场景")
    print("  - 是UserProxyAgent的默认模式")

    assistant_terminate = ConversableAgent(
        name="助手_TERMINATE",
        system_message="你是一个有帮助的助手。如果话题完成，说'DONE'来结束。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "DONE" in msg.get("content", ""),
    )

    user_proxy_terminate = ConversableAgent(
        name="用户代理_TERMINATE",
        system_message="你代表用户参与对话。",
        llm_config=llm_config,
        human_input_mode="TERMINATE",  # 关键参数：智能请求输入
    )

    print("  配置: human_input_mode='TERMINATE'")
    print("  终止条件: 当助手回复包含'DONE'时不请求输入，直接终止")

    # ---------------------------------------------
    # 模式3: NEVER - 完全自动
    # ---------------------------------------------

    print("\n--- human_input_mode = 'NEVER' ---")
    print("特点:")
    print("  - 从不请求人工输入")
    print("  - Agent完全自动运行")
    print("  - 适合自动化流程")
    print("  - 适合代码执行和工具调用")

    assistant_never = ConversableAgent(
        name="助手_NEVER",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",  # 关键参数：完全自动
    )

    user_proxy_never = ConversableAgent(
        name="用户代理_NEVER",
        system_message="你代表用户参与对话。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("  配置: human_input_mode='NEVER'")

    # 实际运行NEVER模式的对话
    print("\n运行NEVER模式对话:")
    result = user_proxy_never.initiate_chat(
        assistant_never,
        message="用一句话介绍Python。",
    )

    print(f"  对话完成，消息数: {len(result.chat_history)}")

    # ---------------------------------------------
    # 总结对比
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("human_input_mode 对比总结")
    print("=" * 60)
    print("\n模式          | 人工输入频率 | 适用场景")
    print("-" * 50)
    print("ALWAYS        | 每次都请求    | 教学演示、严格审核")
    print("TERMINATE     | 智能请求      | 人机协作、迭代优化")
    print("NEVER         | 从不请求      | 自动化流程、代码执行")


# ============================================================
# 第五部分：register_reply 自定义回复逻辑
# ============================================================

def demo_register_reply():
    """
    演示使用 register_reply 注册自定义回复逻辑

    register_reply 是ConversableAgent的核心机制之一：
    1. 允许注册自定义的回复生成函数
    2. 可以覆盖默认的LLM回复
    3. 支持基于规则的回复（关键字匹配、状态机等）
    4. 与generate_reply策略链配合使用

    使用场景：
    - 实现关键字触发的不用LLM的回复
    - 添加安全检查或内容过滤
    - 实现特殊的对话逻辑
    - 集成外部系统或API
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示3: register_reply 自定义回复逻辑")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建带自定义回复的Agent
    # ---------------------------------------------

    assistant = ConversableAgent(
        name="助手_自定义",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建助手Agent")
    print("将为其注册自定义回复逻辑...")

    # ---------------------------------------------
    # 定义自定义回复函数
    # ---------------------------------------------

    def custom_reply_function(
        messages,  # 消息历史
        sender,    # 发送者
        config,    # 配置
    ):
        """
        自定义回复函数

        Args:
            messages: 对话消息历史列表
            sender: 发送消息的Agent
            config: 额外配置

        Returns:
            str: 自定义回复内容，如果不触发则返回None
        """
        # 获取最后一条用户消息
        if not messages:
            return None

        last_message = messages[-1]
        content = last_message.get("content", "").lower()

        # 检测关键字，触发自定义回复
        if "hello" in content or "你好" in content:
            return "你好！我是自定义回复，很高兴为您服务！"
        elif "time" in content or "时间" in content:
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"当前时间是: {now}"
        elif "bye" in content or "再见" in content:
            return "再见！祝您有美好的一天！"

        # 如果没有匹配的关键字，返回None使用默认LLM回复
        return None

    # 注册自定义回复
    # register_reply参数：
    # 1. name: 注册的回复函数名称
    # 2. reply_func: 回复函数
    # 3. override: 是否覆盖已存在的回复（默认False）
    assistant.register_reply(
        reply_func=custom_reply_function,
        name="custom_reply",  # 标识这个reply的名称
    )

    print("\n已注册自定义回复函数:")
    print("  - 触发条件: 消息包含特定关键字")
    print("  - 'hello/你好' -> 自定义问候")
    print("  - 'time/时间' -> 返回当前时间")
    print("  - 'bye/再见' -> 自定义告别")
    print("  - 其他 -> 使用默认LLM回复")

    # ---------------------------------------------
    # 测试自定义回复
    # ---------------------------------------------

    user_proxy = ConversableAgent(
        name="用户",
        system_message="你代表用户。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n测试1: 触发自定义问候")
    print("-" * 40)
    result1 = user_proxy.initiate_chat(
        assistant,
        message="Hello!",
    )
    print(f"结果: 回复数={len(result1.chat_history)}")

    print("\n测试2: 触发自定义时间查询")
    print("-" * 40)
    result2 = user_proxy.initiate_chat(
        assistant,
        message="现在的时间是多少？",
    )
    print(f"结果: 回复数={len(result2.chat_history)}")

    print("\n测试3: 使用默认LLM回复")
    print("-" * 40)
    result3 = user_proxy.initiate_chat(
        assistant,
        message="解释一下什么是机器学习。",
    )
    print(f"结果: 回复数={len(result3.chat_history)}")


# ============================================================
# 第六部分：双人对话的终止控制
# ============================================================

def demo_termination_control():
    """
    演示双人对话中的终止条件控制

    双人对话的终止条件由以下因素控制：
    1. is_termination_msg - Agent级别的终止消息检测
    2. max_consecutive_auto_reply - 最大连续自动回复数
    3. 手动终止 - 外部代码可以强制终止对话

    与GroupChat的区别：
    - GroupChat有max_round限制
    - 双人对话通过max_consecutive_auto_reply控制
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示4: 双人对话的终止控制")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 终止条件1: is_termination_msg
    # ---------------------------------------------

    print("\n--- 方式1: is_termination_msg ---")
    print("特点:")
    print("  - 当回复消息匹配终止条件时终止")
    print("  - 可以是关键字匹配或正则表达式")
    print("  - 适合任务明确完成的场景")

    assistant_term = ConversableAgent(
        name="助手_终止",
        system_message="""你是一个任务助手。
你会评估任务是否完成。如果完成，说'TASK_DONE'来终止对话。
否则继续回答。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TASK_DONE" in msg.get("content", ""),
    )

    user_proxy_term = ConversableAgent(
        name="用户_终止",
        system_message="你代表用户。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    result = user_proxy_term.initiate_chat(
        assistant_term,
        message="请简要介绍一下Python的基本语法。",
    )

    print(f"\n执行结果:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    if any("TASK_DONE" in msg.get("content", "") for msg in result.chat_history):
        print("  - 终止原因: Agent返回包含'TASK_DONE'")
    else:
        print("  - 终止原因: 自然结束")

    # ---------------------------------------------
    # 终止条件2: max_consecutive_auto_reply
    # ---------------------------------------------

    print("\n--- 方式2: max_consecutive_auto_reply ---")
    print("特点:")
    print("  - 限制连续自动回复的最大次数")
    print("  - 达到限制后强制终止")
    print("  - 适合防止无限循环")

    assistant_max = ConversableAgent(
        name="助手_最大",
        system_message="你是一个话多的助手，总是长篇大论。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,  # 关键参数：最多3次连续回复
    )

    user_proxy_max = ConversableAgent(
        name="用户_最大",
        system_message="你代表用户。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    result = user_proxy_max.initiate_chat(
        assistant_max,
        message="给我讲个笑话。",
    )

    print(f"\n执行结果:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 限制: max_consecutive_auto_reply=3")

    # 统计对话轮数
    user_msgs = sum(1 for msg in result.chat_history if msg.get("role") == "user")
    assistant_msgs = sum(1 for msg in result.chat_history if msg.get("role") == "assistant")
    print(f"  - 用户消息数: {user_msgs}")
    print(f"  - 助手消息数: {assistant_msgs}")


# ============================================================
# 第七部分：双人对话 vs GroupChat
# ============================================================

def demo_two_agent_vs_groupchat():
    """
    对比双人对话与GroupChat的区别

    双人对话和GroupChat是AutoGen的两种主要协作模式，
    它们在配置复杂度、适用场景、消息传递方式上有明显区别。
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示5: 双人对话 vs GroupChat 对比")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建相同的Agent用于两种模式
    # ---------------------------------------------

    agent_alpha = ConversableAgent(
        name="Alpha",
        system_message="你是Alpha，团队中的成员。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_beta = ConversableAgent(
        name="Beta",
        system_message="你是Beta，团队中的成员。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_gamma = ConversableAgent(
        name="Gamma",
        system_message="你是Gamma，团队中的成员。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建3个相同的Agent: Alpha, Beta, Gamma")

    # ---------------------------------------------
    # 双人对话模式
    # ---------------------------------------------

    print("\n--- 双人对话模式 ---")
    print("配置:")
    print("  1. 创建两个Agent")
    print("  2. 使用initiate_chat直接通信")
    print("  3. 无需GroupChatManager")

    print("\n执行双人对话 (Alpha -> Beta):")
    result_two = agent_alpha.initiate_chat(
        agent_beta,
        message="Beta，你怎么看待人工智能的未来？",
    )
    print(f"  - 参与者: Alpha, Beta")
    print(f"  - 消息数: {len(result_two.chat_history)}")

    # ---------------------------------------------
    # GroupChat模式
    # ---------------------------------------------

    print("\n--- GroupChat模式 ---")
    print("配置:")
    print("  1. 创建多个Agent")
    print("  2. 创建GroupChat容器")
    print("  3. 创建GroupChatManager")
    print("  4. 所有Agent通过Manager通信")

    groupchat = GroupChat(
        agents=[agent_alpha, agent_beta, agent_gamma],
        messages=[],
        max_round=4,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print("\n执行GroupChat (Alpha发起):")
    result_group = agent_alpha.initiate_chat(
        manager,
        message="各位，让我们讨论一下人工智能的未来。",
    )
    print(f"  - 参与者: Alpha, Beta, Gamma")
    print(f"  - 消息数: {len(result_group.chat_history)}")

    # ---------------------------------------------
    # 关键区别总结
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("双人对话 vs GroupChat 关键区别")
    print("=" * 60)
    print("\n特性          | 双人对话       | GroupChat")
    print("-" * 55)
    print("Agent数量     | 2个            | 多个")
    print("消息传递      | 直接发送        | 通过Manager广播")
    print("配置复杂度    | 低             | 中")
    print("发言控制      | 无（直接通信）  | speaker_selection")
    print("发言均衡      | 不适用         | allow_repeat控制")
    print("适用场景      | 一对一协作      | 团队协作讨论")


# ============================================================
# 第八部分：实际应用场景
# ============================================================

def demo_practical_scenario_code_review():
    """
    实际场景演示：代码审查流程

    场景描述：
    - 程序员编写代码
    - 审查员审查代码
    - 程序员根据审查意见修改代码
    - 审查员确认修改
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示6: 实际场景 - 代码审查流程")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建代码审查团队
    # ---------------------------------------------

    coder = ConversableAgent(
        name="程序员",
        system_message="""你是经验丰富的Python程序员。
职责：
- 编写高质量的Python代码
- 根据审查意见修改代码
- 解释代码实现

你会根据审查员的反馈进行代码修改。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    reviewer = ConversableAgent(
        name="审查员",
        system_message="""你是资深代码审查员。
职责：
- 审查代码质量和安全性
- 提出具体的改进建议
- 确认修改是否满足要求

你会仔细审查程序员提交的代码。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "LGTM" in msg.get("content", ""),
    )

    print("\n代码审查团队:")
    print("  - 程序员: 负责编写和修改代码")
    print("  - 审查员: 负责审查代码")

    # ---------------------------------------------
    # 阶段1: 程序员提交代码
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("阶段1: 程序员提交初始代码")
    print("-" * 40)

    initial_code = '''
def calculate_factorial(n):
    """计算阶乘"""
    if n < 0:
        raise ValueError("负数没有阶乘")
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
'''

    result1 = coder.initiate_chat(
        reviewer,
        message=f"请审查以下代码:\n{initial_code}",
    )

    print(f"  审查完成，消息数: {len(result1.chat_history)}")

    # ---------------------------------------------
    # 阶段2: 程序员修改代码（模拟）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("阶段2: 根据审查意见修改代码")
    print("-" * 40)

    # 注意：在实际场景中，程序员会根据审查意见修改代码
    # 这里演示双人对话的迭代过程

    improved_code = '''
def calculate_factorial(n: int) -> int:
    """计算非负整数的阶乘

    Args:
        n: 非负整数

    Returns:
        n的阶乘结果

    Raises:
        ValueError: 当n为负数时
    """
    if n < 0:
        raise ValueError("负数没有阶乘")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
'''

    result2 = coder.initiate_chat(
        reviewer,
        message=f"我已经根据意见改进了代码:\n{improved_code}\n请再次审查。",
    )

    print(f"  再次审查完成，消息数: {len(result2.chat_history)}")

    # ---------------------------------------------
    # 总结
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("代码审查流程总结")
    print("=" * 60)
    print("\n流程:")
    print("  1. 程序员提交初始代码")
    print("  2. 审查员审查并提出意见")
    print("  3. 程序员修改代码")
    print("  4. 审查员确认通过（说'LGTM'）")
    print("  5. 流程结束")

    print("\n双人对话优势:")
    print("  - 简单的请求-响应模式")
    print("  - 清晰的职责分工")
    print("  - 易于调试和追踪")
    print("  - 适合迭代式工作流")


# ============================================================
# 第九部分：对话模式的选型决策
# ============================================================

def demo_mode_selection_guide():
    """
    对话模式选择决策指南

    根据不同场景选择合适的对话模式：
    1. 双人对话: 一对一协作，简单的请求-响应
    2. GroupChat: 多Agent团队协作，需要发言均衡
    3. 嵌套对话: 需要在对话中调用其他Agent
    4. 异步对话: 需要并发执行，提高性能
    5. 流式对话: 长文本生成，需要实时反馈
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示7: 对话模式选择决策指南")
    print("=" * 60)

    llm_config = build_llm_config()

    print("\n=== 何时使用双人对话 ===")
    print("适用场景:")
    print("  - UserProxyAgent + AssistantAgent 协作")
    print("  - 人机交互式问答")
    print("  - 简单的请求-响应任务")
    print("  - 一对一专家咨询")
    print("  - 代码生成+审查的简单流程")
    print("\n不适用场景:")
    print("  - 需要多个Agent协作")
    print("  - 需要发言均衡")
    print("  - 需要广播消息")

    print("\n=== 何时使用GroupChat ===")
    print("适用场景:")
    print("  - 多个专家团队讨论")
    print("  - 需要LLM自动选择发言者")
    print("  - 需要发言均衡控制")
    print("  - 复杂的团队协作流程")
    print("\n不适用场景:")
    print("  - 只有两个Agent")
    print("  - 需要精确控制发言顺序")
    print("  - 配置简单即可满足需求")

    print("\n=== 决策流程 ===")
    print("  1. Agent数量 = 2?")
    print("     -> 是: 使用双人对话")
    print("     -> 否: 继续判断")
    print("  2. 需要发言均衡?")
    print("     -> 是: 使用GroupChat + allow_repeat='never'")
    print("     -> 否: 继续判断")
    print("  3. 需要LLM自动选择发言者?")
    print("     -> 是: 使用GroupChat")
    print("     -> 否: 考虑manual模式或其他")

    print("\n=== 实际选择示例 ===")

    scenarios = [
        ("客服聊天机器人", "双人对话 - 一对一用户服务"),
        ("代码审查团队", "GroupChat - 程序员+审查员+架构师"),
        ("教学问答系统", "双人对话 - 学生提问，老师回答"),
        ("技术研讨会", "GroupChat - 多个专家讨论"),
        ("自动化测试", "双人对话 - 测试用例生成+执行"),
    ]

    print("\n场景                    | 推荐模式")
    print("-" * 45)
    for scenario, mode in scenarios:
        print(f"{scenario:<20} | {mode}")


# ============================================================
# 第十部分：综合演示
# ============================================================

def demo_comprehensive_two_agent():
    """
    综合演示：双人对话的完整配置

    展示如何配置一个完整的双人对话系统，
    包括LLM配置、Agent创建、对话发起和结果处理。
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("演示8: 双人对话完整配置")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建专业助手
    # ---------------------------------------------

    professional_assistant = ConversableAgent(
        name="专业助手",
        system_message="""你是一位专业的数据分析师。
你的职责：
- 分析数据并提供见解
- 创建可视化建议
- 回答数据相关问题
- 解释分析方法

请始终提供准确、专业的数据分析建议。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "谢谢" in msg.get("content", "") or "DONE" in msg.get("content", ""),
    )

    # ---------------------------------------------
    # 创建用户代理
    # ---------------------------------------------

    user_proxy = ConversableAgent(
        name="业务用户",
        system_message="你是一位业务分析师，需要数据分析支持。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n已创建完整配置的双人对话系统:")
    print("  - 专业助手: 数据分析专家")
    print("  - 业务用户: 数据需求方")

    # ---------------------------------------------
    # 执行对话
    # ---------------------------------------------

    print("\n执行数据分析对话:")
    print("-" * 40)

    result = user_proxy.initiate_chat(
        professional_assistant,
        message="""我有一份销售数据，包含以下字段：
- 日期
- 产品类别
- 销售额
- 客户类型

请分析一下：
1. 哪些产品类别表现最好？
2. 客户类型有什么特点？
3. 有什么改进建议？""",
    )

    print("-" * 40)
    print(f"\n对话完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 最后一条消息长度: {len(result.last_message.get('content', ''))} 字符")

    # ---------------------------------------------
    # 分析对话历史
    # ---------------------------------------------

    print("\n对话历史分析:")
    for i, msg in enumerate(result.chat_history):
        role = msg.get("role", "unknown")
        name = msg.get("name", "unknown")
        content_preview = msg.get("content", "")[:80]
        print(f"  [{i}] {name}({role}): {content_preview}...")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数：运行所有双人对话演示
    """
    print("=" * 60)
    print("ConversableAgent对话模式综合实战 - 双人对话模式")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_basic_two_agent_chat()
    demo_human_input_modes()
    demo_register_reply()
    demo_termination_control()
    demo_two_agent_vs_groupchat()
    demo_practical_scenario_code_review()
    demo_mode_selection_guide()
    demo_comprehensive_two_agent()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()