# human_input_modes.py
# 第17节 UserProxyAgent三种human_input_mode详解 - 三种模式配置演示
#
# 本文件演示 UserProxyAgent 的三种 human_input_mode：
# 1. ALWAYS - 始终需要人类输入
# 2. NEVER - 完全自动回复，无需人类输入
# 3. TERMINATE - 自动回复直到特定条件满足才请求输入
#
# ============================================================
# human_input_mode 核心概念
# ============================================================
#
# human_input_mode 是 UserProxyAgent 的核心配置，决定了何时需要人类介入。
# 这个参数直接影响 Agent 的人机协作方式。
#
# 三种模式对比：
# | 模式        | 触发条件                          | 适用场景                    |
# |-------------|-----------------------------------|-----------------------------|
# | ALWAYS      | 每次回复前都需要人类输入          | 需要完全人工控制的场景      |
# | NEVER       | 完全自动，从不请求人类输入        | 全自动流程、无监督运行      |
# | TERMINATE   | auto回复直到is_termination_msg   | 有人监督但不需要全程监控    |
#
# 重要：human_input_mode 与 max_consecutive_auto_reply 密切相关
# - max_consecutive_auto_reply: 最大连续自动回复次数
# - 达到该次数后，即使模式是TERMINATE也会请求人类输入
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
# 第三部分：UserProxyAgent 与三种 human_input_mode
# ============================================================
#
# UserProxyAgent 是 UserProxy 类的实例，专门用于代表人类用户。
# 它的核心职责是在 AI Agent 和真实人类之间建立桥梁。
#
# UserProxyAgent 继承自 ConversableAgent，因此具有：
# - generate_reply() - 生成回复的核心方法
# - initiate_chat() - 发起对话
# - register_reply() - 注册回复函数
#
# 关键区别在于 human_input_mode 的处理逻辑
#
# ============================================================


def demo_always_mode():
    """
    演示 ALWAYS 模式：始终需要人类输入

    ALWAYS 模式是最严格的人机协作模式：
    - 每次 Agent 要回复之前，都会暂停等待人类输入
    - 人类可以：批准、修改、或拒绝 AI 的建议
    - 适用于：需要完全人工控制的关键决策场景

    工作原理：
    1. Agent 准备生成回复
    2. 在调用 generate_reply 之前，会调用 getHumanInput()
    3. 等待人类输入后才继续执行
    4. 人类的输入成为对话的一部分

    典型应用场景：
    - 金融交易：每笔交易都需要人工确认
    - 医疗诊断：每个诊断建议都需要医生审核
    - 法律文件：重要文件需要人工审核后才能发送
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示1: ALWAYS 模式 - 始终需要人类输入")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建 UserProxyAgent，设置为 ALWAYS 模式
    # ---------------------------------------------

    print("\n创建 UserProxyAgent (ALWAYS 模式):")

    # 创建一个 UserProxyAgent，human_input_mode="ALWAYS"
    user_proxy_always = UserProxyAgent(
        name="用户代理_ALWAYS",
        # 系统提示：这个 Agent 代表人类用户，每次回复前都需要确认
        system_message="""你是人类用户的代理。
每次 AI Agent 给出建议后，你需要：
1. 审查 AI 的建议
2. 决定是否批准或修改
3. 输入你的决定

当你想终止对话时，输入 'exit'。""",
        # 关键配置：human_input_mode = "ALWAYS"
        # 这意味着每次 Agent 回复前都会等待你的输入
        human_input_mode="ALWAYS",
        # 最大自动回复次数（ALWAYS 模式下不生效，但需要设置）
        max_consecutive_auto_reply=0,
        # LLM 配置（用于生成回复）
        llm_config=llm_config,
    )

    # 创建一个简单的 Assistant Agent 来演示
    assistant = ConversableAgent(
        name="AI助手",
        system_message="你是一个友好的 AI 助手，提供有用的建议。",
        llm_config=llm_config,
        human_input_mode="NEVER",  # AI 端不需要人工输入
    )

    print(f"  - Agent 名称: {user_proxy_always.name}")
    print(f"  - human_input_mode: {user_proxy_always.human_input_mode}")
    print(f"  - max_consecutive_auto_reply: {user_proxy_always.max_consecutive_auto_reply}")

    # ---------------------------------------------
    # 展示 ALWAYS 模式的特点
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("ALWAYS 模式的行为特点:")
    print("-" * 40)
    print("  1. 每次 AI 回复前都会暂停等待人类输入")
    print("  2. 人类可以修改 AI 的建议或直接输入新内容")
    print("  3. 对话完全在人类控制之下")
    print("  4. 适合需要完全人工审核的关键任务")

    print("\n  优点:")
    print("    - 完全控制，安全性最高")
    print("    - 可以随时干预 AI 的行为")
    print("    - 适合高风险操作")

    print("\n  缺点:")
    print("    - 无法实现全自动流程")
    print("    - 需要人类全程监控")
    print("    - 效率较低，不适合大规模任务")

    print("\n  适用场景:")
    print("    - 金融交易审批")
    print("    - 医疗诊断建议审核")
    print("    - 法律文件审批")
    print("    - 任何需要 100% 人工确认的高风险操作")

    # ---------------------------------------------
    # 注意：由于 ALWAYS 模式需要人工输入，这里只展示配置
    # 实际运行需要人类交互
    # ---------------------------------------------

    print("\n  注意：实际运行此代码时，每次 AI 回复前都会暂停等待你的输入")
    print("  测试方式：手动调用 initiate_chat 并观察行为")


def demo_never_mode():
    """
    演示 NEVER 模式：完全自动回复

    NEVER 模式是完全无人值守的模式：
    - Agent 永远不会请求人类输入
    - 所有回复都由 AI 自动生成
    - 依靠 is_termination_msg 或 max_consecutive_auto_reply 来停止

    工作原理：
    1. Agent 检查 human_input_mode 是否为 "NEVER"
    2. 如果是，直接进入自动回复流程
    3. 调用 generate_reply 生成回复
    4. 检查终止条件，决定是否继续

    典型应用场景：
    - 批量数据处理：无需人工干预的批处理任务
    - 信息收集：自动收集和整理信息
    - 代码生成：自动生成标准化的代码模板
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示2: NEVER 模式 - 完全自动回复")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建 UserProxyAgent，设置为 NEVER 模式
    # ---------------------------------------------

    print("\n创建 UserProxyAgent (NEVER 模式):")

    user_proxy_never = UserProxyAgent(
        name="用户代理_NEVER",
        system_message="""你是人类用户的代理，但设置为全自动模式。
你不会请求任何人类输入，所有回复都由 AI 自动生成。
当你认为任务完成时，说 'exit' 来结束对话。""",
        # 关键配置：human_input_mode = "NEVER"
        human_input_mode="NEVER",
        # max_consecutive_auto_reply 设置为较大值允许连续自动回复
        max_consecutive_auto_reply=10,
        llm_config=llm_config,
        # is_termination_msg 用于检测何时自动终止
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    assistant = ConversableAgent(
        name="AI助手",
        system_message="你是一个友好的 AI 助手。简洁回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"  - Agent 名称: {user_proxy_never.name}")
    print(f"  - human_input_mode: {user_proxy_never.human_input_mode}")
    print(f"  - max_consecutive_auto_reply: {user_proxy_never.max_consecutive_auto_reply}")
    print(f"  - is_termination_msg: 检测 'exit' 关键词")

    # ---------------------------------------------
    # 执行一个简单的自动对话
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("执行自动对话（无需人类干预）")
    print("-" * 40)

    # 启动对话
    result = user_proxy_never.initiate_chat(
        assistant,
        message="请介绍一下你自己，然后说 'exit' 结束对话。",
    )

    print(f"\n对话完成:")
    print(f"  - 总消息数: {len(result.chat_history)}")
    print(f"  - 无需任何人类输入")

    # ---------------------------------------------
    # NEVER 模式的特点总结
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("NEVER 模式的行为特点:")
    print("-" * 40)
    print("  1. 完全自动运行，从不请求人类输入")
    print("  2. 依靠终止条件来结束对话")
    print("  3. max_consecutive_auto_reply 控制最大连续自动回复")
    print("  4. 适合需要无人值守运行的场景")

    print("\n  优点:")
    print("    - 全自动，效率高")
    print("    - 无需人工监控")
    print("    - 适合批量处理任务")

    print("\n  缺点:")
    print("    - 无法中途干预")
    print("    - 安全性较低（如果 AI 生成有害内容）")
    print("    - 不适合需要人工审核的场景")

    print("\n  适用场景:")
    print("    - 批量信息收集和整理")
    print("    - 自动化测试")
    print("    - 定时任务执行")
    print("    - 无人值守的数据处理")


def demo_terminate_mode():
    """
    演示 TERMINATE 模式：智能混合模式

    TERMINATE 模式是最常用的模式，结合了 ALWAYS 和 NEVER 的优点：
    - 默认情况下自动运行，AI 自主决策
    - 当达到终止条件时，才请求人类输入确认
    - 适合"监督但不干预"的场景

    工作原理：
    1. Agent 检查 human_input_mode 是否为 "TERMINATE"
    2. 如果是，正常调用 generate_reply
    3. 检查 max_consecutive_auto_reply：
       - 如果未达到，继续自动回复
       - 如果已达到，请求人类输入
    4. 检查 is_termination_msg：
       - 如果返回 True，请求人类输入确认
       - 如果返回 False，继续自动回复

    与 max_consecutive_auto_reply 的交互：
    - max_consecutive_auto_reply=3：连续自动回复3次后请求人类输入
    - 每次收到新的用户消息，计数器会重置

    典型应用场景：
    - AI 编程助手：AI 生成代码，人类最后审核
    - 文档助手：AI 起草文档，人类审核定稿
    - 客服机器人：AI 处理常见问题，转人工处理复杂问题
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示3: TERMINATE 模式 - 智能混合模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建 UserProxyAgent，设置为 TERMINATE 模式
    # ---------------------------------------------

    print("\n创建 UserProxyAgent (TERMINATE 模式):")

    user_proxy_terminate = UserProxyAgent(
        name="用户代理_TERMINATE",
        system_message="""你是人类用户的代理，设置为监督模式。
你会让 AI 自动处理任务，但当：
1. 连续自动回复达到限制，或
2. AI 请求确认，或
3. 任务似乎完成时
你会请求人类输入来确认或指导下一步。

当你想终止对话时，输入 'exit'。""",
        # 关键配置：human_input_mode = "TERMINATE"
        human_input_mode="TERMINATE",
        # 关键配置：连续自动回复3次后请求人类输入
        # 这是 TERMINATE 模式的核心控制参数
        max_consecutive_auto_reply=3,
        llm_config=llm_config,
        # 终止消息检测
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    assistant = ConversableAgent(
        name="AI助手",
        system_message="""你是一个有用的 AI 助手。
你会根据用户需求提供帮助。
当你认为任务完成时，说明并等待用户确认。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"  - Agent 名称: {user_proxy_terminate.name}")
    print(f"  - human_input_mode: {user_proxy_terminate.human_input_mode}")
    print(f"  - max_consecutive_auto_reply: {user_proxy_terminate.max_consecutive_auto_reply}")
    print("  - 模式说明: 自动运行，直到达到限制才请求人类确认")

    # ---------------------------------------------
    # 展示 TERMINATE 模式的工作流程
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("TERMINATE 模式的工作流程:")
    print("-" * 40)
    print("  1. 收到用户消息")
    print("  2. 检查 auto_reply 计数器")
    print("  3. 如果计数器 < max_consecutive_auto_reply:")
    print("     - 自动调用 generate_reply")
    print("     - 计数器 +1")
    print("     - 检查 is_termination_msg")
    print("  4. 如果计数器 >= max_consecutive_auto_reply:")
    print("     - 请求人类输入")
    print("     - 重置计数器")
    print("  5. 如果 is_termination_msg 返回 True:")
    print("     - 请求人类输入确认终止")
    print("  6. 收到人类输入后继续或终止")

    # ---------------------------------------------
    # TERMINATE 模式的特点总结
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("TERMINATE 模式的行为特点:")
    print("-" * 40)
    print("  1. 默认自动运行，AI 自主决策")
    print("  2. 达到 max_consecutive_auto_reply 限制后请求人工确认")
    print("  3. 可以中途干预 AI 的行为")
    print("  4. 适合'监督但不全程干预'的场景")

    print("\n  优点:")
    print("    - 平衡效率和安全性")
    print("    - 可以中途干预")
    print("    - 适合大多数实际应用场景")

    print("\n  缺点:")
    print("    - 需要合理设置 max_consecutive_auto_reply")
    print("    - 如果设置不当可能过于频繁请求人工确认")

    print("\n  max_consecutive_auto_reply 设置建议:")
    print("    - 1-2: 高频人工干预，适合简单确认任务")
    print("    - 3-5: 中等干预频率，适合标准任务（推荐）")
    print("    - 10+: 低频干预，适合长对话任务")


def demo_max_consecutive_auto_reply_interaction():
    """
    演示 max_consecutive_auto_reply 与 human_input_mode 的交互

    max_consecutive_auto_reply 是控制自动回复次数的关键参数：
    - 控制连续自动回复的最大次数
    - 达到该次数后，必须请求人类输入才能继续

    与三种 human_input_mode 的交互：
    - ALWAYS: max_consecutive_auto_reply 不生效（每次都请求输入）
    - NEVER: max_consecutive_auto_reply 限制自动回复次数，到达后强制终止
    - TERMINATE: max_consecutive_auto_reply 控制何时请求人类输入
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示4: max_consecutive_auto_reply 与 human_input_mode 交互")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建简单的 Assistant Agent
    # ---------------------------------------------

    assistant = ConversableAgent(
        name="AI助手",
        system_message="你是一个简洁的助手，简短回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 场景1：max_consecutive_auto_reply=1（每次都确认）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("场景1: max_consecutive_auto_reply=1")
    print("-" * 40)

    user_proxy_1 = UserProxyAgent(
        name="用户代理_1次",
        system_message="你是用户代理。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=1,  # 只允许1次自动回复
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    print(f"  - max_consecutive_auto_reply: 1")
    print(f"  - 行为: AI 每回复1次，就请求人类输入确认")
    print(f"  - 效果: 接近 ALWAYS 模式")

    # ---------------------------------------------
    # 场景2：max_consecutive_auto_reply=3（标准配置）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("场景2: max_consecutive_auto_reply=3")
    print("-" * 40)

    user_proxy_3 = UserProxyAgent(
        name="用户代理_3次",
        system_message="你是用户代理。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=3,  # 允许3次自动回复
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    print(f"  - max_consecutive_auto_reply: 3")
    print(f"  - 行为: AI 连续回复3次后才请求人类输入")
    print(f"  - 效果: 平衡效率和安全性（推荐配置）")

    # ---------------------------------------------
    # 场景3：max_consecutive_auto_reply=10（长对话）
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("场景3: max_consecutive_auto_reply=10")
    print("-" * 40)

    user_proxy_10 = UserProxyAgent(
        name="用户代理_10次",
        system_message="你是用户代理。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,  # 允许10次自动回复
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    print(f"  - max_consecutive_auto_reply: 10")
    print(f"  - 行为: AI 可以连续回复10次才请求人类输入")
    print(f"  - 效果: 适合长对话任务")

    # ---------------------------------------------
    # 总结配置建议
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("max_consecutive_auto_reply 配置建议")
    print("=" * 60)

    print("\n| 场景              | max_consecutive_auto_reply | 说明               |")
    print("|-------------------|---------------------------|---------------------|")
    print("| 快速确认任务      | 1-2                        | 需要频繁确认        |")
    print("| 标准任务          | 3-5                        | 平衡效率和安全（推荐）|")
    print("| 复杂长对话        | 10+                        | 适合深入讨论        |")
    print("| 全自动流程        | 较大值+终止条件            | 无人值守运行        |")


def demo_exit_as_termination_signal():
    """
    演示 'exit' 作为强制终止信号的源码实现

    在 AutoGen 中，当人类输入 'exit' 时，会触发强制终止。
    这是通过 get_human_input 方法实现的：
    1. 获取人类输入
    2. 检查是否为 'exit'
    3. 如果是 'exit'，设置标志位触发终止

    源码层面的实现逻辑：
    ```python
    def get_human_input(self, prompt: str) -> str:
        # 获取输入
        reply = input(prompt)  # 或使用 get_input 方法

        # 检查是否为退出命令
        if self.human_input_mode == "ALWAYS" and reply.lower() == "exit":
            # 在 ALWAYS 模式下，exit 不会立即终止
            # 只会设置一个标志
            pass

        return reply
    ```

    关键点：
    1. 'exit' 是一个约定俗成的终止信号
    2. Agent 会在 is_termination_msg 中检查 'exit'
    3. 当检测到 'exit' 时，对话会优雅地终止
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示5: 'exit' 作为强制终止信号")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建演示用的 Agent
    # ---------------------------------------------

    user_proxy = UserProxyAgent(
        name="用户代理",
        system_message="""你是用户代理。
当你想终止对话时，输入 'exit'。
AI 助手会检测到这个信号并优雅地结束对话。""",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        # 关键：is_termination_msg 检测 'exit'
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    assistant = ConversableAgent(
        name="AI助手",
        system_message="""你是一个友好的 AI 助手。
当你认为任务完成时，说 'exit' 来结束对话。
否则继续帮助用户。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        # AI 端也检测 'exit' 作为终止信号
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    print("\n配置说明:")
    print(f"  - UserProxyAgent.is_termination_msg: 检测 'exit'")
    print(f"  - AssistantAgent.is_termination_msg: 检测 'exit'")
    print("  - 双方都检测 'exit' 确保对话可以正确终止")

    # ---------------------------------------------
    # 展示源码层面的实现逻辑
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("'exit' 终止信号的工作原理:")
    print("-" * 40)
    print("  1. 人类用户输入 'exit'")
    print("  2. get_human_input() 方法被调用")
    print("  3. 输入被传递给 Agent")
    print("  4. is_termination_msg() 检查输入")
    print("  5. 如果包含 'exit'，返回 True")
    print("  6. 对话优雅终止")

    print("\n  注意事项:")
    print("    - 'exit' 是约定俗成的信号，AutoGen 默认处理")
    print("    - 可以自定义终止关键词，如 'quit', 'bye' 等")
    print("    - 终止信号应该在整个对话中保持一致")


def demo_mode_selection_guide():
    """
    演示模式选择指南

    根据不同的场景选择合适的 human_input_mode：
    - 场景1：需要 100% 人工控制 -> ALWAYS
    - 场景2：完全无人值守 -> NEVER
    - 场景3：监督但不干预 -> TERMINATE
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示6: human_input_mode 选择指南")
    print("=" * 60)

    print("\n" + "-" * 40)
    print("模式选择决策树:")
    print("-" * 40)
    print("  是否需要人工全程监控？")
    print("    ├─ 是 → ALWAYS 模式")
    print("    └─ 否 → 继续判断")
    print("          ├─ 是否需要完全无人值守？")
    print("          │    ├─ 是 → NEVER 模式")
    print("          │    └─ 否 → TERMINATE 模式")
    print("          └─ 任务复杂度？")
    print("               ├─ 简单任务 → max_consecutive_auto_reply=1-2")
    print("               └─ 复杂任务 → max_consecutive_auto_reply=3-10")

    print("\n" + "-" * 40)
    print("典型场景对应表:")
    print("-" * 40)

    scenarios = [
        ("金融交易审批", "ALWAYS", "每笔交易都需要人工确认"),
        ("医疗诊断辅助", "ALWAYS", "诊断建议需要医生审核"),
        ("批量数据处理", "NEVER", "无人值守的全自动任务"),
        ("自动化测试", "NEVER", "无需人工干预的测试"),
        ("AI编程助手", "TERMINATE", "AI生成代码，人类最终审核"),
        ("文档助手", "TERMINATE", "AI起草文档，人类审核定稿"),
        ("客服机器人", "TERMINATE", "AI处理常见问题，人工处理复杂问题"),
        ("代码审查", "TERMINATE", "AI初步审查，人类最终确认"),
    ]

    print("\n  | 场景              | 推荐模式    | 原因                       |")
    print("  |-------------------|-------------|----------------------------|")
    for scenario, mode, reason in scenarios:
        print(f"  | {scenario:16} | {mode:11} | {reason:28} |")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有 human_input_mode 演示
    """
    print("=" * 60)
    print("UserProxyAgent 三种 human_input_mode 详解")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_always_mode()
    demo_never_mode()
    demo_terminate_mode()
    demo_max_consecutive_auto_reply_interaction()
    demo_exit_as_termination_signal()
    demo_mode_selection_guide()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n总结：")
    print("  - ALWAYS: 每次都请求人工输入，适合高风险操作")
    print("  - NEVER: 完全自动，适合无人值守任务")
    print("  - TERMINATE: 自动运行直到条件满足，适合大多数场景")


if __name__ == "__main__":
    main()