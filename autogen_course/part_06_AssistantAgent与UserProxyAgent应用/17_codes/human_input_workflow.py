# human_input_workflow.py
# 第17节 UserProxyAgent三种human_input_mode详解 - 人机协作工作流配置
#
# 本文件演示人机协作工作流的配置，包括：
# 1. UserProxyAgent与Code Executor的集成方式
# 2. UserProxyAgent作为人类代表在GroupChat中的特殊角色
# 3. 人机协作工作流的配置
# 4. 人类输入'exit'作为强制终止信号的源码实现
#
# ============================================================
# 人机协作工作流概述
# ============================================================
#
# 人机协作工作流是 AutoGen 的核心应用场景之一。
# 通过合理配置 UserProxyAgent，可以实现：
# - 人类监督但不干预的半自动流程
# - 人类审批关键步骤的增强自动化流程
# - 人类完全控制的精确执行流程
#
# 关键组件：
# 1. UserProxyAgent - 人类代理，负责人类输入和审批
# 2. AssistantAgent - AI 助手，负责生成和建议
# 3. Code Executor - 代码执行器，安全执行 AI 生成的代码
# 4. GroupChat - 多Agent协作，协调多个Agent的工作
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
# 第三部分：UserProxyAgent 与 Code Executor 集成
# ============================================================
#
# UserProxyAgent 可以与 Code Executor 集成，实现：
# 1. AI 生成代码后自动执行
# 2. 人类监督执行过程
# 3. 执行完成后人类确认结果
#
# 集成方式：
# - code_execution_config: 配置代码执行参数
# - human_input_mode: 控制何时需要人类输入
# - max_consecutive_auto_reply: 控制自动执行次数
#
# ============================================================


def demo_userproxy_with_code_executor():
    """
    演示 UserProxyAgent 与 Code Executor 的集成

    UserProxyAgent 可以通过 code_execution_config 启用代码执行功能：
    1. 当 AI Agent 需要执行代码时，UserProxyAgent 会自动调用 Code Executor
    2. 代码执行完成后，结果返回给 AI Agent 进行下一步处理
    3. 如果设置了 human_input_mode="TERMINATE"，人类可以监督执行过程

    这种集成特别适合：
    - AI 编程助手：AI 生成代码并执行验证
    - 数据分析助手：AI 生成分析代码并执行
    - 自动化测试：AI 生成测试用例并执行
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示1: UserProxyAgent 与 Code Executor 集成")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建带有代码执行功能的 UserProxyAgent
    # ---------------------------------------------

    print("\n配置 UserProxyAgent 与 Code Executor 集成:")

    # 代码执行配置
    code_execution_config = {
        "work_dir": "./code_workspace",  # 代码执行的工作目录
        "use_docker": False,  # 开发环境不使用 Docker
        "timeout": 60,  # 超时时间 60 秒
        "last_n_messages": 6,  # 错误时参考最近 6 条消息
    }

    # 创建 UserProxyAgent，启用代码执行
    user_proxy_with_code = UserProxyAgent(
        name="代码执行代理",
        system_message="""你是人类用户的代理，负责代码执行监督。
当你收到代码执行请求时：
1. 检查代码安全性
2. 执行代码
3. 将执行结果反馈给用户

如果需要人工确认，输入 'confirm' 来确认执行。
如果想终止，输入 'exit'。""",
        # 关键配置：启用代码执行
        code_execution_config=code_execution_config,
        # human_input_mode="TERMINATE" 允许 AI 自动执行，但人类可以监督
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=3,  # 连续3次自动回复后请求确认
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    # 创建 AssistantAgent
    assistant = ConversableAgent(
        name="AI编程助手",
        system_message="""你是一个 AI 编程助手。
你会根据用户需求生成 Python 代码，并请求执行验证。
生成的代码应该简洁、高效、安全。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"\n  - Agent 名称: {user_proxy_with_code.name}")
    print(f"  - human_input_mode: {user_proxy_with_code.human_input_mode}")
    print(f"  - max_consecutive_auto_reply: {user_proxy_with_code.max_consecutive_auto_reply}")
    print(f"  - code_execution_config.work_dir: {code_execution_config['work_dir']}")
    print(f"  - code_execution_config.use_docker: {code_execution_config['use_docker']}")
    print(f"  - code_execution_config.timeout: {code_execution_config['timeout']}")

    # ---------------------------------------------
    # 展示集成后的工作流程
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("UserProxyAgent 与 Code Executor 集成工作流程:")
    print("-" * 40)
    print("""
    工作流程：

    1. [用户] 发送任务请求
           ↓
    2. [AssistantAgent] 接收请求，生成代码
           ↓
    3. [UserProxyAgent] 接收代码执行请求
           ↓
    4. [Code Executor] 在隔离环境中执行代码
           ↓
    5. [UserProxyAgent] 收集执行结果
           ↓
    6. [AssistantAgent] 接收结果，继续处理
           ↓
    7. 重复步骤 2-6 直到任务完成

    人类监督点：
    - human_input_mode="TERMINATE" 时，每 max_consecutive_auto_reply 次请求一次确认
    - 可以通过输入 'exit' 随时终止
    """)

    # ---------------------------------------------
    # 配置建议
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("代码执行配置建议:")
    print("-" * 40)
    print("  | 场景              | use_docker | timeout | 说明               |")
    print("  |-------------------|------------|---------|---------------------|")
    print("  | 开发调试          | False      | 30-60   | 快速反馈，不使用容器|")
    print("  | 生产环境          | True       | 120+    | 安全隔离            |")
    print("  | 数据分析          | True       | 300     | 处理大数据需要更长时间|")
    print("  | 代码审查          | True       | 30      | 快速检查            |")


def demo_userproxy_in_groupchat():
    """
    演示 UserProxyAgent 作为人类代表在 GroupChat 中的特殊角色

    在 GroupChat 场景中，UserProxyAgent 可以代表人类用户参与讨论：
    1. 接收 GroupChat 中其他 Agent 的消息
    2. 根据 human_input_mode 决定何时需要人类输入
    3. 将人类的反馈传递给 GroupChat

    UserProxyAgent 在 GroupChat 中的特殊角色：
    1. 人类代表：作为人类用户与多个 AI Agent 交互的桥梁
    2. 监督者：监督 AI Agent 之间的协作过程
    3. 决策者：在关键时刻提供人类决策
    """
    from autogen import UserProxyAgent, ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("演示2: UserProxyAgent 在 GroupChat 中的特殊角色")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建参与 GroupChat 的 Agent
    # ---------------------------------------------

    # UserProxyAgent - 代表人类用户
    human_proxy = UserProxyAgent(
        name="人类代表",
        system_message="""你是人类用户的代表，参与团队讨论。
你会根据人类用户的需求发表意见，并在必要时请求人工确认。
当任务完成或用户要求终止时，说 'exit'。""",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    # AI Agent 1 - 程序员
    coder = ConversableAgent(
        name="程序员",
        system_message="""你是团队中的程序员，负责编写代码。
你会根据需求提供代码实现，并与其他团队成员协作。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # AI Agent 2 - 审查员
    reviewer = ConversableAgent(
        name="审查员",
        system_message="""你是团队中的代码审查员，负责审查代码质量。
你会检查代码并提出改进建议。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\nGroupChat 参与者:")
    print("  - 人类代表 (UserProxyAgent): 代表人类用户，参与决策")
    print("  - 程序员 (ConversableAgent): AI 程序员，生成代码")
    print("  - 审查员 (ConversableAgent): AI 审查员，审查代码")

    # ---------------------------------------------
    # 创建 GroupChat
    # ---------------------------------------------

    groupchat = GroupChat(
        agents=[human_proxy, coder, reviewer],
        messages=[],
        max_round=10,
        speaker_selection_method="auto",  # LLM 自动选择下一个发言者
        allow_repeat="never",  # 不允许连续发言
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    print(f"\nGroupChat 配置:")
    print(f"  - 参与者数量: {len(groupchat.agents)}")
    print(f"  - max_round: {groupchat.max_round}")
    print(f"  - speaker_selection_method: {groupchat.speaker_selection_method}")
    print(f"  - allow_repeat: {groupchat.allow_repeat}")

    # ---------------------------------------------
    # 展示 UserProxyAgent 在 GroupChat 中的行为
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("UserProxyAgent 在 GroupChat 中的特殊行为:")
    print("-" * 40)
    print("""
    1. 接收消息：
       - GroupChatManager 将其他 Agent 的消息转发给 UserProxyAgent
       - UserProxyAgent 根据 human_input_mode 决定如何响应

    2. 人类输入处理：
       - ALWAYS 模式：每次收到消息都请求人类输入
       - TERMINATE 模式：自动处理，达到限制后请求确认
       - NEVER 模式：自动回复，不请求人类输入

    3. 消息传递：
       - UserProxyAgent 的回复会被添加到 GroupChat 消息历史
       - 其他 Agent 可以看到 UserProxyAgent 的回复（人类反馈）

    4. 终止控制：
       - UserProxyAgent 可以通过 'exit' 信号终止 GroupChat
       - 也可以通过 is_termination_msg 检测来触发终止
    """)

    print("\n  注意：由于 TERMINATE 模式需要人类输入，这里只展示配置")
    print("  实际运行需要人类在适当时候提供输入")


def demo_human_workflow_patterns():
    """
    演示常见的人机协作工作流模式

    模式1：监督模式（Supervisor Pattern）
    - AI 自动执行，人类监督
    - 适合：AI 编程助手、文档助手

    模式2：审批模式（Approval Pattern）
    - AI 提议，人类审批
    - 适合：金融交易、内容发布

    模式3：协作模式（Collaboration Pattern）
    - 人类和 AI 协作完成任务
    - 适合：复杂问题解决、创意工作
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示3: 人机协作工作流模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 模式1：监督模式
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("模式1: 监督模式 (Supervisor Pattern)")
    print("-" * 40)
    print("  特点:")
    print("    - AI 自主执行大部分任务")
    print("    - 人类监督但不干预")
    print("    - 达到限制或异常时请求人类介入")
    print("  适用场景:")
    print("    - AI 编程助手")
    print("    - 文档自动化助手")
    print("    - 数据分析助手")

    supervisor_proxy = UserProxyAgent(
        name="监督代理",
        system_message="你是监督代理，让 AI 自主工作，只在必要时干预。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,  # 允许 AI 连续执行 10 次
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    assistant = ConversableAgent(
        name="AI助手",
        system_message="你是一个自主工作的 AI 助手，可以自动执行任务。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"\n  配置:")
    print(f"    - human_input_mode: TERMINATE")
    print(f"    - max_consecutive_auto_reply: 10")
    print(f"    - 效果: AI 可以连续执行 10 次后才请求确认")

    # ---------------------------------------------
    # 模式2：审批模式
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("模式2: 审批模式 (Approval Pattern)")
    print("-" * 40)
    print("  特点:")
    print("    - AI 生成提议或建议")
    print("    - 人类审批后执行")
    print("    - 关键步骤需要人工确认")
    print("  适用场景:")
    print("    - 金融交易审批")
    print("    - 内容发布审批")
    print("    - 重要决策确认")

    approver_proxy = UserProxyAgent(
        name="审批代理",
        system_message="你是审批代理，审核 AI 的提议并决定是否批准。",
        human_input_mode="ALWAYS",  # 每次都请求人工审批
        max_consecutive_auto_reply=0,
        llm_config=llm_config,
    )

    advisor = ConversableAgent(
        name="AI顾问",
        system_message="你是一个 AI 顾问，提供建议和方案供人类审批。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"\n  配置:")
    print(f"    - human_input_mode: ALWAYS")
    print(f"    - max_consecutive_auto_reply: 0")
    print(f"    - 效果: AI 每次提议都需要人类审批")

    # ---------------------------------------------
    # 模式3：协作模式
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("模式3: 协作模式 (Collaboration Pattern)")
    print("-" * 40)
    print("  特点:")
    print("    - 人类和 AI 共同完成任务")
    print("    - 人类提供领域知识，AI 提供分析能力")
    print("    - 交替互动，协作完成")
    print("  适用场景:")
    print("    - 复杂问题分析")
    print("    - 创意工作协作")
    print("    - 研究探索")

    collaborator_proxy = UserProxyAgent(
        name="协作者",
        system_message="你是协作者，与 AI 一起工作，分享你的观点和反馈。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=3,  # 中等频率交互
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    collaborator_ai = ConversableAgent(
        name="AI协作者",
        system_message="你是一个协作型 AI，与人类一起工作，提出想法并倾听人类反馈。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"\n  配置:")
    print(f"    - human_input_mode: TERMINATE")
    print(f"    - max_consecutive_auto_reply: 3")
    print(f"    - 效果: 每 3 次 AI 回复后请求人类反馈")

    # ---------------------------------------------
    # 模式选择指南
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("人机协作模式选择指南")
    print("=" * 60)
    print("\n  | 模式      | human_input_mode | max_auto_reply | 适用场景          |")
    print("  |-----------|------------------|----------------|-------------------|")
    print("  | 监督模式  | TERMINATE        | 5-10           | AI编程助手        |")
    print("  | 审批模式  | ALWAYS           | 0              | 金融交易审批      |")
    print("  | 协作模式  | TERMINATE        | 3-5            | 复杂问题分析      |")


def demo_exit_signal_implementation():
    """
    演示 'exit' 作为强制终止信号的源码实现

    在 AutoGen 源码中，'exit' 信号的处理流程如下：

    1. 人类输入触发：
       ```python
       # 在 UserProxyAgent 中
       def get_human_input(self, prompt: str) -> str:
           reply = input(prompt)  # 获取人类输入
           return reply
       ```

    2. 信号检测：
       ```python
       # 在 generate_reply 或相关方法中
       def _check_termination(self, message):
           if self.is_termination_msg(message):
               return True
           return False
       ```

    3. 终止处理：
       ```python
       # 当检测到终止信号时
       if self.human_input_mode == "TERMINATE":
           if self._check_termination(message):
               # 请求人类确认是否真的要终止
               user_input = self.get_human_input("确定要终止吗？(yes/exit): ")
               if user_input.lower() in ["yes", "exit", "y"]:
                   # 终止对话
                   self._stop_reply()
       ```

    关键点：
    1. 'exit' 是约定俗成的终止信号
    2. is_termination_msg 函数检查消息内容
    3. 可以自定义终止关键词
    4. 终止前通常会请求确认
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示4: 'exit' 终止信号的源码实现")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建演示用的 Agent
    # ---------------------------------------------

    user_proxy = UserProxyAgent(
        name="用户代理",
        system_message="""你是用户代理。
当你想终止对话时，输入 'exit'。
AI 会检测到这个信号并终止对话。""",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        # 关键：检测 'exit' 作为终止信号
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    assistant = ConversableAgent(
        name="AI助手",
        system_message="""你是一个友好的 AI 助手。
当你认为任务完成时，说 'exit' 来结束对话。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        # AI 端也检测 'exit'
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    print("\n配置说明:")
    print("  UserProxyAgent:")
    print("    - is_termination_msg: 检测 'exit'")
    print("    - 当检测到 'exit' 时，对话终止")
    print("  ")
    print("  AssistantAgent:")
    print("    - is_termination_msg: 检测 'exit'")
    print("    - AI 说 'exit' 时也会触发终止")

    # ---------------------------------------------
    # 展示源码层面的实现逻辑
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("'exit' 终止信号的源码实现流程:")
    print("-" * 40)
    print("""
    1. 人类输入阶段
       │
       ▼
    ┌─────────────────────────────────────┐
    │  UserProxyAgent.get_human_input()   │
    │  - 等待用户输入                      │
    │  - 返回用户输入内容                  │
    └─────────────────────────────────────┘
       │
       ▼
    2. 信号检测阶段
       │
       ▼
    ┌─────────────────────────────────────┐
    │  is_termination_msg(message)       │
    │  - 检查消息内容是否包含 'exit'       │
    │  - 返回 True/False                  │
    └─────────────────────────────────────┘
       │
       ▼
    3. 终止决策阶段
       │
       ▼
    ┌─────────────────────────────────────┐
    │  generate_reply() 中的终止检查       │
    │  - 如果返回 True，请求确认           │
    │  - 用户确认后，设置终止标志          │
    │  - 对话优雅终止                      │
    └─────────────────────────────────────┘
    """)

    # ---------------------------------------------
    # 自定义终止关键词
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("自定义终止关键词:")
    print("-" * 40)

    # 可以自定义终止关键词检测
    def custom_termination(msg):
        """自定义终止条件：检测多个退出关键词"""
        content = msg.get("content", "").lower()
        exit_keywords = ["exit", "quit", "bye", "再见", "退出"]
        return any(kw in content for kw in exit_keywords)

    user_proxy_custom = UserProxyAgent(
        name="自定义终止代理",
        system_message="你是用户代理，支持多种退出命令。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        # 自定义终止条件函数
        is_termination_msg=custom_termination,
    )

    print("  自定义终止关键词:")
    print("    - 'exit' - 标准退出命令")
    print("    - 'quit' - 退出命令")
    print("    - 'bye' - 再见")
    print("    - '再见' - 中文退出")
    print("    - '退出' - 中文退出")

    print("\n  实现方式:")
    print("    ```python")
    print("    def custom_termination(msg):")
    print("        content = msg.get('content', '').lower()")
    print("        exit_keywords = ['exit', 'quit', 'bye', '再见', '退出']")
    print("        return any(kw in content for kw in exit_keywords)")
    print("    ```")


def demo_termination_in_source_code():
    """
    演示终止条件在源码中的实现细节

    AutoGen 的终止条件处理涉及多个组件：
    1. is_termination_msg - Agent 级别的终止检测
    2. max_consecutive_auto_reply - 连续自动回复次数限制
    3. human_input_mode - 人类输入模式
    4. termination_msg - GroupChat 级别的终止检测

    这些组件在源码中的协作流程：
    1. 在 generate_reply() 中检查终止条件
    2. 如果 is_termination_msg 返回 True，等待人类确认
    3. 如果达到 max_consecutive_auto_reply，请求人类输入
    4. 根据 human_input_mode 决定是否需要人类输入
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示5: 终止条件在源码中的实现细节")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建一个带详细配置的 Agent
    # ---------------------------------------------

    user_proxy = UserProxyAgent(
        name="终止演示代理",
        system_message="你是终止演示代理，用于演示终止条件的工作原理。",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=3,
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    assistant = ConversableAgent(
        name="AI助手",
        system_message="你是一个 AI 助手，简短回复。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print("\n配置:")
    print(f"  - human_input_mode: TERMINATE")
    print(f"  - max_consecutive_auto_reply: 3")
    print(f"  - is_termination_msg: 检测 'exit'")

    # ---------------------------------------------
    # 展示源码中的终止流程
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("generate_reply() 中的终止处理流程:")
    print("-" * 40)
    print("""
    源码流程（简化版）：

    def generate_reply(self, messages, ...):
        # 1. 检查终止条件
        if self.is_termination_msg(messages[-1]):
            # 返回终止响应，等待确认
            return self._generate_termination_response()

        # 2. 检查 max_consecutive_auto_reply
        if self._consecutive_auto_replies >= self.max_consecutive_auto_reply:
            # 需要人类输入
            if self.human_input_mode == "TERMINATE":
                return self.get_human_input("需要确认...")

        # 3. 生成自动回复
        response = self._generate_auto_reply(messages)
        self._consecutive_auto_replies += 1
        return response

    终止条件优先级：
    1. is_termination_msg（最高）- 检测到终止关键词立即处理
    2. max_consecutive_auto_reply（中等）- 达到次数限制后请求确认
    3. human_input_mode（基础）- 根据模式决定是否需要人类输入
    """)

    # ---------------------------------------------
    # 实际执行演示
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("实际执行演示:")
    print("-" * 40)
    print("  注意：以下代码会发起对话，但由于 TERMINATE 模式需要人类输入，")
    print("  实际运行时可能会在达到 max_consecutive_auto_reply=3 后暂停。")

    # 注意：由于实际运行需要人类输入，这里只展示配置
    # result = user_proxy.initiate_chat(
    #     assistant,
    #     message="请介绍一下你自己，然后说 'exit' 结束对话。",
    # )


def demo_complete_workflow_example():
    """
    演示一个完整的人机协作工作流示例

    场景：AI 编程助手
    - 用户提出编程需求
    - AI 分析需求并生成代码
    - Code Executor 执行代码验证
    - AI 根据执行结果修改代码
    - 人类在关键节点确认
    """
    from autogen import UserProxyAgent, ConversableAgent

    print("\n" + "=" * 60)
    print("演示6: 完整人机协作工作流示例 - AI 编程助手")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 创建 AI 编程助手
    # ---------------------------------------------

    print("\n步骤1: 创建 Agent")

    # 用户代理 - 带代码执行功能
    user_proxy = UserProxyAgent(
        name="用户代理",
        system_message="""你是人类用户的代理。
你会：
1. 接收用户的编程需求
2. 监督 AI 编程助手的工作
3. 在必要时提供人工确认

当任务完成或用户要求终止时，输入 'exit'。""",
        # 启用代码执行
        code_execution_config={
            "work_dir": "./code_workspace",
            "use_docker": False,
            "timeout": 60,
        },
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
    )

    # AI 编程助手
    coding_assistant = ConversableAgent(
        name="AI编程助手",
        system_message="""你是一个 AI 编程助手。
你会：
1. 理解用户的编程需求
2. 生成 Python 代码实现需求
3. 解释代码逻辑

当你认为代码正确时，说明完成。
如果需要执行验证，请求 UserProxyAgent 执行代码。""",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    print(f"  - 用户代理: {user_proxy.name}")
    print(f"    * code_execution_config: 已启用")
    print(f"    * human_input_mode: TERMINATE")
    print(f"  - AI编程助手: {coding_assistant.name}")

    # ---------------------------------------------
    # 展示工作流
    # ---------------------------------------------

    print("\n" + "-" * 40)
    print("AI 编程助手工作流:")
    print("-" * 40)
    print("""
    工作流程：

    1. [用户] 提出编程需求
           ↓
    2. [AI编程助手] 分析需求，生成代码
           ↓
    3. [AI编程助手] 请求执行代码验证
           ↓
    4. [用户代理] 接收代码执行请求
           ↓
    5. [Code Executor] 执行代码
           ↓
    6. [用户代理] 收集结果，返回给 AI
           ↓
    7. [AI编程助手] 根据结果修改代码
           ↓
    8. 重复 2-7，直到代码正确
           ↓
    9. [用户] 说 'exit' 终止对话

    人类监督点：
    - 每 max_consecutive_auto_reply=5 次自动回复后请求确认
    - 可以通过输入 'exit' 随时终止
    - 可以要求 AI 解释或修改代码
    """)

    # ---------------------------------------------
    # 总结
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("人机协作工作流配置总结")
    print("=" * 60)
    print("""
    关键配置：

    1. UserProxyAgent 配置
       - human_input_mode: "TERMINATE"（推荐）
       - max_consecutive_auto_reply: 3-5（标准任务）
       - is_termination_msg: 检测 'exit'

    2. Code Executor 配置
       - work_dir: 代码执行目录
       - use_docker: True（生产环境）
       - timeout: 60-120（根据任务复杂度）

    3. AssistantAgent 配置
       - human_input_mode: "NEVER"
       - is_termination_msg: 根据需要设置

    最佳实践：
    - 开发环境：use_docker=False, timeout=30
    - 生产环境：use_docker=True, timeout=120
    - 复杂任务：增加 max_consecutive_auto_reply
    """)


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有人机协作工作流演示
    """
    print("=" * 60)
    print("UserProxyAgent 人机协作工作流配置演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_userproxy_with_code_executor()
    demo_userproxy_in_groupchat()
    demo_human_workflow_patterns()
    demo_exit_signal_implementation()
    demo_termination_in_source_code()
    demo_complete_workflow_example()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n总结：")
    print("  1. UserProxyAgent 与 Code Executor 集成实现自动代码执行")
    print("  2. UserProxyAgent 在 GroupChat 中代表人类参与协作")
    print("  3. 三种工作流模式：监督模式、审批模式、协作模式")
    print("  4. 'exit' 是约定的终止信号，可自定义检测函数")
    print("  5. 终止条件在源码中通过 is_termination_msg 和")
    print("     max_consecutive_auto_reply 协同工作")


if __name__ == "__main__":
    main()