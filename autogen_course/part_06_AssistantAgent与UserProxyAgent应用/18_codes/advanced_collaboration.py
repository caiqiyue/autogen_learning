"""
第18节 - 高级协作模式
====================================
本文件展示 AssistantAgent 与 UserProxyAgent 的高级协作模式

高级主题：
1. 多轮对话与上下文管理
2. 代码执行与工具调用集成
3. 条件终止与复杂工作流设计
4. 多Agent场景下的协作设计
5. 错误处理与恢复机制
"""

import os
import time

# ============================================================
# 第一部分：标准导入与配置
# ============================================================

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# LLM配置 - 在实际使用时需要配置有效的API密钥
llm_config = {
    "model": "gpt-4",
    "api_type": "openai",
    # "api_key": os.getenv("OPENAI_API_KEY"),
}


# ============================================================
# 第二部分：多轮对话与上下文管理
# ============================================================

def example_multi_turn_conversation():
    """
    多轮对话示例：展示如何在多轮对话中保持上下文
    """
    print("=" * 60)
    print("高级模式1：多轮对话与上下文管理")
    print("=" * 60)

    # ---------------------------------------------------------
    # 创建具备专业背景的AssistantAgent
    # ---------------------------------------------------------
    assistant = AssistantAgent(
        name="技术顾问",
        system_message="""你是一位资深技术架构师，专注于系统设计和优化。
        在多轮对话中，你应该：
        1. 记住用户之前的需求和约束条件
        2. 基于上下文提供连贯的建议
        3. 主动询问澄清不确定的问题
        4. 提供分步骤的解决方案

        每次回复后，询问用户是否需要进一步细化。""",
        llm_config=llm_config,
    )

    # 创建UserProxyAgent（自动模式）
    user_proxy = UserProxyAgent(
        name="产品经理",
        human_input_mode="NEVER",
    )

    print("\n[场景] 产品经理与架构师进行多轮技术讨论")

    # ---------------------------------------------------------
    # 第一轮：初始需求
    # ---------------------------------------------------------
    print("\n--- 第1轮对话 ---")
    assistant.initiate_chat(
        recipient=user_proxy,
        message="我需要设计一个日活100万用户的电商系统，请给出架构建议。",
    )

    # ---------------------------------------------------------
    # 第二轮：深入讨论（使用clear_history=False保持上下文）
    # ---------------------------------------------------------
    print("\n--- 第2轮对话 ---")
    # 注意：这里使用不同的recipient进行演示
    # 在实际场景中，可以通过同一个assistant继续对话

    # 如果需要继续同一对话，可以使用 send() 方法
    # assistant.send(recipient=user_proxy, message="关于数据库选型，有什么建议？")


def example_context_preservation():
    """
    上下文保持示例：展示如何在不同对话之间保持状态
    """
    print("\n" + "=" * 60)
    print("上下文保持策略")
    print("=" * 60)

    # 策略1：使用系统消息定义Agent的行为模式
    assistant_with_context = AssistantAgent(
        name="上下文助手",
        system_message="""你是一个任务跟踪助手。
        你会记住：
        1. 用户当前正在完成的任务
        2. 已完成的步骤和剩余的步骤
        3. 用户的偏好和约束

        在每次回复中，简要说明当前进度。""",
        llm_config=llm_config,
    )

    # 策略2：使用显式的状态记录
    task_state = {
        "current_step": 0,
        "total_steps": 5,
        "completed": [],
    }

    print(f"""
    上下文保持的常用策略：

    策略1 - 系统消息持久化
        在system_message中定义Agent的记忆机制
        适用于：固定的角色行为、长期上下文

    策略2 - 外部状态存储
        使用字典、数据库等外部存储保存状态
        适用于：需要跨会话保持状态的场景

    策略3 - 对话摘要
        在每轮对话后生成摘要，保持核心信息
        适用于：长对话场景，减少token消耗

    当前任务状态示例：
        {task_state}
    """)


# ============================================================
# 第三部分：代码执行与工具调用集成
# ============================================================

def example_code_execution_workflow():
    """
    代码执行工作流示例：展示如何集成代码执行功能
    """
    print("\n" + "=" * 60)
    print("高级模式2：代码执行与工具调用")
    print("=" * 60)

    # ---------------------------------------------------------
    # 创建具备代码执行能力的UserProxyAgent
    # ---------------------------------------------------------
    coder_proxy = UserProxyAgent(
        name="代码执行代理",
        human_input_mode="NEVER",  # 自动执行代码
        code_execution_config={
            "work_dir": "coding",  # 代码执行的工作目录
            "use_docker": False,    # 是否使用Docker容器
            # timeout: 代码执行超时时间（秒）
            "timeout": 60,
        },
    )

    # ---------------------------------------------------------
    # 创建编程助手Agent
    # ---------------------------------------------------------
    coding_assistant = AssistantAgent(
        name="Python开发助手",
        system_message="""你是一位专业的Python开发者。
        当需要执行代码时：
        1. 编写清晰、可执行的Python代码
        2. 添加必要的错误处理
        3. 输出结果要格式清晰

        用户要求执行代码时，使用 'print(' TERMINATE') 结束对话。""",
        llm_config=llm_config,
    )

    print("\n[场景] 开发者请求执行数据分析代码")

    # ---------------------------------------------------------
    # 启动协作：助手生成代码 -> 用户代理执行 -> 返回结果
    # ---------------------------------------------------------
    # 在自动模式下，UserProxyAgent会接收助手生成的代码并执行

    # 注意：实际执行需要配置有效的API密钥
    # coding_assistant.initiate_chat(
    #     recipient=coder_proxy,
    #     message="请执行以下Python代码：\nprint('Hello, World!')",
    # )


def example_code_review_workflow():
    """
    代码审查工作流：生成代码 -> 执行 -> 审查结果
    """
    print("\n" + "-" * 60)
    print("代码审查完整工作流")
    print("-" * 60)

    # 1. 代码生成Agent
    code_generator = AssistantAgent(
        name="代码生成器",
        system_message="""你是一个代码生成专家。
        根据用户需求生成高质量Python代码。
        代码要包含：
        1. 完整的函数定义
        2. 类型注解
        3. docstring文档
        4. 必要的异常处理""",
        llm_config=llm_config,
    )

    # 2. 代码执行Agent（用户代理）
    code_executor = UserProxyAgent(
        name="代码执行器",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "review_workspace",
            "use_docker": False,
        },
    )

    # 3. 代码审查Agent
    code_reviewer = AssistantAgent(
        name="代码审查员",
        system_message="""你是一个代码审查专家。
        审查代码并提供改进建议，关注：
        1. 代码质量和可读性
        2. 潜在bug和安全性
        3. 性能优化建议
        4. 最佳实践遵循情况""",
        llm_config=llm_config,
    )

    print("""
    代码审查工作流：

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ 代码生成器  │ ──> │ 代码执行器  │ ──> │ 代码审查员  │
    │ (Assistant)│     │ (UserProxy) │     │ (Assistant) │
    └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          v                   v                   v
       生成代码            执行代码            审查结果
                              │                   │
                              v                   v
                         返回输出            提供建议

    特点：
    - UserProxyAgent负责执行代码，避免助手Agent直接执行风险
    - 审查Agent独立于生成Agent，提供客观评价
    - 可以扩展为自动修复循环
    """)


# ============================================================
# 第四部分：条件终止与复杂工作流
# ============================================================

def example_conditional_termination():
    """
    条件终止示例：展示如何设计智能终止条件
    """
    print("\n" + "=" * 60)
    print("高级模式3：条件终止与复杂工作流")
    print("=" * 60)

    # ---------------------------------------------------------
    # 自定义终止条件函数
    # ---------------------------------------------------------
    def should_terminate(message):
        """
        自定义终止条件判断函数

        参数:
            message: Agent的回复消息

        返回值:
            bool: True表示应该终止对话
        """
        content = message.get("content", "")

        # 条件1：显式终止标记
        if content.rstrip().endswith("TERMINATE"):
            return True

        # 条件2：包含完成标记
        if "任务完成" in content or "完成" in content:
            return True

        # 条件3：错误标记（可以终止或重试）
        if "无法完成" in content or "失败" in content:
            return True

        # 条件4：对话轮次限制（防止无限循环）
        # 注意：需要配合max_round使用
        # if len(chat_history) >= max_turns:
        #     return True

        return False

    # ---------------------------------------------------------
    # 创建带自定义终止条件的Agent
    # ---------------------------------------------------------
    smart_terminator = UserProxyAgent(
        name="智能终止代理",
        human_input_mode="TERMINATE",  # 使用TERMINATE模式
        is_termination_msg=should_terminate,  # 自定义终止条件
    )

    print("""
    智能终止条件设计：

    1. 基于内容的终止
       - 检查消息是否包含特定关键词
       - 如："完成"、"终止"、"TERMINATE"

    2. 基于状态的终止
       - 检查任务是否达到目标
       - 如：结果验证通过、错误恢复成功

    3. 基于轮次的终止
       - 限制最大对话轮次
       - 防止无限循环

    4. 基于质量的终止
       - 检查输出是否满足质量标准
       - 如：代码通过测试、答案足够准确
    """)

    return smart_terminator


# ============================================================
# 第五部分：多Agent协作设计
# ============================================================

def example_multi_agent_collaboration():
    """
    多Agent协作示例：展示如何协调多个Agent的工作
    """
    print("\n" + "=" * 60)
    print("高级模式4：多Agent协作设计")
    print("=" * 60)

    # ---------------------------------------------------------
    # 创建多个专业Agent
    # ---------------------------------------------------------

    # 1. 需求分析师
    requirements_analyst = AssistantAgent(
        name="需求分析师",
        system_message="""你专注于理解和分析用户需求。
        工作方式：
        1. 明确用户目标
        2. 分解需求为具体任务
        3. 识别约束条件和依赖
        4. 输出清晰的需求文档""",
        llm_config=llm_config,
    )

    # 2. 技术架构师
    architect = AssistantAgent(
        name="架构师",
        system_message="""你专注于系统设计和架构。
        工作方式：
        1. 评估技术可行性
        2. 设计系统架构
        3. 选择技术栈
        4. 识别技术风险""",
        llm_config=llm_config,
    )

    # 3. 项目经理（UserProxy）
    pm_proxy = UserProxyAgent(
        name="项目经理",
        human_input_mode="NEVER",
    )

    print("""
    多Agent协作模式：

    模式A - 串行协作（流水线模式）
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ 需求分析师    │ -> │ 架构师       │ -> │ 项目经理     │
    │ 分析需求     │    │ 设计架构     │    │ 协调执行     │
    └──────────────┘    └──────────────┘    └──────────────┘

    模式B - 并行协作（分而治之）
    ┌──────────────┐
    │ 任务分解器   │
    └──────────────┘
          │
    ┌─────┴─────┬────────────┐
    v           v            v
┌────────┐ ┌────────┐ ┌────────┐
│ Agent1 │ │ Agent2 │ │ Agent3 │
│ 子任务1 │ │ 子任务2 │ │ 子任务3 │
└────────┘ └────────┘ └────────┘
    """)

    # ---------------------------------------------------------
    # 使用GroupChat进行多Agent协作
    # ---------------------------------------------------------
    # GroupChat允许多个Agent在同一个群组中交流

    group_chat = GroupChat(
        agents=[
            requirements_analyst,
            architect,
            pm_proxy,
        ],
        messages=[],  # 初始消息列表
        max_round=10,  # 最大对话轮次
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    print("\nGroupChat配置：")
    print(f"  - 参与Agent数量: {len(group_chat.agents)}")
    print(f"  - 最大对话轮次: {group_chat.max_round}")

    return group_chat_manager


# ============================================================
# 第六部分：错误处理与恢复机制
# ============================================================

def example_error_handling():
    """
    错误处理与恢复机制示例
    """
    print("\n" + "=" * 60)
    print("高级模式5：错误处理与恢复机制")
    print("=" * 60)

    # ---------------------------------------------------------
    # 错误处理策略
    # ---------------------------------------------------------

    # 策略1：重试机制
    def retry_with_backoff(max_attempts=3, backoff_factor=2):
        """
        指数退避重试装饰器

        参数:
            max_attempts: 最大尝试次数
            backoff_factor: 退避因子（每次重试等待时间乘以此因子）
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                attempt = 0
                while attempt < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            raise e
                        wait_time = backoff_factor ** attempt
                        print(f"尝试 {attempt} 失败，{wait_time}秒后重试...")
                        time.sleep(wait_time)
            return wrapper
        return decorator

    # 策略2：降级处理
    def graceful_degradation(fallback_value=None):
        """
        降级处理装饰器

        参数:
            fallback_value: 失败时返回的默认值
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"执行失败，使用降级处理: {e}")
                    return fallback_value
            return wrapper
        return decorator

    print("""
    错误处理与恢复策略：

    1. 重试机制 (Retry)
       - 指数退避：每次失败后等待时间指数增长
       - 适用场景：网络请求、临时故障

    2. 降级处理 (Graceful Degradation)
       - 失败时返回默认值或备用方案
       - 适用场景：非关键功能、优雅退出

    3. 超时控制 (Timeout)
       - 设置最大等待时间
       - 超时后终止或切换方案
       - 适用场景：LLM调用、代码执行

    4. 状态回滚 (Rollback)
       - 失败后恢复到之前的状态
       - 适用场景：数据库操作、文件修改

    5. 熔断器模式 (Circuit Breaker)
       - 连续失败达到阈值后暂停调用
       - 定期尝试恢复
       - 适用场景：外部服务调用
    """)


# ============================================================
# 第七部分：协作性能优化
# ============================================================

def example_performance_optimization():
    """
    协作性能优化示例
    """
    print("\n" + "=" * 60)
    print("高级模式6：协作性能优化")
    print("=" * 60)

    # ---------------------------------------------------------
    # 优化策略1：消息压缩
    # ---------------------------------------------------------
    # 对于长对话，使用摘要来减少token消耗

    def summarize_conversation(messages, max_length=500):
        """
        对话摘要函数

        参数:
            messages: 对话消息列表
            max_length: 最大摘要长度

        返回:
            str: 摘要内容
        """
        if not messages:
            return ""

        # 保留首尾消息，压缩中间部分
        if len(messages) <= 3:
            return "\n".join([m.get("content", "")[:200] for m in messages])

        # 策略：保留重要节点
        summary_parts = [
            f"[对话开始] {messages[0].get('content', '')[:200]}",
            f"[中间{len(messages)-2}条消息已省略]",
            f"[对话结束] {messages[-1].get('content', '')[:200]}",
        ]

        return "\n".join(summary_parts)

    # ---------------------------------------------------------
    # 优化策略2：并行处理
    # ---------------------------------------------------------
    # 对于独立任务，并行执行多个Agent

    def parallel_execution(agents, common_task):
        """
        并行执行多个Agent的任务

        参数:
            agents: Agent列表
            common_task: 共同的任务描述

        返回:
            list: 各个Agent的执行结果
        """
        # 注意：实际实现需要异步支持
        results = []
        for agent in agents:
            # 串行执行作为示例
            # 实际可以使用 asyncio.gather 并行执行
            result = agent.initiate_chat(
                recipient=agent,  # 实际场景需要调整
                message=common_task,
            )
            results.append(result)
        return results

    print("""
    协作性能优化策略：

    1. 消息压缩
       - 对长对话进行摘要
       - 保留关键信息，删除冗余
       - 减少token消耗

    2. 并行处理
       - 独立任务并行执行
       - 减少总等待时间
       - 注意资源竞争

    3. 缓存复用
       - 相同请求使用缓存结果
       - 减少重复计算
       - 适用场景：相似查询

    4. 预热策略
       - 提前初始化Agent
       - 减少冷启动时间
       - 适合高延迟场景

    5. 流式输出
       - 使用streaming减少等待感
       - 逐步显示结果
       - 提升用户体验
    """)


# ============================================================
# 第八部分：完整协作工作流示例
# ============================================================

def example_complete_workflow():
    """
    完整协作工作流：从需求到实现的完整流程
    """
    print("\n" + "=" * 60)
    print("完整协作工作流示例")
    print("=" * 60)

    # ---------------------------------------------------------
    # 阶段1：需求分析
    # ---------------------------------------------------------
    print("\n[阶段1] 需求分析")

    analyst = AssistantAgent(
        name="需求分析师",
        system_message="分析用户需求，输出清晰的需求文档",
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="客户",
        human_input_mode="NEVER",
    )

    print("   输入: 用户描述业务需求")
    print("   输出: 结构化需求文档")

    # ---------------------------------------------------------
    # 阶段2：技术设计
    # ---------------------------------------------------------
    print("\n[阶段2] 技术设计")

    designer = AssistantAgent(
        name="技术设计师",
        system_message="基于需求文档，设计技术方案",
        llm_config=llm_config,
    )

    print("   输入: 需求文档")
    print("   输出: 技术设计文档")

    # ---------------------------------------------------------
    # 阶段3：代码实现
    # ---------------------------------------------------------
    print("\n[阶段3] 代码实现")

    coder = AssistantAgent(
        name="开发者",
        system_message="根据设计文档，编写代码",
        llm_config=llm_config,
    )

    executor = UserProxyAgent(
        name="代码执行器",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "project_workspace",
            "use_docker": False,
        },
    )

    print("   输入: 设计文档")
    print("   输出: 可执行代码")

    # ---------------------------------------------------------
    # 阶段4：测试验证
    # ---------------------------------------------------------
    print("\n[阶段4] 测试验证")

    tester = AssistantAgent(
        name="测试工程师",
        system_message="编写和执行测试用例，验证代码质量",
        llm_config=llm_config,
    )

    print("   输入: 代码和需求")
    print("   输出: 测试报告")

    print("""
    完整工作流架构：

    ┌─────────────────────────────────────────────────────────┐
    │                      需求输入                           │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段1: 需求分析 - 需求分析师                             │
    │  • 理解业务目标                                         │
    │  • 分解功能需求                                         │
    │  • 输出需求文档                                         │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段2: 技术设计 - 技术设计师                            │
    │  • 系统架构设计                                         │
    │  • 技术选型决策                                         │
    │  • 输出设计文档                                         │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段3: 代码实现 - 开发者 + 代码执行器                    │
    │  • 编写实现代码                                         │
    │  • 执行验证代码                                         │
    │  • 输出可运行代码                                       │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段4: 测试验证 - 测试工程师                            │
    │  • 单元测试                                             │
    │  • 集成测试                                             │
    │  • 输出测试报告                                         │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      最终交付                           │
    └─────────────────────────────────────────────────────────┘

    关键设计原则：
    1. 每个阶段有明确的输入和输出
    2. Agent之间通过消息传递协作
    3. UserProxyAgent作为执行和验证的桥梁
    4. 支持回滚和迭代优化
    """)


# ============================================================
# 主函数：运行所有示例
# ============================================================

if __name__ == "__main__":
    print("第18节 - AssistantAgent与UserProxyAgent高级协作模式")
    print("=" * 60)

    # 注意：由于没有实际API密钥，下面的示例仅展示代码结构
    # 在实际运行时，需要配置有效的LLM配置

    # 多轮对话示例
    # example_multi_turn_conversation()

    # 上下文保持
    example_context_preservation()

    # 代码执行工作流（需要API密钥）
    # example_code_execution_workflow()

    # 代码审查工作流
    example_code_review_workflow()

    # 条件终止
    example_conditional_termination()

    # 多Agent协作
    # example_multi_agent_collaboration()

    # 错误处理
    example_error_handling()

    # 性能优化
    example_performance_optimization()

    # 完整工作流
    example_complete_workflow()

    print("\n" + "=" * 60)
    print("高级协作模式示例完成")
    print("=" * 60)
    print("""
    总结：

    本节涵盖了AssistantAgent与UserProxyAgent的高级协作模式：

    1. 多轮对话与上下文管理
       - 保持对话上下文连续性
       - 使用系统消息和外部状态

    2. 代码执行与工具调用
       - UserProxyAgent的代码执行能力
       - 完整的代码审查工作流

    3. 条件终止与复杂工作流
       - 自定义终止条件
       - 智能工作流控制

    4. 多Agent协作设计
       - GroupChat多Agent群聊
       - 串行和并行协作模式

    5. 错误处理与恢复
       - 重试机制、降级处理
       - 超时控制和熔断器

    6. 性能优化
       - 消息压缩、并行处理
       - 缓存复用和预热策略

    下一步学习：
    - 深入学习 GroupChat 和 GroupChatManager
    - 探索更复杂的多Agent协作模式
    """)