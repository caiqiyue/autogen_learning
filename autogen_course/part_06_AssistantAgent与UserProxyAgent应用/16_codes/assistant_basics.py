"""
AutoGen AssistantAgent 基础用法

本文件演示 AssistantAgent 的核心能力、默认行为与典型使用场景。

AssistantAgent 是 AutoGen 框架中专门为"AI助手"场景设计的 Agent 子类，
它继承自 ConversableAgent，内置了代码执行能力的集成，使得构建任务执行代理变得非常简单。

主要学习点：
1. AssistantAgent 与 ConversableAgent 的继承关系
2. AssistantAgent 的默认 system_message 配置
3. 代码执行能力的内置集成机制
4. 快速创建 AssistantAgent 的配置模板

运行本文件需要：
- pip install pyautogen
- 设置 OPENAI_API_KEY 环境变量
"""

import os
from typing import Dict, List, Optional

# ============================================================
# 第一部分：AssistantAgent 核心概念
# ============================================================

class AssistantAgentConcept:
    """
    AssistantAgent 核心概念解析

    继承关系：
    ConversableAgent（基类）
        └── AssistantAgent（专门化的AI助手Agent）

    ConversableAgent 的四大核心组件：
    1. LLM（语言模型）
    2. Code Executor（代码执行器）
    3. Tool Executor（工具执行器）
    4. Human Input（人类输入）

    AssistantAgent 的定位：
    - 默认集成了代码执行能力（通过 code_executor）
    - 默认 system_message 包含代码执行相关指令
    - 适合作为"任务执行代理"，帮助用户完成代码编写、数据分析等任务
    """

    @staticmethod
    def explain_inheritance():
        """解释 AssistantAgent 的类继承关系"""
        print("=" * 60)
        print("AssistantAgent 继承关系")
        print("=" * 60)
        print("""
        AssistantAgent 的 MRO（方法解析顺序）：

        1. AssistantAgent  -- 本节主角，专门化的AI助手
           └─ 继承自 ConversableAgent
               └─ 继承自 Agent（基础Agent类）

        2. ConversableAgent 提供了：
           - generate_reply() 策略链机制
           - register_reply() 条件触发注册
           - max_consecutive_auto_reply 轮次控制
           - is_termination_msg 终止条件判断

        3. AssistantAgent 在 ConversableAgent 基础上增加了：
           - 内置 code_executor 配置
           - 默认 system_message 包含代码执行指令
           - 更适合"AI助手"场景的默认行为
        """)

    @staticmethod
    def explain_default_system_message():
        """
        解释 AssistantAgent 的默认 system_message

        AssistantAgent 的默认 system_message 包含以下核心指令：
        1. 代码执行能力：你可以编写和执行代码
        2. 工具使用：你可以使用各种工具完成任务
        3. 角色定位：你是一个专业的AI助手
        """
        print("\n" + "=" * 60)
        print("AssistantAgent 默认 system_message 核心要素")
        print("=" * 60)
        print("""
        默认 system_message 包含的关键指令：

        1. 角色定位：
           "You are a helpful AI assistant."

        2. 代码执行能力：
           - "You can write and execute Python code"
           - 内置的 code_executor 使 LLM 能够生成代码并执行

        3. 工具调用：
           - "You can use tools to help完成任务"
           - 通过 register_function 注册的工具可被调用

        4. 终止条件：
           - 一般由 is_termination_msg 控制何时结束对话
        """)


# ============================================================
# 第二部分：快速创建 AssistantAgent 的配置模板
# ============================================================

class AssistantAgentTemplates:
    """
    快速创建 AssistantAgent 的配置模板集合

    提供多种场景下的配置模板，方便快速构建 Agent
    """

    @staticmethod
    def basic_template():
        """
        最基础的 AssistantAgent 配置模板

        适用场景：简单的单轮对话助手

        最小配置：
        - name: Agent 的名称
        - llm_config: 模型配置（至少需要 model 和 api_key）
        """
        config = {
            "name": "assistant",
            "llm_config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        }
        return config

    @staticmethod
    def code_execution_template():
        """
        带代码执行能力的 AssistantAgent 配置模板

        适用场景：需要执行代码的任务助手（如数据分析、算法验证）

        关键参数：
        - code_executor: 配置代码执行器
          - use_docker: 是否使用 Docker 隔离执行（生产环境建议 True）
          - timeout: 代码执行超时时间
        """
        config = {
            "name": "code_assistant",
            "llm_config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.7,
            },
            "code_executor": {
                "use_docker": False,  # 开发环境设为 False，方便调试
                "timeout": 60,        # 超时时间（秒）
            }
        }
        return config

    @staticmethod
    def research_assistant_template():
        """
        研究助手配置模板

        适用场景：需要联网搜索、分析信息的助手

        特点：
        - 较高 temperature 用于创意分析
        - 可配合 RAG 或搜索工具使用
        """
        config = {
            "name": "research_assistant",
            "llm_config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.8,    # 较高温度，支持创意分析
                "max_tokens": 4096,
            },
            "system_message": """
                你是一个专业的研究助手，擅长：
                1. 信息检索与整理
                2. 数据分析与可视化
                3. 报告撰写
                4. 参考文献管理

                请始终提供准确、客观的信息，并在分析时注明数据来源。
            """
        }
        return config

    @staticmethod
    def legal_advisor_template():
        """
        法律咨询助手配置模板

        适用场景：法律问题咨询、合同审查等

        特点：
        - 明确的专业角色定位
        - 强调免责声明（不是真正的律师）
        """
        config = {
            "name": "legal_advisor",
            "llm_config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.3,    # 较低温度，确保准确性
            },
            "system_message": """
                你是一个专业的法律咨询助手。注意：
                1. 你不是律师，不能提供正式的法律意见
                2. 你的回答仅供参考，不能替代专业法律咨询
                3. 对于复杂法律问题，请建议用户咨询专业律师
                4. 请务必说明你所提供的只是一般性信息

                你的专长领域包括：
                - 合同条款解读
                - 法律流程咨询
                - 权利义务分析
            """
        }
        return config


# ============================================================
# 第三部分：AssistantAgent 与 UserProxyAgent 协作模式
# ============================================================

class AgentCollaboration:
    """
    AssistantAgent 与 UserProxyAgent 的标准协作模式

    在 AutoGen 中，最常见的协作模式是：
    AssistantAgent（AI助手）+ UserProxyAgent（人类代理/代码执行器）

    通信流程：
    UserProxyAgent（用户） -> AssistantAgent（AI助手）
                          <- 响应/代码执行请求
    UserProxyAgent（执行代码） -> AssistantAgent（接收结果）
                                <- 进一步响应
    """

    @staticmethod
    def explain_standard_pattern():
        """解释标准协作模式"""
        print("\n" + "=" * 60)
        print("AssistantAgent + UserProxyAgent 标准模式")
        print("=" * 60)
        print("""
        标准双 Agent 协作架构：

        +------------------+     +------------------+
        |  UserProxyAgent  |     | AssistantAgent   |
        |  (人类代理)       |<--->|  (AI助手)        |
        +------------------+     +------------------+
                 |                        |
                 |                        |
                 v                        v
        +------------------+     +------------------+
        |  代码执行器       |     |  LLM (GPT-4o)   |
        |  (执行代码)       |     +------------------+
        +------------------+

        角色分工：
        - UserProxyAgent: 接收用户输入，执行代码，反馈结果
        - AssistantAgent: 生成响应、编写代码、决策下一步

        典型应用场景：
        1. 数据分析：Assistant 生成代码，UserProxy 执行
        2. 代码调试：Assistant 分析问题，UserProxy 执行测试
        3. 自动化任务：Assistant 规划，UserProxy 执行
        """)

    @staticmethod
    def show_human_input_modes():
        """展示 UserProxyAgent 的三种 human_input_mode"""
        print("\n" + "=" * 60)
        print("UserProxyAgent human_input_mode 三种模式")
        print("=" * 60)
        print("""
        UserProxyAgent 的 human_input_mode 参数控制人类参与方式：

        1. ALWAYS（始终等待人类输入）
           - Agent 每次回复前都会等待人类确认
           - 适用场景：需要人类审批的关键节点
           - 风险：频繁打断可能影响效率

        2. NEVER（完全自动化）
           - Agent 完全自主运行，不需要人类介入
           - 适用场景：批量处理、高度自动化的流程
           - 注意：需要设置合理的终止条件

        3. TERMINATE（智能终止模式）
           - Agent 自动运行，直到满足终止条件
           - 遇到无法判断的情况时请求人类输入
           - 适用场景：大多数标准任务
           - 这是最常用的模式

        代码示例：
        ```python
        from autogen import UserProxyAgent

        # 始终需要人类确认（高安全性场景）
        human_agent = UserProxyAgent("human", human_input_mode="ALWAYS")

        # 完全自动化（批量任务）
        auto_agent = UserProxyAgent("auto", human_input_mode="NEVER")

        # 智能终止（标准模式）
        smart_agent = UserProxyAgent("smart", human_input_mode="TERMINATE")
        ```
        """)


# ============================================================
# 第四部分：配置参数详解
# ============================================================

class ConfigExplainer:
    """
    AssistantAgent 关键配置参数详解
    """

    @staticmethod
    def explain_llm_config():
        """详解 llm_config 参数"""
        print("\n" + "=" * 60)
        print("llm_config 完整参数结构")
        print("=" * 60)
        print("""
        llm_config 是 AssistantAgent 最关键的配置参数：

        必需参数：
        - model: 模型名称（如 "gpt-4o"、"gpt-4o-mini"）
        - api_key: API 密钥（建议使用环境变量）

        可选参数：
        - temperature: 温度参数，控制输出随机性
          * 0.0-0.3：精确任务（代码、翻译）
          * 0.4-0.7：平衡任务（一般对话）
          * 0.8-1.0：创意任务（头脑风暴）

        - max_tokens: 最大生成 token 数
        - top_p: 核采样参数
        - timeout: 请求超时时间
        - max_retries: 最大重试次数
        - cache_seed: 缓存种子（用于实验可重复性）

        扩展参数：
        - fail_safe: 失败安全回调函数
        - functions: 可用工具列表（通过 register_function 注册）

        示例配置：
        ```python
        llm_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 60,
            "max_retries": 3,
        }
        ```
        """)

    @staticmethod
    def explain_code_executor():
        """详解 code_executor 配置"""
        print("\n" + "=" * 60)
        print("code_executor 配置参数")
        print("=" * 60)
        print("""
        code_executor 用于配置代码执行器，使 Agent 能够执行代码。

        配置方式：
        1. use_docker: Docker 隔离执行（推荐生产环境使用）
           - True: 使用 Docker 容器隔离执行（安全）
           - False: 直接在本地执行（方便调试）
           - str: 指定 Docker 镜像名称

        2. timeout: 代码执行超时时间（秒）
           - 建议设置 30-300 秒
           - 过长可能导致资源占用

        3. work_dir: 工作目录
           - 默认使用临时目录
           - 可指定持久化目录用于文件共享

        4. last_n_messages: 回溯消息数量
           - 控制代码执行器可见的历史消息数
           - 节省 token 但可能丢失上下文

        示例：
        ```python
        code_executor={
            "use_docker": False,  # 开发环境
            "timeout": 120,
            "work_dir": "./workspace"
        }
        ```

        生产环境建议使用 Docker：
        ```python
        code_executor={
            "use_docker": True,
            "timeout": 300,
        }
        ```
        """)


# ============================================================
# 第五部分：实战配置示例
# ============================================================

class RealWorldExamples:
    """
    实际业务场景的配置示例
    """

    @staticmethod
    def data_analysis_assistant():
        """
        数据分析助手配置

        场景：用户上传数据，Agent 进行分析和可视化
        """
        config = {
            "agent_name": "data_analyst",
            "system_message": """
                你是一个专业的数据分析师，擅长：
                1. 数据清洗和预处理
                2. 统计分析和趋势发现
                3. 数据可视化（生成图表）
                4. 洞察提炼和报告撰写

                工作流程：
                1. 理解用户的数据分析需求
                2. 编写 Python 代码进行分析
                3. 执行代码并展示结果
                4. 根据结果给出建议
            """,
            "llm_config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.5,  # 中低温度，保证分析准确性
            },
            "code_executor": {
                "use_docker": False,  # 开发环境
                "timeout": 120,
            }
        }
        return config

    @staticmethod
    def code_review_assistant():
        """
        代码审查助手配置

        场景：Agent 帮助审查代码，提出改进建议
        """
        config = {
            "agent_name": "code_reviewer",
            "system_message": """
                你是一个资深的代码审查专家，擅长：
                1. 发现代码中的 bug 和潜在问题
                2. 提出性能优化建议
                3. 检查代码风格和可读性
                4. 提供重构建议

                审查标准：
                - 功能正确性
                - 代码可读性
                - 性能效率
                - 安全漏洞
                - 最佳实践遵循

                注意：
                - 只提供建议，不直接修改代码
                - 解释为什么要这样改
            """,
            "llm_config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.3,  # 低温度，审查需要严谨
            }
        }
        return config

    @staticmethod
    def learning_tutor():
        """
        学习辅导助手配置

        场景：帮助学生学习，解答问题，生成练习题
        """
        config = {
            "agent_name": "learning_tutor",
            "system_message": """
                你是一个耐心的学习辅导老师，擅长：
                1. 解释复杂的概念（用简单的方式）
                2. 回答学科问题
                3. 生成练习题和测验
                4. 提供学习方法和技巧

                教学原则：
                -循序渐进，由浅入深
                - 多用例子和类比
                - 鼓励提问，耐心解答
                - 根据学生水平调整难度

                注意：
                - 不能替代正式的学校教育
                - 对于不确定的问题，诚实告知学生
            """,
            "llm_config": {
                "model": "gpt-4o-mini",  # 成本优化，使用 mini 模型
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.8,  # 较高温度，适合教学互动
            }
        }
        return config


# ============================================================
# 第六部分：AssistantAgent 能力边界
# ============================================================

class CapabilityBoundary:
    """
    AssistantAgent 的能力边界与适用场景分析
    """

    @staticmethod
    def show_capabilities():
        """展示 AssistantAgent 的核心能力"""
        print("\n" + "=" * 60)
        print("AssistantAgent 核心能力")
        print("=" * 60)
        print("""
        AssistantAgent 内置了以下核心能力：

        1. 语言理解与生成
           - 自然语言对话
           - 文本生成与改写
           - 多语言支持

        2. 代码生成与执行
           - Python 代码编写
           - 代码执行与结果反馈
           - 错误诊断与修正

        3. 工具调用
           - 通过 register_function 注册工具
           - Function Calling 集成
           - 工具组合使用

        4. 对话管理
           - 多轮对话上下文维护
           - 终止条件判断
           - 轮次控制
        """)

    @staticmethod
    def show_limitations():
        """展示 AssistantAgent 的局限性"""
        print("\n" + "=" * 60)
        print("AssistantAgent 局限性")
        print("=" * 60)
        print("""
        AssistantAgent 的能力边界：

        1. 模型能力限制
           - 依赖底层 LLM 的能力
           - 无法超越模型本身的知识边界
           - 对于实时信息需要配合搜索工具

        2. 代码执行限制
           - 默认只支持 Python
           - 需要配置 code_executor 才能执行代码
           - 代码执行有超时和资源限制

        3. 状态管理
           - 单 Agent 状态有限
           - 复杂状态需要配合内存或外部存储
           - 多 Agent 协作需要额外设计

        4. 安全性
           - 代码执行可能存在风险（建议使用 Docker）
           - 需要注意 prompt injection 攻击
           - 敏感操作需要人类确认

        适用场景：
        ✓ 代码编写与调试
        ✓ 数据分析与处理
        ✓ 文档生成与编辑
        ✓ 知识问答与解释
        ✓ 任务规划与执行

        不适用场景：
        ✗ 需要其他编程语言（需扩展）
        ✗ 实时性极高的场景（LLM 延迟）
        ✗ 完全离线的严格安全环境（取决于部署方式）
        """)

    @staticmethod
    def decision_framework():
        """展示选择 AssistantAgent 的决策框架"""
        print("\n" + "=" * 60)
        print("选择 AssistantAgent 的决策框架")
        print("=" * 60)
        print("""
        何时使用 AssistantAgent：

        1. 需要 AI 助手功能
           - 需要对话式交互
           - 需要代码执行能力
           - 需要工具调用

        2. 任务导向为主
           - 明确的完成目标
           - 可以分解为步骤
           - 需要结果输出

        3. 人机协作场景
           - 人类提供输入和反馈
           - AI 辅助完成任务
           - 需要人类审批节点

        何时考虑其他方案：

        1. 纯信息检索场景
           - 建议使用 RAG + 简单 Agent

        2. 高度自动化流水线
           - 建议使用 LangChain/LangGraph

        3. 多 Agent 复杂协作
           - 建议使用 GroupChat

        AutoGen 的优势：
        - 对话式 Agent 协作
        - 代码执行内置集成
        - 灵活的人类参与机制
        - 多 Agent 协作支持
        """)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# 第一部分：AssistantAgent 核心概念")
    print("#" * 60)

    AssistantAgentConcept.explain_inheritance()
    AssistantAgentConcept.explain_default_system_message()

    print("\n" + "#" * 60)
    print("# 第二部分：配置模板")
    print("#" * 60)

    templates = AssistantAgentTemplates()
    print("\n基础模板：")
    print(templates.basic_template())

    print("\n代码执行模板：")
    print(templates.code_execution_template())

    print("\n" + "#" * 60)
    print("# 第三部分：协作模式")
    print("#" * 60)

    AgentCollaboration.explain_standard_pattern()
    AgentCollaboration.show_human_input_modes()

    print("\n" + "#" * 60)
    print("# 第四部分：配置参数详解")
    print("#" * 60)

    ConfigExplainer.explain_llm_config()
    ConfigExplainer.explain_code_executor()

    print("\n" + "#" * 60)
    print("# 第五部分：实战配置示例")
    print("#" * 60)

    examples = RealWorldExamples()
    print("\n数据分析助手配置：")
    print(examples.data_analysis_assistant())

    print("\n代码审查助手配置：")
    print(examples.code_review_assistant())

    print("\n学习辅导助手配置：")
    print(examples.learning_tutor())

    print("\n" + "#" * 60)
    print("# 第六部分：能力边界")
    print("#" * 60)

    CapabilityBoundary.show_capabilities()
    CapabilityBoundary.show_limitations()
    CapabilityBoundary.decision_framework()

    print("\n" + "=" * 60)
    print("AssistantAgent 基础用法演示结束")
    print("=" * 60)
    print("""
    学习要点总结：

    1. 继承关系：AssistantAgent 继承自 ConversableAgent
       - ConversableAgent 提供核心对话机制
       - AssistantAgent 增加代码执行集成

    2. 默认行为：
       - 内置 code_executor 配置
       - 默认 system_message 包含代码执行指令
       - 支持 register_function 工具注册

    3. 快速配置模板：
       - 基础模板：name + llm_config
       - 代码执行：增加 code_executor 参数
       - 场景定制：通过 system_message 自定义角色

    4. 典型应用场景：
       - 数据分析助手
       - 代码审查助手
       - 学习辅导助手
       - 研究助理

    下一步：
    - 查看 assistant_scenarios.py 了解不同场景的配置
    - 尝试创建自己的 AssistantAgent 实例
    - 阅读 AutoGen 源码理解 AssistantAgent 实现
    """)