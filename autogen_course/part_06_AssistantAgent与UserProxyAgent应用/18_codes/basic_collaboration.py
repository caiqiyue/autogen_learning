"""
第18节 - 基础协作模式
====================================
本文件展示 AssistantAgent 与 UserProxyAgent 的基础协作模式

核心概念：
1. AssistantAgent - 负责执行任务和生成回复的智能代理
2. UserProxyAgent - 模拟用户行为，可以自动回复或等待人工输入
3. initiate_chat - 启动对话的方法，是Agent间协作的核心接口
"""

import os
import sys

# ============================================================
# 第一部分：理解 AssistantAgent 与 UserProxyAgent 的角色
# ============================================================

# 在开始协作之前，先了解两个Agent的核心职责：
#
# AssistantAgent（助手代理）：
#   - 接收用户或系统指令
#   - 利用LLM能力生成回复和执行任务
#   - 可以调用代码执行器或工具
#   - 典型用途：编写代码、分析问题、提供建议
#
# UserProxyAgent（用户代理）：
#   - 模拟最终用户的行为
#   - 三种模式：
#     a) human_input_mode="ALWAYS" - 始终等待人工输入
#     b) human_input_mode="NEVER"  - 完全自动回复
#     c) human_input_mode="TERMINATE" - 满足条件时自动回复，否则终止
#   - 典型用途：作为人机交互的桥梁，转发用户指令给AssistantAgent

# ============================================================
# 第二部分：基础协作模式
# ============================================================

# 标准导入
from autogen import AssistantAgent, UserProxyAgent, oai

# 配置LLM（使用环境变量或直接指定）
# 注意：在实际使用中，你需要设置合适的模型和API密钥
llm_config = {
    "model": "gpt-4",
    "api_type": "openai",
    # 以下配置根据实际需要调整
    # "api_key": os.getenv("OPENAI_API_KEY"),
}


def example_basic_collaboration():
    """
    基础协作示例：简单的问答协作

    工作流程：
    1. 创建 AssistantAgent（作为AI助手）
    2. 创建 UserProxyAgent（作为用户界面）
    3. 通过 initiate_chat 启动对话
    4. AssistantAgent 处理请求并回复
    5. UserProxyAgent 决定是否终止或继续
    """
    print("=" * 60)
    print("基础协作模式示例")
    print("=" * 60)

    # ---------------------------------------------------------
    # 步骤1：创建 AssistantAgent
    # ---------------------------------------------------------
    # description: 描述Agent的角色，用于日志和调试
    # llm_config: LLM配置，包含模型参数
    assistant = AssistantAgent(
        name="编程助手",  # Agent名称，在多Agent协作时用于识别
        system_message="""你是一位专业的Python编程助手。
        当用户提出编程问题时，你应该：
        1. 先理解问题需求
        2. 提供清晰的解决方案
        3. 附上完整的代码示例
        4. 解释关键知识点

        回答时使用中文，保持专业且友好的语气。""",
        llm_config=llm_config,
    )

    # ---------------------------------------------------------
    # 步骤2：创建 UserProxyAgent
    # ---------------------------------------------------------
    # human_input_mode: 控制是否等待人工输入
    #   - NEVER: 完全自动，不等待人工输入
    #   - ALWAYS: 始终等待人工确认
    #   - TERMINATE: 当回复包含特定终止信号时自动终止
    #
    # code_execution_config: 启用代码执行功能，允许Agent运行代码
    user_proxy = UserProxyAgent(
        name="用户界面",
        description="模拟用户操作的代理",
        human_input_mode="NEVER",  # 自动模式，无需人工干预
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # 是否使用Docker执行代码
        },
    )

    # ---------------------------------------------------------
    # 步骤3：启动对话 - initiate_chat 方法
    # ---------------------------------------------------------
    # initiate_chat 是 AssistantAgent 的核心方法，用于启动对话
    #
    # 参数说明：
    #   recipient: 接收消息的Agent（这里是user_proxy）
    #   message: 发送的消息内容
    #   clear_history: 是否清除对话历史（默认True）
    #   silent: 是否静默模式（默认False，显示详细信息）
    #
    # 返回值：ChatResult对象，包含对话结果信息

    print("\n[步骤1] 用户发起问询...")
    response = assistant.initiate_chat(
        recipient=user_proxy,  # 指定接收者
        message="请用Python写一个快速排序算法，并解释其工作原理。",
        clear_history=True,  # 清除之前的对话历史
    )

    print("\n[响应结果]")
    print(f"对话完成，聊天结果已返回。")

    return response


# ============================================================
# 第三部分：UserProxyAgent 的三种工作模式
# ============================================================

def example_human_input_modes():
    """
    演示 UserProxyAgent 的三种 human_input_mode
    """
    print("\n" + "=" * 60)
    print("UserProxyAgent 工作模式详解")
    print("=" * 60)

    # ---------------------------------------------------------
    # 模式1: NEVER - 完全自动回复
    # ---------------------------------------------------------
    # 适用于：自动化任务、不需要人工干预的场景
    #
    # 特点：
    #   - Agent会自动生成回复，无需等待人工输入
    #   - 适合批量处理、无人值守的任务
    #   - 响应速度快，但缺乏人工审核

    auto_agent = UserProxyAgent(
        name="自动模式代理",
        human_input_mode="NEVER",
        code_execution_config=False,  # 此模式不需要代码执行
    )

    # ---------------------------------------------------------
    # 模式2: ALWAYS - 始终等待人工输入
    # ---------------------------------------------------------
    # 适用于：需要人工确认的关键操作、安全敏感的的场景
    #
    # 特点：
    #   - 每次回复前都会暂停，等待人工确认
    #   - 可以审核、修改或拒绝AI的建议
    #   - 适合高风险操作或需要人工监督的场景

    manual_agent = UserProxyAgent(
        name="人工确认模式代理",
        human_input_mode="ALWAYS",
    )

    # ---------------------------------------------------------
    # 模式3: TERMINATE - 条件触发自动回复
    # ---------------------------------------------------------
    # 适用于：需要人工介入但不需全程监督的场景
    #
    # 特点：
    #   - 当回复包含终止关键词时自动完成
    #   - 否则持续对话直到满足终止条件
    #   - is_termination_msg 方法决定何时终止

    terminate_agent = UserProxyAgent(
        name="终止条件模式代理",
        human_input_mode="TERMINATE",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )

    print("\n三种模式已创建:")
    print("  1. NEVER   - 完全自动，不等待人工")
    print("  2. ALWAYS  - 始终等待人工确认")
    print("  3. TERMINATE - 满足条件时自动终止")


# ============================================================
# 第四部分：双向协作示例
# ============================================================

def example_two_way_collaboration():
    """
    双向协作示例：UserProxyAgent 接收 AssistantAgent 的回复后自动处理
    """
    print("\n" + "=" * 60)
    print("双向协作模式示例")
    print("=" * 60)

    # 创建助手Agent
    assistant = AssistantAgent(
        name="分析助手",
        system_message="""你是一个数据分析专家。
        用户提供数据描述时，你应该：
        1. 分析数据特征
        2. 提出分析方法
        3. 给出建议的可视化方案
        回复结尾请加上"请确认是否继续"来提示用户。""",
        llm_config=llm_config,
    )

    # 创建用户代理（自动模式）
    user_proxy = UserProxyAgent(
        name="数据用户",
        human_input_mode="NEVER",  # 自动模式
    )

    print("\n[场景] 用户提交数据描述，助手分析并建议")

    # 启动协作
    assistant.initiate_chat(
        recipient=user_proxy,
        message="我有一个销售数据集，包含产品类别、地区、月销量。请问应该如何分析？",
    )


# ============================================================
# 第五部分：消息传递机制详解
# ============================================================

def explain_message_flow():
    """
    解释 initiate_chat 的消息传递流程
    """
    print("\n" + "=" * 60)
    print("initiate_chat 消息传递流程")
    print("=" * 60)

    print("""
    initiate_chat 方法的标准流程：

    1. 初始化 (initiate)
       用户/系统 --> initiate_chat() --> 创建对话上下文

    2. 消息发送 (Send)
       发送方Agent --> 生成消息 --> 传递给接收方Agent

    3. 接收处理 (Receive)
       接收方Agent --> 解析消息 --> 调用LLM生成回复

    4. 回复返回 (Response)
       接收方Agent --> 生成回复 --> 返回给发送方

    5. 状态更新 (Update)
       对话历史更新 --> 检查终止条件 --> 决定是否继续

    关键参数说明：
    ┌─────────────────┬──────────────────────────────────┐
    │ 参数            │ 说明                             │
    ├─────────────────┼──────────────────────────────────┤
    │ recipient       │ 消息接收方Agent                  │
    │ message         │ 发送的消息内容                    │
    │ clear_history   │ 是否清除历史（默认True）          │
    │ silent          │ 是否静默模式                      │
    └─────────────────┴──────────────────────────────────┘

    返回值 ChatResult 包含：
    │- summary: 对话摘要
    │- chat_history: 完整对话历史
    │- cost: 消耗的token和成本
    """)


# ============================================================
# 主函数：运行所有示例
# ============================================================

if __name__ == "__main__":
    print("第18节 - AssistantAgent与UserProxyAgent基础协作模式")
    print("=" * 60)

    # 注意：由于没有实际API密钥，下面的示例仅展示代码结构
    # 在实际运行时，需要配置有效的LLM配置

    # 基础协作示例（需要配置llm_config）
    # example_basic_collaboration()

    # 演示工作模式
    example_human_input_modes()

    # 双向协作示例（需要配置llm_config）
    # example_two_way_collaboration()

    # 消息传递机制说明
    explain_message_flow()

    print("\n" + "=" * 60)
    print("基础协作模式示例完成")
    print("=" * 60)
    print("""
    下一步学习：
    - 查看 advanced_collaboration.py 了解高级协作模式
    - 学习 GroupChat 进行多Agent协作
    """)