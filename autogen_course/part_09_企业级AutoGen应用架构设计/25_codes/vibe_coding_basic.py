#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第25节 Vibe Coding基础工作流示例
=====================================

Vibe Coding 理念：
通过AI Agent协作，让开发者专注于创意和设计，
而将重复性的编码、调试、测试等工作交给AI Agent处理。

本示例展示最基础的Vibe Coding工作流：
1. AssistantAgent - 负责任务分析、代码生成
2. UserProxyAgent - 负责代码执行、结果验证
3. ReviewAgent   - 负责代码审查、提出改进建议

这种三角协作模式是Vibe Coding的核心范式。
"""

import os
import sys
import json
from pathlib import Path

# ============================================================
# 添加项目根目录到Python路径，确保可以导入autogen相关模块
# ============================================================
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError as e:
    print(f"❌ 请先安装 autogen 包: pip install autogen")
    print(f"   导入错误: {e}")
    sys.exit(1)

# ============================================================
# 配置说明
# ============================================================
# 设置日志，方便调试和追踪AI Agent的思考过程
# o1模型使用宽松的输出限制，避免截断关键信息

llm_config = {
    "model": "o1",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,  # 适度创意性，允许AI在解决方案上有一定灵活性
}

# ============================================================
# 第一步：创建Vibe Coding三角架构Agent
# ============================================================

def create_vibe_coding_agents():
    """
    创建Vibe Coding工作流的三个核心Agent

    三角架构说明：
    ┌─────────────┐
    │   User      │  (人类开发者，描述需求)
    │  (需求方)   │
    └──────┬──────┘
           │  描述任务
           ▼
    ┌─────────────┐      生成代码      ┌─────────────┐
    │  Assistant   │ ────────────────▶ │   Review    │
    │  (代码生成)   │                   │  (代码审查)  │
    └──────┬───────┘                   └──────┬───────┘
           │                                 │
           │  反馈修正                       │ 审查结果
           │◀────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │   修正后代码 │
    └─────────────┘

    这是一个迭代式协作模式，直到代码达到质量标准。
    """

    # ---- 1. AssistantAgent: 代码生成专家 ----
    # 角色：接收任务描述，生成高质量Python代码
    # 特点：深入理解业务需求，输出可直接运行的代码
    assistant = AssistantAgent(
        name="代码生成专家",
        system_message="""你是一位经验丰富的Python开发专家，专注于生成高质量、可维护的代码。

        你的工作流程：
        1. 仔细分析用户提出的编程任务
        2. 设计清晰、可行的解决方案
        3. 编写符合PEP8规范的Python代码
        4. 在代码中添加详细的注释和文档字符串

        输出要求：
        - 代码必须完整、可直接运行
        - 必须包含输入/输出示例
        - 复杂逻辑必须添加中文注释
        - 提供使用说明和注意事项

        记住：你的代码会被审查Agent检查，所以要确保代码质量。""",
        llm_config=llm_config,
    )

    # ---- 2. ReviewAgent: 代码审查专家 ----
    # 角色：审查生成的代码，提出改进建议
    # 特点：严格把关代码质量，发现潜在问题
    review_agent = AssistantAgent(
        name="代码审查专家",
        system_message="""你是一位资深的代码审查专家，专注于发现代码中的问题和改进空间。

        审查维度：
        1. 正确性 - 代码逻辑是否正确，边界条件是否处理
        2. 安全性 - 是否有潜在的注入、溢出等安全风险
        3. 效率   - 算法复杂度是否合理，是否有性能优化空间
        4. 可读性 - 命名是否清晰，结构是否合理
        5. 可维护性 - 是否易于扩展和修改

        审查输出格式：
        如果代码通过审查：
        ✅ 审查通过 | 无重大问题

        如果需要修正：
        ❌ 需要修正 | 问题描述 | 建议方案

        你的反馈会帮助代码生成专家不断改进。""",
        llm_config=llm_config,
    )

    # ---- 3. UserProxyAgent: 需求方代理 ----
    # 角色：代表最终用户，描述需求并验证结果
    # 特点：作为人与AI之间的桥梁，管理对话流程
    user_proxy = UserProxyAgent(
        name="需求方代理",
        human_input_mode="NEVER",  # 完全自动运行，无需人工干预
        max_consecutive_auto_reply=10,  # 最多连续自动回复10次，防止无限循环
        code_execution_config={
            "executor": "local",  # 本地执行代码
            "use_docker": False,  # 不使用Docker（Windows环境）
        },
        system_message="""你是连接开发者和AI Agent的桥梁。

        你的职责：
        1. 将用户的自然语言需求准确传达给代码生成专家
        2. 协调代码生成和审查的迭代过程
        3. 验证最终生成的代码是否满足需求

        工作模式：
        - 使用group chat进行多Agent协作
        - 监控整个工作流程的执行状态
        - 在代码审查通过后，执行验证测试

        始终保持专业、耐心的态度。""",
    )

    return assistant, review_agent, user_proxy


# ============================================================
# 第二步：创建GroupChat进行多Agent协作
# ============================================================

def create_group_chat(assistant, review_agent, user_proxy):
    """
    创建群聊环境，实现多Agent协作

    GroupChat工作原理：
    ┌────────────────────────────────────────────────────┐
    │                  GroupChat                         │
    │                                                    │
    │   [UserProxy] ───发送消息───▶ [Assistant]         │
    │       ▲                              │           │
    │       │                              ▼           │
    │       │                         [Review]          │
    │       │                              │           │
    │       └──────────反馈修正◀───────────┘           │
    │                                                    │
    └────────────────────────────────────────────────────┘

    消息在三个Agent之间循环，直到任务完成或达到最大迭代次数。
    """

    # 定义群聊成员列表，Agent按照此顺序接收消息
    # 重要：speaker_transitions定义Agent之间的对话流向
    group_chat = GroupChat(
        agents=[user_proxy, assistant, review_agent],
        messages=[],  # 存储对话历史
        max_round=15,  # 最多进行15轮对话，防止无限循环
        speaker_selection_method="round_robin",  # 轮询选择发言者
    )

    # 创建群聊管理器
    # 管理器决定每轮由哪个Agent发言
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
        name="工作流协调员",
    )

    return manager


# ============================================================
# 第三步：执行Vibe Coding工作流示例
# ============================================================

def run_vibe_coding_workflow_simple():
    """
    运行一个简单的Vibe Coding任务示例

    任务：让AI生成一个排序算法并自动审查
    """

    print("\n" + "="*60)
    print("🚀 启动Vibe Coding工作流 - 基础示例")
    print("="*60 + "\n")

    # 创建Agent
    assistant, review_agent, user_proxy = create_vibe_coding_agents()

    # 创建群聊管理器
    manager = create_group_chat(assistant, review_agent, user_proxy)

    # 定义任务
    task = """
    请用Python实现一个学生成绩排序系统。

    要求：
    1. 定义学生类，包含姓名、成绩两个属性
    2. 实现多种排序方式：按成绩升序、按成绩降序
    3. 包含成绩统计功能：最高分、最低分、平均分
    4. 代码需要包含中文注释
    """

    print("📋 任务描述：")
    print(task)
    print("\n" + "-"*60 + "\n")

    # 启动群聊
    # chat_initiation定义初始消息内容
    user_proxy.initiate_chat(
        manager,
        message=task,
        clear_history=True,
    )

    print("\n" + "="*60)
    print("✅ Vibe Coding工作流执行完成")
    print("="*60)


# ============================================================
# 第四步：演示如何直接使用两个Agent进行迭代开发
# ============================================================

def run_direct_agent_iteration():
    """
    直接使用两个Agent进行迭代开发的示例

    这种模式更适合简单的任务，避免群聊的开销

    工作流程：
    Assistant (生成) → 执行代码 → Review (审查)
         ↑                              │
         └──── 修正建议 ←──────────────┘
    """

    print("\n" + "="*60)
    print("🚀 启动直接Agent迭代模式")
    print("="*60 + "\n")

    # 创建两个核心Agent
    assistant = AssistantAgent(
        name="直接代码生成",
        system_message="""你是一位Python开发专家，负责根据需求生成代码。
        直接输出代码即可，不需要过多解释。""",
        llm_config=llm_config,
    )

    review = AssistantAgent(
        name="直接代码审查",
        system_message="""你是一位代码审查专家。
        如果代码有问题，用中文提出具体的修改建议。
        如果代码没问题，回复"审查通过"。""",
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="用户代理",
        human_input_mode="NEVER",
        code_execution_config={"executor": "local", "use_docker": False},
    )

    # 简单任务：生成一个计算器
    task = "生成一个简单的命令行计算器，支持加减乘除运算"

    print(f"📝 任务: {task}\n")

    # 第一步：生成代码
    print("📤 步骤1: 代码生成专家生成代码...")
    assistant_response = assistant.generate_reply(
        messages=[{"role": "user", "content": task}]
    )

    print("📥 生成的代码:")
    print(assistant_response)

    # 第二步：执行代码
    print("\n📤 步骤2: 执行生成的代码...")
    user_proxy.execute_code_blocks(
        code_blocks=[{"language": "python", "code": assistant_response}]
    )

    # 第三步：审查
    print("\n📤 步骤3: 代码审查专家审查...")
    review_response = review.generate_reply(
        messages=[{"role": "user", "content": assistant_response + "\n请审查这段代码"}]
    )
    print(f"🔍 审查结果: {review_response}")

    print("\n" + "="*60)
    print("✅ 直接迭代模式执行完成")
    print("="*60)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    """
    主程序入口

    可以选择运行不同的示例：
    1. run_vibe_coding_workflow_simple() - 完整的群聊工作流
    2. run_direct_agent_iteration()      - 直接Agent迭代模式
    """

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       Vibe Coding 工作流演示 - 第25节学习示例               ║
    ║                                                              ║
    ║  Vibe Coding核心理念：                                       ║
    ║  - AI Agent协作完成开发任务                                  ║
    ║  - 开发者专注于创意和设计                                     ║
    ║  - 代码生成 → 执行验证 → 审查修正 的循环迭代                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 运行基础示例
    try:
        run_vibe_coding_workflow_simple()
    except Exception as e:
        print(f"\n⚠️ 群聊模式出错: {e}")
        print("   尝试使用直接迭代模式...")

    # 运行直接迭代示例
    try:
        run_direct_agent_iteration()
    except Exception as e:
        print(f"\n⚠️ 直接迭代模式出错: {e}")
        print("   请检查API配置和网络连接")