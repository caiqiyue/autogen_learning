"""
ConversableAgent初始化过程示例

本文件展示如何正确初始化AutoGen的ConversableAgent类，
包括各种配置参数的使用方法。

作者：AutoGen学习课程
学习目标：理解ConversableAgent的初始化流程
"""

import os
from typing import Union, List, Optional, Dict, Any

# ============================================================
# 第一部分：导入AutoGen核心组件
# ============================================================

# 导入ConversableAgent类 - 这是AutoGen中最核心的智能体类
# 所有具有对话能力的智能体都继承自此类
from autogen import ConversableAgent

# 导入代码执行器 - 用于执行Python代码
# AutoGen支持多种代码执行器：本地执行器、Docker执行器、Jupyter执行器等
from autogen import LocalExecutableCodeExecutor

# 导入Agent输出类型 - 用于定义智能体的输出结构
from autogen.agentchat import AgentOutput

# ============================================================
# 第二部分：准备LLM配置
# ============================================================

def prepare_llm_config():
    """
    准备语言模型配置

    LLM配置是ConversableAgent最重要的配置之一，
    它决定了智能体的"大脑"能力。

    返回:
        dict: 包含LLM配置的字典
    """
    # 方式1：使用环境变量存储API密钥（推荐方式）
    # 这样可以避免在代码中硬编码敏感信息
    api_key = os.getenv("OPENAI_API_KEY")

    # 方式2：直接传入API密钥（仅用于测试，不推荐用于生产环境）
    # api_key = "your-api-key-here"

    llm_config = {
        # 模型名称：可以是OpenAI的gpt-4、gpt-3.5-turbo，或其他兼容模型
        "model": "gpt-4",

        # API密钥：从环境变量读取，保证安全性
        "api_key": api_key,

        # 温度参数：控制输出的随机性
        # 0.0 = 最确定性输出，适合精确任务
        # 0.7 = 平衡模式，适合一般对话
        # 1.0 = 高度随机，适合创意任务
        "temperature": 0.7,

        # 最大令牌数：限制单次响应的最大长度
        # 适当的值可以防止过长的输出
        "max_tokens": 2000,

        # Top-p参数：另一种控制随机性的方式
        # 值越低，输出越集中；值越高，输出越多样
        "top_p": 0.9,

        # 频率惩罚：减少重复 tokens 的使用
        # 范围：-2.0 到 2.0，正值减少重复
        "frequency_penalty": 0.0,

        # 在场惩罚：鼓励引入新话题
        # 范围：-2.0 到 2.0，正值鼓励新话题
        "presence_penalty": 0.0
    }

    return llm_config


# ============================================================
# 第三部分：基础初始化示例
# ============================================================

def create_basic_agent():
    """
    创建最基本的ConversableAgent实例

    最小配置只需要：
    - name: 智能体名称，用于标识
    - system_message: 系统提示词，定义智能体的角色和行为
    """
    # 系统提示词：定义智能体的角色
    # 这是最重要的配置之一，决定了智能体的行为风格
    system_message = """你是一个乐于助人的AI助手。
    你的特点是：
    1. 回答简洁明了
    2. 主动提供有用的建议
    3. 在不确定时会诚实说明"""

    # 基础配置：只包含必需参数
    basic_agent = ConversableAgent(
        name="basic_assistant",  # 智能体唯一名称，同一会话中不能重复
        system_message=system_message,  # 系统提示词，定义角色
        # llm_config设置为None时，智能体将不能生成LLM回复
        # 但仍然可以执行代码或使用工具
        llm_config=None
    )

    print("=== 基础智能体创建成功 ===")
    print(f"智能体名称: {basic_agent.name}")
    print(f"系统提示词长度: {len(basic_agent.system_message)} 字符")

    return basic_agent


# ============================================================
# 第四部分：完整配置初始化示例
# ============================================================

def create_fully_configured_agent():
    """
    创建完整配置的ConversableAgent

    展示所有核心配置参数的使用方法
    """
    # 准备LLM配置
    llm_config = prepare_llm_config()

    # 系统提示词：定义企业级助手的角色
    system_message = """你是一个企业级数据分析助手。

    你的职责包括：
    1. 帮助用户分析数据并提供洞察
    2. 编写和执行Python代码进行数据处理
    3. 生成可视化图表
    4. 提供数据驱动的决策建议

    你的工作原则：
    - 数据准确，分析严谨
    - 代码高效，遵循最佳实践
    - 回答专业，解释清晰
    - 不确定时主动说明"""

    # 代码执行器配置：指定如何执行代码
    # LocalExecutableCodeExecutor 在本地环境执行Python代码
    code_executor = LocalExecutableCodeExecutor(
        timeout=30,  # 单次代码执行超时时间（秒）
        # max_consecutive_auto_reply=None,  # 最大连续自动回复数，None表示无限制
        sandbox=None  # 沙箱配置，None表示使用默认配置
    )

    # 创建完整配置的智能体
    fully_configured_agent = ConversableAgent(
        # 基础信息
        name="enterprise_data_assistant",  # 唯一标识
        system_message=system_message,  # 角色定义

        # LLM配置
        llm_config=llm_config,

        # 代码执行配置
        code_executor=code_executor,  # 使用本地代码执行器

        # 人工介入模式：控制何时需要人工输入
        # "NEVER" - 从不询问，完全自主运行
        # "TERMINATE" - 只在收到终止消息时询问
        # "ALWAYS" - 每次回复前都询问
        human_input_mode="TERMINATE",

        # 最大连续自动回复数：防止智能体无限循环
        # 当达到此数量后，会强制要求人工输入或终止
        max_consecutive_auto_reply=10,

        # 终止消息判断函数：定义什么情况算是"完成"
        # 这是一个可选参数，默认使用内置判断逻辑
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),

        # 聊天记录配置
        # chat_messages用作初始化聊天记录（可选）
        # chat_messages_dict=None,  # 如果需要从某个状态恢复，可以传入历史消息

        # 混合代理配置（可选）
        # 如果需要使用混合代理，可以在这里配置
        # default_auto_reply="请稍候...",  # 默认自动回复
        # max_reply=100,  # 最大回复总数限制
    )

    print("=== 完整配置智能体创建成功 ===")
    print(f"智能体名称: {fully_configured_agent.name}")
    print(f"代码执行器类型: {type(fully_configured_agent.code_executor)}")
    print(f"人工介入模式: {fully_configured_agent.human_input_mode}")

    return fully_configured_agent


# ============================================================
# 第五部分：不同角色的智能体创建
# ============================================================

def create_role_based_agents():
    """
    创建具有不同角色的智能体

    展示如何根据不同用途创建专属智能体
    """

    # 1. 编程助手智能体
    # 专门用于代码编写、调试和优化
    coding_assistant = ConversableAgent(
        name="coding_assistant",
        system_message="""你是一个专业的Python编程助手。

        你的专长：
        1. 编写高质量的Python代码
        2. 调试和修复代码错误
        3. 优化代码性能和可读性
        4. 解释代码逻辑和实现细节

        代码风格要求：
        - 遵循PEP 8规范
        - 添加必要的注释说明
        - 使用类型提示提高可读性
        - 编写单元测试确保质量""",
        llm_config=prepare_llm_config(),
        # 编程助手使用本地代码执行器来运行代码
        code_executor="local",
        human_input_mode="NEVER"  # 编程任务可以全自动运行
    )

    # 2. 数据分析师智能体
    # 专门用于数据处理和可视化
    data_analyst = ConversableAgent(
        name="data_analyst",
        system_message="""你是一个专业的数据分析师。

        你的职责：
        1. 读取和处理各类数据文件
        2. 进行数据清洗和预处理
        3. 执行统计分析和建模
        4. 生成数据可视化图表
        5. 撰写数据分析报告

        分析原则：
        - 数据质量优先
        - 结果可复现
        - 图表清晰易懂
        - 结论有理有据""",
        llm_config=prepare_llm_config(),
        code_executor="local",
        human_input_mode="TERMINATE"  # 数据分析结果需要确认
    )

    # 3. 客服智能体
    # 用于自动回复用户咨询
    customer_service = ConversableAgent(
        name="customer_service",
        system_message="""你是一个友好的客服助手。

        你的服务理念：
        1. 耐心倾听用户问题
        2. 用简洁易懂的语言解释
        3. 提供准确有效的解决方案
        4. 无法解决时及时转接人工

        服务规范：
        - 语言亲切友好
        - 回答及时准确
        - 保护用户隐私
        - 记录重要信息""",
        llm_config=prepare_llm_config(),
        # 客服不需要代码执行器
        code_executor=None,
        human_input_mode="ALWAYS"  # 客服需要严格的人工监督
    )

    print("=== 角色型智能体创建成功 ===")
    print(f"编程助手: {coding_assistant.name}")
    print(f"数据分析师: {data_analyst.name}")
    print(f"客服: {customer_service.name}")

    return {
        "coding_assistant": coding_assistant,
        "data_analyst": data_analyst,
        "customer_service": customer_service
    }


# ============================================================
# 第六部分：智能体配置验证
# ============================================================

def validate_agent_configuration():
    """
    验证智能体配置是否正确

    检查配置的有效性并提供修改建议
    """

    print("\n=== 智能体配置验证 ===")

    # 检查必要配置
    llm_config = prepare_llm_config()

    # 验证LLM配置
    required_keys = ["model", "api_key"]
    missing_keys = [key for key in required_keys if key not in llm_config]

    if missing_keys:
        print(f"警告：LLM配置缺少以下必要键: {missing_keys}")
    else:
        print("LLM配置验证通过：所有必要键都存在")

    # 验证API密钥
    if not llm_config.get("api_key"):
        print("警告：API密钥未设置，智能体将无法调用LLM")
    else:
        # 检查API密钥格式（简单验证）
        api_key = llm_config["api_key"]
        if api_key.startswith("sk-"):
            print("API密钥格式验证通过")
        else:
            print("警告：API密钥格式可能不正确（应以sk-开头）")

    # 验证数值参数
    temperature = llm_config.get("temperature", 0)
    if 0 <= temperature <= 2:
        print(f"温度参数验证通过: {temperature}")
    else:
        print(f"警告：温度参数应在0-2之间，当前值: {temperature}")

    print("配置验证完成")


# ============================================================
# 第七部分：主函数 - 运行所有示例
# ============================================================

def main():
    """
    主函数：运行所有初始化示例

    展示ConversableAgent的各种初始化方式
    """

    print("=" * 60)
    print("ConversableAgent 初始化过程演示")
    print("=" * 60)

    # 1. 验证配置
    print("\n【步骤1】验证智能体配置...")
    validate_agent_configuration()

    # 2. 创建基础智能体
    print("\n【步骤2】创建基础智能体...")
    basic_agent = create_basic_agent()

    # 3. 创建完整配置智能体
    print("\n【步骤3】创建完整配置智能体...")
    fully_configured_agent = create_fully_configured_agent()

    # 4. 创建角色型智能体
    print("\n【步骤4】创建角色型智能体...")
    role_agents = create_role_based_agents()

    # 5. 打印总结
    print("\n" + "=" * 60)
    print("初始化演示完成！")
    print("=" * 60)
    print("\n创建的智能体列表：")
    print(f"1. {basic_agent.name} - 基础助手")
    print(f"2. {fully_configured_agent.name} - 企业级数据助手")
    print(f"3. coding_assistant - 编程助手")
    print(f"4. data_analyst - 数据分析师")
    print(f"5. customer_service - 客服")
    print("\n注意：由于未设置有效的API密钥，这些智能体无法进行实际的LLM对话。")
    print("在生产环境中，请确保正确配置OPENAI_API_KEY环境变量。")


# ============================================================
# 程序入口点
# ============================================================

if __name__ == "__main__":
    # 运行主函数
    main()

    """
    运行说明：
    1. 确保已安装 autogen 包：pip install autogen
    2. 设置环境变量 OPENAI_API_KEY
    3. 运行命令：python conversable_agent_init.py

    预期输出：
    - 显示各种初始化方式的成功信息
    - 展示不同配置参数的效果
    - 打印创建的所有智能体列表

    注意事项：
    - 本示例中的LLM配置需要有效的API密钥才能正常工作
    - 代码执行功能需要本地Python环境支持
    """