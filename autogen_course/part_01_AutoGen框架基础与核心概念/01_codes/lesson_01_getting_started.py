# lesson_01_getting_started.py
# 第1节 AutoGen概述与开发环境搭建 - 入门示例
#
# 本文件演示如何快速搭建AutoGen开发环境并运行第一个示例程序
#
# 要运行此代码，你需要：
# 1. 安装 AutoGen: pip install pyautogen
# 2. 配置环境变量（在 .env 文件中或系统环境变量中）
#    OPENAI_MODEL=你的模型名称
#    OPENAI_API_KEY=你的API密钥
#    OPENAI_BASE_URL=API基础URL（可选，取决于你使用的API提供商）
#
# 本示例展示如何使用 ConversableAgent 创建一个简单的聊天机器人代理

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

    # 读取文件每一行，解析键值对
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 跳过空行和注释行
        if not line or line.startswith("#") or "=" not in line:
            continue

        # 分割并设置环境变量
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
# 第二部分：创建第一个 AutoGen Agent
# ============================================================

def create_chatbot():
    """
    创建并返回一个简单的聊天机器人代理

    Returns:
        ConversableAgent: 配置好的聊天机器人代理实例
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    # 从环境变量获取配置
    model = get_required_env("OPENAI_MODEL")
    api_key = get_required_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # 可选

    # 构建 LLM 配置字典
    llm_config = {
        "config_list": [{
            "model": model,
            "api_key": api_key,
        }]
    }

    # 如果提供了 base_url，则添加到配置中
    if base_url:
        llm_config["config_list"][0]["base_url"] = base_url

    # 创建 ConversableAgent 实例
    # name: 代理的名称，用于标识不同的代理
    # llm_config: 大语言模型的配置，包含模型、API密钥等
    agent = ConversableAgent(
        name="chatbot",  # 给代理起一个有意义的名字
        llm_config=llm_config,  # LLM 配置
    )

    return agent


def main():
    """
    主函数：运行第一个 AutoGen 示例程序
    """
    print("=" * 60)
    print("AutoGen 入门示例 - 第1个 AutoGen 程序")
    print("=" * 60)

    # 1. 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 2. 创建聊天机器人代理
    print("\n[步骤2] 创建聊天机器人代理...")
    agent = create_chatbot()
    print(f"代理创建成功: {agent.name}")

    # 3. 准备对话消息
    # AutoGen 使用特定的消息格式，包含 role 和 content 字段
    messages = [
        {
            "role": "user",  # 消息来自用户
            "content": "你好，请你介绍一下你自己",  # 用户的输入
        }
    ]

    # 4. 让代理生成回复
    print("\n[步骤3] 让代理生成回复...")
    print("正在等待 LLM 响应...\n")

    reply = agent.generate_reply(messages=messages)

    # 5. 输出结果
    print("=" * 60)
    print("代理回复：")
    print("=" * 60)
    print(reply)
    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 调用主函数运行示例
    main()