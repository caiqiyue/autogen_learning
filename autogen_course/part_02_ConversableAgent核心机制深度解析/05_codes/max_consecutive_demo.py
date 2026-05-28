# max_consecutive_demo.py
# 第5节 max_consecutive_auto_reply与轮次控制 - 核心演示
#
# 本文件演示 max_consecutive_auto_reply 参数对对话轮次控制的原理
#
# 要运行此代码，你需要：
# 1. 安装 AutoGen: pip install pyautogen
# 2. 配置环境变量（在 .env 文件中或系统环境变量中）
#    OPENAI_MODEL=你的模型名称
#    OPENAI_API_KEY=你的API密钥
#    OPENAI_BASE_URL=API基础URL（可选）
#
# ============================================================
# max_consecutive_auto_reply 参数详解
# ============================================================
#
# max_consecutive_auto_reply: 控制单个Agent在对话中的最大连续自动回复次数
#
# 工作原理：
# 1. 每次Agent生成自动回复（非人工输入触发）时，计数器+1
# 2. 当计数器达到max_consecutive_auto_reply设定的值时，Agent停止自动回复
# 3. 此时如果human_input_mode设置恰当，会触发人类输入请求
#
# 与MAX_CONSECUTIVE_AUTO_REPLY类属性的交互：
# - MAX_CONSECUTIVE_AUTO_REPLY是类属性，默认值通常为None（表示无限制）
# - 实例的max_consecutive_auto_reply参数会覆盖类属性
# - 如果实例设置为0，则完全禁用自动回复
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
# 第三部分：演示函数
# ============================================================

def demo_max_consecutive_zero():
    """
    演示 max_consecutive_auto_reply=0 的效果

    设置为0时，Agent完全禁用自动回复功能，每次回复都需要人类输入。
    这适用于需要严格控制对话节奏、每一步都需要人工确认的场景。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示1: max_consecutive_auto_reply=0 (禁用自动回复)")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建助手代理 - max_consecutive_auto_reply=0 表示禁用自动回复
    assistant = ConversableAgent(
        name="助手",
        llm_config=llm_config,
        max_consecutive_auto_reply=0,  # 关键参数：设为0完全禁用自动回复
    )

    # 创建用户代理 - 这里使用简单的模拟消息
    user_proxy = ConversableAgent(
        name="用户代理",
        human_input_mode="NEVER",  # 不请求人类输入，用于模拟
        max_consecutive_auto_reply=0,
    )

    # 构造对话历史
    messages = [
        {"role": "user", "content": "请给我讲一个关于AI的小故事"}
    ]

    print(f"助手名称: {assistant.name}")
    print(f"max_consecutive_auto_reply: {assistant.max_consecutive_auto_reply}")
    print(f"类属性 MAX_CONSECUTIVE_AUTO_REPLY: {ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY}")
    print(f"\n发送消息: {messages[0]['content']}")

    # 注意：当 max_consecutive_auto_reply=0 时，
    # generate_reply 会在内部逻辑中直接返回，不调用 LLM
    reply = assistant.generate_reply(messages=messages)

    print(f"\n回复结果: {reply}")
    print("说明: max_consecutive_auto_reply=0 时，Agent不会调用LLM生成回复")


def demo_max_consecutive_five():
    """
    演示 max_consecutive_auto_reply=5 的效果

    设置为5时，Agent最多连续自动回复5次，之后停止。
    这适用于需要限制对话轮次但允许一定自主性的场景。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示2: max_consecutive_auto_reply=5 (限制5轮自动回复)")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建助手代理 - max_consecutive_auto_reply=5 限制最多5轮连续回复
    assistant = ConversableAgent(
        name="助手",
        llm_config=llm_config,
        max_consecutive_auto_reply=5,  # 关键参数：限制5轮
    )

    print(f"助手名称: {assistant.name}")
    print(f"max_consecutive_auto_reply: {assistant.max_consecutive_auto_reply}")
    print(f"MAX_CONSECUTIVE_AUTO_REPLY (类属性): {ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY}")

    # 模拟多轮对话
    messages = [
        {"role": "user", "content": "帮我写一个Python的快速排序算法"}
    ]

    print(f"\n初始消息: {messages[0]['content']}")
    print("开始多轮对话模拟...")

    # 模拟连续回复场景
    # 每次 generate_reply 调用，内部的 consecutive_auto_reply_counter 会 +1
    for i in range(1, 7):  # 尝试6轮回复
        print(f"\n--- 第 {i} 轮尝试 ---")
        reply = assistant.generate_reply(messages=messages)

        if reply:
            print(f"回复 {i}: {reply[:50]}..." if len(str(reply)) > 50 else f"回复 {i}: {reply}")
            # 添加助手回复到消息历史，模拟下一轮
            messages.append({"role": "assistant", "content": str(reply)})
        else:
            print(f"回复 {i}: None (已达到轮次上限)")
            break

    print("\n说明: 当连续回复达到5次后，generate_reply 返回 None")


def demo_max_consecutive_none():
    """
    演示 max_consecutive_auto_reply=None（默认值）的效果

    当设置为None时，使用类属性 MAX_CONSECUTIVE_AUTO_REPLY 的值。
    类属性默认值通常也是None，表示无限制。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示3: max_consecutive_auto_reply=None (默认值/无限制)")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建助手代理 - 不设置 max_consecutive_auto_reply，使用默认值 None
    assistant = ConversableAgent(
        name="助手",
        llm_config=llm_config,
        # max_consecutive_auto_reply 参数留空，使用默认值 None
    )

    print(f"助手名称: {assistant.name}")
    print(f"max_consecutive_auto_reply (实例属性): {assistant.max_consecutive_auto_reply}")
    print(f"MAX_CONSECUTIVE_AUTO_REPLY (类属性): {ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY}")

    # 对比：显式设置 None 效果相同
    assistant_explicit = ConversableAgent(
        name="助手(显式None)",
        llm_config=llm_config,
        max_consecutive_auto_reply=None,  # 显式设置为 None
    )

    print(f"\n显式设置None的实例 max_consecutive_auto_reply: {assistant_explicit.max_consecutive_auto_reply}")

    print("\n说明: None 表示无限制，不设轮次上限")


def demo_class_attribute_interaction():
    """
    演示实例属性与类属性的交互

    实例的 max_consecutive_auto_reply 参数会覆盖类属性。
    可以通过修改类属性来设置所有实例的默认值。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示4: 实例属性与类属性的交互")
    print("=" * 60)

    # 1. 查看类属性默认值
    print(f"1. 类属性默认值: {ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY}")

    # 2. 创建实例时覆盖
    llm_config = build_llm_config()
    agent1 = ConversableAgent(
        name="代理1",
        llm_config=llm_config,
        max_consecutive_auto_reply=3,
    )
    print(f"2. agent1 (设置=3): {agent1.max_consecutive_auto_reply}")

    agent2 = ConversableAgent(
        name="代理2",
        llm_config=llm_config,
        max_consecutive_auto_reply=10,
    )
    print(f"3. agent2 (设置=10): {agent2.max_consecutive_auto_reply}")

    # 3. 创建实例时使用默认值
    agent3 = ConversableAgent(
        name="代理3",
        llm_config=llm_config,
        # 不设置，使用类属性
    )
    print(f"4. agent3 (默认值): {agent3.max_consecutive_auto_reply}")

    # 4. 修改类属性会影响后续创建的实例（但不改变已创建的实例）
    print(f"\n5. 修改类属性前: agent3使用类属性值")
    ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY = 7
    print(f"6. 修改类属性为: 7")

    agent4 = ConversableAgent(
        name="代理4",
        llm_config=llm_config,
        # 不设置，继承新的类属性值
    )
    print(f"7. agent4 (新实例): {agent4.max_consecutive_auto_reply}")
    print(f"8. agent3 (已有实例，不受影响): {agent3.max_consecutive_auto_reply}")

    # 恢复类属性默认值
    ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY = None
    print(f"\n9. 恢复类属性为 None: {ConversableAgent.MAX_CONSECUTIVE_AUTO_REPLY}")


def demo_consecutive_counter_mechanism():
    """
    演示 consecutive_auto_reply_counter 计数机制

    这个内部计数器追踪连续自动回复的次数。
    当收到人类输入或回复被终止时，计数器会重置。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示5: consecutive_auto_reply_counter 计数机制")
    print("=" * 60)

    llm_config = build_llm_config()

    assistant = ConversableAgent(
        name="助手",
        llm_config=llm_config,
        max_consecutive_auto_reply=3,  # 设置上限为3
    )

    print(f"助手配置: max_consecutive_auto_reply={assistant.max_consecutive_auto_reply}")

    # 模拟对话场景
    messages = [
        {"role": "user", "content": "解释什么是机器学习"}
    ]

    # 模拟连续回复（不实际调用LLM，只演示计数器机制）
    print("\n模拟多轮自动回复场景:")
    for i in range(1, 5):
        print(f"  轮次 {i}: consecutive_auto_reply_counter 将 +1")

        # 检查是否达到上限（实际逻辑中此检查在generate_reply内部）
        if i > assistant.max_consecutive_auto_reply:
            print(f"  轮次 {i}: 达到上限，停止自动回复")
            break

    print("\n计数器重置条件:")
    print("  1. 收到人类输入消息时")
    print("  2. is_termination_msg 返回 True 时")
    print("  3. Agent显式停止时")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有 max_consecutive_auto_reply 演示
    """
    print("=" * 60)
    print("max_consecutive_auto_reply 轮次控制演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_max_consecutive_zero()
    demo_max_consecutive_five()
    demo_max_consecutive_none()
    demo_class_attribute_interaction()
    demo_consecutive_counter_mechanism()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()