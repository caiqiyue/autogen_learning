# async_basic.py
# 第20节 异步Agent基础用法演示
#
# 本文件展示异步Agent的基础用法，包括：
# 1. 同步reply_func与异步reply_func的定义与注册
# 2. a_generate_reply的基本调用方式
# 3. 异步环境下的简单超时控制
#
# ============================================================
# 异步编程核心概念
# ============================================================
#
# 1. 协程（Coroutine）：
#    - 使用 async def 定义的函数称为协程函数
#    - 调用协程函数返回一个协程对象，不会立即执行
#    - 必须使用 await 来执行协程
#
# 2. 事件循环（Event Loop）：
#    - asyncio.create_task() 创建任务
#    - asyncio.gather() 并发执行多个任务
#    - asyncio.wait_for() 添加超时控制
#
# 3. await 的作用：
#    - 等待协程执行完成并返回结果
#    - 在等待期间，事件循环可以执行其他协程（非阻塞）
#    - 只能在 async def 函数中使用
#
# ============================================================

import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# 添加父目录到路径，以便导入autogen
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 尝试导入autogen
try:
    from autogen import ConversableAgent, Agent
except ImportError:
    print("错误：请先安装autogen库：pip install autogen")
    sys.exit(1)


# ============================================================
# 第一部分：环境配置加载（与之前课程保持一致）
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


def build_llm_config():
    """
    构建 AutoGen 的 LLM 配置

    Returns:
        dict: 包含模型配置的字典
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
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
# 第二部分：核心概念演示
# ============================================================

def demo_coroutine_vs_sync_function():
    """
    演示协程函数与普通函数的区别

    关键点：
    1. 普通函数：def 关键字，返回值直接可用
    2. 协程函数：async def 关键字，返回协程对象，需要 await 执行
    3. inspect.iscoroutinefunction() 可以判断函数类型
    """
    print("\n" + "=" * 60)
    print("演示1: 协程函数 vs 普通函数")
    print("=" * 60)

    # 定义一个普通函数（同步函数）
    def sync_function(x):
        """同步函数：直接返回结果"""
        return f"同步函数结果: {x}"

    # 定义一个协程函数（异步函数）
    async def async_function(x):
        """异步函数：返回协程对象，需要await执行"""
        # 模拟一个异步I/O操作（如API调用）
        await asyncio.sleep(0.1)  # 模拟耗时操作
        return f"异步函数结果: {x}"

    # 演示调用方式的区别
    print("\n1. 调用方式区别:")

    # 同步函数直接调用
    sync_result = sync_function("测试")
    print(f"   sync_function('测试') = {sync_result}")
    print(f"   类型: {type(sync_result)}")

    # 异步函数调用返回协程对象（未执行）
    async_obj = async_function("测试")
    print(f"   async_function('测试') = {async_obj}")
    print(f"   类型: {type(async_obj)}")

    # 必须使用await执行异步函数
    async def run_async():
        return await async_function("测试")

    async_result = asyncio.run(run_async())
    print(f"   await async_function('测试') = {async_result}")
    print(f"   类型: {type(async_result)}")

    # 使用inspect判断函数类型
    print("\n2. 使用inspect.iscoroutinefunction判断:")
    print(f"   inspect.iscoroutinefunction(sync_function) = {inspect.iscoroutinefunction(sync_function)}")
    print(f"   inspect.iscoroutinefunction(async_function) = {inspect.iscoroutinefunction(async_function)}")


def demo_sync_reply_func():
    """
    演示同步reply_func的注册与执行

    同步reply_func是最简单的回复策略：
    - 直接返回回复内容字符串
    - 不涉及任何I/O操作
    - 在策略链中直接调用，不阻塞其他任务
    """
    print("\n" + "=" * 60)
    print("演示2: 同步reply_func注册与执行")
    print("=" * 60)

    # 创建一个简单的Agent（不使用LLM）
    agent = ConversableAgent(
        name="同步Agent",
        system_message="你是一个简单的助手。",
        llm_config=False,  # 不使用LLM，纯规则引擎
        human_input_mode="NEVER",
    )

    # 定义同步reply_func
    def sync_reply_handler(messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        同步回复处理器

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 回复内容
        """
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "").lower()

        # 简单的关键词匹配
        if "你好" in content or "hello" in content:
            return "你好！有什么可以帮助你的吗？"

        if "时间" in content:
            import datetime
            return f"现在时间是 {datetime.datetime.now().strftime('%H:%M:%S')}"

        return ""  # 返回空字符串表示不处理

    # 注册同步reply_func
    agent.register_reply(
        trigger=None,  # 无条件触发（后备处理）
        reply_func=sync_reply_handler,
        position=0  # 高优先级
    )

    # 验证函数类型
    print(f"\n回复函数类型验证:")
    print(f"  inspect.iscoroutinefunction(sync_reply_handler) = {inspect.iscoroutinefunction(sync_reply_handler)}")

    # 测试同步reply_func
    print("\n测试同步reply_func:")
    test_messages = [
        {"content": "你好", "role": "user"},
        {"content": "现在几点了", "role": "user"},
        {"content": "今天天气不错", "role": "user"}
    ]

    for msg in test_messages:
        # 使用同步方法generate_reply
        response = agent.generate_reply(
            messages=[msg],
            sender=None
        )
        print(f"  用户: {msg['content']}")
        print(f"  Agent: {response}")
        print()


def demo_async_reply_func():
    """
    演示异步reply_func的注册与执行

    异步reply_func适用于：
    - 需要调用外部API（如LLM）
    - 需要执行I/O密集型操作
    - 需要与其他异步任务并发执行

    注意：异步reply_func必须使用 async def 定义
    """
    print("\n" + "=" * 60)
    print("演示3: 异步reply_func注册与执行")
    print("=" * 60)

    # 创建Agent
    agent = ConversableAgent(
        name="异步Agent",
        system_message="你是一个有帮助的助手。",
        llm_config=False,  # 不使用LLM，纯演示
        human_input_mode="NEVER",
    )

    # 定义异步reply_func
    async def async_reply_handler(messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        异步回复处理器

        这个函数模拟一个异步API调用场景。
        实际应用中，这里可以调用LLM或其他异步服务。

        Args:
            messages: 消息历史列表
            sender: 发送者Agent
            config: 配置字典

        Returns:
            str: 回复内容
        """
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")

        # 模拟异步API调用（实际应用中替换为真实API调用）
        await asyncio.sleep(0.2)  # 模拟网络延迟

        # 根据输入返回不同的响应
        if "hello" in content.lower() or "你好" in content.lower():
            return "你好！我是异步Agent，很高兴为你服务！"

        if "时间" in content:
            import datetime
            return f"现在时间是 {datetime.datetime.now().strftime('%H:%M:%S')}"

        return ""

    # 注册异步reply_func
    agent.register_reply(
        trigger=None,
        reply_func=async_reply_handler,
        position=0
    )

    # 验证函数类型
    print(f"\n回复函数类型验证:")
    print(f"  inspect.iscoroutinefunction(async_reply_handler) = {inspect.iscoroutinefunction(async_reply_handler)}")

    # 测试异步reply_func
    # 注意：必须使用a_generate_reply（异步方法）来调用
    print("\n测试异步reply_func（使用a_generate_reply）:")

    async def test_async_reply():
        """异步测试函数"""
        test_messages = [
            {"content": "你好", "role": "user"},
            {"content": "现在几点了", "role": "user"},
        ]

        for msg in test_messages:
            # 使用异步方法a_generate_reply
            response = await agent.a_generate_reply(
                messages=[msg],
                sender=None
            )
            print(f"  用户: {msg['content']}")
            print(f"  Agent: {response}")
            print()

    # 运行异步测试
    asyncio.run(test_async_reply())


def demo_mixed_sync_async_chain():
    """
    演示混合同步/异步策略链

    AutoGen支持在同一策略链中混合使用同步和异步函数。
    系统会自动根据函数类型选择同步调用或异步await。

    这个演示展示了策略链的执行顺序和类型检测机制。
    """
    print("\n" + "=" * 60)
    print("演示4: 混合同步/异步策略链")
    print("=" * 60)

    # 创建Agent
    agent = ConversableAgent(
        name="混合Agent",
        system_message="你是一个混合处理Agent。",
        llm_config=False,
        human_input_mode="NEVER",
    )

    # 策略1：同步函数 - 检查消息前缀
    def check_prefix(messages: List[Dict], sender: Agent, config: Dict) -> str:
        """同步策略：检查消息前缀"""
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")

        # 如果消息以"/"开头，交给命令处理器
        if content.startswith("/"):
            return f"[命令处理] 收到命令: {content}"

        return ""  # 返回空表示不处理，继续下一个策略

    # 策略2：异步函数 - 处理常规消息
    async def process_regular_message(messages: List[Dict], sender: Agent, config: Dict) -> str:
        """异步策略：处理常规消息"""
        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")

        # 模拟异步处理
        await asyncio.sleep(0.1)

        return f"[异步处理] 收到消息: {content}，长度={len(content)}"

    # 策略3：同步默认策略
    def default_handler(messages: List[Dict], sender: Agent, config: Dict) -> str:
        """默认策略：最终后备"""
        return "[默认处理] 抱歉，我无法理解这个消息"

    # 注册策略链（按优先级从高到低）
    agent.register_reply(
        trigger=None,
        reply_func=check_prefix,
        position=0
    )

    agent.register_reply(
        trigger=None,
        reply_func=process_regular_message,
        position=1
    )

    agent.register_reply(
        trigger=None,
        reply_func=default_handler,
        position=2
    )

    # 打印策略链信息
    print("\n策略链结构:")
    print("  position=0: check_prefix (同步函数)")
    print("  position=1: process_regular_message (异步函数)")
    print("  position=2: default_handler (同步函数)")
    print("\n执行流程:")
    print("  1. 先执行 position=0 的 check_prefix")
    print("  2. 如果返回空，继续执行 position=1 的 process_regular_message")
    print("  3. 如果仍然返回空，执行 position=2 的 default_handler")

    # 测试各种消息
    print("\n测试消息处理:")

    async def test_mixed():
        test_messages = [
            {"content": "/help", "role": "user"},  # 命令消息
            {"content": "你好", "role": "user"},   # 常规消息
            {"content": "unknown", "role": "user"} # 未知消息
        ]

        for msg in test_messages:
            # 使用异步方法触发策略链
            response = await agent.a_generate_reply(
                messages=[msg],
                sender=None
            )
            print(f"  消息: '{msg['content']}'")
            print(f"  结果: {response}")
            print()

    asyncio.run(test_mixed())


def demo_timeout_control():
    """
    演示异步环境下的超时控制

    在异步环境中，可以使用 asyncio.wait_for 或 asyncio.timeout
    来为协程添加超时控制，防止操作无限期等待。

    超时控制对于LLM API调用尤为重要：
    - 网络问题可能导致API调用挂起
    - 无限等待会阻塞整个事件循环
    - 合理的超时设置可以提高系统健壮性
    """
    print("\n" + "=" * 60)
    print("演示5: 异步超时控制")
    print("=" * 60)

    # 创建Agent
    agent = ConversableAgent(
        name="超时测试Agent",
        system_message="你是一个响应较慢的助手。",
        llm_config=False,
        human_input_mode="NEVER",
    )

    # 定义一个模拟慢速响应的异步函数
    async def slow_async_reply(messages: List[Dict], sender: Agent, config: Dict) -> str:
        """
        模拟慢速响应的异步函数

        Args:
            delay: 模拟的延迟时间（秒）

        Returns:
            str: 延迟后的响应
        """
        delay = config.get("delay", 0.5) if config else 0.5

        # 模拟处理时间
        await asyncio.sleep(delay)

        if not messages:
            return ""

        latest_message = messages[-1]
        content = latest_message.get("content", "")
        return f"收到消息 '{content}'，处理完成"

    # 注册慢速回复函数
    agent.register_reply(
        trigger=None,
        reply_func=slow_async_reply,
        position=0,
        config={"delay": 0.5}  # 500ms延迟
    )

    print("\n场景1: 正常完成（不超时）")
    print("  - 处理时间: 0.5秒")
    print("  - 超时设置: 1秒")
    print("  - 预期结果: 正常完成")

    async def test_no_timeout():
        """测试不超时的场景"""
        try:
            # 使用 wait_for 添加超时控制
            response = await asyncio.wait_for(
                agent.a_generate_reply(messages=[{"content": "快速消息", "role": "user"}], sender=None),
                timeout=1.0  # 1秒超时
            )
            print(f"  结果: {response}")
        except asyncio.TimeoutError:
            print("  结果: 超时！")

    asyncio.run(test_no_timeout())

    print("\n场景2: 操作超时")
    print("  - 处理时间: 2秒（实际延迟）")
    print("  - 超时设置: 0.5秒")
    print("  - 预期结果: TimeoutError异常")

    # 修改延迟配置
    agent.register_reply(
        trigger=None,
        reply_func=slow_async_reply,
        position=0,
        config={"delay": 2.0}  # 2秒延迟
    )

    async def test_with_timeout():
        """测试超时的场景"""
        try:
            response = await asyncio.wait_for(
                agent.a_generate_reply(messages=[{"content": "慢速消息", "role": "user"}], sender=None),
                timeout=0.5  # 0.5秒超时
            )
            print(f"  结果: {response}")
        except asyncio.TimeoutError:
            print("  结果: 操作超时！已设置合理的超时控制防止无限等待")

    asyncio.run(test_with_timeout())

    print("\n场景3: 使用asyncio.timeout上下文管理器（Python 3.11+）")
    print("  这是更现代的超时控制方式，推荐使用")

    async def test_timeout_context():
        """使用上下文管理器的超时控制"""
        try:
            async with asyncio.timeout(0.5) as cm:
                # 这个操作会超过0.5秒
                await asyncio.sleep(1)
                return "完成"
        except asyncio.TimeoutError:
            return "超时了！"

    result = asyncio.run(test_timeout_context())
    print(f"  结果: {result}")


def demo_basic_async_agent():
    """
    演示基本的异步Agent工作流

    创建一个简单的异步Agent并执行对话，
    展示async/await在AutoGen中的实际使用方式。
    """
    print("\n" + "=" * 60)
    print("演示6: 基本异步Agent工作流")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建异步友好的Agent
    assistant = ConversableAgent(
        name="异步助手",
        system_message="你是一个简洁的助手，只回答简短的问题。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    user_proxy = ConversableAgent(
        name="用户代理",
        system_message="你是一个用户代理，代表用户发送消息。",
        llm_config=False,  # 不使用LLM
        human_input_mode="NEVER",
    )

    print("\n创建异步Agent对话:")
    print("  - Assistant: 异步助手 (使用LLM)")
    print("  - UserProxy: 用户代理 (不需要LLM)")
    print("  - 对话模式: UserProxy发起，Assistant回复")

    async def run_async_chat():
        """异步执行对话"""

        # 使用a_generate_reply模拟异步对话流程
        # 实际应用中，直接使用initiate_chat方法即可
        # initiate_chat内部已经支持异步调用

        # 模拟一个异步工作流：
        # 1. 用户代理发送消息
        # 2. 助手异步生成回复
        # 3. 收集结果

        message = {"content": "你好，请简短介绍一下你自己", "role": "user"}

        print(f"\n发送消息: {message['content']}")

        # 调用助手的异步回复方法
        response = await assistant.a_generate_reply(
            messages=[message],
            sender=user_proxy
        )

        print(f"收到回复: {response}")

        return response

    # 执行异步对话
    result = asyncio.run(run_async_chat())
    print(f"\n异步对话完成，结果长度: {len(result) if result else 0}")


# ============================================================
# 第三部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有异步基础用法演示
    """
    print("=" * 60)
    print("AutoGen异步Agent基础用法演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_coroutine_vs_sync_function()
    demo_sync_reply_func()
    demo_async_reply_func()
    demo_mixed_sync_async_chain()
    demo_timeout_control()
    demo_basic_async_agent()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n知识点总结:")
    print("  1. async def定义的函数是协程函数，需要await执行")
    print("  2. inspect.iscoroutinefunction()可判断函数类型")
    print("  3. a_generate_reply用于调用异步reply_func")
    print("  4. asyncio.wait_for可添加超时控制")
    print("  5. AutoGen支持同步/异步函数混合的策略链")


if __name__ == "__main__":
    main()