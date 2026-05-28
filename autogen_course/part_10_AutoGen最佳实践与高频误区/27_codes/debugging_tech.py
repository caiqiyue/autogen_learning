# debugging_tech.py
# 第27节 AutoGen高频误区与调试技巧 - 调试技巧代码演示
#
# 本文件展示AutoGen的调试技巧与问题定位方法：
# 1. 日志配置与调试输出
# 2. 消息历史分析
# 3. Agent状态检查
# 4. GroupChat监控
# 5. 性能分析与成本追踪
# 6. 常见问题诊断流程
#
# ============================================================
# 调试技巧概览
# ============================================================
#
# 类别              技巧              难度    效果
# ---------------------------------------------------------
# 日志配置          verbose模式      低      高
# 消息分析          chat_history     低      高
# 状态检查          agent属性检查    中      高
# GroupChat监控     select_speaker    中      高
# 性能分析          token追踪        中      中
# 成本追踪          price计算        低      高
# ============================================================

import os
import re
import time
import logging
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Tuple
from functools import wraps

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
# 第二部分：LLM配置构建
# ============================================================

def build_llm_config():
    """
    构建AutoGen的LLM配置

    Returns:
        dict: 包含模型配置的字典
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-demo")
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
# 第三部分：调试技巧代码
# ============================================================

# ------------------------------------------------------------
# 技巧1：日志配置与调试输出
# ------------------------------------------------------------

def demo_debug_technique_001_logging():
    """
    调试技巧1：日志配置与调试输出

    AutoGen使用Python的logging模块，可以通过配置日志级别
    来获取不同详细程度的调试信息。

    常用日志级别：
    - CRITICAL (50): 严重错误，导致程序无法继续
    - ERROR (40): 错误，但程序可以继续
    - WARNING (30): 警告，可能有问题
    - INFO (20): 一般信息
    - DEBUG (10): 调试信息，最详细

    使用verbose模式：
    - agent.verbose = True 可以启用详细输出
    - 会打印每条消息的详细信息
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("调试技巧1：日志配置与调试输出")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 技巧1.1：配置全局日志级别
    # ---------------------------------------------
    print("\n【技巧1.1】配置全局日志级别")

    print("""
    import logging

    # 配置日志格式
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 获取autogen的logger
    logger = logging.getLogger('autogen')
    logger.setLevel(logging.DEBUG)
    """)

    # 实际代码示例
    import logging as logging_module

    # 配置根日志
    logging_module.basicConfig(
        level=logging_module.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 获取autogen的logger并设置级别
    autogen_logger = logging_module.getLogger('autogen')
    autogen_logger.setLevel(logging_module.DEBUG)

    print("  配置步骤:")
    print("    1. 导入logging模块")
    print("    2. 配置basicConfig设置全局级别")
    print("    3. 获取autogen的logger并设置DEBUG级别")
    print("  效果：可以看到详细的LLM调用和消息传递过程")

    # ---------------------------------------------
    # 技巧1.2：使用verbose模式
    # ---------------------------------------------
    print("\n【技巧1.2】使用verbose模式")

    agent_verbose = ConversableAgent(
        name="VerboseAgent",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        verbose=True,  # 启用verbose模式
    )

    print("  正确代码:")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        system_message='你是一个有帮助的助手。',")
    print("        llm_config=llm_config,")
    print("        verbose=True,  # 启用详细输出")
    print("    )")
    print("  效果：每条消息都会打印详细信息")
    print("  适用场景：开发调试时使用，生产环境建议关闭")

    # ---------------------------------------------
    # 技巧1.3：自定义调试装饰器
    # ---------------------------------------------
    print("\n【技巧1.3】自定义调试装饰器")

    def debug_decorator(func: Callable) -> Callable:
        """
        调试装饰器：打印函数调用前后的状态

        使用方式：
        @debug_decorator
        def my_function(arg1, arg2):
            ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[DEBUG] 调用函数: {func.__name__}")
            print(f"[DEBUG] 参数: args={args}, kwargs={kwargs}")
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[DEBUG] 函数 {func.__name__} 执行完成，耗时: {elapsed:.3f}s")
            return result
        return wrapper

    print("  装饰器代码:")
    print("    def debug_decorator(func):")
    print("        @wraps(func)")
    print("        def wrapper(*args, **kwargs):")
    print("            print(f'[DEBUG] 调用函数: {func.__name__}')")
    print("            print(f'[DEBUG] 参数: args={args}, kwargs={kwargs}')")
    print("            start_time = time.time()")
    print("            result = func(*args, **kwargs)")
    print("            print(f'[DEBUG] 耗时: {time.time() - start_time:.3f}s')")
    print("            return result")
    print("        return wrapper")
    print("  使用：@debug_decorator 装饰需要调试的函数")

    # ---------------------------------------------
    # 技巧1.4：条件调试日志
    # ---------------------------------------------
    print("\n【技巧1.4】条件调试日志")

    # 根据环境变量决定是否启用调试
    DEBUG_MODE = os.getenv("AUTOGEN_DEBUG", "false").lower() == "true"

    def debug_log(message: str, data: Any = None):
        """
        条件调试日志：只在DEBUG模式下输出

        Args:
            message: 日志消息
            data: 附加数据（可选）
        """
        if DEBUG_MODE:
            print(f"[DEBUG] {message}")
            if data is not None:
                print(f"[DEBUG] 数据: {data}")

    print("  代码示例:")
    print("    DEBUG_MODE = os.getenv('AUTOGEN_DEBUG', 'false') == 'true'")
    print("")
    print("    def debug_log(message, data=None):")
    print("        if DEBUG_MODE:")
    print("            print(f'[DEBUG] {message}')")
    print("            if data is not None:")
    print("                print(f'[DEBUG] 数据: {data}')")
    print("  优势：可以通过环境变量控制开关")


# ------------------------------------------------------------
# 技巧2：消息历史分析
# ------------------------------------------------------------

def demo_debug_technique_002_message_history():
    """
    调试技巧2：消息历史分析

    通过分析chat_history可以：
    - 了解对话的完整流程
    - 识别异常消息
    - 检查消息内容是否符合预期
    - 分析对话轮次是否合理

    常用分析方法：
    1. 打印完整消息历史
    2. 统计每个Agent的发言次数
    3. 分析消息长度分布
    4. 检查特定消息内容
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("调试技巧2：消息历史分析")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建两个Agent进行测试
    agent_a = ConversableAgent(
        name="Agent_A",
        system_message="你是Agent A，负责提问。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="Agent_B",
        system_message="你是Agent B，负责回答问题。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
    )

    # 注意：这里只是演示分析方法，实际运行需要有效的LLM配置
    # result = agent_a.initiate_chat(agent_b, message="请介绍一下你自己")

    # ---------------------------------------------
    # 技巧2.1：打印完整消息历史
    # ---------------------------------------------
    print("\n【技巧2.1】打印完整消息历史")

    print("""
    # 在对话结束后打印消息历史
    def print_chat_history(chat_history, max_content_length=100):
        '''
        打印消息历史，每条消息最多显示指定长度

        Args:
            chat_history: 消息历史列表
            max_content_length: 每条消息的最大显示长度
        '''
        print(f"\\n总消息数: {len(chat_history)}")
        print("=" * 60)

        for i, msg in enumerate(chat_history):
            name = msg.get('name', 'unknown')
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')

            # 截断过长的内容
            display_content = content[:max_content_length]
            if len(content) > max_content_length:
                display_content += '...'

            print(f"[{i}] {role}/{name}: {display_content}")

        print("=" * 60)
    """)

    def print_chat_history(chat_history, max_content_length=100):
        """
        打印消息历史，每条消息最多显示指定长度
        """
        print(f"\n总消息数: {len(chat_history)}")
        print("=" * 60)

        for i, msg in enumerate(chat_history):
            name = msg.get('name', 'unknown')
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')

            display_content = content[:max_content_length]
            if len(content) > max_content_length:
                display_content += '...'

            print(f"[{i}] {role}/{name}: {display_content}")

        print("=" * 60)

    # 模拟消息历史
    mock_history = [
        {"role": "user", "name": "user", "content": "请介绍一下你自己"},
        {"role": "assistant", "name": "Agent_B", "content": "我是Agent B，一个智能助手。"},
        {"role": "assistant", "name": "Agent_B", "content": "我可以帮助你完成各种任务。"},
    ]

    print("  示例输出:")
    print("  -" * 60)
    print_chat_history(mock_history)

    # ---------------------------------------------
    # 技巧2.2：统计每个Agent的发言次数
    # ---------------------------------------------
    print("\n【技巧2.2】统计每个Agent的发言次数")

    def analyze_speaker_distribution(chat_history):
        """
        分析发言者分布

        Args:
            chat_history: 消息历史列表

        Returns:
            dict: 每个Agent的发言次数
        """
        speaker_counts = {}

        for msg in chat_history:
            name = msg.get('name', 'unknown')
            speaker_counts[name] = speaker_counts.get(name, 0) + 1

        return speaker_counts

    print("  代码示例:")
    print("    def analyze_speaker_distribution(chat_history):")
    print("        speaker_counts = {}")
    print("        for msg in chat_history:")
    print("            name = msg.get('name', 'unknown')")
    print("            speaker_counts[name] = speaker_counts.get(name, 0) + 1")
    print("        return speaker_counts")
    print("  效果：")
    print(f"    {analyze_speaker_distribution(mock_history)}")

    # ---------------------------------------------
    # 技巧2.3：分析消息长度分布
    # ---------------------------------------------
    print("\n【技巧2.3】分析消息长度分布")

    def analyze_message_lengths(chat_history):
        """
        分析消息长度分布

        Returns:
            dict: 包含统计信息
        """
        lengths = [len(msg.get('content', '')) for msg in chat_history]

        return {
            'total': sum(lengths),
            'average': sum(lengths) / len(lengths) if lengths else 0,
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
        }

    print("  代码示例:")
    print("    def analyze_message_lengths(chat_history):")
    print("        lengths = [len(msg.get('content', '')) for msg in chat_history]")
    print("        return {")
    print("            'total': sum(lengths),")
    print("            'average': sum(lengths) / len(lengths),")
    print("            'min': min(lengths),")
    print("            'max': max(lengths),")
    print("        }")
    print("  效果：")
    for k, v in analyze_message_lengths(mock_history).items():
        print(f"    {k}: {v}")

    # ---------------------------------------------
    # 技巧2.4：查找特定消息
    # ---------------------------------------------
    print("\n【技巧2.4】查找特定消息")

    def find_messages_by_keyword(chat_history, keyword):
        """
        查找包含特定关键词的消息

        Args:
            chat_history: 消息历史列表
            keyword: 关键词

        Returns:
            list: 匹配的消息
        """
        results = []

        for i, msg in enumerate(chat_history):
            content = msg.get('content', '')
            if keyword.lower() in content.lower():
                results.append({
                    'index': i,
                    'name': msg.get('name', 'unknown'),
                    'content': content,
                })

        return results

    print("  代码示例:")
    print("    def find_messages_by_keyword(chat_history, keyword):")
    print("        results = []")
    print("        for i, msg in enumerate(chat_history):")
    print("            if keyword.lower() in msg.get('content', '').lower():")
    print("                results.append({...})")
    print("        return results")
    print("  使用：find_messages_by_keyword(history, '完成')")


# ------------------------------------------------------------
# 技巧3：Agent状态检查
# ------------------------------------------------------------

def demo_debug_technique_003_agent_state():
    """
    调试技巧3：Agent状态检查

    通过检查Agent的各种属性，可以了解Agent的配置状态：
    - llm_config: LLM配置是否正确
    - human_input_mode: 人类输入模式
    - max_consecutive_auto_reply: 最大连续回复数
    - chat_messages: 当前消息历史
    - 等等
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("调试技巧3：Agent状态检查")
    print("=" * 60)

    llm_config = build_llm_config()

    agent = ConversableAgent(
        name="TestAgent",
        system_message="你是一个测试Agent。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
    )

    # ---------------------------------------------
    # 技巧3.1：打印Agent配置状态
    # ---------------------------------------------
    print("\n【技巧3.1】打印Agent配置状态")

    def print_agent_state(agent: ConversableAgent):
        """
        打印Agent的配置状态

        Args:
            agent: ConversableAgent实例
        """
        print(f"\nAgent名称: {agent.name}")
        print("-" * 40)
        print(f"系统消息: {agent.system_message[:50]}...")
        print(f"人类输入模式: {agent.human_input_mode}")
        print(f"最大连续回复数: {agent.max_consecutive_auto_reply}")
        print(f"LLM配置: {'已设置' if agent.llm_config else '未设置'}")

        if hasattr(agent, 'chat_messages') and agent.chat_messages:
            print(f"当前对话数: {len(agent.chat_messages)}")
        else:
            print("当前对话数: 0")

    print("  代码示例:")
    print("    def print_agent_state(agent):")
    print("        print(f'Agent名称: {agent.name}')")
    print("        print(f'人类输入模式: {agent.human_input_mode}')")
    print("        print(f'最大连续回复数: {agent.max_consecutive_auto_reply}')")
    print("        print(f'LLM配置: {\\'已设置\\' if agent.llm_config else \\'未设置\\'}')")

    print("\n  实际输出:")
    print_agent_state(agent)

    # ---------------------------------------------
    # 技巧3.2：检查LLM配置有效性
    # ---------------------------------------------
    print("\n【技巧3.2】检查LLM配置有效性")

    def check_llm_config(agent: ConversableAgent) -> Tuple[bool, str]:
        """
        检查LLM配置是否有效

        Args:
            agent: ConversableAgent实例

        Returns:
            (is_valid, error_message)
        """
        if not agent.llm_config:
            return False, "LLM配置未设置"

        config_list = agent.llm_config.get('config_list', [])
        if not config_list:
            return False, "config_list为空"

        for i, config in enumerate(config_list):
            if 'model' not in config:
                return False, f"config[{i}]缺少model字段"

        return True, "配置有效"

    print("  代码示例:")
    print("    def check_llm_config(agent):")
    print("        if not agent.llm_config:")
    print("            return False, 'LLM配置未设置'")
    print("        config_list = agent.llm_config.get('config_list', [])")
    print("        if not config_list:")
    print("            return False, 'config_list为空'")
    print("        for i, config in enumerate(config_list):")
    print("            if 'model' not in config:")
    print("                return False, f'config[{i}]缺少model字段'")
    print("        return True, '配置有效'")
    print(f"  检查结果: {check_llm_config(agent)}")

    # ---------------------------------------------
    # 技巧3.3：监控对话状态
    # ---------------------------------------------
    print("\n【技巧3.3】监控对话状态")

    def monitor_conversation_state(agent: ConversableAgent):
        """
        监控对话状态

        Returns:
            dict: 对话状态信息
        """
        state = {
            'name': agent.name,
            'chat_messages_count': 0,
            'consecutive_replies': 0,
            'active_chats': [],
        }

        if hasattr(agent, 'chat_messages') and agent.chat_messages:
            state['chat_messages_count'] = len(agent.chat_messages)
            state['active_chats'] = list(agent.chat_messages.keys())

        return state

    print("  代码示例:")
    print("    def monitor_conversation_state(agent):")
    print("        state = {")
    print("            'name': agent.name,")
    print("            'chat_messages_count': len(agent.chat_messages) if agent.chat_messages else 0,")
    print("            'active_chats': list(agent.chat_messages.keys()) if agent.chat_messages else [],")
    print("        }")
    print("        return state")
    print(f"  状态: {monitor_conversation_state(agent)}")


# ------------------------------------------------------------
# 技巧4：GroupChat监控
# ------------------------------------------------------------

def demo_debug_technique_004_groupchat_monitoring():
    """
    调试技巧4：GroupChat监控

    GroupChat的监控重点：
    1. speaker选择过程
    2. 消息广播
    3. 终止条件触发
    4. 发言者分布统计
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("调试技巧4：GroupChat监控")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建多个Agent
    agent_1 = ConversableAgent(
        name="Agent_1",
        system_message="你是Agent 1。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_2 = ConversableAgent(
        name="Agent_2",
        system_message="你是Agent 2。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agent_3 = ConversableAgent(
        name="Agent_3",
        system_message="你是Agent 3。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 技巧4.1：监控speaker选择过程
    # ---------------------------------------------
    print("\n【技巧4.1】监控speaker选择过程")

    def create_monitoring_groupchat(agents, max_round=10):
        """
        创建带监控的GroupChat

        每次选择speaker后打印日志
        """
        # 选择函数的历史记录
        selection_history = []

        def monitored_select_speaker(groupchat: GroupChat, last_speaker=None):
            """
            带监控的speaker选择函数
            """
            # 执行实际的选择逻辑
            selected = groupchat.select_speaker(last_speaker)

            # 记录选择过程
            selection_history.append({
                'last_speaker': last_speaker.name if last_speaker else None,
                'selected': selected.name,
                'round': len(groupchat.messages),
            })

            # 打印日志
            print(f"[MONITOR] Round {len(groupchat.messages)}: "
                  f"{last_speaker.name if last_speaker else 'None'} -> {selected.name}")

            return selected

        groupchat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round,
            speaker_selection_method=monitored_select_speaker,
        )

        return groupchat, selection_history

    print("  监控函数:")
    print("    def monitored_select_speaker(groupchat, last_speaker):")
    print("        selected = groupchat.select_speaker(last_speaker)")
    print("        print(f'[MONITOR] Round {len(groupchat.messages)}: '")
    print("              f'{last_speaker} -> {selected}')")
    print("        return selected")

    # 演示
    print("\n  示例输出:")
    print("    [MONITOR] Round 0: None -> Agent_1")
    print("    [MONITOR] Round 1: Agent_1 -> Agent_2")
    print("    [MONITOR] Round 2: Agent_2 -> Agent_3")

    # ---------------------------------------------
    # 技巧4.2：监控终止条件触发
    # ---------------------------------------------
    print("\n【技巧4.2】监控终止条件触发")

    def create_logged_termination(condition_func: Callable, name: str = "termination"):
        """
        创建带日志的终止条件函数

        Args:
            condition_func: 原始终止条件函数
            name: 条件名称

        Returns:
            包装后的终止条件函数
        """
        def logged_condition(msg):
            result = condition_func(msg)
            content_preview = msg.get('content', '')[:30]
            print(f"[{name.upper()}] '{content_preview}...' -> {result}")
            return result

        return logged_condition

    def sample_termination(msg):
        return "完成" in msg.get("content", "")

    logged_termination = create_logged_termination(sample_termination)

    print("  代码示例:")
    print("    def create_logged_termination(condition_func, name='termination'):")
    print("        def logged_condition(msg):")
    print("            result = condition_func(msg)")
    print("            content_preview = msg.get('content', '')[:30]")
    print("            print(f'[{name.upper()}] \\'{content_preview}\\' -> {result}')")
    print("            return result")
    print("        return logged_condition")
    print("  效果：每次检查终止条件都会打印日志")

    # 演示
    test_msg = {"content": "任务完成，请结束对话"}
    print("\n  测试:")
    logged_termination(test_msg)

    # ---------------------------------------------
    # 技巧4.3：分析发言者分布
    # ---------------------------------------------
    print("\n【技巧4.3】分析发言者分布")

    def analyze_speaker_stats(messages: List[Dict]) -> Dict[str, Any]:
        """
        分析GroupChat中发言者的统计信息

        Args:
            messages: 消息历史列表

        Returns:
            dict: 统计信息
        """
        if not messages:
            return {'total_messages': 0, 'speakers': {}}

        speaker_counts = {}
        message_lengths = {}

        for msg in messages:
            name = msg.get('name', 'unknown')
            content = msg.get('content', '')

            speaker_counts[name] = speaker_counts.get(name, 0) + 1
            message_lengths[name] = message_lengths.get(name, []) + [len(content)]

        # 计算每个发言者的平均消息长度
        avg_lengths = {
            name: sum(lengths) / len(lengths) if lengths else 0
            for name, lengths in message_lengths.items()
        }

        return {
            'total_messages': len(messages),
            'speakers': speaker_counts,
            'average_lengths': avg_lengths,
        }

    print("  代码示例:")
    print("    def analyze_speaker_stats(messages):")
    print("        speaker_counts = {}")
    print("        for msg in messages:")
    print("            name = msg.get('name', 'unknown')")
    print("            speaker_counts[name] = speaker_counts.get(name, 0) + 1")
    print("        return {'speakers': speaker_counts}")

    # 模拟数据
    mock_messages = [
        {"name": "Agent_1", "content": "消息1" * 50},
        {"name": "Agent_2", "content": "消息2" * 30},
        {"name": "Agent_1", "content": "消息3" * 40},
        {"name": "Agent_3", "content": "消息4" * 20},
    ]

    print("\n  示例统计:")
    stats = analyze_speaker_stats(mock_messages)
    print(f"    总消息数: {stats['total_messages']}")
    print(f"    发言者分布: {stats['speakers']}")


# ------------------------------------------------------------
# 技巧5：性能分析与成本追踪
# ------------------------------------------------------------

def demo_debug_technique_005_performance_tracking():
    """
    调试技巧5：性能分析与成本追踪

    通过监控Token消耗和响应时间，可以：
    - 评估系统性能
    - 控制成本
    - 识别性能瓶颈
    """
    from autogen import ChatCompletion

    print("\n" + "=" * 60)
    print("调试技巧5：性能分析与成本追踪")
    print("=" * 60)

    # ---------------------------------------------
    # 技巧5.1：计算对话成本
    # ---------------------------------------------
    print("\n【技巧5.1】计算对话成本")

    def calculate_chat_cost(prompt_tokens: int, completion_tokens: int,
                           model: str = "gpt-4o-mini") -> float:
        """
        计算对话成本

        Args:
            prompt_tokens: 提示token数
            completion_tokens: 完成token数
            model: 模型名称

        Returns:
            float: 成本（美元）
        """
        # 价格配置（每1000 token的价格）
        prices = {
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo": (0.001, 0.002),
        }

        if model not in prices:
            print(f"警告：{model}不在价格表中，使用gpt-4o-mini的价格")
            model = "gpt-4o-mini"

        input_price, output_price = prices[model]

        cost = (prompt_tokens * input_price / 1000 +
                completion_tokens * output_price / 1000)

        return cost

    print("  代码示例:")
    print("    def calculate_chat_cost(prompt_tokens, completion_tokens, model):")
    print("        prices = {")
    print("            'gpt-4o': (0.005, 0.015),")
    print("            'gpt-4o-mini': (0.00015, 0.0006),")
    print("        }")
    print("        input_price, output_price = prices[model]")
    print("        cost = (prompt_tokens * input_price +")
    print("               completion_tokens * output_price) / 1000")
    print("        return cost")

    print("\n  示例计算:")
    cost = calculate_chat_cost(prompt_tokens=1000, completion_tokens=500, model="gpt-4o-mini")
    print(f"    输入1000 token + 输出500 token = ${cost:.6f}")

    # ---------------------------------------------
    # 技巧5.2：追踪响应时间
    # ---------------------------------------------
    print("\n【技巧5.2】追踪响应时间")

    def time_llm_call(func: Callable) -> Callable:
        """
        装饰器：追踪LLM调用时间

        使用方式：
        @time_llm_call
        def call_llm(...):
            ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[TIMING] {func.__name__} 耗时: {elapsed:.3f}s")
            return result
        return wrapper

    print("  代码示例:")
    print("    def time_llm_call(func):")
    print("        @wraps(func)")
    print("        def wrapper(*args, **kwargs):")
    print("            start_time = time.time()")
    print("            result = func(*args, **kwargs)")
    print("            print(f'[TIMING] {func.__name__} 耗时: {time.time() - start_time:.3f}s')")
    print("            return result")
    print("        return wrapper")
    print("  使用：@time_llm_call 装饰需要计时的函数")

    # ---------------------------------------------
    # 技巧5.3：性能监控器
    # ---------------------------------------------
    print("\n【技巧5.3】性能监控器")

    class PerformanceMonitor:
        """
        性能监控器：追踪LLM调用的各项指标
        """

        def __init__(self):
            self.calls = []
            self.total_tokens = 0
            self.total_cost = 0.0

        def record_call(self, prompt_tokens: int, completion_tokens: int,
                       model: str, elapsed_time: float):
            """
            记录一次LLM调用

            Args:
                prompt_tokens: 提示token数
                completion_tokens: 完成token数
                model: 模型名称
                elapsed_time: 耗时（秒）
            """
            cost = calculate_chat_cost(prompt_tokens, completion_tokens, model)

            self.calls.append({
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'model': model,
                'cost': cost,
                'elapsed_time': elapsed_time,
            })

            self.total_tokens += prompt_tokens + completion_tokens
            self.total_cost += cost

        def get_stats(self) -> Dict[str, Any]:
            """
            获取统计信息
            """
            if not self.calls:
                return {'total_calls': 0}

            elapsed_times = [c['elapsed_time'] for c in self.calls]

            return {
                'total_calls': len(self.calls),
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'avg_response_time': sum(elapsed_times) / len(elapsed_times),
                'min_response_time': min(elapsed_times),
                'max_response_time': max(elapsed_times),
            }

        def print_report(self):
            """
            打印性能报告
            """
            stats = self.get_stats()

            print("\n" + "=" * 40)
            print("性能报告")
            print("=" * 40)
            print(f"总调用次数: {stats['total_calls']}")
            print(f"总Token数: {stats['total_tokens']:,}")
            print(f"总成本: ${stats['total_cost']:.6f}")

            if 'avg_response_time' in stats:
                print(f"平均响应时间: {stats['avg_response_time']:.3f}s")
                print(f"最快响应时间: {stats['min_response_time']:.3f}s")
                print(f"最慢响应时间: {stats['max_response_time']:.3f}s")

            print("=" * 40)

    print("  代码示例:")
    print("    class PerformanceMonitor:")
    print("        def __init__(self):")
    print("            self.calls = []")
    print("            self.total_tokens = 0")
    print("            self.total_cost = 0.0")
    print("")
    print("        def record_call(self, prompt_tokens, completion_tokens, model, elapsed_time):")
    print("            self.calls.append({...})")
    print("            self.total_tokens += prompt_tokens + completion_tokens")
    print("")
    print("        def print_report(self):")
    print("            stats = self.get_stats()")
    print("            print(f'总调用次数: {stats[\\'total_calls\\']}')")
    print("            print(f'总成本: ${stats[\\'total_cost\\']:.6f}')")

    # 演示
    monitor = PerformanceMonitor()
    monitor.record_call(1000, 500, "gpt-4o-mini", 1.234)
    monitor.record_call(800, 300, "gpt-4o-mini", 0.987)
    monitor.record_call(1500, 800, "gpt-4o-mini", 2.456)
    monitor.print_report()


# ------------------------------------------------------------
# 技巧6：常见问题诊断流程
# ------------------------------------------------------------

def demo_debug_technique_006_diagnosis_flow():
    """
    调试技巧6：常见问题诊断流程

    提供标准化的诊断流程，帮助快速定位问题
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("调试技巧6：常见问题诊断流程")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 诊断流程：对话无法终止
    # ---------------------------------------------
    print("\n【诊断流程1】对话无法终止")

    print("""
    诊断步骤：
    1. 检查is_termination_msg是否正确配置
       - 条件是否过于严格？
       - 是否永远无法满足？

    2. 检查max_consecutive_auto_reply设置
       - 是否设置为None？
       - 是否设置为过大的值？

    3. 检查GroupChat的max_round
       - 是否设置为过大的值？

    4. 检查Agent的系统提示
       - 是否包含可能导致无限循环的指令？

    5. 检查LLM是否陷入重复模式
       - 多次对话后是否开始重复？

    检查清单：
    [ ] is_termination_msg返回True的条件是否合理？
    [ ] max_consecutive_auto_reply是否设置？
    [ ] GroupChat的max_round是否合理？
    [ ] Agent系统提示是否包含循环指令？
    """)

    # ---------------------------------------------
    # 诊断流程：Agent不响应
    # ---------------------------------------------
    print("\n【诊断流程2】Agent不响应")

    print("""
    诊断步骤：
    1. 检查llm_config配置
       - 是否设置为False？
       - config_list是否正确？

    2. 检查human_input_mode
       - 是否设置为ALWAYS？（会等待人类输入）
       - 是否设置为TERMINATE？（会在特定条件停止）

    3. 检查Agent类型
       - 是否是UserProxyAgent但没有正确配置？

    4. 检查消息传递
       - initiate_chat是否正确调用？
       - 消息是否正确传递给Agent？

    检查清单：
    [ ] llm_config是否正确配置？
    [ ] human_input_mode是否正确？
    [ ] Agent类型是否匹配场景？
    [ ] initiate_chat参数是否正确？
    """)

    # ---------------------------------------------
    # 诊断流程：GroupChat发言不均
    # ---------------------------------------------
    print("\n【诊断流程3】GroupChat发言不均")

    print("""
    诊断步骤：
    1. 检查speaker_selection_method
       - 是否使用auto模式？
       - LLM可能存在偏好

    2. 检查Agent系统提示
       - 不同Agent的提示是否差异过大？
       - 某些Agent是否更健谈？

    3. 检查Agent数量
       - Agent数量是否不均衡？

    4. 检查消息内容
       - 是否某些话题只与特定Agent相关？

    解决方案：
    - 使用round_robin强制均衡
    - 使用manual模式手动控制
    - 自定义均衡选择函数

    检查清单：
    [ ] speaker_selection_method是否合适？
    [ ] Agent系统提示是否平衡？
    [ ] 是否需要使用round_robin？
    """)

    # ---------------------------------------------
    # 诊断流程：成本异常高
    # ---------------------------------------------
    print("\n【诊断流程4】成本异常高")

    print("""
    诊断步骤：
    1. 检查Token消耗
       - 对话轮次是否过多？
       - 每次回复的Token数是否过大？

    2. 检查max_consecutive_auto_reply
       - 是否设置为None或过大的值？
       - 是否触发无限循环？

    3. 检查模型选择
       - 是否使用了过大的模型？
       - 是否需要对不同任务使用不同模型？

    4. 检查price配置
       - 是否为每个模型配置了price字段？
       - 价格是否正确？

    解决方案：
    - 设置max_consecutive_auto_reply限制
    - 使用更小的模型处理简单任务
    - 配置price字段进行成本监控
    - 设置预算上限

    检查清单：
    [ ] max_consecutive_auto_reply是否设置？
    [ ] 是否配置了price字段？
    [ ] 模型选择是否合理？
    [ ] 对话轮次是否过多？
    """)

    # ---------------------------------------------
    # 综合诊断脚本
    # ---------------------------------------------
    print("\n【综合诊断脚本】")

    def diagnose_autogen_issue(agent_or_groupchat):
        """
        AutoGen问题综合诊断函数

        Args:
            agent_or_groupchat: Agent或GroupChat实例

        Returns:
            dict: 诊断结果
        """
        issues = []
        warnings = []

        # 诊断Agent
        if isinstance(agent_or_groupchat, ConversableAgent):
            agent = agent_or_groupchat

            # 检查1：llm_config
            if not agent.llm_config:
                issues.append("llm_config未设置，Agent无法自动回复")

            # 检查2：max_consecutive_auto_reply
            if agent.max_consecutive_auto_reply is None:
                warnings.append("max_consecutive_auto_reply为None，使用默认值")

            # 检查3：human_input_mode
            if agent.human_input_mode == "ALWAYS":
                warnings.append("human_input_mode='ALWAYS'会等待人类输入")

            # 检查4：is_termination_msg
            if agent.is_termination_msg is None:
                warnings.append("is_termination_msg未设置")

        # 诊断GroupChat
        elif isinstance(agent_or_groupchat, GroupChat):
            groupchat = agent_or_groupchat

            # 检查1：speaker_selection_method
            if groupchat.speaker_selection_method == "auto":
                warnings.append("speaker_selection_method='auto'可能导致发言不均")

            # 检查2：max_round
            if groupchat.max_round > 50:
                warnings.append("max_round设置过大，可能导致成本过高")

            # 检查3：termination_msg
            if groupchat.termination_msg is None:
                warnings.append("termination_msg未设置，对话可能无法正常终止")

        return {
            'issues': issues,
            'warnings': warnings,
            'is_healthy': len(issues) == 0,
        }

    print("  代码示例:")
    print("    def diagnose_autogen_issue(agent_or_groupchat):")
    print("        issues = []")
    print("        warnings = []")
    print("        if isinstance(agent, ConversableAgent):")
    print("            if not agent.llm_config:")
    print("                issues.append('llm_config未设置')")
    print("            if agent.max_consecutive_auto_reply is None:")
    print("                warnings.append('max_consecutive_auto_reply为None')")
    print("        elif isinstance(groupchat, GroupChat):")
    print("            if groupchat.speaker_selection_method == 'auto':")
    print("                warnings.append('speaker_selection_method=auto可能导致发言不均')")
    print("        return {'issues': issues, 'warnings': warnings, 'is_healthy': len(issues) == 0}")

    # 演示诊断
    demo_agent = ConversableAgent(
        name="DemoAgent",
        system_message="你是一个测试Agent。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=None,  # 使用默认值
    )

    print("\n  诊断示例:")
    result = diagnose_autogen_issue(demo_agent)
    print(f"    健康状态: {result['is_healthy']}")
    print(f"    问题: {result['issues']}")
    print(f"    警告: {result['warnings']}")


# ============================================================
# 第四部分：调试清单
# ============================================================

def print_debug_checklist():
    """
    打印AutoGen开发调试清单
    """
    print("\n" + "=" * 60)
    print("AutoGen开发调试清单")
    print("=" * 60)

    print("""
    【配置检查】
    [ ] llm_config是否正确配置？（不使用False，除非是纯代码执行）
    [ ] config_list是否包含有效的模型配置？
    [ ] 是否为非OpenAI模型配置了price字段？
    [ ] base_url是否正确（如果使用代理）？

    【Agent配置】
    [ ] max_consecutive_auto_reply是否显式设置？
    [ ] human_input_mode是否正确（NEVER/ALWAYS/TERMINATE）？
    [ ] is_termination_msg是否合理配置？
    [ ] system_message是否包含可能导致问题的指令？

    【GroupChat配置】
    [ ] speaker_selection_method是否合适？
    [ ] max_round是否设置合理？
    [ ] termination_msg是否正确配置？
    [ ] agents列表是否正确？

    【调试技巧】
    [ ] 是否启用了verbose模式进行调试？
    [ ] 是否打印了chat_history进行分析？
    [ ] 是否监控了发言者分布？
    [ ] 是否追踪了Token消耗和成本？

    【常见问题快速修复】
    [ ] 对话无法终止 -> 检查is_termination_msg和max_consecutive_auto_reply
    [ ] Agent不响应 -> 检查llm_config和human_input_mode
    [ ] 发言不均 -> 使用round_robin或调整speaker_selection_method
    [ ] 成本过高 -> 设置max_consecutive_auto_reply，配置price
    """)


# ============================================================
# 第五部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有调试技巧演示
    """
    print("=" * 60)
    print("AutoGen调试技巧代码演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个调试技巧演示
    demo_debug_technique_001_logging()
    demo_debug_technique_002_message_history()
    demo_debug_technique_003_agent_state()
    demo_debug_technique_004_groupchat_monitoring()
    demo_debug_technique_005_performance_tracking()
    demo_debug_technique_006_diagnosis_flow()

    print_debug_checklist()

    print("\n" + "=" * 60)
    print("调试技巧演示完成")
    print("=" * 60)
    print("\n总结：6大调试技巧")
    print("  技巧1: 日志配置与调试输出 - 启用verbose和DEBUG日志")
    print("  技巧2: 消息历史分析 - 打印和分析chat_history")
    print("  技巧3: Agent状态检查 - 检查Agent配置属性")
    print("  技巧4: GroupChat监控 - 监控speaker选择和终止条件")
    print("  技巧5: 性能分析与成本追踪 - 追踪Token和成本")
    print("  技巧6: 常见问题诊断流程 - 提供标准化诊断流程")


if __name__ == "__main__":
    main()