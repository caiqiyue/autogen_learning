# pitfalls_demo.py
# 第27节 AutoGen高频误区与调试技巧 - 常见误区代码演示
#
# 本文件通过代码示例展示AutoGen开发中的8个高频误区：
# 1. is_termination_msg条件设置过宽/过严
# 2. max_consecutive_auto_reply设置为None导致无限循环
# 3. GroupChat中忘记设置speaker_selection_mode
# 4. Tool Call与Code Executor混用导致行为不一致
# 5. 在async代码中使用同步generate_reply导致死锁
# 6. llm_config设置为False但仍期望Agent自动生成回复
# 7. 嵌套GroupChat中的状态污染
# 8. 未配置price字段导致成本计算不准确
#
# ============================================================
# 误区概览
# ============================================================
#
# 误区编号    严重程度    发生频率    排查难度
# ------------------------------------------------
# case_001    高          高          低
# case_002    高          中          低
# case_003    中          高          中
# case_004    高          中          高
# case_005    高          低          高
# case_006    中          高          低
# case_007    高          低          高
# case_008    中          中          低
# ============================================================

import os
import re
import asyncio
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

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
# 第三部分：误区案例代码
# ============================================================

# ------------------------------------------------------------
# 误区1：is_termination_msg条件设置过宽导致对话无法终止
# ------------------------------------------------------------

def demo_pitfall_001_weak_termination():
    """
    误区1：is_termination_msg条件设置过宽导致对话无法终止

    问题描述：
    - 终止条件设置过于宽泛，如任何包含句号的消息都终止
    - 导致对话过早终止，任务未完成
    - 或终止条件过于严格，永远无法满足，对话无法结束

    根因分析：
    - 对消息内容的多样性估计不足
    - 条件判断过于简单，未考虑边界情况
    - 例如：只检查"再见"，但Agent很少说"再见"

    后果评估：
    - 严重程度：高
    - GroupChat陷入死循环或用户无法获得最终结果
    - Token消耗失控
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("误区1：is_termination_msg条件设置过宽")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 错误示例1：条件过于宽泛
    # ---------------------------------------------
    print("\n【错误示例1】条件过于宽泛 - 任何包含'。'就终止")

    def weak_termination_too_broad(msg):
        """
        错误：条件过于宽泛
        任何包含句号的消息都会触发终止
        """
        # 这个条件太宽泛了！
        return "。" in msg.get("content", "")

    # 这会导致第一条包含句号的消息就终止对话
    agent_a = ConversableAgent(
        name="Agent_A",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=weak_termination_too_broad,
    )

    agent_b = ConversableAgent(
        name="Agent_B",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=weak_termination_too_broad,
    )

    print("  错误代码:")
    print("    def weak_termination(msg):")
    print("        return '。' in msg.get('content', '')")
    print("  问题：第一条消息包含句号就终止")

    # ---------------------------------------------
    # 错误示例2：条件过于严格
    # ---------------------------------------------
    print("\n【错误示例2】条件过于严格 - 需要同时满足多个不太可能同时出现的词")

    def weak_termination_too_strict(msg):
        """
        错误：条件过于严格
        需要同时包含多个不太可能同时出现的关键词
        """
        content = msg.get("content", "")
        # 这个条件几乎不可能满足
        return ("任务完成" in content and
                "结论如下" in content and
                "再见" in content)

    print("  错误代码:")
    print("    def strict_termination(msg):")
    print("        return ('任务完成' in content and")
    print("                '结论如下' in content and")
    print("                '再见' in content)")
    print("  问题：条件几乎不可能同时满足，对话无法终止")

    # ---------------------------------------------
    # 正确示例
    # ---------------------------------------------
    print("\n【正确示例】多条件组合的终止条件")

    def correct_termination(msg):
        """
        正确：多条件组合，平衡宽泛和严格

        终止条件（满足任一即可）：
        1. 包含明确的完成标记（多种）
        2. 包含退出指令
        3. 消息来自特定Agent且包含结论性内容
        """
        content = msg.get("content", "").lower()
        name = msg.get("name", "")

        # 完成标记（任一即可）
        completion_markers = ["完成", "结束", "结论", "搞定", "可以了"]
        if any(marker in content for marker in completion_markers):
            return True

        # 退出指令
        exit_markers = ["不需要再讨论", "到此为止", "再见"]
        if any(marker in content for marker in exit_markers):
            return True

        # 来自特定Agent的结论性消息
        if name == "总结员" and len(content) > 50:
            return True

        return False

    print("  正确代码:")
    print("    def correct_termination(msg):")
    print("        content = msg.get('content', '').lower()")
    print("        # 包含任一完成标记即终止")
    print("        completion_markers = ['完成', '结束', '结论', '搞定']")
    print("        if any(m in content for m in completion_markers):")
    print("            return True")
    print("        # 包含退出指令")
    print("        exit_markers = ['不需要再讨论', '到此为止']")
    print("        if any(m in content for m in exit_markers):")
    print("            return True")
    print("        return False")

    print("\n  修复要点:")
    print("    1. 提供多种完成标记，避免单一关键词依赖")
    print("    2. 包含退出指令作为备用终止方式")
    print("    3. 可以根据消息来源设置不同条件")


# ------------------------------------------------------------
# 误区2：max_consecutive_auto_reply设置为None导致无限循环
# ------------------------------------------------------------

def demo_pitfall_002_max_consecutive_none():
    """
    误区2：max_consecutive_auto_reply设置为None导致无限循环

    问题描述：
    - 未显式设置max_consecutive_auto_reply，依赖默认值
    - 误以为None表示无限制，实际上None有默认值
    - Agent持续自说自话，无法停止

    根因分析：
    - MAX_CONSECUTIVE_AUTO_REPLY类属性有默认值(100)
    - 误以为None表示无限制
    - 在某些复杂场景下，默认值仍可能触发大量循环

    后果评估：
    - 严重程度：高
    - Token消耗失控
    - API成本暴涨
    - 可能触发LLM的重复模式
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("误区2：max_consecutive_auto_reply设置为None")
    print("=" * 60)

    llm_config = build_llm_config()

    # ---------------------------------------------
    # 错误示例：依赖默认的max_consecutive_auto_reply
    # ---------------------------------------------
    print("\n【错误示例】不设置max_consecutive_auto_reply")

    # 注意：ConversableAgent的默认max_consecutive_auto_reply=100
    # 但在嵌套对话或复杂场景下可能不够用
    agent_default = ConversableAgent(
        name="Agent_Default",
        system_message="你是一个固执的助手，会不断提问。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        # 没有显式设置max_consecutive_auto_reply
        # 使用默认值100，在复杂场景下可能触发多次循环
    )

    print("  错误代码:")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        system_message='你是一个固执的助手，会不断提问。',")
    print("        llm_config=llm_config,")
    print("        human_input_mode='NEVER',")
    print("        # 没有设置max_consecutive_auto_reply")
    print("    )")
    print("  问题：依赖默认值，在复杂场景下可能触发多次循环")

    # ---------------------------------------------
    # 正确示例：显式设置合理的值
    # ---------------------------------------------
    print("\n【正确示例】显式设置max_consecutive_auto_reply")

    agent_correct = ConversableAgent(
        name="Agent_Correct",
        system_message="你是一个固执的助手，会不断提问。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,  # 显式设置，限制最大连续回复数
    )

    print("  正确代码:")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        system_message='你是一个固执的助手，会不断提问。',")
    print("        llm_config=llm_config,")
    print("        human_input_mode='NEVER',")
    print("        max_consecutive_auto_reply=5,  # 显式设置")
    print("    )")

    print("\n  修复要点:")
    print("    1. 始终显式设置max_consecutive_auto_reply")
    print("    2. 根据任务复杂度选择合适的值:")
    print("       - 简单问答: 1-2")
    print("       - 标准任务: 5-10")
    print("       - 复杂任务: 10-20")
    print("    3. 设置过小可能导致任务未完成就停止，需要权衡")

    # ---------------------------------------------
    # 陷阱：设置为0的特殊含义
    # ---------------------------------------------
    print("\n【陷阱】max_consecutive_auto_reply=0 表示完全禁用自动回复")

    agent_zero = ConversableAgent(
        name="Agent_Zero",
        system_message="这是一个特殊的agent。",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,  # 这意味着完全禁用自动回复！
    )

    print("  max_consecutive_auto_reply=0 的含义:")
    print("    - Agent不会自动生成任何回复")
    print("    - 必须通过其他方式触发回复（如initiate_chat）")
    print("    - 适用于纯人工输入场景")
    print("  适用场景:")
    print("    - 需要完全人工控制的对话")
    print("    - 作为人工代理的入口点")


# ------------------------------------------------------------
# 误区3：GroupChat中忘记设置speaker_selection_mode
# ------------------------------------------------------------

def demo_pitfall_003_speaker_selection_mode():
    """
    误区3：GroupChat中忘记设置speaker_selection_mode导致随机发言

    问题描述：
    - 未理解auto模式的LLM推荐逻辑
    - 可能推荐不合适的下一个发言者
    - 对话质量不可控、发言顺序混乱

    根因分析：
    - 使用默认的auto模式
    - LLM可能根据偏好选择而非任务需求
    - 在复杂对话中，缺乏对发言顺序的控制

    后果评估：
    - 严重程度：中
    - 对话质量不可控
    - 某些Agent可能被忽视
    - 发言顺序可能不符合业务逻辑
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("误区3：GroupChat中忘记设置speaker_selection_mode")
    print("=" * 60)

    llm_config = build_llm_config()

    # 创建三个Agent
    analyst = ConversableAgent(
        name="分析师",
        system_message="你是一位数据分析师，负责分析数据并给出见解。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    developer = ConversableAgent(
        name="开发者",
        system_message="你是一位后端开发者，负责实现技术方案。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    tester = ConversableAgent(
        name="测试员",
        system_message="你是一位测试工程师，负责验证方案的质量。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 错误示例：使用默认的auto模式
    # ---------------------------------------------
    print("\n【错误示例】使用默认的auto模式，不做任何控制")

    # 注意：GroupChat默认speaker_selection_method="auto"
    # 但auto模式的LLM推荐可能不符合预期
    groupchat_default = GroupChat(
        agents=[analyst, developer, tester],
        messages=[],
        max_round=10,
        # 没有显式设置speaker_selection_method
        # 默认是"auto"，LLM可能选择不合适的发言者
    )

    print("  错误代码:")
    print("    groupchat = GroupChat(")
    print("        agents=[analyst, developer, tester],")
    print("        messages=[],")
    print("        max_round=10,")
    print("        # 没有设置speaker_selection_method")
    print("    )")
    print("  问题：LLM可能随机或根据偏好选择发言者")
    print("  场景：")
    print("    - 分析师可能连续发言多次")
    print("    - 测试员的意见可能被忽视")
    print("    - 发言顺序不符合业务逻辑")

    # ---------------------------------------------
    # 正确示例1：使用round_robin强制均衡
    # ---------------------------------------------
    print("\n【正确示例1】使用round_robin强制均衡发言")

    groupchat_rr = GroupChat(
        agents=[analyst, developer, tester],
        messages=[],
        max_round=9,  # 确保每个Agent都能发言3次
        speaker_selection_method="round_robin",  # 强制轮询
    )

    print("  正确代码:")
    print("    groupchat = GroupChat(")
    print("        agents=[analyst, developer, tester],")
    print("        max_round=9,  # 确保轮询完整")
    print("        speaker_selection_method='round_robin',")
    print("    )")
    print("  优势：每个Agent都有均等的发言机会")
    print("  适用场景：需要严格交替发言的流程")

    # ---------------------------------------------
    # 正确示例2：使用manual模式手动控制
    # ---------------------------------------------
    print("\n【正确示例2】使用manual模式人工控制发言顺序")

    groupchat_manual = GroupChat(
        agents=[analyst, developer, tester],
        messages=[],
        max_round=10,
        speaker_selection_method="manual",  # 手动控制
    )

    print("  正确代码:")
    print("    groupchat = GroupChat(")
    print("        agents=[analyst, developer, tester],")
    print("        max_round=10,")
    print("        speaker_selection_method='manual',")
    print("    )")
    print("  优势：人类完全控制发言顺序")
    print("  适用场景：需要人工指导的评审流程")

    # ---------------------------------------------
    # 正确示例3：自定义选择函数
    # ---------------------------------------------
    print("\n【正确示例3】自定义选择函数实现智能控制")

    # 记录每个Agent的发言次数
    speaker_history: Dict[str, int] = {}

    def smart_select_speaker(groupchat: GroupChat, last_speaker=None):
        """
        智能选择函数：
        1. 优先选择发言次数最少的Agent
        2. 排除上一个发言者（除非只有两个）
        3. 根据话题相关性动态调整权重
        """
        agents = groupchat.agents

        if len(agents) == 1:
            return agents[0]

        # 排除上一个发言者
        candidates = [a for a in agents if a != last_speaker]

        if len(candidates) == 1:
            return candidates[0]

        # 选择发言次数最少的候选者
        def get_count(agent):
            return speaker_history.get(agent.name, 0)

        return min(candidates, key=get_count)

    groupchat_custom = GroupChat(
        agents=[analyst, developer, tester],
        messages=[],
        max_round=10,
        speaker_selection_method=smart_select_speaker,  # 自定义函数
    )

    print("  正确代码:")
    print("    def smart_select_speaker(groupchat, last_speaker):")
    print("        # 排除上一个发言者")
    print("        candidates = [a for a in groupchat.agents if a != last_speaker]")
    print("        # 选择发言次数最少的")
    print("        return min(candidates, key=lambda a: speaker_history[a.name])")
    print("  优势：兼顾均衡与灵活性")

    print("\n  修复要点:")
    print("    1. 根据业务场景选择合适的speaker_selection_method")
    print("    2. 严格交替发言使用round_robin")
    print("    3. 需要人工指导使用manual")
    print("    4. 需要智能选择使用auto或自定义函数")


# ------------------------------------------------------------
# 误区4：Tool Call与Code Executor混用导致行为不一致
# ------------------------------------------------------------

def demo_pitfall_004_mixed_executor():
    """
    误区4：Tool Call与Code Executor混用导致行为不一致

    问题描述：
    - register_function和register_for_llm_call配置混淆
    - LLM知道工具存在但无法执行
    - 执行结果未反馈给LLM
    - 工具调用行为不一致

    根因分析：
    - 不理解register_function vs register_for_llm_call vs register_for_exec的区别
    - 混用导致LLM和实际执行不匹配
    - 缺少正确的错误处理

    后果评估：
    - 严重程度：高
    - 工具调用失败但LLM不知道
    - 响应不一致，用户体验差
    - 调试困难，难以定位问题
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("误区4：Tool Call与Code Executor混用")
    print("=" * 60)

    llm_config = build_llm_config()

    # 定义一个计算器函数
    def calculator(expression: str) -> str:
        """
        计算数学表达式

        Args:
            expression: 数学表达式，如 "2+3*5"

        Returns:
            计算结果
        """
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # ---------------------------------------------
    # 错误示例1：只注册给LLM，不注册执行
    # ---------------------------------------------
    print("\n【错误示例1】只注册给LLM，不注册执行")

    agent_llm_only = ConversableAgent(
        name="Agent_LLM_Only",
        system_message="你是一个计算器助手，可以帮助用户计算数学表达式。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 错误：只注册给LLM，没有注册执行
    agent_llm_only.register_for_llm_call(name="calculator", description="计算数学表达式")

    print("  错误代码:")
    print("    agent.register_for_llm_call(name='calculator', description='...')")
    print("    # 缺少：agent.register_for_execution(name='calculator')")
    print("  问题：LLM知道有这个工具，但无法调用它")

    # ---------------------------------------------
    # 错误示例2：register_function配置不当
    # ---------------------------------------------
    print("\n【错误示例2】register_function配置不当")

    agent_wrong = ConversableAgent(
        name="Agent_Wrong",
        system_message="你是一个计算器助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # register_function会同时注册给LLM和执行
    # 但如果不正确配置，可能导致问题
    agent_wrong.register_function(
        function=calculator,
        # 没有正确配置name和description
    )

    print("  错误代码:")
    print("    agent.register_function(function=calculator)")
    print("    # 缺少明确的name和description配置")
    print("  问题：工具描述不清晰，LLM可能无法正确调用")

    # ---------------------------------------------
    # 正确示例：明确分离注册
    # ---------------------------------------------
    print("\n【正确示例】正确分离注册")

    agent_correct = ConversableAgent(
        name="Agent_Correct",
        system_message="你是一个计算器助手，可以帮助用户计算数学表达式。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 正确做法：明确分离注册
    agent_correct.register_for_llm_call(
        name="calculate",
        description="计算数学表达式的值，支持加减乘除和括号"
    )
    agent_correct.register_for_execution(
        name="calculate",
        description="计算数学表达式的值"
    )

    print("  正确代码:")
    print("    # 注册给LLM（让它知道有这个工具）")
    print("    agent.register_for_llm_call(")
    print("        name='calculate',")
    print("        description='计算数学表达式，支持加减乘除'")
    print("    )")
    print("    # 注册给执行器（让它知道如何执行）")
    print("    agent.register_for_execution(")
    print("        name='calculate',")
    print("        description='计算数学表达式的值'")
    print("    )")
    print("  优势：职责分离，清晰可控")

    # ---------------------------------------------
    # 正确示例：使用function_map
    # ---------------------------------------------
    print("\n【正确示例】使用function_map配置")

    agent_fmap = ConversableAgent(
        name="Agent_FMap",
        system_message="你是一个计算器助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 使用function_map注册
    agent_fmap.register_function(
        function=calculator,
        name="calc",
        description="计算数学表达式"
    )

    print("  正确代码:")
    print("    agent.register_function(")
    print("        function=calculator,")
    print("        name='calc',")
    print("        description='计算数学表达式'")
    print("    )")
    print("  说明：register_function会同时注册LLM和执行")

    print("\n  修复要点:")
    print("    1. 使用register_for_llm_call + register_for_execution组合")
    print("    2. 确保name一致，否则会出现找不到函数的问题")
    print("    3. description要清晰，帮助LLM理解何时调用")
    print("    4. 或者使用register_function一次性完成注册")


# ------------------------------------------------------------
# 误区5：在async代码中使用同步generate_reply导致死锁
# ------------------------------------------------------------

def demo_pitfall_005_async_sync_mixing():
    """
    误区5：在async代码中使用同步generate_reply导致死锁

    问题描述：
    - 在异步上下文中调用同步方法
    - 阻塞事件循环
    - 多Agent并发场景下系统假死

    根因分析：
    - 不理解同步和异步方法的区别
    - 在async函数中调用了同步的generate_reply
    - 缺少正确的异步封装

    后果评估：
    - 严重程度：高
    - 系统假死，无法响应
    - 多Agent并发时问题更明显
    - 调试困难，死锁不易复现
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("误区5：在async代码中使用同步generate_reply")
    print("=" * 60)

    llm_config = build_llm_config()

    agent = ConversableAgent(
        name="AsyncAgent",
        system_message="你是一个异步助手。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 错误示例：在async函数中调用同步方法
    # ---------------------------------------------
    print("\n【错误示例】在async函数中调用同步方法")

    async def wrong_async_usage():
        """
        错误：在async函数中调用同步方法
        这会阻塞事件循环
        """
        # 错误：在async函数中调用同步的initiate_chat
        # 这会阻塞事件循环，直到对话完成
        result = agent.initiate_chat(
            recipient=agent,
            message="你好",
            max_consecutive_auto_reply=2
        )
        return result

    print("  错误代码:")
    print("    async def wrong_async_usage():")
    print("        # 错误：在async中调用同步方法")
    print("        result = agent.initiate_chat(...)")
    print("        return result")
    print("  问题：会阻塞事件循环，破坏异步性")

    # ---------------------------------------------
    # 正确示例：使用异步方法
    # ---------------------------------------------
    print("\n【正确示例】使用异步方法")

    async def correct_async_usage():
        """
        正确：使用异步方法进行通信
        使用a_initiate_chat代替initiate_chat
        """
        # 正确：使用异步方法
        result = await agent.a_initiate_chat(
            recipient=agent,
            message="你好",
            max_consecutive_auto_reply=2
        )
        return result

    print("  正确代码:")
    print("    async def correct_async_usage():")
    print("        # 正确：使用异步方法")
    print("        result = await agent.a_initiate_chat(...)")
    print("        return result")

    # ---------------------------------------------
    # 正确示例：并发执行多个对话
    # ---------------------------------------------
    print("\n【正确示例】并发执行多个对话")

    async def concurrent_chats():
        """
        正确：使用asyncio.gather并发执行多个对话
        """
        agent_a = ConversableAgent(
            name="Agent_A",
            system_message="你是Agent A。",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        agent_b = ConversableAgent(
            name="Agent_B",
            system_message="你是Agent B。",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        # 并发执行两个对话
        task1 = agent_a.a_initiate_chat(
            recipient=agent_b,
            message="你好",
            max_consecutive_auto_reply=1
        )

        task2 = agent_b.a_initiate_chat(
            recipient=agent_a,
            message="你好",
            max_consecutive_auto_reply=1
        )

        # 使用asyncio.gather并发执行
        results = await asyncio.gather(task1, task2)
        return results

    print("  正确代码:")
    print("    async def concurrent_chats():")
    print("        # 并发执行多个对话")
    print("        task1 = agent_a.a_initiate_chat(...)")
    print("        task2 = agent_b.a_initiate_chat(...)")
    print("        results = await asyncio.gather(task1, task2)")
    print("        return results")
    print("  优势：充分利用异步并发能力")

    print("\n  修复要点:")
    print("    1. 异步环境中统一使用异步方法（a_initiate_chat）")
    print("    2. 不要在async函数中调用同步方法")
    print("    3. 使用asyncio.gather并发执行多个任务")
    print("    4. 注意：generate_reply有对应的a_generate_reply")


# ------------------------------------------------------------
# 误区6：llm_config设置为False但仍期望Agent自动生成回复
# ------------------------------------------------------------

def demo_pitfall_006_llm_config_false():
    """
    误区6：llm_config设置为False但仍期望Agent自动生成回复

    问题描述：
    - 将llm_config=False理解为禁用LLM
    - 导致Agent无法生成回复
    - Agent始终返回空回复或default_auto_reply

    根因分析：
    - 不理解llm_config=False的含义
    - 误以为False表示"使用默认配置"
    - 实际上False表示不使用LLM

    后果评估：
    - 严重程度：中
    - Agent无法生成有意义的回复
    - 对话无法正常进行
    - 调试困难，错误信息不明确
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("误区6：llm_config设置为False但仍期望自动回复")
    print("=" * 60)

    # ---------------------------------------------
    # 错误示例：llm_config=False但期望自动回复
    # ---------------------------------------------
    print("\n【错误示例】llm_config=False但期望自动回复")

    # 错误：设置了llm_config=False，但仍然期望Agent自动回复
    agent_wrong = ConversableAgent(
        name="Agent_Wrong",
        system_message="你是一个有帮助的助手。",
        llm_config=False,  # 这会禁用LLM！
        human_input_mode="NEVER",
        # 错误：设置了max_consecutive_auto_reply但LLM已被禁用
        max_consecutive_auto_reply=5,
    )

    print("  错误代码:")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        system_message='你是一个有帮助的助手。',")
    print("        llm_config=False,  # 禁用LLM！")
    print("        max_consecutive_auto_reply=5,")
    print("    )")
    print("  问题：llm_config=False会禁用LLM，无法自动回复")
    print("  效果：Agent会使用default_auto_reply或返回空消息")

    # ---------------------------------------------
    # 正确示例1：使用正确的LLM配置
    # ---------------------------------------------
    print("\n【正确示例1】使用正确的LLM配置")

    llm_config = build_llm_config()

    agent_correct = ConversableAgent(
        name="Agent_Correct",
        system_message="你是一个有帮助的助手。",
        llm_config=llm_config,  # 正确的配置
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
    )

    print("  正确代码:")
    print("    llm_config = {")
    print("        'config_list': [{'model': 'gpt-4o-mini', 'api_key': '...'}]")
    print("    }")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        llm_config=llm_config,  # 正确的配置")
    print("    )")

    # ---------------------------------------------
    # 正确示例2：llm_config=False的正确使用场景
    # ---------------------------------------------
    print("\n【正确示例2】llm_config=False的正确使用场景")

    # llm_config=False的正确场景：纯人工输入或纯代码执行
    agent_human = ConversableAgent(
        name="Agent_Human",
        system_message="你是人工代理，只转发人类输入。",
        llm_config=False,  # 禁用LLM，节省资源
        human_input_mode="ALWAYS",  # 等待人类输入
    )

    agent_code = ConversableAgent(
        name="Agent_Code",
        system_message="你是代码执行代理。",
        llm_config=False,  # 禁用LLM，使用代码执行器
        human_input_mode="NEVER",
    )

    print("  正确场景1：纯人工输入代理")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        llm_config=False,  # 不需要LLM")
    print("        human_input_mode='ALWAYS',  # 等待人类输入")
    print("    )")

    print("  正确场景2：纯代码执行代理")
    print("    agent = ConversableAgent(")
    print("        name='agent',")
    print("        llm_config=False,  # 不需要LLM")
    print("        code_execution_config={...},  # 使用代码执行器")
    print("    )")

    print("\n  修复要点:")
    print("    1. llm_config=False表示完全禁用LLM")
    print("    2. 只有在纯人工输入或纯代码执行场景才使用False")
    print("    3. 需要自动回复时必须提供有效的llm_config")
    print("    4. 注意：如果llm_config=False但human_input_mode='NEVER'，Agent将无法响应")


# ------------------------------------------------------------
# 误区7：嵌套GroupChat中的状态污染
# ------------------------------------------------------------

def demo_pitfall_007_nested_groupchat_pollution():
    """
    误区7：嵌套GroupChat中的状态污染

    问题描述：
    - 未理解nested group chat的消息隔离机制
    - 子群聊消息混入父群聊
    - GroupChatManager状态混乱、消息溯源失败

    根因分析：
    - 嵌套GroupChat共享状态管理不当
    - 消息传递参数配置错误
    - 未正确隔离不同层级的消息

    后果评估：
    - 严重程度：高
    - 消息溯源失败
    - 状态混乱导致不可预测行为
    - 对话内容泄露到错误的层级
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    print("\n" + "=" * 60)
    print("误区7：嵌套GroupChat中的状态污染")
    print("=" * 60)

    llm_config = build_llm_config()

    # 定义子群聊的Agent
    subtask_agent_a = ConversableAgent(
        name="SubTask_A",
        system_message="你是子任务Agent A。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    subtask_agent_b = ConversableAgent(
        name="SubTask_B",
        system_message="你是子任务Agent B。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # 定义主群聊的Agent
    main_agent = ConversableAgent(
        name="MainAgent",
        system_message="你是主Agent。",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ---------------------------------------------
    # 错误示例：嵌套GroupChat未正确隔离消息
    # ---------------------------------------------
    print("\n【错误示例】嵌套GroupChat未正确隔离消息")

    # 创建子GroupChat
    sub_groupchat = GroupChat(
        agents=[subtask_agent_a, subtask_agent_b],
        messages=[],
        max_round=5,
    )

    sub_manager = GroupChatManager(
        groupchat=sub_groupchat,
        llm_config=llm_config,
    )

    print("  错误代码:")
    print("    # 创建子GroupChat（未隔离）")
    print("    sub_groupchat = GroupChat(")
    print("        agents=[subtask_a, subtask_b],")
    print("        messages=[],  # 共享空列表？")
    print("        max_round=5,")
    print("    )")
    print("  问题：子群聊的messages可能与父群聊共享")
    print("  后果：子群聊消息混入父群聊，状态污染")

    # ---------------------------------------------
    # 正确示例：正确配置嵌套GroupChat
    # ---------------------------------------------
    print("\n【正确示例】正确配置嵌套GroupChat")

    # 子GroupChat：使用独立的messages列表
    sub_groupchat_isolated = GroupChat(
        agents=[subtask_agent_a, subtask_agent_b],
        messages=[],  # 独立的空列表
        max_round=5,
        # 关键：为子GroupChat设置独立的名称
        name="sub_groupchat",
    )

    sub_manager_isolated = GroupChatManager(
        groupchat=sub_groupchat_isolated,
        llm_config=llm_config,
    )

    print("  正确代码:")
    print("    # 子GroupChat：使用独立配置")
    print("    sub_groupchat = GroupChat(")
    print("        agents=[subtask_a, subtask_b],")
    print("        messages=[],  # 独立的空列表")
    print("        max_round=5,")
    print("        name='sub_groupchat',  # 独立名称")
    print("    )")

    print("  关键配置:")
    print("    1. 为子GroupChat提供独立的messages列表")
    print("    2. 设置唯一的name标识")
    print("    3. 使用嵌套chat机制传递消息")
    print("    4. 在父Agent中正确处理子群聊的结果")

    # ---------------------------------------------
    # 正确的嵌套调用方式
    # ---------------------------------------------
    print("\n【正确的嵌套调用方式】")

    print("""
    # 正确的嵌套调用
    main_agent = ConversableAgent(...)

    # 通过initiate_chat启动子群聊
    result = main_agent.initiate_chat(
        sub_manager_isolated,
        message="请完成子任务",
        # 关键：传递父群聊的消息历史
        chat_messages=main_agent.chat_messages.get(sub_manager_isolated, {}),
    )

    # 从结果中提取子群聊的输出
    sub_output = result.summary or result.last_message()
    """)

    print("  修复要点:")
    print("    1. 每个GroupChat使用独立的messages列表")
    print("    2. 使用chat_messages参数传递消息历史")
    print("    3. 正确区分不同层级的消息")
    print("    4. 使用summary或last_message获取子群聊结果")


# ------------------------------------------------------------
# 误区8：未配置price字段导致成本计算不准确
# ------------------------------------------------------------

def demo_pitfall_008_missing_price():
    """
    误区8：未配置price字段导致成本计算不准确

    问题描述：
    - 使用非OpenAI模型时未配置price字段
    - AutoGen无法计算成本
    - 成本监控失效、预算超支风险

    根因分析：
    - 不理解price字段的作用
    - 误以为price只是用于显示
    - 未考虑成本控制的必要性

    后果评估：
    - 严重程度：中
    - 成本监控失效
    - 预算超支风险
    - 无法进行成本优化
    """
    from autogen import ConversableAgent

    print("\n" + "=" * 60)
    print("误区8：未配置price字段导致成本计算不准确")
    print("=" * 60)

    # ---------------------------------------------
    # 错误示例：缺少price字段
    # ---------------------------------------------
    print("\n【错误示例】缺少price字段")

    config_without_price = {
        "config_list": [{
            "model": "qwen2.5:3b",
            "api_key": "ollama-key",  # 非OpenAI模型
            "base_url": "http://localhost:11434",
        }]
        # 缺少price字段！
    }

    print("  错误代码:")
    print("    config = {")
    print("        'config_list': [{")
    print("            'model': 'qwen2.5:3b',")
    print("            'api_key': '...',")
    print("            'base_url': 'http://localhost:11434',")
    print("            # 缺少 price 字段！")
    print("        }]")
    print("    }")
    print("  问题：AutoGen无法计算成本，成本监控失效")

    # ---------------------------------------------
    # 正确示例：配置price字段
    # ---------------------------------------------
    print("\n【正确示例】配置price字段")

    config_with_price = {
        "config_list": [{
            "model": "qwen2.5:3b",
            "api_key": "ollama-key",
            "base_url": "http://localhost:11434",
            # price字段：[input_price_per_1k, output_price_per_1k]
            "price": [0.0001, 0.0002],  # 成本配置
        }]
    }

    print("  正确代码:")
    print("    config = {")
    print("        'config_list': [{")
    print("            'model': 'qwen2.5:3b',")
    print("            'api_key': '...',")
    print("            'base_url': 'http://localhost:11434',")
    print("            'price': [0.0001, 0.0002],  # 输入和输出价格")
    print("        }]")
    print("    }")

    # ---------------------------------------------
    # 价格配置说明
    # ---------------------------------------------
    print("\n【价格配置说明】")

    print("  price字段格式: [input_price_per_1k, output_price_per_1k]")
    print("    - 第一个值：每1000个输入token的价格（美元）")
    print("    - 第二个值：每1000个输出token的价格（美元）")

    print("\n  常见模型的价格参考:")
    print("    - gpt-4o: [0.005, 0.015]")
    print("    - gpt-4o-mini: [0.00015, 0.0006]")
    print("    - qwen2.5:3b (本地): [0, 0]  # 免费")
    print("    - claude-3-opus: [0.015, 0.075]")

    print("\n  修复要点:")
    print("    1. 为每个模型配置price字段")
    print("    2. 本地模型可以设置为[0, 0]")
    print("    3. 设置price后可以:")
    print("       - 监控对话成本")
    print("       - 设置预算上限")
    print("       - 优化模型选择")
    print("    4. 使用get_model_cost估算对话成本")

    # ---------------------------------------------
    # 使用成本监控
    # ---------------------------------------------
    print("\n【使用成本监控】")

    print("""
    from autogen import ChatCompletion

    # 获取对话成本
    cost = ChatCompletion.get_cost(
        model="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
        price_config=config_with_price,
    )
    print(f"对话成本: ${cost}")

    # 设置预算上限
    def check_budget(msg):
        total_cost = calculate_total_cost()
        if total_cost > budget_limit:
            raise BudgetExceededError("预算超支")
        return False
    """)


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有误区演示
    """
    print("=" * 60)
    print("AutoGen高频误区代码演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个误区演示
    demo_pitfall_001_weak_termination()
    demo_pitfall_002_max_consecutive_none()
    demo_pitfall_003_speaker_selection_mode()
    demo_pitfall_004_mixed_executor()
    demo_pitfall_005_async_sync_mixing()
    demo_pitfall_006_llm_config_false()
    demo_pitfall_007_nested_groupchat_pollution()
    demo_pitfall_008_missing_price()

    print("\n" + "=" * 60)
    print("误区演示完成")
    print("=" * 60)
    print("\n总结：8个高频误区")
    print("  case_001: is_termination_msg条件设置过宽/过严")
    print("  case_002: max_consecutive_auto_reply设置为None")
    print("  case_003: GroupChat忘记设置speaker_selection_mode")
    print("  case_004: Tool Call与Code Executor混用")
    print("  case_005: async代码中使用同步方法")
    print("  case_006: llm_config=False但期望自动回复")
    print("  case_007: 嵌套GroupChat状态污染")
    print("  case_008: 未配置price字段")


if __name__ == "__main__":
    main()