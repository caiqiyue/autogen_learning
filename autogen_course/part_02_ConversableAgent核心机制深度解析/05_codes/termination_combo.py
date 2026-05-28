# termination_combo.py
# 第5节 max_consecutive_auto_reply与轮次控制 - 终止条件组合演示
#
# 本文件演示 is_termination_msg + max_consecutive_auto_reply 组合使用
# 实现精确的对话轮次控制和终止条件管理
#
# 要运行此代码，你需要：
# 1. 安装 AutoGen: pip install pyautogen
# 2. 配置环境变量（在 .env 文件中或系统环境变量中）
#    OPENAI_MODEL=你的模型名称
#    OPENAI_API_KEY=你的API密钥
#    OPENAI_BASE_URL=API基础URL（可选）
#
# ============================================================
# is_termination_msg 编写模式
# ============================================================
#
# is_termination_msg 是一个可选的回调函数，用于判断消息是否应该终止对话。
# 它的签名是: def is_termination_msg(msg) -> bool
#
# 编写规范：
# 1. 接收一个消息字典作为参数
# 2. 从消息中提取关键信息（通常检查 content 字段）
# 3. 返回 True 表示终止对话，False 表示继续
#
# 常见编写模式：
# - 字典键值提取：检查 msg.get("content", "")
# - 多条件组合：使用 and/or 连接多个条件
# - 正则匹配：re.search(pattern, content)
# - 关键词检测：any(keyword in content for keyword in keywords)
# ============================================================
#
# ============================================================
# 三种 human_input_mode 与轮次控制的关系
# ============================================================
#
# 1. ALWAYS - 每次回复都请求人类输入
#    - 无论 max_consecutive_auto_reply 如何设置，每次回复前都会请求人类输入
#    - 对话总是受人类控制，但可能频繁打断
#
# 2. NEVER - 从不请求人类输入，完全依赖自动终止条件
#    - 需要依靠 max_consecutive_auto_reply 和/或 is_termination_msg 来终止
#    - 适用于完全自动化的对话场景
#
# 3. TERMINATE - 当满足终止条件时请求人类确认
#    - 当 is_termination_msg 返回 True 或达到 max_consecutive_auto_reply 上限时
#    - 请求人类输入来确认是否真正终止
#    - 适用于需要人工审批关键决策的场景
# ============================================================

import os
import re
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
# 第三部分：is_termination_msg 编写模式演示
# ============================================================

def demo_termination_msg_patterns():
    """
    演示 is_termination_msg 的几种编写模式

    is_termination_msg 用于判断是否应该终止对话。
    以下展示不同复杂度的编写方式。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示1: is_termination_msg 编写模式")
    print("=" * 60)

    llm_config = build_llm_config()

    # ============================================================
    # 模式1: 简单关键词检测
    # ============================================================
    def termination_by_keyword(msg):
        """
        模式1: 基于关键词的终止判断

        当消息包含特定关键词时终止对话。
        适用于已知对话结束标记的场景。
        """
        # 从消息字典中提取内容
        content = msg.get("content", "")
        # 定义终止关键词列表
        termination_keywords = ["再见", "结束", "终止", "告辞", "goodbye", "exit", "terminate"]
        # 检查是否包含任意终止关键词
        return any(keyword in content.lower() for keyword in termination_keywords)

    # ============================================================
    # 模式2: 正则表达式匹配
    # ============================================================
    def termination_by_regex(msg):
        """
        模式2: 基于正则表达式的终止判断

        使用正则表达式匹配特定模式。
        适用于需要更灵活匹配规则（如特定格式输出）的场景。
        """
        content = msg.get("content", "")
        # 匹配常见的终止模式
        # 例如：包含"任务完成"、"结果如下"后跟句号
        patterns = [
            r"任务完成[。.]",
            r"结果如下[：:]",  # 注意：这里用中文冒号也可以匹配中文冒号：
            r"^完成$",
            r"^再见[。!]*$",
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    # ============================================================
    # 模式3: 多条件组合
    # ============================================================
    def termination_by_multiple_conditions(msg):
        """
        模式3: 多条件组合的终止判断

        结合多个条件进行判断，提供更精确的终止控制。
        适用于复杂业务规则场景。
        """
        content = msg.get("content", "")
        role = msg.get("role", "")

        # 条件1: 消息来自助手
        condition1 = role == "assistant"

        # 条件2: 包含完成标记
        condition2 = any(keyword in content for keyword in ["完成", "结束", "结果"])

        # 条件3: 内容长度适中（表示不是简单敷衍）
        condition3 = 20 < len(content) < 1000

        # 条件4: 不包含追问类词汇
        condition4 = not any(keyword in content for keyword in ["请告诉我", "你能", "可以", "吗", "？"])

        # 所有条件都满足时才终止
        return condition1 and condition2 and condition3 and condition4

    # ============================================================
    # 模式4: 字典结构深度提取
    # ============================================================
    def termination_by_dict_extraction(msg):
        """
        模式4: 从嵌套字典结构中提取信息

        AutoGen的消息可能包含更复杂的结构。
        这个模式展示如何处理嵌套字典。
        """
        # 处理可能嵌套的消息结构
        if isinstance(msg, dict):
            # 尝试多种可能的键名
            content = msg.get("content", "")
            # 有些消息格式使用 "text" 字段
            if not content:
                content = msg.get("text", "")
            # 有些消息格式使用嵌套结构
            if not content and "message" in msg:
                content = msg.get("message", {}).get("content", "")
        else:
            content = str(msg)

        # 基于内容判断是否终止
        return "终止" in content or "结束对话" in content

    # 测试每种模式
    test_messages = [
        {"role": "assistant", "content": "好的，我已经完成了任务。结果如下：分析完成。"},
        {"role": "assistant", "content": "再见，感谢您的咨询。"},
        {"role": "assistant", "content": "这个问题的答案 是42。是的，答案是42。"},
        {"role": "user", "content": "请详细解释一下什么是AutoGen"},
    ]

    patterns = [
        ("关键词检测", termination_by_keyword),
        ("正则匹配", termination_by_regex),
        ("多条件组合", termination_by_multiple_conditions),
        ("字典深度提取", termination_by_dict_extraction),
    ]

    for name, termination_func in patterns:
        print(f"\n--- {name} ---")
        for i, msg in enumerate(test_messages, 1):
            result = termination_func(msg)
            content_preview = msg.get("content", "")[:30]
            print(f"  消息{i}: {content_preview}... -> 终止={result}")


def demo_termination_combo_with_max_consecutive():
    """
    演示 is_termination_msg + max_consecutive_auto_reply 组合

    两种终止条件可以组合使用，形成"双重保险"机制：
    1. is_termination_msg 控制内容相关的终止（如输出完成）
    2. max_consecutive_auto_reply 控制轮次上限（如防止无限循环）

    这种组合特别适用于需要严格控制对话长度的生产环境。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示2: is_termination_msg + max_consecutive_auto_reply 组合")
    print("=" * 60)

    llm_config = build_llm_config()

    # 定义终止消息判断函数
    def my_termination_msg(msg):
        """
        自定义终止条件：消息包含"完成"或"再见"时终止
        """
        content = msg.get("content", "")
        return "完成" in content or "再见" in content

    # 创建助手代理：同时设置 is_termination_msg 和 max_consecutive_auto_reply
    assistant = ConversableAgent(
        name="助手",
        llm_config=llm_config,
        # 组合策略：
        # 1. is_termination_msg: 当输出包含"完成"时终止
        # 2. max_consecutive_auto_reply=5: 最多自动回复5轮
        is_termination_msg=my_termination_msg,
        max_consecutive_auto_reply=5,
    )

    print(f"助手配置:")
    print(f"  - 名称: {assistant.name}")
    print(f"  - max_consecutive_auto_reply: {assistant.max_consecutive_auto_reply}")
    print(f"  - is_termination_msg: 已设置")

    # 模拟对话场景
    messages = [
        {"role": "user", "content": "帮我分析一下Python和JavaScript的区别"}
    ]

    print(f"\n初始用户消息: {messages[0]['content']}")
    print("\n模拟对话（双重终止条件）:")

    for i in range(1, 7):
        reply = assistant.generate_reply(messages=messages)

        if reply:
            print(f"\n轮次 {i}:")
            print(f"  回复: {reply[:60]}..." if len(str(reply)) > 60 else f"  回复: {reply}")
            messages.append({"role": "assistant", "content": str(reply)})

            # 检查终止条件
            if my_termination_msg({"content": str(reply)}):
                print(f"  -> is_termination_msg 返回 True，对话终止")
                break
        else:
            print(f"\n轮次 {i}: generate_reply 返回 None")
            print(f"  -> 可能的原因为:")
            print(f"     1. max_consecutive_auto_reply 达到上限 (5)")
            print(f"     2. 其他内部终止条件触发")
            break

    print("\n说明: 双重保险确保对话不会无限进行")


def demo_human_input_mode_combination():
    """
    演示 human_input_mode 与轮次控制参数的组合

    human_input_mode 决定何时请求人类输入，与轮次控制参数配合使用。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示3: human_input_mode 与轮次控制组合")
    print("=" * 60)

    llm_config = build_llm_config()

    def termination_msg(msg):
        """简单终止条件：消息包含"完成"时终止"""
        content = msg.get("content", "")
        return "完成" in content

    # 配置场景1: ALWAYS 模式
    print("\n--- 场景1: human_input_mode='ALWAYS' ---")
    print("特点: 每次回复都请求人类输入，完全由人类控制对话节奏")
    print("max_consecutive_auto_reply 参数在 ALWAYS 模式下作用较小")

    agent_always = ConversableAgent(
        name="助手_ALWAYS",
        llm_config=llm_config,
        human_input_mode="ALWAYS",  # 总是请求人类输入
        max_consecutive_auto_reply=3,
        is_termination_msg=termination_msg,
    )
    print(f"配置: human_input_mode={agent_always.human_input_mode}, max_consecutive_auto_reply={agent_always.max_consecutive_auto_reply}")

    # 配置场景2: NEVER 模式
    print("\n--- 场景2: human_input_mode='NEVER' ---")
    print("特点: 从不请求人类输入，完全依赖自动终止条件")
    print("需要依靠 max_consecutive_auto_reply 和/或 is_termination_msg 来终止")

    agent_never = ConversableAgent(
        name="助手_NEVER",
        llm_config=llm_config,
        human_input_mode="NEVER",  # 从不请求人类输入
        max_consecutive_auto_reply=3,
        is_termination_msg=termination_msg,
    )
    print(f"配置: human_input_mode={agent_never.human_input_mode}, max_consecutive_auto_reply={agent_never.max_consecutive_auto_reply}")

    # 配置场景3: TERMINATE 模式
    print("\n--- 场景3: human_input_mode='TERMINATE' ---")
    print("特点: 当满足终止条件时请求人类确认")
    print("适用于关键节点需要人工审批的工作流")

    agent_terminate = ConversableAgent(
        name="助手_TERMINATE",
        llm_config=llm_config,
        human_input_mode="TERMINATE",  # 终止条件满足时请求确认
        max_consecutive_auto_reply=3,
        is_termination_msg=termination_msg,
    )
    print(f"配置: human_input_mode={agent_terminate.human_input_mode}, max_consecutive_auto_reply={agent_terminate.max_consecutive_auto_reply}")

    print("\n三种模式对比:")
    print("| 模式      | 人类控制 | 自动终止 | 适用场景                    |")
    print("|-----------|----------|----------|-----------------------------|")
    print("| ALWAYS    | 高       | 低       | 需要全程人工监督的敏感任务    |")
    print("| NEVER     | 无       | 高       | 完全自动化的批量处理场景      |")
    print("| TERMINATE | 中       | 中       | 需要在关键节点人工审批的工作流|")
    print("|-----------|----------|----------|-----------------------------|")


def demo_production_workflow():
    """
    演示生产环境工作流：复杂终止条件组合

    展示一个更复杂的实际应用场景：
    - 对话在3种情况下终止：
      1. 达到最大轮次 (max_consecutive_auto_reply)
      2. 用户明确表示结束 (is_termination_msg)
      3. 助手输出包含完成标记 (is_termination_msg)
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示4: 生产环境工作流 - 复杂终止条件组合")
    print("=" * 60)

    llm_config = build_llm_config()

    # ============================================================
    # 生产级终止条件函数
    # ============================================================
    def production_termination_msg(msg):
        """
        生产级终止条件：综合考虑多个因素

        终止条件：
        1. 用户明确说再见/退出
        2. 助手输出了任务完成标记
        3. 助手输出包含"已解决"等完成信号
        """
        content = msg.get("content", "").lower()
        role = msg.get("role", "")

        # 条件1: 用户明确要求退出
        user_exit_keywords = ["再见", "退出", "exit", "quit", "取消", "算了"]
        if role == "user" and any(kw in content for kw in user_exit_keywords):
            return True

        # 条件2: 助手输出包含完成标记
        completion_keywords = ["完成", "解决", "结束了", "搞定", "搞定了"]
        if role == "assistant" and any(kw in content for kw in completion_keywords):
            return True

        # 条件3: 助手输出包含"谢谢"且较短（表示正常结束语）
        if role == "assistant" and "谢谢" in content and len(content) < 100:
            return True

        return False

    # 创建生产级代理
    production_agent = ConversableAgent(
        name="生产助手",
        llm_config=llm_config,
        max_consecutive_auto_reply=10,  # 最多自动回复10轮
        is_termination_msg=production_termination_msg,
        human_input_mode="TERMINATE",  # 终止时请求确认
    )

    print("生产助手配置:")
    print(f"  - max_consecutive_auto_reply: {production_agent.max_consecutive_auto_reply}")
    print(f"  - human_input_mode: {production_agent.human_input_mode}")
    print(f"  - is_termination_msg: 已配置（多条件组合）")

    print("\n终止条件规则:")
    print("  1. 用户说'再见'、'退出'等退出关键词时终止")
    print("  2. 助手输出包含'完成'、'解决'等完成标记时终止")
    print("  3. 助手输出包含'谢谢'且内容较短时终止")
    print("  4. 达到10轮自动回复上限时终止")
    print("  5. 终止时请求人类确认（human_input_mode='TERMINATE'）")

    # 模拟测试场景
    test_scenarios = [
        # 场景1: 用户主动退出
        {
            "messages": [
                {"role": "user", "content": "帮我解释一下什么是API"},
                {"role": "assistant", "content": "API是应用程序编程接口..."},
                {"role": "user", "content": "好的，我明白了，再见"},
            ],
            "name": "用户主动退出"
        },
        # 场景2: 助手输出完成标记
        {
            "messages": [
                {"role": "user", "content": "写一个计算器程序"},
                {"role": "assistant", "content": "好的，这是一个简单的计算器程序。代码已完成。"},
            ],
            "name": "助手输出完成标记"
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- 测试场景 {i}: {scenario['name']} ---")
        for msg in scenario["messages"]:
            should_terminate = production_termination_msg(msg)
            content_preview = msg.get("content", "")[:30]
            print(f"  消息({msg['role']}): {content_preview}... -> 终止={should_terminate}")
            if should_terminate:
                print(f"  -> 对话将在此消息后终止")
                break


def demo_anti_patterns():
    """
    演示 is_termination_msg 编写的常见反模式

    帮助开发者避免常见的编写错误。
    """
    from autogen.agentchat.conversable_agent import ConversableAgent

    print("\n" + "=" * 60)
    print("演示5: is_termination_msg 常见反模式（避坑指南）")
    print("=" * 60)

    # ============================================================
    # 反模式1: 条件过宽 - 几乎所有消息都终止
    # ============================================================
    def termination_too_broad(msg):
        """
        反模式1: 条件过宽

        问题: 几乎所有包含句子的消息都会被判定为终止
        后果: 对话过早终止，用户无法获得完整回答
        """
        content = msg.get("content", "")
        # 错误：任何包含句号的消息都被判定为终止
        return "." in content or "。" in content

    # ============================================================
    # 反模式2: 条件过窄 - 几乎没有消息会终止
    # ============================================================
    def termination_too_narrow(msg):
        """
        反模式2: 条件过窄

        问题: 只有极少数特定短语能触发终止
        后果: 对话无法正常终止，可能无限进行
        """
        content = msg.get("content", "")
        # 错误：只有精确匹配"任务完成"才会终止
        return content == "任务完成"

    # ============================================================
    # 反模式3: 未处理消息结构的多样性
    # ============================================================
    def termination_no_dict_check(msg):
        """
        反模式3: 未考虑消息格式多样性

        问题: 假设消息总是字典且总有content字段
        后果: 当收到不同格式的消息时会出错或误判
        """
        # 错误：直接访问content而不检查msg是否为字典
        return "." in msg["content"]

    # 测试反模式
    test_messages = [
        {"role": "assistant", "content": "好的，我来帮你分析这个问题。"},
        {"role": "assistant", "content": "让我解释一下Python的基本语法。"},
        {"role": "assistant", "content": "任务完成"},
        {"role": "user", "content": "继续说"},
    ]

    patterns = [
        ("反模式1(过宽)", termination_too_broad),
        ("反模式2(过窄)", termination_too_narrow),
        ("反模式3(格式问题)", termination_no_dict_check),
    ]

    for name, func in patterns:
        print(f"\n--- {name} ---")
        for i, msg in enumerate(test_messages, 1):
            try:
                result = func(msg)
                content_preview = msg.get("content", "")[:25]
                print(f"  消息{i}: {content_preview}... -> 终止={result}")
            except Exception as e:
                print(f"  消息{i}: 错误 - {type(e).__name__}: {e}")

    print("\n正确编写建议:")
    print("  1. 使用多条件组合，避免单一条件过宽或过窄")
    print("  2. 使用 msg.get('content', '') 安全地提取内容")
    print("  3. 考虑消息的 role 属性，区分用户和助手消息")
    print("  4. 添加日志输出，便于调试和监控")
    print("  5. 编写单元测试验证终止条件的正确性")


# ============================================================
# 第四部分：主函数
# ============================================================

def main():
    """
    主函数：运行所有终止条件组合演示
    """
    print("=" * 60)
    print("is_termination_msg + max_consecutive_auto_reply 组合演示")
    print("=" * 60)

    # 加载环境变量
    print("\n[步骤1] 加载环境变量...")
    load_env()
    print("环境变量加载完成")

    # 运行各个演示
    demo_termination_msg_patterns()
    demo_termination_combo_with_max_consecutive()
    demo_human_input_mode_combination()
    demo_production_workflow()
    demo_anti_patterns()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()