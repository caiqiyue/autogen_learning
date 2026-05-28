# generate_reply_flow.py
# 第4节 generate_reply策略链与register_reply深度掌握
# 演示 generate_reply 的完整执行流程与策略链机制
#
# 本文件展示 generate_reply 方法的核心执行流程：
# 1. 策略链遍历顺序（reply_func_list 的四种函数类型）
# 2. final 标志中断机制
# 3. 消息处理与回复生成逻辑
#
# 要运行此代码，你需要：
# 1. 安装 AutoGen: pip install pyautogen
# 2. 配置环境变量 OPENAI_API_KEY

# ============================================================
# AutoGen 导入说明
# ============================================================
# 以下导入语句仅在配置好真实环境后使用：
# ```python
# try:
#     from autogen import ConversableAgent
#     # 如需使用真实API，取消下面的注释并确保.env配置正确
#     # load_env()  # 加载环境变量
# except ImportError as e:
#     print(f"AutoGen 未安装或导入失败: {e}")
#     print("演示模式：代码逻辑仍可正常展示")
# ```
#
# 本文件中的代码为模拟演示逻辑，展示了 AutoGen 的核心机制
# 实际运行时需要配置真实的 API 密钥和环境
# ============================================================

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# ============================================================
# 第一部分：环境配置与辅助函数
# ============================================================

def load_env(env_path: str = ".env") -> None:
    """
    从 .env 文件加载环境变量

    Args:
        env_path: .env 文件路径
    """
    path = Path(env_path)
    if not path.exists():
        print(f"警告：未找到 {env_path} 文件")
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_env_or_default(key: str, default: str = "") -> str:
    """获取环境变量，不存在则返回默认值"""
    return os.getenv(key, default)


# ============================================================
# 第二部分：reply_func_list 的四种函数类型定义
# ============================================================

class ReplyFuncType(Enum):
    """
    _reply_func_list 中存储的四种回复函数类型

    AutoGen 的回复策略链支持四种类型的回复函数，每种都有不同的触发机制和返回值：
    """
    # 类型1：消息前缀匹配触发 (prefix-match)
    # - trigger: str 或 tuple[str, ...]，匹配消息 content 的前缀
    # - 触发条件：message["content"] 以 trigger 指定的前缀开头
    # - 典型用途：处理特殊命令、VIP用户、格式化输出等
    PREFIX = "prefix"

    # 类型2：正则表达式匹配触发 (regex-match)
    # - trigger: str，正则表达式模式
    # - 触发条件：message["content"] 与正则表达式匹配
    # - 典型用途：模式识别、特定格式消息处理
    REGEX = "regex"

    # 类型3：函数条件判断触发 (func-match)
    # - trigger: Callable 函数，接收消息和发送者，返回 bool
    # - 触发条件：trigger(message, sender) 返回 True
    # - 典型用途：复杂业务逻辑判断
    FUNC = "func"

    # 类型4：可执行内容触发 (executable)
    # - trigger: 不使用（为 None）
    # - 触发条件：消息包含可执行内容（如代码块）
    # - 典型用途：代码执行结果处理
    EXECUTABLE = "executable"


@dataclass
class ReplyFuncRecord:
    """
    回复函数记录结构

    存储在 _reply_func_list 中的每条记录包含：
    - reply_func: 回调函数
    - trigger: 触发条件（类型因 ReplyFuncType 而异）
    - config: 可选配置字典
    - priority: 优先级（数值越小越先执行）
    """
    reply_func: Callable
    trigger: Union[str, tuple, Callable, None]
    config: Optional[Dict[str, Any]]
    priority: int  # 优先级，数值越小优先级越高，越先执行


# ============================================================
# 第三部分：generate_reply 模拟实现（核心流程演示）
# ============================================================

def simulate_generate_reply(
    messages: List[Dict[str, Any]],
    reply_func_list: List[ReplyFuncRecord],
    max_consecutive_auto_reply: int = 10,
    send_history_indexes: Optional[List[int]] = None
) -> str:
    """
    模拟 generate_reply 的核心执行流程

    generate_reply 是 ConversableAgent 接收消息后的核心回复生成方法。
    它遍历策略链（reply_func_list），按照优先级顺序检查每个回复函数，
    一旦某个函数返回非空回复立即终止（除非设置了 final=False）。

    策略链遍历顺序：
    1. 按 priority 从小到大排序（priority=0 的先执行）
    2. 对于相同优先级的函数，按注册顺序执行
    3. 遇到第一个返回非空值的函数立即停止（final 标志中断机制）

    参数:
        messages: 消息历史列表，每条消息是包含 role 和 content 的字典
        reply_func_list: 回复函数列表（按优先级排序）
        max_consecutive_auto_reply: 最大连续自动回复次数
        send_history_indexes: 发送历史索引列表

    返回:
        str: 生成的回复内容，如果没有任何函数返回内容则返回空字符串
    """
    print("\n" + "=" * 60)
    print("generate_reply 执行流程开始")
    print("=" * 60)

    # 步骤1：输入验证
    if not messages:
        print("[步骤1] 消息列表为空，直接返回空回复")
        return ""

    # 获取最新消息
    current_message = messages[-1]
    sender = current_message.get("role", "unknown")
    content = current_message.get("content", "")

    print(f"[步骤1] 当前消息 - 发送者: {sender}")
    print(f"[步骤1] 消息内容: {content[:80]}{'...' if len(content) > 80 else ''}")

    # 步骤2：检查是否超过最大连续回复次数
    consecutive_count = 0  # 这里简化处理，实际代码会计数连续自动回复
    if consecutive_count >= max_consecutive_auto_reply:
        print(f"[步骤2] 已达到最大连续回复次数 ({max_consecutive_auto_reply})，停止生成回复")
        return ""

    # 步骤3：遍历策略链
    print(f"\n[步骤3] 开始遍历策略链，共 {len(reply_func_list)} 个回复函数")
    print("-" * 60)

    # 排序：按 priority 从小到大
    sorted_funcs = sorted(reply_func_list, key=lambda x: x.priority)

    for idx, func_record in enumerate(sorted_funcs):
        print(f"\n--- 检查函数 {idx + 1}: {func_record.reply_func.__name__} ---")
        print(f"    优先级: {func_record.priority}")
        print(f"    触发类型: {type(func_record.trigger).__name__}")
        print(f"    触发条件: {func_record.trigger}")

        # 判断是否应该触发
        triggered = False
        trigger_reason = ""

        if isinstance(func_record.trigger, str):
            # 类型1：前缀匹配
            if content.startswith(func_record.trigger):
                triggered = True
                trigger_reason = f"消息以 '{func_record.trigger}' 开头"
        elif isinstance(func_record.trigger, tuple) and len(func_record.trigger) > 0:
            # 前缀匹配（tuple 形式，支持多个前缀）
            if any(content.startswith(p) for p in func_record.trigger):
                triggered = True
                trigger_reason = f"消息匹配 tuple 中的某个前缀"
        elif isinstance(func_record.trigger, type(re.compile(r""))):
            # 类型2：正则表达式匹配
            pattern = func_record.trigger
            if pattern.search(content):
                triggered = True
                trigger_reason = f"消息匹配正则表达式 '{pattern.pattern}'"
        elif callable(func_record.trigger):
            # 类型3：函数条件判断
            try:
                if func_record.trigger(current_message, sender):
                    triggered = True
                    trigger_reason = "触发函数返回 True"
            except Exception as e:
                print(f"    警告：触发函数执行出错: {e}")
        elif func_record.trigger is None:
            # 类型4：可执行内容（无条件触发，用于代码执行结果）
            triggered = True
            trigger_reason = "无条件触发（可执行内容）"

        if not triggered:
            print(f"    未触发，继续检查下一个函数")
            continue

        print(f"    [触发] {trigger_reason}")

        # 步骤4：执行回复函数
        print(f"    执行回复函数...")
        try:
            reply = func_record.reply_func(
                messages=messages,
                sender=sender,
                config=func_record.config
            )
        except Exception as e:
            print(f"    错误：回复函数执行失败: {e}")
            reply = None

        # 步骤5：检查返回值
        print(f"    返回值: {reply}")
        if reply is not None and reply != "":
            print("-" * 60)
            print(f"[步骤5] 获得有效回复，策略链中断")
            print(f"         （如果此函数设定了 final=False，则继续遍历）")
            print("=" * 60)
            return reply
        else:
            print(f"    返回值为空，继续检查下一个函数")

    # 步骤6：没有任何函数返回有效回复
    print("\n" + "-" * 60)
    print("[步骤6] 所有策略函数均未返回有效回复")
    print("         将进入 LLM 生成回复流程（如果配置了 llm_config）")
    print("=" * 60)
    return ""


# ============================================================
# 第四部分：演示用的回复函数
# ============================================================

def my_prefix_reply(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
    """
    前缀匹配回复示例：处理以 "/help" 开头的消息

    当用户输入以 "/help" 开头时，返回帮助信息
    """
    latest_message = messages[-1]
    content = latest_message.get("content", "")

    if content.startswith("/help"):
        return """
        可用命令：
        /help - 显示此帮助信息
        /status - 查看系统状态
        /quit - 退出对话
        """
    return ""


def my_regex_reply(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
    """
    正则匹配回复示例：处理包含邮箱格式的消息

    当消息中包含邮箱地址时，提取并确认
    """
    latest_message = messages[-1]
    content = latest_message.get("content", "")

    # 匹配邮箱正则表达式
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    matches = email_pattern.findall(content)

    if matches:
        return f"检测到邮箱地址: {', '.join(matches)}，请问需要我帮您做什么？"
    return ""


def my_func_reply(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
    """
    函数条件判断回复示例：VIP用户专属回复

    只有当发送者是 "vip_user" 时才触发
    """
    latest_message = messages[-1]
    sender_role = latest_message.get("role", "")

    # 检查发送者是否是 VIP 用户
    is_vip = sender_role.lower() == "vip_user" or (
        isinstance(config, dict) and config.get("vip_users", []).count(sender_role) > 0
    )

    if is_vip:
        return "欢迎 VIP 用户！您享有优先响应权。"
    return ""


def my_executable_reply(messages: List[Dict], sender: str, config: Optional[Dict] = None) -> str:
    """
    可执行内容回复示例：代码执行结果处理

    当消息包含代码执行结果时进行处理
    """
    latest_message = messages[-1]
    content = latest_message.get("content", "")

    # 检测是否包含代码执行结果
    if "```output" in content or "[代码执行结果]" in content:
        return "已收到代码执行结果，正在分析..."
    return ""


# ============================================================
# 第五部分：演示场景
# ============================================================

def demo_basic_flow():
    """
    演示1：基本策略链执行流程
    """
    print("\n" + "#" * 60)
    print("# 演示1：基本策略链执行流程")
    print("#" * 60)

    # 构建策略链（按 priority 从小到大排列）
    reply_func_list = [
        # priority=0：最高优先级，前缀匹配
        ReplyFuncRecord(
            reply_func=my_prefix_reply,
            trigger="/help",
            config=None,
            priority=0
        ),
        # priority=1：次优先级，正则匹配
        ReplyFuncRecord(
            reply_func=my_regex_reply,
            trigger=re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            config=None,
            priority=1
        ),
        # priority=2：第三优先级，函数判断
        ReplyFuncRecord(
            reply_func=my_func_reply,
            trigger=lambda msg, sender: True,  # 简化：始终触发
            config={"vip_users": ["alice", "bob"]},
            priority=2
        ),
    ]

    # 模拟用户消息
    test_messages = [
        {"role": "user", "content": "你好，请问有什么帮助？"}
    ]

    print("\n>>> 测试消息: '你好，请问有什么帮助？'")
    result = simulate_generate_reply(test_messages, reply_func_list)
    print(f">>> 最终回复: {result if result else '(无回复，将使用 LLM 生成)'}")


def demo_vip_flow():
    """
    演示2：VIP用户优先响应
    """
    print("\n" + "#" * 60)
    print("# 演示2：VIP用户优先响应")
    print("#" + "#" * 60)

    # 构建策略链
    reply_func_list = [
        ReplyFuncRecord(
            reply_func=my_func_reply,
            trigger=lambda msg, sender: sender.lower() == "vip_user" if sender else False,
            config={"vip_users": ["vip_user"]},
            priority=0  # VIP 最高优先级
        ),
        ReplyFuncRecord(
            reply_func=my_prefix_reply,
            trigger="/help",
            config=None,
            priority=1
        ),
    ]

    # VIP 用户消息
    vip_messages = [
        {"role": "vip_user", "content": "我要查询我的账户状态"}
    ]

    print("\n>>> VIP用户消息: '我要查询我的账户状态'")
    result = simulate_generate_reply(vip_messages, reply_func_list)
    print(f">>> 最终回复: {result}")


def demo_email_flow():
    """
    演示3：邮箱识别与处理
    """
    print("\n" + "#" * 60)
    print("# 演示3：邮箱识别与处理")
    print("#" * 60)

    reply_func_list = [
        ReplyFuncRecord(
            reply_func=my_regex_reply,
            trigger=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            config=None,
            priority=0
        ),
    ]

    email_messages = [
        {"role": "user", "content": "请发送到这个邮箱: test@example.com"}
    ]

    print("\n>>> 包含邮箱的消息: '请发送到这个邮箱: test@example.com'")
    result = simulate_generate_reply(email_messages, reply_func_list)
    print(f">>> 最终回复: {result}")


def demo_final_flag_flow():
    """
    演示4：final 标志中断机制

    注意：这是一个概念演示，实际的 final 标志行为需要查看 AutoGen 源码
    在真正的 AutoGen 中，某些回复函数可以设置 flags 来控制是否继续遍历
    """
    print("\n" + "#" * 60)
    print("# 演示4：final 标志中断机制（概念说明）")
    print("#" * 60)

    print("""
    final 标志中断机制说明：
    ─────────────────────────────────────────────────────────
    1. 默认情况下，当某个回复函数返回非空内容时，遍历立即停止
    2. 这称为"短路效应"——第一个匹配的回复立即生效

    3. 如果某个回复函数希望返回内容但允许继续遍历，
       需要设置特殊的 flags 参数（视具体实现而定）

    4. 常见的 final=False 场景：
       - 日志记录函数：记录日志但不影响后续处理
       - 统计函数：收集信息但允许正常回复继续
       - 预处理函数：对消息进行预处理后继续传递

    5. 实际应用中：
       - 大部分回复函数是 final=True（一旦返回就停止）
       - 只有需要"副作用"的函数才会设为 final=False
    ─────────────────────────────────────────────────────────
    """)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    # 加载环境变量
    load_env()

    # 运行所有演示
    demo_basic_flow()
    demo_vip_flow()
    demo_email_flow()
    demo_final_flag_flow()

    print("\n" + "=" * 60)
    print("generate_reply 流程演示结束")
    print("=" * 60)
    print("""
    下一步学习：
    - 查看 register_reply_demo.py 了解如何注册自定义回复函数
    - 阅读 AutoGen 源码中的 ConversableAgent.generate_reply() 实现
    - 实践：编写自己的 reply_func 并注册到策略链中
    """)