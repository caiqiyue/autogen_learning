"""
消息处理示例 - 展示ConversableAgent的消息发送、接收和处理机制

本文件展示AutoGen中消息的工作原理，包括：
- 消息的创建和发送
- 消息的接收和处理
- 消息队列的管理
- 对话状态机的状态转换

作者：AutoGen学习课程
学习目标：掌握消息管理机制和对话状态转换
"""

import os
import time
from typing import Union, List, Optional, Dict, Any
from enum import Enum

# ============================================================
# 第一部分：导入AutoGen核心组件
# ============================================================

# 导入ConversableAgent类 - AutoGen中最核心的智能体类
# 用于创建具有对话能力的智能体
from autogen import ConversableAgent

# 导入LocalExecutableCodeExecutor - 用于执行Python代码
from autogen import LocalExecutableCodeExecutor

# ============================================================
# 第二部分：消息格式定义
# ============================================================

class MessageRole(Enum):
    """
    消息角色枚举

    在AutoGen和OpenAI的API中，消息角色包括：
    - system: 系统消息，用于设置智能体的行为和角色
    - user: 用户消息，来自用户或外部系统的输入
    - assistant: 助手消息，由智能体生成的回答
    - tool: 工具消息，包含工具执行结果
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


def create_message(role: str, content: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    创建标准格式的消息

    参数:
        role: 消息角色，可以是 "system"、"user"、"assistant"、"tool"
        content: 消息的正文内容
        name: 可选的发送者名称

    返回:
        dict: 标准化的消息字典

    示例:
        >>> msg = create_message("user", "你好，请帮我分析数据")
        >>> print(msg)
        {'role': 'user', 'content': '你好，请帮我分析数据'}
    """
    message = {
        "role": role,
        "content": content
    }

    # 如果提供了发送者名称，则添加到消息中
    if name:
        message["name"] = name

    return message


def create_tool_call_message(
    content: str,
    tool_calls: List[Dict[str, Any]],
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建包含工具调用的消息

    这种消息类型用于智能体请求执行工具的场景

    参数:
        content: 消息内容，通常描述要执行的操作
        tool_calls: 工具调用列表，每个元素包含工具名和参数
        name: 可选的发送者名称

    返回:
        dict: 包含工具调用的消息字典

    示例:
        >>> tool_msg = create_tool_call_message(
        ...     content="我需要查询天气",
        ...     tool_calls=[
        ...         {"name": "get_weather", "arguments": {"city": "北京"}}
        ...     ]
        ... )
    """
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls
    }

    if name:
        message["name"] = name

    return message


# ============================================================
# 第三部分：消息处理演示（模拟）
# ============================================================

def demonstrate_message_flow():
    """
    演示消息在ConversableAgent中的流动过程

    这个函数模拟了消息从创建、发送到接收、处理的完整流程
    """

    print("=" * 60)
    print("消息处理流程演示")
    print("=" * 60)

    # 步骤1：创建消息
    print("\n【步骤1】创建消息...")
    user_message = create_message(
        role="user",
        content="请帮我用Python写一个快速排序算法",
        name="user_001"
    )
    print(f"创建的消息: {user_message}")

    # 步骤2：准备LLM配置
    print("\n【步骤2】准备LLM配置...")
    llm_config = {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY", "dummy-key"),
        "temperature": 0.7,
        "max_tokens": 2000
    }
    print(f"LLM配置: model={llm_config['model']}, temperature={llm_config['temperature']}")

    # 步骤3：创建智能体
    print("\n【步骤3】创建智能体...")
    coding_agent = ConversableAgent(
        name="coding_assistant_demo",
        system_message="""你是一个专业的Python编程助手。
        你的专长是编写高质量、易读的Python代码。
        请总是添加详细的中文注释来解释代码逻辑。""",
        llm_config=llm_config,
        code_executor=LocalExecutableCodeExecutor(timeout=30),
        human_input_mode="NEVER"  # 编程任务全自动运行
    )
    print(f"智能体创建成功: {coding_agent.name}")

    # 步骤4：模拟消息发送
    print("\n【步骤4】模拟消息发送...")
    print("注意：在实际运行中，消息发送是通过agent.send()或agent.generate_reply()实现的")

    # 模拟发送消息
    messages_to_send = [
        create_message("system", "你是一个乐于助人的AI助手", name="system"),
        user_message,
        create_message("user", "代码需要包含单元测试", name="user_002")
    ]

    print(f"准备发送 {len(messages_to_send)} 条消息:")
    for i, msg in enumerate(messages_to_send, 1):
        print(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")

    # 步骤5：演示消息接收
    print("\n【步骤5】演示消息接收机制...")
    print("智能体接收消息后会：")
    print("  1. 解析消息内容和角色")
    print("  2. 检查是否需要执行工具或代码")
    print("  3. 生成回复内容")
    print("  4. 更新内部消息队列")

    # 步骤6：模拟回复生成
    print("\n【步骤6】模拟回复内容...")
    # 在实际场景中，这会由LLM生成
    simulated_response = """以下是快速排序算法的实现：

```python
def quicksort(arr):
    '''
    快速排序算法

    参数:
        arr: 待排序的列表

    返回:
        list: 排序后的列表

    原理：
        1. 选择基准元素（pivot）
        2. 分区：将小于基准的元素放在左边，大于基准的放在右边
        3. 递归地对左右两部分继续排序
    '''
    # 基本情况：空列表或单元素列表不需要排序
    if len(arr) <= 1:
        return arr

    # 选择基准元素（这里选择中间元素）
    pivot = arr[len(arr) // 2]

    # 分区操作
    left = [x for x in arr if x < pivot]   # 小于基准的元素
    middle = [x for x in arr if x == pivot]  # 等于基准的元素
    right = [x for x in arr if x > pivot]   # 大于基准的元素

    # 递归排序并合并结果
    return quicksort(left) + middle + quicksort(right)

# 测试代码
if __name__ == "__main__":
    test_list = [64, 34, 25, 12, 22, 11, 90]
    print(f"原始列表: {test_list}")
    sorted_list = quicksort(test_list.copy())
    print(f"排序后: {sorted_list}")
```
"""
    print("模拟回复已生成（实际运行中由LLM生成）")

    return {
        "user_messages": messages_to_send,
        "assistant_response": simulated_response,
        "agent_name": coding_agent.name
    }


# ============================================================
# 第四部分：消息队列管理
# ============================================================

class MessageQueueManager:
    """
    消息队列管理器（模拟ConversableAgent内部的消息管理机制）

    这个类展示了如何管理对话消息队列，包括：
    - 添加消息
    - 获取消息历史
    - 清空消息
    - 消息搜索
    """

    def __init__(self):
        """
        初始化消息队列

        使用字典存储消息，按会话ID组织
        """
        # 消息存储结构：{session_id: [messages]}
        self._messages: Dict[str, List[Dict[str, Any]]] = {}

        # 当前活跃的会话ID
        self._current_session_id: Optional[str] = None

        # 消息计数器，用于生成唯一消息ID
        self._message_counter = 0

    def create_session(self, session_id: str) -> None:
        """
        创建新会话

        参数:
            session_id: 会话唯一标识符
        """
        if session_id not in self._messages:
            self._messages[session_id] = []
            print(f"创建新会话: {session_id}")
        else:
            print(f"会话已存在: {session_id}")

        self._current_session_id = session_id

    def append_message(
        self,
        role: str,
        content: str,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加消息到当前会话

        参数:
            role: 消息角色
            content: 消息内容
            name: 可选的发送者名称

        返回:
            dict: 创建的消息对象
        """
        if not self._current_session_id:
            # 如果没有活跃会话，创建一个默认会话
            self.create_session("default")

        # 创建消息
        message = create_message(role, content, name)

        # 添加元数据
        self._message_counter += 1
        message["message_id"] = self._message_counter
        message["timestamp"] = time.time()

        # 添加到队列
        self._messages[self._current_session_id].append(message)

        print(f"消息已添加: [{role}] {content[:30]}...")
        return message

    def get_messages(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取会话消息历史

        参数:
            session_id: 可选的会话ID，如果为None则获取当前会话

        返回:
            list: 消息列表
        """
        target_session = session_id or self._current_session_id or "default"

        if target_session not in self._messages:
            return []

        return self._messages[target_session]

    def clear_messages(self, session_id: Optional[str] = None) -> None:
        """
        清空会话消息

        参数:
            session_id: 可选的会话ID，如果为None则清空当前会话
        """
        target_session = session_id or self._current_session_id

        if target_session and target_session in self._messages:
            self._messages[target_session] = []
            print(f"已清空会话消息: {target_session}")

    def search_messages(
        self,
        keyword: str,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        在消息中搜索关键词

        参数:
            keyword: 搜索关键词
            session_id: 可选的会话ID

        返回:
            list: 包含关键词的消息列表
        """
        messages = self.get_messages(session_id)
        results = [
            msg for msg in messages
            if keyword.lower() in msg.get("content", "").lower()
        ]
        print(f"搜索结果: 找到 {len(results)} 条包含 '{keyword}' 的消息")
        return results

    def get_last_message(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取最近一条消息

        参数:
            session_id: 可选的会话ID

        返回:
            dict: 最近的消息，如果没有消息则返回None
        """
        messages = self.get_messages(session_id)
        return messages[-1] if messages else None


def demonstrate_message_queue():
    """
    演示消息队列管理功能
    """

    print("\n" + "=" * 60)
    print("消息队列管理演示")
    print("=" * 60)

    # 创建消息队列管理器
    manager = MessageQueueManager()

    # 创建会话
    print("\n【步骤1】创建会话...")
    manager.create_session("conversation_001")

    # 添加消息
    print("\n【步骤2】添加消息...")
    manager.append_message("system", "你是一个专业的Python编程助手")
    manager.append_message("user", "请帮我写一个快速排序算法")
    manager.append_message("assistant", "当然可以，以下是快速排序算法的实现...")
    manager.append_message("user", "能否加上单元测试？")
    manager.append_message("assistant", "好的，我来添加单元测试...")

    # 获取消息历史
    print("\n【步骤3】获取消息历史...")
    messages = manager.get_messages()
    print(f"当前会话共有 {len(messages)} 条消息")

    # 搜索消息
    print("\n【步骤4】搜索消息...")
    search_results = manager.search_messages("快速排序")
    print(f"找到 {len(search_results)} 条相关消息")

    # 获取最近消息
    print("\n【步骤5】获取最近消息...")
    last_msg = manager.get_last_message()
    if last_msg:
        print(f"最近消息: [{last_msg['role']}] {last_msg['content'][:40]}...")

    # 清空消息
    print("\n【步骤6】清空消息...")
    manager.clear_messages()
    remaining = manager.get_messages()
    print(f"清空后会话消息数: {len(remaining)}")


# ============================================================
# 第五部分：状态机演示
# ============================================================

class AgentState(Enum):
    """
    智能体状态枚举

    表示ConversableAgent可能处于的状态：
    - IDLE: 空闲状态，等待输入
    - RUNNING: 运行状态，正在处理请求
    - WAITING: 等待状态，等待人工输入或工具执行
    - TERMINATED: 终止状态，对话结束
    """
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"


class StateTransitionDemo:
    """
    状态转换演示类

    展示ConversableAgent如何在不同状态之间转换
    """

    def __init__(self):
        """
        初始化状态机
        """
        self.current_state = AgentState.IDLE
        self.transition_history = []

    def transition_to(self, new_state: AgentState, reason: str) -> None:
        """
        执行状态转换

        参数:
            new_state: 目标状态
            reason: 转换原因
        """
        old_state = self.current_state
        self.current_state = new_state

        # 记录转换历史
        self.transition_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time()
        })

        print(f"状态转换: {old_state.value} -> {new_state.value} (原因: {reason})")

    def receive_message(self) -> None:
        """
        接收消息触发转换：IDLE -> RUNNING
        """
        if self.current_state == AgentState.IDLE:
            self.transition_to(AgentState.RUNNING, "收到用户消息")

    def need_human_input(self) -> None:
        """
        需要人工输入触发转换：RUNNING -> WAITING
        """
        if self.current_state == AgentState.RUNNING:
            self.transition_to(AgentState.WAITING, "需要人工确认")

    def human_input_received(self) -> None:
        """
        人工输入完成触发转换：WAITING -> RUNNING
        """
        if self.current_state == AgentState.WAITING:
            self.transition_to(AgentState.RUNNING, "人工输入完成")

    def need_termination(self) -> None:
        """
        需要终止触发转换：RUNNING -> TERMINATED
        """
        if self.current_state == AgentState.RUNNING:
            self.transition_to(AgentState.TERMINATED, "收到终止信号")

    def reset(self) -> None:
        """
        重置状态机
        """
        self.transition_to(AgentState.IDLE, "重置状态机")
        self.transition_history.clear()


def demonstrate_state_machine():
    """
    演示状态机的工作原理
    """

    print("\n" + "=" * 60)
    print("状态机工作原理演示")
    print("=" * 60)

    # 创建状态机演示实例
    state_machine = StateTransitionDemo()

    # 场景1：正常对话流程
    print("\n【场景1】正常对话流程...")
    print("初始状态:", state_machine.current_state.value)

    state_machine.receive_message()  # IDLE -> RUNNING
    # ... 执行处理
    state_machine.need_termination()  # RUNNING -> TERMINATED

    print("最终状态:", state_machine.current_state.value)

    # 重置
    state_machine.reset()

    # 场景2：需要人工介入的流程
    print("\n【场景2】需要人工介入的流程...")
    print("初始状态:", state_machine.current_state.value)

    state_machine.receive_message()  # IDLE -> RUNNING
    state_machine.need_human_input()  # RUNNING -> WAITING
    state_machine.human_input_received()  # WAITING -> RUNNING
    state_machine.need_termination()  # RUNNING -> TERMINATED

    print("最终状态:", state_machine.current_state.value)

    # 打印转换历史
    print("\n【转换历史】")
    for i, transition in enumerate(state_machine.transition_history, 1):
        print(f"  {i}. {transition['from']} -> {transition['to']} ({transition['reason']})")


# ============================================================
# 第六部分：完整消息处理流程
# ============================================================

def demonstrate_complete_flow():
    """
    演示完整的消息处理流程

    整合消息创建、发送、接收、响应生成的完整过程
    """

    print("\n" + "=" * 60)
    print("完整消息处理流程")
    print("=" * 60)

    # 步骤1：初始化
    print("\n【步骤1】初始化系统...")
    print("  - 创建消息队列管理器")
    print("  - 初始化LLM配置")
    print("  - 创建智能体实例")

    # 步骤2：用户发送消息
    print("\n【步骤2】用户发送消息...")
    user_msg = create_message("user", "请用Python实现归并排序算法", name="user_001")
    print(f"  用户消息: {user_msg['content']}")

    # 步骤3：智能体接收并处理
    print("\n【步骤3】智能体接收并处理...")
    print("  - 解析消息内容")
    print("  - 理解用户意图（请求编写归并排序）")
    print("  - 生成代码实现")

    # 步骤4：智能体生成回复
    print("\n【步骤4】智能体生成回复...")
    assistant_response = """以下是归并排序算法的Python实现：

```python
def merge_sort(arr):
    '''
    归并排序算法

    参数:
        arr: 待排序的列表

    返回:
        list: 排序后的列表

    算法复杂度:
        - 时间复杂度: O(n log n)
        - 空间复杂度: O(n)

    原理:
        1. 分解：将数组分成两半
        2. 递归：对左右两部分分别排序
        3. 合并：将两个有序数组合并成一个有序数组
    '''
    # 基本情况：空数组或单元素数组不需要排序
    if len(arr) <= 1:
        return arr

    # 分解：将数组从中间分成两半
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # 递归：对左右两部分分别排序
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)

    # 合并：将两个有序数组合并成一个有序数组
    return merge(left_sorted, right_sorted)


def merge(left, right):
    '''
    合并两个有序数组

    参数:
        left: 左边的有序数组
        right: 右边的有序数组

    返回:
        list: 合并后的有序数组
    '''
    result = []
    i = j = 0

    # 比较两个数组的元素，按顺序放入结果数组
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 将剩余的元素添加到结果数组
    result.extend(left[i:])
    result.extend(right[j:])

    return result


# 测试代码
if __name__ == "__main__":
    test_list = [64, 34, 25, 12, 22, 11, 90, 5]
    print(f"原始列表: {test_list}")
    sorted_list = merge_sort(test_list.copy())
    print(f"排序后: {sorted_list}")
```
"""
    print("  智能体生成了包含代码的回复")

    # 步骤5：执行代码（如需要）
    print("\n【步骤5】代码执行（可选）...")
    print("  - 如果用户请求执行代码，智能体会使用代码执行器运行")
    print("  - 本例中仅展示代码，不自动执行")

    # 步骤6：更新对话历史
    print("\n【步骤6】更新对话历史...")
    print("  - 用户消息和助手回复都被保存到_oai_messages")
    print("  - 对话历史用于维持上下文理解")

    print("\n流程完成！")


# ============================================================
# 主函数 - 运行所有演示
# ============================================================

def main():
    """
    主函数：运行所有消息处理演示
    """

    print("=" * 60)
    print("AutoGen 消息处理机制演示")
    print("=" * 60)

    # 1. 消息格式创建演示
    print("\n" + "-" * 60)
    print("第一部分：消息格式定义")
    print("-" * 60)
    print("创建了标准消息格式和工具调用消息格式")

    # 2. 消息流动演示
    print("\n" + "-" * 60)
    print("第二部分：消息流动过程")
    print("-" * 60)
    demonstrate_message_flow()

    # 3. 消息队列管理演示
    print("\n" + "-" * 60)
    print("第三部分：消息队列管理")
    print("-" * 60)
    demonstrate_message_queue()

    # 4. 状态机演示
    print("\n" + "-" * 60)
    print("第四部分：状态机工作原理")
    print("-" * 60)
    demonstrate_state_machine()

    # 5. 完整流程演示
    print("\n" + "-" * 60)
    print("第五部分：完整消息处理流程")
    print("-" * 60)
    demonstrate_complete_flow()

    # 总结
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("""
本演示涵盖的关键概念：

1. 消息格式
   - role: 消息角色（system/user/assistant/tool）
   - content: 消息内容
   - name: 发送者名称
   - tool_calls: 工具调用信息

2. 消息管理
   - 消息队列按会话组织
   - 支持消息添加、搜索、清空
   - 消息历史用于上下文理解

3. 状态机
   - IDLE: 等待输入
   - RUNNING: 处理请求
   - WAITING: 等待人工/工具
   - TERMINATED: 对话结束

4. 完整流程
   用户消息 -> 智能体接收 -> 理解意图 ->
   生成回复 -> 执行代码（如需要）-> 更新历史
    """)


# ============================================================
# 程序入口点
# ============================================================

if __name__ == "__main__":
    # 运行主函数
    main()

    """
    运行说明：
    1. 确保已安装 autogen 包：pip install autogen
    2. 设置环境变量 OPENAI_API_KEY（如果需要实际LLM调用）
    3. 运行命令：python message_handling.py

    注意：
    - 本演示主要展示消息处理的原理和机制
    - 不需要实际的API密钥即可运行基本演示
    - 状态机和消息队列的演示是纯逻辑性的
    """