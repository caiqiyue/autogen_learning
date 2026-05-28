# code_executor_config.py
# 第6节 Code Executor配置与安全机制 - Code Executor配置详解
#
# 本文件展示AutoGen中Code Executor的各种配置方式，包括：
# 1. 本地代码执行器（LocalSingleCodeExecutor）
# 2. 容器化代码执行器（DockerExecutor）
# 3. 代码执行配置参数详解
#
# Code Executor是AutoGen框架中用于安全执行LLM生成代码的核心组件

from pathlib import Path
import time
import shutil
import os

# ============================================================
# 第一部分：Code Executor 概述与执行模型
# ============================================================
#
# AutoGen的Code Executor设计理念：
# 1. 隔离性：通过容器或单独进程执行代码，防止恶意代码损害主机
# 2. 超时控制：为每次执行设置超时限制，避免无限循环占用资源
# 3. 结果解析：统一格式化代码执行结果，便于后续处理
#
# 执行流程：
#   LLM生成代码 → Code Executor接收 → 创建执行环境 → 运行代码 → 捕获输出 → 返回结果

def explain_executor_architecture():
    """
    解释Code Executor的架构设计
    """
    print("=" * 60)
    print("Code Executor 架构设计")
    print("=" * 60)
    print("""
AutoGen的Code Executor采用分层设计：

1. 代码生成层（LLM）
   - LLM根据用户需求生成Python或其他语言代码

2. 执行器抽象层（CodeExecutor基类）
   - 定义统一的接口：execute_code()
   - 支持不同执行后端：本地、容器、远程

3. 执行环境隔离层
   - LocalSingleCodeExecutor：单次执行，进程级隔离
   - DockerExecutor：容器级隔离，更高安全性

4. 结果处理层
   - 捕获 stdout/stderr
   - 收集执行时间、内存使用等信息
   - 错误分类与格式化输出
    """)


# ============================================================
# 第二部分：code_execution_config 参数详解
# ============================================================

def explain_code_execution_config():
    """
    详解code_execution_config各参数的作用
    """
    print("\n" + "=" * 60)
    print("code_execution_config 参数详解")
    print("=" * 60)
    print("""
code_execution_config 是配置代码执行器的核心字典，包含以下关键参数：

1. work_dir - 工作目录
   - 类型：str 或 Path
   - 作用：代码执行的根目录，生成的代码文件将保存在此目录下
   - 默认值：None（使用系统临时目录）
   - 示例：work_dir="./code_workspace" 或 work_dir=Path("./my_code")

2. use_docker - 是否使用Docker容器
   - 类型：bool 或 str（容器名）
   - 作用：是否在Docker容器中执行代码，提供更强的隔离
   - 默认值：True（推荐在生产环境使用）
   - False值：用于开发调试模式，直接在本地执行

3. timeout - 执行超时时间
   - 类型：int（秒）
   - 作用：单次代码执行的最大时长，超过则终止
   - 默认值：30秒
   - 示例：timeout=60 表示60秒超时

4. last_n_messages - 参考消息数量
   - 类型：int
   - 作用：决定代码执行错误时，向LLM反馈多少条历史消息
   - 默认值：6
   - 设为0表示只反馈最后一次执行结果

5. temperature - 代码生成的温度参数
   - 类型：float
   - 作用：控制代码生成的多样性（通常较低以保证稳定性）
   - 默认值：0.0（确定性生成）
    """)


# ============================================================
# 第三部分：本地代码执行器配置
# ============================================================

def create_local_code_executor():
    """
    演示如何创建本地代码执行器
    本地执行器适用于开发调试阶段，直接在当前进程执行代码
    """
    from autogen.code_utils import create_virtual_env_dict

    # 构建本地执行器的配置字典
    local_executor_config = {
        "work_dir": "./code_output",      # 代码文件保存目录
        "use_docker": False,              # 不使用Docker，用于本地调试
        "timeout": 60,                    # 超时时间60秒
        "last_n_messages": 10,            # 错误反馈时参考最近10条消息
    }

    print("\n" + "=" * 60)
    print("本地代码执行器配置示例")
    print("=" * 60)
    print(f"配置内容：{local_executor_config}")
    print("""
使用场景：
- 开发调试阶段，快速迭代代码
- 单机环境，无Docker支持
- 需要快速查看执行结果

注意事项：
- use_docker=False 时，代码直接在本地执行
- 存在安全风险，LLM生成的代码可能访问敏感资源
- 仅在可信环境或开发阶段使用
    """)

    # 演示如何创建虚拟环境字典配置
    # 虚拟环境可以隔离不同项目的依赖
    virtual_env_dict = create_virtual_env_dict(
        name="autogen_code_env",          # 环境名称
        python_path=None,                # Python解释器路径，None则使用当前环境
    )
    print(f"虚拟环境配置：{virtual_env_dict}")

    return local_executor_config


# ============================================================
# 第四部分：Docker容器化执行器配置
# ============================================================

def create_docker_executor():
    """
    演示如何创建Docker容器化代码执行器
    容器化执行器适用于生产环境，提供更强的隔离和安全保障
    """
    # 构建Docker执行器的配置字典
    docker_executor_config = {
        "work_dir": "./docker_code_workspace",   # 容器内的工作目录
        "use_docker": "autogen-code-executor",   # Docker镜像名称，也可设为True
        "timeout": 120,                          # 生产环境超时时间可以设长一些
        "last_n_messages": 6,                    # 默认值
    }

    print("\n" + "=" * 60)
    print("Docker容器化代码执行器配置示例")
    print("=" * 60)
    print(f"配置内容：{docker_executor_config}")
    print("""
use_docker 参数的三种形式：

1. use_docker=True（推荐）
   - AutoGen自动选择合适的镜像
   - 简化配置，适合大多数场景

2. use_docker="镜像名称:标签"（自定义镜像）
   - 使用指定的Docker镜像
   - 适合有特殊依赖要求的场景
   - 示例："autogen-code-executor:latest"

3. use_docker=False（不推荐）
   - 仅用于开发调试
   - 存在安全风险

Docker执行器的优势：
- 操作系统级隔离，防止恶意代码损害主机
- 环境一致性好，跨平台部署无问题
- 资源限制，可控制CPU/内存使用
    """)

    return docker_executor_config


# ============================================================
# 第五部分：组合配置示例
# ============================================================

def demonstrate_combined_config():
    """
    演示如何根据不同场景组合配置Code Executor
    """
    print("\n" + "=" * 60)
    print("不同场景的配置组合示例")
    print("=" * 60)

    # 场景1：开发调试模式 - 快速反馈，不使用Docker
    dev_config = {
        "work_dir": "./dev_workspace",
        "use_docker": False,
        "timeout": 30,           # 快速失败
        "last_n_messages": 3,    # 简洁错误信息
    }

    # 场景2：生产环境模式 - 安全隔离，使用Docker
    prod_config = {
        "work_dir": "/app/code_workspace",   # Docker容器内路径
        "use_docker": True,                   # 使用默认Docker镜像
        "timeout": 120,                       # 更长的超时时间
        "last_n_messages": 6,                 # 完整的上下文
    }

    # 场景3：数据分析场景 - 大文件处理，长超时
    data_config = {
        "work_dir": "./data_analysis_workspace",
        "use_docker": "python:3.11-slim",     # 自定义Python镜像
        "timeout": 300,                       # 5分钟超时，处理大数据
        "last_n_messages": 10,
    }

    # 场景4：代码审查场景 - 短超时，快速反馈
    review_config = {
        "work_dir": "./code_review_workspace",
        "use_docker": True,
        "timeout": 15,                        # 快速检查
        "last_n_messages": 2,
    }

    scenarios = [
        ("开发调试", dev_config),
        ("生产环境", prod_config),
        ("数据分析", data_config),
        ("代码审查", review_config),
    ]

    for name, config in scenarios:
        print(f"\n【{name}】配置：")
        for key, value in config.items():
            print(f"  {key}: {value}")


# ============================================================
# 第六部分：代码执行器的实际使用
# ============================================================

def show_executor_usage_with_agent():
    """
    展示如何在Agent中使用Code Executor
    """
    print("\n" + "=" * 60)
    print("在ConversableAgent中使用Code Executor")
    print("=" * 60)

    # 定义在Agent初始化时传入代码执行配置
    # 以下配置将创建一个支持代码执行的UserProxyAgent

    code_execution_config = {
        "work_dir": "./code_execution_workspace",
        "use_docker": True,
        "timeout": 60,
        "last_n_messages": 6,
    }

    print("""
代码示例：

from autogen.agentchat.user_proxy_agent import UserProxyAgent

# 创建UserProxyAgent时传入code_execution_config
user_proxy = UserProxyAgent(
    name="code_executor_agent",
    code_execution_config=code_execution_config,  # 启用代码执行
    human_input_mode="NEVER",                    # 不需要人工介入
)

# 当Agent接收到需要执行代码的请求时，
# 会自动调用Code Executor执行代码并返回结果
    """)

    # 演示如何检查Docker是否可用
    def check_docker_available():
        """检查Docker环境是否可用"""
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    docker_available = check_docker_available()
    print(f"\nDocker环境检查：{'可用' if docker_available else '不可用'}")
    if not docker_available:
        print("提示：请安装Docker以使用容器化代码执行器")
        print("安装指南：https://docs.docker.com/get-docker/")


# ============================================================
# 第七部分：错误处理与结果解析
# ============================================================

def explain_error_handling():
    """
    解释Code Executor的错误处理机制
    """
    print("\n" + "=" * 60)
    print("Code Executor 错误处理与结果解析")
    print("=" * 60)
    print("""
Code Executor执行后返回的结果包含以下信息：

1. output - 标准输出内容
   - 代码的print输出、正常结果等

2. error - 错误信息
   - 语法错误、运行时异常、超时信息等

3. exit_code - 退出码
   - 0表示成功
   - 非0表示失败

4. elapsed_time - 执行耗时
   - 单位：秒
   - 用于性能分析和超时调整

5. code_file - 代码文件路径
   - 执行的代码保存在此文件

错误分类：
- 语法错误（SyntaxError）：代码不符合Python语法
- 运行时错误（RuntimeError）：代码执行时抛出异常
- 超时错误（TimeoutError）：执行时间超过限制
- 权限错误（PermissionError）：文件/目录访问被拒绝
    """)


# ============================================================
# 程序入口
# ============================================================

def main():
    """
    主函数：运行Code Executor配置示例
    """
    print("=" * 70)
    print("第6节 Code Executor配置与安全机制 - 配置详解")
    print("=" * 70)

    # 1. 解释架构设计
    explain_executor_architecture()

    # 2. 详解配置参数
    explain_code_execution_config()

    # 3. 本地执行器配置
    create_local_code_executor()

    # 4. Docker执行器配置
    create_docker_executor()

    # 5. 组合配置示例
    demonstrate_combined_config()

    # 6. Agent中使用示例
    show_executor_usage_with_agent()

    # 7. 错误处理说明
    explain_error_handling()

    print("\n" + "=" * 70)
    print("示例运行完成")
    print("=" * 70)
    print("""
下一步学习：
- 学习 docker_sandbox.py 了解Docker隔离环境的具体配置
- 理解LocalSingleCodeExecutor与DockerExecutor的适用场景
    """)


if __name__ == "__main__":
    main()