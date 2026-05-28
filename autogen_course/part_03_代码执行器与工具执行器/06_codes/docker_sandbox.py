# docker_sandbox.py
# 第6节 Code Executor配置与安全机制 - Docker隔离环境配置
#
# 本文件展示如何配置Docker沙箱环境进行安全的代码执行
# 涵盖Docker镜像选择、容器配置、资源限制等
#
# Docker隔离是AutoGen生产环境中推荐的安全机制

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List

# ============================================================
# 第一部分：Docker环境基础检查
# ============================================================

def check_docker_environment() -> Dict[str, bool]:
    """
    检查Docker环境是否正确配置

    Returns:
        Dict[str, bool]: 包含各项检查结果的字典
    """
    print("=" * 60)
    print("Docker 环境检查")
    print("=" * 60)

    results = {
        "docker_installed": False,
        "docker_running": False,
        "docker_accessible": False,
    }

    # 检查1：Docker是否安装
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            results["docker_installed"] = True
            version_line = result.stdout.strip()
            print(f"[OK] Docker已安装: {version_line}")
        else:
            print("[FAIL] Docker未正确安装")
    except FileNotFoundError:
        print("[FAIL] 未找到docker命令，请安装Docker")
    except subprocess.TimeoutExpired:
        print("[FAIL] Docker命令执行超时")
    except Exception as e:
        print(f"[FAIL] 检查Docker时出错: {e}")

    # 检查2：Docker守护进程是否运行
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            results["docker_running"] = True
            print("[OK] Docker守护进程正在运行")
        else:
            print("[FAIL] Docker守护进程未运行，请启动Docker")
    except Exception as e:
        print(f"[FAIL] 检查Docker守护进程时出错: {e}")

    # 检查3：是否有权限访问Docker
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            results["docker_accessible"] = True
            print("[OK] 当前用户可以访问Docker")
        else:
            print("[FAIL] 当前用户无Docker访问权限")
            print("提示：确保用户属于docker组，或使用sudo运行")
    except Exception as e:
        print(f"[FAIL] 检查Docker权限时出错: {e}")

    return results


# ============================================================
# 第二部分：AutoGen Docker镜像管理
# ============================================================

class DockerImageManager:
    """
    Docker镜像管理器，用于管理AutoGen代码执行所需的Docker镜像
    """

    # AutoGen官方推荐的代码执行镜像
    OFFICIAL_IMAGES = {
        "python3.8": "python:3.8-slim",
        "python3.9": "python:3.9-slim",
        "python3.10": "python:3.10-slim",
        "python3.11": "python:3.11-slim",
        "autogen-base": "ag，阳光-ai/autogen-code-executor:latest",
    }

    def __init__(self):
        self.pulled_images: List[str] = []

    def pull_image(self, image_name: str) -> bool:
        """
        拉取Docker镜像

        Args:
            image_name: 镜像名称，格式：name:tag

        Returns:
            bool: 拉取是否成功
        """
        print(f"\n正在拉取镜像: {image_name}")
        try:
            result = subprocess.run(
                ["docker", "pull", image_name],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            if result.returncode == 0:
                print(f"[OK] 镜像 {image_name} 拉取成功")
                self.pulled_images.append(image_name)
                return True
            else:
                print(f"[FAIL] 镜像拉取失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"[FAIL] 镜像拉取超时（超过5分钟）")
            return False
        except Exception as e:
            print(f"[FAIL] 拉取镜像时出错: {e}")
            return False

    def list_local_images(self) -> List[str]:
        """
        列出本地已有的AutoGen相关镜像

        Returns:
            List[str]: 镜像名称列表
        """
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                images = [line.strip() for line in result.stdout.splitlines()]
                # 过滤出Python相关或AutoGen相关镜像
                relevant = [img for img in images if "python" in img.lower() or "autogen" in img.lower()]
                return relevant
            return []
        except Exception:
            return []

    def cleanup_unused_images(self) -> int:
        """
        清理未使用的镜像，释放磁盘空间

        Returns:
            int: 清理的镜像数量
        """
        print("\n正在清理未使用的Docker镜像...")
        try:
            result = subprocess.run(
                ["docker", "image", "prune", "-a", "-f"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                # 统计清理的镜像数量
                lines = result.stdout.splitlines()
                for line in lines:
                    if "Total reclaimed space" in line:
                        print(f"[OK] {line}")
                        return 1
                return 0
            return 0
        except Exception:
            return 0


# ============================================================
# 第三部分：Docker容器配置详解
# ============================================================

def explain_docker_container_config():
    """
    解释Docker容器配置的各项参数
    """
    print("\n" + "=" * 60)
    print("Docker容器配置参数详解")
    print("=" * 60)
    print("""
AutoGen的DockerExecutor支持以下容器配置：

1. 镜像配置（use_docker参数）
   - True：使用AutoGen默认镜像
   - "镜像名:标签"：使用自定义镜像

2. 工作目录（work_dir）
   - 容器内的代码执行目录
   - 建议使用绝对路径：/app/workspace

3. 环境变量
   - 可以传递API密钥等敏感信息
   - 格式：{"KEY": "VALUE"}

4. 资源限制
   - memory：内存限制，如"512m"表示512MB
   - cpu_period：CPU周期限制
   - cpu_quota：CPU配额

5. 网络配置
   - network_disabled：禁用网络（更安全）
   - 网络模式：bridge/host/none

容器安全建议：
- 禁用网络(--network none)防止数据泄露
- 限制内存防止恶意代码耗尽资源
- 设置只读文件系统防止写入敏感目录
- 使用非root用户运行代码
    """)


# ============================================================
# 第四部分：DockerExecutor配置示例
# ============================================================

def demonstrate_docker_executor_config():
    """
    演示不同场景下DockerExecutor的配置方法
    """
    print("\n" + "=" * 60)
    print("DockerExecutor 配置示例")
    print("=" * 60)

    # 示例1：基础Docker执行器配置
    basic_docker_config = {
        "work_dir": "/app/code_workspace",
        "use_docker": True,  # 使用默认镜像
        "timeout": 60,
        "last_n_messages": 6,
    }

    # 示例2：自定义Python版本
    custom_python_config = {
        "work_dir": "/app/workspace",
        "use_docker": "python:3.11-slim",  # 指定Python版本
        "timeout": 120,
        "last_n_messages": 8,
    }

    # 示例3：高安全级别配置
    high_security_config = {
        "work_dir": "/app/secure_workspace",
        "use_docker": "python:3.10-slim",
        "timeout": 30,
        "last_n_messages": 3,
        # 容器运行时选项（高级）
        # 注意：这些选项需要在创建容器时手动配置
        # AutoGen不直接支持，但了解这些参数有助于安全部署
        # "container_options": {
        #     "mem_limit": "256m",        # 内存限制256MB
        #     "cpu_period": 100000,       # CPU周期
        #     "cpu_quota": 50000,         # CPU配额（50%）
        #     "network_disabled": True,   # 禁用网络
        #     "read_only": True,          # 只读根文件系统
        # },
    }

    # 示例4：数据分析场景配置
    data_analysis_config = {
        "work_dir": "/app/data_workspace",
        "use_docker": "python:3.11-slim",
        "timeout": 600,  # 10分钟超时
        "last_n_messages": 10,
    }

    scenarios = [
        ("基础配置", basic_docker_config),
        ("自定义Python", custom_python_config),
        ("高安全级别", high_security_config),
        ("数据分析", data_analysis_config),
    ]

    for name, config in scenarios:
        print(f"\n【{name}】")
        for key, value in config.items():
            print(f"  {key}: {value}")


# ============================================================
# 第五部分：use_docker=False 的开发调试模式
# ============================================================

def explain_local_dev_mode():
    """
    解释use_docker=False的开发调试模式
    """
    print("\n" + "=" * 60)
    print("use_docker=False 开发调试模式")
    print("=" * 60)
    print("""
use_docker=False 模式下，代码直接在本地环境执行：

适用场景：
1. 开发调试阶段
   - 快速迭代代码
   - 便于使用调试器
   - 减少容器开销

2. 已确认代码安全性
   - LLM生成的代码经过验证
   - 不包含危险操作

3. 环境限制
   - 无Docker支持
   - 资源受限

风险警告：
- 代码可以直接访问主机所有资源
- 可能执行恶意操作：删除文件、窃取数据等
- 仅在可信环境和开发阶段使用

调试技巧：
- 设置短超时时间（timeout=10）快速失败
- 使用较小的last_n_messages减少上下文
- 配合日志记录代码执行过程
    """)


# ============================================================
# 第六部分：Docker安全最佳实践
# ============================================================

class DockerSecurityBestPractices:
    """
    Docker安全最佳实践集合
    """

    @staticmethod
    def get_security_options() -> Dict[str, any]:
        """
        获取推荐的安全配置选项

        Returns:
            Dict: 安全配置字典
        """
        return {
            # 资源限制
            "memory": "256m",              # 限制256MB内存
            "memory_swap": "256m",          # 限制交换空间
            "cpu_period": 100000,           # CPU周期100ms
            "cpu_quota": 50000,             # 50% CPU使用率

            # 安全限制
            "network_disabled": True,       # 禁用网络
            "read_only": True,              # 只读根文件系统
            "tmpfs": ["/tmp"],             # 使用tmpfs挂载/tmp

            # 用户权限
            "user": "1000:1000",            # 非root用户
        }

    @staticmethod
    def validate_code_safety(code: str) -> Dict[str, any]:
        """
        代码安全性预检查（示例实现）

        Args:
            code: 待执行的代码字符串

        Returns:
            Dict: 检查结果，包含is_safe布尔值和警告信息
        """
        warnings = []
        dangerous_patterns = [
            ("os.remove", "文件删除操作"),
            ("subprocess.call", "子进程调用"),
            ("requests.get", "网络请求"),
            ("eval(", "动态代码执行"),
            ("exec(", "代码执行"),
            ("__import__", "动态导入"),
            ("open(", "文件操作"),
            ("sys.exit", "系统退出"),
        ]

        for pattern, description in dangerous_patterns:
            if pattern in code:
                warnings.append(f"检测到{description}：{pattern}")

        return {
            "is_safe": len(warnings) == 0,
            "warnings": warnings,
        }


# ============================================================
# 第七部分：实际使用示例
# ============================================================

def demonstrate_practical_usage():
    """
    演示Code Executor的实际使用流程
    """
    print("\n" + "=" * 60)
    print("Code Executor 实际使用示例")
    print("=" * 60)

    print("""
在AutoGen中使用Docker执行器的基本流程：

1. 导入必要的模块
   from autogen.agentchat.user_proxy_agent import UserProxyAgent
   from autogen.code_utils import create_docker_executor

2. 准备代码执行配置
   code_exec_config = {
       "work_dir": "./workspace",
       "use_docker": True,
       "timeout": 60,
       "last_n_messages": 6,
   }

3. 创建UserProxyAgent
   code_agent = UserProxyAgent(
       name="code_executor",
       code_execution_config=code_exec_config,
       human_input_mode="NEVER",
   )

4. Agent自动处理代码执行
   - 当LLM生成代码时，Agent调用Code Executor
   - Executor在Docker容器中执行代码
   - 结果返回给Agent进行下一步处理

示例代码：
""")

    # 示例代码片段
    sample_code = '''
# 完整示例：使用代码执行器解决数学问题
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

# 创建助手代理（LLM驱动）
assistant = ConversableAgent(
    name="math_assistant",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]},
)

# 创建代码执行代理
code_executor = UserProxyAgent(
    name="code_executor",
    code_execution_config={
        "work_dir": "./workspace",
        "use_docker": True,
        "timeout": 60,
    },
    human_input_mode="NEVER",
)

# 启动对话
assistant.initiate_chat(
    code_executor,
    message="请计算前100个素数的和，然后输出结果",
)
'''
    print(sample_code)


# ============================================================
# 第八部分：故障排查指南
# ============================================================

def docker_troubleshooting_guide():
    """
    Docker相关问题的故障排查指南
    """
    print("\n" + "=" * 60)
    print("Docker 故障排查指南")
    print("=" * 60)
    print("""
常见问题及解决方案：

1. Docker守护进程未运行
   症状：docker命令返回 "Cannot connect to the Docker daemon"
   解决：
   - Linux/Mac: sudo systemctl start docker
   - Windows: 启动Docker Desktop应用

2. 权限被拒绝
   症状：Got permission denied while trying to connect
   解决：
   - Linux: sudo usermod -aG docker $USER
   - Mac/Windows: 确保Docker Desktop以当前用户运行

3. 镜像拉取失败
   症状：docker pull超时或失败
   解决：
   - 检查网络连接
   - 配置Docker镜像加速器
   - 手动拉取镜像

4. 容器启动失败
   症状：Code Executor报错 "Failed to create container"
   解决：
   - 检查Docker版本（需要17.0+）
   - 清理未使用的容器：docker container prune
   - 检查系统资源是否充足

5. 执行超时
   症状：代码执行总是超时
   解决：
   - 适当增加timeout值
   - 检查代码是否有死循环
   - 考虑优化代码或分段执行
    """)


# ============================================================
# 程序入口
# ============================================================

def main():
    """
    主函数：运行Docker沙箱配置示例
    """
    print("=" * 70)
    print("第6节 Code Executor配置与安全机制 - Docker隔离环境配置")
    print("=" * 70)

    # 1. 检查Docker环境
    check_results = check_docker_environment()

    # 2. 解释Docker容器配置
    explain_docker_container_config()

    # 3. 演示DockerExecutor配置
    demonstrate_docker_executor_config()

    # 4. 解释本地开发模式
    explain_local_dev_mode()

    # 5. 安全最佳实践
    security = DockerSecurityBestPractices()
    print("\n" + "=" * 60)
    print("Docker安全最佳实践")
    print("=" * 60)
    options = security.get_security_options()
    print("推荐的安全配置：")
    for key, value in options.items():
        print(f"  {key}: {value}")

    # 6. 代码安全性检查示例
    print("\n" + "=" * 60)
    print("代码安全性预检查示例")
    print("=" * 60)
    sample_code = "import os; os.remove('/etc/passwd')"
    safety_result = security.validate_code_safety(sample_code)
    print(f"待检查代码：{sample_code}")
    print(f"是否安全：{safety_result['is_safe']}")
    print(f"警告信息：{safety_result['warnings']}")

    # 7. 实际使用示例
    demonstrate_practical_usage()

    # 8. 故障排查指南
    docker_troubleshooting_guide()

    # 镜像管理示例
    manager = DockerImageManager()
    local_images = manager.list_local_images()
    print("\n" + "=" * 60)
    print("本地Docker镜像检查")
    print("=" * 60)
    if local_images:
        print(f"找到 {len(local_images)} 个相关镜像：")
        for img in local_images:
            print(f"  - {img}")
    else:
        print("未找到Python/AutoGen相关镜像")

    print("\n" + "=" * 70)
    print("Docker沙箱配置示例运行完成")
    print("=" * 70)
    print("""
学习要点总结：
1. 生产环境应优先使用use_docker=True
2. 根据场景选择合适的超时时间和镜像
3. 关注代码执行的安全性，检查危险操作
4. 熟悉Docker故障排查方法
    """)


if __name__ == "__main__":
    main()