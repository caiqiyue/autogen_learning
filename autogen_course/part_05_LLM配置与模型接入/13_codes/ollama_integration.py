"""
Ollama 本地模型接入 AutoGen 完整指南

本文件展示如何将 Ollama 本地部署的大语言模型接入 AutoGen 框架。
Ollama 是一款开源的本地模型运行工具，支持多种开源模型（如 Llama 2、Qwen、Mistral 等），
无需云端 API，适合隐私敏感或内网环境。

主要功能：
1. Ollama 服务启动与模型管理
2. AutoGen 通过 Ollama API 调用本地模型
3. 多模型 fallback 配置
4. 常见问题排查

安装 Ollama：https://ollama.com/download
"""

import os
from typing import Dict, List, Optional


# =============================================================================
# 第一部分：Ollama 服务配置基础
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 Ollama 服务地址配置
# -----------------------------------------------------------------------------

# Ollama 默认监听 localhost:11434
# 如果使用了自定义端口或远程服务器，需要相应调整

OLLAMA_BASE_URL = "http://localhost:11434"  # 默认 Ollama 服务地址
OLLAMA_MODEL = "qwen2.5:3b"                # 默认模型名称（示例：通义千问2.5 3B参数）

# 注意：Ollama 模型名称格式为 "模型名:版本标签"
# 常用模型名称示例：
#   - llama3:latest
#   - qwen2.5:3b
#   - mistral:latest
#   - codellama:7b
#   - deepseek-r1:1.5b


# -----------------------------------------------------------------------------
# 1.2 创建 Ollama 专用的 config_list
# -----------------------------------------------------------------------------

def create_ollama_config(
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    num_ctx: int = 4096,  # Ollama 特有的上下文窗口大小参数
    timeout: int = 120
) -> List[Dict]:
    """
    创建 Ollama 模型接入 AutoGen 的配置

    参数:
        model: Ollama 模型名称，格式为 "模型名:版本标签"
        base_url: Ollama 服务的地址，默认 http://localhost:11434
        temperature: 温度参数，控制输出随机性
        num_ctx: Ollama 特有参数，设置上下文窗口 token 数
        timeout: 请求超时时间（秒）

    返回:
        config_list: 符合 AutoGen 格式的配置列表
    """

    config_list = [
        {
            # ---------- 必需参数 ----------
            "model": model,                    # Ollama 模型名称
            "base_url": base_url + "/v1",     # Ollama 的 API 兼容 OpenAI 格式

            # ---------- API 配置 ----------
            # 注意：Ollama 不需要真实的 API key，但 AutoGen 要求 api_key 字段
            # 可以设置为任意非空字符串或 "ollama" 占位
            "api_key": "ollama",

            # ---------- 生成参数 ----------
            "temperature": temperature,       # 控制随机性
            "max_tokens": 2048,               # 最大生成 token 数

            # ---------- Ollama 特有参数 ----------
            # num_ctx 设置上下文窗口大小，根据模型和内存情况调整
            # 较大的值可以处理更长的对话，但会消耗更多内存
            "num_ctx": num_ctx,               # Ollama 特有参数

            # ---------- 价格配置 ----------
            # 本地模型成本为 0，用于成本监控
            "price": [0.0, 0.0],

            # ---------- 标签（用于过滤） ----------
            "tags": ["ollama", "local", "qwen"],
        }
    ]

    return config_list


# =============================================================================
# 第二部分：AutoGen Agent 集成 Ollama
# =============================================================================

def create_ollama_agent_config():
    """
    创建配置 Ollama 模型的 Agent 示例

    本节展示如何创建一个使用 Ollama 模型的 ConversableAgent
    """

    # -------------------------------------------------------------------------
    # 方式1：直接传入 config_list
    # -------------------------------------------------------------------------

    # from autogen import ConversableAgent

    ollama_config = create_ollama_config(model="qwen2.5:3b")

    # 创建使用 Ollama 的 AssistantAgent
    # ollama_agent = ConversableAgent(
    #     name="ollama_assistant",
    #     system_message="你是一个有帮助的AI助手，由 Ollama 本地模型驱动。",
    #     llm_config={
    #         "config_list": ollama_config,
    #         "temperature": 0.7,
    #     }
    # )

    # -------------------------------------------------------------------------
    # 方式2：使用 environment 变量配置
    # -------------------------------------------------------------------------

    # 如果 Ollama 在远程服务器或非默认端口，可以这样配置：
    remote_ollama_config = [
        {
            "model": "qwen2.5:3b",
            "base_url": "http://192.168.1.100:11434/v1",  # 远程 Ollama 服务
            "api_key": "ollama",
            "price": [0.0, 0.0],
        }
    ]

    return ollama_config, remote_ollama_config


# =============================================================================
# 第三部分：多模型 Fallback 配置
# =============================================================================

def create_ollama_fallback_config() -> List[Dict]:
    """
    创建多模型 Fallback 配置示例

    在生产环境中，建议配置多个模型作为 fallback：
    1. 首选：本地 Ollama 模型（成本低、隐私好）
    2. 备选：云端 API（质量高、可用性高）

    AutoGen 会按顺序尝试每个模型，失败时自动切换到下一个
    """

    config_list = [
        # ---------- 第一优先级：本地 Ollama ----------
        {
            "model": "qwen2.5:3b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "price": [0.0, 0.0],
            "tags": ["ollama", "local", "primary"],
        },

        # ---------- 第二优先级：本地 Ollama（备用模型） ----------
        {
            "model": "llama3:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "price": [0.0, 0.0],
            "tags": ["ollama", "local", "backup"],
        },

        # ---------- 第三优先级：云端 API（最后保障） ----------
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "price": [0.15, 0.60],  # $0.15/1M 输入, $0.60/1M 输出
            "tags": ["openai", "cloud", "fallback"],
        },
    ]

    return config_list


# =============================================================================
# 第四部分：Ollama 模型管理命令
# =============================================================================

def get_ollama_management_commands():
    """
    常用 Ollama 管理命令参考

    这些命令通过命令行执行，用于管理本地模型
    """

    commands = {
        # ---------- 模型操作 ----------
        "list_models": "ollama list",
        # 列出所有已下载的模型

        "pull_model": "ollama pull qwen2.5:3b",
        # 下载/更新模型（首次运行需要下载）

        "remove_model": "ollama rm qwen2.5:3b",
        # 删除已下载的模型

        "show_model": "ollama show qwen2.5:3b",
        # 显示模型详细信息

        # ---------- 服务操作 ----------
        "start_server": "ollama serve",
        # 启动 Ollama 服务（通常自动启动）

        "run_model": "ollama run qwen2.5:3b",
        # 直接运行模型（交互模式）

        # ---------- API 测试 ----------
        "api_test": """
# 测试 Ollama API 是否正常工作
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Hello, how are you?",
  "stream": false
}'
        """,
    }

    return commands


# =============================================================================
# 第五部分：完整集成示例
# =============================================================================

def run_ollama_integration_example():
    """
    完整的 Ollama + AutoGen 集成示例

    该示例展示如何：
    1. 配置 Ollama 连接参数
    2. 创建 Agent
    3. 进行对话测试
    """

    # -------------------------------------------------------------------------
    # 步骤1：准备配置
    # -------------------------------------------------------------------------
    config_list = create_ollama_fallback_config()

    # -------------------------------------------------------------------------
    # 步骤2：创建 Agent（实际运行时取消注释）
    # -------------------------------------------------------------------------

    # from autogen import ConversableAgent
    #
    # assistant = ConversableAgent(
    #     name="assistant",
    #     system_message="你是一个有帮助的AI助手。",
    #     llm_config={
    #         "config_list": config_list,
    #         "temperature": 0.7,
    #     }
    # )
    #
    # user_proxy = ConversableAgent(
    #     name="user_proxy",
    #     human_input_mode="NEVER",
    #     max_consecutive_auto_reply=5,
    # )
    #
    # # -------------------------------------------------------------------------
    # # 步骤3：发起对话
    # # -------------------------------------------------------------------------
    #
    # user_proxy.initiate_chat(
    #     assistant,
    #     message="请用一句话介绍一下自己。"
    # )

    # -------------------------------------------------------------------------
    # 返回配置用于验证
    # -------------------------------------------------------------------------
    return config_list


# =============================================================================
# 第六部分：常见问题排查
# =============================================================================

def troubleshoot_ollama_issues():
    """
    Ollama 接入 AutoGen 的常见问题及解决方案

    问题1：Connection Error - Ollama 服务未启动
    解决：
        - Windows: 在任务管理器或服务中启动 Ollama
        - macOS: 在菜单栏找到 Ollama 图标，点击 "Start Server"
        - 或命令行运行: ollama serve

    问题2：模型不存在 (model not found)
    解决：
        - 运行 ollama list 查看已安装模型
        - 如需安装，运行: ollama pull <模型名>
        - 确认模型名称格式正确（包含版本标签）

    问题3：上下文长度不足
    解决：
        - 在配置中增加 num_ctx 参数（如 num_ctx: 8192）
        - 或减少 max_tokens 限制

    问题4：响应速度慢
    解决：
        - 使用更小的模型（如 3B 参数而非 70B）
        - 减少 max_tokens 限制
        - 确保有足够的系统内存
    """

    issues = [
        {
            "problem": "Connection Error",
            "cause": "Ollama 服务未启动",
            "solution": "运行 'ollama serve' 启动服务"
        },
        {
            "problem": "Model not found",
            "cause": "模型未安装或名称错误",
            "solution": "运行 'ollama list' 检查已安装模型，或 'ollama pull <model>' 下载"
        },
        {
            "problem": "Context length exceeded",
            "cause": "对话长度超过模型上下文窗口",
            "solution": "增加 num_ctx 参数或减少 max_tokens"
        },
        {
            "problem": "Slow response",
            "cause": "模型太大或系统资源不足",
            "solution": "使用更小的模型，或确保有足够内存"
        },
    ]

    return issues


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Ollama 接入 AutoGen 配置示例")
    print("=" * 60)

    # 示例1：创建基础配置
    print("\n【示例1】创建 Ollama 基础配置")
    config = create_ollama_config()
    print(f"模型: {config[0]['model']}")
    print(f"地址: {config[0]['base_url']}")

    # 示例2：创建 Fallback 配置
    print("\n【示例2】多模型 Fallback 配置")
    fallback_config = create_ollama_fallback_config()
    print(f"共配置 {len(fallback_config)} 个模型:")
    for i, cfg in enumerate(fallback_config):
        print(f"  {i+1}. {cfg['model']} (tags: {cfg['tags']})")

    # 示例3：Ollama 管理命令
    print("\n【示例3】常用 Ollama 管理命令")
    commands = get_ollama_management_commands()
    for cmd, desc in commands.items():
        print(f"  {cmd}: {desc[:50]}...")

    # 示例4：问题排查
    print("\n【示例4】常见问题排查")
    issues = troubleshoot_ollama_issues()
    for issue in issues:
        print(f"  问题: {issue['problem']}")
        print(f"  解决: {issue['solution']}\n")