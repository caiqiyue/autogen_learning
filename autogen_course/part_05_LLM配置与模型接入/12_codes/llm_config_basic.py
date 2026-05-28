"""
AutoGen LLM配置基础 - 演示llm_config的完整结构与基本配置方法

本文件展示如何配置AutoGen的LLM参数，包括：
1. 基础的config_list配置
2. llm_config的完整参数结构
3. 常用模型供应商的配置示例
"""

from typing import Dict, List, Optional

# =============================================================================
# 第一部分：基础 config_list 配置
# =============================================================================

# config_list是AutoGen中最常用的配置格式，用于指定可用的模型列表
# 它是一个列表，每个元素包含一个模型的完整配置信息

# -----------------------------------------------------------------------------
# 示例1：最简单的 config_list 配置（单模型）
# -----------------------------------------------------------------------------
config_list_simple = [
    {
        "model": "gpt-4o",           # 模型名称
        "api_key": "your-api-key",   # API密钥（生产环境应使用环境变量）
    }
]

# -----------------------------------------------------------------------------
# 示例2：包含完整参数的 config_list 配置
# -----------------------------------------------------------------------------
config_list_full = [
    {
        # ---------- 必需参数 ----------
        "model": "gpt-4o",                    # 模型标识符

        # ---------- API配置 ----------
        "api_key": "your-api-key",             # API密钥
        "base_url": "https://api.openai.com/v1",  # API端点（可选，默认为OpenAI官方）

        # ---------- 模型行为参数 ----------
        "temperature": 0.7,                    # 温度参数：控制输出随机性
                                               # 0.0 = 确定性输出，2.0 = 高随机性
        "max_tokens": 4096,                     # 最大生成token数
        "top_p": 1.0,                          # 核采样参数：控制候选token的多样性
        "frequency_penalty": 0.0,               # 频率惩罚：减少重复已使用的词
        "presence_penalty": 0.0,               # 存在惩罚：鼓励生成新话题

        # ---------- 请求控制 ----------
        "timeout": 120,                        # 请求超时时间（秒）
        "max_retries": 3,                      # 最大重试次数

        # ---------- 扩展参数 ----------
        "tags": ["primary", "fast"],           # 标签：用于过滤和分组
        "price": [0.005, 0.015],              # 价格：[输入价格/1M tokens, 输出价格/1M tokens]
    }
]

# =============================================================================
# 第二部分：llm_config 完整结构
# =============================================================================

# llm_config是ConversableAgent的LLM配置参数，包含更丰富的控制选项

def create_llm_config():
    """
    创建完整的llm_config配置示例

    llm_config支持的完整参数：
    - model: 模型名称（必需）
    - api_key/api_type/base_url: API配置
    - temperature/max_tokens/top_p: 生成控制
    - timeout/max_retries: 请求控制
    - cache_seed: 缓存种子（用于缓存实验的可重复性）
    - fail_safe: 失败安全回调函数
    - functions: 可用函数列表（工具调用）
    """

    llm_config: Dict = {
        # ---------- 核心模型配置 ----------
        "model": "gpt-4o",

        # ---------- API配置 ----------
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1",

        # ---------- 生成参数 ----------
        # temperature控制输出的随机性
        # 0.0 = 几乎确定性输出，适合代码生成、精确问答
        # 0.7 = 中等随机性，适合创意写作
        # 1.0+ = 高随机性，适合头脑风暴
        "temperature": 0.7,

        # max_tokens限制单次回复的最大长度
        # 设置过低会截断回复，设置过高可能浪费配额
        "max_tokens": 2048,

        # top_p是核采样参数
        # 1.0 = 考虑所有token；0.9 = 只考虑top 90%累积概率的token
        "top_p": 1.0,

        # ---------- 重试与超时 ----------
        "timeout": 120,           # 单次请求超时（秒）
        "max_retries": 3,         # 失败时最大重试次数

        # ---------- 缓存配置 ----------
        # cache_seed用于缓存实验的 Reproducibility
        # 相同seed + 相同输入 = 相同输出
        "cache_seed": 42,

        # ---------- 失败安全机制 ----------
        # 当所有模型都失败时的回调函数
        # 可以在这里实现优雅降级逻辑
        "fail_safe": lambda: "抱歉，当前服务暂时不可用，请稍后重试。",

        # ---------- 函数调用配置 ----------
        # 定义可用的工具函数，AutoGen会自动处理函数选择和调用
        "functions": [
            {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如：北京、上海"
                        }
                    },
                    "required": ["city"]
                }
            }
        ]
    }

    return llm_config


# =============================================================================
# 第三部分：多供应商配置示例
# =============================================================================

def create_multi_provider_config():
    """
    演示如何配置来自不同供应商的多个模型

    AutoGen支持混合使用多个供应商的模型，
    可以实现负载均衡、成本优化等功能
    """

    # -----------------------------------------------------------------------------
    # 示例：混合使用 OpenAI 和 Azure OpenAI
    # -----------------------------------------------------------------------------
    config_list_multi_provider = [
        {
            # ---------- OpenAI 模型 ----------
            "model": "gpt-4o",
            "api_key": "sk-openai-xxxx",
            "base_url": "https://api.openai.com/v1",
            "price": [5.0, 15.0],         # $5/1M输入，$15/1M输出
            "tags": ["openai", "primary"],  # 标签用于后续过滤
        },
        {
            # ---------- Azure OpenAI 模型 ----------
            "model": "gpt-4o-azure",          # Azure模型名称可能不同
            "api_key": "your-azure-key",
            "base_url": "https://xxx.openai.azure.com",  # Azure特定端点
            "api_type": "azure",              # 指定供应商类型
            "api_version": "2024-02-01",      # Azure API版本
            "price": [3.0, 10.0],             # Azure通常更便宜
            "tags": ["azure", "backup"],      # 标记为备用模型
        }
    ]

    return config_list_multi_provider


# =============================================================================
# 第四部分：环境变量安全存储
# =============================================================================

# 重要：生产环境中，API密钥不应硬编码在代码中
# 应使用环境变量或安全的密钥管理服务

def create_config_with_env():
    """
    演示如何使用环境变量存储敏感信息
    """

    import os

    # 方式1：直接使用 os.getenv
    config_list = [
        {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),  # 从环境变量读取
        }
    ]

    # 方式2：使用 pyautoGen 的 config_util 加载配置
    # from autogen.agentchat.contrib.config_util import select_config

    # 方式3：使用 .env 文件 + python-dotenv
    # from dotenv import load_dotenv
    # load_dotenv()

    return config_list


# =============================================================================
# 第五部分：快速创建默认配置的工具函数
# =============================================================================

def quick_create_config(model: str = "gpt-4o",
                       api_key: Optional[str] = None,
                       temperature: float = 0.7) -> List[Dict]:
    """
    快速创建基础配置的辅助函数

    参数:
        model: 模型名称，默认为 gpt-4o
        api_key: API密钥，None则从环境变量 OPENAI_API_KEY 读取
        temperature: 温度参数

    返回:
        config_list: 标准格式的配置列表
    """

    import os

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "placeholder-key")

    return [
        {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
        }
    ]


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 示例1：使用完整配置
    print("=" * 60)
    print("示例1：llm_config 完整配置")
    print("=" * 60)
    config = create_llm_config()
    print(f"模型: {config['model']}")
    print(f"温度: {config['temperature']}")
    print(f"超时: {config['timeout']}秒")

    # 示例2：多供应商配置
    print("\n" + "=" * 60)
    print("示例2：多供应商配置")
    print("=" * 60)
    multi_config = create_multi_provider_config()
    for i, cfg in enumerate(multi_config):
        print(f"模型 {i+1}: {cfg['model']}")
        print(f"  供应商: {cfg.get('api_type', 'openai')}")
        print(f"  标签: {cfg['tags']}")

    # 示例3：快速创建配置
    print("\n" + "=" * 60)
    print("示例3：快速创建配置")
    print("=" * 60)
    quick_config = quick_create_config("gpt-4o-mini", temperature=0.5)
    print(f"快速配置: {quick_config}")