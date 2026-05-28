"""
LiteLLM 统一代理层接入 AutoGen 完整指南

LiteLLM 是一个开源的统一 API 抽象层，可以将 100+ 种大模型 API
（OpenAI、Anthropic、Azure、Hugging Face、Ollama、vLLM 等）
统一为 OpenAI 兼容格式。

主要优势：
1. 统一接口：无需关心底层模型差异
2. 多模型负载均衡：自动在多个模型间分配请求
3. 成本控制：支持预算和速率限制
4. 简化部署：企业内网只需暴露一个 LiteLLM 端点

本文件展示：
1. LiteLLM 服务部署配置
2. AutoGen 通过 LiteLLM 代理接入多种模型
3. 多模型负载均衡和 fallback 策略
"""

import os
from typing import Dict, List, Optional


# =============================================================================
# 第一部分：LiteLLM 服务配置
# =============================================================================

# LiteLLM 服务地址（默认配置）
LITELLM_PROXY_URL = "http://localhost:4000"  # LiteLLM 代理服务地址
LITELLM_MASTER_KEY = "sk-1234567890"          # LiteLLM 主密钥（生产环境应从环境变量读取）

# 注意：LiteLLM 支持两种模式：
# 1. 代理模式（Proxy Mode）：作为统一网关，需要配置 model_list
# 2. 直连模式（Direct Mode）：直接调用，不经过代理


# =============================================================================
# 第二部分：LiteLLM 代理配置示例
# =============================================================================

def create_litellm_proxy_config(
    proxy_url: str = "http://localhost:4000",
    master_key: Optional[str] = None
) -> List[Dict]:
    """
    创建通过 LiteLLM 代理接入 AutoGen 的配置

    参数:
        proxy_url: LiteLLM 代理服务地址
        master_key: LiteLLM 主密钥，None 则从环境变量 LITELLM_MASTER_KEY 读取

    返回:
        config_list: 符合 AutoGen 格式的配置列表
    """

    if master_key is None:
        master_key = os.getenv("LITELLM_MASTER_KEY", "mock-key")

    config_list = [
        {
            # ---------- 模型标识 ----------
            # LiteLLM 中模型名称格式：provider/model-name
            # 例如：openai/gpt-4o, anthropic/claude-3-opus,ollama/qwen2.5:3b
            "model": "openai/gpt-4o-mini",

            # ---------- 代理地址 ----------
            "base_url": proxy_url + "/v1",

            # ---------- 认证信息 ----------
            # LiteLLM 使用 master_key 作为统一认证
            "api_key": master_key,

            # ---------- 生成参数 ----------
            "temperature": 0.7,
            "max_tokens": 2048,

            # ---------- 成本配置 ----------
            "price": [0.15, 0.60],  # OpenAI GPT-4o-mini 价格

            # ---------- 标签 ----------
            "tags": ["litellm", "proxy", "openai"],
        }
    ]

    return config_list


# =============================================================================
# 第三部分：LiteLLM 多模型负载均衡配置
# =============================================================================

def create_litellm_load_balance_config() -> List[Dict]:
    """
    创建 LiteLLM 多模型负载均衡配置

    LiteLLM 支持两种多模型配置方式：
    1. 模型组（model_group）：多个同类型模型自动负载均衡
    2. fallback：按优先级依次尝试

    以下配置展示如何在 AutoGen 中配置 LiteLLM 的模型组
    """

    master_key = os.getenv("LITELLM_MASTER_KEY", "mock-key")
    proxy_url = "http://localhost:4000"

    # -------------------------------------------------------------------------
    # 方式1：LiteLLM 模型组（自动负载均衡）
    # -------------------------------------------------------------------------
    # 在 LiteLLM 配置文件中定义 model_group，AutoGen 只需指定组名

    config_with_model_group = [
        {
            # 使用模型组，LiteLLM 会自动在组内模型间分配请求
            # 模型组在 LiteLLM 代理配置文件中定义
            "model": "openai/gpt-4o-mini",  # 或者使用定义的模型组名如 "gpt-group"
            "base_url": proxy_url + "/v1",
            "api_key": master_key,
            "price": [0.15, 0.60],
            "tags": ["litellm", "model_group"],
        }
    ]

    # -------------------------------------------------------------------------
    # 方式2：AutoGen 级别的 fallback（与 Ollama 配合）
    # -------------------------------------------------------------------------
    # 如果需要在 AutoGen 层面控制 fallback，可以这样配置

    config_with_fallback = [
        # 第一选择：LiteLLM 代理的 GPT-4o-mini
        {
            "model": "openai/gpt-4o-mini",
            "base_url": proxy_url + "/v1",
            "api_key": master_key,
            "price": [0.15, 0.60],
            "tags": ["litellm", "cloud", "primary"],
        },
        # 第二选择：本地 Ollama
        {
            "model": "ollama/qwen2.5:3b",
            "base_url": proxy_url + "/v1",
            "api_key": master_key,
            "price": [0.0, 0.0],
            "tags": ["litellm", "ollama", "local", "fallback"],
        },
    ]

    return config_with_fallback


# =============================================================================
# 第四部分：LiteLLM 配置文件示例
# =============================================================================

def get_litellm_config_yaml() -> str:
    """
    返回 LiteLLM 代理配置文件示例（config.yaml 格式）

    这个配置文件定义了 LiteLLM 的模型列表、负载均衡、速率限制等
    将此内容保存为 config.yaml，然后运行 litellm --config config.yaml
    """

    config_yaml = """
# LiteLLM 代理配置文件示例
# 保存为 config.yaml，然后运行: litellm --config config.yaml

model_list:
  # ========== OpenAI 模型 ==========
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OpenAI_API_KEY
      rpm: 500  # 每分钟请求数限制

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OpenAI_API_KEY
      rpm: 200

  # ========== Ollama 本地模型 ==========
  - model_name: qwen-local
    litellm_params:
      model: ollama/qwen2.5:3b
      api_key: "ollama"
      base_url: http://localhost:11434

  # ========== Anthropic 模型 ==========
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  # ========== 模型组（负载均衡） ==========
  - model_name: gpt-group
    litellm_params:
      model: openai/gpt-4o-mini  # 指向组内的主模型
      rpm: 1000

# 负载均衡策略
router_settings:
  routing_strategy: least-cost  # 按成本选择 | latencies-optimized | simple-round-robin
  redis_host: localhost
  redis_port: 6379

# 价格配置（用于成本优化）
module_settings:
  flatten_user_models: true
  user_api_key_aliases:
    sk-hello: sk-1234567890

# 速率限制
general_settings:
  master_key: sk-1234567890
  database_type: dynamodb
"""

    return config_yaml


# =============================================================================
# 第五部分：AutoGen + LiteLLM 完整集成示例
# =============================================================================

def create_litellm_complete_config() -> List[Dict]:
    """
    创建完整的 LiteLLM 集成配置

    适用于企业内网场景：
    - 所有模型通过统一的 LiteLLM 代理暴露
    - AutoGen 只需连接一个端点
    - 支持多模型 fallback 和负载均衡
    """

    master_key = os.getenv("LITELLM_MASTER_KEY", "mock-key")
    proxy_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")

    config_list = [
        # ---------- 云端优先 ----------
        {
            "model": "openai/gpt-4o-mini",
            "base_url": proxy_url + "/v1",
            "api_key": master_key,
            "price": [0.15, 0.60],
            "tags": ["cloud", "primary", "fast"],
        },
        # ---------- 本地备用 ----------
        {
            "model": "ollama/qwen2.5:3b",
            "base_url": proxy_url + "/v1",
            "api_key": master_key,
            "price": [0.0, 0.0],
            "num_ctx": 8192,  # Ollama 特有参数
            "tags": ["local", "fallback", "privacy"],
        },
        # ---------- Claude 备用 ----------
        {
            "model": "anthropic/claude-3-opus",
            "base_url": proxy_url + "/v1",
            "api_key": master_key,
            "price": [15.0, 75.0],  # Claude 较贵
            "tags": ["cloud", "fallback", "quality"],
        },
    ]

    return config_list


# =============================================================================
# 第六部分：vLLM 接入 LiteLLM 配置
# =============================================================================

def get_vllm_in_litellm_config() -> Dict:
    """
    展示 vLLM 如何通过 LiteLLM 接入

    vLLM 是一个高性能的推理服务，特别适合本地部署
    接入方式：在 LiteLLM 配置文件中添加 vLLM 模型
    """

    vllm_config = {
        # vLLM 模型配置（添加到 LiteLLM config.yaml）
        "model_name": "vllm-llama3",
        "litellm_params": {
            "model": "openai/llama3",  # 使用 OpenAI 兼容格式
            "api_key": "unused",
            "base_url": "http://localhost:8000/v1",  # vLLM 服务地址
        }
    }

    return vllm_config


# =============================================================================
# 第七部分：LiteLLM vs Ollama vs vLLM 对比
# =============================================================================

def compare_local_model_solutions():
    """
    三种本地模型接入方案对比

    适用场景：
    - Ollama：个人开发、快速原型、简单部署
    - vLLM：高性能生产环境、需要批量推理
    - LiteLLM：企业内网、需要统一管理多模型
    """

    comparison = {
        "Ollama": {
            "部署难度": "★☆☆☆☆（极简）",
            "性能": "★★☆☆☆",
            "适用场景": "个人开发、简单部署、快速原型",
            "优点": "安装简单、一行命令运行、模型管理方便",
            "缺点": "性能一般、不支持多模型统一管理",
            "AutoGen接入": "直接通过 base_url=http://localhost:11434/v1",
        },
        "vLLM": {
            "部署难度": "★★★★☆（复杂）",
            "性能": "★★★★★",
            "适用场景": "生产环境、高吞吐、低延迟需求",
            "优点": "高性能、支持PagedAttention、吞吐量高",
            "缺点": "部署复杂、需要GPU支持、模型支持有限",
            "AutoGen接入": "通过 LiteLLM 代理或直接 base_url=http://localhost:8000/v1",
        },
        "LiteLLM": {
            "部署难度": "★★★☆☆（中等）",
            "性能": "★★★☆☆",
            "适用场景": "企业内网、多模型统一管理、负载均衡",
            "优点": "统一接口、100+模型支持、负载均衡、成本控制",
            "缺点": "额外一层服务、需要单独部署和维护",
            "AutoGen接入": "通过 LiteLLM 代理 base_url=http://localhost:4000/v1",
        }
    }

    return comparison


# =============================================================================
# 第八部分：企业内网架构建议
# =============================================================================

def get_enterprise_architecture_guide():
    """
    企业内网模型部署架构建议

    推荐架构：
    1. 边缘层：LiteLLM 统一代理（暴露单一端点）
    2. 模型层：按需求组合 Ollama/vLLM/云端API
    3. 安全层：认证、限流、审计日志
    4. AutoGen 层：直接连接 LiteLLM 代理
    """

    architecture = {
        "component_layers": [
            {
                "layer": "AutoGen Agent",
                "description": "业务逻辑层，使用单一端点调用模型",
                "connection": "http://litellm-proxy:4000"
            },
            {
                "layer": "LiteLLM Proxy",
                "description": "统一网关，处理认证、限流、负载均衡",
                "connection": "litellm:4000"
            },
            {
                "layer": "Model Providers",
                "description": "实际模型服务：Ollama/vLLM/云端API",
                "connection": "localhost:11434, localhost:8000, api.openai.com"
            }
        ],
        "security": [
            "使用 master_key 认证",
            "配置 RPM（每分钟请求数）限制",
            "启用详细审计日志",
            "配置 CORS 限制"
        ],
        "high_availability": [
            "LiteLLM 支持 Redis 连接，实现会话粘性",
            "多模型 fallback 配置确保服务连续性",
            "定期健康检查自动移除故障节点"
        ]
    }

    return architecture


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LiteLLM 统一代理层接入 AutoGen 配置示例")
    print("=" * 70)

    # 示例1：基础代理配置
    print("\n【示例1】LiteLLM 代理配置")
    config = create_litellm_proxy_config()
    print(f"代理地址: {config[0]['base_url']}")
    print(f"模型: {config[0]['model']}")

    # 示例2：多模型配置
    print("\n【示例2】多模型 Fallback 配置")
    complete_config = create_litellm_complete_config()
    print(f"共配置 {len(complete_config)} 个模型:")
    for cfg in complete_config:
        print(f"  - {cfg['model']} (tags: {cfg['tags']})")

    # 示例3：方案对比
    print("\n【示例3】本地模型接入方案对比")
    comparison = compare_local_model_solutions()
    for solution, details in comparison.items():
        print(f"\n  {solution}:")
        print(f"    部署难度: {details['部署难度']}")
        print(f"    性能: {details['性能']}")
        print(f"    AutoGen接入: {details['AutoGen接入']}")

    # 示例4：企业架构
    print("\n【示例4】企业内网架构建议")
    arch = get_enterprise_architecture_guide()
    print("  组件层级:")
    for item in arch['component_layers']:
        print(f"    {item['layer']} -> {item['connection']}")