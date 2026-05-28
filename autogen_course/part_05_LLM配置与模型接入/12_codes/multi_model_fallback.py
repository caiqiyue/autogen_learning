"""
AutoGen 多模型Fallback机制 - 展示多模型fallback配置与重试逻辑

本文件展示如何配置多模型fallback机制，包括：
1. config_list优先级配置
2. 多模型fallback链配置
3. tags过滤策略与filter_config用法
4. 基于price的成本控制策略
"""

from typing import Dict, List, Optional, Callable

# =============================================================================
# 第一部分：config_list 优先级与基本Fallback配置
# =============================================================================

# AutoGen的config_list按顺序优先级排列
# 当前一个模型失败时，会自动尝试列表中的下一个模型

# -----------------------------------------------------------------------------
# 示例1：基本的多模型Fallback配置
# -----------------------------------------------------------------------------

# 定义一个三层的fallback链：优先GPT-4o，失败后尝试GPT-4o-mini，最后尝试GPT-3.5
config_list_fallback_basic = [
    # ---------- 第一优先级：主力模型 ----------
    {
        "model": "gpt-4o",
        "api_key": "sk-primary-key",
        "price": [5.0, 15.0],          # 高成本高性能
        "tags": ["primary", "high-quality"],
        "max_retries": 2,              # 允许重试2次
    },
    # ---------- 第二优先级：备用模型 ----------
    {
        "model": "gpt-4o-mini",
        "api_key": "sk-backup-key",
        "price": [0.15, 0.6],          # 低成本高效率
        "tags": ["backup", "fast"],
        "max_retries": 2,
    },
    # ---------- 第三优先级：降级模型 ----------
    {
        "model": "gpt-3.5-turbo",
        "api_key": "sk-fallback-key",
        "price": [0.5, 1.5],           # 中等成本
        "tags": ["fallback", "budget"],
        "max_retries": 3,              # 降级模型允许更多重试
    }
]


# =============================================================================
# 第二部分：基于Tags的智能过滤策略
# =============================================================================

# Tags是AutoGen中重要的分组和过滤机制
# 可以用tags来标记模型的特性（速度、成本、质量等）

# -----------------------------------------------------------------------------
# 示例2：使用filter_config进行Tags过滤
# -----------------------------------------------------------------------------

def create_tagged_config_list() -> List[Dict]:
    """
    创建带tags的模型配置列表

    常用的Tags策略：
    - primary/backup/fallback: 表示优先级
    - fast/slow: 表示速度特性
    - cheap/expensive: 表示成本
    - high-quality/basic: 表示质量等级
    - openai/azure/claude: 表示供应商
    """

    config_list_with_tags = [
        # GPT-4系列 - 高质量
        {
            "model": "gpt-4o",
            "api_key": "sk-openai-primary",
            "tags": ["openai", "primary", "high-quality", "fast"],
            "price": [5.0, 15.0],
        },
        {
            "model": "gpt-4o-mini",
            "api_key": "sk-openai-mini",
            "tags": ["openai", "fast", "cheap"],
            "price": [0.15, 0.6],
        },
        # Claude系列 - 另一供应商
        {
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-anthropic-key",
            "tags": ["anthropic", "primary", "high-quality"],
            "price": [3.0, 15.0],
        },
        {
            "model": "claude-opus-4-20250514",
            "api_key": "sk-anthropic-key",
            "tags": ["anthropic", "premium", "high-quality"],
            "price": [15.0, 75.0],
        },
        # 本地模型 - 完全免费，可作为最后防线
        {
            "model": "ollama/llama3",
            "api_key": "not-needed",     # 本地模型不需要API key
            "base_url": "http://localhost:11434/v1",
            "tags": ["local", "free", "fallback"],
            "price": [0.0, 0.0],         # 免费使用
        }
    ]

    return config_list_with_tags


def filter_by_tags(config_list: List[Dict], required_tags: List[str]) -> List[Dict]:
    """
    根据Tags过滤config_list

    参数:
        config_list: 原始配置列表
        required_tags: 必须包含的标签列表（AND逻辑）

    返回:
        过滤后的配置列表
    """

    filtered = []
    for config in config_list:
        config_tags = config.get("tags", [])
        # 检查所有required_tags是否都在config_tags中
        if all(tag in config_tags for tag in required_tags):
            filtered.append(config)

    return filtered


def filter_by_any_tag(config_list: List[Dict], tags: List[str]) -> List[Dict]:
    """
    根据Tags过滤config_list（OR逻辑）

    参数:
        config_list: 原始配置列表
        tags: 任意匹配一个即可的标签列表

    返回:
        过滤后的配置列表
    """

    filtered = []
    for config in config_list:
        config_tags = config.get("tags", [])
        # 检查是否有任意一个tag匹配
        if any(tag in config_tags for tag in tags):
            filtered.append(config)

    return filtered


# =============================================================================
# 第三部分：filter_config 高级过滤
# =============================================================================

# filter_config是一个更强大的过滤机制，支持自定义过滤函数

def create_filter_config():
    """
    演示filter_config的用法

    filter_config可以是:
    1. 一个字典: {"tags": ["primary"]} - 基于已有字段过滤
    2. 一个函数: callable - 自定义过滤逻辑
    """

    # 方式1：使用字典进行简单过滤
    filter_config_dict = {
        "tags": ["primary"],  # 只使用包含"primary"标签的模型
    }

    # 方式2：使用自定义过滤函数
    def custom_filter(config: Dict) -> bool:
        """
        自定义过滤函数

        这个函数定义了什么条件下模型会被使用
        """
        # 价格过滤：拒绝过于昂贵的模型（输入价格>$10/1M tokens）
        if config.get("price") and config["price"][0] > 10.0:
            return False

        # 供应商偏好：优先使用OpenAI模型
        tags = config.get("tags", [])
        if "openai" in tags:
            return True  # OpenAI模型直接通过

        # 非OpenAI模型只在使用"allow_other"标签时才通过
        return "allow_other" in tags

    filter_config_function = custom_filter

    return filter_config_dict, filter_config_function


# =============================================================================
# 第四部分：基于价格的自适应Fallback策略
# =============================================================================

def create_price_aware_fallback():
    """
    创建基于价格的自适应Fallback策略

    策略思路：
    1. 优先使用最便宜的模型处理简单请求
    2. 只有在复杂请求或失败时才升级到更贵的模型
    3. 通过设置不同的price阈值来实现成本控制
    """

    # 定义一个智能的模型列表，按价格从低到高排序
    # 但优先级不是单纯依赖顺序，而是根据任务复杂度动态选择

    price_aware_config = [
        # Level 1: 最便宜 - 简单任务
        {
            "model": "gpt-4o-mini",
            "api_key": "sk-mini-key",
            "tags": ["level-1", "budget", "simple-tasks"],
            "price": [0.15, 0.6],
            "max_tokens": 1024,           # 限制输出长度控制成本
        },
        # Level 2: 中等成本 - 一般任务
        {
            "model": "gpt-4o-mini",
            "api_key": "sk-mini-key",
            "tags": ["level-2", "standard"],
            "price": [0.15, 0.6],
            "max_tokens": 4096,
        },
        # Level 3: 高性能 - 复杂任务
        {
            "model": "gpt-4o",
            "api_key": "sk-gpt4o-key",
            "tags": ["level-3", "high-quality", "complex-tasks"],
            "price": [5.0, 15.0],
            "max_tokens": 8192,
        },
        # Level 4: 高级模型 - 关键任务
        {
            "model": "gpt-4-turbo",
            "api_key": "sk-gpt4-key",
            "tags": ["level-4", "premium", "critical"],
            "price": [10.0, 30.0],
            "max_tokens": 16384,
        },
    ]

    return price_aware_config


def calculate_cost_savings(primary_config: Dict,
                          fallback_config: Dict,
                          request_count: int = 1000) -> Dict:
    """
    计算使用Fallback策略的潜在成本节省

    参数:
        primary_config: 主要模型配置
        fallback_config: 备用模型配置
        request_count: 请求数量

    返回:
        成本分析字典
    """

    # 假设60%的请求会成功使用主模型，40%会fallback
    primary_rate = 0.6
    fallback_rate = 0.4

    # 计算平均每个请求的成本
    primary_cost_per_1k = primary_config["price"][0]  # 输入价格/1M
    fallback_cost_per_1k = fallback_config["price"][0]

    # 如果只有主模型
    only_primary_cost = (request_count / 1_000_000) * primary_cost_per_1k * 1000

    # 使用Fallback策略（假设主模型处理60%请求，备用处理40%）
    with_fallback_cost = (
        (request_count * primary_rate / 1_000_000) * primary_cost_per_1k * 1000 +
        (request_count * fallback_rate / 1_000_000) * fallback_cost_per_1k * 1000
    )

    savings = only_primary_cost - with_fallback_cost
    savings_percent = (savings / only_primary_cost) * 100 if only_primary_cost > 0 else 0

    return {
        "only_primary_cost": round(only_primary_cost, 2),
        "with_fallback_cost": round(with_fallback_cost, 2),
        "savings": round(savings, 2),
        "savings_percent": round(savings_percent, 1)
    }


# =============================================================================
# 第五部分：自定义重试策略与错误处理
# =============================================================================

def create_retry_strategy():
    """
    创建智能重试策略配置

    AutoGen支持针对不同错误类型配置不同的重试策略
    """

    retry_config = {
        # 速率限制（Rate Limit）重试策略
        "rate_limit_retry": {
            "max_attempts": 5,              # 最多重试5次
            "initial_delay": 1,             # 初始延迟1秒
            "backoff_factor": 2,            # 指数退避：1s -> 2s -> 4s -> 8s -> 16s
            "max_delay": 60,                # 最大延迟60秒
        },

        # 超时重试策略
        "timeout_retry": {
            "max_attempts": 3,
            "initial_delay": 0.5,
            "backoff_factor": 1.5,
            "max_delay": 10,
        },

        # 服务器错误重试策略（5xx错误）
        "server_error_retry": {
            "max_attempts": 4,
            "initial_delay": 2,
            "backoff_factor": 2,
            "max_delay": 120,               # 服务器问题可能需要更长的等待
        },

        # 配额错误策略（配额用尽）
        "quota_exceeded": {
            "fail_fast": True,              # 配额问题应该快速失败并切换模型
            "notify": True,                 # 通知用户配额问题
        }
    }

    return retry_config


class FallbackErrorHandler:
    """
    自定义Fallback错误处理器

    用于实现更精细的错误处理和降级逻辑
    """

    def __init__(self):
        self.error_counts = {}          # 记录每个模型的错误次数
        self.last_error_times = {}      # 记录上次错误时间
        self.success_counts = {}        # 记录成功次数

    def record_success(self, model: str):
        """记录成功调用"""
        self.success_counts[model] = self.success_counts.get(model, 0) + 1
        # 成功后可以清零错误计数（可选）
        # self.error_counts[model] = 0

    def record_error(self, model: str, error_type: str):
        """记录错误"""
        self.error_counts[model] = self.error_counts.get(model, 0) + 1
        self.last_error_times[model] = error_type

    def should_disable_model(self, model: str, threshold: int = 5) -> bool:
        """
        判断是否应该禁用某个模型

        参数:
            model: 模型名称
            threshold: 错误次数阈值，超过则禁用

        返回:
            True表示模型应该被禁用
        """
        return self.error_counts.get(model, 0) >= threshold

    def get_best_available_model(self, config_list: List[Dict]) -> Optional[Dict]:
        """
        获取当前可用的最佳模型

        会排除错误次数过多的模型
        """
        for config in config_list:
            model = config["model"]
            if not self.should_disable_model(model):
                return config
        return None


# =============================================================================
# 第六部分：完整的Fallback配置示例
# =============================================================================

def create_production_fallback_config() -> Dict:
    """
    创建一个生产环境级别的Fallback配置示例

    这个配置展示了如何构建一个健壮的多模型Fallback系统
    """

    # 1. 主体配置列表
    config_list = [
        # OpenAI主力模型
        {
            "model": "gpt-4o",
            "api_key": "sk-primary-openai",
            "tags": ["openai", "tier-1", "primary"],
            "price": [5.0, 15.0],
            "max_retries": 2,
            "timeout": 60,
        },
        # OpenAI Mini - 快速备用
        {
            "model": "gpt-4o-mini",
            "api_key": "sk-mini-openai",
            "tags": ["openai", "tier-2", "fast"],
            "price": [0.15, 0.6],
            "max_retries": 3,
            "timeout": 30,
        },
        # Anthropic Claude - 高质量选择
        {
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-anthropic",
            "tags": ["anthropic", "tier-1", "high-quality"],
            "price": [3.0, 15.0],
            "max_retries": 2,
            "timeout": 60,
        },
        # 本地Ollama - 免费最后防线
        {
            "model": "ollama/llama3",
            "api_key": "not-needed",
            "base_url": "http://localhost:11434/v1",
            "tags": ["local", "tier-3", "free", "emergency"],
            "price": [0.0, 0.0],
            "max_retries": 1,
            "timeout": 120,               # 本地模型可能需要更长时间
        }
    ]

    # 2. 过滤配置 - 只使用primary和fast标签的模型
    filter_config = {
        "tags": ["tier-1", "tier-2"],  # 默认只使用前两层
    }

    # 3. 错误处理器
    error_handler = FallbackErrorHandler()

    # 4. 成本控制配置
    cost_control = {
        "max_cost_per_request": 0.50,     # 单个请求最大成本$0.50
        "monthly_budget": 100.0,          # 月度预算$100
        "enable_cost_alert": True,        # 启用成本警报
        "alert_threshold": 0.80,          # 80%预算时警报
    }

    return {
        "config_list": config_list,
        "filter_config": filter_config,
        "error_handler": error_handler,
        "cost_control": cost_control,
    }


# =============================================================================
# 第七部分：Ollama本地模型配置
# =============================================================================

def create_ollama_config():
    """
    创建Ollama本地模型配置

    Ollama允许在本地运行开源LLM模型，如Llama3、Qwen等
    适合离线环境或不想消耗云端配额时使用
    """

    ollama_config = {
        "model": "ollama/llama3",              # Ollama模型标识符
        "api_key": "not-needed",               # 本地模型不需要API key
        "base_url": "http://localhost:11434/v1",  # Ollama默认端口
        "tags": ["local", "free", "ollama"],
        "price": [0.0, 0.0],                   # 完全免费
        "timeout": 180,                        # 本地模型推理可能较慢
        "max_retries": 1,                      # 本地模型重试意义不大
    }

    return ollama_config


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("多模型Fallback机制演示")
    print("=" * 70)

    # 示例1：基于Tags过滤
    print("\n【示例1】Tags过滤演示")
    all_configs = create_tagged_config_list()
    print(f"全部配置数量: {len(all_configs)}")

    primary_only = filter_by_tags(all_configs, ["primary"])
    print(f"仅primary标签: {len(primary_only)} 个")
    for cfg in primary_only:
        print(f"  - {cfg['model']}: {cfg['tags']}")

    openai_or_fast = filter_by_any_tag(all_configs, ["openai", "fast"])
    print(f"openai或fast标签: {len(openai_or_fast)} 个")
    for cfg in openai_or_fast:
        print(f"  - {cfg['model']}: {cfg['tags']}")

    # 示例2：成本节省计算
    print("\n【示例2】成本节省分析")
    gpt4o_config = {"price": [5.0, 15.0]}
    gpt35_config = {"price": [0.5, 1.5]}
    savings = calculate_cost_savings(gpt4o_config, gpt35_config, request_count=10000)
    print(f"仅使用GPT-4o成本: ${savings['only_primary_cost']}")
    print(f"使用Fallback策略成本: ${savings['with_fallback_cost']}")
    print(f"节省: ${savings['savings']} ({savings['savings_percent']}%)")

    # 示例3：生产环境配置
    print("\n【示例3】生产环境Fallback配置")
    prod_config = create_production_fallback_config()
    print(f"配置层级数量: {len(prod_config['config_list'])}")
    for i, cfg in enumerate(prod_config['config_list']):
        print(f"  Tier {i+1}: {cfg['model']}")
        print(f"    价格: ${cfg['price'][0]}/1M输入")
        print(f"    标签: {cfg['tags']}")

    # 示例4：Ollama本地配置
    print("\n【示例4】Ollama本地模型配置")
    ollama = create_ollama_config()
    print(f"模型: {ollama['model']}")
    print(f"端点: {ollama['base_url']}")
    print(f"用途: 本地离线环境或免费使用")