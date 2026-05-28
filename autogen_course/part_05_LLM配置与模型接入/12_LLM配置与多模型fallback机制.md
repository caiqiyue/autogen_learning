---
lesson_id: lesson_12
title: LLM配置与多模型fallback机制
module: LLM配置与模型接入
---

# 第12节 LLM配置与多模型Fallback机制

## 学习目标

1. 掌握llm_config的完整结构与配置项
2. 理解多模型fallback机制的工作原理
3. 掌握重试逻辑的配置方法

---

## 12.1 LLM配置概述

### 12.1.1 什么是llm_config

llm_config是AutoGen中用于配置大语言模型（LLM）的核心参数集合。通过llm_config，可以指定使用哪个模型、API密钥、请求参数等关键信息。

### 12.1.2 llm_config的完整结构

```python
llm_config = {
    # 模型配置
    "model": "gpt-4o-mini",

    # API配置
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL"),

    # 请求配置
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 1.0,

    # 超时配置
    "timeout": 120,
    "max_retries": 3,

    # 其他配置
    "cache_seed": None,
    "tags": ["production"],
}
```

---

## 12.2 基础LLM配置

### 12.2.1 环境变量配置

在使用llm_config之前，推荐通过`.env`文件管理API密钥：

```env
# .env文件配置示例
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 12.2.2 基础配置示例

以下示例展示如何创建基础LLM配置：

```python
import os
from dotenv import load_dotenv
from autogen import ConversableAgent

# 加载环境变量
load_dotenv()

# 基础llm_config配置
llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,
}

# 创建代理
agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
)
```

---

## 12.3 多模型Fallback机制

### 12.3.1 为什么需要Fallback

在生产环境中，LLM服务可能出现以下情况：
- API配额用尽
- 服务暂时不可用
- 响应超时

Fallback机制允许配置多个模型，当主模型不可用时自动切换到备用模型。

### 12.3.2 配置多模型Fallback

```python
# 多模型fallback配置
llm_config = {
    "model": "gpt-4o",  # 主模型
    "api_key": os.getenv("OPENAI_API_KEY"),

    # fallback模型列表
    "fallback_models": [
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
        {"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")},
    ],

    # 重试配置
    "max_retries": 3,
    "retry_delay": 2,  # 秒
}
```

### 12.3.3 Fallback工作流程

```
请求 → 主模型(gpt-4o)
         │
         ├─ 成功 → 返回结果
         │
         └─ 失败 → 重试(max_retries次)
                     │
                     ├─ 仍失败 → 切换到fallback模型
                     │           │
                     │           ├─ 成功 → 返回结果
                     │           │
                     │           └─ 失败 → 继续下一个fallback
                     │
                     └─ 成功 → 返回结果
```

---

## 12.4 重试逻辑配置

### 12.4.1 自动重试机制

AutoGen内置自动重试机制，可以通过以下参数配置：

```python
llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),

    # 重试配置
    "max_retries": 3,
    "retry_delay": 1,  # 初始重试延迟（秒）
    "exponential_backoff": True,  # 指数退避
    "max_retry_delay": 60,  # 最大重试延迟（秒）
}
```

### 12.4.2 指数退避策略

启用指数退避后，重试间隔会以指数方式增长：

| 重试次数 | 延迟计算 | 总延迟 |
|---------|---------|-------|
| 第1次 | 1s | 1s |
| 第2次 | 2s | 3s |
| 第3次 | 4s | 7s |
| 第4次 | 8s | 15s |

### 12.4.3 自定义重试策略

```python
from autogen import ConversableAgent
from autogen.retriever import RetryConfig

# 自定义重试配置
retry_config = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=30,
    exponential_base=2,
    non retryable errors=["AuthenticationError", "RateLimitError"],
)

llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "retry_config": retry_config,
}
```

---

## 12.5 代码示例说明

### 12.5.1 示例文件路径

本节包含以下代码示例：

- **llm_config_basic.py**：基础LLM配置示例
  - 路径：`part_05_LLM配置与模型接入/12_codes/llm_config_basic.py`

- **multi_model_fallback.py**：多模型fallback与重试逻辑示例
  - 路径：`part_05_LLM配置与模型接入/12_codes/multi_model_fallback.py`

### 12.5.2 运行说明

**前置条件**：
1. 确保已安装AutoGen：`pip install pyautogen`
2. 确保已在`.env`文件中配置好API密钥

**运行方式**：

```bash
# 进入代码目录
cd part_05_LLM配置与模型接入/12_codes

# 运行基础配置示例
python llm_config_basic.py

# 运行多模型fallback示例
python multi_model_fallback.py
```

**预期输出**：
- llm_config_basic.py：展示基础LLM配置创建代理的过程
- multi_model_fallback.py：展示主模型失败后自动切换到备用模型的过程

---

## 12.6 常见问题与解决方案

### Q1: fallback_models不生效怎么办？

**问题**：配置了fallback_models但没有自动切换

**解决方案**：
1. 确认fallback模型配置正确
2. 检查max_retries是否设置足够大
3. 查看日志确认是否触发了重试

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Q2: 如何确认使用了哪个模型？

**解决方案**：
通过代理的`model`属性查看当前使用的模型：

```python
print(f"当前使用模型: {agent.model}")
```

### Q3: 重试次数过多影响性能

**解决方案**：
1. 设置合理的max_retries（建议2-3次）
2. 启用指数退避减少无效重试
3. 设置最大重试延迟

---

## 12.7 本章小结

通过本章学习，你已经：

1. **掌握llm_config结构**：了解了模型配置、API配置、请求配置等完整参数
2. **理解多模型fallback机制**：当主模型不可用时自动切换到备用模型
3. **掌握重试逻辑配置**：包括指数退避、自定义重试策略等

下一章我们将学习如何接入本地模型，包括Ollama、vLLM和LiteLLM。

---

## 扩展阅读

- [AutoGen LLM配置文档](https://microsoft.github.io/autogen/docs/topics/LLM-configuration)
- [AutoGen Fallback机制](https://microsoft.github.io/autogen/docs/topics/Fallback)
- [重试策略最佳实践](https://docs.python.org/3/library/retrying.html)