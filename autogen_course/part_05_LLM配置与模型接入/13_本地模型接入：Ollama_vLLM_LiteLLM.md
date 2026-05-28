---
lesson_id: lesson_13
title: 本地模型接入：Ollama、vLLM与LiteLLM
module: LLM配置与模型接入
---

# 第13节 本地模型接入：Ollama、vLLM与LiteLLM

## 学习目标

1. 掌握Ollama的部署与AutoGen集成方法
2. 掌握vLLM高性能推理服务的配置
3. 理解LiteLLM的统一接口方案

---

## 13.1 本地模型概述

### 13.1.1 为什么使用本地模型

使用本地模型有以下优势：
- **成本可控**：无需支付API调用费用
- **数据隐私**：数据不出本地服务器
- **自定义微调**：可以根据需求微调模型
- **离线可用**：无需网络连接

### 13.1.2 主流本地模型方案

| 方案 | 特点 | 适用场景 |
|-----|------|---------|
| **Ollama** | 轻量级、易用、跨平台 | 个人开发者、小规模部署 |
| **vLLM** | 高吞吐、PagedAttention | 生产环境、大规模部署 |
| **LiteLLM** | 统一接口、多后端支持 | 需要混合使用多种模型的场景 |

---

## 13.2 Ollama部署与集成

### 13.2.1 Ollama简介

Ollama是一款轻量级的本地大语言模型运行框架，支持多种开源模型，如Llama 2、 Mistral、Code Llama等。

### 13.2.2 Ollama安装

**Windows/macOS安装**：

```bash
# 下载安装包
# 访问 https://ollama.com/download

# 或者使用命令行安装（macOS）
brew install ollama
```

**Linux安装**：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 13.2.3 常用Ollama命令

```bash
# 拉取模型
ollama pull llama2
ollama pull mistral

# 查看已安装模型
ollama list

# 运行模型
ollama run llama2 "你好，请介绍一下自己"

# 启动API服务
ollama serve
```

### 13.2.4 AutoGen集成Ollama

```python
import os
from autogen import ConversableAgent

# Ollama配置
llm_config = {
    "model": "llama2",  # Ollama模型名称
    "base_url": "http://localhost:11434/v1",  # Ollama API地址
    "api_key": "ollama",  # Ollama不需要真实API Key，但需要填写
    "temperature": 0.7,
}

# 创建代理
agent = ConversableAgent(
    name="ollama_assistant",
    llm_config=llm_config,
)

# 测试对话
response = agent.generate_reply(
    messages=[{"role": "user", "content": "你好"}]
)
print(f"回复: {response}")
```

### 13.2.5 高级Ollama配置

```python
# 更多Ollama配置选项
llm_config = {
    "model": "llama2",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",

    # 模型参数
    "temperature": 0.7,
    "top_p": 0.9,
    "num_ctx": 4096,  # 上下文长度
    "num_keep": 128,  # 保留的上下文token数

    # 系统提示
    "system": "你是一个有帮助的AI助手",
}
```

---

## 13.3 vLLM高性能推理

### 13.3.1 vLLM简介

vLLM是NVIDIA开源的高性能推理框架，使用PagedAttention技术，显著提升推理吞吐量。

### 13.3.2 vLLM安装

```bash
# 使用pip安装
pip install vllm

# 或者从源码编译
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 13.3.3 启动vLLM服务

```bash
# 启动vLLM OpenAI兼容API服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2
```

### 13.3.4 AutoGen集成vLLM

```python
from autogen import ConversableAgent

# vLLM配置
llm_config = {
    "model": "meta-llama/Llama-2-7b-hf",
    "base_url": "http://localhost:8000/v1",
    "api_key": "EMPTY",  # vLLM不需要API Key
    "temperature": 0.7,
    "max_tokens": 2000,
}

# 创建代理
agent = ConversableAgent(
    name="vllm_assistant",
    llm_config=llm_config,
)
```

### 13.3.5 vLLM性能优化

```bash
# 启动参数优化示例
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \        # GPU数量
    --gpu-memory-utilization 0.9 \    # GPU内存使用率
    --max-model-len 8192 \            # 最大模型长度
    --enforce-eager                   # 禁用CUDA图优化（减少内存）
```

---

## 13.4 LiteLLM统一接口

### 13.4.1 LiteLLM简介

LiteLLM是一个统一的LLM调用库，可以用一个接口调用100+个LLM服务（包括OpenAI、Azure、Ollama、vLLM等）。

### 13.4.2 LiteLLM安装

```bash
pip install litellm
```

### 13.4.3 LiteLLM代理模式

LiteLLM支持代理模式，提供统一的OpenAI兼容API：

```bash
# 启动LiteLLM代理
litellm --model ollama/llama2 --drop-on-err
litellm --model vllm/meta-llama/Llama-2-7b-hf

# 多模型代理
litellm \
    --model ollama/llama2 \
    --model vllm/meta-llama/Llama-2-7b-hf \
    --model azure/gpt-4o-mini
```

### 13.4.4 AutoGen集成LiteLLM

```python
from autogen import ConversableAgent

# LiteLLM配置
llm_config = {
    "model": "ollama/llama2",  # 格式：provider/model
    "base_url": "http://localhost:4000/v1",  # LiteLLM代理地址
    "api_key": "anything",  # LiteLLM代理不需要真实API Key
    "temperature": 0.7,
}

# 创建代理
agent = ConversableAgent(
    name="litellm_assistant",
    llm_config=llm_config,
)
```

### 13.4.5 LiteLLM多模型配置

```python
# LiteLLM配置文件示例
# config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: 2024-05-01

  - model_name: llama2-local
    litellm_params:
      model: ollama/llama2
      api_base: http://localhost:11434

  - model_name: vllm-llama2
    litellm_params:
      model: vllm/meta-llama/Llama-2-7b-hf
      api_base: http://localhost:8000
```

---

## 13.5 代码示例说明

### 13.5.1 示例文件路径

本节包含以下代码示例：

- **ollama_integration.py**：Ollama部署与AutoGen集成示例
  - 路径：`part_05_LLM配置与模型接入/13_codes/ollama_integration.py`

- **litellm_proxy.py**：LiteLLM统一接口配置示例
  - 路径：`part_05_LLM配置与模型接入/13_codes/litellm_proxy.py`

### 13.5.2 运行说明

**Ollama集成示例前置条件**：
1. 安装Ollama：`brew install ollama`（macOS）或从官网下载（Windows）
2. 启动Ollama服务：`ollama serve`
3. 拉取模型：`ollama pull llama2`

**运行方式**：

```bash
# 进入代码目录
cd part_05_LLM配置与模型接入/13_codes

# 运行Ollama集成示例
python ollama_integration.py

# 运行LiteLLM代理示例
python litellm_proxy.py
```

**预期输出**：
- ollama_integration.py：展示通过AutoGen调用本地Ollama模型
- litellm_proxy.py：展示通过LiteLLM统一接口调用多个后端模型

---

## 13.6 常见问题与解决方案

### Q1: Ollama模型下载慢怎么办？

**解决方案**：
1. 使用代理或VPN
2. 手动下载模型文件后导入
3. 使用国内镜像源（如果有）

```bash
# 设置代理
export https_proxy=http://127.0.0.1:7890
ollama pull llama2
```

### Q2: vLLM内存不足怎么解决？

**解决方案**：
1. 减小tensor-parallel-size
2. 降低gpu-memory-utilization
3. 使用更小的模型
4. 启用enforce-eager模式

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.7
```

### Q3: LiteLLM代理无法连接

**解决方案**：
1. 确认LiteLLM代理已启动
2. 检查端口是否正确（默认4000）
3. 验证模型名称格式正确

```bash
# 检查LiteLLM代理状态
curl http://localhost:4000/health
```

---

## 13.7 本章小结

通过本章学习，你已经：

1. **掌握Ollama集成**：安装配置Ollama，通过AutoGen调用本地模型
2. **掌握vLLM配置**：使用PagedAttention技术提升推理性能
3. **理解LiteLLM方案**：通过统一接口调用多个LLM后端

下一章我们将学习AutoGen的缓存机制与成本控制。

---

## 扩展阅读

- [Ollama官方文档](https://github.com/ollama/ollama)
- [vLLM GitHub仓库](https://github.com/vllm-project/vllm)
- [LiteLLM官方文档](https://docs.litellm.ai/)
- [本地模型性能对比](https://arxiv.org/abs/2401.04188)