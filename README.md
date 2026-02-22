# AutoGen Learning

一个基于 Microsoft AutoGen 的学习型项目，包含：
- 对 OpenAI 兼容接口的连通性检测脚本
- 基于 `ConversableAgent` 的最小可运行示例
- 两份 AutoGen 学习 Notebook（入门与源码解析）

## 项目结构

```text
.
├─ autogen_case_1.py
├─ connect_openai.py
├─ requirements.txt
├─ .env.example
├─ 1. MicroSoft AutoGen 基础入门.ipynb
├─ 2. MicroSoft AutoGen 代理对话与人机交互源码解析.ipynb
├─ autogen_learning.md
├─ assests/
└─ source_code/   # 本地参考源码快照（非运行必需）
```

## 环境要求

- Python 3.10 - 3.12（建议 3.11）
- 可访问的 OpenAI 兼容 API（如 AiHubMix）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

1. 复制示例配置：

```bash
cp .env.example .env
```

Windows PowerShell 可用：

```powershell
Copy-Item .env.example .env
```

2. 按需填写 `.env`：

```env
AIHUBMIX_API_KEY=your_api_key
AIHUBMIX_BASE_URL=https://aihubmix.com/v1
OPENAI_MODEL=gpt-4o-free
```

## 运行脚本

### 1) 检查 API 与模型可用性

```bash
python connect_openai.py
```

常用参数：

```bash
python connect_openai.py --list-models-only
python connect_openai.py --message "Hello"
python connect_openai.py --use-env-proxy
```

### 2) 运行 AutoGen 最小示例

```bash
python autogen_case_1.py
```

该脚本会从 `.env` 读取：
- `AIHUBMIX_API_KEY`
- `AIHUBMIX_BASE_URL`
- `OPENAI_MODEL`

## 版本说明（重要）

本项目示例使用的是 AutoGen 0.2.x 风格 API：

```python
from autogen.agentchat.conversable_agent import ConversableAgent
```

因此依赖选择为 `pyautogen==0.2.40`。如果只安装 `autogen-agentchat` 0.7+，将是新版包体系，API 与本项目示例不完全兼容。

## Notebook 说明

- `1. MicroSoft AutoGen 基础入门.ipynb`：基础概念与入门示例。
- `2. MicroSoft AutoGen 代理对话与人机交互源码解析.ipynb`：对代理对话机制与源码结构的学习记录。

## 常见问题

1. `Missing required environment variable`：检查 `.env` 是否存在且变量名正确。
2. 连接超时/网络异常：先运行 `connect_openai.py` 定位是密钥、模型还是网络问题。
3. `ConversableAgent` 导入失败：确认安装的是 `pyautogen==0.2.40`，并在当前解释器环境执行。
