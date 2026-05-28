---
lesson_id: lesson_06
title: Code Executor配置与安全机制
module: 代码执行器与工具执行器
---

# 第6节 Code Executor配置与安全机制

## 学习目标

1. 理解Code Executor的执行模型
2. 掌握不同代码解释器的配置方法
3. 能够处理代码执行异常与结果解析

---

## 6.1 Code Executor 执行模型概述

### 6.1.1 什么是Code Executor

Code Executor是AutoGen框架中用于安全执行LLM生成代码的核心组件。当LLM生成Python或其他代码时，Code Executor负责创建隔离的执行环境、运行代码并返回结果。

### 6.1.2 执行流程

```
LLM生成代码 → Code Executor接收 → 创建执行环境 → 运行代码 → 捕获输出 → 返回结果
```

### 6.1.3 架构设计

AutoGen的Code Executor采用分层设计：

| 层级 | 说明 |
|------|------|
| **代码生成层（LLM）** | LLM根据用户需求生成Python或其他语言代码 |
| **执行器抽象层** | 定义统一接口`execute_code()`，支持不同执行后端 |
| **执行环境隔离层** | LocalSingleCodeExecutor（进程级）、DockerExecutor（容器级） |
| **结果处理层** | 捕获stdout/stderr，收集执行时间、内存使用等信息 |

---

## 6.2 code_execution_config 参数详解

`code_execution_config`是配置代码执行器的核心字典，包含以下关键参数：

### 6.2.1 work_dir - 工作目录

- **类型**：`str` 或 `Path`
- **作用**：代码执行的根目录，生成的代码文件将保存在此目录下
- **默认值**：`None`（使用系统临时目录）
- **示例**：`work_dir="./code_workspace"`

### 6.2.2 use_docker - Docker容器配置

- **类型**：`bool` 或 `str`（容器名）
- **作用**：是否在Docker容器中执行代码，提供更强的隔离
- **默认值**：`True`（生产环境推荐）
- **可选值**：
  - `True`：使用AutoGen默认镜像
  - `"镜像名:标签"`：使用自定义镜像（如`"python:3.11-slim"`）
  - `False`：开发调试模式，直接在本地执行

### 6.2.3 timeout - 执行超时时间

- **类型**：`int`（秒）
- **作用**：单次代码执行的最大时长，超过则终止
- **默认值**：`30`秒
- **示例**：`timeout=60`表示60秒超时

### 6.2.4 last_n_messages - 参考消息数量

- **类型**：`int`
- **作用**：决定代码执行错误时，向LLM反馈多少条历史消息
- **默认值**：`6`
- **说明**：设为`0`表示只反馈最后一次执行结果

### 6.2.5 完整配置示例

```python
code_execution_config = {
    "work_dir": "./code_output",
    "use_docker": True,           # 推荐生产环境使用
    "timeout": 60,                 # 超时时间60秒
    "last_n_messages": 6,         # 错误反馈参考最近6条消息
}
```

---

## 6.3 本地代码执行器配置

### 6.3.1 适用场景

本地执行器适用于以下情况：

- 开发调试阶段，快速迭代代码
- 单机环境，无Docker支持
- 需要快速查看执行结果

### 6.3.2 配置方法

```python
local_executor_config = {
    "work_dir": "./code_output",
    "use_docker": False,           # 不使用Docker，用于本地调试
    "timeout": 60,
    "last_n_messages": 10,
}
```

### 6.3.3 风险提示

> **警告**：`use_docker=False`时，代码直接在本地执行，存在安全风险。仅在可信环境或开发阶段使用。

### 6.3.4 代码文件参考

详见 `06_codes/code_executor_config.py` 第三部分：本地代码执行器配置

---

## 6.4 Docker容器化代码执行器配置

### 6.4.1 Docker执行器的优势

- **操作系统级隔离**：防止恶意代码损害主机
- **环境一致性**：跨平台部署无问题
- **资源限制**：可控制CPU/内存使用

### 6.4.2 配置方法

```python
docker_executor_config = {
    "work_dir": "/app/code_workspace",   # 容器内的工作目录
    "use_docker": "autogen-code-executor",  # Docker镜像名称
    "timeout": 120,                     # 生产环境超时可设长一些
    "last_n_messages": 6,
}
```

### 6.4.3 use_docker参数的三种形式

| 形式 | 示例 | 说明 |
|------|------|------|
| `True` | `use_docker=True` | AutoGen自动选择合适的镜像 |
| `"镜像名:标签"` | `use_docker="python:3.11-slim"` | 使用指定镜像 |
| `False` | `use_docker=False` | 本地执行（仅用于开发调试） |

### 6.4.4 不同场景的配置组合

**开发调试模式** - 快速反馈：

```python
dev_config = {
    "work_dir": "./dev_workspace",
    "use_docker": False,
    "timeout": 30,           # 快速失败
    "last_n_messages": 3,    # 简洁错误信息
}
```

**生产环境模式** - 安全隔离：

```python
prod_config = {
    "work_dir": "/app/code_workspace",
    "use_docker": True,
    "timeout": 120,
    "last_n_messages": 6,
}
```

**数据分析场景** - 大文件处理：

```python
data_config = {
    "work_dir": "./data_analysis_workspace",
    "use_docker": "python:3.11-slim",
    "timeout": 300,          # 5分钟超时
    "last_n_messages": 10,
}
```

### 6.4.5 代码文件参考

详见 `06_codes/code_executor_config.py` 第四部分：Docker容器化代码执行器配置

---

## 6.5 Docker环境配置与隔离机制

### 6.5.1 Docker环境基础检查

在使用Docker执行器前，应检查Docker环境是否可用：

```python
import subprocess

def check_docker_available():
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
```

### 6.5.2 Docker容器安全配置

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `memory` | 内存限制 | `"256m"` |
| `network_disabled` | 禁用网络 | `True` |
| `read_only` | 只读文件系统 | `True` |
| `tmpfs` | 临时文件系统 | `["/tmp"]` |
| `user` | 运行用户 | `"1000:1000"` |

### 6.5.3 安全最佳实践

1. **禁用网络**：防止数据泄露和外部攻击
2. **限制内存**：防止恶意代码耗尽资源
3. **设置只读文件系统**：防止写入敏感目录
4. **使用非root用户**：降低权限风险

### 6.5.4 代码文件参考

详见 `06_codes/docker_sandbox.py`

---

## 6.6 use_docker=False 的开发调试模式

### 6.6.1 适用场景

1. **开发调试阶段**：快速迭代代码，便于使用调试器
2. **已确认代码安全性**：LLM生成的代码经过验证
3. **环境限制**：无Docker支持或资源受限

### 6.6.2 调试技巧

- 设置短超时时间（`timeout=10`）快速失败
- 使用较小的`last_n_messages`减少上下文
- 配合日志记录代码执行过程

### 6.6.3 风险警告

> **重要**：本地执行模式下，代码可以直接访问主机所有资源，可能执行恶意操作（删除文件、窃取数据等）。仅在可信环境和开发阶段使用。

---

## 6.7 错误处理与结果解析

### 6.7.1 执行结果结构

Code Executor执行后返回的结果包含以下信息：

| 字段 | 说明 |
|------|------|
| `output` | 标准输出内容（print输出、正常结果等） |
| `error` | 错误信息（语法错误、运行时异常等） |
| `exit_code` | 退出码（0表示成功，非0表示失败） |
| `elapsed_time` | 执行耗时（单位：秒） |
| `code_file` | 代码文件保存路径 |

### 6.7.2 错误分类

- **语法错误（SyntaxError）**：代码不符合Python语法
- **运行时错误（RuntimeError）**：代码执行时抛出异常
- **超时错误（TimeoutError）**：执行时间超过限制
- **权限错误（PermissionError）**：文件/目录访问被拒绝

---

## 6.8 在Agent中使用Code Executor

### 8.8.1 配置方式

在创建UserProxyAgent时传入`code_execution_config`：

```python
from autogen.agentchat.user_proxy_agent import UserProxyAgent

code_execution_config = {
    "work_dir": "./code_execution_workspace",
    "use_docker": True,
    "timeout": 60,
    "last_n_messages": 6,
}

user_proxy = UserProxyAgent(
    name="code_executor_agent",
    code_execution_config=code_execution_config,
    human_input_mode="NEVER",
)
```

### 6.8.2 工作流程

1. 用户向AssistantAgent发送任务请求
2. AssistantAgent生成代码
3. UserProxyAgent接收代码并调用Code Executor
4. Executor在Docker容器中执行代码
5. 结果返回给AssistantAgent进行下一步处理

---

## 6.9 常见问题与解决方案

### Q1: Docker守护进程未运行

**症状**：`Cannot connect to the Docker daemon`

**解决**：
- Linux/Mac: `sudo systemctl start docker`
- Windows: 启动Docker Desktop应用

### Q2: 权限被拒绝

**症状**：`Got permission denied while trying to connect`

**解决**：
- Linux: `sudo usermod -aG docker $USER`
- Mac/Windows: 确保Docker Desktop以当前用户运行

### Q3: 镜像拉取失败

**症状**：`docker pull`超时或失败

**解决**：
- 检查网络连接
- 配置Docker镜像加速器
- 手动拉取镜像

### Q4: 执行超时

**症状**：代码执行总是超时

**解决**：
- 适当增加`timeout`值
- 检查代码是否有死循环
- 考虑优化代码或分段执行

---

## 6.10 本章小结

通过本章学习，你已经：

1. **理解Code Executor执行模型**：掌握从代码生成到执行结果返回的完整流程
2. **掌握配置方法**：学会配置`work_dir`、`use_docker`、`timeout`、`last_n_messages`等参数
3. **区分执行环境**：了解本地执行器和Docker执行器的适用场景
4. **处理异常结果**：学会解析执行结果和处理各类错误

---

## 扩展阅读

- [AutoGen官方文档 - Code Executor](https://microsoft.github.io/autogen/)
- [Docker官方文档](https://docs.docker.com/get-started/)
- 代码示例：参见 `06_codes/code_executor_config.py` 和 `06_codes/docker_sandbox.py`

---

**framework_ref**: `mod_004`