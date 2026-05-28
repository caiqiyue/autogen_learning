# AutoGen 课程总览

本课程系统讲解 Microsoft AutoGen 框架，涵盖从基础概念到企业级架构设计的完整知识体系。

---

## 学习路径

1. **夯实基础** (part_01) - AutoGen 框架基础与核心概念
2. **深入核心** (part_02) - ConversableAgent 核心机制深度解析
3. **掌握工具** (part_03) - 代码执行器与工具执行器
4. **协作进阶** (part_04) - 多 Agent 协作与 GroupChat 高级机制
5. **模型配置** (part_05) - LLM 配置与模型接入
6. **实战应用** (part_06) - AssistantAgent 与 UserProxyAgent 应用
7. **性能优化** (part_07) - 异步 Agent 与并发协作
8. **生态集成** (part_08) - AutoGen 与主流技术栈集成
9. **架构设计** (part_09) - 企业级 AutoGen 应用架构设计
10. **最佳实践** (part_10) - AutoGen 最佳实践与高频误区

---

## 课程结构

| 模块 | 名称 | 课时 |
|------|------|------|
| part_01 | AutoGen 框架基础与核心概念 | lesson_01, lesson_02 |
| part_02 | ConversableAgent 核心机制深度解析 | lesson_03, lesson_04, lesson_05 |
| part_03 | 代码执行器与工具执行器 | lesson_06, lesson_07, lesson_08 |
| part_04 | 多 Agent 协作与 GroupChat 高级机制 | lesson_09, lesson_10, lesson_11 |
| part_05 | LLM 配置与模型接入 | lesson_12, lesson_13, lesson_14 |
| part_06 | AssistantAgent 与 UserProxyAgent 应用 | lesson_15, lesson_16, lesson_17, lesson_18, lesson_19 |
| part_07 | 异步 Agent 与并发协作 | lesson_20, lesson_21 |
| part_08 | AutoGen 与主流技术栈集成 | lesson_22, lesson_23 |
| part_09 | 企业级 AutoGen 应用架构设计 | lesson_24, lesson_25, lesson_26 |
| part_10 | AutoGen 最佳实践与高频误区 | lesson_27, lesson_28 |

**总计：28 节课，10 个模块**

---

## 快速开始指南

### 环境准备

```bash
pip install autogen-agentchat autogen-agentchat[openai]
```

### 基础示例

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# 创建 Agent
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
)

# 运行对话
async def main():
    result = await agent.run(
        messages=[TextMessage(content="你好，介绍下自己", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(result.output_messages[-1].content)

import asyncio
asyncio.run(main())
```

### 学习建议

- 每节课先通读概念，再动手实践
- 结合源码理解核心机制
- 完成课后思考题巩固知识