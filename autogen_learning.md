# 一、AutoGen框架介绍

&emsp;&emsp;`AutoGen` 是 `Microsoft` 在大模型技术领域推出的开创性 `AI Agent` 开发框架，也是目前最受欢迎的`AI Agent`开发工具之一。自 `2023` 年 `10` 月发布以来，`AutoGen` 凭借其前沿的技术迭代目前在 `GitHub` 上已累计获得超过 36.7k 次的星标，同时截至`2025年01月03日`，`AutoGen`开源的最新版本是`0.2.40`，期间已经迭代了`65`个小版本，具有非常广泛的用户关注度和社区活跃度。如下图所示：

[microsoft/autogen: A programming framework for agentic AI](https://github.com/microsoft/autogen)

![{0FED2041-CB50-4033-81AC-D2F3323D3165}](E:\autogen_learning\assests\{0FED2041-CB50-4033-81AC-D2F3323D3165}.png)

&emsp;&emsp;***\*****`AutoGen`** **框架的一大特色是支持创建对话式应用\****。也就是说它构建多代理的方式是使多个智能体能够相互交流，从而促进不同智能体之间的合作以完成最终的任务。简单的理解就是这个框架可以让不同的`Agent`建立起通信的连接，然后***\*它提供给开发者的使用方式是，其一可以为每个****`Agent`****自定义大模型、角色、工具及行为，其二可以创建不同的对话模式，包括一问一答、联合聊天、分层聊天等等\****，从而实现高度个性化的应用场景设计。如下图所示：

\> MicroSoft AutoGen Docs：https://microsoft.github.io/autogen/0.2/docs/Getting-Started

![{222B3E94-F02D-4215-B86D-09C43606359D}](E:\autogen_learning\assests\{222B3E94-F02D-4215-B86D-09C43606359D}.png)

## AutoGen框架特征

***\*Agent Customization\***

&emsp;&emsp;`AutoGen`支持开发者根据特定需求定制代理（Agent）。这种定制化***\*允许开发者定义代理的行为、响应方式和功能，使其适应不同的应用场景\****。例如，我们可以创建自定义代理来执行特定的算术操作或其他任务。此外，***\*****`AutoGen`** **还允许集成自定义的大模型，进一步增强代理的能力\****。

\- ***\*Multi-Agent Conversations\****

&emsp;&emsp;`AutoGen` 支持多代理对话（Multi-Agent Conversations）。在 `AutoGen` 中，***\*多个代理（Agents）可以通过对话相互交流，协作完成复杂任务\****。这些代理可以由大语言模型（LLMs）、人类输入或工具驱动，具备可定制和可对话的特性。

\- ***\*Flexible Conversation Patterns\****

&emsp;&emsp;`AutoGen` 支持灵活的对话模式（Flexible Conversation Patterns），允许开发者根据应用需求设计多种代理交互方式。比如***\*联合聊天（Joint Chat）\****，在这种模式下，多个代理共同参与同一个对话线程，所有代理共享相同的上下文。而***\*层次化聊天（Hierarchical Chat）\**** 这种模式涉及将一个工作流封装为单个代理，以便在更大的工作流中重复使用。除此以外，像最基本的对话形式（Two-Agent Chat）用来构建两个代理之间的交流，顺序对话（Sequential Chat）通过上下文传递将前一次对话的摘要带入下一次对话等等多样化的对话模式，能够覆盖绝大部分的应用需求。

&emsp;&emsp;基于上述提到的这些功能和模式，`AutoGen`的开源仓库中提供了非常多如`MathChat`、`文本分析`等应用模板，可以在我们的项目中自由修改和部署，这也是它非常受欢迎的一个主要原因。

&emsp;&emsp;***\*****`AutoGen`****框架的第二大优势，则是与其他****`AI Agent`****开发框架（如** **`LangChain`****、****`LangGraph`****）、****`RAG`** **以及函数调用等功能集成在一起\****，所以我们可以非常方便的通过额外的知识源去增强基于大模型的代理能力，从而使基于`AutoGen`构建的智能体可以解决相对复杂的问题和更多元化的应用场景。



## AutoGen 中“对话式应用”的含义

1. 概念本质

所谓“对话式应用”（Conversational Application），核心并不是“像聊天机器人那样与用户对话”，而是：

> **系统内部的多个 Agent 通过对话进行协作，以完成任务。**

重点是 **Agent 与 Agent 之间的对话机制**，而不是用户界面形式。

------

2. 与传统单模型应用的区别

传统 LLM 应用模式

典型流程：

```
User → Prompt → 单个大模型 → Response
```

特点：

- 所有任务由一个模型完成
- 逻辑集中在 Prompt 工程
- 难以拆分复杂职责
- 模型既要“查资料”又要“分析”又要“生成”

------

AutoGen 的对话式模式

典型流程：

```
User → 多个 Agent → Agent 之间对话 → 最终结果
```

特点：

- 多 Agent 分工
- Agent 之间通过自然语言通信
- 协作过程可控
- 更接近“团队工作模式”

------

示例：自动生成行业分析报告

场景描述

用户输入：

> “生成一份关于新能源汽车市场的分析报告”

------

Agent 设计

可以创建多个 Agent：

1. ResearchAgent
   - 职责：检索信息、收集数据
2. AnalystAgent
   - 职责：数据分析、趋势推理
3. WriterAgent
   - 职责：撰写报告
4. ReviewerAgent
   - 职责：校对、质量评估

------

系统内部协作方式

ResearchAgent：

> 我找到了市场规模数据和增长率。

AnalystAgent：

> 请提供数据，我进行趋势分析。

WriterAgent：

> 分析结果完成后，我将生成报告结构。

ReviewerAgent：

> 报告中关于增长预测的论证需要补充依据。

------

用户视角

用户只看到：

- 输入需求
- 输出完整报告

用户并不直接看到 Agent 对话，但应用是由“对话驱动协作”完成的。

这就是“对话式应用”。

------

“支持多种对话模式”是什么意思

AutoGen 不只是让 Agent 随意聊天，而是允许设计不同协作结构。

------

1. 一问一答模式（Two-Agent Chat）

结构：

```
User ↔ AssistantAgent
```

适用于：

- 简单问答
- 单任务执行

------

2. 联合聊天模式（Group Chat）

结构：

```
User ↔ 多 Agent 群聊
```

适用于：

- 多角色讨论
- 方案对比 / 辩论

------

3. 分层聊天模式（Hierarchical Chat）

结构：

```
ManagerAgent → WorkerAgents
```

适用于：

- 任务拆解
- 指挥与执行分离

------

4. 协作推理模式（Collaborative Reasoning）

结构：

```
Agent A → Agent B → Agent C
```

适用于：

- 逐步推理
- 校验与修正链条

------

与其他框架的对比

LangChain

**核心定位：**

- Chain（流程编排）
- Tool（工具调用）
- Agent（决策执行）

**特点：**

- 强在工作流/工具链
- 多 Agent 对话不是核心机制
- Agent 更像“智能路由器”

**更像：**

> 函数式调用 + 流程管道

------

**LlamaIndex**

**核心定位：**

- RAG
- 文档索引与检索

**特点：**

- 强在知识库问答
- 不强调 Agent 协作

**更像：**

> 智能搜索系统

------

CrewAI

**核心定位：**

- 多 Agent 团队协作

**特点：**

- 与 AutoGen 相似
- 偏应用封装
- AutoGen 更偏底层与研究灵活性

------

AutoGen

**核心定位：**

- 多 Agent 对话系统
- 对话即协作机制

**特点：**

- Agent 天生通过对话协同
- 对话模式高度可控
- 非常适合复杂任务拆解

**更像：**

> AI 团队模拟框架

------

AutoGen 适用场景

适合：

- 复杂任务拆解
- 多角色协作
- 自动编程 / 调试 / 执行
- 需要交叉验证的任务
- 研究 / 实验型系统

不一定适合：

- 简单问答
- 轻量级工具调用
- 单 Agent 足够的任务

------

总结

对话式应用在 AutoGen 中可以理解为：

> **通过多个 Agent 之间的对话来驱动任务执行的应用架构。**

它的创新点不是 UI，而是：

- 通信机制
- 协作结构
- 角色化智能体系统

------

如果你需要，我可以下一步提供：

- AutoGen 与 LangChain 的架构图对比
- 一个最小可运行的 AutoGen 示例（代码）
- 如何设计 Agent 角色的实践指南



# 二、AutoGen组件

 AutoGen ConversableAgent：https://microsoft.github.io/autogen/0.2/docs/tutorial/introduction



## ConversableAgent



&emsp;&emsp;在 `AutoGen` 框架中，代理（Agent）可以由多种组件驱动，主要包括：

- **大语言模型（LLMs）**：例如 `GPT-4` 、`GLM 4`等，用于自然语言处理和生成。
- **人类输入**：代理可以接受人类的直接输入，进行交互或获取指令。
- **代码执行器**：如 IPython 内核，允许代理执行代码，实现动态计算和任务处理。
- **其他可插拔和可定制的组件**：根据具体需求，代理可以集成其他工具或功能模块，以扩展其能力。

&emsp;&emsp;这四种不同类型的代理之间能够进行对话交互，而 `AutoGen`做的事情就是提供多种不同的实现方法来支持这些交互。其中，我们**需要优先学习和掌握的，是`AutoGen`框架内置的 `ConversableAgent`。该类作为一个基础的代理类，提供了非常灵活的接口，允许我们根据具体需求启用或禁用特定功能，并进行相应的配置**。除此以外，`AssistantAgent` 和 `UserProxyAgent` 都是 `ConversableAgent` 的子类，分别用于执行任务处理、调用 `API` 和逻辑推理，以及模拟用户输入和执行代码等，我们将在后续的课程中再逐步的展开介绍。

![image-20260221231008308](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260221231008308.png)



1. **核心组成模块**

从结构上看，ConversableAgent 并不是单一 LLM，而是一个**多能力聚合体**，通常包括以下组件：

1.1 LLM（语言模型）

作用：

- 理解消息（来自用户或其他 Agent）
- 生成回复
- 决定下一步行为（是否调用工具 / 执行代码 / 请求人工）

特点：

- 可自定义模型（GPT、Claude、本地模型等）
- 可设定 system prompt（角色、风格、策略）

本质地位：

- **认知与推理中枢**

------

1.2 Code Executor（代码执行器）

作用：

- 执行 Agent 生成的代码
- 返回运行结果（stdout / 错误 / 文件）

典型用途：

- 数据分析
- 数值计算
- 画图
- 自动调试

设计意义：

- 让 Agent 从“文本生成器”变成“可行动体”

------

1.3 Tool Executor（工具执行器）

作用：

- 调用外部工具或 API
- 将调用结果反馈给 Agent

典型工具：

- 搜索引擎
- 数据库
- 自定义函数
- 企业内部系统接口

与 Code Executor 区别：

| Code Executor | Tool Executor        |
| ------------- | -------------------- |
| 执行代码      | 调用封装好的工具/API |
| 更灵活        | 更安全/可控          |
| 适合计算/编程 | 适合业务能力扩展     |

------

1.4 Human-in-the-loop（人类参与接口）

作用：

- 在关键决策点请求人工反馈
- 人类可修正 / 批准 / 提供额外信息

典型场景：

- 高风险操作
- 需要主观判断
- 复杂策略确认

设计意义：

- 避免 Agent 自主决策失控
- 提供安全兜底机制

------

2. “**Custom**” 模块代表什么

图中底部的 **Custom** 表示：

> ConversableAgent 支持扩展任意能力模块

例如：

- 自定义 Memory（长期记忆）
- 自定义 Planner（任务规划器）
- 自定义安全策略模块
- 自定义业务逻辑组件

这体现的是 AutoGen 的一个关键理念：

> Agent 是一个“能力组合框架”，而非固定结构。

------

3. **这些模块如何协同工作**

ConversableAgent 的运行逻辑大致是：

```
接收消息
  ↓
LLM 解析与推理
  ↓
判断是否需要动作
  ├─ 直接回复
  ├─ 调用 Tool Executor
  ├─ 调用 Code Executor
  └─ 请求 Human-in-the-loop
  ↓
获得结果
  ↓
生成下一轮对话
```

关键点：

- LLM 负责“思考与决策”
- Executor 负责“执行”
- 对话负责“状态推进”

------

4. **为什么要这样设计**

这种构成解决了传统 LLM 应用中的几个问题：

4.1 单模型能力受限

LLM 只能输出文本，不能真正“操作世界”。

→ Executor 提供行动能力

------

4.2 复杂任务难以闭环

例如：

- 写代码 → 运行 → 报错 → 修改 → 重试

→ Code Executor + 对话循环形成闭环

------

4.3 缺乏可靠性与校验

→ Human-in-the-loop 提供人工审查

------

4.4 能力扩展困难

→ Tool Executor + Custom 模块支持无限扩展

------

5. **从抽象层面理解** ConversableAgent

可以将其理解为：

| 抽象视角 | 对应解释                                 |
| -------- | ---------------------------------------- |
| 软件架构 | 控制器（LLM） + 执行层（Executor）       |
| 组织结构 | 大脑（LLM） + 手（工具/代码） + 人类顾问 |
| 运行机制 | 对话驱动的状态机                         |

6.**源码解析**

![{D6F9492A-2468-4150-B361-04B51090619E}](E:\autogen_learning\assests\{D6F9492A-2468-4150-B361-04B51090619E}.png)



```python
    def __init__(
        self,
        name: str,
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
    ):
```

&emsp;&emsp;在 `ConversableAgent` 类的 `__init__` 方法中，参数分为必填和选填两类。必填参数在实例化时必须提供，而选填参数有默认值，可根据需要进行配置。以下是这些参数的详细说明：

| 参数名 | 类型 | 默认值 | 必填/选填 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 无 | 必填 | 代理的名称。 |
| `system_message` | `Optional[Union[str, List]]` | `"You are a helpful AI Assistant."` | 选填 | 用于 ChatCompletion 推理的系统消息。 |
| `is_termination_msg` | `Optional[Callable[[Dict], bool]]` | `None` | 选填 | 判断接收到的消息是否为终止消息的函数。该函数接受一个字典形式的消息，返回布尔值。字典可能包含以下键："content"、"role"、"name"、"function_call"。 |
| `max_consecutive_auto_reply` | `Optional[int]` | `None` | 选填 | 连续自动回复的最大次数。默认为 `None`（无特定限制，此时将使用类属性 `MAX_CONSECUTIVE_AUTO_REPLY` 作为限制）。设置为 `0` 时，不会生成自动回复。 |
| `human_input_mode` | `Literal["ALWAYS", "NEVER", "TERMINATE"]` | `"TERMINATE"` | 选填 | 指定在每次收到消息时是否请求人类输入。可能的取值包括：<br> - `"ALWAYS"`：每次收到消息时都请求人类输入。在此模式下，当人类输入为 `"exit"`，或 `is_termination_msg` 返回 `True` 且没有人类输入时，对话停止。<br> - `"TERMINATE"`：仅在收到终止消息或自动回复次数达到 `max_consecutive_auto_reply` 时请求人类输入。<br> - `"NEVER"`：从不请求人类输入。在此模式下，当自动回复次数达到 `max_consecutive_auto_reply` 或 `is_termination_msg` 返回 `True` 时，对话停止。 |
| `function_map` | `Optional[Dict[str, Callable]]` | `None` | 选填 | 将函数名称映射到可调用函数的字典，用于工具调用。 |
| `code_execution_config` | `Union[Dict, Literal[False]]` | `False` | 选填 | 代码执行的配置。若设置为 `False`，则禁用代码执行。否则，应设置为包含以下键的字典：<br> - `work_dir`（可选，`str`）：代码执行的工作目录。若为 `None`，将使用默认工作目录，通常为 `autogen` 路径下的 "extensions" 目录。<br> - `use_docker`（可选，`list`、`str` 或 `bool`）：用于代码执行的 Docker 镜像。默认为 `True`，表示将在 Docker 容器中执行代码，并使用默认的镜像列表。若提供了镜像名称的列表或字符串，将在成功拉取的第一个镜像中执行代码。若为 `False`，将在当前环境中执行代码。强烈建议使用 Docker 进行代码执行。<br> - `timeout`（可选，`int`）：代码执行的最大时间（秒）。<br> - `last_n_messages`（实验性，`int` 或 `str`）：用于代码执行的消息回溯数量。若设置为 `'auto'`，将向后扫描自代理上次发言以来的所有消息，通常是上次尝试执行代码以来的消息。 |
| `llm_config` | `Optional[Union[Dict, Literal[False]]]` | `None` | 选填 | LLM 推理配置。请参考 [OpenAIWrapper.create](https://microsoft.github.io/autogen/0.2/docs/reference/oai/client#create) 获取可用选项。使用 OpenAI 或 Azure OpenAI 时，请在 `llm_config` 或 `llm_config` 中的 `config_list` 的每个配置中指定非空的 `'model'`。若设置为 `False`，则禁用基于 LLM 的自动回复。若为 `None`，将使用 `self.DEFAULT_CONFIG`，默认为 `False`。 |
| `default_auto_reply` | `Union[str, Dict]` | `""` | 选填 | 当未生成代码执行或基于 LLM 的回复时的默认自动回复。 |
| `description` | `Optional[str]` | `None` | 选填 | 代理的简短描述。其他代理（如 `GroupChatManager`）可根据此描述决定何时调用该代理。默认为 `system_message`。 |
| `chat_messages` | `Optional[Dict[Agent, List[Dict]]]` | `None` | 选填 | 代理与其他代理之间的先前聊天消息记录。可用于提供聊天历史，从而使代理具备记忆功能，能够恢复之前的对话。默认为空的聊天历史。 |

&emsp;&emsp;这些参数允许在实例化 `ConversableAgent` 时进行灵活配置，以满足不同的对话需求。而仔细查看`ConversableAgent(LLMAgent)`中定义的子方法后，能够明确其核心在于：**它用于处理消息的接收与发送、管理对话记录、执行代码与工具调用，并支持与大模型（LLM）的集成。它具备灵活的自动回复配置能力，通过注册自定义触发条件和回复函数来实现高度定制化的交互。除此之外，类还提供异步与嵌套对话支持，确保复杂对话场景的高效处理。这些特性使 ConversableAgent 成为开发多功能对话系统的基础工具。**



### LLM为主的agent

&emsp;&emsp;`AutoGen` 框架支持以下三种类型的大模型接入，分别是：

\- ***\*OpenAI 模型\****：如 `gpt-3.5-turbo`、`gpt-4` 等。

\- ***\*Azure OpenAI 模型\****：通过 `Azure` 平台提供的 `OpenAI` 服务。

\- ***\*其他兼容 OpenAI API 的模型\****：如 `Anthropic` 的 `Claude` 系列模型，`Ollama`、`Vllm`接入的本地开源大模型。

AutoGen LLM Configuration：https://microsoft.github.io/autogen/0.2/docs/topics/llm_configuration/



![{59E7796E-B8D2-4301-B5E0-670A490C0752}](E:\autogen_learning\assests\{59E7796E-B8D2-4301-B5E0-670A490C0752}.png)

`Azure OpenAI`服务是微软与 `OpenAI` 合作推出的云服务，通过将 `OpenAI` 的模型与微软 `Azure` 的企业级安全性和基础设施相结合，用户可以在 `Azure` 平台上访问和使用 `OpenAI` 的模型，直接的好处是在国内网络`Azure OpenAI` 服务，而无需使用翻墙工具。使用国内的信用卡、手机号和 IP 地址即可在 `Azure` 上注册账户并申请 `OpenAI` 服务。 但目前好像已暂停中国个人用户的 `OpenAI` 服务，链接：https://azure.microsoft.com/en-us/products/ai-services/openai-service

&emsp;&emsp;除此以外，对于非`OpenAI`的`GPT`系列模型，`AutoGen`框架还支持接入任意符合`OpenAI` 兼容 `API` 的代理服务器，Anthropic Claude`、`Gemini`, `GLM`等，`Ollama`、`vllm`等都可以无缝接入到`AutoGen`中构建大模型驱动的代理。



AutoGen Non-OpenAI Models：https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/about-using-nonopenai-models

![{AA88C830-17F6-4B49-86F9-4C05A17F9EC0}](E:\autogen_learning\assests\{AA88C830-17F6-4B49-86F9-4C05A17F9EC0}.png)



