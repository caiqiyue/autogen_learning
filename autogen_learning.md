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



### LLM为主的agent（case1）

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





&emsp;&emsp;`class ConversableAgent(LLMAgent)` 是通用可对话代理的类，因此***\*它的核心功能是可以直接针对用户输入的问题生成大模型的响应，并且返回到用户端。 另外，因为需要接入大模型去驱动代理，所以这里需要使用** **`ConversableAgent`** **类中的****`llm_config`****参数来指定大模型实例。\**** 而关于如何使用，我们需要关注一下`class ConversableAgent(LLMAgent)`源码中的定义逻辑，主要有以下两个关注点：

&emsp;&emsp;首先，根据`class ConversableAgent(LLMAgent)`类的定义，`llm_config`需要接收的是一个字典：

```
llm_config: Optional[Union[Dict, Literal[False]]] = None,
```



&emsp;&emsp;其次，在`_validate_llm_config`方法中，其校验的逻辑如下：

```python
    def _validate_llm_config(self, llm_config):
        assert llm_config in (None, False) or isinstance(
            llm_config, dict
        ), "llm_config must be a dict or False or None."
        if llm_config is None:
            llm_config = self.DEFAULT_CONFIG
        self.llm_config = self.DEFAULT_CONFIG if llm_config is None else llm_config
        # TODO: more complete validity check
        if self.llm_config in [{}, {"config_list": []}, {"config_list": [{"model": ""}]}]:
            raise ValueError(
                "When using OpenAI or Azure OpenAI endpoints, specify a non-empty 'model' either in 'llm_config' or in each config of 'config_list'."
            )
        self.client = None if self.llm_config is False else OpenAIWrapper(**self.llm_config)
```



&emsp;&emsp;这里会验证 `llm_config` 必须是 `None`、`False` 或字典。如果 `llm_config` 包含键 `config_list`，则该键值必须是一个列表，其中每个字典对象需要包含一个有效的 'model'。如果没有提供 config_list 或其中的 model 为空，则会抛出错误。因此，如果我们这里想接入一个`OpenAI`的`GPT`模型，接入规范就如下所示：

```python
import os

from autogen import ConversableAgent

agent = ConversableAgent(
    name="chatbot",
    llm_config={"config_list": 
                [{"model": "gpt-4o-mini", 
                  "api_key": os.environ.get("OPENAI_API_KEY")}]},
)
```



&emsp;&emsp;`config_list` 允许指定不同的端点和配置 被使用。在此方法中，`config_list` 中的每个字典必须包含**至少一个 `model` 和 `api_key` 的组合**。如果缺少 `model` 或 `api_key`，会触发 ValueError。其可以使用的参数如下：

| 参数名称       | 类型        | 必需性    | 描述                                                                 |
| -------------- | ----------- | --------- | -------------------------------------------------------------------- |
| `model`        | `str`       | 必需      | 要使用的模型的标识符，例如 `'gpt-4'`，`'gpt-3.5-turbo'`。            |
| `api_key`      | `str`       | 可选      | 验证模型 API 端点请求所需的 API 密钥。                               |
| `api_rate_limit` | `float`   | 可选      | 指定每秒允许的最大 API 请求数。                                      |
| `base_url`     | `str`       | 可选      | API 端点的基本 URL，这是 API 调用所定向的根地址。                   |
| `tags`         | `List[str]` | 可选      | 可用于过滤的标签。                                                  |



&emsp;&emsp;最后，如果想要触发大模型调用并且得到最终的响应，则需要在定义的`agent`实例中（即`ConversableAgent`）调用`generate_reply`方法。`generate_reply` 是 `ConversableAgent` 的核心功能之一，此函数会根据接收到的消息和配置，通过一系列注册的处理函数和回复生成函数，来产生一个回复。

```python
  async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        messages = self.process_all_messages_before_reply(messages)

        messages = self.process_last_received_message(messages)

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue

            if self._match_trigger(reply_func_tuple["trigger"], sender):
                if inspect.iscoroutinefunction(reply_func):
                    final, reply = await reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                else:
                    final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return self._default_auto_reply
    
它实现了一个“可插拔回复策略链”：

准备/标准化消息

遍历策略列表（可排除）

策略按 trigger 决定是否执行

执行后返回 (final, reply)

第一个 final=True 直接作为最终回复，否则返回默认回复
```



&emsp;&emsp;源码 `generate_reply` 的逻辑主要围绕以下两点：`

- **第一步：根据消息内容和发送方，决定是否执行普通问答逻辑或触发与其他 Agent 的交互**。
  - 普通问答: 如果当前的 `messages` 表示普通对话内容，那么直接按消息上下文调用适当的回复逻辑（如调用 LLM 或简单地使用默认回复）。
  - 如果有一个明确的 `sender`，则消息会被理解为来自某个特定 `Agent` 的请求。
  <br>
  <br> 
- **第二步：在会话过程中，检查是否需要调用不同类型的回复函数，并按优先级顺序依次尝试生成响应**。回复函数的类型与优先级为：
  - check_termination_and_human_reply：检查是否需要终止会话或请求人类输入。
  - generate_function_call_reply：处理函数调用（已废弃，建议使用 tool_calls）。
  - generate_tool_calls_reply：生成工具调用相关的回复。
  - generate_code_execution_reply：根据消息中的代码块执行代码并返回结果。
  - generate_oai_reply：通过 LLM 模型生成对话回复。



&emsp;&emsp;`ConversableAgent`类中`generate_reply`方法定义过程其实是比较复杂的，但使用非常简单。对于普通的大模型对话交互过程，我们只需要传入`messages`字段，`generate_reply` 就会自动处理所有的内部逻辑，因此调用代码如下：

```python
# 调用代理生成回复
reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "你好，请你非常详细的介绍一下你自己"
        }
    ]
)

# 打印生成的回复
print(reply)
```



#### caching缓存机制

&emsp;&emsp;在成功使用`ConversableAgent`接入在线大模型生成问答的回复，这里有一点需要格外注意：***\*当我们使用相同的大模型并提出相同的问题时，会发现其回复速度非常快，且内容与之前一致\****。

&emsp;&emsp;这是因为 `AutoGen` 框架的设计采用了缓存机制。该对话过程支持对 `API` 请求进行缓存，以便在发出相同请求时可以重复使用之前的响应结果。这种机制在重复或持续的实验中非常有用，有助于提高结果的可重复性并节省成本。***\*从版本** **`0.2.8`** **开始，****`AutoGen`** **提供了一个可配置的上下文管理器，允许我们轻松配置** **`LLM`** **缓存，支持多种缓存类型，如磁盘缓存（DiskCache）、Redis 缓存（RedisCache）或 Azure Cosmos DB 缓存。\****

&emsp;&emsp;其配置的方法在 `llm_config` 参数中，默认会开启缓存机制并存储在磁盘的上下文管理器中。如果想在调用`generate_reply`方法时禁用缓存，可以通过在代理的 `llm_config` 中设置 `cache_seed` 参数为 `None` 来实现。代码如下：



```
import os

from autogen import ConversableAgent

agent = ConversableAgent(
    name="chatbot",
    llm_config={
        "cache_seed": None,  # 禁用缓存
        "config_list": 
                [{"model": "gpt-4o-mini", 
                  "api_key": os.environ.get("OPENAI_API_KEY")}]},
)

# 调用代理生成回复
reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "请你介绍一下什么是大模型？"
        }
    ]
)

# 打印生成的回复
print(reply)
```



&emsp;&emsp;此时能够发现在上述代码中，通过在 `llm_config` 中设置 `"cache_seed": None`后可以禁用缓存功能。因此每次调用 `generate_reply` 时，都会直接向模型发送请求，而不会使用之前的缓存结果，从而每次都能得到不同的结果。&emsp;&emsp;除此以外，还可以改变`cache_seed`参数以获得不同的大模型输出，同时仍然使用缓存。比如：

```
import os

from autogen import ConversableAgent

agent = ConversableAgent(
    name="chatbot",
    llm_config={
        "cache_seed": 24,  # 设置随机数种子
        "config_list": 
                [{"model": "gpt-4o-mini", 
                  "api_key": os.environ.get("OPENAI_API_KEY")}]},
)

# 调用代理生成回复
reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "请你介绍一下什么是大模型？"
        }
    ]
)

# 打印生成的回复
print(reply)
```

![{A482B0A5-036D-4E68-BE7F-A407DECD059A}](E:\autogen_learning\assests\{A482B0A5-036D-4E68-BE7F-A407DECD059A}.png)

可以看到自动在当前目录下创建了 .cacahe 文件夹里面保存了 以seed为名的缓存

&emsp;&emsp;当然，如果有添加缓存的需求，除了默认的在磁盘中，也可以使用`RedisCache`或 `Cosmos DB Cache` 轻松配置大模型缓存，比如`RedisCache`的示例代码为：

```python
agent = ConversableAgent(
    name="chatbot",
    llm_config={
        "config_list": 
                [{"model": "gpt-4o-mini", 
                  "api_key": os.environ.get("OPENAI_API_KEY")}]},
)


with Cache.redis(redis_url="redis://localhost:6379/0") as cache:
    # 使用 Redis 缓存
    reply = agent.generate_reply(
        messages=[{"role": "user", "content": "请问什么是大模型"}],
        cache=cache
    )
```

&emsp;&emsp;通过使用缓存，`AutoGen` 可以在相同的输入下直接返回之前的响应结果，而无需再次调用底层的语言模型服务。既能提高响应速度，同时还可以减少对外部 `API` 的调用次数，从而降低使用成本。在重复查询、开发、测试Agent业务阶段以及在多代理系统中缓存共享的上下文信息等场景中均有实际的使用价值。



#### 本地开源模型

&emsp;&emsp;如果大家想使用本地的开源大模型应用`AutoGen`框架构建`Agent`应用程序，因为`AutoGen`可以支持兼容 `OpenAI API` 的模型接入，***\*主要通过三种方式支持开源模型的接入，分别是：****`LiteLLM`****、****`Ollama`****和****`vLLM`****。\**** 其中`Ollama`和`vLLM`我们课程中已经有过重点的讲解，这里就不再重复性的说明，而`LiteLLM`类似于`Ollama`也是一个轻量级的大模型推理框架，会对大模型进行剪枝、量化和蒸馏等技术的应用，从而减少模型的体积和计算需求，同时保持其较好的推理能力，目的主要是在不牺牲性能的情况下，减少内存使用和计算资源的消耗。

同时可以使用 `OpenAI` 格式调用所有`LLM API` [Bedrock、Huggingface、VertexAI、TogetherAI、Azure、OpenAI、Groq 等] 。

LiteLLM Github：https://github.com/BerriAI/litellm

![{29CAF6D9-E621-4B83-9C95-DFCB7D21729D}](E:\autogen_learning\assests\{29CAF6D9-E621-4B83-9C95-DFCB7D21729D}.png)

&emsp;&emsp;最方便快捷的一种接入本地开源模型的方法就是使用`Ollama`。 该框架提供了与 `OpenAI API` 的兼容性，使得通过 `Ollama` 启动的开源模型可以与支持 `OpenAI API` 的应用程序进行集成。&emsp;&emsp;这里我们使用`Ollama`启动的`Qwen2.5：32B`进行接入，启动本地的`Ollama`模型后，我们只需要将 `API` 请求的主机名更改为 `https://127.0.0.1:11434`， 即可通过本地运行的 `Ollama` 实例与这些模型进行交互。代码如下所示：

```
import os

from autogen import ConversableAgent

agent = ConversableAgent(
    name="ollama_chatbot",
    llm_config={"config_list": 
                [{"model": "qwen2.5:32b",
                  "base_url": "http://192.168.110.131:11434/v1/"}]},
)

# 调用代理生成回复
reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "你好，请你详细地介绍一下你自己啊",
        }
    ]
)

# 打印生成的回复
print(reply)
```

&emsp;&emsp;同样，无论使用什么模型作为`Agent`的底层驱动，生成对输入问题的答复都统一使用 `generate_reply`方法。



&emsp;&emsp;可以看到：使用 `AutoGen` 集成 `Ollama` 模型时，虽然已成功接收到回复，但会出现以下警告：

> [!WARNING]
>
> [autogen.oai.client: 02-26 14:45:09] {351} WARNING - Model qwen2.5:3b is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
> 你好，我是来自阿里云的大规模语言模型，我叫通义千问。我会回答各种问题和完成包括生成文章、撰写报告、回答问答、创作音乐等在内的多种任务。
>
> 我在多个领域积累了知识，并且能够处理不同种类的任务。下面我对我的一些主要功能进行了一些介绍：
>
> 1. 知识查询：我能回答大量领域的专业问题，如科技、文化、历史、艺术、心理和健康等。
> 2. 写作辅助：我可以生成文章、撰写故事开始或结束段落以及描述场景，也可以撰写邮件、报告甚至是软件代码。
> 3. 聊天话题：我能够提供对话信息、发表见解和进行游戏等。
> 4. 对话模拟：我能帮助完成客户服务任务，例如解答关于产品、价格的常见问题；也可用作商务谈判中的助手或在客服代表与客户之间的桥梁角色。 
>
> 总的来说，我是阿里云开发的一台非常智能且强大的机器学习系统。作为一个基于大规模语言数据训练的语言模型，我能够理解和交流多种形式的问题和信息，以便为用户提供更好的服务体验。

&emsp;&emsp;出现此警告表示 `AutoGen` 未找到名为 `qwen2.5:32b` 的模型，因此无法计算相应的费用。如果想避免，我们可以在 `config_list` 中添加 `price` 字段，以指定每 1000 个提示（prompt）和完成（completion）标记的费用。注意，这里我们添加了不使用`LLM Cache`的配置。

```
agent = ConversableAgent(
    name="ollama_chatbot",
    llm_config={
        "cache_seed": None,  # 禁用缓存
        "config_list": [
            {
                "model": "qwen2.5:3b",
                "base_url": "http://localhost:11434/v1/",
                "price": [0.00, 0.00]
            }
        ]
    },
)
```



\- ***\*system_message 参数详解\****

&emsp;&emsp;接下来我们可以测试下`ConversableAgent`类初始化参数中的`system_message`，该参数用于为代理提供上下文或设定特定的行为和规则。其默认设置如下:

```
system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant."
```

&emsp;&emsp;从代码的定义中，`system_message` 默认值是 "You are a helpful AI Assistant."，也就是说，如果没有给 `system_message` 传递其他值，它会自动使用这个默认的字符串。在很多场景`system_message` 用于指导模型如何与用户交互、设定模型的角色、语言风格或行为限制等。为大模型提供系统级的指令或上下文信息，通常在聊天开始时就设定。这些消息的作用是：

\- 设定模型的角色：比如你可以告诉模型它是一个客服、技术支持或产品顾问等。

\- 限定模型行为：例如，告诉模型它只能回答某些问题、需要使用正式语言或者避免某些话题。

\- 指定响应的风格或格式：如要求回答简洁、详细，或者以某种特定的方式呈现。

&emsp;&emsp;下面是一个基于法律咨询助手的 `system_message` 示例，用于指导大模型在法律相关问题上的行为，代码如下：

```
agent = ConversableAgent(
    name="lawyer_assistant",
    # 添加系统信息
    system_message = """
    你是一个法律咨询助手，专注于提供法律相关的咨询服务。你的任务是为用户解答法律问题，但请注意，你的回答仅供参考，不能作为正式的法律意见。
    回答应当基于法律条文和普遍的法律原则，避免提供任何违法或误导性的信息。你需要：
    1. 提供清晰、简洁、准确的法律知识。
    2. 在回答中避免使用非专业术语，尽量让普通用户易于理解。
    3. 如果问题涉及具体案件，建议用户寻求专业律师的帮助。
    4. 尊重用户隐私，不涉及任何个人数据的收集或存储。
    """,
    
    llm_config={
        "cache_seed": None,  # 禁用缓存
        "config_list": [
            {
                "model": "qwen2.5:3b",
                "base_url": "http://localhost:11434/v1/",
                "price": [0.00, 0.00]
            }
        ]
    },
)

# 调用代理生成回复
reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "如果我遇到合同纠纷，应该如何维权？",
        }
    ]
)

# 打印生成的回复
print(reply)
```

&emsp;&emsp;在很多应用中，直接让大模型从头开始理解其角色和行为约束会很低效，尤其是当大模型的默认行为可能与实际需求不符时。使用 `system_message`可以清晰地告诉大模型它的角色和需要遵循的行为规范，使得大模型的行为更加符合预期，这在`Agent`的构建过程中是非常关键的。



\- ***\*description 参数详解\****

&emsp;&emsp;与`system messages`类似，在`ConversableAgent`中 还有一个 `description` 可选参数用来描述代理的角色和行为。



```
agent.name

lawyer_assistant

agent.system_message
    你是一个法律咨询助手，专注于提供法律相关的咨询服务。你的任务是为用户解答法律问题，但请注意，你的回答仅供参考，不能作为正式的法律意见。
    回答应当基于法律条文和普遍的法律原则，避免提供任何违法或误导性的信息。你需要：
    1. 提供清晰、简洁、准确的法律知识。
    2. 在回答中避免使用非专业术语，尽量让普通用户易于理解。
    3. 如果问题涉及具体案件，建议用户寻求专业律师的帮助。
    4. 尊重用户隐私，不涉及任何个人数据的收集或存储。

agent.description
    你是一个法律咨询助手，专注于提供法律相关的咨询服务。你的任务是为用户解答法律问题，但请注意，你的回答仅供参考，不能作为正式的法律意见。
    回答应当基于法律条文和普遍的法律原则，避免提供任何违法或误导性的信息。你需要：
    1. 提供清晰、简洁、准确的法律知识。
    2. 在回答中避免使用非专业术语，尽量让普通用户易于理解。
    3. 如果问题涉及具体案件，建议用户寻求专业律师的帮助。
    4. 尊重用户隐私，不涉及任何个人数据的收集或存储。
```



&emsp;&emsp;默认情况下，`description` 会自动继承 `system_message` 的内容，其源码定义位置如下：

![{960E8998-FB9C-47E0-B3A9-335EFE8CE9BD}](E:\autogen_learning\assests\{960E8998-FB9C-47E0-B3A9-335EFE8CE9BD}.png)



&emsp;&emsp;而如果我们希望为代理提供更具体的信息，而不仅仅依赖 `system_message`，可以显式设置 description，比如：

```
agent = ConversableAgent(
    name="lawyer_assistant",
    # 添加系统信息
    system_message = "你是一个法律咨询助手，专注于提供法律相关的咨询服务",

    # 添加对代理的更具体描述
    description="专门解答有关合同、诉讼等方面的法律问题。",  # 提供更加具体的描述

    llm_config={
        "cache_seed": None,  # 禁用缓存
        "config_list": [
            {
                "model": "qwen2.5:3b",
                "base_url": "http://localhost:11434/v1/",
                "price": [0.00, 0.00]
            }
        ]
    },
)

# 调用代理生成回复
reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "如果我遇到合同纠纷，应该如何维权？",
        }
    ]
)

# 打印生成的回复
print(reply)
```



&emsp;&emsp;那么何时使用默认的 `description` ，什么时候该自定义 `description`呢？ 大家可以从以下几个方面来考虑：

1. 如果希望提供更具体或更详细的信息，或者 `system_message` 的内容并不能完全概括代理的角色时，可以使用 `description` 来传递更多的上下文。`description` 可以更简洁、精确地描述该代理的具体用途或目标。例如，如果某个代理的任务是执行特定类型的计算或查询，`description` 可以是类似“执行代码分析和计算”这样的描述。
2. `description` 在多代理架构中用来帮助其他代理了解当前代理的用途和角色。这不仅仅是为了向用户展示信息，更主要的是提供给其他代理一种上下文，用于合理地调用或路由任务。例如，在一个多代理系统中，如果多个代理有不同的任务和功能（比如一个是法律咨询代理，一个是技术支持代理），那么 `description` 可以帮助其他代理理解该代理的具体用途，便于做出正确的任务调度决策。
3. 指导函数调用行为：在函数调用和任务调度中，`description` 通常是决定哪个代理应当接管某个任务的依据。如果系统需要将任务分配给一个合适的代理，它会查看代理的 `description` 来判断是否符合任务要求。例如，如果任务是法律相关的咨询，系统可能会根据代理的 `description` 选择一个具有法律咨询功能的代理进行处理。例如：法律咨询代理的 `description` 示例为 法律咨询助手，专门解答有关合同、诉讼等方面的法律问题， 技术支持代理的 `description` 示例为 技术支持助手，专门解答关于硬件和软件故障排查的问题等等。



#### LLM配置过滤方法

&emsp;&emsp;接下来我们要考虑的是，`AutoGen`中的`llm_config`为什么要设计成一个列表？当`llm_config`是一个列表的时候，意味着我们定义代理的时候可以使用的多个模型。这在构建`Agent`的过程中非常有用，主要原因如下：

\- 如果一个大模型超时或失败，代理可以尝试另一种模型。

\- 有一个全局模型列表，可以根据某些键（例如名称、标签）对其进行过滤，以便将选择的大模型传递给某个代理（例如，使用更便宜的 GPT 3.5 来让代理解决更简单的任务）

&emsp;&emsp;`config_list`中的工作原理是默认使用配置的第一个大模型，并针对该大模型进行调用。如果调用失败（例如 API 限制），代理将针对第二个大模型发起重试请求，依此类推，直到收到提示完成（或者如果没有大模型成功完成请求，则抛出错误）。因此，我们可以通过下面的形式进行定义：

```
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "tags": ["openai"]
        },
        {
            "model": "qwen2.5:3b",
            "base_url": "http://localhost:11434/v1/",
            "price": [0.00, 0.00],
            "tags": ["ollama"]
        }
    ]
}
```

&emsp;&emsp;对于大模型实例的字典，我们在使用的时候可以基于某些标准来过滤该列表。如上所示，使用 `tags`参数来为不同的代理分配特定的大模型实例。通过在 `llm_config` 的 `config_list` 中为每个模型配置添加 `tags`，然后在创建代理时使用 `filter_config`方法就可以进行筛选：

![{9C14AA41-D2C2-48D3-8F62-FA611F82E653}](E:\autogen_learning\assests\{9C14AA41-D2C2-48D3-8F62-FA611F82E653}.png)

```python
import autogen

# 过滤出包含 'ollama' 标签的模型配置
filter_model = {"tags": ["ollama"]}

config_model = autogen.filter_config(
    config_list=llm_config["config_list"], 
    filter_dict=filter_model)
    
agent = ConversableAgent(
    name="ollama_chatbot",
    llm_config={"config_list": config_model}  # 这里使用 config_model
)

reply = agent.generate_reply(messages=[{"role": "user", "content": "请问你是什么大模型呀",}])
print(reply)
```

&emsp;&emsp;掌握`AutoGen`框架中模型过滤的技巧非常关键，这种配置方式提供了非常便捷的灵活性，能够根据具体需求选择和定制模型。在实际应用中，合理配置 `llm_config` 可以实现多代理协作、自动代码生成、复杂任务处理等功能。在开发过程中对于构建高效、智能的对话系统和自动化工作流都有非常广泛的使用场景和开发需求。



### CodeExcutor
