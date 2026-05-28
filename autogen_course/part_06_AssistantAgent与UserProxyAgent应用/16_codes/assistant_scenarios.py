"""
AutoGen AssistantAgent 典型应用场景配置

本文件展示 AssistantAgent 在不同应用场景下的配置方式，包括：
1. 法律咨询助手
2. 数据分析助手
3. 代码审查助手
4. 学习辅导助手
5. 研究助理
6. 自动化工作流代理

每个场景都包含完整的配置模板和关键参数说明。

运行本文件需要：
- pip install pyautogen
- 设置 OPENAI_API_KEY 环境变量
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# ============================================================
# 第一部分：场景配置模板定义
# ============================================================

@dataclass
class ScenarioConfig:
    """
    场景配置数据类

    用于标准化管理不同应用场景的配置参数
    """
    name: str                           # Agent 名称
    description: str                     # 场景描述
    system_message: str                 # 系统消息
    model: str                           # 使用的模型
    temperature: float                   # 温度参数
    code_executor_enabled: bool         # 是否启用代码执行器
    use_docker: bool                    # 是否使用 Docker
    max_tokens: Optional[int] = None    # 最大 token 数
    timeout: int = 120                  # 超时时间（秒）

    def to_llm_config(self) -> Dict:
        """转换为 AutoGen 的 llm_config 格式"""
        config = {
            "model": self.model,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": self.temperature,
        }
        if self.max_tokens:
            config["max_tokens"] = self.max_tokens
        return config

    def to_agent_config(self) -> Dict:
        """转换为完整的 Agent 配置"""
        config = {
            "name": self.name,
            "system_message": self.system_message,
            "llm_config": self.to_llm_config(),
        }
        if self.code_executor_enabled:
            config["code_executor"] = {
                "use_docker": self.use_docker,
                "timeout": self.timeout,
            }
        return config


# ============================================================
# 第二部分：法律咨询助手场景
# ============================================================

class LegalAdvisorScenario:
    """
    法律咨询助手场景配置

    适用场景：
    - 法律问题初步咨询
    - 合同条款解读
    - 法律流程说明
    - 权利义务分析

    配置要点：
    - 低 temperature（0.2-0.3）确保准确性和严谨性
    - 明确的免责声明
    - 专业角色定位
    """

    @staticmethod
    def get_config() -> ScenarioConfig:
        """获取法律咨询助手的完整配置"""
        return ScenarioConfig(
            name="legal_advisor",
            description="法律咨询助手 - 提供法律问题初步解答和指导",
            system_message="""
                你是一个专业但友善的法律咨询助手。

                你的专长：
                1. 合同条款解读与分析
                2. 法律流程说明
                3. 权利义务分析
                4. 常见法律问题解答

                重要原则：
                1. 你不是律师，不能提供正式的法律意见
                2. 你的回答仅供参考，不能替代专业法律咨询
                3. 对于复杂或重要的法律问题，请务必建议用户咨询专业律师
                4. 在回答中要明确说明这一点

                回答风格：
                - 专业但易懂，避免过多法律术语
                - 结构清晰，分点说明
                - 必要时提供相关法律条文参考
            """,
            model="gpt-4o",
            temperature=0.3,
            code_executor_enabled=False,  # 法律咨询不需要代码执行
            use_docker=False,
        )

    @staticmethod
    def explain_usage():
        """解释法律咨询助手的使用方式"""
        print("=" * 60)
        print("法律咨询助手使用说明")
        print("=" * 60)
        print("""
        使用示例：

        ```python
        from autogen import AssistantAgent

        config = LegalAdvisorScenario.get_config()

        legal_agent = AssistantAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=config.to_llm_config()
        )

        # 与用户对话
        response = legal_agent.generate_reply([
            {"role": "user", "content": "我想了解一下购房合同的注意事项"}
        ])
        ```

        典型问答示例：
        Q: 购房合同中的定金和订金有什么区别？
        A: [详细解释两者区别及法律后果]

        Q: 公司签合同有什么需要注意的？
        A: [提供公司签约的注意事项清单]
        """)


# ============================================================
# 第三部分：数据分析助手场景
# ============================================================

class DataAnalysisScenario:
    """
    数据分析助手场景配置

    适用场景：
    - 数据清洗和预处理
    - 统计分析和建模
    - 数据可视化
    - 洞察报告生成

    配置要点：
    - 启用代码执行器（需要执行 Python 分析代码）
    - 中低 temperature（0.5左右）保证分析准确性
    - use_docker=False 方便开发调试
    """

    @staticmethod
    def get_config() -> ScenarioConfig:
        """获取数据分析助手的完整配置"""
        return ScenarioConfig(
            name="data_analyst",
            description="数据分析助手 - 帮助用户进行数据处理、分析和可视化",
            system_message="""
                你是一个专业的数据分析助手，擅长使用 Python 进行数据分析。

                你的能力：
                1. 数据清洗：处理缺失值、异常值、重复数据
                2. 数据转换：格式转换、特征工程、数据聚合
                3. 统计分析：描述统计、相关性分析、假设检验
                4. 数据可视化：生成各种图表（折线图、柱状图、散点图等）
                5. 报告撰写：总结分析结果，提供业务建议

                工作流程：
                1. 理解用户的数据分析需求
                2. 编写 Python 代码（使用 pandas、numpy、matplotlib 等库）
                3. 执行代码并展示结果
                4. 解释分析结果并给出建议

                注意事项：
                - 代码要清晰注释，便于用户理解
                - 图表要标注清楚，包括标题、轴标签、图例
                - 分析结果要结合业务场景解释
            """,
            model="gpt-4o",
            temperature=0.5,
            code_executor_enabled=True,
            use_docker=False,  # 开发环境使用本地执行
            timeout=180,        # 数据分析可能耗时较长
        )

    @staticmethod
    def example_analysis_task():
        """展示数据分析的示例任务"""
        print("=" * 60)
        print("数据分析助手示例任务")
        print("=" * 60)
        print("""
        示例1：销售数据分析

        用户：分析一下我们公司近一年的销售数据

        Agent 响应：
        1. 编写 Python 代码读取销售数据
        2. 进行数据清洗（处理缺失值）
        3. 计算关键指标（月销售额、同比增长、环比增长）
        4. 生成可视化图表
        5. 总结洞察和建议

        示例2：用户行为分析

        用户：分析一下App用户的使用习惯

        Agent 响应：
        1. 读取用户行为日志
        2. 分析用户活跃度、留存率、转化率
        3. 生成用户分群
        4. 可视化关键指标
        5. 提出产品优化建议
        """)


# ============================================================
# 第四部分：代码审查助手场景
# ============================================================

class CodeReviewScenario:
    """
    代码审查助手场景配置

    适用场景：
    - 代码质量审查
    - Bug 发现与修复建议
    - 性能优化建议
    - 安全漏洞检测
    - 代码风格检查

    配置要点：
    - 低 temperature（0.2-0.3）确保审查严谨
    - 可选代码执行（用于运行测试验证）
    - 强调专业性和建设性反馈
    """

    @staticmethod
    def get_config() -> ScenarioConfig:
        """获取代码审查助手的完整配置"""
        return ScenarioConfig(
            name="code_reviewer",
            description="代码审查助手 - 帮你审查代码质量并提供改进建议",
            system_message="""
                你是一个资深的代码审查专家，有多年的软件开发经验。

                你的审查维度：
                1. 功能正确性：代码是否实现了需求功能
                2. 代码质量：可读性、可维护性、复杂度
                3. 性能效率：算法复杂度、资源使用、扩展性
                4. 安全漏洞：注入风险、敏感信息处理、认证授权
                5. 最佳实践：设计模式、编码规范、测试覆盖

                审查原则：
                - 建设性反馈：指出问题的同时提供改进建议
                - 具体可操作：每个问题都要有具体的修复方案
                - 分优先级：区分关键问题和次要建议
                - 解释原因：说明为什么要这样改

                输出格式：
                - 问题描述
                - 严重程度（高/中/低）
                - 具体位置（文件、行号）
                - 改进建议
                - 示例代码（改进后）
            """,
            model="gpt-4o",
            temperature=0.3,
            code_executor_enabled=True,  # 可以运行测试验证
            use_docker=True,             # 代码执行建议使用 Docker
            timeout=120,
        )

    @staticmethod
    def example_review():
        """展示代码审查的示例"""
        print("=" * 60)
        print("代码审查助手示例")
        print("=" * 60)
        print("""
        示例：审查 Python 代码

        用户代码：
        ```python
        def get_user(user_id):
            user = db.query("SELECT * FROM users WHERE id = " + user_id)
            return user
        ```

        Agent 审查结果：
        ═══════════════════════════════════════════════════
        高优先级问题：SQL 注入漏洞
        ═══════════════════════════════════════════════════
        文件: app.py, 第 3 行

        问题描述：
        直接拼接用户输入到 SQL 查询中，存在严重的 SQL 注入风险。
        攻击者可以通过 user_id 参数执行恶意 SQL 语句。

        修复建议：
        使用参数化查询（Prepared Statements）

        改进代码：
        ```python
        def get_user(user_id):
            stmt = "SELECT * FROM users WHERE id = %s"
            user = db.query(stmt, (user_id,))
            return user
        ```
        ═══════════════════════════════════════════════════

        中优先级问题：缺少错误处理
        ═══════════════════════════════════════════════════
        问题：没有处理数据库查询可能失败的情况

        修复建议：添加 try-except 块
        ```python
        def get_user(user_id):
            try:
                stmt = "SELECT * FROM users WHERE id = %s"
                user = db.query(stmt, (user_id,))
                return user
            except db.Error as e:
                logging.error(f"Database error: {e}")
                return None
        ```
        """)


# ============================================================
# 第五部分：学习辅导助手场景
# ============================================================

class LearningTutorScenario:
    """
    学习辅导助手场景配置

    适用场景：
    - 学科问题解答
    - 概念解释
    - 学习计划制定
    - 练习题生成
    - 知识点复习

    配置要点：
    - 较高 temperature（0.7-0.8）支持创意教学
    - 可以使用更便宜的 mini 模型降低成本
    - 强调耐心和鼓励
    """

    @staticmethod
    def get_config() -> ScenarioConfig:
        """获取学习辅导助手的完整配置"""
        return ScenarioConfig(
            name="learning_tutor",
            description="学习辅导助手 - 帮助学生学习、解答问题和制定学习计划",
            system_message="""
                你是一个耐心、友好的学习辅导老师。

                你的教学理念：
                1. 循序渐进：由浅入深，先打基础再深入
                2. 多样化表达：用文字、图表、例子、类比多种方式讲解
                3. 鼓励提问：没有愚蠢的问题，鼓励学生多问
                4. 即时反馈：及时纠正错误，给予肯定

                你的能力：
                1. 学科辅导：数学、科学、编程、语言等
                2. 概念解释：用简单易懂的方式解释复杂概念
                3. 学习规划：帮助制定学习计划和目标
                4. 练习生成：根据学生水平生成合适的练习题
                5. 复习助手：帮助回顾和巩固知识点

                注意事项：
                - 不要假设学生理解所有背景知识
                - 遇到不确定的问题，诚实告知学生
                - 多用例子和实践来帮助理解
                - 定期检查学生是否真正理解了
            """,
            model="gpt-4o-mini",  # 使用 mini 模型降低成本
            temperature=0.8,
            code_executor_enabled=True,  # 可以执行代码示例
            use_docker=False,
            timeout=60,
        )

    @staticmethod
    def example_teaching():
        """展示学习辅导的示例"""
        print("=" * 60)
        print("学习辅导助手示例")
        print("=" * 60)
        print("""
        示例：教授 Python 函数概念

        学生问：什么是函数？为什么要有函数？

        Agent 响应：

        好问题！让我用一个生活中的例子来解释。

        【类比】想象一下做饭：
        - 你每天要做很多次饭，如果每次都从头开始洗米、加水、煮饭...
        - 那多麻烦啊！

        所以我们发明了"电饭煲"：
        - 把洗好的米和水放进去
        - 按下按钮
        - 等一段时间后，饭就熟了

        这个"电饭煲"就是一个"函数"！
        它把一组操作打包起来，需要的时候一键调用就行。

        【编程中的函数】

        不使用函数（重复代码）：
        ```python
        # 计算两个数的和
        result1 = 10 + 20
        print(result1)

        result2 = 30 + 40
        print(result2)
        ```

        使用函数：
        ```python
        def add(a, b):
            return a + b

        print(add(10, 20))  # 调用函数
        print(add(30, 40))  # 再次调用
        ```

        函数的优点：
        ✓ 代码不重复：写一次，用多次
        ✓ 容易维护：改一处，所有地方都生效
        ✓ 容易测试：可以单独测试每个函数
        ✓ 可读性高：函数名说明它在做什么

        想不想试试写一个你自己的函数？
        """)


# ============================================================
# 第六部分：研究助理场景
# ============================================================

class ResearchAssistantScenario:
    """
    研究助理场景配置

    适用场景：
    - 文献检索与整理
    - 研究方法建议
    - 数据分析
    - 报告撰写
    - 参考文献管理

    配置要点：
    - 中高 temperature（0.6-0.7）支持综合分析
    - 可配合搜索工具使用
    - 强调信息准确性和引用规范
    """

    @staticmethod
    def get_config() -> ScenarioConfig:
        """获取研究助理的完整配置"""
        return ScenarioConfig(
            name="research_assistant",
            description="研究助理 - 帮助进行学术研究和文献整理",
            system_message="""
                你是一个专业的研究助理，擅长学术研究工作。

                你的研究能力：
                1. 文献检索：帮助查找相关论文和资料
                2. 文献整理：总结论文要点、整理参考文献
                3. 研究设计：协助设计研究方案和方法论
                4. 数据分析：协助进行数据处理和统计分析
                5. 报告撰写：帮助撰写研究报告和论文

                研究原则：
                1. 准确性：确保信息的准确性，注明来源
                2. 客观性：呈现多方观点，避免偏见
                3. 完整性：全面考虑问题的各个维度
                4. 创新性：在已有研究基础上提出新见解

                学术规范：
                - 引用要规范，注明作者、年份、期刊
                - 区分直接引用和转述
                - 标注研究方法的局限性
                - 遵守学术道德

                注意事项：
                - 对于复杂的研究问题，建议咨询专家
                - 不要编造不存在的文献或数据
                - 保持研究的严谨性
            """,
            model="gpt-4o",
            temperature=0.6,
            code_executor_enabled=True,
            use_docker=False,
            max_tokens=4096,  # 研究可能需要更长的回复
            timeout=180,
        )

    @staticmethod
    def example_research():
        """展示研究助理的示例"""
        print("=" * 60)
        print("研究助理示例")
        print("=" * 60)
        print("""
        示例：协助文献综述

        用户：我想了解人工智能在医疗诊断中的应用研究现状

        Agent 响应：

        好的，我来帮你梳理这个领域的研究现状。

        一、主要研究方向

        1. 医学影像诊断
           - 应用：X光、CT、MRI 等影像分析
           - 代表性研究：DeepMind 的眼底疾病检测
           - 主要技术：CNN、Transformer

        2. 疾病预测与风险评估
           - 应用：癌症早筛、心血管风险预测
           - 主要技术：生存分析、机器学习

        3. 药物研发
           - 应用：靶点发现、药物设计
           - 主要技术：生成模型、强化学习

        二、关键论文推荐

        1. "Deep learning for medical imaging" (Nature, 2017)
           - 作者：Esteva et al.
           - 贡献：展示了深度学习在皮肤癌诊断上的潜力

        2. "A deep learning system for breast cancer screening" (Nature, 2020)
           - 作者：McKinney et al.
           - 贡献：AI系统在乳腺癌筛查上超越人类放射科医生

        三、研究挑战

        1. 数据隐私和伦理问题
        2. 模型可解释性
        3. 数据质量和标注问题
        4. 临床验证的复杂性

        四、未来方向

        1. 多模态融合（影像 + 病历 + 基因）
        2. 联邦学习保护隐私
        3. 可解释AI在医疗的应用

        需要我深入讲解某个方向吗？或者帮你搜索最新的论文？
        """)


# ============================================================
# 第七部分：自动化工作流代理场景
# ============================================================

class WorkflowAutomationScenario:
    """
    自动化工作流代理场景配置

    适用场景：
    - 业务流程自动化
    - 任务编排和执行
    - 多步骤数据处理
    - 定时任务执行

    配置要点：
    - 低 temperature（0.2-0.4）确保任务执行的确定性
    - 启用代码执行器
    - use_docker=True 确保安全性
    - 强调错误处理和日志记录
    """

    @staticmethod
    def get_config() -> ScenarioConfig:
        """获取自动化工作流代理的完整配置"""
        return ScenarioConfig(
            name="workflow_automation",
            description="自动化工作流代理 - 自动执行多步骤业务流程",
            system_message="""
                你是一个专业的自动化工作流代理。

                你的职责：
                1. 任务理解：准确理解用户要完成的自动化任务
                2. 步骤规划：将大任务分解为可执行的小步骤
                3. 执行监控：执行过程中监控状态，处理异常
                4. 结果汇报：清晰汇报任务执行结果和日志

                设计原则：
                1. 幂等性：同一任务可以安全地重复执行
                2. 可恢复：失败后可以从断点恢复
                3. 可观测：每个步骤都有清晰的日志记录
                4. 错误处理：每个步骤都要有错误处理逻辑

                工作流程：
                1. 接收任务描述
                2. 分析任务需求
                3. 编写自动化脚本
                4. 执行并监控
                5. 报告结果

                注意事项：
                - 重要操作前先确认用户
                - 保留执行日志便于排查问题
                - 设置合理的超时时间
                - 关键节点要有人工审批
            """,
            model="gpt-4o",
            temperature=0.3,
            code_executor_enabled=True,
            use_docker=True,  # 生产环境使用 Docker
            timeout=300,      # 复杂工作流可能需要更长时间
        )

    @staticmethod
    def example_workflow():
        """展示自动化工作流的示例"""
        print("=" * 60)
        print("自动化工作流代理示例")
        print("=" * 60)
        print("""
        示例：每日数据报表生成

        用户：帮我创建一个每天早上9点自动生成销售报表的工作流

        Agent 响应：

        好的，我来帮你设计这个自动化工作流。

        工作流设计：
        ╔═══════════════════════════════════════════════════╗
        ║  Step 1: 数据提取                                ║
        ║  - 连接数据库读取昨日销售数据                      ║
        ║  - 导出为 CSV 格式                                ║
        ║  - 记录数据量：X 条记录                          ║
        ╠═══════════════════════════════════════════════════╣
        ║  Step 2: 数据清洗                                ║
        ║  - 处理缺失值和异常值                             ║
        ║  - 计算关键指标                                   ║
        ╠═══════════════════════════════════════════════════╣
        ║  Step 3: 报表生成                                ║
        ║  - 生成销售汇总图表                               ║
        ║  - 生成同比环比分析                               ║
        ║  - 导出 PDF 格式                                  ║
        ╠═══════════════════════════════════════════════════╣
        ║  Step 4: 邮件发送                                ║
        ║  - 发送报表给相关人员                             ║
        ║  - 记录发送状态                                   ║
        ╚═══════════════════════════════════════════════════╝

        执行日志示例：
        [2024-01-20 08:59:00] 工作流启动
        [2024-01-20 08:59:01] Step 1 完成：提取数据 1,234 条
        [2024-01-20 08:59:15] Step 2 完成：清洗后有效数据 1,230 条
        [2024-01-20 08:59:45] Step 3 完成：生成报表 3 页
        [2024-01-20 08:59:58] Step 4 完成：邮件已发送
        [2024-01-20 09:00:00] 工作流完成

        需要我帮你部署这个工作流吗？
        """)


# ============================================================
# 第八部分：配置使用示例
# ============================================================

class UsageExamples:
    """
    展示如何使用配置创建 Agent
    """

    @staticmethod
    def show_basic_usage():
        """展示基础使用方式"""
        print("\n" + "=" * 60)
        print("基础使用方式")
        print("=" * 60)
        print("""
        方式1：使用 ScenarioConfig 类

        ```python
        from autogen import AssistantAgent

        # 获取配置
        config = DataAnalysisScenario.get_config()

        # 创建 Agent
        agent = AssistantAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=config.to_llm_config(),
            code_executor={"use_docker": config.use_docker}
        )
        ```

        方式2：直接使用字典配置

        ```python
        agent = AssistantAgent(
            name="my_agent",
            system_message="你是一个专业的数据分析师...",
            llm_config={
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.5,
            }
        )
        ```

        方式3：自定义配置模板

        ```python
        def create_custom_agent(name, role, temperature=0.7):
            return AssistantAgent(
                name=name,
                system_message=f"你是一个{role}专家...",
                llm_config={
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "temperature": temperature,
                }
            )

        # 使用
        analyst = create_custom_agent("analyst", "数据分析")
        legal = create_custom_agent("legal", "法律咨询", temperature=0.3)
        ```
        """)

    @staticmethod
    def show_advanced_usage():
        """展示高级使用方式"""
        print("\n" + "=" * 60)
        print("高级使用方式")
        print("=" * 60)
        print("""
        高级配置1：自定义代码执行器

        ```python
        from autogen import AssistantAgent, CodeExecutor

        # 创建自定义代码执行器
        code_executor = CodeExecutor(
            use_docker=True,
            timeout=300,
            work_dir="./workspace"
        )

        # 创建带自定义执行器的 Agent
        agent = AssistantAgent(
            name="data_analyst",
            system_message="你是一个数据分析专家...",
            llm_config=llm_config,
            code_executor=code_executor
        )
        ```

        高级配置2：注册自定义工具

        ```python
        agent = AssistantAgent(
            name="assistant",
            llm_config=llm_config
        )

        # 注册自定义函数
        def query_database(sql: str) -> dict:
            '''查询数据库'''
            # 执行 SQL 查询
            return {"result": "查询结果"}

        agent.register_function(query_database)
        ```

        高级配置3：多 Agent 协作

        ```python
        from autogen import AssistantAgent, UserProxyAgent

        # 创建助手 Agent
        assistant = AssistantAgent(
            name="assistant",
            llm_config=llm_config
        )

        # 创建用户代理（执行代码）
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",  # 完全自动化
            code_executor={"use_docker": False}
        )

        # 启动对话
        user_proxy.initiate_chat(
            assistant,
            message="帮我分析一下这个CSV文件"
        )
        ```
        """)


# ============================================================
# 第九部分：场景选择指南
# ============================================================

class ScenarioSelectionGuide:
    """
    帮助用户选择合适的场景配置
    """

    @staticmethod
    def show_decision_tree():
        """展示场景选择的决策树"""
        print("\n" + "=" * 60)
        print("场景选择决策树")
        print("=" * 60)
        print("""
        问题1：需要代码执行能力吗？

        ├─ 是 → 问题2：需要什么级别的安全性？
        │         ├─ 高（生产环境）→ 使用 Docker → 代码审查/数据分析
        │         └─ 低（开发环境）→ 本地执行 → 学习辅导/工作流自动化
        │
        └─ 否 → 问题3：需要多正式的语气？

                ├─ 正式严谨 → 法律咨询/研究助理
                └─ 友好亲切 → 学习辅导

        ─────────────────────────────────────────────────

        快速选择指南：

        场景                    │ 推荐模型    │ Temperature │ 代码执行
        ────────────────────────┼─────────────┼─────────────┼─────────
        法律咨询                │ gpt-4o      │ 0.3         │ 否
        数据分析                │ gpt-4o      │ 0.5         │ 是（本地）
        代码审查                │ gpt-4o      │ 0.3         │ 是（Docker）
        学习辅导                │ gpt-4o-mini │ 0.8         │ 是（本地）
        研究助理                │ gpt-4o      │ 0.6         │ 是（本地）
        工作流自动化            │ gpt-4o      │ 0.3         │ 是（Docker）
        """)

    @staticmethod
    def show_comparison():
        """展示各场景配置对比"""
        print("\n" + "=" * 60)
        print("场景配置对比表")
        print("=" * 60)

        scenarios = [
            LegalAdvisorScenario.get_config(),
            DataAnalysisScenario.get_config(),
            CodeReviewScenario.get_config(),
            LearningTutorScenario.get_config(),
            ResearchAssistantScenario.get_config(),
            WorkflowAutomationScenario.get_config(),
        ]

        print(f"\n{'场景':<15} {'模型':<12} {'温度':<8} {'代码执行':<10} {'Docker'}")
        print("-" * 60)

        for s in scenarios:
            code_exec = "是" if s.code_executor_enabled else "否"
            docker = "是" if s.use_docker else "否"
            print(f"{s.name:<15} {s.model:<12} {s.temperature:<8} {code_exec:<10} {docker}")

        print("\n说明：")
        print("- 温度：0.2-0.3（严谨），0.4-0.6（平衡），0.7-1.0（创意）")
        print("- 代码执行：根据任务需求选择是否启用")
        print("- Docker：生产环境建议使用，本地开发可用本地执行")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# 第一部分：法律咨询助手场景")
    print("#" * 60)
    LegalAdvisorScenario.explain_usage()
    print("\n配置：")
    config = LegalAdvisorScenario.get_config()
    print(f"名称: {config.name}")
    print(f"模型: {config.model}")
    print(f"温度: {config.temperature}")

    print("\n" + "#" * 60)
    print("# 第二部分：数据分析助手场景")
    print("#" * 60)
    DataAnalysisScenario.example_analysis_task()

    print("\n" + "#" * 60)
    print("# 第三部分：代码审查助手场景")
    print("#" * 60)
    CodeReviewScenario.example_review()

    print("\n" + "#" * 60)
    print("# 第四部分：学习辅导助手场景")
    print("#" * 60)
    LearningTutorScenario.example_teaching()

    print("\n" + "#" * 60)
    print("# 第五部分：研究助理场景")
    print("#" * 60)
    ResearchAssistantScenario.example_research()

    print("\n" + "#" * 60)
    print("# 第六部分：工作流自动化场景")
    print("#" * 60)
    WorkflowAutomationScenario.example_workflow()

    print("\n" + "#" * 60)
    print("# 第七部分：使用示例")
    print("#" * 60)
    UsageExamples.show_basic_usage()
    UsageExamples.show_advanced_usage()

    print("\n" + "#" * 60)
    print("# 第八部分：场景选择指南")
    print("#" * 60)
    ScenarioSelectionGuide.show_decision_tree()
    ScenarioSelectionGuide.show_comparison()

    print("\n" + "=" * 60)
    print("AssistantAgent 典型应用场景演示结束")
    print("=" * 60)
    print("""
    学习要点总结：

    1. 法律咨询场景
       - 低温度（0.3）确保严谨
       - 明确免责声明
       - 不需要代码执行

    2. 数据分析场景
       - 启用代码执行器
       - 中等温度（0.5）
       - 可视化图表生成

    3. 代码审查场景
       - 低温度（0.3）确保严谨
       - Docker 执行保证安全
       - 建设性反馈

    4. 学习辅导场景
       - 高温度（0.8）支持创意教学
       - 友好、耐心的语气
       - 使用 mini 模型降低成本

    5. 研究助理场景
       - 中高温度（0.6）
       - 强调引用规范
       - 完整的信息来源

    6. 工作流自动化
       - 低温度（0.3）确保确定性
       - Docker 执行保证安全
       - 完善的错误处理

    选择建议：
    - 根据任务需求选择合适的场景模板
    - 调整温度参数控制输出风格
    - 根据安全需求决定是否使用 Docker
    - 根据任务复杂度调整超时时间

    下一步：
    - 根据你的业务场景选择合适的配置
    - 自定义 system_message 适配具体需求
    - 尝试创建自己的 Agent 实例
    """)