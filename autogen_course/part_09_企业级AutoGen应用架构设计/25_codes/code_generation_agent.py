#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第25节 代码生成Agent深度示例
=====================================

本文件展示更复杂的代码生成Agent系统，包括：
1. 专门化的代码生成Agent（前端、后端、数据科学等）
2. 智能调试Agent的集成
3. 自动化测试与验证流程
4. 企业级代码生成工作流设计

通过本示例，你将学会如何构建专业的AI代码生成团队。
"""

import os
import sys
import json
import traceback
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 添加项目根目录到Python路径
# ============================================================
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent
except ImportError as e:
    print(f"❌ 请先安装 autogen 包: pip install autogen")
    sys.exit(1)

# ============================================================
# 配置说明
# ============================================================
# 使用o1模型进行代码生成，它在代码生成任务上表现优异

llm_config = {
    "model": "o1",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.5,  # 较低温度，保证代码生成的稳定性
}


# ============================================================
# 第一步：定义专门的代码生成Agent工厂
# ============================================================

@dataclass
class CodeGenAgentConfig:
    """
    代码生成Agent的配置类

    通过配置不同的角色和系统提示，
    我们可以创建专注于不同领域的代码生成专家。
    """
    name: str                          # Agent名称
    specialty: str                     # 专业领域
    tech_stack: List[str] = field(default_factory=list)  # 技术栈
    system_message_template: str = ""  # 系统提示模板
    max_iterations: int = 5            # 最大迭代次数


class CodeGenAgentFactory:
    """
    代码生成Agent工厂类

    负责创建和管理各种专门化的代码生成Agent

    支持的专门化领域：
    - 前端开发 (HTML/CSS/JavaScript/React)
    - 后端开发 (Python/Java/Go/Database)
    - 数据科学 (ML/DL/DataAnalysis)
    - DevOps (CI/CD/Docker/K8s)
    - 移动开发 (iOS/Android)
    """

    # 预定义的Agent模板
    AGENT_TEMPLATES = {
        "frontend": """
你是一位专业的前端开发工程师，精通以下技术：

技术栈：HTML5, CSS3, JavaScript (ES6+), React, Vue.js, TypeScript

你的工作职责：
1. 根据需求描述生成美观、响应式的前端代码
2. 确保代码符合Web标准和无障碍访问要求
3. 使用现代CSS框架（如Tailwind、Bootstrap）提高效率
4. 编写清晰的组件结构和API交互逻辑

输出规范：
- 生成的HTML/CSS/JS代码必须结构清晰
- 包含必要的meta标签和SEO优化
- 使用语义化的HTML标签
- 代码注释使用中文""",

        "backend": """
你是一位经验丰富的后端开发工程师，精通以下技术：

技术栈：Python (FastAPI/Django/Flask), Java (Spring Boot), Go (Gin), Node.js, SQL

你的工作职责：
1. 设计并实现RESTful API接口
2. 编写清晰的数据模型和数据库交互代码
3. 实现业务逻辑层和数据访问层的分离
4. 包含适当的错误处理和日志记录

输出规范：
- 遵循各语言的编码规范（PEP8, Airbnb JS, etc.）
- API设计要符合REST最佳实践
- 代码注释使用中文
- 提供API使用示例""",

        "data_science": """
你是一位专业的数据科学家，精通以下技术：

技术栈：Python, pandas, numpy, scikit-learn, TensorFlow, PyTorch, Jupyter

你的工作职责：
1. 数据处理和分析代码生成
2. 机器学习模型的实现
3. 数据可视化和报告生成
4. 统计分析和实验设计

输出规范：
- 代码必须包含数据处理流程的详细注释
- 模型训练和评估代码要完整
- 可视化代码要清晰易读
- 代码注释使用中文""",

        "devops": """
你是一位资深的DevOps工程师，精通以下技术：

技术栈：Docker, Kubernetes, Jenkins, GitHub Actions, Terraform, Ansible

你的工作职责：
1. 基础设施即代码（IaC）的编写
2. CI/CD流水线配置
3. 容器化应用部署
4. 监控和日志配置

输出规范：
- 配置文件必须结构清晰
- 包含必要的环境变量配置
- 有详细的部署说明
- 代码注释使用中文""",
    }

    @classmethod
    def create_agent(cls, config: CodeGenAgentConfig) -> AssistantAgent:
        """
        根据配置创建代码生成Agent

        Args:
            config: Agent配置对象

        Returns:
            配置好的AssistantAgent实例
        """
        # 获取对应的模板或使用默认模板
        template = cls.AGENT_TEMPLATES.get(
            config.specialty.lower(),
            "你是一位专业的Python开发工程师，负责生成高质量代码。"
        )

        # 添加工具说明和输出格式要求
        system_message = f"""
{template}

额外要求：
1. 代码必须完整、可运行
2. 复杂逻辑必须添加详细注释
3. 包含必要的安全检查和错误处理
4. 提供使用示例和测试方法

工作流程：
1. 分析需求 → 2. 设计方案 → 3. 编写代码 → 4. 自测验证
"""

        agent = AssistantAgent(
            name=config.name,
            system_message=system_message,
            llm_config=llm_config,
        )

        return agent


# ============================================================
# 第二步：定义智能调试Agent
# ============================================================

class IntelligentDebugger:
    """
    智能调试Agent

    功能：
    1. 捕获并分析代码执行错误
    2. 分析错误堆栈，定位问题根源
    3. 生成修复建议和修正后的代码
    4. 验证修复是否成功

    工作原理：
    ┌─────────────┐
    │   捕获错误  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     分析     ┌─────────────┐
    │   错误分析   │ ──────────▶ │  堆栈追踪   │
    └──────┬──────┘              └──────┬──────┘
           │                           │
           ▼                           ▼
    ┌─────────────┐              ┌─────────────┐
    │  根因定位    │              │  修复建议   │
    └──────┬──────┘              └──────┬──────┘
           │                           │
           └──────────┬────────────────┘
                      │
                      ▼
              ┌─────────────┐
              │  生成修复代码│
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  验证修复    │
              └─────────────┘
    """

    def __init__(self):
        """初始化调试Agent"""
        self.debug_agent = AssistantAgent(
            name="智能调试专家",
            system_message="""你是一位专业的代码调试专家，擅长分析和修复程序错误。

            你的工作流程：
            1. 分析错误信息和堆栈追踪
            2. 定位问题根源（不只是表象）
            3. 提出具体的修复方案
            4. 生成修复后的完整代码

            输出格式：
            ## 错误分析
            [详细说明错误原因]

            ## 问题根源
            [定位到的根本原因]

            ## 修复方案
            [具体的修复步骤]

            ## 修正后的代码
            ```python
            [完整的修正代码]
            ```

            注意：修复后的代码必须完整、可运行。""",
            llm_config=llm_config,
        )

        self.fix_verifier = AssistantAgent(
            name="修复验证专家",
            system_message="""你是一位代码质量专家，负责验证修复是否正确。

            验证维度：
            1. 原错误是否已修复
            2. 是否引入新的问题
            3. 代码质量是否达标
            4. 测试用例是否通过

            回复格式：
            - 如果修复正确：✅ 验证通过
            - 如果还需要调整：❌ 需要修正（说明原因）""",
            llm_config=llm_config,
        )

    def debug_and_fix(self, code: str, error: str) -> str:
        """
        调试并修复代码

        Args:
            code: 原始代码
            error: 错误信息

        Returns:
            修复后的代码
        """
        print("\n" + "="*50)
        print("🔧 启动智能调试流程")
        print("="*50)

        # 分析错误
        debug_request = f"""
原始代码：
```python
{code}
```

错误信息：
```
{error}
```

请分析错误原因并提供修复代码。
"""

        # 生成修复建议
        debug_response = self.debug_agent.generate_reply(
            messages=[{"role": "user", "content": debug_request}]
        )

        print("📋 调试分析结果：")
        print(debug_response)

        # 从响应中提取修复后的代码
        fixed_code = self._extract_code_from_response(debug_response)

        if fixed_code:
            # 验证修复
            verify_request = f"""
原始代码：{code}
原始错误：{error}
修复后代码：{fixed_code}

请验证修复是否正确。
"""
            verification = self.fix_verifier.generate_reply(
                messages=[{"role": "user", "content": verify_request}]
            )

            if "通过" in verification:
                print("\n✅ 修复验证通过")
                return fixed_code
            else:
                print(f"\n⚠️ 验证反馈：{verification}")
                # 进入下一轮调试
                return self.debug_and_fix(fixed_code, verification)

        return code  # 返回原始代码如果无法修复

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """从调试响应中提取代码块"""
        import re
        # 匹配 ```python ... ``` 格式的代码块
        pattern = r'```python\s*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return None


# ============================================================
# 第三步：定义自动化测试Agent
# ============================================================

class AutoTestAgent:
    """
    自动化测试Agent

    职责：
    1. 根据代码生成测试用例
    2. 执行测试并收集结果
    3. 生成测试报告
    4. 提供覆盖率分析
    """

    def __init__(self):
        self.test_agent = AssistantAgent(
            name="测试工程师",
            system_message="""你是一位专业的测试工程师，负责生成高质量的测试用例。

            测试策略：
            1. 等价类划分 - 覆盖正常、边界、异常情况
            2. 边界值分析 - 测试边界条件
            3. 决策表测试 - 测试复杂逻辑分支
            4. 场景测试 - 测试实际使用场景

            输出要求：
            - 使用pytest框架
            - 包含setup和teardown
            - 测试函数命名清晰
            - 包含必要的assert断言
            - 代码注释使用中文

            工作流程：
            1. 分析被测代码
            2. 设计测试用例
            3. 编写测试代码
            4. 验证测试可运行""",
            llm_config=llm_config,
        )

    def generate_tests(self, code: str, code_type: str = "python") -> str:
        """
        根据代码生成测试用例

        Args:
            code: 待测试的代码
            code_type: 代码语言类型

        Returns:
            测试代码
        """
        test_request = f"""
请为以下{code_type}代码生成测试用例：

```python
{code}
```

要求：
1. 使用pytest框架
2. 覆盖主要功能和边界情况
3. 测试代码需要包含中文注释
4. 生成完整可运行的测试文件
"""

        response = self.test_agent.generate_reply(
            messages=[{"role": "user", "content": test_request}]
        )

        return response

    def run_tests(self, test_code: str) -> Dict[str, Any]:
        """
        执行测试并返回结果

        Args:
            test_code: 测试代码

        Returns:
            测试结果字典，包含通过率、失败信息等
        """
        # 创建临时测试文件
        test_file = Path("temp_test.py")
        test_file.write_text(test_code, encoding="utf-8")

        try:
            # 执行测试
            import subprocess
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            # 清理临时文件
            if test_file.exists():
                test_file.unlink()


# ============================================================
# 第四步：构建完整的企业级代码生成工作流
# ============================================================

class EnterpriseCodeGenWorkflow:
    """
    企业级代码生成工作流

    这是一个完整的多Agent协作系统，包含：
    1. 需求分析Agent
    2. 架构设计Agent
    3. 代码生成Agent团队
    4. 智能调试Agent
    5. 自动化测试Agent
    6. 代码审查Agent

    工作流程图：
    ┌──────────────────────────────────────────────────────────────┐
    │                      需求输入                                 │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               阶段1：需求分析与架构设计                       │
    │  ┌─────────────┐    ┌─────────────┐                         │
    │  │ 需求分析Agent │ → │ 架构设计Agent│                         │
    │  └─────────────┘    └─────────────┘                         │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               阶段2：代码生成（多Agent并行）                  │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
    │  │  前端Agent   │    │  后端Agent   │    │ 数据科学Agent│       │
    │  └─────────────┘    └─────────────┘    └─────────────┘       │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               阶段3：智能调试                                 │
    │  ┌─────────────────────────────────────────────┐             │
    │  │         IntelligentDebugger                  │             │
    │  │  错误分析 → 根因定位 → 修复方案 → 验证确认   │             │
    │  └─────────────────────────────────────────────┘             │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               阶段4：自动化测试                               │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
    │  │ 测试生成    │ → │ 测试执行    │ → │ 覆盖率分析  │       │
    │  └─────────────┘    └─────────────┘    └─────────────┘       │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               阶段5：代码审查与交付                           │
    │  ┌─────────────────────────────────────────────┐             │
    │  │         代码审查 + 最终验证                   │             │
    │  └─────────────────────────────────────────────┘             │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
                         输出：高质量代码
    """

    def __init__(self):
        """初始化工作流组件"""
        print("\n" + "="*60)
        print("🏢 企业级代码生成工作流初始化")
        print("="*60)

        # 创建专门化的代码生成Agent
        self.frontend_agent = CodeGenAgentFactory.create_agent(
            CodeGenAgentConfig(
                name="前端开发专家",
                specialty="frontend",
                tech_stack=["HTML", "CSS", "JavaScript", "React"],
            )
        )

        self.backend_agent = CodeGenAgentFactory.create_agent(
            CodeGenAgentConfig(
                name="后端开发专家",
                specialty="backend",
                tech_stack=["Python", "FastAPI", "PostgreSQL"],
            )
        )

        self.data_agent = CodeGenAgentFactory.create_agent(
            CodeGenAgentConfig(
                name="数据科学专家",
                specialty="data_science",
                tech_stack=["Python", "pandas", "scikit-learn"],
            )
        )

        # 创建智能调试和测试Agent
        self.debugger = IntelligentDebugger()
        self.tester = AutoTestAgent()

        # 创建用户代理
        self.user_proxy = UserProxyAgent(
            name="工作流协调员",
            human_input_mode="NEVER",
            code_execution_config={"executor": "local", "use_docker": False},
        )

        print("✅ 所有Agent初始化完成\n")

    def analyze_requirement(self, requirement: str) -> Dict[str, Any]:
        """
        需求分析阶段

        分析用户需求，确定需要哪些专门的Agent参与
        """
        print("\n" + "-"*50)
        print("📊 阶段1：需求分析")
        print("-"*50)
        print(f"需求描述：{requirement}")

        # 简单的需求分析逻辑
        # 实际应用中可以用更复杂的NLP方法
        analysis_result = {
            "requires_frontend": any(kw in requirement.lower() for kw in ["前端", "界面", "网页", "html", "react"]),
            "requires_backend": any(kw in requirement.lower() for kw in ["后端", "api", "服务", "server"]),
            "requires_data": any(kw in requirement.lower() for kw in ["数据", "分析", "机器学习", "ml"]),
            "complexity": "high" if len(requirement) > 100 else "medium",
        }

        print(f"分析结果：需要前端={analysis_result['requires_frontend']}, "
              f"后端={analysis_result['requires_backend']}, "
              f"数据={analysis_result['requires_data']}")

        return analysis_result

    def generate_code(self, requirement: str, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        代码生成阶段

        根据需求分析结果，调用相应的专门化Agent生成代码
        """
        print("\n" + "-"*50)
        print("⚙️ 阶段2：代码生成")
        print("-"*50)

        generated_code = {}

        # 调用前端Agent
        if analysis["requires_frontend"]:
            print("\n📤 调用前端开发专家...")
            response = self.frontend_agent.generate_reply(
                messages=[{"role": "user", "content": requirement}]
            )
            generated_code["frontend"] = response
            print("✅ 前端代码生成完成")

        # 调用后端Agent
        if analysis["requires_backend"]:
            print("\n📤 调用后端开发专家...")
            response = self.backend_agent.generate_reply(
                messages=[{"role": "user", "content": requirement}]
            )
            generated_code["backend"] = response
            print("✅ 后端代码生成完成")

        # 调用数据科学Agent
        if analysis["requires_data"]:
            print("\n📤 调用数据科学专家...")
            response = self.data_agent.generate_reply(
                messages=[{"role": "user", "content": requirement}]
            )
            generated_code["data"] = response
            print("✅ 数据科学代码生成完成")

        return generated_code

    def test_and_debug(self, code: str, code_type: str = "python") -> str:
        """
        测试与调试阶段

        对生成的代码进行测试和调试
        """
        print("\n" + "-"*50)
        print("🧪 阶段3：测试与调试")
        print("-"*50)

        # 生成测试用例
        print("📤 生成测试用例...")
        test_code = self.tester.generate_tests(code, code_type)
        print("✅ 测试用例生成完成")

        # 执行测试
        print("\n📤 执行测试...")
        test_result = self.tester.run_tests(test_code)

        if test_result["success"]:
            print("✅ 所有测试通过")
            return code
        else:
            # 测试失败，启动调试
            print("❌ 测试失败，启动智能调试...")
            error_info = test_result.get("stderr", "未知错误")
            fixed_code = self.debugger.debug_and_fix(code, error_info)
            return fixed_code

    def run(self, requirement: str) -> Dict[str, Any]:
        """
        运行完整的企业级代码生成工作流

        Args:
            requirement: 用户的需求描述

        Returns:
            工作流执行结果，包含生成的代码和元数据
        """
        print("\n" + "="*60)
        print("🚀 启动企业级代码生成工作流")
        print("="*60)
        print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"需求：{requirement}")

        start_time = datetime.now()

        # 阶段1：需求分析
        analysis = self.analyze_requirement(requirement)

        # 阶段2：代码生成
        generated_code = self.generate_code(requirement, analysis)

        # 阶段3-4：测试与调试（对每个生成的代码）
        final_code = {}
        for code_type, code in generated_code.items():
            final_code[code_type] = self.test_and_debug(code, code_type)

        # 阶段5：总结输出
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*60)
        print("📦 工作流执行完成")
        print("="*60)
        print(f"执行时间：{duration:.2f}秒")
        print(f"生成代码类型：{list(final_code.keys())}")

        return {
            "requirement": requirement,
            "analysis": analysis,
            "generated_code": final_code,
            "execution_time": duration,
            "timestamp": end_time.isoformat(),
        }


# ============================================================
# 第五步：演示完整工作流
# ============================================================

def demo_simple_code_gen():
    """
    演示简单的代码生成流程
    """
    print("\n" + "="*60)
    print("📝 演示：简单代码生成流程")
    print("="*60)

    # 创建后端Agent
    backend_agent = CodeGenAgentFactory.create_agent(
        CodeGenAgentConfig(
            name="后端开发专家",
            specialty="backend",
        )
    )

    user_proxy = UserProxyAgent(
        name="用户",
        human_input_mode="NEVER",
    )

    # 简单任务
    task = "创建一个用户管理API，包含用户注册和登录功能"

    print(f"任务：{task}\n")

    # 生成代码
    print("📤 生成代码...")
    response = backend_agent.generate_reply(
        messages=[{"role": "user", "content": task}]
    )

    print("\n📥 生成的代码：")
    print(response)

    # 分析生成的代码
    print("\n" + "-"*40)
    print("🔍 代码分析：")

    if "class" in response or "def" in response:
        print("✅ 包含类和函数定义")
    if "import" in response:
        print("✅ 包含导入语句")
    if "#" in response or "中文" in response:
        print("✅ 包含中文注释")

    return response


def demo_intelligent_debug():
    """
    演示智能调试功能
    """
    print("\n" + "="*60)
    print("🔧 演示：智能调试功能")
    print("="*60)

    # 有问题的代码
    buggy_code = """
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

result = calculate_average([1, 2, 3, 4, 5])
print(f"平均值: {result}")

# 测试空列表
result = calculate_average([])
"""

    error_message = """
ZeroDivisionError: division by zero

During handling of the above exception, another exception occurred:

File "test.py", line 7, in <module>
    result = calculate_average([])
"""

    debugger = IntelligentDebugger()
    fixed_code = debugger.debug_and_fix(buggy_code, error_message)

    print("\n" + "-"*40)
    print("修正后的代码：")
    print(fixed_code)

    return fixed_code


def demo_multi_agent_collab():
    """
    演示多Agent协作生成完整项目
    """
    print("\n" + "="*60)
    print("🤝 演示：多Agent协作生成完整项目")
    print("="*60)

    # 创建工作流
    workflow = EnterpriseCodeGenWorkflow()

    # 复杂需求
    requirement = """
创建一个完整的数据分析系统，功能包括：
1. 数据导入：支持CSV和Excel文件
2. 数据清洗：处理缺失值和异常值
3. 数据分析：描述性统计和相关性分析
4. 数据可视化：生成图表和报告
5. 后端API：提供REST接口供前端调用
"""

    # 运行工作流
    result = workflow.run(requirement)

    print("\n" + "="*60)
    print("📊 工作流执行摘要")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    """
    主程序入口

    可选择运行不同的演示：
    1. demo_simple_code_gen()      - 简单代码生成
    2. demo_intelligent_debug()    - 智能调试演示
    3. demo_multi_agent_collab()    - 多Agent协作
    """

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         企业级代码生成Agent系统 - 第25节学习示例               ║
    ║                                                              ║
    ║  核心组件：                                                   ║
    ║  1. CodeGenAgentFactory - 代码生成Agent工厂                   ║
    ║  2. IntelligentDebugger  - 智能调试Agent                      ║
    ║  3. AutoTestAgent        - 自动化测试Agent                    ║
    ║  4. EnterpriseCodeGenWorkflow - 企业级工作流                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 运行所有演示
    try:
        demo_simple_code_gen()
    except Exception as e:
        print(f"\n⚠️ 简单代码生成演示出错: {e}")

    try:
        demo_intelligent_debug()
    except Exception as e:
        print(f"\n⚠️ 智能调试演示出错: {e}")

    try:
        demo_multi_agent_collab()
    except Exception as e:
        print(f"\n⚠️ 多Agent协作演示出错: {e}")
        print("   请检查API配置和网络连接")