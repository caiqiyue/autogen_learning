from autogen import ConversableAgent
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
# reply = agent.generate_reply(
#     messages=[
#         {
#             "role": "user",
#             "content": "如果我遇到合同纠纷，应该如何维权？",
#         }
#     ]
# )

# 打印生成的回复
print(agent.name)
print(agent.description)
print(agent.system_message)