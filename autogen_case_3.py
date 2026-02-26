import os

from autogen import ConversableAgent

agent = ConversableAgent(
    name="ollama_chatbot",
    llm_config={"config_list": 
                [{"model": "qwen2.5:3b",
                  "base_url": "http://localhost:11434/v1"}]},
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