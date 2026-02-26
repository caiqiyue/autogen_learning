import os
import dotenv

dotenv.load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": os.environ.get("OPENAI_MODEL"),
            "api_key": os.environ.get("AIHUBMIX_API_KEY"),
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

import autogen
from autogen.agentchat import ConversableAgent

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
