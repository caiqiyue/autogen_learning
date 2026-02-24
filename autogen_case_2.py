
import os
from pathlib import Path


from autogen.agentchat.conversable_agent import ConversableAgent


def load_local_env(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


load_local_env()
model = get_required_env("OPENAI_MODEL")
api_key = get_required_env("AIHUBMIX_API_KEY")
base_url = get_required_env("AIHUBMIX_BASE_URL")


# agent = ConversableAgent(
#     name="chatbot",
#     llm_config={"config_list": [{"model": model, "api_key": api_key,'base_url':base_url }],"cache_seed": None,},
# )


agent = ConversableAgent(
    name="chatbot",
    llm_config={"config_list": [{"model": model, "api_key": api_key,'base_url':base_url}], "cache_seed": 24},
)

reply = agent.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "你好，请你非常详细地介绍一下你自己",
        }
    ]
)

print(reply)