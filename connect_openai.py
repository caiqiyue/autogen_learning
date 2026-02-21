import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request


def load_local_env(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def configure_proxy(ignore_proxy: bool) -> None:
    if ignore_proxy:
        opener = request.build_opener(request.ProxyHandler({}))
        request.install_opener(opener)


def api_request(base_url: str, api_key: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{normalize_base_url(base_url)}{path}"
    data = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, data=data, headers=headers, method="POST" if payload is not None else "GET")

    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {path}: {raw[:500]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error calling {path}: {exc.reason}") from exc


def list_accessible_models(base_url: str, api_key: str) -> list[str]:
    result = api_request(base_url, api_key, "/models")
    data = result.get("data", [])
    model_ids = sorted({item.get("id") for item in data if isinstance(item, dict) and item.get("id")})
    return model_ids


def run_chat(base_url: str, api_key: str, model: str, message: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
    }
    result = api_request(base_url, api_key, "/chat/completions", payload=payload)
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError(f"Chat response has no choices: {result}")

    message_obj = choices[0].get("message", {})
    content = message_obj.get("content")
    if not content:
        raise RuntimeError(f"Chat response content is empty: {result}")
    return content


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(description="OpenAI-compatible API connectivity checker")
    parser.add_argument(
        "--message",
        default=os.getenv("OPENAI_USER_MESSAGE", "Hello, how are you?"),
        help="User message for chat completion test.",
    )
    parser.add_argument(
        "--list-models-only",
        action="store_true",
        help="Only list accessible models and skip chat test.",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from environment. Default is disabled to avoid local invalid proxy issues.",
    )
    args = parser.parse_args()

    configure_proxy(ignore_proxy=not args.use_env_proxy)

    api_key = get_required_env("AIHUBMIX_API_KEY")
    base_url = get_required_env("AIHUBMIX_BASE_URL")
    model = get_required_env("OPENAI_MODEL")

    print("Checking model access...")
    models = list_accessible_models(base_url, api_key)
    if models:
        print("Accessible models:")
        for model_id in models:
            print(f"- {model_id}")
    else:
        print("No accessible models returned by API.")

    if args.list_models_only:
        return

    print("\nTesting chat completion...")
    reply = run_chat(base_url, api_key, model, args.message)
    print(f"Model: {model}")
    print("Reply:")
    print(reply)


if __name__ == "__main__":
    main()
