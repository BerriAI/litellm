"""Mini-agent feature recipe using the Router interface.

Requires ``LITELLM_ENABLE_MINI_AGENT=1`` and the experimental mini-agent
components (codex CLI + MCP tooling) to be available locally.
"""

import asyncio
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv
from litellm import Router

load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_MINI_AGENT") != "1":
    print("Set LITELLM_ENABLE_MINI_AGENT=1 before running this recipe.")
    sys.exit(1)

BASE_MODEL = (
    os.getenv("LITELLM_DEFAULT_CHUTES_MODEL")
    or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
    or "deepseek-ai/DeepSeek-R1"
)

model_list = [
    {
        "model_name": "mini-agent",
        "litellm_params": {
            "model": "mini-agent",
            "custom_llm_provider": "mini-agent",
            "target_model": BASE_MODEL,
            "allowed_languages": ["python", "rust", "go", "javascript"],
            "max_iterations": 6,
            "max_seconds": 180,
        },
    }
]

router = Router(model_list=model_list)

PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a meticulous analyst. Use exec_python to verify numbers before you answer."
        ),
    },
    {
        "role": "user",
        "content": (
            "A startup tracks weekly active users: [1320, 1588, 1710, 1895, 2044, 2102, 1998, 2234]. "
            "Using Python, compute the week-over-week percentage change, highlight the largest spike, "
            "and provide a recommendation."
        ),
    },
]


async def main() -> None:
    print(json.dumps({"model_list": model_list, "prompt": PROMPT}, indent=2))
    response = await router.acompletion(model="mini-agent", messages=PROMPT)
    payload = response.model_dump() if hasattr(response, "model_dump") else str(response)
    print(json.dumps({"request": PROMPT, "response": payload}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
