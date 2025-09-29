"""Feature recipe: batching prompts with Router.parallel_acompletions."""

import asyncio
import os
from typing import List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router

MODEL_ALIAS = (
    os.getenv("SCENARIO_PARALLEL_MODEL")
    or os.getenv("LITELLM_DEFAULT_MODEL")
    or "ollama/qwen2.5-coder:14b"
)
PROVIDER = None
if "/" in MODEL_ALIAS:
    PROVIDER = MODEL_ALIAS.split("/", 1)[0]

model_list = [
    {
        "model_name": "parallel-demo",
        "litellm_params": {"model": MODEL_ALIAS},
    }
]

router = Router(model_list=model_list)

messages_list: List[List[dict]] = [
    [{"role": "user", "content": "List three programming languages."}],
    [{"role": "user", "content": "Give me a fun fact about pandas."}],
    [{"role": "user", "content": "Summarize the benefits of unit testing."}],
]

async def main() -> None:
    requests = [
        {
            "model": "parallel-demo",
            "messages": messages,
        }
        for messages in messages_list
    ]
    results = await router.parallel_acompletions(
        requests,
        preserve_order=True,
        return_exceptions=True,
    )
    for item in results:
        print({
            "index": item.index,
            "content": item.content,
            "error": str(item.error) if item.error else None,
        })


if __name__ == "__main__":
    asyncio.run(main())
