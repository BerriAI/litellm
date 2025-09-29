"""Feature recipe: batching prompts with Router.parallel_acompletions."""

import asyncio
import os
from typing import List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router

MODEL_ALIAS = os.getenv("LITELLM_DEFAULT_MODEL") or "ollama/qwen2.5-coder:14b"

model_list = [
    {
        "model_name": "parallel-demo",
        "litellm_params": {"model": MODEL_ALIAS},
    }
]

router = Router(model_list=model_list)

messages_list: List[List[dict]] = [
    [
        {
            "role": "user",
            "content": "Describe this image: https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg",
        }
    ],
    [
        {
            "role": "user",
            "content": f"Summarize this local diagram: {os.path.abspath('local/images/sample_chart.png')}",
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe both of these images."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Test.svg",
                    },
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": "Summarize the benefits of unit testing in a paragraph.",
        }
    ],
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
            "request": item.request.messages,
            "response": getattr(item.response, "choices", item.response),
            "error": str(item.error) if item.error else None,
        })


if __name__ == "__main__":
    asyncio.run(main())
