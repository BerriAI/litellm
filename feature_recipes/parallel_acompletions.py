"""Parallel-acompletions feature recipe.

Illustrates batching heterogeneous prompts through ``Router.parallel_acompletions``:

- Plain text prompt with inline image URL reference.
- OpenAI multimodal payload using ``[{"type": "text"}, {"type": "image_url"}, ...]``.
- Local file reference embedded in text content.
- Mixed request shapes (``RouterParallelRequest`` and plain dict).
- Request-level knobs such as ``temperature``, ``stream``, ``max_tokens``, ``top_p``.
- Per-result output that includes the original request, response, content, and error.

Execute with ``python feature_recipes/parallel_acompletions.py`` after setting
``LITELLM_DEFAULT_MODEL`` (or relying on its default).
"""

import asyncio
import os
from typing import List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest

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
            "content": f"Summarize this local diagram: {os.path.abspath('local/images/sample_chart.png')}",
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
        RouterParallelRequest("parallel-demo", messages_list[0], temperature=0.3),
        RouterParallelRequest("parallel-demo", messages_list[1], stream=True),
        {
            "model": "parallel-demo",
            "messages": messages_list[2],
            "kwargs": {"max_tokens": 128},
        },
        RouterParallelRequest("parallel-demo", messages_list[3], top_p=0.95),
    ]

    results = await router.parallel_acompletions(
        requests,
        preserve_order=True,
        return_exceptions=True,
        concurrency=2,
    )

    for item in results:
        request_messages = getattr(item.request, "messages", None)
        print({
            "index": item.index,
            "request": request_messages,
            "response": getattr(item.response, "choices", item.response),
            "content": item.content,
            "error": str(item.error) if item.error else None,
        })


if __name__ == "__main__":
    asyncio.run(main())
