"""
Example: Parallel acompletions with Router (text + local image + http image).

Prereqs:
  export OPENAI_API_KEY=sk-...

Run:
  python examples/router_parallel_multimodal.py
"""
from __future__ import annotations

import asyncio
import os
from litellm import Router
from litellm.extras.images import compress_image


def msg_text(txt: str):
    return [{"role": "user", "content": txt}]


def msg_text_with_local_image(txt: str, path: str):
    data_url = compress_image(path, max_kb=256, cache_dir=".cache")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


def msg_text_with_http_image(txt: str, url: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


async def main():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini-mm",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ]
    )

    reqs = [
        {"model": "gpt-4o-mini-mm", "messages": msg_text("Say hello briefly.")},
        {
            "model": "gpt-4o-mini-mm",
            "messages": msg_text_with_local_image(
                "Describe this image in one line.", "local/images/eiffel_winter.png"
            ),
        },
        {
            "model": "gpt-4o-mini-mm",
            "messages": msg_text_with_http_image(
                "What’s in this photo?",
                "https://upload.wikimedia.org/wikipedia/commons/5/5f/Alaskan_Malamute.jpg",
            ),
        },
    ]

    # Parallel via gather
    outs = await asyncio.gather(
        *[router.acompletion(**r) for r in reqs], return_exceptions=True
    )
    for i, r in enumerate(outs):
        if isinstance(r, Exception):
            print(f"[{i}] error:", r)
        else:
            print(f"[{i}] ->", getattr(r.choices[0].message, "content", "").strip())


if __name__ == "__main__":
    asyncio.run(main())

