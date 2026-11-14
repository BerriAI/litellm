"""A deterministic OpenAI-compatible upstream for the pricing currency demo.

Every chat completion reports 1000 prompt and 500 completion tokens, so a deployment's bill is fixed by its prices
alone. A model name containing "reported-cost" also reports a provider cost of 9.99 USD, which a deployment with
custom prices must ignore. Responses are synthetic and say so, and no model is called.
"""

import json
import sys
from typing import Final

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

PROMPT_TOKENS: Final = 1000
COMPLETION_TOKENS: Final = 500
CREATED: Final = 1_700_000_000
REPORTED_COST: Final = 9.99


async def chat_completions(request: Request) -> JSONResponse:
    body: Final = json.loads(await request.body() or b"{}")
    model: Final = str(body.get("model", "synthetic"))
    return JSONResponse(
        {
            "id": "chatcmpl-synthetic-demo",
            "object": "chat.completion",
            "created": CREATED,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Synthetic demo reply from {model}."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                **({"cost": REPORTED_COST} if "reported-cost" in model else {}),
            },
        }
    )


async def models(_: Request) -> JSONResponse:
    return JSONResponse({"object": "list", "data": []})


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


app: Final = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models),
        Route("/health", health),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8090, log_level="warning")
