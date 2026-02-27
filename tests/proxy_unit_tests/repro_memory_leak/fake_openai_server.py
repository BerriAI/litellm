"""
Minimal fake OpenAI-compatible server for memory leak reproduction.
Responds instantly to /v1/chat/completions with a canned response.
"""

import json
import time
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


CANNED_RESPONSE = {
    "id": "chatcmpl-fake123",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a fake response for memory leak testing.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25,
    },
}

MODELS_RESPONSE = {
    "data": [
        {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
    ],
    "object": "list",
}


async def chat_completions(request: Request) -> Response:
    return JSONResponse(CANNED_RESPONSE)


async def models(request: Request) -> Response:
    return JSONResponse(MODELS_RESPONSE)


async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def catch_all(request: Request) -> Response:
    return JSONResponse(CANNED_RESPONSE)


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/models", models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/{path:path}", catch_all, methods=["GET", "POST", "PUT", "DELETE"]),
    ],
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18080, log_level="warning")
