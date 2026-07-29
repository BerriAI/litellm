"""Canned OpenAI-shaped mock server for the CI proxy E2Es.

Several CI jobs run the litellm proxy (often in its own Docker container) against
a model whose ``api_base`` is a fake OpenAI endpoint that returns canned
responses, so the run costs nothing and does not depend on a real provider. That
endpoint used to be a single shared deployment; when it went down every one of
those jobs failed with ``404 Application not found`` even though nothing in the
PR was broken.

This process is the local stand-in. A model points its ``api_base`` here and
gets back a well-formed chat/text/embedding response with realistic ``usage`` so
cost tracking and spend accounting still exercise their real code paths. The one
behavioral special case mirrors the old hosted mock: a request whose ``model``
is ``429`` returns HTTP 429 so rate-limit and cooldown tests still have
something to trip on.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Final

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

_CANNED_CONTENT: Final = "Hello! This is a mock response from the fake OpenAI endpoint."
_RATE_LIMIT_MODEL: Final = "429"
_SLOW_MODEL: Final = "slow-endpoint"
_SLOW_RESPONSE_SECONDS: Final = 3.0
_PROMPT_TOKENS: Final = 20
_COMPLETION_TOKENS: Final = 20


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": _PROMPT_TOKENS,
        "completion_tokens": _COMPLETION_TOKENS,
        "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
    }


def _requested_model(body: dict[str, object]) -> str:
    model = body.get("model")
    return model if isinstance(model, str) else "mock-model"


def _wants_stream(body: dict[str, object]) -> bool:
    return body.get("stream") is True


def _wants_stream_usage(body: dict[str, object]) -> bool:
    options = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


async def _parse_body(request: Request) -> dict[str, object]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rate_limit_response(model: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": f"Rate limit reached for model `{model}` (mock).",
                "type": "rate_limit_error",
                "code": "429",
            }
        },
    )


def _chat_completion_body(model: str) -> dict[str, object]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _CANNED_CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(),
    }


async def _chat_completion_stream(model: str, with_usage: bool) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict[str, object], finish_reason: str | None) -> dict[str, object]:
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': _CANNED_CONTENT}, None))}\n\n"
    yield f"data: {json.dumps(chunk({}, 'stop'))}\n\n"
    if with_usage:
        final = chunk({}, None) | {"choices": [], "usage": _usage()}
        yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def chat_completions(request: Request) -> Response:
    body = await _parse_body(request)
    model = _requested_model(body)
    if model == _RATE_LIMIT_MODEL:
        return _rate_limit_response(model)
    if model == _SLOW_MODEL:
        await asyncio.sleep(_SLOW_RESPONSE_SECONDS)
    if _wants_stream(body):
        return StreamingResponse(
            _chat_completion_stream(model, _wants_stream_usage(body)),
            media_type="text/event-stream",
        )
    return JSONResponse(_chat_completion_body(model))


def _text_completion_body(model: str) -> dict[str, object]:
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": _CANNED_CONTENT,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(),
    }


async def _text_completion_stream(model: str, with_usage: bool) -> AsyncIterator[str]:
    response_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(text: str, finish_reason: str | None) -> dict[str, object]:
        return {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
        }

    yield f"data: {json.dumps(chunk(_CANNED_CONTENT, None))}\n\n"
    yield f"data: {json.dumps(chunk('', 'stop'))}\n\n"
    if with_usage:
        final = chunk("", None) | {"choices": [], "usage": _usage()}
        yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def completions(request: Request) -> Response:
    body = await _parse_body(request)
    model = _requested_model(body)
    if model == _RATE_LIMIT_MODEL:
        return _rate_limit_response(model)
    if model == _SLOW_MODEL:
        await asyncio.sleep(_SLOW_RESPONSE_SECONDS)
    if _wants_stream(body):
        return StreamingResponse(
            _text_completion_stream(model, _wants_stream_usage(body)),
            media_type="text/event-stream",
        )
    return JSONResponse(_text_completion_body(model))


async def embeddings(request: Request) -> Response:
    body = await _parse_body(request)
    raw_input = body.get("input", "")
    count = len(raw_input) if isinstance(raw_input, list) else 1
    return JSONResponse(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": [0.0] * 1536} for i in range(max(count, 1))],
            "model": _requested_model(body),
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    )


async def triton_embeddings(_request: Request) -> Response:
    return JSONResponse(
        {
            "model_name": "my-triton-model",
            "outputs": [
                {
                    "name": "output",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [0.1, 0.2],
                }
            ],
        }
    )


async def list_models(_request: Request) -> Response:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"id": "fake", "object": "model", "owned_by": "mock"},
                {"id": "my-fake-model", "object": "model", "owned_by": "mock"},
            ],
        }
    )


async def health(_request: Request) -> Response:
    return PlainTextResponse("ok")


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/", health, methods=["GET"]),
        Route("/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/completions", completions, methods=["POST"]),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/embeddings", embeddings, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/triton/embeddings", triton_embeddings, methods=["POST"]),
        Route("/models", list_models, methods=["GET"]),
        Route("/v1/models", list_models, methods=["GET"]),
    ]
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8190)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
