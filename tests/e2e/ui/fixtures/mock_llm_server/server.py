"""
Mock LLM server for UI e2e tests.
Responds to OpenAI-format endpoints with canned responses.
"""

import time
import asyncio
import json
import re
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


app = FastAPI(title="Mock LLM Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "fake-gpt-4", "object": "model", "owned_by": "mock"},
            {"id": "fake-claude", "object": "model", "owned_by": "mock"},
        ],
    }


HOLD_MARKER = re.compile(r"e2e-hold-(\d+)s")


def _hold_seconds(body: dict) -> float:
    """Let a test keep a request in flight by putting `e2e-hold-<n>s` in the prompt.

    The active-requests page only shows requests that have not finished, so a
    browser test needs a completion that is still running while it asserts.
    """
    messages = body.get("messages") or []
    for message in reversed(messages):
        content = message.get("content")
        match = HOLD_MARKER.search(content) if isinstance(content, str) else None
        if match:
            return min(float(match.group(1)), 60.0)
    return 0.0


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model")
    stream = body.get("stream", False)
    hold_seconds = _hold_seconds(body)
    if hold_seconds:
        await asyncio.sleep(hold_seconds)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:

        async def stream_generator():
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "This is a mock response.",
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            done_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a mock response."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input", [""])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.0] * 1536}
            for i in range(len(inputs))
        ],
        "model": body.get("model", "mock-embedding"),
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090)
