#!/usr/bin/env python3
"""
Mock OpenAI-compatible server for benchmarking LiteLLM proxy overhead.
Returns fast streaming responses to isolate proxy overhead from provider latency.
"""

import json
import time
import uuid
import asyncio
from aiohttp import web


WORDS = [
    "The", "history", "of", "artificial", "intelligence", "began", "in",
    "antiquity", "with", "myths", "stories", "and", "rumors", "of",
    "artificial", "beings", "endowed", "with", "intelligence", "or",
    "consciousness", "by", "master", "craftsmen.", "The", "seeds", "of",
    "modern", "AI", "were", "planted", "by", "philosophers", "who",
    "attempted", "to", "describe", "the", "process", "of", "human",
    "thinking", "as", "the", "mechanical", "manipulation", "of", "symbols.",
    "This", "work", "culminated", "in", "the", "invention", "of", "the",
    "programmable", "digital", "computer.", "The", "field", "of", "AI",
    "research", "was", "founded", "at", "a", "workshop", "held", "on",
    "the", "campus", "of", "Dartmouth", "College.", "Many", "predicted",
    "that", "a", "machine", "as", "intelligent", "as", "a", "human",
    "would", "exist", "in", "no", "more", "than", "a", "generation.",
    "Eventually", "it", "became", "obvious", "that", "researchers", "had",
    "grossly", "underestimated", "the", "difficulty.", "The", "game", "of",
    "chess", "has", "long", "been", "viewed", "as", "a", "litmus", "test.",
    "In", "1997", "Deep", "Blue", "became", "the", "first", "computer",
    "to", "beat", "a", "reigning", "world", "chess", "champion.", "AI",
    "gradually", "restored", "its", "reputation", "in", "the", "late",
    "1990s", "and", "early", "21st", "century.", "Progress", "in", "large",
    "language", "models", "has", "sparked", "both", "excitement", "and",
    "concern.", "Researchers", "continue", "to", "push", "boundaries.",
]

# Pre-build chunks for common token counts to avoid per-request JSON serialization
_CHUNK_CACHE: dict[int, bytes] = {}


def _build_streaming_body(model: str, num_tokens: int) -> bytes:
    """Pre-build the entire SSE body as a single bytes object."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    parts = []
    for i in range(num_tokens):
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": WORDS[i % len(WORDS)] + " "},
                "finish_reason": None,
            }],
        }
        if i == num_tokens - 1:
            chunk["choices"][0]["finish_reason"] = "stop"
            chunk["usage"] = {
                "prompt_tokens": 500,
                "completion_tokens": num_tokens,
                "total_tokens": 500 + num_tokens,
            }
        parts.append(f"data: {json.dumps(chunk)}\n\n")

    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


async def handle_chat_completions(request):
    body = await request.json()
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 100)
    model = body.get("model", "gpt-4.1-mini")

    if not stream:
        text = " ".join(WORDS[:min(max_tokens, len(WORDS))])
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": min(max_tokens, len(WORDS)),
                "total_tokens": 500 + min(max_tokens, len(WORDS)),
            },
        }
        return web.json_response(response)

    # Streaming response - write all chunks at once for maximum throughput
    num_tokens = min(max_tokens, len(WORDS))
    body_bytes = _build_streaming_body(model, num_tokens)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)
    await response.write(body_bytes)
    await response.write_eof()
    return response


async def handle_models(request):
    return web.json_response({
        "object": "list",
        "data": [
            {"id": "gpt-4.1-mini", "object": "model", "owned_by": "mock"},
        ],
    })


app = web.Application()
app.router.add_post("/v1/chat/completions", handle_chat_completions)
app.router.add_post("/chat/completions", handle_chat_completions)
app.router.add_get("/v1/models", handle_models)
app.router.add_get("/models", handle_models)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    print(f"Starting mock OpenAI server on port {args.port}")
    web.run_app(app, host="0.0.0.0", port=args.port)
