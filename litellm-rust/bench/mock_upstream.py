"""Mock upstream LLM provider for benchmarking.

Returns a fixed chat completion response with minimal latency.
Simulates token generation by sleeping for a configurable duration.
"""

import json
import asyncio
from aiohttp import web

MOCK_RESPONSE = {
    "id": "chatcmpl-bench",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a benchmark response from the mock upstream provider."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 12,
        "total_tokens": 22
    }
}

async def handle_chat_completions(request):
    body = await request.json()
    await asyncio.sleep(0.005)  # 5ms simulated processing
    return web.json_response(MOCK_RESPONSE)

async def handle_health(request):
    return web.json_response({"status": "ok"})

app = web.Application()
app.router.add_post("/v1/chat/completions", handle_chat_completions)
app.router.add_get("/health", handle_health)

if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=11434, print=None, access_log=None)
