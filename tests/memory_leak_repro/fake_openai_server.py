"""
Fake OpenAI-compatible server for memory leak reproduction.

Returns minimal valid chat completion responses as fast as possible.
Run: python fake_openai_server.py
"""

import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Pre-built response to avoid per-request allocation
_RESPONSE_TEMPLATE = {
    "id": "chatcmpl-fake-00000",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock response for memory leak testing.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    await request.body()
    resp = dict(_RESPONSE_TEMPLATE)
    resp["created"] = int(time.time())
    return JSONResponse(resp)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=18080, log_level="warning", access_log=False
    )
