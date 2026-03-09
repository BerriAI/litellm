"""
Ultra-fast mock OpenAI server for load testing.
Pre-serializes the response body so the mock adds near-zero overhead.
"""
import orjson
import uvicorn
from fastapi import FastAPI, Request, Response

app = FastAPI()

_BODY = orjson.dumps(
    {
        "id": "chatcmpl-mock-loadtest",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mock."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    await request.body()
    return Response(content=_BODY, media_type="application/json")


@app.get("/health")
async def health():
    return Response(content=b'{"status":"ok"}', media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18888, log_level="warning", access_log=False)
