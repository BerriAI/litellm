"""
Ultra-fast mock OpenAI server for load testing.
Returns minimal valid responses with near-zero processing time.
"""
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

MOCK_RESPONSE = {
    "id": "chatcmpl-mock-loadtest",
    "object": "chat.completion",
    "created": 0,
    "model": "fake-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Mock response for load testing."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()
    response = MOCK_RESPONSE.copy()
    response["created"] = int(time.time())
    return JSONResponse(response)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18888, log_level="warning", access_log=False)
