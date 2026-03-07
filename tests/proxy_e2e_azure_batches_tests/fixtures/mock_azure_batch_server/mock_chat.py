import json
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def get_request_details(request: Request, body: dict = None) -> str:
    details = {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "headers": dict(request.headers),
        "query_params": dict(request.query_params),
    }
    return json.dumps(details, indent=2)


def data_generator(response_details: str, model: str):
    response_id = uuid.uuid4().hex
    content = response_details
    chunk_size = 50
    for i in range(0, len(content), chunk_size):
        text_chunk = content[i : i + chunk_size]
        chunk = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text_chunk}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    final_chunk = {
        "id": f"chatcmpl-{response_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def setup_chat_routes(app: FastAPI):
    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    @app.post("/openai/deployments/{model:path}/chat/completions")
    async def completion(request: Request):
        data = await request.json()
        model = data.get("model", "unknown")
        request_details = get_request_details(request, data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_details = f"Request:{request_details}, Canned Response:{timestamp}"

        if data.get("stream"):
            return StreamingResponse(
                content=data_generator(response_details, model),
                media_type="text/event-stream",
            )
        else:
            response_id = uuid.uuid4().hex
            response = {
                "id": f"chatcmpl-{response_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "system_fingerprint": "fp_mock_server",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_details,
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            }
            return response

    @app.post("/completions")
    @app.post("/v1/completions")
    async def text_completion(request: Request):
        data = await request.json()
        model = data.get("model", "unknown")
        request_details = get_request_details(request, data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_details = f"Request:{request_details}, Canned Response:{timestamp}"

        if data.get("stream"):
            return StreamingResponse(
                content=data_generator(response_details, model),
                media_type="text/event-stream",
            )
        else:
            response = {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "text": response_details,
                    },
                ],
                "created": int(time.time()),
                "model": model,
                "object": "text_completion",
                "system_fingerprint": None,
                "usage": {
                    "completion_tokens": 16,
                    "prompt_tokens": 10,
                    "total_tokens": 26,
                },
            }
            return response
