import json
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, Request


def get_request_details(request: Request, body: dict = None) -> str:
    details = {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "headers": dict(request.headers),
        "query_params": dict(request.query_params),
    }
    return json.dumps(details, indent=2)


def setup_responses_routes(app: FastAPI):
    @app.post("/responses")
    @app.post("/v1/responses")
    @app.post("/openai/responses")
    async def responses_api(request: Request):
        data = await request.json()
        model = data.get("model", "unknown")
        request_details = get_request_details(request, data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_details = f"Request:{request_details}, Canned Response:{timestamp}"
        response_id = uuid.uuid4().hex
        message_id = f"msg_{uuid.uuid4().hex[:34]}"
        return {
            "id": f"resp_{response_id}",
            "created_at": int(time.time()),
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "model": model,
            "object": "response",
            "output": [
                {
                    "id": message_id,
                    "content": [
                        {
                            "annotations": [],
                            "text": response_details,
                            "type": "output_text",
                            "logprobs": [],
                        },
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": True,
            "temperature": data.get("temperature", 1.0),
            "tool_choice": data.get("tool_choice", "auto"),
            "tools": data.get("tools", []),
            "top_p": data.get("top_p", 1.0),
            "max_output_tokens": data.get("max_output_tokens"),
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "status": "completed",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "truncation": "disabled",
            "usage": {
                "input_tokens": 11,
                "input_tokens_details": {
                    "audio_tokens": None,
                    "cached_tokens": 0,
                    "text_tokens": None,
                },
                "output_tokens": 19,
                "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": None},
                "total_tokens": 30,
                "cost": None,
            },
            "user": None,
            "store": True,
            "background": False,
            "content_filters": None,
            "max_tool_calls": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
            "service_tier": "default",
            "top_logprobs": 0,
        }



