import json
import re
import time
import uuid
from datetime import datetime

from typing import Any

from fastapi import FastAPI, Request, HTTPException


# Header to identify which model/deployment this request targets (simulates Azure model-specific encryption).
# When set, the mock validates that encrypted_content in input was produced by this model.
MOCK_AZURE_MODEL_HEADER = "X-Mock-Azure-Model"

# Prefix we use in mock encrypted_content: gAAA_model_<model_id>_<32hex uuid>
# Model id can contain underscores (e.g. gpt-5.1-codex-openai-2).
ENCRYPTED_CONTENT_MODEL_PREFIX = re.compile(r"^gAAA_model_(.+)_[0-9a-f]{32}$")


def _extract_model_from_encrypted_content(encrypted: str) -> str | None:
    """Extract model id from our mock encrypted_content format, or None if not our format."""
    if not isinstance(encrypted, str) or not encrypted.startswith("gAAA"):
        return None
    m = ENCRYPTED_CONTENT_MODEL_PREFIX.match(encrypted)
    return m.group(1) if m else None


def _collect_encrypted_contents(obj, out: list[str]) -> None:
    """Recursively collect all encrypted_content string values from input structure."""
    if isinstance(obj, dict):
        if "encrypted_content" in obj and obj["encrypted_content"]:
            out.append(obj["encrypted_content"])
        for v in obj.values():
            _collect_encrypted_contents(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_encrypted_contents(item, out)


def _validate_encrypted_content_model(request_model: str | None, input_data: Any) -> str | None:
    """
    If request_model is set, check that all encrypted_content in input was produced by this model.
    Returns error message if validation fails, else None.
    Content with our format (gAAA_model_<id>_) must match request_model.
    """
    if not request_model:
        return None
    encrypted_values: list[str] = []
    _collect_encrypted_contents(input_data, encrypted_values)
    for enc in encrypted_values:
        content_model = _extract_model_from_encrypted_content(enc)
        if content_model is not None and content_model != request_model:
            err = enc[:50] + "..." if len(enc) > 50 else enc
            return f"The encrypted content {err} could not be verified."
    return None


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

        # Simulate Azure: encrypted content from one model cannot be verified by another.
        input_data = data.get("input")
        err_msg = _validate_encrypted_content_model(model, input_data)
        if err_msg is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": err_msg,
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_encrypted_content",
                    }
                },
            )

        request_details = get_request_details(request, data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_details = f"Request:{request_details}, Canned Response:{timestamp}"
        response_id = uuid.uuid4().hex
        message_id = f"msg_{uuid.uuid4().hex[:34]}"
        reasoning_id = f"rs_{uuid.uuid4().hex[:34]}"

        output_items: list[dict[str, Any]] = [
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
        ]

        if model:
            output_items.append(
                {
                    "id": reasoning_id,
                    "type": "reasoning",
                    "status": "completed",
                    "encrypted_content": f"gAAA_model_{model}_{uuid.uuid4().hex}",
                }
            )

        return {
            "id": f"resp_{response_id}",
            "created_at": int(time.time()),
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "model": model,
            "object": "response",
            "output": output_items,
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
