"""
Chat completions endpoint and helper functions.
Can be mounted with: app.include_router(chat_completions_router)
"""

import asyncio
import json
import os
import random
import uuid
from typing import Optional

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

router = APIRouter()

# Only print when MOCK_CHAT_VERBOSE=1 (silence non-error logs by default)
_verbose = os.getenv("MOCK_CHAT_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _log(msg: str) -> None:
    if _verbose:
        print(msg)


def _normalize_model_for_response(model: str) -> str:
    """Normalize deployment model IDs (e.g. gpt-4o-mini-data-zone) to client-facing names (gpt-4o-mini)
    so LiteLLM does not log 'response model mismatch' when backend returns a deployment variant."""
    if not model or not isinstance(model, str):
        return model or ""
    # Strip known deployment suffixes so response matches what the client requested
    for suffix in ("-data-zone", "-eu", "-us", "-preview"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def get_histogram_sleep_time() -> float:
    """
    Returns a sleep time based on a histogram distribution:
    - Most requests (~70%): ~10 seconds (8-12s range)
    - Some requests (~25%): ~60 seconds (55-65s range)
    - Very few (~5%): 60+ seconds (60-300s range)

    This simulates realistic degraded provider behavior where most requests
    are slow but not terrible, some are very slow, and a few are extremely slow.
    """
    rand = random.random()

    if rand < 0.70:  # 70% of requests: ~10 seconds
        sleep_time = max(1.0, random.gauss(10, 2))
        return sleep_time
    elif rand < 0.95:  # 25% of requests: ~60 seconds
        sleep_time = max(30.0, random.gauss(60, 5))
        return sleep_time
    else:  # 5% of requests: 60+ seconds (up to 5 minutes)
        sleep_time = random.uniform(60, 300)
        return sleep_time


def data_generator(model: Optional[str] = None):
    """Stream OpenAI-format chat completion chunks."""
    response_id = uuid.uuid4().hex
    sentence = "Hello this is a test response from a fixed OpenAI endpoint."
    words = sentence.split(" ")
    _model = model if isinstance(model, str) else "gpt-3.5-turbo-0125"
    for word in words:
        word = word + " "
        chunk = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": _model,
            "choices": [{"index": 0, "delta": {"content": word}}],
        }
        try:
            yield f"data: {json.dumps(chunk)}\n\n"
        except Exception:
            yield f"data: {json.dumps(chunk)}\n\n"


def _validate_auth(request: Request, authorization: Optional[str] = None, required: bool = True) -> bool:
    """Validate authentication - accepts Bearer token and x-goog-api-key. Returns True if valid."""
    if authorization and authorization.startswith("Bearer "):
        return True
    api_key = request.headers.get("x-goog-api-key") or request.headers.get("X-Goog-Api-Key")
    if api_key:
        return True
    if required:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Authorization header. Use 'Authorization: Bearer <token>' or 'x-goog-api-key: <key>'",
        )
    return False


def has_valid_auth(request: Request, authorization: Optional[str] = None) -> bool:
    """Check if request has valid authentication - returns True/False without raising."""
    try:
        return _validate_auth(request, authorization, required=False)
    except HTTPException:
        return False


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
@router.post("/openai/deployments/{model:path}/chat/completions")
async def completion(request: Request, authorization: Optional[str] = Header(None)):
    """OpenAI-compatible chat completions (and Azure path). Supports streaming and LiteLLM model id."""
    has_valid_auth(request, authorization)

    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)
    if _time_to_sleep is not None:
        _log("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))

    data = await request.json()
    data = data if isinstance(data, dict) else {}

    if data.get("model") == "429":
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")

    if data.get("model") == "random_sleep":
        sleep_time = random.randint(1, 10)
        _log("sleeping for " + str(sleep_time) + " seconds")
        await asyncio.sleep(sleep_time)

    if data.get("model") in ["degraded", "slow_provider", "blocked"]:
        sleep_time = get_histogram_sleep_time()
        _log(f"[DEGRADED MODE] Sleeping for {sleep_time:.1f} seconds ({sleep_time/60:.1f} minutes) - histogram distribution")
        await asyncio.sleep(sleep_time)

    _path_for_claude = request.url.path
    if data.get("stream") is True and "/v1/chat/completions" not in _path_for_claude:
        _path_params_stream = getattr(request, "path_params", None) or {}
        _stream_model = (
            data.get("litellm_model_id")
            or request.headers.get("X-LiteLLM-Model")
            or request.headers.get("x-litellm-model")
            or data.get("model")
            or data.get("model_name")
            or data.get("modelId")
            or data.get("model_id")
            or _path_params_stream.get("model")
            or "gpt-3.5-turbo-0125"
        )
        _stream_model = _stream_model if isinstance(_stream_model, str) else "gpt-3.5-turbo-0125"
        if _stream_model.strip() == "*":
            _stream_model = (
                _path_params_stream.get("model")
                or request.headers.get("X-Model")
                or request.headers.get("x-model")
                or "gpt-3.5-turbo-0125"
            )
            _stream_model = _stream_model if isinstance(_stream_model, str) else "gpt-3.5-turbo-0125"
        _stream_model = _normalize_model_for_response(_stream_model)
        return StreamingResponse(
            content=data_generator(model=_stream_model),
            media_type="text/event-stream",
        )

    # Prefer client-requested model (litellm_model_id / headers) so LiteLLM does not log response model mismatch
    _path_params = getattr(request, "path_params", None) or {}
    _model = (
        data.get("litellm_model_id")
        or request.headers.get("X-LiteLLM-Model")
        or request.headers.get("x-litellm-model")
        or data.get("model")
        or data.get("model_name")
        or data.get("modelId")
        or data.get("model_id")
        or _path_params.get("model")
        or (getattr(request, "query_params", None) or {}).get("model")
        or (data.get("generationConfig") or {}).get("model")
        or (data.get("metadata") or {}).get("model")
        or (data.get("metadata") or {}).get("model_name")
        or request.headers.get("X-Model")
        or request.headers.get("x-model")
        or ""
    )
    _model = _model if isinstance(_model, str) else ""

    if not _model and "contents" in data and "messages" not in data:
        _model = "claude"
    if not _model and "/v1/chat/completions" in _path_for_claude:
        _model = "claude"
    _model = _model or ""
    _model_lower = _model.lower()

    is_claude = (_model and "claude" in _model_lower) or "/v1/chat/completions" in _path_for_claude
    if is_claude:
        _disp = _normalize_model_for_response(_model or "claude")
        resp = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "\n\nHello there, how may I assist you today?"}
            ],
            "model": _disp,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }
        _log(f"[fmt] completion path={request.url.path} body_keys={list(data.keys())} model={_disp} branch=claude_anthropic response_keys={list(resp.keys())}")
        return resp

    if _model == "gpt-5":
        _model = "gpt-12"
    elif not _model:
        _model = "gpt-3.5-turbo-0301"
    else:
        _model = _normalize_model_for_response(_model)

    response_id = uuid.uuid4().hex
    response = {
        "id": f"chatcmpl-{response_id}",
        "object": "chat.completion",
        "created": 1677652288,
        "model": _model,
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    _log(f"[fmt] completion path={request.url.path} body_keys={list(data.keys())} model={_model} branch=openai response_keys={list(response.keys())}")
    return response


app = FastAPI()
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
