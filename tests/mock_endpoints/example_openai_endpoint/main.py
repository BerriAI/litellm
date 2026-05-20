from fastapi import FastAPI, Request, status, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, UploadFile, Form
from fastapi.responses import StreamingResponse, Response, PlainTextResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import asyncio
import os
import time
import random
import warnings
import logging
from dotenv import load_dotenv
from slowapi import Limiter
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel

from batch_and_files_api import router as batch_files_router

from slowapi import Limiter, _rate_limit_exceeded_handler


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
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# Preloaded audio in three sizes: small, medium, large
# Structure: _PRELOADED_AUDIO[format][size] = bytes
_PRELOADED_AUDIO: Dict[str, Dict[str, bytes]] = {}

# Size definitions in bytes
AUDIO_SIZES = {
    "small": 20_000,   # ~20 KB - for short inputs (< 100 chars)
    "medium": 100_000, # ~100 KB - for medium inputs (100-1000 chars)
    "large": 400_000,  # ~400 KB - for large inputs (> 1000 chars)
}


def generate_minimal_audio(format: str = "mp3") -> bytes:
    """Generate minimal valid audio data for different formats."""
    if format == "mp3":
        # Minimal valid MP3 frame (silent audio, ~1 second)
        # This is a minimal MP3 header + frame
        mp3_header = bytes([
            0xFF, 0xFB, 0x90, 0x00,  # MP3 sync word + header
        ])
        # Add some minimal frame data using a non-zero pattern
        frame_data = bytes([0x55] * 100)  # 0x55 pattern instead of all zeros
        return mp3_header + frame_data
    elif format == "opus":
        # Minimal Opus header (OggS)
        opus_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
        # Use a simple non-zero pattern for payload
        return opus_header + bytes([0x33] * 50)
    elif format == "aac":
        # Minimal AAC header
        aac_header = bytes([0xFF, 0xF1])  # ADTS sync word
        # Non-zero pattern payload
        return aac_header + bytes([0x77] * 50)
    elif format == "flac":
        # Minimal FLAC header (fLaC)
        flac_header = b"fLaC"
        # Non-zero pattern payload
        return flac_header + bytes([0x99] * 50)
    elif format == "pcm":
        # Minimal PCM WAV header
        # WAV header structure
        wav_header = (
            b"RIFF" +  # ChunkID
            (36).to_bytes(4, byteorder="little") +  # ChunkSize
            b"WAVE" +  # Format
            b"fmt " +  # Subchunk1ID
            (16).to_bytes(4, byteorder="little") +  # Subchunk1Size
            (1).to_bytes(2, byteorder="little") +  # AudioFormat (PCM)
            (1).to_bytes(2, byteorder="little") +  # NumChannels
            (16000).to_bytes(4, byteorder="little") +  # SampleRate
            (32000).to_bytes(4, byteorder="little") +  # ByteRate
            (2).to_bytes(2, byteorder="little") +  # BlockAlign
            (16).to_bytes(2, byteorder="little") +  # BitsPerSample
            b"data" +  # Subchunk2ID
            (0).to_bytes(4, byteorder="little")  # Subchunk2Size
        )
        return wav_header
    else:
        # Default to MP3
        return generate_minimal_audio("mp3")


def generate_audio_by_size(format: str, target_size: int, speed: float = 1.0) -> bytes:
    """
    Generate audio data of approximately the target size.
    
    Args:
        format: Audio format (mp3, opus, aac, flac, pcm)
        target_size: Target size in bytes
        speed: Speech speed (affects duration, so faster = smaller for same text)
    
    Returns:
        bytes: Audio data approximately matching target_size
    """
    # Adjust target size based on speed (faster speed = shorter duration = smaller file)
    # Speed affects duration linearly, so we divide by speed
    adjusted_size = int(target_size / speed)
    
    # Get the minimal header/base for this format
    base = generate_minimal_audio(format)
    base_size = len(base)
    
    # If target is smaller than base, return base (minimum valid audio)
    if adjusted_size <= base_size:
        return base
    
    # Calculate how much payload we need
    payload_size = adjusted_size - base_size
    
    # Generate payload data with a pattern that looks like audio
    # Use a repeating pattern that varies to avoid compression artifacts
    pattern = [0x55, 0xAA, 0x33, 0xCC, 0x66, 0x99, 0x11, 0xEE]
    payload = bytes([pattern[i % len(pattern)] for i in range(payload_size)])
    
    return base + payload


def _load_preloaded_audio() -> None:
    """
    Preload three sizes (small, medium, large) of audio for each format.
    This is done once at startup to avoid per-request generation overhead.
    """
    global _PRELOADED_AUDIO
    
    # Formats we support for the speech endpoint
    formats = ["mp3", "opus", "aac", "flac", "pcm"]
    
    for fmt in formats:
        _PRELOADED_AUDIO[fmt] = {}
        for size_name, target_size in AUDIO_SIZES.items():
            # Generate audio of the target size
            audio_data = generate_audio_by_size(fmt, target_size, speed=1.0)
            _PRELOADED_AUDIO[fmt][size_name] = audio_data


# Initialize preloaded audio at import time
_load_preloaded_audio()


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
        # Normal distribution around 10 seconds, std dev of 2
        sleep_time = max(1.0, random.gauss(10, 2))
        return sleep_time
    elif rand < 0.95:  # 25% of requests: ~60 seconds
        # Normal distribution around 60 seconds, std dev of 5
        sleep_time = max(30.0, random.gauss(60, 5))
        return sleep_time
    else:  # 5% of requests: 60+ seconds (up to 5 minutes)
        # Uniform distribution between 60 and 300 seconds
        sleep_time = random.uniform(60, 300)
        return sleep_time


def get_request_url(request: Request):
    return str(request.url)


def format_detailed_error(
    error: Exception,
    request: Request,
    context: str = "",
    include_body: bool = False
) -> dict:
    """Format a detailed error response with request context"""
    error_detail = {
        "error": {
            "message": str(error),
            "type": type(error).__name__,
            "context": context,
            "request": {
                "path": request.url.path,
                "method": request.method,
                "query": str(request.url.query) if request.url.query else None,
            }
        }
    }
    
    # Clearer message for KeyError (e.g. 'content' or 'candidates') to aid debugging format mismatches
    if isinstance(error, KeyError):
        key = str(error).strip("'\"")
        error_detail["error"]["message"] = (
            f"Missing required key '{key}' in response. "
            "Claude/Anthropic models expect a top-level 'content' array; Gemini models expect 'candidates'. "
            f"This may indicate a response-format mismatch. Original: {error}."
        )
    
    # Add sanitized headers (mask sensitive values)
    headers_dict = dict(request.headers)
    sanitized_headers = {}
    for key, value in headers_dict.items():
        if key.lower() in ["authorization", "api-key", "x-api-key", "cookie"]:
            # Show only first 20 chars of sensitive headers
            sanitized_headers[key] = f"{value[:20]}..." if len(value) > 20 else value
        else:
            sanitized_headers[key] = value
    error_detail["error"]["request"]["headers"] = sanitized_headers
    
    # Optionally include request body preview
    if include_body:
        try:
            # Try to get body without consuming it (if already read)
            # This might not work if body was already consumed, but worth trying
            pass  # Body reading is async, so we'll do it in the caller if needed
        except:
            pass
    
    return error_detail


async def get_error_detail_with_body(
    error: Exception,
    request: Request,
    context: str = ""
) -> dict:
    """Get detailed error response including request body if possible"""
    error_detail = format_detailed_error(error, request, context, include_body=False)
    
    # Try to get request body
    try:
        body = await request.body()
        if body:
            body_str = body.decode('utf-8', errors='ignore')
            # Limit body size in error response (first 500 chars)
            error_detail["error"]["request"]["body_preview"] = body_str[:500]
            if len(body_str) > 500:
                error_detail["error"]["request"]["body_preview"] += "... (truncated)"
    except Exception:
        error_detail["error"]["request"]["body_preview"] = "Unable to read request body"
    
    return error_detail


def _validate_response_format(response: dict, model: Optional[str], is_gemini: bool) -> Optional[dict]:
    """
    Validate that the response matches the expected format for the model.
    Returns an error dict for HTTP 500 if invalid, None if ok.
    Logs clearly on the server when a format mismatch or structural bug is detected.
    """
    model_lower = (model or "").lower()
    has_candidates = "candidates" in response
    has_content = "content" in response and isinstance(response.get("content"), list)

    # Misclassification: Claude model but we built Gemini format (causes KeyError: 'content' in LiteLLM)
    if "claude" in model_lower and has_candidates and not has_content:
        log.error(
            "Format misclassification: Claude model '%s' received Gemini format (candidates, no top-level content). "
            "Response keys: %s. This will cause KeyError: 'content' in LiteLLM Anthropic transformation.",
            model, list(response.keys())
        )
        return {
            "error": {
                "message": (
                    f"Response format mismatch: Model '{model}' is a Claude/Anthropic model but the server produced "
                    "a Gemini-format response (with 'candidates' instead of 'content'). "
                    "The LiteLLM/Anthropic client expects a top-level 'content' array. "
                    "This is a format-detection bug on the server. Please report."
                ),
                "code": "FORMAT_MISMATCH_CLAUDE_GOT_GEMINI",
            }
        }
    # Misclassification: Gemini model but we built Anthropic format
    if "gemini" in model_lower and has_content and not has_candidates:
        log.error(
            "Format misclassification: Gemini model '%s' received Anthropic format (content, no candidates). "
            "Response keys: %s.",
            model, list(response.keys())
        )
        return {
            "error": {
                "message": (
                    f"Response format mismatch: Model '{model}' is a Gemini model but the server produced "
                    "an Anthropic-format response (with 'content' instead of 'candidates'). "
                    "The Vertex/Gemini client expects 'candidates'. This is a format-detection bug on the server. Please report."
                ),
                "code": "FORMAT_MISMATCH_GEMINI_GOT_ANTHROPIC",
            }
        }

    # Structural validation: required top-level key must exist
    if is_gemini:
        if not has_candidates:
            log.error("Invalid Gemini response: missing 'candidates'. Model: %s, response keys: %s", model, list(response.keys()))
            return {
                "error": {
                    "message": f"Response format error: Gemini format requires a top-level 'candidates' array. Model: {model}. This is a server-side bug.",
                    "code": "INVALID_GEMINI_RESPONSE",
                }
            }
    else:
        if not has_content:
            log.error("Invalid Anthropic response: missing or invalid 'content'. Model: %s, response keys: %s", model, list(response.keys()))
            return {
                "error": {
                    "message": f"Response format error: Anthropic/Claude format requires a top-level 'content' array. Model: {model}. This is a server-side bug.",
                    "code": "INVALID_ANTHROPIC_RESPONSE",
                }
            }
    return None


limiter = Limiter(key_func=get_request_url)
load_dotenv()

# Suppress python-multipart's "Skipping data after last boundary" warning
# This warning appears even in patched versions (0.0.18+) where the DoS vulnerability is fixed.
# The warning is harmless but clutters logs. We suppress it via logging configuration.
# Note: Upgraded to python-multipart 0.0.20 which has better handling, but warning may still appear.
logging.getLogger("multipart").setLevel(logging.ERROR)  # Only show errors, not warnings

log = logging.getLogger(__name__)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a clear, non-leaking error to the client; log the full traceback on the server."""
    if isinstance(exc, HTTPException):
        raise exc
    log.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error. The request could not be processed. Check server logs for details.",
                "code": "INTERNAL_ERROR",
            }
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the batch and files router with /v1 prefix
app.include_router(batch_files_router, prefix="/v1")


def data_generator(model=None):
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
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"


# for completion
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/openai/deployments/{model:path}/chat/completions")  # azure compatible endpoint
async def completion(request: Request, authorization: str = Header(None)):
    # Accept both Authorization: Bearer and x-goog-api-key (optional for OpenAI-compatible endpoints)
    # This allows Gemini models routed through /chat/completions to work
    has_valid_auth(request, authorization)
    
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))

    data = await request.json()
    data = data if isinstance(data, dict) else {}

    if data.get("model") == "429":
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")

    if data.get("model") == "random_sleep":
        # sleep for a random time between 1 and 10 seconds
        sleep_time = random.randint(1, 10)
        print("sleeping for " + str(sleep_time) + " seconds")
        await asyncio.sleep(sleep_time)
    
    # Degraded provider simulation: sleep with histogram distribution to simulate hanging requests
    # This causes event loop blocking, file descriptor accumulation, and tests LiteLLM's
    # behavior under degraded provider conditions
    if data.get("model") in ["degraded", "slow_provider", "blocked"]:
        # Histogram distribution: most ~10s, some ~60s, few 60s+
        sleep_time = get_histogram_sleep_time()
        print(f"[DEGRADED MODE] Sleeping for {sleep_time:.1f} seconds ({sleep_time/60:.1f} minutes) - histogram distribution")
        await asyncio.sleep(sleep_time)
    # For /v1/chat/completions, LiteLLM's Vertex AI Anthropic expects a body with "content".
    # data_generator() yields OpenAI format (choices/delta); returning it causes KeyError: 'content'.
    # So when path is /v1/chat/completions, do NOT stream—fall through to return Anthropic JSON.
    _path_for_claude = request.url.path
    if data.get("stream") == True and "/v1/chat/completions" not in _path_for_claude:
        _stream_model = (
            data.get("litellm_model_id") or data.get("model") or data.get("model_name")
            or data.get("modelId") or data.get("model_id")
            or (getattr(request, "path_params", None) or {}).get("model")
            or "gpt-3.5-turbo-0125"
        )
        _stream_model = _normalize_model_for_response(
            _stream_model if isinstance(_stream_model, str) else "gpt-3.5-turbo-0125"
        )
        return StreamingResponse(
            content=data_generator(model=_stream_model),
            media_type="text/event-stream",
        )
    # Model: body (model, model_name, modelId, model_id, litellm_model_id), path (Azure), query,
    # generationConfig, metadata, or headers (X-Model, X-LiteLLM-Model). LiteLLM may put model
    # in different places when calling /v1/chat/completions for Vertex AI Anthropic.
    _path_params = getattr(request, "path_params", None) or {}
    _model = (
        data.get("model")
        or data.get("model_name")
        or data.get("modelId")
        or data.get("model_id")
        or data.get("litellm_model_id")
        or _path_params.get("model")
        or (getattr(request, "query_params", None) or {}).get("model")
        or (data.get("generationConfig") or {}).get("model")
        or (data.get("metadata") or {}).get("model")
        or (data.get("metadata") or {}).get("model_name")
        or request.headers.get("X-Model")
        or request.headers.get("x-model")
        or request.headers.get("X-LiteLLM-Model")
        or request.headers.get("x-litellm-model")
        or ""
    )
    # Coerce to str so .lower() is safe; non-string (e.g. dict) treated as no model.
    _model = _model if isinstance(_model, str) else ""
    # Vertex-style body ("contents") without "model" or "messages": LiteLLM may omit model;
    # assume Anthropic so we return "content" and avoid KeyError in anthropic transformation.
    if not _model and "contents" in data and "messages" not in data:
        _model = "claude"
    # LiteLLM routes Vertex AI Anthropic (Claude) to /v1/chat/completions and may omit model
    # in the body. If we still have no model and path is exactly /v1/chat/completions,
    # assume Claude so we return Anthropic format (content) and avoid KeyError.
    if not _model and "/v1/chat/completions" in _path_for_claude:
        _model = "claude"
    _model = _model or ""
    _model_lower = _model.lower()

    # When LiteLLM routes Claude (Vertex AI Anthropic) to /v1/chat/completions, it
    # expects Anthropic Messages API format (top-level "content" array). OpenAI format
    # (choices/usage) causes KeyError: 'content' in litellm anthropic transformation.
    # Also: path /v1/chat/completions is used by LiteLLM for Vertex AI Anthropic; the
    # model in the body may be omitted or not contain "claude", so treat that path as Claude.
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
        print(f"[fmt] completion path={request.url.path} body_keys={list(data.keys())} model={_disp} branch=claude_anthropic response_keys={list(resp.keys())}")
        return resp

    if _model == "gpt-5":
        _model = "gpt-12"
    elif not _model:
        _model = "gpt-3.5-turbo-0301"  # fallback when no model in request
    else:
        _model = _normalize_model_for_response(_model)  # e.g. gpt-4o-mini-data-zone -> gpt-4o-mini
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
    print(f"[fmt] completion path={request.url.path} body_keys={list(data.keys())} model={_model} branch=openai response_keys={list(response.keys())}")
    return response


# for completion
@app.post("/completions")
@app.post("/v1/completions")
async def text_completion(request: Request):
    data = await request.json()
    _model = (
        data.get("litellm_model_id") or data.get("model") or data.get("model_name")
        or data.get("modelId") or data.get("model_id") or "unknown"
    )
    _model = _normalize_model_for_response(_model if isinstance(_model, str) else "unknown")

    if data.get("stream") == True:
        return StreamingResponse(
            content=data_generator(model=_model),
            media_type="text/event-stream",
        )
    else:
        response_id = uuid.uuid4().hex
        response = {
            "id": "cmpl-9B2ycsf0odECdLmrVzm2y8Q12csjW",
            "choices": [
                {
                "finish_reason": "length",
                "index": 0,
                "logprobs": None,
                "text": "\n\nA test request, how intriguing\nAn invitation for knowledge bringing\nWith words"
                }
            ],
            "created": 1712420078,
            "model": _model,
            "object": "text_completion",
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": 16,
                "prompt_tokens": 10,
                "total_tokens": 26
            }
        }

        return response




# for completion
@app.post("/invocations")
@app.post("/invocations/")
async def invocation(request: Request):
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))
    data = await request.json()
    if data.get("model") == "429":
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
    else:
        response_id = uuid.uuid4().hex
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

@app.post("/embeddings")
@app.post("/v1/embeddings")
@app.post("/openai/deployments/{model:path}/embeddings")  # azure compatible endpoint
async def embeddings(request: Request, authorization: str = Header(None)):
    # Accept both Authorization: Bearer and x-goog-api-key (optional for OpenAI-compatible endpoints)
    # This allows Gemini models routed through /embeddings to work
    has_valid_auth(request, authorization)
    
    _small_embedding = [
        -0.006929283495992422,
        -0.005336422007530928,
        -4.547132266452536e-05,
        -0.024047505110502243,
    ]

    big_embedding = _small_embedding * 100
    return {
        "object": "list",
        "data": [
            {
            "object": "embedding",
            "index": 0,
            "embedding": big_embedding
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """OpenAI Audio Speech endpoint - Text to Speech"""
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))

    data = await request.json()
    
    # Extract parameters
    model = data.get("model", "tts-1")
    input_text = data.get("input", "")
    voice = data.get("voice", "alloy")
    response_format = data.get("response_format", "mp3")
    speed = data.get("speed", 1.0)
    
    # Validate required parameters
    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameter: input"
        )
    
    # Validate voice
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid voice. Must be one of: {', '.join(valid_voices)}"
        )
    
    # Validate response_format
    valid_formats = ["mp3", "opus", "aac", "flac", "pcm"]
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid response_format. Must be one of: {', '.join(valid_formats)}"
        )
    
    # Validate speed
    if not (0.25 <= speed <= 4.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="speed must be between 0.25 and 4.0"
        )
    
    # Select audio size based on input text length
    # This allows testing different response sizes without dynamic generation overhead
    text_length = len(input_text)
    
    if text_length < 100:
        size_key = "small"
    elif text_length < 1000:
        size_key = "medium"
    else:
        size_key = "large"
    
    # Get preloaded audio data for the selected size
    audio_data = _PRELOADED_AUDIO.get(response_format, {}).get(size_key)
    if audio_data is None:
        # Fallback to on-demand minimal generation if something is missing
        audio_data = generate_minimal_audio(response_format)
    
    # Adjust for speed parameter (faster speed = smaller file)
    # For simplicity, we'll just use the preloaded data as-is
    # In a real implementation, speed would affect the audio duration
    # Optional: Log the size for testing/debugging (can be enabled via env var)
    if os.getenv("LOG_AUDIO_SIZES", "").lower() == "true":
        print(f"[Audio Speech] Input length: {text_length} chars, Format: {response_format}, "
              f"Size: {size_key}, Actual size: {len(audio_data)} bytes, Speed: {speed}")
    
    # Set appropriate content type
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/wav"
    }
    content_type = content_types.get(response_format, "audio/mpeg")
    
    # Return binary audio response with preloaded size based on input text length
    return Response(
        content=audio_data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{response_format}"'
        },
    )


# Pre-computed responses for maximum speed
_TRANSCRIPTION_TEXT = "This is a mock transcription of the audio file. The audio has been processed and transcribed to text."
_TRANSCRIPTION_JSON = {"text": _TRANSCRIPTION_TEXT}
_TRANSCRIPTION_VERBOSE_JSON = {
    "text": _TRANSCRIPTION_TEXT,
    "language": "en",
    "duration": 5.0,
    "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": _TRANSCRIPTION_TEXT}]
}
_TRANSCRIPTION_SRT = f"1\n00:00:00,000 --> 00:00:05,000\n{_TRANSCRIPTION_TEXT}\n\n"
_TRANSCRIPTION_VTT = f"WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n{_TRANSCRIPTION_TEXT}\n"

@app.post("/audio/transcriptions")
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile,
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json")
):
    """OpenAI Audio Transcriptions endpoint - Speech to Text"""
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))

    # Minimal validation - only check critical errors
    if model != "whisper-1":
        raise HTTPException(status_code=400, detail="Invalid model. Must be one of: whisper-1")
    
    # Fast path: return pre-computed responses based on format
    # Use simple dict returns (FastAPI auto-encodes to JSON) for maximum speed
    if response_format == "text":
        return PlainTextResponse(content=_TRANSCRIPTION_TEXT)
    elif response_format == "srt":
        return PlainTextResponse(content=_TRANSCRIPTION_SRT, media_type="text/plain")
    elif response_format == "vtt":
        return PlainTextResponse(content=_TRANSCRIPTION_VTT, media_type="text/vtt")
    elif response_format == "verbose_json":
        # Only modify language if different from default
        if language and language != "en":
            return {"text": _TRANSCRIPTION_TEXT, "language": language, "duration": 5.0, "segments": _TRANSCRIPTION_VERBOSE_JSON["segments"]}
        return _TRANSCRIPTION_VERBOSE_JSON
    else:  # json or default
        return _TRANSCRIPTION_JSON




@app.post("/triton/embeddings")
async def embeddings(request: Request):
    try:
        input_data = await request.json()
        assert "inputs" in input_data

        inputs = input_data["inputs"]
        element_one = inputs[0]

        assert "name" in element_one, "Missing name in inputs"
        assert "shape" in element_one, "Missing shape in inputs"
        assert "datatype" in element_one, "Missing datatype in inputs"
        assert "data" in element_one, "Missing data in inputs"


    except (ValueError, KeyError) as e:
        return HTTPException(status_code=400, detail=str(e))

    output_data = {
        "model_name": "triton-embeddings",
        "model_version": "1",
        "parameters": {
            "sequence_id": 0,
            "sequence_start": False,
            "sequence_end": False
        },
        "outputs": [
            {
                "name": "embedding_output",
                "datatype": "FP32",
                "shape": [2, 2],
                "data": [0.1, 0.2]  # Replace with actual output data
            }
        ]
    }

    return output_data


@app.post("/openai/fine_tuning/jobs")  # azure compatible endpoint
async def fine_tuning(request: Request):
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)

    print("inside fine tuning /jobs endpoint")
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))

    data = await request.json()

    if data.get("model") == "429":
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
    
    print("got request=" + json.dumps(data))

    return {
        "object": "fine_tuning.job",
        "id": "ftjob-abc123",
        "model": "davinci-002",
        "created_at": 1692661014,
        "finished_at": 1692661190,
        "fine_tuned_model": "ft:davinci-002:my-org:custom_suffix:7q8mpxmy",
        "organization_id": "org-123",
        "result_files": [
            "file-abc123"
        ],
        "status": "succeeded",
        "validation_file": None,
        "training_file": "file-abc123",
        "hyperparameters": {
            "n_epochs": 4,
            "batch_size": 1,
            "learning_rate_multiplier": 1.0
        },
        "trained_tokens": 5768,
        "integrations": [],
        "seed": 0,
        "estimated_finish": 0
    }


@app.get("/openai/fine_tuning/jobs")  # azure compatible endpoint
async def list_fine_tuning(request: Request):
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)

    return {
        "object": "list",
        "data": [
            {
            "object": "fine_tuning.job.event",
            "id": "ft-event-TjX0lMfOniCZX64t9PUQT5hn",
            "created_at": 1689813489,
            "level": "warn",
            "message": "Fine tuning process stopping due to job cancellation",
            "data": None,
            "type": "message"
            },
        ], "has_more": True
    }



@app.post("/openai/fine_tuning/jobs/{fine_tuning_job_id:path}/cancel")  # azure compatible endpoint
async def cancel_fine_tuning(request: Request):
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)

    return {
        "object": "fine_tuning.job",
        "id": "ftjob-abc123",
        "model": "gpt-4o-mini-2024-07-18",
        "created_at": 1721764800,
        "fine_tuned_model": None,
        "organization_id": "org-123",
        "result_files": [],
        "hyperparameters": {
            "n_epochs":  "auto"
        },
        "status": "cancelled",
        "validation_file": "file-abc123",
        "training_file": "file-abc123"
    }





@app.post("/openai/files")  # azure compatible endpoint
async def openai_files(request: Request):
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)

    print("inside fine tuning /jobs endpoint")
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))


    return {
        "id": "file-abc123",
        "object": "file",
        "bytes": 120000,
        "created_at": 1677610602,
        "filename": "mydata.jsonl",
        "purpose": "fine-tune",
    }


### FAKE BEDROCK ENDPOINT ### 

@app.post("/model/{modelId}/converse")
async def fake_bedrock_endpoint(request: Request):
    return {"metrics":{"latencyMs":393},"output":{"message":{"content":[{"text":"Good morning to you too! I am not Claude, however. Claude is a large language model trained by Google, while I am Gemini, a multi-modal AI model, developed by Google as well. Is there anything I can help you with today?"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":37,"outputTokens":8,"totalTokens":45}}

@app.post("/model/{modelId}/invoke")
async def fake_bedrock_invoke(request: Request, modelId: str):
    """Bedrock invoke endpoint for text generation and embeddings"""
    data = await request.json()
    modelId_lower = modelId.lower()
    
    print(f"[bedrock_invoke] modelId={modelId}, modelId_lower={modelId_lower}")
    
    # Check for Mistral models FIRST (before embedding check)
    # Mistral model IDs: mistral.mistral-7b-instruct-v0:2, mistral.mixtral-8x7b-instruct-v0:1
    if "mistral" in modelId_lower:
        print(f"[bedrock_invoke] Detected Mistral model, returning outputs format")
        # Mistral models expect "outputs" array with "text" and "stop_reason"
        # Reference: litellm/llms/bedrock/chat/invoke_handler.py line 657-661
        return {
            "outputs": [
                {
                    "text": "This is a mock response from Bedrock Mistral model.",
                    "stop_reason": "stop"
                }
            ]
        }
    # Check if this is an embedding model
    elif "embed" in modelId_lower or "titan-embed" in modelId_lower:
        print(f"[bedrock_invoke] Detected embedding model")
        # Return embedding format
        input_text = data.get("inputText", data.get("text", ""))
        if isinstance(input_text, list):
            input_text = " ".join(str(t) for t in input_text)
        
        # Generate fake embedding (768 dimensions for most models, 1024 for Cohere v3)
        embedding_dim = 1024 if "cohere.embed" in modelId_lower else 768
        embedding = [random.uniform(-0.15, 0.15) for _ in range(embedding_dim)]
        
        # Cohere models expect "embeddings" (plural) in an array
        # Model IDs: cohere.embed-english-v3, cohere.embed-multilingual-v3
        if "cohere.embed" in modelId_lower:
            print(f"[bedrock_invoke] Detected Cohere embedding model, returning embeddings array")
            return {
                "embeddings": [embedding],
                "id": f"embed_{uuid.uuid4().hex}",
                "response_type": "embeddings_floats"
            }
        else:
            # Titan and other embedding models expect "embedding" (singular)
            print(f"[bedrock_invoke] Detected Titan/other embedding model, returning embedding format")
            return {
                "embedding": embedding,
                "inputTextTokenCount": len(input_text.split()) if input_text else 0
            }
    else:
        # Return text generation format for other models (Amazon Titan text, etc.)
        print(f"[bedrock_invoke] Default to Amazon Titan text format with results")
        return {
            "results": [
                {
                    "outputText": "This is a mock response from Bedrock.",
                    "completionReason": "FINISH",
                    "tokenCount": {
                        "inputTokens": 10,
                        "outputTokens": 8,
                        "totalTokens": 18
                    }
                }
            ]
        }

### FAKE VERTEX ENDPOINT ### 

def validate_google_auth(request: Request, authorization: str = None, required: bool = True):
    """Validate authentication for Google endpoints - accepts both Bearer token and x-goog-api-key"""
    # Check Authorization header (Vertex AI format)
    if authorization and authorization.startswith("Bearer "):
        return True
    
    # Check x-goog-api-key header (Gemini API format)
    api_key = request.headers.get("x-goog-api-key") or request.headers.get("X-Goog-Api-Key")
    if api_key:
        return True
    
    # No valid auth found
    if required:
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization header. Use 'Authorization: Bearer <token>' or 'x-goog-api-key: <key>'")
    return False

def has_valid_auth(request: Request, authorization: str = None) -> bool:
    """Check if request has valid authentication (either format) - returns True/False without raising"""
    try:
        return validate_google_auth(request, authorization, required=False)
    except HTTPException:
        return False

@app.post("/generateContent")
@app.post("/v1/projects/adroit-crow-413218/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision-001:generateContent")
@app.post("/v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision-001:generateContent")
@app.post("/v1beta/models/gemini-1.5-flash:generateContent")
async def generate_content(request: Request, authorization: str = Header(None)):
    validate_google_auth(request, authorization)

    data = await request.json()
    
    request_path = request.url.path
    
    # Extract model name from path
    model = None
    if "/models/" in request_path:
        try:
            model_part = request_path.split("/models/")[1].split(":")[0]
            if model_part:
                model = model_part
        except:
            pass
    
    # Also check request body for model information (fallback). Prefer litellm_model_id (client-facing).
    if isinstance(data, dict):
        model = (
            data.get("litellm_model_id") or model
            or data.get("model") or data.get("model_name") or data.get("modelId") or data.get("model_id")
        )
    
    # Determine if this is a Gemini model: check path OR model name
    # Claude models MUST get Anthropic format (with 'content'). Gemini gets Gemini format (with 'candidates', 'usageMetadata').
    # Check path (only /models/gemini or /models/claude to avoid false positives from project IDs)
    request_path_lower = request_path.lower()
    path_has_gemini = "/models/gemini" in request_path_lower
    path_has_claude = "/models/claude" in request_path_lower
    
    # Check model name from path or body
    model_lower = (model or "").lower()
    model_has_gemini = "gemini" in model_lower if model else False
    model_has_claude = "claude" in model_lower if model else False
    
    # Claude models MUST get Anthropic format. Gemini MUST get Gemini format (usageMetadata, candidates).
    # When model/path give no signal: default to Gemini (Vertex generateContent is mostly Gemini; returning
    # Anthropic for Gemini causes "usageMetadata not found" in LiteLLM).
    is_gemini = (path_has_gemini or model_has_gemini or (not path_has_claude and not model_has_claude)) and not (path_has_claude or model_has_claude)
    
    print(f"[fmt] generate_content path={request_path} model={model} is_gemini={is_gemini} path_g={path_has_gemini} path_c={path_has_claude} model_g={model_has_gemini} model_c={model_has_claude}")
    
    # CRITICAL: Gemini models MUST get Gemini format, Anthropic models MUST get Anthropic format
    if is_gemini:
        # Return Gemini format - DO NOT return Anthropic format for Gemini models
        # Skip to Gemini format section below
        pass
    else:
        # Return Anthropic format for Anthropic/Claude models only (we only reach here when we detected Claude)
        # Anthropic Messages API format - MUST include 'content' array and 'usage' (not usageMetadata)
        response = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [  # REQUIRED: This field must be present and be an array
                {
                    "type": "text",
                    "text": "Hello! This is a mock response from the Vertex AI Anthropic endpoint. I'm processing your request."
                }
            ],
            "model": _normalize_model_for_response(model or "unknown"),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        }
        # Ensure content field exists
        if 'content' not in response or not isinstance(response.get('content'), list):
            response['content'] = [{"type": "text", "text": "Hello! This is a mock response from the Vertex AI Anthropic endpoint. I'm processing your request."}]
        
        err = _validate_response_format(response, model, False)
        if err:
            return JSONResponse(status_code=500, content=err)
        print(f"[fmt] generate_content branch=anthropic response_keys={list(response.keys())}")
        return response
    
    # Return Vertex AI Gemini format (only reached if is_gemini is True)
    # IMPORTANT: Gemini format requires "usageMetadata" (not "usage")
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": f"Hello! This is a mock response from Vertex AI Gemini endpoint. Model: {model or 'gemini'}"
                        }
                    ]
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.037353516,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.03515625
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.017944336,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.020019531
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.06738281,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.03173828
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.11279297,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.057373047
                    }
                ],
                "avgLogprobs": -0.30250951355578853
            }
        ],
        "usageMetadata": {  # REQUIRED: Gemini format uses "usageMetadata" (not "usage")
            "promptTokenCount": 5,
            "candidatesTokenCount": 51,
            "totalTokenCount": 56
        }
    }

    err = _validate_response_format(response, model, True)
    if err:
        return JSONResponse(status_code=500, content=err)
    print(f"[fmt] generate_content branch=gemini response_keys={list(response.keys())}")
    return response


import random

request_counter = 0

@app.post("/generateContent")
@app.post("/v1/projects/bad-adroit-crow-413218/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision-001:generateContent")
@limiter.limit("10000/minute")
async def generate_content_bad(request: Request, authorization: str = Header(None)):
    global request_counter
    request_counter += 1

    validate_google_auth(request, authorization)

    # Raise an error for every 200th request
    if request_counter % 200 == 0:
        raise HTTPException(status_code=500, detail="Internal Server Error: Simulated error for every 200th request")

    # Introduce a 0.5% chance of error for other requests
    if random.random() < 0.005:
        raise HTTPException(status_code=500, detail="Internal Server Error: Random error (0.5% chance)")

    data = await request.json()
    
    # You can process the input data here if needed
    # For now, we'll just return the hardcoded response

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "Good morning to you too! I am not Claude, however. Claude is a large language model trained by Google, while I am Gemini, a multi-modal AI model, developed by Google as well. Is there anything I can help you with today?"
                        }
                    ]
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.037353516,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.03515625
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.017944336,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.020019531
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.06738281,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.03173828
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.11279297,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.057373047
                    }
                ],
                "avgLogprobs": -0.30250951355578853
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 51,
            "totalTokenCount": 56
        }
    }

    return response


@app.post("/models/{model}:batchEmbedContents")
async def gemini_batch_embed_contents(request: Request, model: str, authorization: str = Header(None)):
    """Gemini batch embedding endpoint"""
    validate_google_auth(request, authorization)
    
    data = await request.json()
    
    # Extract requests from the body
    requests_list = data.get("requests", [])
    if not requests_list:
        raise HTTPException(status_code=400, detail="Missing 'requests' field in body")
    
    # Generate embeddings for each request
    embeddings = []
    for req in requests_list:
        # Extract text from the request
        text = ""
        if "text" in req:
            text = req["text"]
        elif "content" in req:
            content = req["content"]
            if isinstance(content, list) and len(content) > 0:
                if "parts" in content[0]:
                    parts = content[0]["parts"]
                    if isinstance(parts, list) and len(parts) > 0:
                        text = parts[0].get("text", "")
        
        # Generate fake embedding (768 dimensions)
        embedding = [random.uniform(-0.15, 0.15) for _ in range(768)]
        
        embeddings.append({
            "values": embedding
        })
    
    return {
        "embeddings": embeddings
    }


@app.post("/predict")
@app.post("/rawPredict")
@app.post("/v1/projects/adroit-crow-413218/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict")
@app.post("/v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict")
async def predict(request: Request, authorization: str = Header(None)):
    validate_google_auth(request, authorization)

    data = await request.json()
    
    # Extract model from path, body, or headers to determine if this is a Claude model
    request_path = request.url.path
    model_from_path = None
    if "/models/" in request_path:
        try:
            model_part = request_path.split("/models/")[1].split(":")[0]
            if model_part:
                model_from_path = model_part
        except:
            pass
    
    # Also check request body for model information (prefer litellm_model_id = client-facing name)
    model_from_body = None
    if isinstance(data, dict):
        model_from_body = (
            data.get("litellm_model_id") or data.get("model") or data.get("model_name")
            or data.get("modelId") or data.get("model_id")
        )
    
    # Check headers for model information (LiteLLM may pass model in headers)
    model_from_headers = (
        request.headers.get("X-Model") or
        request.headers.get("x-model") or
        request.headers.get("X-LiteLLM-Model") or
        request.headers.get("x-litellm-model")
    )
    
    # Determine final model (prefer body = client-facing name over path = deployment ID)
    final_model = model_from_body or model_from_path or model_from_headers or ""
    model_lower = (final_model or "").lower()
    
    # Check request body structure: Claude requests use "contents" (Anthropic format) or "messages" (chat format),
    # while embedding requests use "instances" (Vertex AI embedding format)
    has_contents = isinstance(data, dict) and "contents" in data
    has_messages = isinstance(data, dict) and "messages" in data
    has_instances = isinstance(data, dict) and "instances" in data
    # If it has messages or contents (but not instances), it's a Claude request
    body_suggests_claude = (has_contents or has_messages) and not has_instances
    
    # Check if this is a Claude model
    path_has_claude = "/models/claude" in request_path.lower()
    model_has_claude = "claude" in model_lower if final_model else False
    
    # For :rawPredict or :predict endpoints, if we can't determine the model and the request
    # doesn't have "instances" (embedding format), assume it's Claude since LiteLLM uses
    # these endpoints for Claude models via Vertex AI
    is_raw_predict_endpoint = ":rawPredict" in request_path or request_path.endswith("rawPredict") or ":predict" in request_path or request_path.endswith("predict")
    no_instances = not has_instances
    fallback_to_claude = is_raw_predict_endpoint and no_instances and not final_model
    
    # If body has "contents" but no "instances", it's likely a Claude request
    # Or if it's a rawPredict/predict endpoint without instances and no model detected, assume Claude
    is_claude = path_has_claude or model_has_claude or body_suggests_claude or fallback_to_claude
    
    # Log for debugging
    print(f"[predict] path={request_path} model={final_model} path_has_claude={path_has_claude} model_has_claude={model_has_claude} has_contents={has_contents} has_messages={has_messages} has_instances={has_instances} body_suggests_claude={body_suggests_claude} fallback_to_claude={fallback_to_claude} is_claude={is_claude}")
    
    # If this is a Claude model, return Anthropic format (not embedding format)
    if is_claude:
        response = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"Hello! This is a mock response from Vertex AI Anthropic endpoint via predict/rawPredict. Model: {final_model or 'claude'}"
                }
            ],
            "model": _normalize_model_for_response(final_model or "claude"),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        }
        # Ensure content field exists
        if 'content' not in response or not isinstance(response.get('content'), list):
            response['content'] = [{"type": "text", "text": f"Hello! This is a mock response from Vertex AI Anthropic endpoint via predict/rawPredict. Model: {final_model or 'claude'}"}]
        
        err = _validate_response_format(response, final_model or "claude", False)
        if err:
            return JSONResponse(status_code=500, content=err)
        print(f"[fmt] predict/rawPredict path={request_path} model={final_model} branch=anthropic response_keys={list(response.keys())}")
        return response
    
    # Otherwise, return embedding format (for embedding models)
    # Process the input data
    instances = data.get('instances', [])
    num_instances = len(instances)
    
    # Generate fake embeddings
    predictions = []
    for _ in range(num_instances):
        embedding = [random.uniform(-0.15, 0.15) for _ in range(768)]  # 768-dimensional embedding
        predictions.append({
            "embeddings": {
                "values": embedding,
                "statistics": {
                    "truncated": False,
                    "token_count": random.randint(4, 10)
                }
            }
        })

    # Calculate billable character count
    billable_character_count = sum(len(instance.get('content', '')) for instance in instances)

    response = {
        "predictions": predictions,
        "metadata": {
            "billableCharacterCount": billable_character_count
        }
    }

    return response


# Add catch-all routes for Vertex AI endpoints
# These must come AFTER the specific routes but handle any project/location/model combination

@app.post("/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent")
async def vertex_generate_content_catchall(request: Request, project: str, location: str, model: str, authorization: str = Header(None)):
    """Catch-all endpoint for Vertex AI generateContent - accepts any project/location/model"""
    validate_google_auth(request, authorization)

    data = await request.json()
    
    # Also check request body for model information (prefer litellm_model_id = client-facing name)
    model_from_body = None
    if isinstance(data, dict):
        model_from_body = (
            data.get("litellm_model_id") or data.get("model") or data.get("model_name")
            or data.get("modelId") or data.get("model_id")
        )
    
    # Use model from body (client-facing) or path
    final_model = model_from_body or model or ""
    
    # Model detection: only /models/gemini or /models/claude in path to avoid false positives (e.g. "gemini" in project ID)
    request_path_lower = request.url.path.lower()
    path_has_gemini = "/models/gemini" in request_path_lower
    path_has_claude = "/models/claude" in request_path_lower
    
    model_lower = (final_model or "").lower()
    model_has_gemini = "gemini" in model_lower if final_model else False
    model_has_claude = "claude" in model_lower if final_model else False
    
    # Claude models MUST get Anthropic format (content, usage). Gemini MUST get Gemini format (candidates, usageMetadata).
    # When model/path give no signal: default to Gemini (Vertex generateContent is mostly Gemini).
    is_gemini = (path_has_gemini or model_has_gemini or (not path_has_claude and not model_has_claude)) and not (path_has_claude or model_has_claude)
    
    print(f"[fmt] vertex_catchall path={request.url.path} model={final_model} is_gemini={is_gemini} path_g={path_has_gemini} path_c={path_has_claude} model_g={model_has_gemini} model_c={model_has_claude}")
    
    # CRITICAL: Gemini models MUST get Gemini format, Anthropic models MUST get Anthropic format
    if is_gemini:
        # Return Gemini format - DO NOT return Anthropic format for Gemini models
        # Skip to Gemini format section below
        pass
    else:
        # Return Anthropic format for Anthropic/Claude models only (we only reach here when we detected Claude)
        response = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"Hello! This is a mock response from Vertex AI Anthropic endpoint. Model: {final_model or model}"
                }
            ],
            "model": _normalize_model_for_response(final_model or model or "unknown"),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        }
        # Safety check: Ensure content field exists and is a list
        if 'content' not in response or not isinstance(response.get('content'), list):
            response['content'] = [{"type": "text", "text": f"Hello! This is a mock response from Vertex AI Anthropic endpoint. Model: {final_model or model}"}]
        
        err = _validate_response_format(response, final_model or model, False)
        if err:
            return JSONResponse(status_code=500, content=err)
        print(f"[fmt] vertex_catchall branch=anthropic response_keys={list(response.keys())}")
        return response
    
    # Return Vertex AI Gemini format (only reached if is_gemini is True)
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "Hello! This is a mock response from Vertex AI. Model: " + model
                        }
                    ]
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.037353516,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.03515625
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.017944336,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.020019531
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.06738281,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.03173828
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.11279297,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.057373047
                    }
                ],
                "avgLogprobs": -0.30250951355578853
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 51,
            "totalTokenCount": 56
        }
    }
    err = _validate_response_format(response, final_model or model, True)
    if err:
        return JSONResponse(status_code=500, content=err)
    print(f"[fmt] vertex_catchall branch=gemini response_keys={list(response.keys())}")
    return response


@app.post("/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict")
async def vertex_predict_catchall(request: Request, project: str, location: str, model: str, authorization: str = Header(None)):
    """Catch-all endpoint for Vertex AI predict (embeddings) - accepts any project/location/model"""
    validate_google_auth(request, authorization)

    data = await request.json()
    
    # Check headers for model information (LiteLLM may pass model in headers)
    model_from_headers = (
        request.headers.get("X-Model") or
        request.headers.get("x-model") or
        request.headers.get("X-LiteLLM-Model") or
        request.headers.get("x-litellm-model")
    )
    
    # Use model from body (prefer litellm_model_id), headers, or path
    model_from_body = None
    if isinstance(data, dict):
        model_from_body = (
            data.get("litellm_model_id") or data.get("model") or data.get("model_name")
            or data.get("modelId") or data.get("model_id")
        )
    final_model = model_from_body or model or model_from_headers or ""
    
    # Check request body structure: Claude requests use "contents" (Anthropic format) or "messages" (chat format),
    # while embedding requests use "instances" (Vertex AI embedding format)
    has_contents = isinstance(data, dict) and "contents" in data
    has_messages = isinstance(data, dict) and "messages" in data
    has_instances = isinstance(data, dict) and "instances" in data
    # If it has messages or contents (but not instances), it's a Claude request
    body_suggests_claude = (has_contents or has_messages) and not has_instances
    
    # Check if this is a Claude model
    model_lower = (final_model or "").lower()
    model_has_claude = "claude" in model_lower if final_model else False
    request_path = request.url.path
    path_has_claude = "/models/claude" in request_path.lower()
    
    # For :predict endpoints, if we can't determine the model and the request
    # doesn't have "instances" (embedding format), assume it's Claude since LiteLLM uses
    # these endpoints for Claude models via Vertex AI
    is_predict_endpoint = ":predict" in request_path or request_path.endswith("predict")
    no_instances = not has_instances
    fallback_to_claude = is_predict_endpoint and no_instances and not final_model
    
    is_claude = path_has_claude or model_has_claude or body_suggests_claude or fallback_to_claude
    
    # Log for debugging
    print(f"[vertex_predict_catchall] path={request_path} model={final_model or model} path_has_claude={path_has_claude} model_has_claude={model_has_claude} has_contents={has_contents} has_messages={has_messages} has_instances={has_instances} body_suggests_claude={body_suggests_claude} fallback_to_claude={fallback_to_claude} is_claude={is_claude}")
    
    # If this is a Claude model, return Anthropic format (not embedding format)
    if is_claude:
        response = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"Hello! This is a mock response from Vertex AI Anthropic endpoint via predict. Model: {final_model or model}"
                }
            ],
            "model": _normalize_model_for_response(final_model or model or "claude"),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        }
        # Ensure content field exists
        if 'content' not in response or not isinstance(response.get('content'), list):
            response['content'] = [{"type": "text", "text": f"Hello! This is a mock response from Vertex AI Anthropic endpoint via predict. Model: {final_model or model}"}]
        
        err = _validate_response_format(response, final_model or model, False)
        if err:
            return JSONResponse(status_code=500, content=err)
        print(f"[fmt] vertex_predict_catchall path={request_path} model={final_model or model} branch=anthropic response_keys={list(response.keys())}")
        return response
    
    # Otherwise, return embedding format (for embedding models)
    instances = data.get('instances', [])
    num_instances = len(instances)
    
    predictions = []
    for _ in range(num_instances):
        embedding = [random.uniform(-0.15, 0.15) for _ in range(768)]
        predictions.append({
            "embeddings": {
                "values": embedding,
                "statistics": {
                    "truncated": False,
                    "token_count": random.randint(4, 10)
                }
            }
        })

    billable_character_count = sum(len(instance.get('content', '')) for instance in instances)

    return {
        "predictions": predictions,
        "metadata": {
            "billableCharacterCount": billable_character_count
        }
    }


@app.post("/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:rawPredict")
async def vertex_raw_predict_catchall(request: Request, project: str, location: str, model: str, authorization: str = Header(None)):
    """Catch-all endpoint for Vertex AI rawPredict - accepts any project/location/model"""
    validate_google_auth(request, authorization)
    # rawPredict is similar to predict, so delegate to the same handler
    return await vertex_predict_catchall(request, project, location, model, authorization)


@app.post("/runs")
@app.post("/runs/batch")
async def runs(request: Request):
    start_time = time.perf_counter()
    
    # Simulate some minimal processing
    data = await request.json()
    
    # Create a simple response
    response = {
        "id": str(uuid.uuid4()),
        "status": "completed",
        "created_at": int(time.time()),
        "request": data
    }
    
    # Ensure the response takes at least 0.05 ms
    elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    if elapsed_time < 0.05:
        time.sleep((0.05 - elapsed_time) / 1000)  # Convert back to seconds for sleep
    
    return response
    

@app.post("/traces")
async def traces(request: Request):
    try:
        start_time = time.perf_counter()
        
        # Attempt to parse the request body
        try:
            data = await request.json()
        except json.JSONDecodeError:
            # If JSON parsing fails, try to read the raw body
            body = await request.body()
            return HTTPException(status_code=400, detail=f"Invalid JSON: {body.decode('utf-8', errors='ignore')}")
        except UnicodeDecodeError:
            # If decoding fails, return an error about invalid encoding
            return HTTPException(status_code=400, detail="Request body is not valid UTF-8 encoded")
        
        # Rest of the function remains the same
        response = {
            "id": str(uuid.uuid4()),
            "status": "completed",
            "created_at": int(time.time()),
            "trace_data": {
                "events": [
                    {
                        "timestamp": int(time.time()),
                        "type": "start",
                        "details": "Trace started"
                    },
                    {
                        "timestamp": int(time.time()) + 1,
                        "type": "end",
                        "details": "Trace completed"
                    }
                ],
            }
        }
        
        # Ensure the response takes at least 0.05 ms
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        if elapsed_time < 0.05:
            time.sleep((0.05 - elapsed_time) / 1000)  # Convert back to seconds for sleep
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTTPException(status_code=500, detail=str(e))

import gzip
import io

@app.post("/api/v2/logs")
async def logs(request: Request):
    await asyncio.sleep(60)  # Wait for 1 second
    return {"status": "done"}
    start_time = time.perf_counter()
    
    # Check if the content is gzipped
    content_encoding = request.headers.get("Content-Encoding", "").lower()
    
    # Read the raw body
    body = await request.body()
    
    # Decompress if gzipped
    if content_encoding == "gzip":
        try:
            body = gzip.decompress(body)
        except gzip.BadGzipFile:
            return HTTPException(status_code=400, detail="Invalid gzip data")
    
    # Attempt to parse the request body
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return HTTPException(status_code=400, detail=f"Invalid JSON: {body.decode('utf-8', errors='ignore')}")
    except UnicodeDecodeError:
        return HTTPException(status_code=400, detail="Request body is not valid UTF-8 encoded")
    
    # Create a log response
    response = {
        "id": str(uuid.uuid4()),
        "timestamp": int(time.time()),
        "level": "info",
        "message": "Log entry received",
        "data": data
    }
    
    # Ensure the response takes at least 0.05 ms
    elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    if elapsed_time < 0.05:
        time.sleep((0.05 - elapsed_time) / 1000)  # Convert back to seconds for sleep
    
    return Response(
        content=json.dumps(response),
        status_code=202,
    )
slack_requests = deque(maxlen=10)
slack_requests = deque(maxlen=10)

class SlackRequest(BaseModel):
    timestamp: datetime
    data: Dict[str, Any]

@app.post("/slack")
async def slack_endpoint(request: Request):
    current_time = datetime.now()
    request_data = await request.json()

    # Add the current request to the deque
    slack_requests.append(SlackRequest(timestamp=current_time, data=request_data))

    # Remove requests older than 10 minutes
    slack_requests_list = list(slack_requests)
    slack_requests_list = [req for req in slack_requests_list if current_time - req.timestamp <= timedelta(minutes=10)]
    slack_requests.clear()
    slack_requests.extend(slack_requests_list)

    return {"message": "Request received and stored"}

@app.get("/slack/history", response_model=List[SlackRequest])
async def get_slack_history():
    return list(slack_requests)





def data_generator_anthropic(model=None):
    response_id = uuid.uuid4().hex
    sentence = "Hello this is a test response from a fixed OpenAI endpoint."
    words = sentence.split(" ")
    _model = model if isinstance(model, str) else "claude-3-opus-20240229"
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
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"



# for completion
@app.post("/v1/messages")
async def completion_anthropic(request: Request):
    data = await request.json()
    _model = (
        data.get("litellm_model_id")  # Prefer client-facing model name
        or data.get("model") or data.get("model_name") or data.get("modelId") or data.get("model_id")
        or request.headers.get("X-LiteLLM-Model") or request.headers.get("x-litellm-model")
        or "claude-3-opus-20240229"
    )
    _model = _model if isinstance(_model, str) else "claude-3-opus-20240229"
    _model = _normalize_model_for_response(_model)

    if data.get("stream") == True:
        return StreamingResponse(
            content=data_generator_anthropic(model=_model),
            media_type="text/event-stream",
        )
    else:
        response = {
            "id": "msg_01G7MsdWPT2JZMUuc1UXRavn",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "I'm sorry, but the string of characters \"123450000s0 p kk\" doesn't appear to have any clear meaning or context. It seems to be a random combination of numbers and letters. If you could provide more information or clarify what you're trying to communicate, I'll do my best to assist you."
                }
            ],
            "model": _model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 17,
                "output_tokens": 71,
                "total_tokens": 88
            }
        }

        return response

@app.post("/load_test/api/public/ingestion")
async def mock_ingestion(request: Request):
    time.sleep(0.5)
    return {"status": "done"}


seen_langfuse_request_ids = set()

@app.post("/api/public/ingestion")
async def ingestion(request: Request):
    try:
        global seen_langfuse_request_ids
        data = await request.json()
        
        # Extract request IDs from the batch
        for item in data.get('batch', []):
            if item.get('type') == 'generation-create':
                full_request_id = item.get('body', {}).get('id')
                if full_request_id and '_' in full_request_id:
                    # Split on underscore and take the second part (the chatcmpl ID)
                    clean_request_id = full_request_id.split('_')[1]
                    seen_langfuse_request_ids.add(clean_request_id)
        
        print(f"Stored request IDs (total: {len(seen_langfuse_request_ids)}): {seen_langfuse_request_ids}")
        await asyncio.sleep(1)  # Original delay
        return {"status": "done", "stored_ids_count": len(seen_langfuse_request_ids)}
    
    except Exception as e:
        print(f"Error processing ingestion request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/langfuse/trace/{request_id}")
async def has_request_id(request_id: str):
    return {
        "exists": request_id in seen_langfuse_request_ids,
        "request_id": request_id
    }


# for responses
def responses_data_generator(input_text="", model=None):
    """Generator for streaming Responses API chunks"""
    # Ensure input_text is always a string
    if isinstance(input_text, list):
        input_text = " ".join(str(msg) for msg in input_text)
    elif isinstance(input_text, dict):
        input_text = str(input_text)
    else:
        input_text = str(input_text) if input_text else ""
    
    _model = _normalize_model_for_response(model) if isinstance(model, str) else "gpt-4.1"
    response_id = uuid.uuid4().hex
    item_id = f"msg_{uuid.uuid4().hex}"
    sentence = f"Hello! I received your input: '{input_text}'. This is a mock response from the Responses API."
    words = sentence.split(" ")
    current_time = int(time.time())
    
    # 1. Send response.created event
    response_created = {
        'type': 'response.created',
        'response': {
            'id': f'resp_{response_id}',
            'object': 'response',
            'created_at': current_time,
            'model': _model,
            'status': 'in_progress',
            'output': []
        }
    }
    yield f"data: {json.dumps(response_created)}\n\n"
    
    # 2. Send output_item.added event
    output_item_added = {
        'type': 'response.output_item.added',
        'output_index': 0,
        'item': {
            'type': 'message',
            'id': item_id,
            'role': 'assistant',
            'status': 'in_progress',
            'content': []
        }
    }
    yield f"data: {json.dumps(output_item_added)}\n\n"
    
    # 3. Send text deltas
    for word in words:
        text_delta = {
            'type': 'response.output_text.delta',
            'item_id': item_id,
            'output_index': 0,
            'content_index': 0,
            'delta': word + ' '
        }
        yield f"data: {json.dumps(text_delta)}\n\n"
    
    # 4. Send output_text.done event
    output_text_done = {
        'type': 'response.output_text.done',
        'item_id': item_id,
        'output_index': 0,
        'content_index': 0,
        'text': sentence
    }
    yield f"data: {json.dumps(output_text_done)}\n\n"
    
    # 5. Send response.completed event
    response_completed = {
        'type': 'response.completed',
        'response': {
            'id': f'resp_{response_id}',
            'object': 'response',
            'created_at': current_time,
            'model': _model,
            'status': 'completed',
            'output': [{
                'type': 'message',
                'id': item_id,
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': sentence}]
            }],
            'usage': {
                'input_tokens': len(input_text.split()) if input_text else 0,
                'output_tokens': len(sentence.split()),
                'total_tokens': (len(input_text.split()) if input_text else 0) + len(sentence.split())
            }
        }
    }
    yield f"data: {json.dumps(response_completed)}\n\n"
    
    # 6. Send [DONE]
    yield "data: [DONE]\n\n"


async def _send_realtime_text_response(
    websocket: WebSocket,
    text: str,
    model: str,
    next_event_id: Callable[[], str],
    conversation_id: str,
) -> None:
    """Emit realtime events that align with OpenAI's websocket schema."""

    current_time = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "created_at": current_time,
                "status": "in_progress",
                "model": model,
                "conversation_id": conversation_id,
                "output": [],
            },
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.content_part.added",
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "part": {
                "type": "text",
                "text": "",
            },
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.text.delta",
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "delta": text,
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.text.done",
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "text": text,
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.content_part.done",
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "part": {
                "type": "text",
                "text": text,
            },
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            },
        }
    )

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "response.done",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "model": model,
                "conversation_id": conversation_id,
                "output": [
                    {
                        "object": "realtime.item",
                        "type": "message",
                        "id": item_id,
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": len(text.split()),
                    "total_tokens": len(text.split()),
                },
            },
        }
    )


@app.post("/responses")
@app.post("/v1/responses")
async def create_response(request: Request):
    """OpenAI Responses API endpoint - Create a response"""
    _time_to_sleep = os.getenv("TIME_TO_SLEEP", None)
    if _time_to_sleep is not None:
        print("sleeping for " + _time_to_sleep)
        await asyncio.sleep(float(_time_to_sleep))

    data = await request.json()

    # Handle error cases
    if data.get("model") == "429":
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")

    if data.get("model") == "random_sleep":
        sleep_time = random.randint(1, 10)
        print("sleeping for " + str(sleep_time) + " seconds")
        await asyncio.sleep(sleep_time)
    
    # Degraded provider simulation for responses endpoint
    if data.get("model") in ["degraded", "slow_provider", "blocked"]:
        sleep_time = get_histogram_sleep_time()
        print(f"[DEGRADED MODE] Sleeping for {sleep_time:.1f} seconds ({sleep_time/60:.1f} minutes) - histogram distribution")
        await asyncio.sleep(sleep_time)

    # Non-streaming response setup
    response_id = uuid.uuid4().hex
    model = (
        data.get("litellm_model_id") or data.get("model") or data.get("model_name")
        or data.get("modelId") or data.get("model_id") or "gpt-4.1"
    )
    model = _normalize_model_for_response(model if isinstance(model, str) else "gpt-4.1")
    input_data = data.get("input", "")
    
    # Handle input: can be string, list, or dict (Azure format)
    if isinstance(input_data, list):
        # Convert list of messages to string representation for token counting
        input_text = " ".join(str(msg) for msg in input_data)
    elif isinstance(input_data, dict):
        # Handle dict format (e.g., {"messages": [...]})
        input_text = str(input_data)
    else:
        # Assume it's a string
        input_text = str(input_data) if input_data else ""
    
    if data.get("stream") == True:
        return StreamingResponse(
            content=responses_data_generator(input_text=input_text, model=model),
            media_type="text/event-stream",
        )
    tools = data.get("tools", [])
    reasoning = data.get("reasoning", {})
    background = data.get("background", False)
    
    # Generate output text based on input (use original input_data for display)
    input_display = input_data if isinstance(input_data, str) else str(input_data)
    output_text = f"Hello! I received your input: '{input_display}'. This is a mock response from the Responses API."
    
    # Handle background mode
    if background:
        # In background mode, return a job-like response
        return {
            "id": f"resp_{response_id}",
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "status": "processing",
            "background": True,
            "output": [],  # Empty output for processing state
            "tools": tools,
            "reasoning": reasoning
        }
    
    # Regular response
    response = {
        "id": f"resp_{response_id}",
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text
                    }
                ]
            }
        ],
        "tools": tools if tools else [],
        "reasoning": reasoning if reasoning else {},
        "usage": {
            "input_tokens": len(input_text.split()) if input_text else 0,  
            "output_tokens": len(output_text.split()),
            "total_tokens": len(input_text.split()) + len(output_text.split()) if input_text else len(output_text.split())
        }
    }
    
    return response


@app.post("/openai/responses")
@app.post("/openai/v1/responses")
async def azure_responses_api(request: Request):
    """Azure Responses API endpoint - delegates to the same handler as /responses"""
    try:
        # Reuse the exact same logic as your existing /responses endpoint
        return await create_response(request)
    except Exception as e:
        # Return detailed error response to client
        error_detail = await get_error_detail_with_body(e, request, "azure_responses_api")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    """OpenAI Responses API endpoint - Get a response by ID"""
    # Return a mock response with the given ID
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": "gpt-4.1",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "This is a mock response retrieved by ID."
                    }
                ]
            }
        ],
        "tools": [],
        "reasoning": {},
        "status": "completed",
        "usage": {
            "input_tokens": 5,       
            "output_tokens": 10,     
            "total_tokens": 15
        }
    }


@app.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket):
    """Simplified realtime endpoint compatible with LiteLLM proxy tests."""

    model = websocket.query_params.get("model", "gpt-4o-realtime-preview-2024-10-01")
    session_id = f"sess_{uuid.uuid4().hex}"
    conversation_id = f"conv_{uuid.uuid4().hex}"

    event_counter = 0

    def next_event_id() -> str:
        nonlocal event_counter
        event_counter += 1
        return f"evt_{session_id}_{event_counter}"

    await websocket.accept()

    await websocket.send_json(
        {
            "event_id": next_event_id(),
            "type": "session.created",
            "session": {
                "id": session_id,
                "model": model,
                "created_at": int(time.time()),
                "modalities": ["text"],
            },
        }
    )

    try:
        while True:
            try:
                incoming = await websocket.receive()
            except RuntimeError as e:
                # Connection was closed by client
                if "already completed" in str(e) or "websocket.close" in str(e):
                    break
                raise

            if incoming.get("type") == "websocket.close":
                break

            message_text = incoming.get("text")
            if message_text is None:
                # Skip non-text messages (binary, ping/pong, etc.)
                continue

            try:
                payload = json.loads(message_text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Payload must be valid JSON",
                        },
                    }
                )
                continue

            event_type = payload.get("type")

            if event_type == "session.update":
                await websocket.send_json(
                    {
                        "event_id": next_event_id(),
                        "type": "session.updated",
                        "session": {
                            "id": session_id,
                            "model": model,
                            "modalities": payload.get("session", {}).get("modalities", ["text"]),
                        },
                    }
                )
            elif event_type == "response.create":
                response_payload = payload.get("response", {})
                instructions = response_payload.get("instructions", "")
                prefix = "[fake-realtime] "
                output_text = (
                    f"{prefix}{instructions}"
                    if instructions
                    else f"{prefix}Hello! This is a realtime response from the fake endpoint."
                )
                await _send_realtime_text_response(
                    websocket,
                    output_text,
                    model,
                    next_event_id,
                    conversation_id,
                )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "type": "unsupported_event",
                            "message": f"Unsupported realtime event: {event_type}",
                        },
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.close(code=1011, reason=str(exc))
        except RuntimeError:
            pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


# Catch-all route for Vertex AI endpoints with colons (e.g., :generateContent, :predict)
# This must come at the END after all other routes
# Only matches paths that look like Vertex AI endpoints (contains /v1/projects/ or ends with colon methods)
@app.post("/{path:path}")
async def catch_all_vertex_with_colons(request: Request, path: str):
    """Catch-all for Vertex AI endpoints with colons (e.g., :generateContent, :predict)"""
    # Extract the actual path from the request URL
    request_path = request.url.path
    
    # Only handle paths that look like Vertex AI endpoints
    # Check for Vertex AI patterns: /v1/projects/, paths with colons, or common Vertex methods
    is_vertex_path = (
        "/v1/projects/" in request_path or 
        ":generateContent" in request_path or 
        ":predict" in request_path or
        ":rawPredict" in request_path or
        request_path.endswith("generateContent") or
        request_path.endswith("predict") or
        request_path.endswith("rawPredict") or
        request_path.endswith(":generateContent") or
        request_path.endswith(":predict") or
        request_path.endswith(":rawPredict") or
        path.endswith("generateContent") or
        path.endswith("predict") or
        path.endswith("rawPredict") or
        path.endswith(":generateContent") or
        path.endswith(":predict") or
        path.endswith(":rawPredict")
    )
    
    if not is_vertex_path:
        # Not a Vertex AI path, return 404 with details
        error_detail = format_detailed_error(
            Exception("Path does not match Vertex AI endpoint patterns"),
            request,
            "catch_all_vertex_path_check"
        )
        error_detail["error"]["message"] = f"No handler found for path: {request_path}. Supported Vertex AI patterns: /v1/projects/.../:generateContent, /v1/projects/.../:predict, /:generateContent, /:predict, /:rawPredict"
        raise HTTPException(status_code=404, detail=error_detail)
    
    # Check if this is a generateContent endpoint (multiple patterns)
    is_generate_content = (
        ":generateContent" in request_path or 
        request_path.endswith("generateContent") or
        request_path.endswith(":generateContent") or
        ":generateContent" in path or
        path.endswith("generateContent") or
        path.endswith(":generateContent")
    )
    
    if is_generate_content:
        authorization = request.headers.get("authorization") or request.headers.get("Authorization")
        try:
            return await generate_content(request, authorization)
        except Exception as e:
            # Return detailed error response to client
            error_detail = await get_error_detail_with_body(e, request, "vertex_generate_content")
            raise HTTPException(status_code=500, detail=error_detail)
    
    # Check if this is a predict endpoint (multiple patterns)
    is_predict = (
        ":predict" in request_path or 
        request_path.endswith("predict") or
        request_path.endswith(":predict") or
        ":predict" in path or
        path.endswith("predict") or
        path.endswith(":predict")
    )
    
    if is_predict:
        authorization = request.headers.get("authorization") or request.headers.get("Authorization")
        try:
            return await predict(request, authorization)
        except Exception as e:
            # Return detailed error response to client
            error_detail = await get_error_detail_with_body(e, request, "vertex_predict")
            raise HTTPException(status_code=500, detail=error_detail)
    
    # Check if this is a rawPredict endpoint (multiple patterns)
    # rawPredict is similar to predict, so we'll use the same handler
    is_raw_predict = (
        ":rawPredict" in request_path or 
        request_path.endswith("rawPredict") or
        request_path.endswith(":rawPredict") or
        ":rawPredict" in path or
        path.endswith("rawPredict") or
        path.endswith(":rawPredict")
    )
    
    if is_raw_predict:
        authorization = request.headers.get("authorization") or request.headers.get("Authorization")
        try:
            return await predict(request, authorization)
        except Exception as e:
            # Return detailed error response to client
            error_detail = await get_error_detail_with_body(e, request, "vertex_raw_predict")
            raise HTTPException(status_code=500, detail=error_detail)
    
    # If it's a Vertex path but doesn't match our handlers, return 404 with details
    error_detail = format_detailed_error(
        Exception(f"Vertex AI path recognized but no handler available"),
        request,
        "catch_all_vertex_no_handler"
    )
    error_detail["error"]["message"] = f"Vertex AI path '{request_path}' recognized but no handler matched. Supported handlers: generateContent, predict, rawPredict"
    raise HTTPException(status_code=404, detail=error_detail)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")