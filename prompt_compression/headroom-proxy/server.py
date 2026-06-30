"""
Minimal headroom-compatible proxy: /v1/compress and /v1/retrieve/{hash}.

Implements CCR (Compress-Cache-Retrieve) without any ML models.
Compresses by truncating long messages and embedding a retrieval hash marker.
The full original content is kept in an in-process LRU cache (TTL 30 min).
"""

import hashlib
import re
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="headroom-proxy-mock")

_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 30 * 60
_MIN_TOKENS_TO_COMPRESS = 500
_CHARS_PER_TOKEN = 4
_TARGET_CHARS = _MIN_TOKENS_TO_COMPRESS * _CHARS_PER_TOKEN


def _evict_expired() -> None:
    now = time.time()
    expired = [k for k, (_, ts) in _cache.items() if now - ts > _CACHE_TTL]
    for k in expired:
        del _cache[k]


def _store(content: str) -> str:
    h = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:24]
    _cache[h] = (content, time.time())
    return h


def _compress_content(text: str) -> tuple[str, str | None]:
    char_count = len(text)
    if char_count <= _TARGET_CHARS:
        return text, None
    kept = text[:_TARGET_CHARS]
    overflow = text[_TARGET_CHARS:]
    h = _store(overflow)
    token_estimate = char_count // _CHARS_PER_TOKEN
    kept_estimate = _TARGET_CHARS // _CHARS_PER_TOKEN
    marker = (
        f"[{token_estimate - kept_estimate} tokens compressed to 0. "
        f"Retrieve more: hash={h}]"
    )
    return kept + "\n" + marker, h


class CompressRequest(BaseModel):
    messages: list[dict[str, Any]]
    model: str | None = None


@app.post("/v1/compress")
async def compress(req: CompressRequest) -> JSONResponse:
    _evict_expired()
    original_chars = sum(len(str(m.get("content", ""))) for m in req.messages)
    compressed_messages = []
    for msg in req.messages:
        content = msg.get("content")
        if isinstance(content, str):
            new_content, _ = _compress_content(content)
            compressed_messages.append({**msg, "content": new_content})
        else:
            compressed_messages.append(msg)

    compressed_chars = sum(len(str(m.get("content", ""))) for m in compressed_messages)
    tokens_before = max(1, original_chars // _CHARS_PER_TOKEN)
    tokens_after = max(1, compressed_chars // _CHARS_PER_TOKEN)
    return JSONResponse({
        "messages": compressed_messages,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "compression_ratio": tokens_after / tokens_before,
        "transforms_applied": ["mock:ccr_truncate"],
    })


@app.get("/v1/retrieve/{hash_value}")
async def retrieve(hash_value: str, query: str | None = None) -> JSONResponse:
    _evict_expired()
    if not re.match(r"^[a-f0-9]{24}$", hash_value):
        raise HTTPException(status_code=400, detail="invalid hash format")
    entry = _cache.get(hash_value)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"hash {hash_value} not found or expired")
    original_content, _ = entry
    return JSONResponse({"original_content": original_content, "hash": hash_value})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "healthy", "ready": True, "version": "mock-0.1"})


@app.get("/v1/compress")
async def compress_get() -> JSONResponse:
    return JSONResponse({"error": "use POST"}, status_code=405)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8787"))
    uvicorn.run(app, host="0.0.0.0", port=port)
