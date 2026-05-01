from __future__ import annotations

import os
from typing import Any, Optional

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

CASSETTE_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"


def redis_key_for(cassette_path: str) -> str:
    rel = os.path.relpath(str(cassette_path))
    if rel.endswith(".yaml"):
        rel = rel[: -len(".yaml")]
    rel = rel.replace("/cassettes/", "/").lstrip("./")
    return f"{REDIS_KEY_PREFIX}{rel}"


def _redis_url_from_env() -> Optional[str]:
    for var in ("REDIS_URL", "REDIS_SSL_URL"):
        url = os.environ.get(var)
        if url:
            return url
    host = os.environ.get("REDIS_HOST")
    if not host:
        return None
    scheme = "rediss" if os.environ.get("REDIS_SSL", "").lower() == "true" else "redis"
    auth = ""
    if os.environ.get("REDIS_PASSWORD"):
        user = os.environ.get("REDIS_USERNAME", "")
        auth = f"{user}:{os.environ['REDIS_PASSWORD']}@"
    port = os.environ.get("REDIS_PORT", "6379")
    return f"{scheme}://{auth}{host}:{port}"


def _build_default_client():
    import redis

    url = _redis_url_from_env()
    if not url:
        raise RuntimeError(
            "Set REDIS_URL, REDIS_SSL_URL, or REDIS_HOST to enable the VCR persister"
        )
    return redis.Redis.from_url(
        url,
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,
    )


def make_redis_persister(
    client: Optional[Any] = None,
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
):
    redis_client = client if client is not None else _build_default_client()

    class _RedisPersister:
        @staticmethod
        def load_cassette(cassette_path, serializer):
            data = redis_client.get(redis_key_for(cassette_path))
            if data is None:
                raise CassetteNotFoundError()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return deserialize(data, serializer)

        @staticmethod
        def save_cassette(cassette_path, cassette_dict, serializer):
            data = serialize(cassette_dict, serializer)
            payload = data.encode("utf-8") if isinstance(data, str) else data
            redis_client.set(redis_key_for(cassette_path), payload, ex=ttl_seconds)

    return _RedisPersister


def filter_non_2xx_response(response):
    # Returning None tells vcrpy to skip persisting; see Cassette.append.
    if not isinstance(response, dict):
        return response
    status = response.get("status")
    code = status.get("code") if isinstance(status, dict) else status
    if not isinstance(code, int):
        return response
    return response if 200 <= code < 300 else None


_PATCHED_AIOHTTP_RECORD = False


def patch_vcrpy_aiohttp_record_path() -> None:
    """Re-feed the response body into aiohttp's StreamReader after vcrpy's
    record_response drains it, so downstream consumers (e.g.
    LiteLLMAiohttpTransport.AiohttpResponseStream) can still read it."""
    global _PATCHED_AIOHTTP_RECORD
    if _PATCHED_AIOHTTP_RECORD:
        return
    import vcr.stubs.aiohttp_stubs as _aiohttp_stubs

    _orig_record_response = _aiohttp_stubs.record_response

    async def _record_response_preserving_body(cassette, vcr_request, response):
        await _orig_record_response(cassette, vcr_request, response)
        body = getattr(response, "_body", None) or b""
        if body:
            response.content.unread_data(body)

    _aiohttp_stubs.record_response = _record_response_preserving_body
    _PATCHED_AIOHTTP_RECORD = True
