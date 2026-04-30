from __future__ import annotations

import os
from typing import Any, Optional

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

CASSETTE_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"


def redis_key_for(cassette_path: str) -> str:
    return f"{REDIS_KEY_PREFIX}{os.path.relpath(str(cassette_path))}"


def _build_default_client():
    import redis

    host = os.environ.get("REDIS_HOST")
    if not host:
        raise RuntimeError("REDIS_HOST is not set")
    return redis.Redis(
        host=host,
        port=int(os.environ.get("REDIS_PORT", 6379)),
        password=os.environ.get("REDIS_PASSWORD") or None,
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
