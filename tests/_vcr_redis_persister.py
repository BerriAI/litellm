"""Redis-backed cassette persister for vcrpy.

Stores the same serialized cassette payload that ``FilesystemPersister``
would write to disk, but under a Redis key with a 24h TTL. Cassettes
auto-expire so the next CI run after the rollover re-records against the
live provider, surfacing API drift within a day instead of waiting for a
human to refresh ``cassettes/*.yaml`` by hand.

On a cache miss we raise ``CassetteNotFoundError``; vcrpy's record-mode
machinery catches that and falls through to a live HTTP call, which then
gets persisted via ``save_cassette``. Non-2xx responses are filtered out
upstream by ``conftest.before_record_response`` so a transient provider
failure can't poison the cache for 24h.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

CASSETTE_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"


def redis_key_for(cassette_path: str) -> str:
    """Map a cassette file path to a stable Redis key.

    Uses the path relative to CWD so keys are stable across machines.
    """
    rel = os.path.relpath(str(cassette_path))
    return f"{REDIS_KEY_PREFIX}{rel}"


def _build_default_client():
    import redis

    host = os.environ.get("REDIS_HOST")
    if not host:
        raise RuntimeError(
            "REDIS_HOST is not set; cannot build Redis cassette persister"
        )
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
    """Build a vcrpy-compatible persister bound to a Redis client.

    The returned object exposes ``load_cassette`` / ``save_cassette`` and is
    a drop-in replacement for ``vcr.persisters.filesystem.FilesystemPersister``.
    Pass an explicit ``client`` in tests; production callers omit it and let
    the persister build a client from ``REDIS_HOST`` / ``REDIS_PORT`` /
    ``REDIS_PASSWORD``.
    """
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
            redis_client.set(
                redis_key_for(cassette_path),
                payload,
                ex=ttl_seconds,
            )

    return _RedisPersister


def filter_non_2xx_response(response):
    """vcrpy ``before_record_response`` hook that drops non-2xx responses.

    Returning ``None`` tells vcrpy to skip persisting the response (see
    ``vcr.cassette.Cassette.append``). This prevents transient 5xx/429
    failures from being baked into the cache for the rest of the TTL window.
    """
    if not isinstance(response, dict):
        return response
    status = response.get("status")
    code = None
    if isinstance(status, dict):
        code = status.get("code")
    elif isinstance(status, int):
        code = status
    if code is None:
        return response
    if 200 <= int(code) < 300:
        return response
    return None
