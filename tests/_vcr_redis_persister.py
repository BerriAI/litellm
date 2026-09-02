from __future__ import annotations

import os
from typing import Any

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

from tests._vcr_persister import (
    CASSETTE_BACKEND_ENV,
    CASSETTE_REDIS_URL_ENV,
    CASSETTE_TTL_SECONDS,
    CASSETTE_TTL_SECONDS_ENV,
    MAX_EPISODES_PER_CASSETTE,
    VCR_VERBOSE_ENV,
    BaseCassettePersister,
    CorruptCassetteError,
    FilesystemBackend,
    VCRCassetteCacheWarning,
    _cache_health,
    _record_cache_failure,
    cassette_cache_capacity_snapshot,
    cassette_cache_health,
    filter_non_2xx_response,
    format_vcr_verdict,
    make_persister,
    mark_test_outcome_for_cassette,
    patch_vcrpy_aiohttp_record_path,
    reset_cassette_cache_health,
    resolve_cassette_ttl,
    set_cassette_ttl_override,
    vcr_verbose_enabled,
)

REDIS_KEY_PREFIX = "litellm:vcr:cassette:"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def redis_key_for(cassette_path: str) -> str:
    abs_path = os.path.abspath(str(cassette_path))
    try:
        rel = os.path.relpath(abs_path, start=_REPO_ROOT)
    except ValueError:
        rel = os.path.basename(abs_path)
    rel = rel.removesuffix(".yaml")
    rel = rel.replace("/cassettes/", "/").lstrip("./")
    return f"{REDIS_KEY_PREFIX}{rel}"


def _redis_url_from_env() -> str | None:
    return os.environ.get(CASSETTE_REDIS_URL_ENV) or None


def _build_default_client():
    import redis
    from redis.backoff import ExponentialBackoff
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError
    from redis.retry import Retry

    url = _redis_url_from_env()
    if not url:
        raise RuntimeError(
            f"Set {CASSETTE_REDIS_URL_ENV} to enable the VCR persister. "
            "Cassette Redis is intentionally separate from the application "
            "Redis (REDIS_URL/REDIS_HOST) to avoid being flushed by tests."
        )
    return redis.Redis.from_url(
        url,
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,
        retry=Retry(ExponentialBackoff(cap=2, base=0.1), retries=2),
        retry_on_error=[RedisConnectionError, RedisTimeoutError],
    )


class RedisBackend(BaseCassettePersister):
    backend_name = "redis"

    def __init__(self, client: Any, ttl_seconds: int | None = None) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._client = client

    def _load(self, cassette_path, serializer):
        data = self._client.get(redis_key_for(cassette_path))
        if data is None:
            raise CassetteNotFoundError()
        try:
            text = data.decode("utf-8") if isinstance(data, bytes) else data
            return deserialize(text, serializer)
        except Exception as exc:
            raise CorruptCassetteError(str(exc)) from exc

    def _save(
        self, cassette_path, cassette_dict, serializer, *, ttl_seconds: int
    ) -> None:
        data = serialize(cassette_dict, serializer)
        payload = data.encode("utf-8") if isinstance(data, str) else data
        key = redis_key_for(cassette_path)
        if ttl_seconds <= 0:
            self._client.set(key, payload)
            return
        self._client.set(key, payload, ex=ttl_seconds)

    def capacity_snapshot(self) -> dict | None:
        try:
            info = self._client.info(section="memory")
            used = int(info["used_memory"])
            maxmem = int(info["maxmemory"])
        except Exception:
            return None
        if not used or not maxmem or maxmem <= 0:
            return None
        return {
            "used_memory_bytes": used,
            "maxmemory_bytes": maxmem,
            "used_pct": (used / maxmem) * 100.0,
        }


def make_redis_persister(client: Any | None = None, ttl_seconds: int | None = None):
    redis_client = client if client is not None else _build_default_client()
    return RedisBackend(redis_client, ttl_seconds=ttl_seconds)


__all__ = [
    "CASSETTE_BACKEND_ENV",
    "CASSETTE_REDIS_URL_ENV",
    "CASSETTE_TTL_SECONDS",
    "CASSETTE_TTL_SECONDS_ENV",
    "MAX_EPISODES_PER_CASSETTE",
    "VCR_VERBOSE_ENV",
    "FilesystemBackend",
    "RedisBackend",
    "VCRCassetteCacheWarning",
    "_build_default_client",
    "_cache_health",
    "_record_cache_failure",
    "_redis_url_from_env",
    "cassette_cache_capacity_snapshot",
    "cassette_cache_health",
    "filter_non_2xx_response",
    "format_vcr_verdict",
    "make_persister",
    "make_redis_persister",
    "mark_test_outcome_for_cassette",
    "patch_vcrpy_aiohttp_record_path",
    "redis_key_for",
    "reset_cassette_cache_health",
    "resolve_cassette_ttl",
    "set_cassette_ttl_override",
    "vcr_verbose_enabled",
]
