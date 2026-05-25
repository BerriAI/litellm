from __future__ import annotations

import hashlib
import logging
import os
import sys
import warnings
from typing import Any, Optional

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

CASSETTE_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"
CASSETTE_REDIS_URL_ENV = "CASSETTE_REDIS_URL"
VCR_VERBOSE_ENV = "LITELLM_VCR_VERBOSE"
MAX_EPISODES_PER_CASSETTE = 50
# GPT-image-1 cost diagnostic: when set, save_cassette dumps per-episode
# (method, host, path, body-len, body-sha12) tuples to stderr so we can
# observe whether multipart/image-edit requests are stable or every CI
# run is generating fresh request bytes (which would force live calls
# under record_mode="new_episodes"). Mirrors the temporary diagnostic in
# PR #28110 that was reverted once the boundary fix landed.
EPISODE_BODY_AUDIT_ENV = "LITELLM_VCR_EPISODE_BODY_AUDIT"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_log = logging.getLogger(__name__)
_passed_by_cassette_key: dict[str, bool] = {}


class VCRCassetteCacheWarning(UserWarning):
    """Emitted when the cassette Redis cache fails to load or save.

    Surfaced in pytest's session-end warnings summary so failures are
    visible in CI logs even when the underlying tests pass.
    """


# Per-process counters; surfaced via :func:`cassette_cache_health` so
# conftests can emit a session-end banner when failures occurred.
_cache_health = {
    "save_failures": 0,
    "save_failure_last_error": "",
    "load_failures": 0,
    "load_failure_last_error": "",
}


def _record_cache_failure(kind: str, exc: BaseException) -> None:
    err = f"{type(exc).__name__}: {exc}"
    if kind == "save":
        _cache_health["save_failures"] = int(_cache_health["save_failures"]) + 1
        _cache_health["save_failure_last_error"] = err
    elif kind == "load":
        _cache_health["load_failures"] = int(_cache_health["load_failures"]) + 1
        _cache_health["load_failure_last_error"] = err


def cassette_cache_health() -> dict:
    return dict(_cache_health)


def reset_cassette_cache_health() -> None:
    _cache_health["save_failures"] = 0
    _cache_health["save_failure_last_error"] = ""
    _cache_health["load_failures"] = 0
    _cache_health["load_failure_last_error"] = ""


def cassette_cache_capacity_snapshot(client: Optional[Any] = None) -> Optional[dict]:
    """Probe Redis ``INFO memory`` and return used/max bytes and percent.

    Returns ``None`` if Redis is unreachable, the server didn't report
    ``maxmemory``, or ``maxmemory`` is 0 (uncapped). Best-effort: any
    exception turns into ``None`` so this never breaks a test session.
    """
    try:
        if client is None:
            client = _build_default_client()
        info = client.info(section="memory")
    except Exception:  # pragma: no cover - best-effort probe
        return None
    used = info.get("used_memory")
    maxmem = info.get("maxmemory")
    try:
        used = int(used) if used is not None else None
        maxmem = int(maxmem) if maxmem is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not used or not maxmem or maxmem <= 0:
        return None
    return {
        "used_memory_bytes": used,
        "maxmemory_bytes": maxmem,
        "used_pct": (used / maxmem) * 100.0,
    }


def mark_test_outcome_for_cassette(cassette_path: str, passed: bool) -> None:
    _passed_by_cassette_key[redis_key_for(cassette_path)] = passed


def redis_key_for(cassette_path: str) -> str:
    abs_path = os.path.abspath(str(cassette_path))
    try:
        rel = os.path.relpath(abs_path, start=_REPO_ROOT)
    except ValueError:
        rel = os.path.basename(abs_path)
    if rel.endswith(".yaml"):
        rel = rel[: -len(".yaml")]
    rel = rel.replace("/cassettes/", "/").lstrip("./")
    return f"{REDIS_KEY_PREFIX}{rel}"


def _redis_url_from_env() -> Optional[str]:
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


def make_redis_persister(
    client: Optional[Any] = None,
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
):
    redis_client = client if client is not None else _build_default_client()

    try:
        from redis.exceptions import RedisError
    except ImportError:  # pragma: no cover - redis is a hard test dep
        RedisError = Exception  # type: ignore[assignment,misc]

    class _RedisPersister:
        @staticmethod
        def load_cassette(cassette_path, serializer):
            try:
                data = redis_client.get(redis_key_for(cassette_path))
            except RedisError as exc:
                _record_cache_failure("load", exc)
                msg = (
                    f"VCR redis load failed for {cassette_path}; treating "
                    f"as cache miss: {type(exc).__name__}: {exc}"
                )
                _log.warning(msg)
                warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)
                raise CassetteNotFoundError() from exc
            if data is None:
                raise CassetteNotFoundError()
            try:
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                return deserialize(data, serializer)
            except Exception as exc:
                _record_cache_failure("load", exc)
                msg = (
                    f"VCR redis load failed for {cassette_path}; cached "
                    f"payload is corrupt, treating as cache miss: "
                    f"{type(exc).__name__}: {exc}"
                )
                _log.warning(msg)
                warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)
                raise CassetteNotFoundError() from exc

        @staticmethod
        def save_cassette(cassette_path, cassette_dict, serializer):
            key = redis_key_for(cassette_path)
            passed = _passed_by_cassette_key.pop(key, True)
            episode_count = len(cassette_dict.get("requests", []) or [])
            if os.environ.get(EPISODE_BODY_AUDIT_ENV) == "1" and passed:
                _dump_episode_body_audit(key, cassette_dict)
            if episode_count > MAX_EPISODES_PER_CASSETTE:
                _log.warning(
                    "VCR redis save refused for %s; cassette has %d episodes "
                    "(> MAX_EPISODES_PER_CASSETTE=%d). The test likely produces "
                    "non-deterministic request bodies (e.g. uuid) and is "
                    "appending instead of replaying. Opt it out with the "
                    "no-vcr list in conftest, or stabilize its request body.",
                    cassette_path,
                    episode_count,
                    MAX_EPISODES_PER_CASSETTE,
                )
                return
            if not passed:
                _log.info(
                    "VCR redis save skipped for %s; test did not pass — "
                    "leaving any prior cassette intact",
                    cassette_path,
                )
                return
            data = serialize(cassette_dict, serializer)
            payload = data.encode("utf-8") if isinstance(data, str) else data
            try:
                redis_client.set(key, payload, ex=ttl_seconds)
            except RedisError as exc:
                # Cassette persistence is strictly best-effort: connection
                # blips, timeouts, OOM at the maxmemory cap, READONLY
                # replicas, etc. should all degrade gracefully to "test
                # passed but cassette not cached" rather than failing the
                # test on teardown. We still want a loud signal so the
                # failure shows up in pytest's warnings summary at the
                # end of the session and feeds the session-end banner.
                _record_cache_failure("save", exc)
                msg = (
                    f"VCR redis save failed for {cassette_path}; cassette "
                    f"not persisted: {type(exc).__name__}: {exc}"
                )
                _log.warning(msg)
                warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)

    return _RedisPersister


def _coerce_body_to_bytes(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8", errors="replace")
    return repr(body).encode("utf-8", errors="replace")


def _dump_episode_body_audit(redis_key: str, cassette_dict: dict) -> None:
    """Emit one stderr line per episode about to be saved.

    Triggered by ``LITELLM_VCR_EPISODE_BODY_AUDIT=1``. Each line includes
    method, host, path (no query), body length and a 12-char SHA-256
    prefix so a reviewer can tell at a glance whether the bodies of two
    consecutive runs are identical (cassette-stable, will replay) or
    diverging (will record new episodes and burn live API spend on every
    run). Intentionally read-only and best-effort — never raises.
    """
    try:
        reqs = cassette_dict.get("requests") or []
        resps = cassette_dict.get("responses") or []
        sys.stderr.write(
            f"[VCR_EPISODE_AUDIT] cassette={redis_key} episodes={len(reqs)}\n"
        )
        for idx, req in enumerate(reqs):
            method = getattr(req, "method", None) or (
                req.get("method") if isinstance(req, dict) else "?"
            )
            uri = (
                getattr(req, "uri", None)
                or getattr(req, "url", None)
                or (req.get("uri") or req.get("url") if isinstance(req, dict) else "?")
            )
            host_path = (uri or "?").split("?", 1)[0]
            body = getattr(req, "body", None)
            if body is None and isinstance(req, dict):
                body = req.get("body")
            body_bytes = _coerce_body_to_bytes(body)
            body_sha = hashlib.sha256(body_bytes).hexdigest()[:12]
            # Response status (helpful: a 4xx body sometimes still bills).
            resp = resps[idx] if idx < len(resps) else {}
            status_code = "?"
            if isinstance(resp, dict):
                status = resp.get("status") or {}
                if isinstance(status, dict):
                    status_code = status.get("code", "?")
                elif isinstance(status, int):
                    status_code = status
            sys.stderr.write(
                f"[VCR_EPISODE_AUDIT]   ep{idx:02d} {method} {host_path} "
                f"body_len={len(body_bytes)} body_sha={body_sha} status={status_code}\n"
            )
        sys.stderr.flush()
    except Exception as exc:  # pragma: no cover - diagnostic only
        sys.stderr.write(f"[VCR_EPISODE_AUDIT] audit failed: {exc}\n")
        sys.stderr.flush()


def filter_non_2xx_response(response):
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


def vcr_verbose_enabled() -> bool:
    return os.environ.get(VCR_VERBOSE_ENV) == "1"


def format_vcr_verdict(cassette: Any) -> str:
    if cassette is None:
        return "[VCR NOOP]"
    played = getattr(cassette, "play_count", 0) or 0
    dirty = getattr(cassette, "dirty", False)
    total = len(cassette) if hasattr(cassette, "__len__") else 0
    if played == 0 and not dirty:
        return "[VCR NOOP] (no http traffic)"
    if played > 0 and not dirty:
        return f"[VCR HIT] {played} replayed, 0 new ({total} cassette entries)"
    if played == 0 and dirty:
        return f"[VCR MISS] 0 replayed, recorded new ({total} cassette entries)"
    return (
        f"[VCR PARTIAL] {played} replayed + new recordings ({total} cassette entries)"
    )
