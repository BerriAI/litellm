from __future__ import annotations

import logging
import os
from typing import Any, Optional

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

CASSETTE_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"

# Healthy cassettes hold 1–5 episodes (a single test rarely makes more than a
# handful of distinct HTTP calls). When a cassette balloons past this, it
# usually means a test produces non-deterministic request bodies (e.g. uuid)
# under record_mode=new_episodes, and every CI run is appending fresh
# unmatched episodes instead of replaying. That growth is unbounded over
# time and silently inflates Redis. Refuse to persist past this threshold so
# the pathology surfaces loudly instead.
MAX_EPISODES_PER_CASSETTE = 50

_log = logging.getLogger(__name__)


# Per-process map: cassette key -> "did the test that produced this cassette
# pass?". The conftest's pytest_runtest_makereport hook sets True when the test
# body succeeds; save_cassette consults it to avoid persisting recordings from
# failed runs. We key by the redis cache key so retries (which may produce a
# fresh cassette object each time but write to the same key) interleave
# correctly.
_passed_by_cassette_key: dict[str, bool] = {}


def mark_test_outcome_for_cassette(cassette_path: str, passed: bool) -> None:
    """Record whether the test that owns ``cassette_path`` passed.

    Called from a pytest hook in conftest. The recorded value is consulted by
    ``save_cassette`` so failed-attempt recordings (e.g. a flaky test that
    asserts on provider state) don't poison the cache for future runs.
    """
    _passed_by_cassette_key[redis_key_for(cassette_path)] = passed


def redis_key_for(cassette_path: str) -> str:
    rel = os.path.relpath(str(cassette_path))
    if rel.endswith(".yaml"):
        rel = rel[: -len(".yaml")]
    rel = rel.replace("/cassettes/", "/").lstrip("./")
    return f"{REDIS_KEY_PREFIX}{rel}"


CASSETTE_REDIS_URL_ENV = "CASSETTE_REDIS_URL"


def _redis_url_from_env() -> Optional[str]:
    # Use a dedicated cassette Redis URL so the VCR cache is isolated from any
    # application Redis used by tests (which may be flushed by other suites).
    # Intentionally do NOT fall back to REDIS_URL/REDIS_HOST — sharing a Redis
    # with the app cache risks cassettes being wiped by flushdb/flushall.
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
    # Managed Redis providers (e.g. Upstash) drop idle TLS connections; retry on
    # connection/timeout errors so a single dropped socket doesn't fail teardown.
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

    # Lazily resolve the redis exception classes so callers can pass any
    # client (incl. fakeredis) without importing the real `redis` package.
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        _transient_errors: tuple = (RedisConnectionError, RedisTimeoutError)
    except ImportError:  # pragma: no cover - redis is a hard test dep
        _transient_errors = ()

    class _RedisPersister:
        @staticmethod
        def load_cassette(cassette_path, serializer):
            try:
                data = redis_client.get(redis_key_for(cassette_path))
            except _transient_errors as exc:
                # Treat a Redis outage on read as a cassette miss so tests fall
                # through to a live call instead of erroring in setup.
                _log.warning(
                    "VCR redis load failed for %s; treating as cache miss: %s",
                    cassette_path,
                    exc,
                )
                raise CassetteNotFoundError() from exc
            if data is None:
                raise CassetteNotFoundError()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return deserialize(data, serializer)

        @staticmethod
        def save_cassette(cassette_path, cassette_dict, serializer):
            key = redis_key_for(cassette_path)
            # Only persist successful runs. A failed test (incl. all the failed
            # retries before a passing one) would otherwise poison the cache —
            # e.g. a flaky test that observes provider state across two calls
            # could capture a "bad luck" response that deterministically fails
            # every future replay. We default to True if the hook didn't run
            # (e.g. cassette saved outside a test context) so non-test usage
            # still works.
            passed = _passed_by_cassette_key.pop(key, True)
            episode_count = len(cassette_dict.get("requests", []) or [])
            if episode_count > MAX_EPISODES_PER_CASSETTE:
                # Pathology: the test is producing non-deterministic request
                # bodies and accumulating unbounded episodes. Refuse the save
                # so the cassette can't keep ballooning, and surface a loud
                # warning so someone investigates / opts the test out.
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
            except _transient_errors as exc:
                # Cassette persistence is a cache, not test correctness. A Redis
                # outage on save should not fail an otherwise-passing test —
                # the next run will simply re-record.
                _log.warning(
                    "VCR redis save failed for %s; cassette not persisted: %s",
                    cassette_path,
                    exc,
                )

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


VCR_VERBOSE_ENV = "LITELLM_VCR_VERBOSE"


def vcr_verbose_enabled() -> bool:
    return os.environ.get(VCR_VERBOSE_ENV) == "1"


def emit_vcr_verbose_line(line: str) -> None:
    """Write a one-line VCR verdict to the un-redirected stderr.

    Pytest's stdout/stderr capture would otherwise hide this output (and
    only surface it as 'Captured stdout teardown' on failing tests). Under
    xdist each worker has its own stderr; CircleCI aggregates them. Writing
    to ``sys.__stderr__`` bypasses pytest's capture entirely so the line
    reaches the live CI log alongside the per-test PASSED/FAILED markers.
    """
    import sys

    try:
        stream = sys.__stderr__
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()
            return
    except Exception:  # pragma: no cover - last-ditch fallback
        pass
    # If __stderr__ isn't writable for some reason, fall back to the
    # captured stderr — better than swallowing.
    print(line, file=sys.stderr)


def format_vcr_verdict(cassette: Any) -> str:
    """Build a one-line hit/miss verdict for a vcrpy Cassette.

    HIT  — at least one request was served from cache and nothing new was
           recorded. (Pure replay.)
    MISS — nothing from cache; one or more requests went live and were
           recorded. (Cold cache.)
    PARTIAL — mix of replay and new recordings. Usually means the cassette
              matches some but not all requests for this test (e.g. retries,
              new branches, or vcrpy match_on too strict).
    NOOP — test made no HTTP calls (or VCR not engaged for it).
    """
    if cassette is None:
        return "[VCR NOOP]"
    played = getattr(cassette, "play_count", 0) or 0
    # cassette.data is the recorded request/response list; len(cassette) counts
    # recorded episodes. New recordings during this test = len - prior_len, but
    # we don't have prior_len here, so we use cassette.dirty (set when an append
    # happened during this run) as the "new recording" signal.
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
