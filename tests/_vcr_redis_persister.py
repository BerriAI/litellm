from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import warnings
from typing import Any, Optional, Protocol

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize

CASSETTE_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"
CASSETTE_REDIS_URL_ENV = "CASSETTE_REDIS_URL"
CASSETTE_S3_BUCKET_ENV = "CASSETTE_S3_BUCKET"
CASSETTE_S3_ENDPOINT_ENV = "CASSETTE_S3_ENDPOINT"
CASSETTE_S3_REGION_ENV = "CASSETTE_S3_REGION"
CASSETTE_LOCAL_CACHE_DIR_ENV = "CASSETTE_LOCAL_CACHE_DIR"
CASSETTE_DISABLE_COMPRESSION_ENV = "CASSETTE_DISABLE_COMPRESSION"
VCR_VERBOSE_ENV = "LITELLM_VCR_VERBOSE"
MAX_EPISODES_PER_CASSETTE = 50

# zstd standard frame magic. We rely on this to discriminate at load time
# between legacy uncompressed YAML (which never starts with these four
# bytes — YAML cassettes always begin with "!!python/object" or
# "interactions:") and freshly compressed payloads. That keeps the
# rollout backward compatible with any cassettes already cached as raw
# YAML when this code first ships.
_ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_COMPRESSION_LEVEL = 10

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_log = logging.getLogger(__name__)
_passed_by_cassette_key: dict[str, bool] = {}


class VCRCassetteCacheWarning(UserWarning):
    """Emitted when the cassette cache fails to load or save.

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

    Only meaningful for the Redis backend; the S3/R2 backend has no
    fixed capacity ceiling so this returns ``None`` there too.
    """
    try:
        if client is None:
            client = _maybe_build_redis_client()
            if client is None:
                return None
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
    """Return the canonical cassette key used by every backend.

    The name is preserved for backward compatibility — see
    :func:`cassette_key_for` for the alias used by new call sites.
    """
    abs_path = os.path.abspath(str(cassette_path))
    try:
        rel = os.path.relpath(abs_path, start=_REPO_ROOT)
    except ValueError:
        rel = os.path.basename(abs_path)
    if rel.endswith(".yaml"):
        rel = rel[: -len(".yaml")]
    rel = rel.replace("/cassettes/", "/").lstrip("./")
    return f"{REDIS_KEY_PREFIX}{rel}"


cassette_key_for = redis_key_for


def _redis_url_from_env() -> Optional[str]:
    return os.environ.get(CASSETTE_REDIS_URL_ENV) or None


def _s3_bucket_from_env() -> Optional[str]:
    return os.environ.get(CASSETTE_S3_BUCKET_ENV) or None


def _local_cache_dir_from_env() -> Optional[str]:
    return os.environ.get(CASSETTE_LOCAL_CACHE_DIR_ENV) or None


def _compression_disabled() -> bool:
    return os.environ.get(CASSETTE_DISABLE_COMPRESSION_ENV) == "1"


# ---------------------------------------------------------------------------
# Compression: zstd with magic-byte sniffing for legacy cassettes.
# ---------------------------------------------------------------------------


def _compress(payload: bytes) -> bytes:
    """Compress ``payload`` with zstd. Idempotent on already-compressed input.

    YAML cassettes are extremely repetitive (JSON-in-string bodies, SSE
    chunk headers, repeated header keys) and routinely shrink 6-12x.
    The compressed bytes start with the zstd frame magic, which the
    YAML serializer never emits, so :func:`_decompress` can sniff the
    format on load without a separate metadata key.
    """
    if payload.startswith(_ZSTD_FRAME_MAGIC):
        return payload
    if _compression_disabled():
        return payload
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - dev dependency
        return payload
    return zstandard.ZstdCompressor(level=_ZSTD_COMPRESSION_LEVEL).compress(payload)


def _decompress(payload: bytes) -> bytes:
    """Decompress zstd-framed payloads; pass legacy bytes through unchanged.

    The frame-magic check means we can roll this out without flushing
    the existing Redis cache: payloads written before this change start
    with ``"!!python/object"`` (or ``"interactions:"``) — neither of
    which collides with ``b"\\x28\\xb5\\x2f\\xfd"`` — and load as-is.
    """
    if not payload.startswith(_ZSTD_FRAME_MAGIC):
        return payload
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - dev dependency
        raise
    return zstandard.ZstdDecompressor().decompress(payload)


# ---------------------------------------------------------------------------
# Backends: Redis (legacy) and S3/R2 (zero-egress alternative).
# ---------------------------------------------------------------------------


class CassetteBackend(Protocol):
    """Pluggable cassette store. Implementations only deal in raw bytes."""

    name: str
    transient_error_types: tuple

    def get(self, key: str) -> Optional[bytes]: ...

    def set(self, key: str, payload: bytes, ttl_seconds: int) -> None: ...


def _maybe_build_redis_client() -> Optional[Any]:
    url = _redis_url_from_env()
    if not url:
        return None
    try:
        import redis
        from redis.backoff import ExponentialBackoff
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError
        from redis.retry import Retry
    except ImportError:  # pragma: no cover - redis is a hard test dep
        return None
    return redis.Redis.from_url(
        url,
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,
        retry=Retry(ExponentialBackoff(cap=2, base=0.1), retries=2),
        retry_on_error=[RedisConnectionError, RedisTimeoutError],
    )


def _build_default_client():
    """Backward-compat shim used by the legacy ``make_redis_persister``."""
    client = _maybe_build_redis_client()
    if client is None:
        raise RuntimeError(
            f"Set {CASSETTE_REDIS_URL_ENV} to enable the VCR persister. "
            "Cassette Redis is intentionally separate from the application "
            "Redis (REDIS_URL/REDIS_HOST) to avoid being flushed by tests."
        )
    return client


class _RedisBackend:
    name = "redis"

    def __init__(self, client: Any) -> None:
        self._client = client
        try:
            from redis.exceptions import RedisError
        except ImportError:  # pragma: no cover - redis is a hard test dep
            RedisError = Exception  # type: ignore[assignment,misc]
        self.transient_error_types = (RedisError,)

    def get(self, key: str) -> Optional[bytes]:
        return self._client.get(key)

    def set(self, key: str, payload: bytes, ttl_seconds: int) -> None:
        self._client.set(key, payload, ex=ttl_seconds)

    def info_memory(self):
        return self._client.info(section="memory")


class _S3Backend:
    """S3-compatible object store backend. Works with AWS S3, Cloudflare
    R2, Backblaze B2, MinIO, etc.

    TTL is enforced on read by checking the object's ``LastModified``
    timestamp; objects older than ``ttl_seconds`` are treated as cache
    misses so the test re-records and re-uploads. Bucket lifecycle
    rules should be configured separately to actually evict the bytes
    (otherwise reads succeed-but-treat-as-miss until the lifecycle
    sweep runs). With Cloudflare R2, this pattern means storage stays
    near-zero cost, no egress fees on cassette reads, and the soft
    TTL semantics match the original Redis persister.
    """

    name = "s3"

    def __init__(
        self,
        client: Any,
        bucket: str,
        ttl_seconds: int = CASSETTE_TTL_SECONDS,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._ttl = ttl_seconds
        try:
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:  # pragma: no cover - boto3 is a dev dep
            ClientError = Exception  # type: ignore[assignment,misc]
            BotoCoreError = Exception  # type: ignore[assignment,misc]
        self._client_error = ClientError
        self.transient_error_types = (ClientError, BotoCoreError, OSError)

    def get(self, key: str) -> Optional[bytes]:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
        except self._client_error as exc:
            code = self._error_code(exc)
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        last_modified = obj.get("LastModified")
        if last_modified is not None:
            try:
                age = time.time() - last_modified.timestamp()
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                age = 0.0
            if age > self._ttl:
                return None
        body = obj["Body"].read()
        return body if isinstance(body, (bytes, bytearray)) else bytes(body)

    def set(self, key: str, payload: bytes, ttl_seconds: int) -> None:
        # ttl_seconds is unused on the write path — the bucket's
        # lifecycle policy plus the read-side age check enforce TTL.
        self._client.put_object(Bucket=self._bucket, Key=key, Body=payload)

    @staticmethod
    def _error_code(exc: BaseException) -> str:
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            err = response.get("Error", {})
            if isinstance(err, dict):
                return str(err.get("Code", "")) or ""
        return ""


def _maybe_build_s3_backend(
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
) -> Optional[_S3Backend]:
    bucket = _s3_bucket_from_env()
    if not bucket:
        return None
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.config import Config  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dev dependency
        warnings.warn(
            f"{CASSETTE_S3_BUCKET_ENV} is set but boto3 is not installed: {exc}",
            VCRCassetteCacheWarning,
            stacklevel=2,
        )
        return None
    endpoint = os.environ.get(CASSETTE_S3_ENDPOINT_ENV) or None
    region = os.environ.get(CASSETTE_S3_REGION_ENV) or "auto"
    config = Config(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=10,
        # R2 requires path-style addressing; AWS S3 accepts it too.
        s3={"addressing_style": "path"},
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=config,
    )
    return _S3Backend(client=client, bucket=bucket, ttl_seconds=ttl_seconds)


# ---------------------------------------------------------------------------
# Local-disk L1 cache. Wraps any backend and is read-through / write-through.
# CircleCI's `restore_cache` / `save_cache` mounts the directory across runs
# so most replays never touch the remote backend at all.
# ---------------------------------------------------------------------------


class _LocalDiskL1Cache:
    name = "disk-l1"

    def __init__(
        self,
        base_dir: str,
        inner: CassetteBackend,
        ttl_seconds: int = CASSETTE_TTL_SECONDS,
    ) -> None:
        self._base = os.path.abspath(base_dir)
        self._inner = inner
        self._ttl = ttl_seconds
        os.makedirs(self._base, exist_ok=True)
        # Inherit the inner backend's exception classification so the
        # persister's best-effort ``except`` block still catches remote
        # failures even when L1 is in front.
        self.transient_error_types = tuple(
            set(getattr(inner, "transient_error_types", (Exception,))) | {OSError}
        )

    def _path_for(self, key: str) -> str:
        # Hash the full key to avoid filesystem-illegal characters and
        # path-length issues on Windows / shallow filesystems. Two-byte
        # shard prefix keeps any single directory under ~256 entries on
        # average — friendly to ``ls`` and to CircleCI cache restore.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self._base, digest[:2], digest)

    def get(self, key: str) -> Optional[bytes]:
        path = self._path_for(key)
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return self._fall_through(key, path)
        except OSError as exc:
            _log.debug("L1 stat failed for %s: %s", path, exc)
            return self._fall_through(key, path)
        if (time.time() - stat.st_mtime) > self._ttl:
            return self._fall_through(key, path)
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError as exc:
            _log.debug("L1 read failed for %s: %s", path, exc)
            return self._fall_through(key, path)

    def _fall_through(self, key: str, path: str) -> Optional[bytes]:
        data = self._inner.get(key)
        if data is not None:
            self._write_through(path, data)
        return data

    def set(self, key: str, payload: bytes, ttl_seconds: int) -> None:
        # Write to the remote first — the local copy is only useful if
        # the same payload is accessible to other shards via the remote
        # store. If the remote write fails, the persister's outer
        # ``except`` clause handles it; we still update L1 so the
        # current process at least has a consistent local view.
        try:
            self._inner.set(key, payload, ttl_seconds)
        finally:
            self._write_through(self._path_for(key), payload)

    def info_memory(self):
        # Forward the capacity probe through to the remote backend so
        # the existing health banner keeps reporting Redis usage.
        inner_probe = getattr(self._inner, "info_memory", None)
        if callable(inner_probe):
            return inner_probe()
        return {}

    @staticmethod
    def _write_through(path: str, payload: bytes) -> None:
        directory = os.path.dirname(path)
        try:
            os.makedirs(directory, exist_ok=True)
            # Atomic-ish write: tmp + rename so a crashed CI run never
            # leaves a half-written file that the next run misreads as
            # a cache hit. tempfile.mkstemp is on the same filesystem
            # so rename is atomic.
            fd, tmp = tempfile.mkstemp(prefix=".vcrtmp", dir=directory)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
                os.replace(tmp, path)
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            _log.debug("L1 write-through failed for %s: %s", path, exc)


def _select_remote_backend(
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
) -> CassetteBackend:
    """Pick a remote backend based on env vars. S3 takes precedence
    because it's the cheaper option once configured.
    """
    s3_backend = _maybe_build_s3_backend(ttl_seconds=ttl_seconds)
    if s3_backend is not None:
        return s3_backend
    redis_client = _maybe_build_redis_client()
    if redis_client is not None:
        return _RedisBackend(client=redis_client)
    raise RuntimeError(
        f"No cassette backend configured. Set {CASSETTE_S3_BUCKET_ENV} for "
        f"S3/R2 or {CASSETTE_REDIS_URL_ENV} for Redis. The cassette store is "
        "intentionally separate from the application's Redis "
        "(REDIS_URL/REDIS_HOST) to avoid being flushed by tests."
    )


def _maybe_layer_l1(
    backend: CassetteBackend,
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
) -> CassetteBackend:
    cache_dir = _local_cache_dir_from_env()
    if not cache_dir:
        return backend
    try:
        return _LocalDiskL1Cache(cache_dir, backend, ttl_seconds=ttl_seconds)
    except OSError as exc:
        warnings.warn(
            f"Failed to initialize VCR L1 cache at {cache_dir!r}: {exc}",
            VCRCassetteCacheWarning,
            stacklevel=2,
        )
        return backend


# ---------------------------------------------------------------------------
# Persister factory.
# ---------------------------------------------------------------------------


def make_persister(
    backend: Optional[CassetteBackend] = None,
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
):
    """Build a vcrpy persister wired to the configured backend.

    Selection rules:

    1. If an explicit ``backend`` is passed, use it (mainly for tests).
    2. If ``CASSETTE_S3_BUCKET`` is set, use the S3/R2 backend.
    3. Otherwise fall back to the Redis backend.
    4. If ``CASSETTE_LOCAL_CACHE_DIR`` is set, layer a local-disk L1
       cache in front of the chosen backend.
    """
    if backend is None:
        backend = _select_remote_backend(ttl_seconds=ttl_seconds)
    backend = _maybe_layer_l1(backend, ttl_seconds=ttl_seconds)
    transient_error_types = tuple(
        getattr(backend, "transient_error_types", (Exception,))
    )

    class _Persister:
        @staticmethod
        def load_cassette(cassette_path, serializer):
            key = redis_key_for(cassette_path)
            try:
                data = backend.get(key)
            except transient_error_types as exc:
                _record_cache_failure("load", exc)
                msg = (
                    f"VCR cache load failed for {cassette_path}; treating "
                    f"as cache miss: {type(exc).__name__}: {exc}"
                )
                _log.warning(msg)
                warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)
                raise CassetteNotFoundError() from exc
            if data is None:
                raise CassetteNotFoundError()
            try:
                data = _decompress(data)
            except Exception as exc:  # pragma: no cover - defensive
                _record_cache_failure("load", exc)
                msg = (
                    f"VCR cache decompress failed for {cassette_path}; "
                    f"treating as miss: {type(exc).__name__}: {exc}"
                )
                _log.warning(msg)
                warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)
                raise CassetteNotFoundError() from exc
            if isinstance(data, (bytes, bytearray)):
                data = bytes(data).decode("utf-8")
            return deserialize(data, serializer)

        @staticmethod
        def save_cassette(cassette_path, cassette_dict, serializer):
            key = redis_key_for(cassette_path)
            passed = _passed_by_cassette_key.pop(key, True)
            episode_count = len(cassette_dict.get("requests", []) or [])
            if episode_count > MAX_EPISODES_PER_CASSETTE:
                _log.warning(
                    "VCR cache save refused for %s; cassette has %d episodes "
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
                    "VCR cache save skipped for %s; test did not pass — "
                    "leaving any prior cassette intact",
                    cassette_path,
                )
                return
            data = serialize(cassette_dict, serializer)
            payload = data.encode("utf-8") if isinstance(data, str) else data
            payload = _compress(payload)
            try:
                backend.set(key, payload, ttl_seconds)
            except transient_error_types as exc:
                # Cassette persistence is strictly best-effort: connection
                # blips, timeouts, OOM at the maxmemory cap, READONLY
                # replicas, S3 5xx, etc. should all degrade gracefully to
                # "test passed but cassette not cached" rather than failing
                # the test on teardown. We still want a loud signal so the
                # failure shows up in pytest's warnings summary at the end
                # of the session and feeds the session-end banner.
                _record_cache_failure("save", exc)
                msg = (
                    f"VCR cache save failed for {cassette_path}; cassette "
                    f"not persisted: {type(exc).__name__}: {exc}"
                )
                _log.warning(msg)
                warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)

    return _Persister


def make_redis_persister(
    client: Optional[Any] = None,
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
):
    """Backward-compatible factory used by existing tests and conftests.

    Prefer :func:`make_persister`. Passing ``client`` selects the Redis
    backend explicitly, bypassing env-var detection — useful for unit
    tests that wire up a fakeredis instance.
    """
    if client is None:
        return make_persister(ttl_seconds=ttl_seconds)
    return make_persister(
        backend=_RedisBackend(client=client),
        ttl_seconds=ttl_seconds,
    )


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
