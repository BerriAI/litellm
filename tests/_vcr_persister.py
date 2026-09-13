from __future__ import annotations

import logging
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serialize import serialize
from vcr.serializers import compat

CASSETTE_TTL_SECONDS = 24 * 60 * 60
CASSETTE_TTL_SECONDS_ENV = "CASSETTE_TTL_SECONDS"
CASSETTE_BACKEND_ENV = "CASSETTE_BACKEND"
CASSETTE_REDIS_URL_ENV = "CASSETTE_REDIS_URL"
VCR_VERBOSE_ENV = "LITELLM_VCR_VERBOSE"
MAX_EPISODES_PER_CASSETTE = 50

TTLValue = int | str

_log = logging.getLogger(__name__)
_passed_by_cassette_key: dict[str, bool] = {}
_ttl_by_cassette_key: dict[str, TTLValue] = {}


class VCRCassetteCacheWarning(UserWarning):
    """Emitted when cassette persistence fails to load or save."""


_cache_health = {
    "save_failures": 0,
    "save_failure_last_error": "",
    "load_failures": 0,
    "load_failure_last_error": "",
}


def _cassette_key(cassette_path: str) -> str:
    return os.path.abspath(str(cassette_path))


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


def mark_test_outcome_for_cassette(cassette_path: str, passed: bool) -> None:
    _passed_by_cassette_key[_cassette_key(cassette_path)] = passed


def set_cassette_ttl_override(cassette_path: str, ttl: TTLValue) -> None:
    _ttl_by_cassette_key[_cassette_key(cassette_path)] = ttl


def _parse_ttl(value: TTLValue) -> int:
    if isinstance(value, str) and value.lower() == "inf":
        return 0
    return int(value)


def resolve_cassette_ttl(cassette_path: str, fallback: int | None = None) -> int:
    key = _cassette_key(cassette_path)
    override = _ttl_by_cassette_key.pop(key, None)
    if override is not None:
        return _parse_ttl(override)
    env_value = os.environ.get(CASSETTE_TTL_SECONDS_ENV)
    if env_value is not None:
        return _parse_ttl(env_value)
    return CASSETTE_TTL_SECONDS if fallback is None else fallback


class CorruptCassetteError(ValueError):
    pass


class BaseCassettePersister(ABC):
    backend_name = "cassette"

    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._fallback_ttl_seconds = ttl_seconds

    def load_cassette(self, cassette_path, serializer):
        try:
            return self._load(cassette_path, serializer)
        except CassetteNotFoundError:
            raise
        except Exception as exc:
            corrupt = isinstance(exc, CorruptCassetteError)
            reported_exc = (
                exc.__cause__ if corrupt and exc.__cause__ is not None else exc
            )
            _record_cache_failure("load", reported_exc)
            detail = (
                "cached payload is corrupt, treating as cache miss"
                if corrupt
                else "treating as cache miss"
            )
            msg = (
                f"VCR {self.backend_name} load failed for {cassette_path}; {detail}: "
                f"{type(reported_exc).__name__}: {reported_exc}"
            )
            _log.warning(msg)
            warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)
            raise CassetteNotFoundError() from exc

    def save_cassette(self, cassette_path, cassette_dict, serializer):
        key = _cassette_key(cassette_path)
        passed = _passed_by_cassette_key.pop(key, True)
        episode_count = len(cassette_dict.get("requests", []) or [])
        if episode_count > MAX_EPISODES_PER_CASSETTE:
            _ttl_by_cassette_key.pop(key, None)
            _log.warning(
                "VCR %s save refused for %s; cassette has %d episodes "
                "(> MAX_EPISODES_PER_CASSETTE=%d). The test likely produces "
                "non-deterministic request bodies (e.g. uuid) and is "
                "appending instead of replaying. Opt it out with the "
                "no-vcr list in conftest, or stabilize its request body.",
                self.backend_name,
                cassette_path,
                episode_count,
                MAX_EPISODES_PER_CASSETTE,
            )
            return
        if not passed:
            _ttl_by_cassette_key.pop(key, None)
            _log.info(
                "VCR %s save skipped for %s; test did not pass — "
                "leaving any prior cassette intact",
                self.backend_name,
                cassette_path,
            )
            return
        try:
            ttl_seconds = resolve_cassette_ttl(
                cassette_path, self._fallback_ttl_seconds
            )
            self._save(
                cassette_path,
                cassette_dict,
                serializer,
                ttl_seconds=ttl_seconds,
            )
        except Exception as exc:
            _record_cache_failure("save", exc)
            msg = (
                f"VCR {self.backend_name} save failed for {cassette_path}; cassette "
                f"not persisted: {type(exc).__name__}: {exc}"
            )
            _log.warning(msg)
            warnings.warn(msg, VCRCassetteCacheWarning, stacklevel=2)

    def capacity_snapshot(self) -> dict | None:
        return None

    @abstractmethod
    def _load(self, cassette_path, serializer):
        raise NotImplementedError

    @abstractmethod
    def _save(
        self, cassette_path, cassette_dict, serializer, *, ttl_seconds: int
    ) -> None:
        raise NotImplementedError


class FilesystemBackend(BaseCassettePersister):
    backend_name = "filesystem"

    def __init__(
        self,
        ttl_seconds: int | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        replace: Callable[[str, str], None] = os.replace,
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._now = now
        self._replace = replace

    def _load(self, cassette_path, serializer):
        path = Path(cassette_path)
        if not path.is_file():
            raise CassetteNotFoundError()
        try:
            data = serializer.deserialize(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CorruptCassetteError(str(exc)) from exc
        if not isinstance(data, dict):
            raise CorruptCassetteError("cassette payload must be a mapping")
        if "recorded_at" not in data:
            raise CassetteNotFoundError()
        try:
            recorded_at = datetime.fromisoformat(data["recorded_at"])
            ttl_seconds = _parse_ttl(data["ttl_seconds"])
            interactions = data["interactions"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptCassetteError(str(exc)) from exc
        if recorded_at.tzinfo is None:
            raise CorruptCassetteError("recorded_at must include a timezone")
        if (
            ttl_seconds > 0
            and (self._now() - recorded_at).total_seconds() > ttl_seconds
        ):
            raise CassetteNotFoundError()
        try:
            requests = [Request._from_dict(item["request"]) for item in interactions]
            responses = [
                compat.convert_to_bytes(item["response"]) for item in interactions
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptCassetteError(str(exc)) from exc
        return requests, responses

    def _save(
        self, cassette_path, cassette_dict, serializer, *, ttl_seconds: int
    ) -> None:
        path = Path(cassette_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = serializer.deserialize(serialize(cassette_dict, serializer))
        data = {
            "version": normalized["version"],
            "recorded_at": self._now().isoformat(),
            "ttl_seconds": ttl_seconds,
            "interactions": normalized["interactions"],
        }
        payload = serializer.serialize(data)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as temporary:
                temporary.write(payload)
                temp_path = temporary.name
            self._replace(temp_path, str(path))
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)


def make_persister():
    backend = os.environ.get(CASSETTE_BACKEND_ENV)
    if backend == "filesystem":
        return FilesystemBackend()
    if backend == "redis" or os.environ.get(CASSETTE_REDIS_URL_ENV):
        from tests._vcr_redis_persister import make_redis_persister

        return make_redis_persister()
    if backend is not None:
        raise ValueError(f"Unsupported {CASSETTE_BACKEND_ENV}: {backend}")
    return FilesystemBackend()


def cassette_cache_capacity_snapshot(client: Any | None = None) -> dict | None:
    if client is not None:
        from tests._vcr_redis_persister import make_redis_persister

        return make_redis_persister(client=client).capacity_snapshot()
    return make_persister().capacity_snapshot()


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
