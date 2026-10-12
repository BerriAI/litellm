"""
In-memory cache of resolved Vertex/Gemini explicit context-cache ids.

The discovery path in ``vertex_ai_context_caching.py`` otherwise re-resolves the
cachedContent id on every request via a live ``cachedContents`` LIST (p50 1552ms). This
caches the resolved ``name`` in-memory so a warm request skips that round trip.

Off by default; enable via ``litellm.enable_vertex_context_cache_id_caching``.

Correctness invariants:
- Each entry's ttl is derived from the backing cachedContent's real ``expireTime``
  minus a safety margin, never a fixed span from insertion, so a hit can never
  outlive its backing cache (which would fail downstream generateContent with NOT_FOUND).
- The key is namespaced by the resolving endpoint/tenant so an id created under one
  provider/project/location/endpoint/credential is never served to another.
- ttl bounds natural expiry, but a cachedContent can also be deleted/revoked server-side
  before then. ``invalidate_cache_id`` drops such a poisoned entry when generateContent
  rejects it downstream (Vertex reports a deleted cache as HTTP 400 INVALID_ARGUMENT,
  "Invalid resource state for cache content <id>"), so the next request re-resolves.
"""

import hmac
import json
import re
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

import litellm
from litellm.caching.in_memory_cache import InMemoryCache

SAFETY_MARGIN_SECONDS: Final = 5
_MAX_ENTRIES: Final = 1024


class _CacheStore(Protocol):
    def get_cache(self, key: str) -> object | None: ...

    def set_cache(self, key: str, value: str, *, ttl: float) -> None: ...

    def delete_cache(self, key: str) -> None: ...


def _new_cache_store() -> _CacheStore:
    return InMemoryCache(max_size_in_memory=_MAX_ENTRIES)


_EXPLICIT_CACHE_ID_CACHE: Final[_CacheStore] = _new_cache_store()
_CACHE_LOCK: Final = threading.Lock()
_CACHE_KEY_SECRET: Final = secrets.token_bytes(32)

_RFC3339_RE: Final = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})$")


@dataclass(frozen=True, slots=True)
class ResolvedCacheId:
    name: str
    expire_time: str | None


def _rfc3339_to_epoch(expire_time: str) -> float | None:
    """Vertex/Gemini return e.g. ``2024-10-02T15:01:23.045123456Z`` (nanosecond, Z).

    ``datetime.fromisoformat`` only tolerates the ``Z`` suffix and >6 fractional
    digits on Python 3.11+, but litellm supports 3.10, so normalize first.
    """
    match: Final = _RFC3339_RE.match(expire_time.strip())
    if match is None:
        return None
    base, frac, tz = match.group(1), match.group(2), match.group(3)
    micros: Final = frac[:6].ljust(6, "0") if frac else "000000"
    offset: Final = "+00:00" if tz == "Z" else tz if ":" in tz else f"{tz[:3]}:{tz[3:]}"
    try:
        return datetime.fromisoformat(f"{base}.{micros}{offset}").timestamp()
    except ValueError:
        return None


def expire_time_to_ttl(expire_time: str | None, now: float | None = None) -> float | None:
    """Seconds until ``expire_time`` minus the safety margin, or ``None`` when the
    value is absent, unparseable, or already within the margin (i.e. not cacheable)."""
    if not expire_time:
        return None
    epoch: Final = _rfc3339_to_epoch(expire_time)
    if epoch is None:
        return None
    ttl: Final = epoch - (time.time() if now is None else now) - SAFETY_MARGIN_SECONDS
    return ttl if ttl > 0 else None


def make_cache_id_key(
    content_key: str,
    custom_llm_provider: str,
    vertex_project: str | None,
    vertex_location: str | None,
    api_base: str | None,
    api_key: str | None,
    vertex_auth_header: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> str:
    """Namespace the content hash by everything that determines which cachedContents
    endpoint/tenant the LIST would target, so ids never leak across tenants.

    The credential is folded in as a stable, non-reversible hash (never the raw secret).
    """
    credentials: Final = json.dumps(
        {
            "api_key": api_key,
            "vertex_auth_header": vertex_auth_header,
            "extra_headers": sorted((name.lower(), value) for name, value in (extra_headers or {}).items()),
        },
        sort_keys=True,
    )
    auth_id: Final = hmac.new(_CACHE_KEY_SECRET, credentials.encode(), digestmod="sha256").hexdigest()
    return "|".join(
        (
            custom_llm_provider,
            vertex_project or "",
            vertex_location or "",
            api_base or "",
            auth_id,
            content_key,
        )
    )


def lookup_cache_id(key: str) -> str | None:
    """Resolved cachedContent name for a live entry, else ``None``. No-op (miss) when
    the feature is disabled. ``str(...)`` guards ``InMemoryCache.get_cache``'s
    ``json.loads`` coercion, which would turn a numeric Gemini id into an int."""
    if not litellm.enable_vertex_context_cache_id_caching:
        return None
    with _CACHE_LOCK:
        hit: Final = _EXPLICIT_CACHE_ID_CACHE.get_cache(key)
    if hit is None:
        return None
    return str(hit)


def store_cache_id(key: str, name: str | None, expire_time: str | None, now: float | None = None) -> None:
    """Cache ``name`` under ``key`` with a ttl bounded by ``expire_time``. No-op when the
    feature is disabled, the name is empty, or the ttl is not derivable (safe degrade to
    always-LIST)."""
    if not name or not litellm.enable_vertex_context_cache_id_caching:
        return
    ttl: Final = expire_time_to_ttl(expire_time, now=now)
    if ttl is None:
        return
    with _CACHE_LOCK:
        if _EXPLICIT_CACHE_ID_CACHE.get_cache(key) is None:
            _EXPLICIT_CACHE_ID_CACHE.set_cache(key, name, ttl=ttl)


def invalidate_cache_id(key: str) -> None:
    """Drop a resolved id after a downstream NOT_FOUND (server-side deleted/revoked
    cachedContent), so the next request re-resolves via LIST/create instead of re-serving
    the dead id. Safe no-op when the entry is already absent."""
    with _CACHE_LOCK:
        _EXPLICIT_CACHE_ID_CACHE.delete_cache(key)
