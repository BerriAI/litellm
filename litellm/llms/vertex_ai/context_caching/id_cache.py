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
"""

import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import litellm
from litellm.caching.in_memory_cache import InMemoryCache

SAFETY_MARGIN_SECONDS = 5
_MAX_ENTRIES = 1024

_EXPLICIT_CACHE_ID_CACHE = InMemoryCache(max_size_in_memory=_MAX_ENTRIES)

_RFC3339_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})$")


@dataclass(frozen=True, slots=True)
class ResolvedCacheId:
    name: str
    expire_time: Optional[str]


def _rfc3339_to_epoch(expire_time: str) -> Optional[float]:
    """Vertex/Gemini return e.g. ``2024-10-02T15:01:23.045123456Z`` (nanosecond, Z).

    ``datetime.fromisoformat`` only tolerates the ``Z`` suffix and >6 fractional
    digits on Python 3.11+, but litellm supports 3.10, so normalize first.
    """
    match = _RFC3339_RE.match(expire_time.strip())
    if match is None:
        return None
    base, frac, tz = match.group(1), match.group(2), match.group(3)
    micros = frac[:6].ljust(6, "0") if frac else "000000"
    if tz == "Z":
        offset = "+00:00"
    elif ":" in tz:
        offset = tz
    else:
        offset = f"{tz[:3]}:{tz[3:]}"
    try:
        return datetime.fromisoformat(f"{base}.{micros}{offset}").timestamp()
    except ValueError:
        return None


def expire_time_to_ttl(expire_time: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Seconds until ``expire_time`` minus the safety margin, or ``None`` when the
    value is absent, unparseable, or already within the margin (i.e. not cacheable)."""
    if not expire_time:
        return None
    epoch = _rfc3339_to_epoch(expire_time)
    if epoch is None:
        return None
    ttl = epoch - (time.time() if now is None else now) - SAFETY_MARGIN_SECONDS
    return ttl if ttl > 0 else None


def make_cache_id_key(
    content_key: str,
    custom_llm_provider: str,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
) -> str:
    """Namespace the content hash by everything that determines which cachedContents
    endpoint/tenant the LIST would target, so ids never leak across tenants.

    The credential is folded in as a stable, non-reversible hash (never the raw secret).
    """
    auth_id = hashlib.sha256((api_key or "").encode()).hexdigest()[:16]
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


def lookup_cache_id(key: str) -> Optional[str]:
    """Resolved cachedContent name for a live entry, else ``None``. No-op (miss) when
    the feature is disabled. ``str(...)`` guards ``InMemoryCache.get_cache``'s
    ``json.loads`` coercion, which would turn a numeric Gemini id into an int."""
    if not litellm.enable_vertex_context_cache_id_caching:
        return None
    hit = _EXPLICIT_CACHE_ID_CACHE.get_cache(key)
    if hit is None:
        return None
    return str(hit)


def store_cache_id(key: str, name: Optional[str], expire_time: Optional[str], now: Optional[float] = None) -> None:
    """Cache ``name`` under ``key`` with a ttl bounded by ``expire_time``. No-op when the
    feature is disabled, the name is empty, or the ttl is not derivable (safe degrade to
    always-LIST)."""
    if not name or not litellm.enable_vertex_context_cache_id_caching:
        return
    ttl = expire_time_to_ttl(expire_time, now=now)
    if ttl is None:
        return
    _EXPLICIT_CACHE_ID_CACHE.set_cache(key, name, ttl=ttl)
