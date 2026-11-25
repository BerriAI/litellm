import asyncio
import contextlib
import re
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Final, Literal

KNOWN_CACHE_EXPIRY_MARGIN_SECONDS: Final = 10.0
MAX_KNOWN_CACHED_CONTENTS: Final = 10_000
_CACHE_REJECTED_STATUS_CODES: Final = frozenset({400, 403, 404})
_NAMES_CACHED_CONTENT: Final = re.compile(r"cached?\s*content", re.IGNORECASE)
_FORBIDS_FIELDS_BESIDE_A_CACHE: Final = re.compile(r"system[_\s]instruction", re.IGNORECASE)

_RFC3339_FRACTION: Final = re.compile(r"\.(\d{1,6})\d*")


def names_rejected_cache(status_code: int, message: str) -> bool:
    return (
        status_code in _CACHE_REJECTED_STATUS_CODES
        and _NAMES_CACHED_CONTENT.search(message) is not None
        and _FORBIDS_FIELDS_BESIDE_A_CACHE.search(message) is None
    )


def parse_expire_time(expire_time: object) -> float | None:
    """Parse a Vertex RFC 3339 ``expireTime`` (nanosecond precision, ``Z`` suffix) to epoch seconds."""
    if not isinstance(expire_time, str) or not expire_time:
        return None
    normalized: Final = _RFC3339_FRACTION.sub(lambda m: "." + m.group(1), expire_time.replace("Z", "+00:00"), count=1)
    try:
        parsed: Final = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    return parsed.timestamp()


class KnownCachedContents:
    """Process-local record of the cachedContents this process created or found.

    Vertex meters every cachedContents list page against a per-minute, per-location quota,
    so listing on every cache-marked request exhausts it under load. A cache is remembered
    until ``KNOWN_CACHE_EXPIRY_MARGIN_SECONDS`` before the ``expireTime`` Google reported,
    and after that requests list again, as they did before the record existed.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock: Final = clock
        self._entries: dict[tuple[tuple[str, str], str], tuple[str, float]] = {}  # mutable-ok: guarded by _lock
        self._lock: Final = threading.Lock()
        self._inflight: weakref.WeakKeyDictionary[  # mutable-ok: private, guarded by _lock
            asyncio.AbstractEventLoop, dict[tuple[tuple[str, str], str], asyncio.Future[str | None]]
        ] = weakref.WeakKeyDictionary()

    @staticmethod
    def scope(
        base_url: str,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        headers: Mapping[str, str],
        vertex_project: str | None,
        vertex_location: str | None,
    ) -> tuple[str, str]:
        """Where a cache name is valid: a Vertex project and location, or the project a Gemini key belongs to.

        A custom ``api_base`` can replace the whole Vertex path, so the project and location are kept beside the URL.
        The Gemini key is read from the headers actually sent, so an ``extra_headers`` override scopes the
        cache to the caller's project. It is kept as-is in this in-memory scope and never logged or persisted.
        """
        if custom_llm_provider != "gemini":
            return (base_url, f"{vertex_project}/{vertex_location}")
        sent_keys: Final = [value for name, value in headers.items() if name.lower() == "x-goog-api-key"]
        return (base_url, sent_keys[-1] if sent_keys else "")

    def has_not_expired(self, expire_time: object) -> bool:
        expires_at: Final = parse_expire_time(expire_time)
        return expires_at is None or expires_at > self._clock()

    def get(self, scope: tuple[str, str], cache_key: str) -> str | None:
        with self._lock:
            entry: Final = self._entries.get((scope, cache_key))
            if entry is None:
                return None
            name, reuse_until = entry
            if reuse_until > self._clock():
                return name
            del self._entries[(scope, cache_key)]
            return None

    def remember(self, scope: tuple[str, str], cache_key: str, name: str | None, expire_time: object) -> None:
        expires_at: Final = parse_expire_time(expire_time)
        if not name or expires_at is None:
            return
        reuse_until: Final = expires_at - KNOWN_CACHE_EXPIRY_MARGIN_SECONDS
        now: Final = self._clock()
        if reuse_until <= now:
            return
        with self._lock:
            if len(self._entries) >= MAX_KNOWN_CACHED_CONTENTS:
                self._entries = {k: v for k, v in self._entries.items() if v[1] > now}
                while len(self._entries) >= MAX_KNOWN_CACHED_CONTENTS:
                    del self._entries[next(iter(self._entries))]
            self._entries[(scope, cache_key)] = (name, reuse_until)

    def forget_rejected(self, cached_content: str | None, status_code: int) -> None:
        if cached_content is None or status_code not in _CACHE_REJECTED_STATUS_CODES:
            return
        with self._lock:
            self._entries = {key: entry for key, entry in self._entries.items() if entry[0] != cached_content}

    async def single_flight(
        self,
        scope: tuple[str, str],
        cache_key: str,
        resolve: Callable[[], Awaitable[str | None]],
        wait_seconds: float | None,
    ) -> str | None:
        """Run ``resolve`` once per (scope, cache_key) at a time within this event loop.

        Concurrent requests for the same prefix wait for the first one's lookup-or-create
        instead of each listing (and creating a duplicate cache). A waiter resolves on its
        own when that first attempt fails or has not finished within ``wait_seconds``, and
        never cancels it.
        """
        loop: Final = asyncio.get_running_loop()
        own: Final[asyncio.Future[str | None]] = loop.create_future()
        with self._lock:
            inflight: Final = self._inflight.setdefault(loop, {})
            leader: Final = inflight.setdefault((scope, cache_key), own)
        if leader is not own:
            with contextlib.suppress(asyncio.TimeoutError):
                if name := await asyncio.wait_for(asyncio.shield(leader), wait_seconds):
                    return name
            return await resolve()
        try:
            resolved: Final = await resolve()
        except BaseException:
            own.set_result(None)
            raise
        else:
            own.set_result(resolved)
            return resolved
        finally:
            with self._lock:
                inflight.pop((scope, cache_key), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._inflight.clear()


known_cached_contents: Final = KnownCachedContents()
