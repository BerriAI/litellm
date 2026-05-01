"""mitmproxy addon: cache HTTPS request/response pairs in Redis.

Loaded by ``mitmdump -s addon.py``. Two hooks:

- ``request(flow)``: derive cache key, look up in Redis, short-circuit
  the response if we have one.
- ``response(flow)``: persist the upstream's response under that same
  cache key for the next run.

This intentionally caches *every* upstream call that flows through the
proxy, regardless of host. The expensive surface (LLM provider APIs)
is what we care about, but capturing everything also dedupes the
boring stuff (token endpoints, model-list calls, control-plane pings)
for free.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from mitmproxy import ctx, http  # type: ignore[import-not-found]

from tests.e2e_cassette_proxy.cache_key import (
    DEFAULT_HEADER_ALLOWLIST,
    DEFAULT_HEADER_BLOCKLIST,
    derive_cache_key,
)
from tests.e2e_cassette_proxy.redis_store import (
    CachedResponse,
    RedisCassetteStore,
)

_log = logging.getLogger("litellm.e2e_cassette_proxy.addon")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[E2ECASS] %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


# Hosts we should *never* cache — Redis itself, the proxy admin UI,
# anything pointed at localhost. Extended via env var
# ``LITELLM_E2E_CASS_PASSTHROUGH_HOSTS`` (comma-separated).
_PASSTHROUGH_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "host.docker.internal",
}


def _passthrough_hosts_from_env() -> set[str]:
    raw = os.environ.get("LITELLM_E2E_CASS_PASSTHROUGH_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _record_only() -> bool:
    return os.environ.get("LITELLM_E2E_CASS_RECORD_ONLY", "").lower() in ("1", "true")


def _replay_only() -> bool:
    return os.environ.get("LITELLM_E2E_CASS_REPLAY_ONLY", "").lower() in ("1", "true")


def _shape_summary(flow: http.HTTPFlow) -> str:
    return f"{flow.request.method} {flow.request.pretty_host}{flow.request.path}"


def _is_2xx(status_code: int) -> bool:
    return 200 <= status_code < 300


class CassetteAddon:
    """mitmproxy addon class. Created once at startup; ``request`` and
    ``response`` are invoked per flow."""

    def __init__(self, store: Optional[RedisCassetteStore] = None) -> None:
        self._store: Optional[RedisCassetteStore] = store
        self._passthrough = _PASSTHROUGH_HOSTS | _passthrough_hosts_from_env()
        self._record_only = _record_only()
        self._replay_only = _replay_only()
        self._stats = {"hit": 0, "miss": 0, "stored": 0, "skipped": 0}

    def load(self, loader) -> None:  # noqa: ARG002 - mitmproxy hook signature
        if self._store is None:
            try:
                self._store = RedisCassetteStore()
            except Exception as exc:
                _log.info(
                    f"redis-init-failed err={type(exc).__name__}: {exc}; "
                    f"running in passthrough mode"
                )
                self._store = None
        _log.info(
            f"loaded record_only={self._record_only} replay_only={self._replay_only} "
            f"passthrough_hosts={sorted(self._passthrough)}"
        )

    def done(self) -> None:
        _log.info(
            f"summary hits={self._stats['hit']} misses={self._stats['miss']} "
            f"stored={self._stats['stored']} skipped={self._stats['skipped']}"
        )

    def _should_skip(self, flow: http.HTTPFlow) -> bool:
        host = flow.request.pretty_host.lower()
        if host in self._passthrough:
            return True
        # CONNECT tunnels are handled implicitly by mitmproxy; only filter
        # the inner HTTP request/response. Method allowlist keeps the keys
        # readable in Redis.
        if flow.request.method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            return True
        return False

    def _key_for(self, flow: http.HTTPFlow) -> str:
        return derive_cache_key(
            method=flow.request.method,
            url=flow.request.pretty_url,
            body=flow.request.raw_content or b"",
            headers=dict(flow.request.headers),
            allowlist=DEFAULT_HEADER_ALLOWLIST,
            blocklist=DEFAULT_HEADER_BLOCKLIST,
        )

    def request(self, flow: http.HTTPFlow) -> None:
        if self._store is None or self._should_skip(flow):
            self._stats["skipped"] += 1
            return
        if self._record_only:
            return
        key = self._key_for(flow)
        cached = self._store.get(key)
        if cached is None:
            self._stats["miss"] += 1
            _log.info(f"miss key={key} {_shape_summary(flow)}")
            if self._replay_only:
                # Don't fall through to the upstream; serve a stable 599 so
                # the test surfaces the missing recording loudly.
                flow.response = http.Response.make(
                    599,
                    b"e2e-cassette-proxy: replay-only and no recording for this request",
                    {"content-type": "text/plain"},
                )
            return
        _log.info(
            f"hit  key={key} {_shape_summary(flow)} "
            f"status={cached.status_code} bytes={len(cached.body)}"
        )
        self._stats["hit"] += 1
        flow.response = http.Response.make(
            cached.status_code,
            cached.body,
            list(cached.headers),
        )

    def response(self, flow: http.HTTPFlow) -> None:
        if self._store is None or self._should_skip(flow):
            return
        if self._replay_only:
            return
        # If we already served from cache, don't re-store.
        if flow.response is None or flow.response.status_code == 599:
            return
        # Don't cache redirects, errors, or rate-limit responses — they're
        # not useful for replay and would just churn keys.
        if not _is_2xx(flow.response.status_code):
            return
        key = self._key_for(flow)
        cached = CachedResponse(
            status_code=flow.response.status_code,
            headers=tuple((k, v) for k, v in flow.response.headers.items()),
            body=flow.response.raw_content or b"",
            reason=flow.response.reason or "",
        )
        if self._store.set(key, cached):
            self._stats["stored"] += 1
            _log.info(
                f"store key={key} {_shape_summary(flow)} "
                f"status={cached.status_code} bytes={len(cached.body)}"
            )


# mitmproxy entrypoint.
addons = [CassetteAddon()]


# Re-exported for parity with mitmproxy quirks (some versions look for
# ``ctx`` symbol availability at import time).
__all__ = ["CassetteAddon", "addons", "ctx"]
