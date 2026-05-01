"""Addon-level tests using a minimal fake-mitmproxy flow.

Mitmproxy itself is heavy and fiddly to install in CI for unit tests.
The addon only touches a small surface of the ``http.HTTPFlow`` API
(``request.method``, ``request.pretty_host``, ``request.pretty_url``,
``request.path``, ``request.headers``, ``request.raw_content``,
``response.status_code``, ``response.headers``, ``response.raw_content``,
``response.reason``, plus ``http.Response.make``), so we stub exactly
that subset and exercise the addon directly.
"""

from __future__ import annotations

import os
import sys
import types

import fakeredis

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

# Prefer the *real* mitmproxy if it's installed: that way the contract
# tests in ``test_replay_response_contract.py`` exercise the same
# objects this file does, and any future skew between our fake and
# real mitmproxy can't silently hide regressions. Fall through to the
# fake stub when mitmproxy isn't available (the typical unit-test env).
try:
    import mitmproxy  # noqa: F401  pre-load so setdefault below is a no-op
    import mitmproxy.http  # noqa: F401
except ImportError:
    pass

# Install a fake `mitmproxy` package *before* importing the addon, so the
# real mitmproxy doesn't need to be present in the unit-test env.
_mitmproxy_pkg = types.ModuleType("mitmproxy")
_http_mod = types.ModuleType("mitmproxy.http")


class _FakeHeaders(dict):
    def items(self, multi: bool = False):
        # Real mitmproxy ``Headers.items(multi=True)`` would return one
        # tuple per repeated header (e.g. multiple ``Set-Cookie``). The
        # fake doesn't model duplicates, so the kwarg is a no-op — we
        # just need to accept it without ``TypeError``.
        return list(super().items())


class _FakeResponse:
    """Fake mitmproxy ``http.Response`` that accepts both the real
    constructor signature (used by the addon's replay path) and a
    simplified positional signature (used by these tests when faking an
    upstream response).
    """

    def __init__(
        self,
        *args,
        status_code=None,
        body=None,
        headers=None,
        reason="",
        # real-mitmproxy kwargs (the addon passes these via _build_replay_response)
        http_version=b"HTTP/1.1",
        content=None,
        trailers=None,
        timestamp_start=None,
        timestamp_end=None,
    ):
        # Positional support: _FakeResponse(status_code, body=..., headers=...)
        if args:
            if status_code is None:
                status_code = args[0]
            if len(args) > 1 and body is None:
                body = args[1]
        # ``content`` (real mitmproxy) and ``body`` (test shorthand) are
        # the same field — raw_content.
        if body is None:
            body = content if content is not None else b""
        if isinstance(headers, list):
            # Bytes-tuple list (real mitmproxy contract) → str dict for
            # the fake's case-insensitive lookups.
            headers = {
                (k.decode("latin-1") if isinstance(k, bytes) else k): (
                    v.decode("latin-1") if isinstance(v, bytes) else v
                )
                for k, v in headers
            }
        if isinstance(reason, bytes):
            reason = reason.decode("latin-1")
        self.status_code = status_code
        self.raw_content = body
        self.headers = _FakeHeaders(headers or {})
        self.reason = reason
        self.http_version = http_version
        self.trailers = trailers
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end

    @classmethod
    def make(cls, status_code, body=b"", headers=None):
        if isinstance(headers, list):
            headers = dict(headers)
        return cls(status_code, body=body, headers=headers)


class _FakeRequest:
    def __init__(
        self,
        method,
        url,
        body=b"",
        headers=None,
        host="api.openai.com",
        path="/v1/chat/completions",
    ):
        self.method = method
        self.pretty_url = url
        self.pretty_host = host
        self.path = path
        self.raw_content = body
        self.headers = _FakeHeaders(headers or {})


class _FakeFlow:
    def __init__(self, request, response=None):
        self.request = request
        self.response = response


_http_mod.Response = _FakeResponse  # type: ignore[attr-defined]
_http_mod.HTTPFlow = _FakeFlow  # type: ignore[attr-defined]
_mitmproxy_pkg.http = _http_mod  # type: ignore[attr-defined]
_mitmproxy_pkg.ctx = types.SimpleNamespace(log=types.SimpleNamespace(info=lambda *_a, **_kw: None))  # type: ignore[attr-defined]
sys.modules.setdefault("mitmproxy", _mitmproxy_pkg)
sys.modules.setdefault("mitmproxy.http", _http_mod)

from tests.e2e_cassette_proxy.addon import CassetteAddon  # noqa: E402
from tests.e2e_cassette_proxy.redis_store import (  # noqa: E402
    CachedResponse,
    RedisCassetteStore,
)


def _addon_with_fake_redis():
    fake = fakeredis.FakeStrictRedis()
    store = RedisCassetteStore(client=fake)
    return fake, CassetteAddon(store=store)


def _make_request(body=b'{"model":"gpt-4o","messages":[]}'):
    return _FakeRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        body=body,
        headers={"content-type": "application/json", "authorization": "Bearer sk-1"},
    )


def test_should_pass_through_when_no_cache_entry_exists():
    _, addon = _addon_with_fake_redis()
    flow = _FakeFlow(_make_request())
    addon.request(flow)
    assert flow.response is None  # mitmproxy will then hit the real upstream
    assert addon._stats["miss"] == 1


def test_should_persist_response_on_response_hook_when_2xx():
    fake, addon = _addon_with_fake_redis()
    flow = _FakeFlow(_make_request())
    addon.request(flow)  # miss
    flow.response = _FakeResponse(
        200,
        body=b'{"id":"chatcmpl-1"}',
        headers={"content-type": "application/json"},
        reason="OK",
    )
    addon.response(flow)
    assert addon._stats["stored"] == 1
    assert any(k.startswith(b"litellm:e2ecass:") for k in fake.keys("*"))


def test_should_short_circuit_on_subsequent_request_with_cache_hit():
    _, addon = _addon_with_fake_redis()
    first = _FakeFlow(_make_request())
    addon.request(first)
    first.response = _FakeResponse(
        200,
        body=b'{"id":"chatcmpl-1"}',
        headers={"content-type": "application/json"},
    )
    addon.response(first)

    # Second request: same canonical shape, different auth header.
    second_req = _make_request()
    second_req.headers["authorization"] = "Bearer sk-different"
    second = _FakeFlow(second_req)
    addon.request(second)
    assert second.response is not None
    assert second.response.status_code == 200
    assert second.response.raw_content == b'{"id":"chatcmpl-1"}'
    assert addon._stats["hit"] == 1


def test_should_not_persist_non_2xx_response():
    fake, addon = _addon_with_fake_redis()
    flow = _FakeFlow(_make_request())
    addon.request(flow)
    flow.response = _FakeResponse(
        500,
        body=b'{"error":"internal"}',
        headers={"content-type": "application/json"},
    )
    addon.response(flow)
    assert addon._stats["stored"] == 0
    assert len(fake.keys("*")) == 0


def test_should_skip_passthrough_hosts_completely():
    _, addon = _addon_with_fake_redis()
    req = _FakeRequest(
        method="POST",
        url="http://localhost:4000/key/generate",
        body=b"{}",
        headers={"content-type": "application/json"},
        host="localhost",
        path="/key/generate",
    )
    flow = _FakeFlow(req)
    addon.request(flow)
    assert flow.response is None
    assert addon._stats["skipped"] == 1


def test_replay_only_should_serve_599_on_miss():
    fake, _ = _addon_with_fake_redis()
    addon = CassetteAddon(store=RedisCassetteStore(client=fake))
    addon._replay_only = True

    flow = _FakeFlow(_make_request())
    addon.request(flow)
    assert flow.response is not None
    assert flow.response.status_code == 599


def test_record_only_should_not_short_circuit_even_on_hit():
    fake = fakeredis.FakeStrictRedis()
    store = RedisCassetteStore(client=fake)
    pre_addon = CassetteAddon(store=store)
    flow = _FakeFlow(_make_request())
    pre_addon.request(flow)
    flow.response = _FakeResponse(
        200,
        body=b'{"id":"chatcmpl-1"}',
        headers={"content-type": "application/json"},
    )
    pre_addon.response(flow)

    # Now flip a new addon into record-only mode and verify it never serves
    # from cache.
    record_only_addon = CassetteAddon(store=store)
    record_only_addon._record_only = True
    second = _FakeFlow(_make_request())
    record_only_addon.request(second)
    assert second.response is None
