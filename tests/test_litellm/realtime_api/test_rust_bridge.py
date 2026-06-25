"""Tests for the optional Rust-backed realtime path
(``litellm/realtime_api/rust_bridge.py``).

Mirrors the OCR rust_bridge tests in shape so the two stay reviewable side by
side. Covers the toggle/loader plumbing, the websockets-ClientConnection
adapter contract, and the `_arealtime` rust-vs-python routing decisions
(including the SSL fallback guard).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

import litellm

realtime_main = importlib.import_module("litellm.realtime_api.main")
rust_bridge = importlib.import_module("litellm.realtime_api.rust_bridge")


class RecordingConnect:
    """Fake ``RustRealtimeConnect`` that records calls and returns a
    controllable ``RecordingUpstream``."""

    def __init__(self, *, recv_frames: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.upstream = RecordingUpstream(recv_frames=recv_frames or [])

    def __call__(self, url: str, headers: dict[str, str]):
        self.calls.append({"url": url, "headers": headers})

        async def _connect():
            return self.upstream

        return _connect()


class RecordingUpstream:
    def __init__(self, *, recv_frames: list[str]) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._recv_frames = list(recv_frames)

    def send(self, text: str):
        async def _send():
            self.sent.append(text)

        return _send()

    def recv(self):
        async def _recv():
            if not self._recv_frames:
                raise StopAsyncIteration("upstream closed")
            return self._recv_frames.pop(0)

        return _recv()

    def close(self):
        async def _close():
            self.closed = True

        return _close()


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge.set_rust_realtime(False, connect=None)
    yield
    rust_bridge.set_rust_realtime(False, connect=None)


def test_use_litellm_rust_toggles_realtime_flag():
    assert rust_bridge.rust_realtime_enabled() is False
    litellm.use_litellm_rust()
    assert rust_bridge.rust_realtime_enabled() is True
    litellm.use_litellm_rust(False)
    assert rust_bridge.rust_realtime_enabled() is False


def test_load_rust_realtime_returns_injected_impl():
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    assert rust_bridge.load_rust_realtime() is bridge


def test_toggle_without_realtime_arg_preserves_injected_impl():
    """Regression guard mirroring the OCR one: routine enable/disable calls must
    not clobber a prior injection."""
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)

    litellm.use_litellm_rust(False)
    assert rust_bridge.load_rust_realtime() is bridge
    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_realtime() is bridge


def test_explicit_realtime_none_clears_injected_impl(monkeypatch):
    """After clearing an injected impl the loader must report nothing
    injected. We block the compiled extension import for this case so the test
    runs the same way in environments that did and didn't build the wheel."""
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)
    builtins = __import__("builtins")
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "litellm_python_bridge":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    litellm.use_litellm_rust(True, realtime=None)
    assert rust_bridge.load_rust_realtime() is None


def test_load_rust_realtime_none_when_extension_absent(monkeypatch):
    """With no injection and no compiled wheel, the loader returns None so the
    proxy degrades to the Python path instead of raising."""
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)
    builtins = __import__("builtins")
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "litellm_python_bridge":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_realtime() is None


def test_load_rust_realtime_uses_compiled_extension(monkeypatch):
    """With no injected impl but a compiled extension importable, the loader
    returns the extension's ``realtime_connect`` callable. The native wheel
    isn't built in CI, so we stand in a fake module via ``sys.modules``."""
    fake_module = types.ModuleType("litellm_python_bridge")
    fake_module.realtime_connect = (  # type: ignore[attr-defined]
        lambda url, headers: None
    )
    monkeypatch.setitem(sys.modules, "litellm_python_bridge", fake_module)

    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_realtime() is fake_module.realtime_connect


@pytest.mark.asyncio
async def test_rust_backend_websocket_send_and_async_iter():
    """The adapter must (a) marshal text frames through to the Rust handle,
    (b) iterate text frames lazily, and (c) stop iteration on a clean close."""
    upstream = RecordingUpstream(recv_frames=["frame-a", "frame-b"])
    backend = rust_bridge.RustBackendWebsocket(upstream)

    await backend.send("hello")
    assert upstream.sent == ["hello"]

    collected = [msg async for msg in backend]
    assert collected == ["frame-a", "frame-b"]


@pytest.mark.asyncio
async def test_rust_backend_websocket_recv_raises_connection_closed_ok():
    """``RealTimeStreaming.backend_to_client_send_messages`` catches
    ``websockets.exceptions.ConnectionClosed`` (which ``ConnectionClosedOK``
    subclasses) to flush logging cleanly. A clean close on the Rust side must
    surface as that exception so the logging path stays unaffected."""
    import websockets

    upstream = RecordingUpstream(recv_frames=[])
    backend = rust_bridge.RustBackendWebsocket(upstream)

    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await backend.recv()


@pytest.mark.asyncio
async def test_rust_backend_websocket_close_is_idempotent():
    upstream = RecordingUpstream(recv_frames=[])
    backend = rust_bridge.RustBackendWebsocket(upstream)

    await backend.close()
    await backend.close()
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_rust_backend_connect_factory_signature_matches_websockets():
    """The factory must accept the exact ``(url, *, additional_headers,
    max_size, ssl)`` kwargs ``OpenAIRealtime.async_realtime`` passes to
    ``websockets.connect`` so it drops in without branching the dial site."""
    bridge = RecordingConnect(recv_frames=[])
    factory = rust_bridge.rust_backend_connect_factory(bridge)

    async with factory(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime",
        additional_headers={"Authorization": "Bearer sk-test"},
        max_size=1234,
        ssl=None,
    ) as backend_ws:
        assert isinstance(backend_ws, rust_bridge.RustBackendWebsocket)

    assert bridge.calls == [
        {
            "url": "wss://api.openai.com/v1/realtime?model=gpt-realtime",
            "headers": {"Authorization": "Bearer sk-test"},
        }
    ]
    assert bridge.upstream.closed is True


@pytest.mark.asyncio
async def test_rust_backend_connect_factory_rejects_verify_off():
    """A user who set ``ssl_verify=False`` explicitly opted out of TLS
    verification; the Rust transport always verifies, so the factory must
    refuse rather than silently re-verifying behind their back."""
    bridge = RecordingConnect()
    factory = rust_bridge.rust_backend_connect_factory(bridge)

    with pytest.raises(NotImplementedError):
        async with factory("wss://example", additional_headers={}, ssl=False):
            pass


def test_maybe_rust_backend_connect_returns_none_when_disabled():
    assert realtime_main._maybe_rust_backend_connect(None) is None


def test_maybe_rust_backend_connect_returns_factory_when_enabled():
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    factory = realtime_main._maybe_rust_backend_connect(None)
    assert factory is not None and callable(factory)


def test_maybe_rust_backend_connect_falls_back_when_bridge_missing(monkeypatch):
    """Enabled but no impl: caller must fall back to Python."""
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)
    builtins = __import__("builtins")
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "litellm_python_bridge":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    litellm.use_litellm_rust(True)
    assert realtime_main._maybe_rust_backend_connect(None) is None


def test_maybe_rust_backend_connect_falls_back_when_verify_off(monkeypatch):
    """When the proxy explicitly disables TLS verification the Rust path must
    opt out so the operator's intent is preserved (Rust always verifies)."""
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    monkeypatch.setattr(
        realtime_main,
        "get_shared_realtime_ssl_context",
        lambda: False,
    )
    assert realtime_main._maybe_rust_backend_connect(None) is None


@pytest.mark.asyncio
async def test_openai_handler_passes_backend_connect_to_dial(monkeypatch):
    """End-to-end wiring proof: when ``_arealtime`` injects a Rust factory,
    ``OpenAIRealtime.async_realtime`` must hand it the same ``additional_headers``
    + ``ssl`` it would have passed ``websockets.connect`` (so OpenAI-Beta and
    query params round-trip identically through both transports)."""
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLogging,
    )
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    captured: dict[str, Any] = {}

    class _FakeBackend:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    def fake_connect(url, *, additional_headers, max_size, ssl):
        captured.update(
            {
                "url": url,
                "additional_headers": dict(additional_headers),
                "max_size": max_size,
                "ssl": ssl,
            }
        )
        return _FakeBackend()

    class _FakeClientWS:
        scope = {"headers": [(b"openai-beta", b"realtime=v1")]}

        async def close(self, *args, **kwargs):
            return None

    async def _fake_forward(self):
        captured["streaming_started"] = True

    monkeypatch.setattr(
        "litellm.litellm_core_utils.realtime_streaming.RealTimeStreaming.bidirectional_forward",
        _fake_forward,
    )

    logging_obj = LiteLLMLogging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="realtime",
        start_time=__import__("datetime").datetime.now(),
        litellm_call_id="test-call",
        function_id="fn",
    )

    await handler.async_realtime(
        model="gpt-realtime",
        websocket=_FakeClientWS(),
        logging_obj=logging_obj,
        api_base="https://api.openai.com/",
        api_key="sk-test",
        backend_connect=fake_connect,
    )

    assert captured["url"].startswith(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    )
    assert captured["additional_headers"]["Authorization"] == "Bearer sk-test"
    # Client-sent OpenAI-Beta must round-trip onto the upstream so the GA-vs-beta
    # protocol choice is preserved through the Rust transport.
    assert captured["additional_headers"]["OpenAI-Beta"] == "realtime=v1"
    assert captured["streaming_started"] is True
