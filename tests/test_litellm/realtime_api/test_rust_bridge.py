"""Tests for the optional Rust-backed realtime path."""

from __future__ import annotations

import importlib
import asyncio
import ssl
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm

realtime_main = importlib.import_module("litellm.realtime_api.main")
rust_bridge = importlib.import_module("litellm.realtime_api.rust_bridge")


class RecordingUpstream:
    def __init__(self, *, recv_frames: list[str | bytes] | None = None) -> None:
        self.sent: list[str | bytes] = []
        self.closed = False
        self._recv_frames = list(recv_frames or [])

    def send(self, message: str | bytes):
        async def _send() -> None:
            self.sent.append(message)

        return _send()

    def recv(self):
        async def _recv() -> str | bytes:
            if not self._recv_frames:
                raise StopAsyncIteration("upstream closed")
            return self._recv_frames.pop(0)

        return _recv()

    def close(self):
        async def _close() -> None:
            self.closed = True

        return _close()


class RecordingConnect:
    def __init__(self, *, recv_frames: list[str | bytes] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.upstream = RecordingUpstream(recv_frames=recv_frames)

    def __call__(self, url: str, headers: dict[str, str], max_size: int | None = None):
        self.calls.append({"url": url, "headers": headers, "max_size": max_size})

        async def _connect() -> RecordingUpstream:
            return self.upstream

        return _connect()


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    ssl_verify = litellm.ssl_verify
    ssl_certificate = litellm.ssl_certificate
    ssl_security_level = litellm.ssl_security_level
    ssl_ecdh_curve = litellm.ssl_ecdh_curve
    rust_bridge.set_rust_realtime(False, connect=None)
    yield
    litellm.ssl_verify = ssl_verify
    litellm.ssl_certificate = ssl_certificate
    litellm.ssl_security_level = ssl_security_level
    litellm.ssl_ecdh_curve = ssl_ecdh_curve
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
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)

    litellm.use_litellm_rust(False)
    assert rust_bridge.load_rust_realtime() is bridge
    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_realtime() is bridge


def test_explicit_realtime_none_clears_injected_impl(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)

    litellm.use_litellm_rust(True, realtime=None)

    assert rust_bridge.load_rust_realtime() is None


def test_load_rust_realtime_none_when_extension_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)
    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_realtime() is None


def test_load_rust_realtime_uses_compiled_extension(monkeypatch):
    fake_module = types.ModuleType("litellm_python_bridge")
    fake_module.realtime_connect = lambda url, headers, max_size=None: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm_python_bridge", fake_module)

    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_realtime() is fake_module.realtime_connect


@pytest.mark.asyncio
async def test_rust_backend_connect_factory_matches_websockets_shape():
    bridge = RecordingConnect()
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
            "max_size": 1234,
        }
    ]
    assert bridge.upstream.closed is True


@pytest.mark.asyncio
async def test_rust_backend_connect_factory_maps_http_status():
    import websockets.exceptions as websocket_exceptions

    def connect(url: str, headers: dict[str, str], max_size: int | None = None):
        async def _connect() -> RecordingUpstream:
            _ = (url, headers, max_size)
            raise RuntimeError("upstream websocket HTTP status 401")

        return _connect()

    factory = rust_bridge.rust_backend_connect_factory(connect)

    with pytest.raises(websocket_exceptions.InvalidStatusCode) as exc:
        async with factory("wss://example.test", additional_headers={}, max_size=1):
            pass
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_rust_backend_connect_factory_rejects_custom_ssl_context():
    bridge = RecordingConnect()
    factory = rust_bridge.rust_backend_connect_factory(bridge)

    with pytest.raises(NotImplementedError):
        async with factory(
            "wss://example.test",
            additional_headers={},
            ssl=ssl.create_default_context(),
        ):
            pass


@pytest.mark.asyncio
async def test_rust_backend_websocket_recv_raises_connection_closed_ok():
    import websockets

    backend = rust_bridge.RustBackendWebsocket(RecordingUpstream())

    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await backend.recv()


@pytest.mark.asyncio
async def test_rust_backend_websocket_preserves_binary_frames():
    upstream = RecordingUpstream(recv_frames=[b"abc"])
    backend = rust_bridge.RustBackendWebsocket(upstream)

    await backend.send(b"client-binary")

    assert await backend.recv() == b"abc"
    assert upstream.sent == [b"client-binary"]


def test_maybe_rust_backend_connect_returns_none_when_disabled():
    assert realtime_main._maybe_rust_backend_connect() is None


def test_maybe_rust_backend_connect_returns_factory_when_enabled():
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    factory = realtime_main._maybe_rust_backend_connect()
    assert factory is not None and callable(factory)


def test_maybe_rust_backend_connect_falls_back_when_bridge_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)
    litellm.use_litellm_rust(True)
    assert realtime_main._maybe_rust_backend_connect() is None


def test_maybe_rust_backend_connect_falls_back_when_verify_off(monkeypatch):
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    monkeypatch.setattr(
        realtime_main,
        "get_shared_realtime_ssl_context",
        lambda: False,
    )
    assert realtime_main._maybe_rust_backend_connect() is None


def test_maybe_rust_backend_connect_falls_back_for_custom_tls(monkeypatch):
    bridge = RecordingConnect()
    litellm.use_litellm_rust(True, realtime=bridge)
    monkeypatch.setattr(litellm, "ssl_verify", ssl.create_default_context())
    monkeypatch.setattr(
        realtime_main,
        "get_shared_realtime_ssl_context",
        ssl.create_default_context,
    )
    assert realtime_main._maybe_rust_backend_connect() is None

    monkeypatch.setattr(litellm, "ssl_verify", "/tmp/custom-ca.pem")
    monkeypatch.setattr(
        realtime_main,
        "get_shared_realtime_ssl_context",
        lambda: "/tmp/custom-ca.pem",
    )
    assert realtime_main._maybe_rust_backend_connect() is None


@pytest.mark.asyncio
async def test_openai_handler_passes_backend_connect_to_dial(monkeypatch):
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    captured: dict[str, Any] = {}

    class FakeBackend:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    def fake_connect(url: str, *, additional_headers, max_size, ssl):
        captured.update(
            {
                "url": url,
                "additional_headers": dict(additional_headers),
                "max_size": max_size,
                "ssl": ssl,
            }
        )
        return FakeBackend()

    class FakeClientWebsocket:
        scope = {"headers": [(b"openai-beta", b"realtime=v1")]}

        async def close(self, *args, **kwargs):
            return None

    async def fake_forward(self):
        captured["streaming_started"] = True

    monkeypatch.setattr(
        "litellm.litellm_core_utils.realtime_streaming.RealTimeStreaming.bidirectional_forward",
        fake_forward,
    )

    logging_obj = LiteLLMLogging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="realtime",
        start_time=datetime.now(),
        litellm_call_id="test-call",
        function_id="fn",
    )
    pre_call_payload: dict[str, Any] = {}
    original_pre_call = logging_obj.pre_call

    def capture_pre_call(*args: Any, **kwargs: Any):
        pre_call_payload.update(kwargs)
        return original_pre_call(*args, **kwargs)

    monkeypatch.setattr(logging_obj, "pre_call", capture_pre_call)

    await handler.async_realtime(
        model="gpt-realtime",
        websocket=FakeClientWebsocket(),
        logging_obj=logging_obj,
        api_base="https://api.openai.com/",
        api_key="sk-test",
        backend_connect=fake_connect,
    )

    assert captured["url"].startswith(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    )
    assert captured["additional_headers"]["Authorization"] == "Bearer sk-test"
    assert captured["additional_headers"]["OpenAI-Beta"] == "realtime=v1"
    assert captured["ssl"] is None
    assert captured["streaming_started"] is True
    complete_input = pre_call_payload["additional_args"]["complete_input_dict"]
    assert complete_input["rust_bridge"] is True


@pytest.mark.asyncio
async def test_realtime_logging_flushes_through_rust_backend_adapter():
    from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

    client_ws = MagicMock()
    client_ws.send_text = AsyncMock()
    backend_ws = rust_bridge.RustBackendWebsocket(
        RecordingUpstream(
            recv_frames=[
                '{"type":"session.created","session":{"id":"sess_1"}}',
                '{"type":"response.done","response":{"output":[]}}',
            ]
        )
    )
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.success_handler = MagicMock()

    streaming = RealTimeStreaming(client_ws, backend_ws, logging_obj)
    await streaming.backend_to_client_send_messages()
    await asyncio.sleep(0)

    assert client_ws.send_text.await_count == 2
    logging_obj.async_success_handler.assert_awaited_once()
    logged_events = logging_obj.async_success_handler.call_args.args[0]
    assert [event["type"] for event in logged_events] == [
        "session.created",
        "response.done",
    ]
