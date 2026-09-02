from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import bindings, streaming
from litellm.rust_bridge.runtime import BridgeErrorContext


class _FakeEventStream:
    def __init__(self, events: tuple[Mapping[str, object], ...]) -> None:
        self.metadata: Final = {
            "status_code": 200,
            "provider": "anthropic",
            "transport": "http",
            "response_headers": [{"name": "x-test", "value": "ready"}],
        }
        self._events: Final = iter(events)
        self.closed = False

    def next_event(self) -> Mapping[str, object] | None:
        if self.closed:
            return None
        return next(self._events, None)

    async def anext_event(self) -> Mapping[str, object] | None:
        return self.next_event()

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


class _RecordingOpen:
    def __init__(self, events: tuple[Mapping[str, object], ...]) -> None:
        self._events: Final = events
        self.calls = 0

    def __call__(
        self,
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> _FakeEventStream:
        self.calls += 1
        return _FakeEventStream(self._events)


class _RecordingAsyncOpen:
    def __init__(self, events: tuple[Mapping[str, object], ...]) -> None:
        self._events: Final = events
        self.calls = 0

    async def __call__(
        self,
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> _FakeEventStream:
        self.calls += 1
        return _FakeEventStream(self._events)


class _Declined(Exception):
    pass


class _Upstream(Exception):
    pass


class _NativeErrors:
    RustBridgeDeclined = _Declined
    RustUpstreamError = _Upstream


class _FailingOpen:
    def __init__(self, error: Exception) -> None:
        self._error: Final = error

    def __call__(
        self,
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> _FakeEventStream:
        raise self._error


class _FailingEventStream(_FakeEventStream):
    def next_event(self) -> Mapping[str, object] | None:
        raise _Upstream(502, "stream interrupted")


@pytest.fixture(autouse=True)
def reset_bridge() -> Iterator[None]:
    streaming.set_rust_streaming(
        capability=None,
        chat=None,
        achat=None,
        messages=None,
        amessages=None,
        responses=None,
        aresponses=None,
    )
    yield
    streaming.set_rust_streaming(
        capability=None,
        chat=None,
        achat=None,
        messages=None,
        amessages=None,
        responses=None,
        aresponses=None,
    )


def _chat_event(text: str) -> Mapping[str, object]:
    return {
        "text": text,
        "tool_use": None,
        "is_finished": False,
        "finish_reason": "",
        "usage": None,
    }


def _context() -> BridgeErrorContext:
    return BridgeErrorContext(route="chat completions stream", provider="anthropic", model="claude")


def test_no_native_capability_keeps_every_provider_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener: Final = _RecordingOpen((_chat_event("unused"),))
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    streaming.set_rust_streaming(chat=opener)

    result: Final = streaming.open_stream(
        api="chat_completions",
        provider="anthropic",
        request={"model": "claude", "messages": []},
        credentials={"api_key": "test"},
        api_base=None,
        extra_headers=None,
        timeout=None,
    )

    assert result is None
    assert opener.calls == 0


def test_disabled_capability_never_calls_native() -> None:
    opener: Final = _RecordingOpen((_chat_event("unused"),))
    streaming.set_rust_streaming(capability=lambda api, provider, transport: False, chat=opener)

    result: Final = streaming.open_stream(
        api="chat_completions",
        provider="anthropic",
        request={"model": "claude", "messages": []},
        credentials=None,
        api_base=None,
        extra_headers=None,
        timeout=None,
    )

    assert result is None
    assert opener.calls == 0


def test_declined_open_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _NativeErrors())
    streaming.set_rust_streaming(
        capability=lambda api, provider, transport: True,
        chat=_FailingOpen(_Declined("unsupported request")),
    )

    result: Final = streaming.open_stream(
        api="chat_completions",
        provider="anthropic",
        request={"model": "claude", "messages": []},
        credentials=None,
        api_base=None,
        extra_headers=None,
        timeout=None,
    )

    assert result is None


def test_upstream_open_failure_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _NativeErrors())
    streaming.set_rust_streaming(
        capability=lambda api, provider, transport: True,
        chat=_FailingOpen(_Upstream(503, "connection closed after request")),
    )

    with pytest.raises(APIError, match="connection closed after request") as caught:
        streaming.open_stream(
            api="chat_completions",
            provider="anthropic",
            request={"model": "claude", "messages": []},
            credentials=None,
            api_base=None,
            extra_headers=None,
            timeout=None,
        )
    assert caught.value.headers["x-litellm-core"] == "rust"


def test_midstream_failure_preserves_rust_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _NativeErrors())
    events: Final = streaming.TypedEventStreamAdapter(_FailingEventStream(()), _context())

    with pytest.raises(APIError, match="stream interrupted") as caught:
        next(events)
    assert caught.value.headers["x-litellm-core"] == "rust"


def test_sync_typed_events_preserve_shape_metadata_and_close() -> None:
    opener: Final = _RecordingOpen((_chat_event("one"), _chat_event("two")))
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True, chat=opener)
    result: Final = streaming.open_stream(
        api="chat_completions",
        provider="anthropic",
        request={"model": "claude", "messages": []},
        credentials=None,
        api_base=None,
        extra_headers=None,
        timeout=1.0,
    )

    assert result is not None
    assert tuple(event["text"] for event in result) == ("one", "two")
    assert result.metadata["provider"] == "anthropic"
    result.close()
    assert opener.calls == 1


def test_chat_events_flow_through_custom_stream_wrapper() -> None:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.types.utils import ModelResponseStream

    native: Final = _FakeEventStream((_chat_event("hello"),))
    events: Final = streaming.TypedEventStreamAdapter(native, _context())
    wrapper: Final = CustomStreamWrapper(
        completion_stream=events,
        model="claude",
        logging_obj=MagicMock(),
        custom_llm_provider="anthropic",
    )

    chunk: Final = next(wrapper)
    assert isinstance(chunk, ModelResponseStream)
    assert chunk.choices[0].delta.content == "hello"


@pytest.mark.asyncio
async def test_async_typed_events_and_cancellation() -> None:
    opener: Final = _RecordingAsyncOpen((_chat_event("one"), _chat_event("two")))
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True, achat=opener)
    result: Final = await streaming.aopen_stream(
        api="chat_completions",
        provider="anthropic",
        request={"model": "claude", "messages": []},
        credentials=None,
        api_base=None,
        extra_headers=None,
        timeout=None,
    )

    assert result is not None
    assert tuple(event["text"] for event in [event async for event in result]) == ("one", "two")
    await result.aclose()


def test_messages_events_are_wrapped_in_existing_sse_bytes() -> None:
    events: Final = streaming.TypedEventStreamAdapter(_FakeEventStream(({"type": "message_stop"},)), _context())
    messages: Final = streaming.MessagesSseStreamAdapter(events)

    assert tuple(messages) == (b'data: {"type":"message_stop"}\n\n',)


def test_responses_events_are_validated_into_existing_sdk_objects() -> None:
    from litellm.types.llms.openai import OutputTextDeltaEvent

    native: Final = _FakeEventStream(
        (
            {
                "type": "response.output_text.delta",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
        )
    )
    responses: Final = streaming.ResponsesSdkEventStreamAdapter(streaming.TypedEventStreamAdapter(native, _context()))

    event: Final = next(iter(responses))
    assert isinstance(event, OutputTextDeltaEvent)
    assert event.delta == "hello"


@pytest.mark.asyncio
async def test_rejects_mixed_sync_and_async_consumption() -> None:
    events: Final = streaming.TypedEventStreamAdapter(_FakeEventStream((_chat_event("one"),)), _context())
    assert tuple(events) == (_chat_event("one"),)

    with pytest.raises(RuntimeError, match="cannot mix"):
        async for _ in events:
            pass
