import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterable, Callable
from itertools import chain
from typing import Final

import httpx
import pytest
from integration._support.anthropic_sse import (
    SseEvent,
    delta_text,
    dropping_reply,
    event_types,
    message_stream,
    parse_sse,
    stream_reply,
    user_prompt,
)
from integration._support.client import eventually, object_value
from integration._support.openai_wire import answering_model_discovery, chat_stream, posted_targets
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

_BACKEND: Final = "gpt-4o-mini"
_PROVIDER_KEY: Final = "integration-provider-key"
_UPSTREAM_TARGET: Final = "/v1/chat/completions"
_TEXT: Final = "Hello"
_DROPS_AFTER_CONTENT: Final = "drops-after-content"
_SUCCEEDS: Final = "succeeds"


class _CallbackProbe(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._lock: Final = threading.Lock()
        self._failures: Final[list[object]] = []
        self._successes: Final[list[object]] = []
        self._failure_logs: Final[list[object]] = []

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        with self._lock:
            self._failures.append(kwargs.get("exception"))
            self._failure_logs.append(kwargs.get("standard_logging_object"))

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        with self._lock:
            self._successes.append((kwargs.get("litellm_call_id"), kwargs.get("standard_logging_object")))

    def failures(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._failures)

    def successes(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._successes)

    def failure_logs(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._failure_logs)


def _upstream(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _UPSTREAM_TARGET), request
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}", request.headers
        body: Final = object_value(json.loads(request.body))
        assert body["model"] == _BACKEND and body["stream"] is True, body
        outcome, _, scripted_marker = user_prompt(body).partition(":")
        assert scripted_marker == marker, body
        chunks: Final = chat_stream(f"chunk-{outcome}-{marker}", _BACKEND, _TEXT)
        if outcome == _DROPS_AFTER_CONTENT:
            return dropping_reply(chunks, abort_after=2)
        return stream_reply(chunks)

    return answering_model_discovery(respond)


async def _stream(wire: Wire, prompt: str, model: str = f"hosted_vllm/{_BACKEND}") -> tuple[SseEvent, ...]:
    response: Final = await litellm.anthropic.messages.acreate(
        model=model,
        api_base=wire.url if model.startswith("anthropic/") else wire.url + "/v1",
        api_key=_PROVIDER_KEY,
        max_tokens=16,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
        litellm_call_id=prompt,
    )
    assert isinstance(response, AsyncIterable), response
    frames: Final = [frame async for frame in response]
    assert all(isinstance(frame, bytes) for frame in frames), frames
    return parse_sse(b"".join(frame for frame in frames if isinstance(frame, bytes)).decode())


async def _settled(read: Callable[[], tuple[object, ...]]) -> tuple[object, ...]:
    return await asyncio.to_thread(eventually, read, lambda seen: len(seen) >= 1)


async def test_sdk_bridged_stream_failing_after_content_reports_the_provider_error_once() -> None:
    probe: Final = _CallbackProbe()
    litellm.callbacks.append(probe)
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream(marker)) as wire:
        failing: Final = await _stream(wire, f"{_DROPS_AFTER_CONTENT}:{marker}")
        assert delta_text(failing) == _TEXT, failing
        assert event_types(failing)[-1] == "error" and "message_stop" not in event_types(failing), failing
        await _settled(probe.failures)
        succeeding: Final = await _stream(wire, f"{_SUCCEEDS}:{marker}")
        assert event_types(succeeding)[-1] == "message_stop", succeeding
        assert delta_text(succeeding) == _TEXT, succeeding
        await _settled(probe.successes)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,) * 2
    failures: Final = probe.failures()
    assert len(failures) == 1, failures
    assert isinstance(failures[0], Exception), failures
    assert not isinstance(failures[0], MidStreamFallbackError), failures
    assert len(probe.successes()) == 1, probe.successes()


async def test_sdk_native_messages_stream_delivers_one_success_callback_with_usage() -> None:
    probe: Final = _CallbackProbe()
    litellm.callbacks.append(probe)
    prompt: Final = uuid.uuid4().hex
    model: Final = "claude-sonnet-5-5"

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages")
        assert request.headers["x-api-key"] == _PROVIDER_KEY
        assert json.loads(request.body) == {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 16,
        }
        return stream_reply(message_stream("msg_success", model, _TEXT))

    with wire_server(respond) as wire:
        events: Final = await _stream(wire, prompt, f"anthropic/{model}")
        assert delta_text(events) == _TEXT
        assert event_types(events) == (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        )
        await _settled(probe.successes)
        await GLOBAL_LOGGING_WORKER.flush()
        successes: Final = probe.successes()
        assert len(successes) == 1, successes
        success: Final = TypeAdapter(tuple[object, object]).validate_python(successes[0])
        assert success[0] == prompt
        logging: Final = TypeAdapter(dict[str, object]).validate_python(success[1])
        assert (logging["prompt_tokens"], logging["completion_tokens"], logging["total_tokens"]) == (5, 3, 8)
        assert probe.failures() == ()
        assert posted_targets(wire) == ("/v1/messages",)


@pytest.mark.parametrize("ending", ("complete", "eof", "drop", "cancel"))
async def test_native_http_stream_partial_usage_has_one_failed_result(
    ending: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    monkeypatch.setenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", "all")
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    probe: Final = _CallbackProbe()
    litellm.callbacks.append(probe)
    call_id: Final = str(uuid.uuid4())
    model: Final = "claude-sonnet-5-5"
    frames: Final = message_stream("msg_synthetic_partial", model, _TEXT)
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages")
        assert request.headers["x-api-key"] == _PROVIDER_KEY
        assert json.loads(request.body) == {
            "model": model,
            "messages": [{"role": "user", "content": "synthetic prompt"}],
            "stream": True,
            "max_tokens": 16,
        }
        if ending == "complete":
            return stream_reply(tuple(bytes((byte,)) for byte in b"".join(frames)))
        if ending == "cancel":
            return Reply(content_type="text/event-stream", chunks=(frames[0], frames[-1]), gate_after_first=gate)
        if ending == "drop":
            return dropping_reply((frames[0], frames[-1]), abort_after=1)
        return stream_reply((frames[0][:13], frames[0][13:]))

    with wire_server(respond) as wire:
        stream: Final = await litellm.anthropic.messages.acreate(
            model=f"anthropic/{model}",
            api_base=wire.url,
            api_key=_PROVIDER_KEY,
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": "synthetic prompt"}],
            litellm_call_id=call_id,
        )
        from litellm.llms.anthropic.pass_through.messages.streaming_iterator import (
            AnthropicMessagesStreamingResponse,
        )

        assert isinstance(stream, AnthropicMessagesStreamingResponse)
        try:
            if ending == "drop":
                with pytest.raises(
                    httpx.TransportError,
                    match=r"(incomplete|closed|payload|transfer)",
                ):
                    async for _ in stream:
                        pass
            elif ending == "cancel":
                assert b"message_start" in await stream.__anext__()
                await stream.aclose()
            else:
                output: Final = b"".join([chunk async for chunk in stream])
                assert (b"message_stop" in output) is (ending == "complete")
        finally:
            await stream.aclose()
            gate.set()
        await _settled(probe.successes if ending == "complete" else probe.failures)
        await GLOBAL_LOGGING_WORKER.flush()
        assert len(probe.successes()) + len(probe.failures()) == 1
        if ending == "complete":
            assert probe.failures() == ()
        else:
            assert probe.successes() == ()
            log: Final = TypeAdapter(dict[str, object]).validate_python(probe.failure_logs()[0])
            assert log["status"] == "failure" and log["prompt_tokens"] == 5
        summaries: Final = tuple(
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("anthropic_stream_summary ")
        )
        assert len(summaries) == 1
        assert call_id in summaries[0] and "msg_synthetic_partial" in summaries[0]
        assert _PROVIDER_KEY not in summaries[0] and "synthetic prompt" not in summaries[0]
        assert posted_targets(wire) == ("/v1/messages",)


@pytest.mark.parametrize("transport", ("text", "bytes"))
@pytest.mark.parametrize("newline", (b"\n", b"\r\n", b"\r"), ids=("lf", "crlf", "cr"))
@pytest.mark.parametrize("split_at", ("header", "json", "delimiter"))
async def test_native_fragmented_http_frames_are_not_corrupted_by_proxy_keepalives(
    transport: str, newline: bytes, split_at: str
) -> None:
    from collections.abc import AsyncGenerator

    from anthropic._streaming import SSEDecoder

    from litellm.proxy.common_utils.sse_keepalive import (
        ANTHROPIC_PING_SSE_CHUNK,
        SSE_COMMENT_PING_BYTES,
        wrap_passthrough_sse_bytes_with_keepalive_pings,
        wrap_sse_stream_with_keepalive_pings,
    )

    model: Final = "claude-sonnet-5-5"
    frames: Final = tuple(
        frame.replace(b"\n", newline) for frame in message_stream("msg_synthetic_keepalive", model, _TEXT)
    )

    def fragments(frame: bytes) -> tuple[bytes, ...]:
        boundary: Final = {
            "header": 13,
            "json": frame.index(b"data:") + len(b"data:") + 12,
            "delimiter": len(frame) - 1,
        }[split_at]
        return frame[:boundary], frame[boundary:]

    chunks: Final = (frames[0], *chain.from_iterable(fragments(frame) for frame in frames[1:]))
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages")
        assert json.loads(request.body) == {
            "model": model,
            "messages": [{"role": "user", "content": "synthetic"}],
            "stream": True,
            "max_tokens": 16,
        }
        return Reply(content_type="text/event-stream", chunks=chunks, gate_after_first=gate, pause_between_chunks=0.03)

    with wire_server(respond) as wire:
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"anthropic/{model}",
            api_base=wire.url,
            api_key=_PROVIDER_KEY,
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": "synthetic"}],
        )
        assert isinstance(response, AsyncIterable)

        async def raw() -> AsyncGenerator[bytes, None]:
            async for chunk in response:
                assert isinstance(chunk, bytes)
                yield chunk
                gate.set()

        async def serialized() -> AsyncGenerator[str, None]:
            async for chunk in raw():
                yield chunk.decode()

        received: Final[bytes]
        if transport == "bytes":
            byte_stream: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
                raw(), 0.005, {"content-type": "text/event-stream"}
            )
            received = b"".join([chunk async for chunk in byte_stream])
        else:
            text_stream: Final = wrap_sse_stream_with_keepalive_pings(serialized(), 0.005)
            received = "".join([chunk async for chunk in text_stream]).encode()

        ping: Final = SSE_COMMENT_PING_BYTES if transport == "bytes" else ANTHROPIC_PING_SSE_CHUNK.encode()
        assert ping in received
        events: Final = tuple(SSEDecoder().iter_bytes(iter((received,))))
        for event in events:
            assert TypeAdapter(dict[str, object]).validate_json(event.data)["type"] == event.event
        expected: Final = tuple((event.event, event.data) for event in SSEDecoder().iter_bytes(iter(frames)))
        assert tuple((event.event, event.data) for event in events if event.event != "ping") == expected
        assert received.replace(ping, b"") == b"".join(frames)


@pytest.mark.parametrize("mode", ("all", "errors", "off"))
async def test_routed_stream_diagnostics_correlate_provider_and_router_without_content(
    mode: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    monkeypatch.setenv("LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS", mode)
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    model: Final = "claude-sonnet-5-5"
    call_id: Final = str(uuid.uuid4())
    frames: Final = message_stream("msg_synthetic_routed", model, "SENSITIVE_RESPONSE")

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        return stream_reply(tuple(piece for frame in frames for piece in (frame[:13], frame[13:])))

    with wire_server(respond) as wire:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "diagnostic-test",
                    "litellm_params": {"model": f"anthropic/{model}", "api_base": wire.url, "api_key": _PROVIDER_KEY},
                }
            ],
            num_retries=1,
        )
        response: Final = await router.aanthropic_messages(
            model="diagnostic-test",
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": "SENSITIVE_PROMPT"}],
            litellm_call_id=call_id,
        )
        assert isinstance(response, AsyncIterable)
        received: Final = b"".join([chunk async for chunk in response])
        assert received == b"".join(frames)
        await GLOBAL_LOGGING_WORKER.flush()
        summaries: Final = tuple(
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("anthropic_stream_summary ")
        )
        assert len(summaries) == (2 if mode == "all" else 0)
        for summary in summaries:
            value: Final = TypeAdapter(dict[str, object]).validate_json(summary.split(" ", 1)[1])
            assert value["call_id"] == call_id
            assert value["text_deltas"] == 1 and value["complete"] is True
            assert value["message_id"] == "msg_synthetic_routed"
            assert "SENSITIVE_PROMPT" not in summary and "SENSITIVE_RESPONSE" not in summary
            assert _PROVIDER_KEY not in summary
