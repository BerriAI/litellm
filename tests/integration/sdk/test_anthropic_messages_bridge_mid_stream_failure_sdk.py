import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterable, Callable
from typing import Final, Literal

import pytest
from integration._support.anthropic_sse import (
    Attempts,
    SseEvent,
    delta_text,
    dropping_reply,
    event_types,
    message_id,
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

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        with self._lock:
            self._failures.append(kwargs.get("exception"))

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


class _AgenticHookProbe(_CallbackProbe):
    async def async_should_run_agentic_loop(
        self,
        response: object,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        return False, {}


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


@pytest.mark.parametrize("stream_wrapper", ["none", "cache-miss", "agentic-hook"])
async def test_sdk_router_stream_retried_before_content_logs_the_message_id_the_client_saw(
    stream_wrapper: Literal["none", "cache-miss", "agentic-hook"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cache", litellm.Cache() if stream_wrapper == "cache-miss" else litellm.cache)
    probe: Final = _AgenticHookProbe() if stream_wrapper == "agentic-hook" else _CallbackProbe()
    litellm.callbacks.append(probe)
    prompt: Final = uuid.uuid4().hex
    model: Final = "claude-sonnet-5-5"
    attempts: Final = Attempts()

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages")
        attempt: Final = attempts.record(user_prompt(object_value(json.loads(request.body))))
        frames: Final = message_stream(f"msg_attempt_{attempt}", model, _TEXT)
        return dropping_reply(frames, abort_after=1) if attempt == 1 else stream_reply(frames)

    with wire_server(respond) as wire:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "claude",
                    "litellm_params": {"model": f"anthropic/{model}", "api_base": wire.url, "api_key": _PROVIDER_KEY},
                }
            ],
            num_retries=1,
        )
        response: Final = await router.aanthropic_messages(
            model="claude",
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": prompt}],
            caching=stream_wrapper == "cache-miss",
        )
        assert isinstance(response, AsyncIterable), response
        frames: Final = [frame async for frame in response]
        events: Final = parse_sse(b"".join(frame for frame in frames if isinstance(frame, bytes)).decode())
        assert delta_text(events) == _TEXT, events
        assert event_types(events).count("message_start") == 1, events
        assert attempts.count(prompt) == 2
        await _settled(probe.successes)
        await GLOBAL_LOGGING_WORKER.flush()
    successes: Final = probe.successes()
    assert len(successes) == 1, successes
    logged: Final = TypeAdapter(dict[str, object]).validate_python(
        TypeAdapter(tuple[object, object]).validate_python(successes[0])[1]
    )
    assert logged["id"] == message_id(events) == "msg_attempt_1"
