import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterable, Callable
from typing import Final

from integration._support.anthropic_sse import (
    SseEvent,
    delta_text,
    dropping_reply,
    event_types,
    parse_sse,
    stream_reply,
    user_prompt,
)
from integration._support.client import eventually, object_value
from integration._support.openai_wire import answering_model_discovery, chat_stream, posted_targets
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger

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
            self._successes.append(kwargs.get("litellm_call_id"))

    def failures(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._failures)

    def successes(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._successes)


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


async def _stream(wire: Wire, prompt: str) -> tuple[SseEvent, ...]:
    response: Final = await litellm.anthropic.messages.acreate(
        model=f"hosted_vllm/{_BACKEND}",
        api_base=wire.url + "/v1",
        api_key=_PROVIDER_KEY,
        max_tokens=16,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
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
