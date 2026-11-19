from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from functools import reduce
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from integration._support import codex_vendor as cv
from integration._support.client import eventually
from integration._support.raw_response_probe import (
    CHAT_BODY,
    CHAT_ID,
    CHAT_MODEL,
    EMBEDDING_BODY,
    EMBEDDING_MODEL,
    IMAGE_B64,
    IMAGE_BODY,
    IMAGE_MODEL,
    encoded,
    fingerprint,
    raw_response_fields,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, JsonValue

import litellm
from litellm import _logging as litellm_logging_settings
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

pytestmark: Final = pytest.mark.timeout(180)

_KEY: Final = "synthetic-provider-key"
_MESSAGES: Final = ({"role": "user", "content": "raw response retention"},)
_FAILURE: Final = "raw-response-scripted-failure"
_RAW_RESPONSE_LINE: Final = "RAW RESPONSE"
_RECOVERED: Final = "text recovered from the raw stream"
_CALLBACK_LISTS: Final = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "input_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "_async_input_callback",
)


@dataclass(frozen=True, slots=True)
class _Call:
    name: str
    sync: bool
    call_type: str
    function: Callable[..., object]
    model: str
    arguments: Mapping[str, object]
    route: str
    body: Mapping[str, JsonValue]
    content_path: tuple[str | int, ...]


def _image(name: str, sync: bool, call_type: str, function: Callable[..., object]) -> _Call:
    return _Call(
        name,
        sync,
        call_type,
        function,
        IMAGE_MODEL,
        {"prompt": "raw response retention"},
        "/v1/images/generations",
        IMAGE_BODY,
        ("data", 0, "b64_json"),
    )


def _chat(name: str, sync: bool, call_type: str, function: Callable[..., object]) -> _Call:
    return _Call(
        name,
        sync,
        call_type,
        function,
        CHAT_MODEL,
        {"messages": [dict(message) for message in _MESSAGES]},
        "/v1/chat/completions",
        CHAT_BODY,
        ("choices", 0, "message", "content"),
    )


def _embedding(name: str, sync: bool, call_type: str, function: Callable[..., object]) -> _Call:
    return _Call(
        name,
        sync,
        call_type,
        function,
        EMBEDDING_MODEL,
        {"input": ["raw response retention"]},
        "/v1/embeddings",
        EMBEDDING_BODY,
        ("data", 0, "embedding"),
    )


IMAGE_SYNC: Final = _image("image-sync", True, "image_generation", litellm.image_generation)
IMAGE_ASYNC: Final = _image("image-async", False, "aimage_generation", litellm.aimage_generation)
CHAT_SYNC: Final = _chat("chat-sync", True, "completion", litellm.completion)
CHAT_ASYNC: Final = _chat("chat-async", False, "acompletion", litellm.acompletion)
EMBEDDING_SYNC: Final = _embedding("embedding-sync", True, "embedding", litellm.embedding)
EMBEDDING_ASYNC: Final = _embedding("embedding-async", False, "aembedding", litellm.aembedding)
ALL_CALLS: Final = (IMAGE_SYNC, IMAGE_ASYNC, CHAT_SYNC, CHAT_ASYNC, EMBEDDING_SYNC, EMBEDDING_ASYNC)
SERIALIZED_CALLS: Final = (IMAGE_SYNC, IMAGE_ASYNC, CHAT_SYNC, CHAT_ASYNC, EMBEDDING_ASYNC)
LOGGER_FN_CALLS: Final = (IMAGE_SYNC, IMAGE_ASYNC, CHAT_SYNC, CHAT_ASYNC)


def _call_id(call: _Call) -> str:
    return call.name


def _json_reply(body: Mapping[str, JsonValue], status: int = 200) -> Callable[[Request], Reply]:
    payload: Final = encoded(body)

    def respond(_: Request) -> Reply:
        return Reply(status=status, body=payload)

    return respond


def _failure_reply(_: Request) -> Reply:
    return Reply(status=400, body=encoded({"error": {"message": _FAILURE, "type": "invalid_request_error"}}))


async def _run(call: _Call, api_base: str, extra: Mapping[str, object] = MappingProxyType({})) -> object:
    arguments: Final[Mapping[str, object]] = {
        "model": call.model,
        "api_base": api_base,
        "api_key": _KEY,
        "max_retries": 0,
        **call.arguments,
        **extra,
    }
    if call.sync:
        return await asyncio.to_thread(call.function, **arguments)
    pending: Final = call.function(**arguments)
    assert inspect.isawaitable(pending)
    return await pending


def _base(wire: Wire) -> str:
    return f"{wire.url}/v1"


def _request_logging(call: _Call, *, stream: bool = False) -> Logging:
    return Logging(
        model=call.model,
        messages=[],
        stream=stream,
        call_type=call.call_type,
        start_time=datetime.now(),  # noqa: DTZ005  # litellm's own naive start time
        litellm_call_id=f"raw-response-{call.name}",
        function_id="raw-response-retention",
    )


def _step(current: JsonValue, key: str | int) -> JsonValue:
    if isinstance(current, list) and isinstance(key, int):
        return current[key]
    if isinstance(current, dict) and isinstance(key, str):
        return current[key]
    raise AssertionError(f"no {key!r} in a {type(current).__name__}")


def _served_content(response: object, path: tuple[str | int, ...]) -> JsonValue:
    assert isinstance(response, BaseModel), type(response)
    dumped: Final[JsonValue] = json.loads(json.dumps(response.model_dump(), default=str))
    return reduce(_step, path, dumped)


def _upstream_content(call: _Call) -> JsonValue:
    return reduce(_step, call.content_path, dict(call.body))


def _held(logging_obj: Logging) -> object:
    assert "original_response" in logging_obj.model_call_details
    return logging_obj.model_call_details["original_response"]


def _assert_only_a_short_string_is_held(logging_obj: Logging, call: _Call) -> None:
    held: Final = _held(logging_obj)
    size: Final = len(encoded(call.body))
    assert isinstance(held, str), f"original_response is a {type(held).__name__}"
    assert len(held) < size / 100, f"{len(held)} characters held after a {size} byte response"


def _assert_raw_matches(raw: object, body: Mapping[str, JsonValue]) -> None:
    assert fingerprint(raw_response_fields(raw)) == fingerprint(body), "raw response differs from the upstream body"


@dataclass(frozen=True, slots=True)
class _Seen:
    hook: str
    raw: object
    exception: str


class _Recorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._lock: Final = threading.Lock()
        self.events: tuple[_Seen, ...] = ()

    def _record(self, hook: str, kwargs: Mapping[str, object]) -> None:
        seen: Final = _Seen(hook, kwargs.get("original_response"), str(kwargs.get("exception") or ""))
        with self._lock:
            self.events = (*self.events, seen)

    def of(self, hook: str) -> tuple[_Seen, ...]:
        return tuple(event for event in self.events if event.hook == hook)

    def log_post_api_call(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("post", kwargs)

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("success", kwargs)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("failure", kwargs)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("success", kwargs)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record("failure", kwargs)


class _FunctionRecorder:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self.raw: tuple[object, ...] = ()

    def _record(self, kwargs: Mapping[str, object]) -> None:
        with self._lock:
            self.raw = (*self.raw, kwargs.get("original_response"))

    def sync_callback(self, kwargs: Mapping[str, object], response: object, start: object, end: object) -> None:
        self._record(kwargs)

    async def async_callback(self, kwargs: Mapping[str, object], response: object, start: object, end: object) -> None:
        self._record(kwargs)


class _LoggerFn:
    def __init__(self) -> None:
        self.raw: tuple[object, ...] = ()

    def __call__(self, details: Mapping[str, object]) -> None:
        if details.get("log_event_type") == "post_api_call":
            self.raw = (*self.raw, details.get("original_response"))


class _LoggedLines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: tuple[str, ...] = ()

    def emit(self, record: logging.LogRecord) -> None:
        message: Final = record.getMessage()
        if _RAW_RESPONSE_LINE in message:
            self.lines = (*self.lines, message)


def _clear_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])


@pytest.fixture
def nothing_configured(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _clear_callbacks(monkeypatch)
    monkeypatch.setattr(litellm, "set_verbose", False)
    monkeypatch.setattr(litellm_logging_settings, "set_verbose", False)
    caplog.set_level(logging.WARNING, logger=verbose_logger.name)


@pytest.fixture
def logged_lines() -> Iterator[_LoggedLines]:
    handler: Final = _LoggedLines()
    verbose_logger.addHandler(handler)
    try:
        yield handler
    finally:
        verbose_logger.removeHandler(handler)


@pytest.fixture(scope="module")
def chatgpt_login(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cv.login(tmp_path_factory.mktemp("raw-response-chatgpt"))


async def _settled() -> None:
    await GLOBAL_LOGGING_WORKER.flush()


async def _await_count(read: Callable[[], int], minimum: int = 1) -> None:
    await asyncio.to_thread(eventually, read, lambda count: count >= minimum, 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", SERIALIZED_CALLS, ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_raw_response_is_not_held_when_nothing_reads_it(call: _Call) -> None:
    logging_obj: Final = _request_logging(call)
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire), {"litellm_logging_obj": logging_obj})
    await _settled()
    _assert_only_a_short_string_is_held(logging_obj, call)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", (IMAGE_SYNC, CHAT_ASYNC), ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_raw_response_is_not_held_when_logger_fn_is_explicitly_none(call: _Call) -> None:
    logging_obj: Final = _request_logging(call)
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire), {"litellm_logging_obj": logging_obj, "logger_fn": None})
    await _settled()
    _assert_only_a_short_string_is_held(logging_obj, call)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", (IMAGE_ASYNC, CHAT_SYNC), ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_raw_response_is_not_held_once_debug_logging_is_turned_off(
    call: _Call, caplog: pytest.LogCaptureFixture
) -> None:
    logging_obj: Final = _request_logging(call)
    with wire_server(_json_reply(call.body)) as wire:
        caplog.set_level(logging.DEBUG, logger=verbose_logger.name)
        await _run(call, _base(wire))
        caplog.set_level(logging.WARNING, logger=verbose_logger.name)
        await _run(call, _base(wire), {"litellm_logging_obj": logging_obj})
    await _settled()
    _assert_only_a_short_string_is_held(logging_obj, call)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", (IMAGE_SYNC, CHAT_ASYNC), ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_raw_response_is_not_held_once_the_custom_callback_is_removed(
    call: _Call, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = _Recorder()
    logging_obj: Final = _request_logging(call)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire))
        await _settled()
        _assert_raw_matches(recorder.of("post")[0].raw, call.body)
        _clear_callbacks(monkeypatch)
        await _run(call, _base(wire), {"litellm_logging_obj": logging_obj})
    await _settled()
    _assert_only_a_short_string_is_held(logging_obj, call)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ALL_CALLS, ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_caller_receives_the_complete_response(call: _Call) -> None:
    with wire_server(_json_reply(call.body)) as wire:
        response: Final = await _run(call, _base(wire))
        requests: Final = wire.drain()
    assert [(request.method, request.target) for request in requests] == [("POST", call.route)]
    served: Final = _served_content(response, call.content_path)
    assert fingerprint(served) == fingerprint(_upstream_content(call)), "the caller's response content changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ALL_CALLS, ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_custom_callback_receives_the_full_raw_response(call: _Call, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with wire_server(_json_reply(call.body)) as wire:
        response: Final = await _run(call, _base(wire))
    await _settled()
    await _await_count(lambda: len(recorder.of("success")))
    events: Final = (*recorder.of("post"), *recorder.of("success"))
    assert len(recorder.of("post")) == 1, [event.hook for event in recorder.events]
    for event in events:
        _assert_raw_matches(event.raw, call.body)
    assert fingerprint(_served_content(response, call.content_path)) == fingerprint(_upstream_content(call))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "scope"),
    (
        *((call, "global") for call in ALL_CALLS),
        (IMAGE_ASYNC, "request"),
        (CHAT_SYNC, "request"),
        (CHAT_ASYNC, "request"),
        (EMBEDDING_SYNC, "request"),
        (EMBEDDING_ASYNC, "request"),
    ),
    ids=lambda value: value.name if isinstance(value, _Call) else value,
)
@pytest.mark.usefixtures("nothing_configured")
async def test_callback_function_receives_the_full_raw_response(
    call: _Call, scope: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = _FunctionRecorder()
    callback: Final = recorder.sync_callback if call.sync else recorder.async_callback
    extra: Final[Mapping[str, object]] = {"success_callback": [callback]} if scope == "request" else {}
    monkeypatch.setattr(litellm, "success_callback", [callback] if scope == "global" else [])
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire), extra)
    await _settled()
    await _await_count(lambda: len(recorder.raw))
    for raw in recorder.raw:
        _assert_raw_matches(raw, call.body)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", LOGGER_FN_CALLS, ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_logger_fn_receives_the_full_raw_response(call: _Call) -> None:
    logger_fn: Final = _LoggerFn()
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire), {"logger_fn": logger_fn})
    assert len(logger_fn.raw) == 1, "logger_fn did not see the provider response exactly once"
    _assert_raw_matches(logger_fn.raw[0], call.body)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", (IMAGE_ASYNC, CHAT_SYNC), ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_raw_response_is_not_logged_when_debug_logging_is_off(call: _Call, logged_lines: _LoggedLines) -> None:
    with wire_server(_json_reply(call.body)) as wire:
        response: Final = await _run(call, _base(wire))
    assert fingerprint(_served_content(response, call.content_path)) == fingerprint(_upstream_content(call))
    assert logged_lines.lines == (), "the raw response reached the log with debug logging off"


@pytest.mark.asyncio
@pytest.mark.parametrize("call", (IMAGE_SYNC, CHAT_ASYNC), ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_upstream_error_reaches_failure_callbacks(call: _Call, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with wire_server(_failure_reply) as wire, pytest.raises(litellm.BadRequestError, match=_FAILURE):
        await _run(call, _base(wire))
    await _settled()
    await _await_count(lambda: len(recorder.of("failure")))
    failure: Final = recorder.of("failure")[0]
    assert _FAILURE in failure.exception, failure.exception


def _tracing_logger() -> tuple[OpenTelemetry, InMemorySpanExporter]:
    exporter: Final = InMemorySpanExporter()
    provider: Final = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetry(tracer_provider=provider), exporter


def _named_spans(exporter: InMemorySpanExporter, name: str) -> tuple[ReadableSpan, ...]:
    return tuple(span for span in exporter.get_finished_spans() if span.name == name)


def _provider_fields(span: ReadableSpan) -> Mapping[str, str]:
    attributes: Final = dict(span.attributes or {})
    return {key.rsplit(".", 1)[1]: str(value) for key, value in attributes.items() if key.startswith("llm.")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "field", "expected"),
    (
        (CHAT_SYNC, "id", CHAT_ID),
        (CHAT_ASYNC, "id", CHAT_ID),
        (IMAGE_SYNC, "data", IMAGE_B64),
        (IMAGE_ASYNC, "data", IMAGE_B64),
        (EMBEDDING_ASYNC, "object", "list"),
    ),
    ids=("chat-sync", "chat-async", "image-sync", "image-async", "embedding-async"),
)
@pytest.mark.usefixtures("nothing_configured")
async def test_opentelemetry_raw_span_carries_the_provider_response_fields(
    call: _Call, field: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    otel, exporter = _tracing_logger()
    monkeypatch.setattr(litellm, "callbacks", [otel])
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire))
    await _settled()
    await _await_count(lambda: len(_named_spans(exporter, "raw_gen_ai_request")))
    fields: Final = _provider_fields(_named_spans(exporter, "raw_gen_ai_request")[0])
    assert field in fields, sorted(fields)
    assert expected in fields[field], f"{field} does not carry the provider value"


@pytest.mark.asyncio
@pytest.mark.parametrize("call", (CHAT_SYNC, CHAT_ASYNC, IMAGE_ASYNC), ids=_call_id)
@pytest.mark.usefixtures("nothing_configured")
async def test_opentelemetry_skips_the_raw_span_when_message_logging_is_off(
    call: _Call, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    otel, exporter = _tracing_logger()
    runs_after_otel: Final = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [otel, runs_after_otel])
    with wire_server(_json_reply(call.body)) as wire:
        await _run(call, _base(wire))
    await _settled()
    await _await_count(lambda: len(runs_after_otel.of("success")))
    assert _named_spans(exporter, "litellm_request"), "the OpenTelemetry success handler did not run"
    assert _named_spans(exporter, "raw_gen_ai_request") == (), "message logging is off but the raw span was exported"


def _chatgpt_stream(*events: Mapping[str, JsonValue]) -> Callable[[Request], Reply]:
    payload: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    def respond(_: Request) -> Reply:
        return Reply(body=payload.encode(), content_type="text/event-stream")

    return respond


_COMPLETED_WITHOUT_OUTPUT: Final[Mapping[str, JsonValue]] = {
    "type": "response.completed",
    "response": {
        "id": "resp_raw_response",
        "object": "response",
        "created_at": 1_700_000_000,
        "status": "completed",
        "model": "gpt-6.2-sol",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
}
_OUTPUT_TEXT_DONE: Final[Mapping[str, JsonValue]] = {
    "type": "response.output_text.done",
    "output_index": 0,
    "content_index": 0,
    "item_id": "msg_raw_response",
    "text": _RECOVERED,
}
_CHATGPT_MODEL: Final = "chatgpt/gpt-6.2-sol"


async def _chatgpt_completion(sync: bool, logging_obj: Logging | None = None) -> object:
    extra: Final[Mapping[str, object]] = {} if logging_obj is None else {"litellm_logging_obj": logging_obj}
    messages: Final = [dict(message) for message in _MESSAGES]
    if sync:
        return await asyncio.to_thread(litellm.completion, model=_CHATGPT_MODEL, messages=messages, **extra)
    return await litellm.acompletion(model=_CHATGPT_MODEL, messages=messages, **extra)


def _use_chatgpt(monkeypatch: pytest.MonkeyPatch, login: Path, wire: Wire) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(login))
    monkeypatch.setenv("CHATGPT_API_BASE", wire.url)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (True, False), ids=("sync", "async"))
@pytest.mark.usefixtures("nothing_configured")
async def test_chatgpt_stream_with_empty_final_output_recovers_text_from_raw_events(
    sync: bool, chatgpt_login: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging_obj: Final = _request_logging(replace(CHAT_SYNC if sync else CHAT_ASYNC, model=_CHATGPT_MODEL))
    with wire_server(_chatgpt_stream(_OUTPUT_TEXT_DONE, _COMPLETED_WITHOUT_OUTPUT)) as wire:
        _use_chatgpt(monkeypatch, chatgpt_login, wire)
        response: Final = await _chatgpt_completion(sync, logging_obj)
    assert _served_content(response, ("choices", 0, "message", "content")) == _RECOVERED
    held: Final = _held(logging_obj)
    assert isinstance(held, str) and _RECOVERED in held, type(held)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (True, False), ids=("sync", "async"))
@pytest.mark.usefixtures("nothing_configured")
async def test_chatgpt_stream_with_no_output_anywhere_is_an_error(
    sync: bool, chatgpt_login: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with wire_server(_chatgpt_stream(_COMPLETED_WITHOUT_OUTPUT)) as wire:
        _use_chatgpt(monkeypatch, chatgpt_login, wire)
        with pytest.raises(litellm.APIConnectionError, match="Unknown items"):
            await _chatgpt_completion(sync)


@pytest.mark.asyncio
@pytest.mark.usefixtures("nothing_configured")
async def test_opentelemetry_raw_span_keeps_a_non_json_raw_response_as_text(
    chatgpt_login: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    otel, exporter = _tracing_logger()
    monkeypatch.setattr(litellm, "callbacks", [otel])
    with wire_server(_chatgpt_stream(_OUTPUT_TEXT_DONE, _COMPLETED_WITHOUT_OUTPUT)) as wire:
        _use_chatgpt(monkeypatch, chatgpt_login, wire)
        await _chatgpt_completion(False)
    await _settled()
    await _await_count(lambda: len(_named_spans(exporter, "raw_gen_ai_request")))
    fields: Final = _provider_fields(_named_spans(exporter, "raw_gen_ai_request")[0])
    assert _RECOVERED in fields["stringified_raw_response"], sorted(fields)
