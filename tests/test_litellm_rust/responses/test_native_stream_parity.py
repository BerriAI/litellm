"""Native Responses streams must be indistinguishable from the Python implementation.

Each case streams one public Responses call through Python and through the native route against the same recorded
OpenAI event stream, then compares what reached the provider, the events the caller read, and what the success
callback logged. The native leg fails instead of silently falling back to Python
"""

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

import litellm
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from tests.test_litellm_rust.responses.support import (
    PUBLIC_ARESPONSES,
    PUBLIC_RESPONSES,
    PYTHON_ARESPONSES,
    PYTHON_RESPONSES,
    FallbackProbe,
)
from tests.test_litellm_rust.responses.test_native_parity import (
    DEPLOYMENT_METADATA,
    FUNCTION_CALL_RESPONSE,
    MANTLE_MODEL,
    TEXT_MESSAGE,
    TEXT_RESPONSE,
    WEATHER_TOOL,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "openai/gpt-6.1-sol"
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LOGGED: Final = TypeAdapter(dict[str, object])
_COMPARED_LOG_FIELDS: Final = (
    "call_type",
    "stream",
    "custom_llm_provider",
    "model",
    "model_id",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "response_cost",
    "status",
    "api_base",
    "response",
)
_COMPARED_HIDDEN_PARAMS: Final = ("custom_llm_provider", "model_id", "api_base")
_COMPARED_RESPONSE_HEADERS: Final = (
    "llm_provider-content-type",
    "llm_provider-x-request-id",
    "llm_provider-openai-processing-ms",
    "request-id",
)
_SYNC_HANDLER_GRACE: Final = 0.2
_UPSTREAM_HEADERS: Final = MappingProxyType({"x-request-id": "req_stream", "openai-processing-ms": "12"})


def _opened(response: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**response, "status": "in_progress", "output": [], "usage": None}


def _events(response: Mapping[str, JsonValue]) -> tuple[tuple[str, object], ...]:
    """The event stream OpenAI sends for a response with one output item"""
    output: Final = response["output"]
    assert isinstance(output, list)
    item: Final = output[-1]
    assert isinstance(item, dict)
    opened: Final = (
        ("response.created", {"type": "response.created", "sequence_number": 0, "response": _opened(response)}),
        (
            "response.in_progress",
            {"type": "response.in_progress", "sequence_number": 1, "response": _opened(response)},
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 0,
                "item": {**item, "status": "in_progress"},
            },
        ),
    )
    body: Final = (
        (
            ("response.output_text.delta", _text_delta(item, 3, "Par")),
            ("response.output_text.delta", _text_delta(item, 4, "is")),
        )
        if item["type"] == "message"
        else (
            (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 3,
                    "item_id": item["id"],
                    "output_index": 0,
                    "delta": item["arguments"],
                },
            ),
        )
    )
    closed: Final = (
        (
            "response.output_item.done",
            {"type": "response.output_item.done", "sequence_number": 5, "output_index": 0, "item": item},
        ),
        ("response.completed", {"type": "response.completed", "sequence_number": 6, "response": dict(response)}),
    )
    return (*opened, *body, *closed)


def _text_delta(item: Mapping[str, JsonValue], sequence: int, delta: str) -> dict[str, JsonValue]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence,
        "item_id": item["id"],
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
        "logprobs": [],
    }


@dataclass(frozen=True, slots=True)
class Leg:
    request_path: str
    request_body: Mapping[str, JsonValue]
    events: tuple[Mapping[str, JsonValue], ...]
    logged: Mapping[str, object]
    hidden_params: Mapping[str, object]
    hooks: tuple[str, ...]
    fell_back: bool


ASPECTS: Final = ("dispatch", "request", "events", "logged", "hooks")


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    kwargs: Mapping[str, object]
    upstream: Mapping[str, JsonValue] = TEXT_RESPONSE
    synchronous: bool = False

    def params(self) -> Iterator[object]:
        for aspect in ASPECTS:
            yield pytest.param(self, aspect, id=f"{self.name}-{aspect}")


CASES: Final = (
    Case("plain_text", {"input": "What is the capital of France?"}),
    Case(
        "function_tool",
        {"input": "Weather in Paris?", "tools": [dict(WEATHER_TOOL)], "tool_choice": "auto"},
        upstream=FUNCTION_CALL_RESPONSE,
    ),
    Case(
        "deployment_ids_and_encrypted_content",
        {"input": "hi", "litellm_metadata": dict(DEPLOYMENT_METADATA)},
        upstream={**TEXT_RESPONSE, "output": [{**TEXT_MESSAGE, "encrypted_content": "enc"}]},
    ),
    Case("bedrock_mantle", {"model": MANTLE_MODEL, "input": "hi"}),
    Case("sync_plain_text", {"input": "What is the capital of France?"}, synchronous=True),
)


def _event(value: object) -> Mapping[str, JsonValue]:
    assert isinstance(value, BaseModel)
    return _OBJECT.validate_python(value.model_dump(mode="json", exclude_none=True))


def _hidden_params(stream: object) -> Mapping[str, object]:
    hidden: Final = _LOGGED.validate_python(getattr(stream, "_hidden_params"))
    headers: Final = _LOGGED.validate_python(hidden["additional_headers"])
    return MappingProxyType(
        {
            **{name: hidden.get(name) for name in _COMPARED_HIDDEN_PARAMS},
            **{name: headers.get(name) for name in _COMPARED_RESPONSE_HEADERS},
        }
    )


def _sync_read(native: bool, kwargs: Mapping[str, object]) -> tuple[object, tuple[object, ...]]:
    stream: Final = (PUBLIC_RESPONSES if native else PYTHON_RESPONSES)(**kwargs)
    assert isinstance(stream, SyncResponsesAPIStreamingIterator)
    return stream, tuple(stream)


async def _async_stream(native: bool, kwargs: Mapping[str, object]) -> ResponsesAPIStreamingIterator:
    stream: Final = await (PUBLIC_ARESPONSES if native else PYTHON_ARESPONSES)(**kwargs)
    assert isinstance(stream, ResponsesAPIStreamingIterator)
    return stream


async def _read(case: Case, kwargs: Mapping[str, object], native: bool) -> tuple[object, tuple[object, ...]]:
    if case.synchronous:
        return await asyncio.to_thread(_sync_read, native, kwargs)
    stream: Final = await _async_stream(native, kwargs)
    return stream, tuple([event async for event in stream])


async def _leg(case: Case, server: RecordingServer, probe: FallbackProbe | None) -> Leg:
    with isolated_callback_registries():
        recorder: Final = RecordingLogger()
        server.enqueue(ResponseSpec(body=None, events=_events(case.upstream), headers=dict(_UPSTREAM_HEADERS)))
        kwargs: Final = {
            "model": MODEL,
            "api_key": "sk-test",
            "api_base": server.base_url,
            "callbacks": [recorder],
            "stream": True,
            **case.kwargs,
        }
        fallbacks_before: Final = 0 if probe is None else probe.calls
        stream, events = await _read(case, kwargs, native=probe is not None)
        await drain_logging()
        success: Final = (
            await asyncio.to_thread(recorder.wait_for, "log_success_event")
            if case.synchronous
            else await recorder.wait_for_async("async_log_success_event")
        )
        sent: Final = server.requests[-1]
        standard: Final = _LOGGED.validate_python(_LOGGED.validate_python(success[0].kwargs)["standard_logging_object"])
        return Leg(
            request_path=sent.path,
            request_body=_OBJECT.validate_python(sent.body),
            events=tuple(map(_event, events)),
            logged=MappingProxyType({name: standard.get(name) for name in _COMPARED_LOG_FIELDS}),
            hidden_params=_hidden_params(stream),
            hooks=recorder.names,
            fell_back=probe is not None and probe.calls > fallbacks_before,
        )


@pytest.fixture
def parity_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.expected_requests = 2
    return recording_server


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "aspect"), tuple(param for case in CASES for param in case.params()))
async def test_native_responses_streams_match_python(
    case: Case, aspect: str, parity_server: RecordingServer, native_responses: FallbackProbe
) -> None:
    python: Final = await _leg(case, parity_server, probe=None)
    native: Final = await _leg(case, parity_server, probe=native_responses)
    match aspect:
        case "dispatch":
            assert not native.fell_back
        case "request":
            assert (native.request_path, native.request_body) == (python.request_path, python.request_body)
        case "events":
            assert native.events == python.events
            assert len(native.events) == len(_events(case.upstream))
        case "logged":
            assert (native.logged, native.hidden_params) == (python.logged, python.hidden_params)
        case _:
            assert native.hooks == python.hooks


async def _truncated_leg(server: RecordingServer, probe: FallbackProbe | None) -> tuple[str, str, tuple[str, ...]]:
    with isolated_callback_registries():
        recorder: Final = RecordingLogger()
        server.enqueue(ResponseSpec(body=None, events=_events(TEXT_RESPONSE)[:-1]))
        stream: Final = await _async_stream(
            probe is not None,
            {
                "model": MODEL,
                "api_key": "sk-test",
                "api_base": server.base_url,
                "callbacks": [recorder],
                "stream": True,
                "input": "hi",
            },
        )
        with pytest.raises(litellm.APIConnectionError) as raised:
            _ = [event async for event in stream]
        await drain_logging()
        await recorder.wait_for_async("async_log_failure_event")
        await asyncio.sleep(_SYNC_HANDLER_GRACE)
        return type(raised.value).__name__, str(raised.value), recorder.names


@pytest.mark.asyncio
async def test_a_native_stream_that_ends_early_fails_like_python(
    parity_server: RecordingServer, native_responses: FallbackProbe
) -> None:
    python: Final = await _truncated_leg(parity_server, probe=None)
    native: Final = await _truncated_leg(parity_server, probe=native_responses)
    assert native_responses.calls == 0
    assert native == python
