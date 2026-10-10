from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])
JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
JSON_MESSAGES: Final[TypeAdapter[list[dict[str, JsonValue]]]] = TypeAdapter(list[dict[str, JsonValue]])

CHAT_TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]

RESPONSES_TOOLS: Final = [
    {
        "type": "function",
        "name": "lookup_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    }
]

ANTHROPIC_TOOLS: Final = [
    {
        "name": "lookup_weather",
        "description": "Get weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    }
]

_TOOL_PREFIX: Final = "llm.output_messages.{message}.message.tool_calls.{tool}.tool_call."


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owned: OwnedProxy
    model: str
    provider: Wire
    destination: Wire


def _json_object(body: bytes) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(body)


def _json_object_value(value: JsonValue) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_python(value)


def _json_messages(value: str) -> list[dict[str, JsonValue]]:
    return JSON_MESSAGES.validate_json(value)


def _assert_chat_request(
    request: Request,
    *,
    messages: JsonValue,
    tools: JsonValue = CHAT_TOOLS,
    tool_choice: JsonValue | None = None,
    include_tools: bool = True,
    stream: bool | None = None,
    stream_options: JsonValue | None = None,
    n: int | None = None,
) -> dict[str, JsonValue]:
    request_tools: Final = (
        {
            "tools": tools,
            "tool_choice": (
                tool_choice if tool_choice is not None else {"type": "function", "function": {"name": "lookup_weather"}}
            ),
        }
        if include_tools
        else {}
    )
    expected: Final[dict[str, JsonValue]] = {
        "model": "gpt-4o-mini",
        "messages": messages,
        **request_tools,
        **({"stream": stream} if stream is not None else {}),
        **({"stream_options": stream_options} if stream_options is not None else {}),
        **({"n": n} if n is not None else {}),
    }
    observed: Final = _json_object(request.body)
    assert observed == expected, (observed, expected)
    return observed


def _chat_request_marker(request: Request) -> str:
    body: Final = _json_object(request.body)
    messages: Final = JSON_MESSAGES.validate_python(body["messages"])
    marker: Final = messages[0].get("content")
    assert isinstance(marker, str), body
    return marker


def _assert_responses_request(
    request: Request,
    *,
    marker: str,
    input_value: str = "weather in Paris?",
    stream: bool = False,
) -> None:
    expected: Final[dict[str, JsonValue]] = {
        "model": "gpt-4o-mini",
        "input": input_value,
        "tools": RESPONSES_TOOLS,
        "tool_choice": {"type": "function", "name": "lookup_weather"},
        "metadata": {"trace_marker": marker},
        **({"stream": True} if stream else {}),
    }
    observed: Final = _json_object(request.body)
    assert observed == expected, request


def _assert_messages_request(
    request: Request,
    *,
    marker: str,
    prompt: str = "weather in Paris?",
    stream: bool = False,
) -> None:
    expected: Final[dict[str, JsonValue]] = {
        "model": "claude-opus-5-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
        "tools": ANTHROPIC_TOOLS,
        "tool_choice": {"type": "auto"},
        "metadata": {},
        "stream": stream,
    }
    observed: Final = _json_object(request.body)
    assert observed == expected, request


def _stream_values(reply: Reply) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _json_object(chunk.split(b"data: ", 1)[1].splitlines()[0])
        for chunk in reply.chunks or ()
        if b"data: " in chunk and b"[DONE]" not in chunk
    )


def _sse_json_values(body: bytes) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _json_object(line.removeprefix(b"data: ").strip())
        for line in body.splitlines()
        if line.startswith(b"data: ") and line != b"data: [DONE]"
    )


def _chat_caller_response(reply: Reply, model: str) -> dict[str, JsonValue]:
    body: Final = _json_object(reply.body)
    choices: Final = JSON_MESSAGES.validate_python(body["choices"])
    return {
        **body,
        "model": model,
        "choices": [
            {
                **choice,
                "message": _chat_caller_message(_json_object_value(choice["message"])),
                "provider_specific_fields": {},
            }
            for choice in choices
        ],
    }


def _chat_caller_message(message: dict[str, JsonValue]) -> dict[str, JsonValue]:
    raw_tool_calls: Final = message.get("tool_calls")
    tool_calls: Final = JSON_MESSAGES.validate_python(raw_tool_calls) if isinstance(raw_tool_calls, list) else ()
    return {
        **{key: value for key, value in message.items() if key != "tool_calls"},
        **({"tool_calls": [_chat_caller_tool_call(call) for call in tool_calls]} if tool_calls else {}),
        "provider_specific_fields": {"refusal": None},
    }


def _chat_caller_tool_call(call: dict[str, JsonValue]) -> dict[str, JsonValue]:
    function: Final = _json_object_value(call["function"])
    arguments: Final = function.get("arguments")
    return {
        **call,
        "function": {
            **function,
            **(
                {"arguments": json.dumps(arguments)}
                if "arguments" in function and not isinstance(arguments, str)
                else {}
            ),
        },
    }


def _chat_output_tool_call(call: dict[str, JsonValue]) -> dict[str, JsonValue]:
    function: Final = _json_object_value(call["function"])
    return {
        **call,
        "id": call["id"] if isinstance(call.get("id"), str) else None,
        "function": {
            **function,
            **({"name": None} if "name" not in function else {}),
        },
    }


def _chat_output_value(reply: Reply) -> str:
    body: Final = _json_object(reply.body)
    choices: Final = JSON_MESSAGES.validate_python(body["choices"])
    assert len(choices) == 1, body
    message: Final = _chat_caller_message(_json_object_value(choices[0]["message"]))
    tool_calls: Final = JSON_MESSAGES.validate_python(message["tool_calls"]) if "tool_calls" in message else ()
    return json.dumps(
        [
            {
                **{
                    key: value
                    for key, value in message.items()
                    if key not in {"provider_specific_fields", "tool_calls"}
                },
                **({"tool_calls": [_chat_output_tool_call(call) for call in tool_calls]} if tool_calls else {}),
            }
        ]
    )


def _chat_caller_stream(reply: Reply, model: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(chain.from_iterable(_chat_caller_stream_events(event, model) for event in _stream_values(reply)))


def _chat_cache_hit_caller_stream(
    marker: str, model: str, calls: Sequence[dict[str, JsonValue]]
) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "id": marker,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"index": index, **call} for index, call in enumerate(calls)],
                    },
                }
            ],
        },
        {
            "id": marker,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    )


def _chat_caller_stream_event(event: dict[str, JsonValue], model: str) -> dict[str, JsonValue]:
    choices: Final = JSON_MESSAGES.validate_python(event["choices"])
    return {
        "id": event["id"],
        "object": event["object"],
        "created": 1,
        "model": model,
        "choices": [_chat_caller_stream_choice(choice) for choice in choices],
    }


def _chat_caller_stream_events(event: dict[str, JsonValue], model: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        _chat_caller_stream_event(event, model),
        *((_chat_caller_stream_usage_event(event, model),) if "usage" in event else ()),
    )


def _chat_caller_stream_usage_event(event: dict[str, JsonValue], model: str) -> dict[str, JsonValue]:
    return {
        "id": event["id"],
        "object": event["object"],
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": {}}],
        "usage": {**_json_object_value(event["usage"]), "cost": 4.05e-6},
    }


def _chat_caller_stream_choice(choice: dict[str, JsonValue]) -> dict[str, JsonValue]:
    delta: Final = _json_object_value(choice["delta"])
    tool_calls: Final = JSON_MESSAGES.validate_python(delta["tool_calls"]) if "tool_calls" in delta else None
    finish_reason: Final = choice.get("finish_reason")
    return {
        **{
            key: value
            for key, value in choice.items()
            if key not in {"delta", "finish_reason", "index"} and value is not None
        },
        "index": choice["index"],
        "delta": {
            **{key: value for key, value in delta.items() if key != "tool_calls" and value is not None},
            **(
                {"tool_calls": [_chat_caller_stream_tool_call(call) for call in tool_calls]}
                if tool_calls is not None
                else {}
            ),
        },
        **({"finish_reason": finish_reason} if finish_reason is not None else {}),
    }


def _chat_caller_stream_tool_call(call: dict[str, JsonValue]) -> dict[str, JsonValue]:
    function: Final = _json_object_value(call["function"])
    return {
        **{key: value for key, value in call.items() if key not in {"function", "type"}},
        "type": "function",
        "function": {**function, **({"arguments": ""} if "arguments" not in function else {})},
    }


def _normalize_chat_caller_stream(events: Sequence[JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return tuple({**_json_object_value(event), "created": 1} for event in events)


def _responses_caller_body(body: dict[str, JsonValue], model: str) -> dict[str, JsonValue]:
    usage: Final = _json_object_value(body["usage"])
    output: Final = JSON_MESSAGES.validate_python(body["output"])
    return {
        **body,
        "id": "<response-id>",
        "model": model,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "parallel_tool_calls": None,
        "temperature": None,
        "tool_choice": None,
        "tools": None,
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "text": None,
        "truncation": None,
        "user": None,
        "store": None,
        "output": [{**item, **({"namespace": None} if item.get("type") == "function_call" else {})} for item in output],
        "usage": {
            "input_tokens_details": None,
            "output_tokens_details": None,
            "cost": None,
            **usage,
        },
    }


def _responses_stream_caller_body(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    response_id: Final = body.get("id")
    assert isinstance(response_id, str), body
    usage: Final = _json_object_value(body["usage"]) if body.get("status") == "completed" else {}
    return {
        **body,
        "id": "<response-id>",
        **({"usage": {**usage, "cost": 4.05e-06}} if usage else {}),
    }


def _responses_caller_response(reply: Reply, model: str) -> dict[str, JsonValue]:
    return _responses_caller_body(_json_object(reply.body), model)


def _normalize_responses_caller_body(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    response_id: Final = body.get("id")
    assert isinstance(response_id, str) and response_id.startswith("resp_"), body
    return {**body, "id": "<response-id>"}


def _responses_caller_stream(reply: Reply, model: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            **event,
            "model": model,
            **(
                {"response": _responses_stream_caller_body(_json_object_value(event["response"]))}
                if "response" in event
                else {}
            ),
        }
        for event in _stream_values(reply)
    )


def _normalize_responses_caller_stream(
    events: Sequence[dict[str, JsonValue]],
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            **event,
            **(
                {"response": _normalize_responses_caller_body(_json_object_value(event["response"]))}
                if "response" in event
                else {}
            ),
        }
        for event in events
    )


def _messages_caller_response(reply: Reply, model: str) -> dict[str, JsonValue]:
    return {**_json_object(reply.body), "model": model}


def _messages_caller_raw_stream(reply: Reply, model: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            **event,
            **(
                {"message": {**_json_object_value(event["message"]), "model": model}}
                if event.get("type") == "message_start"
                else {}
            ),
        }
        for event in _stream_values(reply)
    )


def _messages_caller_stream_response(reply: Reply, model: str) -> dict[str, JsonValue]:
    body: Final = _messages_caller_response(reply, model)
    content: Final = JSON_MESSAGES.validate_python(body["content"])
    return {
        **body,
        "content": [{**block, **({"caller": None} if block.get("type") == "tool_use" else {})} for block in content],
    }


def _messages_caller_stream(
    reply: Reply, model: str, *, final_message: dict[str, JsonValue]
) -> tuple[dict[str, JsonValue], ...]:
    events: Final = tuple(
        chain.from_iterable(
            _messages_caller_stream_event(event, model, final_message) for event in _stream_values(reply)
        )
    )
    return (*events, {"type": "message_stop", "message": final_message})


def _messages_caller_stream_event(
    event: dict[str, JsonValue], model: str, final_message: dict[str, JsonValue]
) -> tuple[dict[str, JsonValue], ...]:
    if event.get("type") == "message_stop":
        return ()
    if event.get("type") == "content_block_stop":
        index: Final = event["index"]
        assert isinstance(index, int), event
        content: Final = JSON_MESSAGES.validate_python(final_message["content"])
        return ({**event, "content_block": content[index]},)
    caller_event: Final = {
        **event,
        **({"message": {**_json_object_value(event["message"]), "model": model}} if "message" in event else {}),
    }
    if event.get("type") != "content_block_delta":
        return (caller_event,)
    delta: Final = _json_object_value(event["delta"])
    if delta.get("type") != "input_json_delta":
        return (caller_event,)
    partial_json: Final = delta["partial_json"]
    assert isinstance(partial_json, str), event
    return (
        caller_event,
        {
            "type": "input_json",
            "partial_json": partial_json,
            "snapshot": JSON_OBJECT.validate_json(partial_json),
        },
    )


def _span_attributes(request: Request) -> Iterator[dict[str, str]]:
    if request.headers.get("content-type") != "application/x-protobuf":
        return
    batch: Final = ExportTraceServiceRequest.FromString(request.body)
    for resource_spans in batch.resource_spans:
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                yield {attribute.key: _attribute_text(attribute.value) for attribute in span.attributes}


def _attribute_text(value: object) -> str:
    assert isinstance(value, AnyValue)
    match value.WhichOneof("value"):
        case "string_value":
            return value.string_value
        case "int_value":
            return str(value.int_value)
        case "double_value":
            return str(value.double_value)
        case "bool_value":
            return str(value.bool_value)
        case _:
            return ""


def _spans(requests: tuple[Request, ...]) -> Iterator[dict[str, str]]:
    for request in requests:
        yield from _span_attributes(request)


def _canonical_response_id(value: str) -> str:
    try:
        return base64.b64decode(value.removeprefix("resp_").encode()).decode()
    except (ValueError, UnicodeDecodeError):
        return value


def _matching_llm_spans(requests: tuple[Request, ...], response_id: str) -> Iterator[dict[str, str]]:
    wanted: Final = _canonical_response_id(response_id)
    for attributes in _spans(requests):
        if attributes.get("openinference.span.kind") != "LLM":
            continue
        observed: Final = attributes.get("gen_ai.response.id", "")
        if _canonical_response_id(observed) == wanted or response_id in attributes.values():
            yield attributes


def _matching_marker_spans(requests: tuple[Request, ...], marker: str) -> Iterator[dict[str, str]]:
    for attributes in _spans(requests):
        if attributes.get("openinference.span.kind") == "LLM" and marker in attributes.values():
            yield attributes


def _single_span(spans: tuple[dict[str, str], ...]) -> dict[str, str]:
    assert len(spans) == 1, spans
    return spans[0]


def _matching_span(destination: Wire, response_id: str) -> dict[str, str]:
    spans: Final = eventually(
        lambda: tuple(_matching_llm_spans(destination.drain(), response_id)),
        bool,
        seconds=30,
    )
    return _single_span(spans)


def _matching_marker_span(destination: Wire, marker: str) -> dict[str, str]:
    spans: Final = eventually(
        lambda: tuple(_matching_marker_spans(destination.drain(), marker)),
        bool,
        seconds=30,
    )
    return _single_span(spans)


def _matching_output_value_span(destination: Wire, marker: str) -> dict[str, str]:
    spans: Final = eventually(
        lambda: tuple(
            attributes
            for attributes in _spans(destination.drain())
            if attributes.get("openinference.span.kind") == "LLM" and marker in attributes.get("output.value", "")
        ),
        bool,
        seconds=30,
    )
    return _single_span(spans)


def _matching_genai_marker_span(destination: Wire, marker: str) -> dict[str, str]:
    spans: Final = eventually(
        lambda: tuple(
            attributes
            for attributes in _spans(destination.drain())
            if attributes.get("gen_ai.operation.name") == "chat" and marker in attributes.values()
        ),
        bool,
        seconds=30,
    )
    return _single_span(spans)


def _matching_any_marker_span(destination: Wire, marker: str) -> dict[str, str]:
    def matches(requests: tuple[Request, ...]) -> tuple[dict[str, str], ...]:
        return tuple(attributes for attributes in _spans(requests) if marker in attributes.values())

    spans: Final = eventually(
        lambda: matches(destination.drain()),
        bool,
        seconds=30,
    )
    return _single_span(spans)


def _collect_marker_spans(
    destination: Wire, markers: tuple[str, ...], *, timeout_seconds: float = 30
) -> tuple[dict[str, str], ...]:
    expected: Final = frozenset(markers)

    def matches(requests: tuple[Request, ...]) -> tuple[dict[str, str], ...]:
        return tuple(
            attributes
            for attributes in _spans(requests)
            if attributes.get("openinference.span.kind") == "LLM"
            and any(marker in attributes.values() for marker in expected)
        )

    def complete(spans: tuple[dict[str, str], ...]) -> bool:
        return all(any(marker in attributes.values() for attributes in spans) for marker in expected)

    def collect(previous: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
        current: Final = eventually(lambda: matches(destination.drain()), bool, seconds=timeout_seconds)
        combined: Final = (*previous, *current)
        return combined if complete(combined) else collect(combined)

    return collect(())


def _llm_spans_through_markers(destination: Wire, markers: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    expected: Final = frozenset(markers)

    def llm_spans(requests: tuple[Request, ...]) -> tuple[dict[str, str], ...]:
        return tuple(
            attributes for attributes in _spans(requests) if attributes.get("openinference.span.kind") == "LLM"
        )

    def complete(spans: tuple[dict[str, str], ...]) -> bool:
        return all(any(marker in attributes.values() for attributes in spans) for marker in expected)

    def collect(previous: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
        current: Final = eventually(lambda: llm_spans(destination.drain()), bool, seconds=30)
        combined: Final = (*previous, *current)
        return combined if complete(combined) else collect(combined)

    return collect(())


def _write_config(
    directory: Path,
    *,
    callbacks: Sequence[str] = ("arize",),
    callback_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    loaded: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **loaded,
        "litellm_settings": {
            **loaded["litellm_settings"],
            "callbacks": list(callbacks),
            **(litellm_settings or {}),
        },
        **({"callback_settings": dict(callback_settings)} if callback_settings is not None else {}),
        "general_settings": {
            **loaded["general_settings"],
            **(general_settings or {}),
        },
    }
    path: Final = directory / f"openinference-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _environment(destination: Wire, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "LITELLM_OTEL_V2": "1",
        "OTEL_BSP_SCHEDULE_DELAY": "100",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only",
        "LITELLM_OTEL_BAGGAGE_METADATA_KEYS": "requester_metadata.trace_marker",
        "ARIZE_HTTP_ENDPOINT": destination.url + "/v1/traces",
        "ARIZE_SPACE_ID": "integration-space",
        "ARIZE_API_KEY": "integration-arize-key",
        **(extra or {}),
    }


def _collector(_request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


def _owned_sink_handler(sink: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def handle(request: Request) -> Reply:
        if (
            request.method != "POST"
            or not request.body
            or request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
            != "application/x-protobuf"
        ):
            return Reply(body=b"", content_type="application/x-protobuf")
        try:
            return sink(request)
        except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
            raise AssertionError(f"{request.method} {request.target}: {error!r}") from error

    return handle


def _provider_handler(upstream: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def handle(request: Request) -> Reply:
        if (
            request.method != "POST"
            or not request.body
            or request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower() != "application/json"
        ):
            return Reply(body=b"{}", content_type="application/json")
        try:
            return upstream(request)
        except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
            raise AssertionError(f"{request.method} {request.target}: {error!r}") from error

    return handle


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    upstream: Callable[[Request], Reply],
    *,
    callbacks: Sequence[str] = ("arize",),
    callback_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    remove_environment: tuple[str, ...] = (),
    disabled_environment: tuple[str, ...] = (),
    model_name: str = "openai/gpt-4o-mini",
    api_base_suffix: str = "/v1",
    destination_handler: Callable[[Request], Reply] | None = None,
    destination_wire: Wire | None = None,
    fresh_client_connections: bool = False,
    workers: int = 2,
) -> Iterator[Rig]:
    destination_context: Final[AbstractContextManager[Wire]] = (
        nullcontext(destination_wire)
        if destination_wire is not None
        else wire_server(_owned_sink_handler(destination_handler or _collector))
    )
    with wire_server(_provider_handler(upstream)) as provider, destination_context as destination:
        resolved_settings: Final = {
            key: (
                {**value, "endpoint": destination.url + "/v1/traces"}
                if key == "otel" and isinstance(value, dict) and value.get("endpoint") == "unused"
                else value
            )
            for key, value in (callback_settings or {}).items()
        }
        config: Final = _write_config(
            directory,
            callbacks=callbacks,
            callback_settings=resolved_settings,
            litellm_settings=litellm_settings,
            general_settings=general_settings,
        )
        preset_environment: Final = {
            **({"PHOENIX_COLLECTOR_ENDPOINT": destination.url + "/v1/traces"} if "arize_phoenix" in callbacks else {}),
            **(
                {
                    "WANDB_HOST": destination.url,
                    "WANDB_API_KEY": "integration-weave-key",
                    "WANDB_PROJECT_ID": "integration/project",
                }
                if "weave_otel" in callbacks
                else {}
            ),
            **(
                {
                    "LANGFUSE_OTEL_HOST": destination.url,
                    "LANGFUSE_PUBLIC_KEY": "integration-public",
                    "LANGFUSE_SECRET_KEY": "integration-secret",
                }
                if "langfuse_otel" in callbacks
                else {}
            ),
            **(
                {
                    "LEVOAI_API_KEY": "integration-levo-key",
                    "LEVOAI_ORG_ID": "integration-org",
                    "LEVOAI_WORKSPACE_ID": "integration-workspace",
                    "LEVOAI_COLLECTOR_URL": destination.url + "/v1/traces",
                }
                if "levo" in callbacks
                else {}
            ),
            **(
                {
                    "SIGNOZ_INGESTION_ENDPOINT": destination.url + "/v1/traces",
                    "SIGNOZ_INGESTION_KEY": "integration-signoz-key",
                }
                if "signoz" in callbacks
                else {}
            ),
            **(
                {"OTEL_EXPORTER_OTLP_ENDPOINT": destination.url}
                if any(callback_name in callbacks for callback_name in ("langtrace", "newrelic", "agentops"))
                else {}
            ),
        }
        overrides: Final = {
            key: value
            for key, value in _environment(destination, {**preset_environment, **(environment or {})}).items()
            if key not in disabled_environment
        }
        with (
            owned_proxy_process(
                gateway,
                directory,
                overrides,
                config=config,
                remove_environment=remove_environment,
                workers=workers,
            ) as owned,
            ExitStack() as resources,
        ):
            proxy_client: Final = (
                resources.enter_context(
                    httpx.Client(
                        base_url=str(owned.gateway.client.base_url),
                        timeout=15,
                        trust_env=False,
                        limits=httpx.Limits(max_keepalive_connections=0),
                    )
                )
                if fresh_client_connections
                else owned.gateway.client
            )
            proxy: Final = Gateway(
                client=proxy_client,
                key=owned.gateway.key,
                upstream_url=owned.gateway.upstream_url,
            )
            scenario: Final = resources.enter_context(proxy.scenario())
            model: Final = scenario.model(
                model=model_name,
                api_base=provider.url + api_base_suffix,
            )
            yield Rig(proxy, owned, model, provider, destination)


def _chat_tool_call(identity: str, city: str = "Paris") -> dict[str, JsonValue]:
    return {
        "id": "call_" + identity,
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "arguments": json.dumps({"city": city}),
        },
    }


def _chat_response(identity: str, calls: Sequence[dict[str, JsonValue]] | None = None) -> Reply:
    tool_calls: Final = list(calls if calls is not None else (_chat_tool_call(identity),))
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _chat_plain_response(identity: str, content: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        ).encode()
    )


def _chat_stream_response(identity: str, calls: Sequence[dict[str, JsonValue]], *, include_usage: bool = True) -> Reply:
    def first_chunk(index: int, call: dict[str, JsonValue]) -> dict[str, JsonValue]:
        fields: Final = object_value(JSON_VALUE.validate_python(call))
        return {
            "index": index,
            "id": string_value(fields["id"]),
            "type": "function",
            "function": {"name": "lookup_weather"},
        }

    def arguments_chunk(index: int, call: dict[str, JsonValue]) -> dict[str, JsonValue]:
        fields: Final = object_value(JSON_VALUE.validate_python(call))
        function: Final = object_value(fields["function"])
        return {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": string_value(function["arguments"])},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

    chunks: Final = [
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [first_chunk(index, call) for index, call in enumerate(calls)],
                    },
                    "finish_reason": None,
                }
            ],
        },
        *(arguments_chunk(index, call) for index, call in enumerate(calls)),
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            **({"usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}} if include_usage else {}),
        },
    ]
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + (b"data: [DONE]\n\n",),
    )


def _responses_response(identity: str, calls: Sequence[dict[str, JsonValue]] | None = None) -> Reply:
    def response_item(call: dict[str, JsonValue]) -> dict[str, JsonValue]:
        fields: Final = object_value(JSON_VALUE.validate_python(call))
        function: Final = object_value(fields["function"])
        call_id: Final = string_value(fields["id"])
        return {
            "type": "function_call",
            "id": "fc_" + call_id,
            "call_id": call_id,
            "name": string_value(function["name"]),
            "arguments": string_value(function["arguments"]),
            "status": "completed",
        }

    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [response_item(call) for call in calls if calls is not None]
        if calls is not None
        else [response_item(_chat_tool_call(identity))],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }
    return Reply(body=json.dumps(response).encode())


def _responses_stream_response(identity: str, calls: Sequence[dict[str, JsonValue]]) -> Reply:
    response: Final = _json_object(_responses_response(identity, calls).body)
    output: Final = response["output"]
    items: Final = tuple(object_value(item) for item in output) if isinstance(output, list) else ()
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        *(
            {
                "type": "response.output_item.added",
                "sequence_number": index + 1,
                "output_index": index,
                "item": item,
            }
            for index, item in enumerate(items)
        ),
        *(
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": index + len(items) + 1,
                "item_id": str(item["id"]),
                "output_index": index,
                "delta": str(item["arguments"]),
            }
            for index, item in enumerate(items)
        ),
        {
            "type": "response.completed",
            "sequence_number": len(items) * 2 + 1,
            "response": response,
        },
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _anthropic_response(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_" + identity,
                        "name": "lookup_weather",
                        "input": {"city": "Paris"},
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 4},
            }
        ).encode()
    )


def _anthropic_stream_response(identity: str) -> Reply:
    message: Final = {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {
                "type": "tool_use",
                "id": "call_" + identity,
                "name": "lookup_weather",
                "input": {"city": "Paris"},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }
    events: Final = (
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_" + identity,
                "name": "lookup_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _tool_call_attributes(
    attributes: Mapping[str, str],
    *,
    message_index: int = 0,
    tool_index: int = 0,
) -> tuple[str, str, dict[str, JsonValue]]:
    prefix: Final = _TOOL_PREFIX.format(message=message_index, tool=tool_index)
    return (
        attributes[prefix + "id"],
        attributes[prefix + "function.name"],
        JSON_OBJECT.validate_json(attributes[prefix + "function.arguments"]),
    )


def _assert_tool_span(
    attributes: Mapping[str, str],
    *,
    marker: str,
    output: JsonValue,
    calls: Sequence[tuple[str, str, Mapping[str, JsonValue]]],
    metadata: Mapping[str, JsonValue] | None = None,
    baggage: Mapping[str, str] | None = None,
) -> None:
    for index, (call_id, name, arguments) in enumerate(calls):
        observed: Final = _tool_call_attributes(attributes, tool_index=index)
        assert observed == (call_id, name, dict(arguments)), f"tool call {index} for {marker}: {observed!r}"
    observed_output: Final = JSON_MESSAGES.validate_json(attributes["output.value"])
    assert observed_output == output, (marker, observed_output)
    if metadata is None:
        assert "metadata" not in attributes, attributes
    else:
        assert json.loads(attributes["metadata"]) == dict(metadata), attributes
    for key, value in (baggage or {}).items():
        assert attributes.get("litellm.metadata." + key) == value, attributes


def _response_tool_calls(identity: str, cities: Sequence[str] = ("Paris",)) -> list[dict[str, JsonValue]]:
    return [_chat_tool_call(identity + "-" + city, city) for city in cities]
