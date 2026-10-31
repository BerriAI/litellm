import asyncio
import itertools
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal

import anthropic
import httpx
import openai
import pytest
from anthropic.types import ToolParam
from integration._support.anthropic_sse import SseEvent, event_types, message_id, parse_sse, user_prompt
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, posted_targets
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall, ResponseOutputItemDoneEvent
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_BACKEND: Final = "mistral-multi-tool"
_PROVIDER_MODEL: Final = f"mistral/{_BACKEND}"
_PROVIDER_KEY: Final = "integration-provider-key"
_UPSTREAM_TARGET: Final = "/v1/chat/completions"
_CALL_ID_HEADER: Final = "x-litellm-call-id"
_ROWS: Final = "SELECT status FROM \"LiteLLM_SpendLogs\" WHERE metadata->>'litellm_call_id' = %s"
_ROW_SECONDS: Final = 70
_CACHE_ROWS: Final = "SELECT status, cache_hit FROM \"LiteLLM_SpendLogs\" WHERE metadata->>'litellm_call_id' = %s"
_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOL_SPECS: Final = (("Bash", "command"), ("Read", "file_path"), ("Glob", "pattern"))
_TOOLS_JSON: Final[list[JsonValue]] = [
    {
        "name": name,
        "description": f"{name} tool",
        "input_schema": {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
    }
    for name, field in _TOOL_SPECS
]
_ANTHROPIC_TOOLS: Final[list[ToolParam]] = [
    {
        "name": name,
        "description": f"{name} tool",
        "input_schema": {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
    }
    for name, field in _TOOL_SPECS
]
_OPENAI_TOOLS: Final[list[ChatCompletionToolParam]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
        },
    }
    for name, field in _TOOL_SPECS
]
_RESPONSES_TOOLS: Final[list[FunctionToolParam]] = [
    {
        "type": "function",
        "name": name,
        "description": f"{name} tool",
        "parameters": {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
        "strict": False,
    }
    for name, field in _TOOL_SPECS
]

ShapeName = Literal[
    "together",
    "five_together",
    "together_then_fragments",
    "held",
    "text_then_together",
    "together_usage_only_between",
    "together_no_finish",
    "together_with_finish",
    "single_fragments",
    "empty_arguments",
    "shared_index",
    "two_choices",
    "nonstream",
    "drop_after_calls",
]
_SHAPE: Final = TypeAdapter[ShapeName](ShapeName)
RecordProperty = Callable[[str, object], None]
Field = Literal["index", "id", "type", "name", "arguments"]
Variant = Literal["int", "list", "empty", "5kb", "twice"]
_FIELDS: Final[tuple[Field, ...]] = ("index", "id", "type", "name", "arguments")
_VARIANTS: Final[tuple[Variant, ...]] = ("int", "list", "empty", "5kb", "twice")
_TEXT: Final = "Checking both."
_PLACEHOLDER: Final = "__AUDIT_FIELD__"


@dataclass(frozen=True, slots=True)
class _Call:
    index: int
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class _Block:
    kind: str
    call_id: str | None
    name: str | None
    content: str


@dataclass(frozen=True, slots=True)
class _Sent:
    call_id: str
    events: tuple[SseEvent, ...]


def _two(tag: str) -> tuple[_Call, ...]:
    return (
        _Call(0, f"call_tmp_{tag}", "Bash", json.dumps({"command": f"ls -d /tmp/{tag}"})),
        _Call(1, f"call_var_{tag}", "Bash", json.dumps({"command": f"ls -d /var/{tag}"})),
    )


def _five(tag: str) -> tuple[_Call, ...]:
    reads: Final = tuple(
        _Call(index, f"call_read{index}_{tag}", "Read", json.dumps({"file_path": f"{name}-{tag}.md"}))
        for index, name in enumerate(("a", "b", "c"))
    )
    globs: Final = tuple(
        _Call(index + 3, f"call_glob{index}_{tag}", "Glob", json.dumps({"pattern": f"*-{tag}.{suffix}"}))
        for index, suffix in enumerate(("yml", "json"))
    )
    return (*reads, *globs)


def _one(tag: str) -> tuple[_Call, ...]:
    return (_Call(0, f"call_one_{tag}", "Bash", json.dumps({"command": f"echo {tag}"})),)


def _no_arguments(tag: str) -> tuple[_Call, ...]:
    return tuple(_Call(call.index, call.call_id, call.name, "") for call in _two(tag))


def _calls(shape: ShapeName, tag: str) -> tuple[_Call, ...]:
    match shape:
        case "five_together":
            return _five(tag)
        case "single_fragments":
            return _one(tag)
        case "empty_arguments":
            return _no_arguments(tag)
        case _:
            return _two(tag)


def _opening(call: _Call, arguments: str) -> dict[str, JsonValue]:
    return {
        "index": call.index,
        "id": call.call_id,
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    }


def _continuing(call: _Call, arguments: str) -> dict[str, JsonValue]:
    return {"index": call.index, "function": {"arguments": arguments}}


def _frame(identity: str, choices: list[JsonValue], usage: dict[str, JsonValue] | None = None) -> bytes:
    body: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _BACKEND,
        "choices": choices,
        **({"usage": usage} if usage is not None else {}),
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _delta_frame(identity: str, delta: dict[str, JsonValue], finish: str | None = None) -> bytes:
    return _frame(identity, [{"index": 0, "delta": delta, "finish_reason": finish}])


def _tool_frame(identity: str, entries: Sequence[JsonValue], *, first: bool) -> bytes:
    role: Final[dict[str, JsonValue]] = {"role": "assistant"} if first else {}
    return _delta_frame(identity, {**role, "tool_calls": list(entries)})


def _finish(identity: str) -> bytes:
    return _frame(identity, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], _USAGE)


_DONE: Final = b"data: [DONE]\n\n"


def _halves(text: str) -> tuple[str, str]:
    middle: Final = len(text) // 2
    return text[:middle], text[middle:]


def _together(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    return (
        _tool_frame(identity, [_opening(call, call.arguments) for call in calls], first=True),
        _finish(identity),
        _DONE,
    )


def _together_then_fragments(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    opened: Final = _tool_frame(identity, [_opening(call, _halves(call.arguments)[0]) for call in calls], first=True)
    continued: Final = tuple(
        _tool_frame(identity, [_continuing(call, _halves(call.arguments)[1])], first=False) for call in calls
    )
    return (opened, *continued, _finish(identity), _DONE)


def _held(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    first, second = calls
    first_head, first_tail = _halves(first.arguments)
    second_head, second_tail = _halves(second.arguments)
    return (
        _tool_frame(identity, [_opening(first, first_head)], first=True),
        _tool_frame(identity, [_continuing(first, first_tail), _opening(second, second_head)], first=False),
        _tool_frame(identity, [_continuing(second, second_tail)], first=False),
        _finish(identity),
        _DONE,
    )


def _text_then_together(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    return (
        _delta_frame(identity, {"role": "assistant", "content": _TEXT}),
        _tool_frame(identity, [_opening(call, call.arguments) for call in calls], first=False),
        _finish(identity),
        _DONE,
    )


def _together_usage_only_between(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    return (
        _tool_frame(identity, [_opening(call, call.arguments) for call in calls], first=True),
        _frame(identity, [], _USAGE),
        _delta_frame(identity, {}, "tool_calls"),
        _DONE,
    )


def _together_no_finish(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    return (_tool_frame(identity, [_opening(call, call.arguments) for call in calls], first=True), _DONE)


def _together_with_finish(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    entries: Final[list[JsonValue]] = [_opening(call, call.arguments) for call in calls]
    choice: Final[dict[str, JsonValue]] = {
        "index": 0,
        "delta": {"role": "assistant", "tool_calls": entries},
        "finish_reason": "tool_calls",
    }
    return (_frame(identity, [choice], _USAGE), _DONE)


def _single_fragments(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    (call,) = calls
    head, tail = _halves(call.arguments)
    return (
        _tool_frame(identity, [_opening(call, "")], first=True),
        _tool_frame(identity, [_continuing(call, head)], first=False),
        _tool_frame(identity, [_continuing(call, tail)], first=False),
        _finish(identity),
        _DONE,
    )


def _shared_index(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    shared: Final = tuple(_Call(0, call.call_id, call.name, call.arguments) for call in calls)
    return _together(identity, shared)


def _two_choices(identity: str, calls: tuple[_Call, ...]) -> tuple[bytes, ...]:
    choices: Final[list[JsonValue]] = [
        {"index": position, "delta": {"role": "assistant", "tool_calls": [_opening(call, call.arguments)]}}
        for position, call in enumerate(calls)
    ]
    return (_frame(identity, choices), _finish(identity), _DONE)


def _completion(identity: str, calls: tuple[_Call, ...]) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                            for call in calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": _USAGE,
        }
    ).encode()


def _stream(chunks: tuple[bytes, ...], *, pause: float = 0, abort_after: int | None = None) -> Reply:
    return Reply(content_type="text/event-stream", chunks=chunks, pause_between_chunks=pause, abort_after=abort_after)


def _reply(shape: ShapeName, identity: str, calls: tuple[_Call, ...], pause: float) -> Reply:
    match shape:
        case "together" | "five_together" | "empty_arguments":
            return _stream(_together(identity, calls), pause=pause)
        case "together_then_fragments":
            return _stream(_together_then_fragments(identity, calls), pause=pause)
        case "held":
            return _stream(_held(identity, calls), pause=pause)
        case "text_then_together":
            return _stream(_text_then_together(identity, calls), pause=pause)
        case "together_usage_only_between":
            return _stream(_together_usage_only_between(identity, calls), pause=pause)
        case "together_no_finish":
            return _stream(_together_no_finish(identity, calls), pause=pause)
        case "together_with_finish":
            return _stream(_together_with_finish(identity, calls), pause=pause)
        case "single_fragments":
            return _stream(_single_fragments(identity, calls), pause=pause)
        case "shared_index":
            return _stream(_shared_index(identity, calls), pause=pause)
        case "two_choices":
            return _stream(_two_choices(identity, calls), pause=pause)
        case "drop_after_calls":
            return _stream(_together(identity, calls), pause=0.2, abort_after=1)
        case "nonstream":
            return Reply(body=_completion(identity, calls))


def _prompt(shape: ShapeName, tag: str) -> str:
    return f"{shape}:{tag}"


def _assert_upstream_request(request: Request, *, stream: bool) -> dict[str, JsonValue]:
    assert (request.method, request.target) == ("POST", _UPSTREAM_TARGET), request
    assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}", request.headers
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _BACKEND, body
    assert bool(body.get("stream", False)) is stream, body
    names: Final = tuple(
        string_value(object_value(object_value(tool)["function"])["name"]) for tool in _list(body["tools"])
    )
    assert names == tuple(name for name, _ in _TOOL_SPECS), body
    return body


def _list(value: JsonValue) -> Sequence[JsonValue]:
    assert isinstance(value, list), value
    return value


def _upstream(pause: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        shape_name, tag = user_prompt(_JSON_OBJECT.validate_json(request.body)).split(":")
        shape: Final = _SHAPE.validate_python(shape_name)
        _assert_upstream_request(request, stream=shape != "nonstream")
        return _reply(shape, f"chatcmpl-{uuid.uuid4().hex}", _calls(shape, tag), pause)

    return answering_model_discovery(respond)


def _openai_base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _messages_body(model: str, prompt: str, *, stream: bool, cached: bool = True) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 256,
        "stream": stream,
        "tools": _TOOLS_JSON,
        "messages": [{"role": "user", "content": prompt}],
        **({} if cached else {"cache": {"no-cache": True}}),
    }


def _send(gateway: Gateway, model: str, shape: ShapeName, tag: str, *, cached: bool = True) -> _Sent:
    call_id: Final = uuid.uuid4().hex
    response: Final = gateway.request(
        "POST",
        "/v1/messages",
        _messages_body(model, _prompt(shape, tag), stream=shape != "nonstream", cached=cached),
        headers={_CALL_ID_HEADER: call_id},
    )
    assert response.status_code == 200, response.text
    assert response.headers[_CALL_ID_HEADER] == call_id, response.headers
    return _Sent(call_id, parse_sse(response.text))


def _delta_piece(delta: Mapping[str, JsonValue]) -> str:
    return str(delta.get("partial_json", delta.get("text", "")))


def _is_delta_of(event: SseEvent, index: JsonValue) -> bool:
    return event.event == "content_block_delta" and event.data["index"] == index


def _block(start: SseEvent, events: tuple[SseEvent, ...]) -> _Block:
    index: Final = start.data["index"]
    block: Final = object_value(start.data["content_block"])
    pieces: Final = tuple(
        _delta_piece(object_value(event.data["delta"])) for event in events if _is_delta_of(event, index)
    )
    call_id: Final = block.get("id")
    name: Final = block.get("name")
    return _Block(
        kind=string_value(block["type"]),
        call_id=call_id if isinstance(call_id, str) else None,
        name=name if isinstance(name, str) else None,
        content="".join(pieces),
    )


def _blocks(events: tuple[SseEvent, ...]) -> tuple[_Block, ...]:
    return tuple(_block(event, events) for event in events if event.event == "content_block_start")


def _expected_tool_blocks(calls: tuple[_Call, ...]) -> tuple[_Block, ...]:
    return tuple(_Block("tool_use", call.call_id, call.name, call.arguments) for call in calls)


def _stop_reason(events: tuple[SseEvent, ...]) -> JsonValue:
    delta: Final = next(event for event in events if event.event == "message_delta")
    return object_value(delta.data["delta"])["stop_reason"]


def _block_indexes(events: tuple[SseEvent, ...], kind: str) -> tuple[JsonValue, ...]:
    return tuple(event.data["index"] for event in events if event.event == kind)


def _assert_completed(sent: _Sent) -> None:
    types: Final = event_types(sent.events)
    assert types[0] == "message_start" and types[-1] == "message_stop", sent.events
    assert "error" not in types, sent.events
    starts: Final = _block_indexes(sent.events, "content_block_start")
    assert starts == tuple(range(len(starts))), sent.events
    assert _block_indexes(sent.events, "content_block_stop") == starts, sent.events


def _assert_tool_blocks(sent: _Sent, calls: tuple[_Call, ...], *, text: str = "") -> None:
    _assert_completed(sent)
    blocks: Final = _blocks(sent.events)
    texts: Final = tuple(block for block in blocks if block.kind == "text" and block.content)
    assert "".join(block.content for block in texts) == text, blocks
    assert tuple(block for block in blocks if block.kind == "tool_use") == _expected_tool_blocks(calls), blocks


def _assert_tool_turn(sent: _Sent, calls: tuple[_Call, ...], *, text: str = "") -> None:
    _assert_tool_blocks(sent, calls, text=text)
    assert _stop_reason(sent.events) == "tool_use", sent.events


def _statuses(call_id: str) -> tuple[JsonValue, ...]:
    return tuple(row["status"] for row in read_rows(_ROWS, (call_id,)))


def _assert_one_row(call_id: str, status: str) -> None:
    rows: Final = eventually(lambda: _statuses(call_id), lambda found: len(found) >= 1, seconds=_ROW_SECONDS)
    assert rows == (status,), rows


def _model(scenario_model: Callable[..., str], url: str) -> str:
    return scenario_model(model=_PROVIDER_MODEL, api_base=url + "/v1", api_key=_PROVIDER_KEY)


_STREAMED_SHAPES: Final[tuple[ShapeName, ...]] = (
    "together",
    "five_together",
    "together_then_fragments",
    "held",
    "together_usage_only_between",
    "together_with_finish",
    "empty_arguments",
)


@pytest.mark.parametrize("shape", _STREAMED_SHAPES)
def test_each_streamed_tool_call_becomes_its_own_tool_use_block(gateway: Gateway, shape: ShapeName) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        sent: Final = _send(gateway, model, shape, tag)
        _assert_tool_turn(sent, _calls(shape, tag))
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(sent.call_id, "success")


def test_calls_still_open_when_the_stream_ends_are_released_as_blocks(
    gateway: Gateway, record_property: RecordProperty
) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        sent: Final = _send(gateway, model, "together_no_finish", tag)
        record_property("stop_reason", json.dumps(_stop_reason(sent.events)))
        _assert_tool_blocks(sent, _two(tag))
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(sent.call_id, "success")


def test_text_before_two_calls_stays_one_text_block(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        sent: Final = _send(gateway, model, "text_then_together", tag)
        _assert_tool_turn(sent, _two(tag), text=_TEXT)
        assert [block.kind for block in _blocks(sent.events)] == ["text", "tool_use", "tool_use"]
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(sent.call_id, "success")


@pytest.mark.parametrize("shape", ("single_fragments", "nonstream"))
def test_single_call_and_non_streaming_turns_are_unchanged(gateway: Gateway, shape: ShapeName) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        if shape == "nonstream":
            call_id: Final = uuid.uuid4().hex
            response: Final = gateway.request(
                "POST",
                "/v1/messages",
                _messages_body(model, _prompt(shape, tag), stream=False),
                headers={_CALL_ID_HEADER: call_id},
            )
            assert response.status_code == 200, response.text
            content: Final = _list(_JSON_OBJECT.validate_json(response.content)["content"])
            tool_uses: Final = tuple(
                object_value(block) for block in content if object_value(block)["type"] == "tool_use"
            )
            assert tuple((block["id"], block["name"], block["input"]) for block in tool_uses) == tuple(
                (call.call_id, call.name, json.loads(call.arguments)) for call in _two(tag)
            ), content
            _assert_one_row(call_id, "success")
            return
        sent: Final = _send(gateway, model, shape, tag)
        _assert_tool_turn(sent, _one(tag))
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(sent.call_id, "success")


def test_anthropic_sdk_sync_stream_gets_five_tool_use_blocks(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        call_id: Final = uuid.uuid4().hex
        with client.messages.stream(
            model=model,
            max_tokens=256,
            tools=_ANTHROPIC_TOOLS,
            messages=[{"role": "user", "content": _prompt("five_together", tag)}],
            extra_headers={_CALL_ID_HEADER: call_id},
        ) as stream:
            final: Final = stream.get_final_message()
        assert final.stop_reason == "tool_use", final
        assert tuple(
            (block.id, block.name, block.input) for block in final.content if block.type == "tool_use"
        ) == tuple((call.call_id, call.name, json.loads(call.arguments)) for call in _five(tag)), final
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(call_id, "success")


async def test_anthropic_sdk_async_stream_gets_two_complete_blocks_from_fragments(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        call_id: Final = uuid.uuid4().hex
        async with client.messages.stream(
            model=model,
            max_tokens=256,
            tools=_ANTHROPIC_TOOLS,
            messages=[{"role": "user", "content": _prompt("together_then_fragments", tag)}],
            extra_headers={_CALL_ID_HEADER: call_id},
        ) as stream:
            final: Final = await stream.get_final_message()
        assert tuple(
            (block.id, block.name, block.input) for block in final.content if block.type == "tool_use"
        ) == tuple((call.call_id, call.name, json.loads(call.arguments)) for call in _two(tag)), final
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(call_id, "success")


def _grouped(entries: Sequence[ChoiceDeltaToolCall]) -> tuple[tuple[str | None, str | None, str], ...]:
    indexes: Final = tuple(dict.fromkeys(entry.index for entry in entries))
    return tuple(_merged(tuple(entry for entry in entries if entry.index == index)) for index in indexes)


def _merged(entries: Sequence[ChoiceDeltaToolCall]) -> tuple[str | None, str | None, str]:
    functions: Final = tuple(entry.function for entry in entries if entry.function is not None)
    return (
        next((entry.id for entry in entries if entry.id), None),
        next((function.name for function in functions if function.name), None),
        "".join(function.arguments or "" for function in functions),
    )


def test_chat_completions_stream_still_carries_both_calls(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        client: Final = openai.OpenAI(base_url=_openai_base(gateway), api_key=gateway.key, max_retries=0)
        call_id: Final = uuid.uuid4().hex
        chunks: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt("together", tag)}],
                tools=_OPENAI_TOOLS,
                stream=True,
                extra_headers={_CALL_ID_HEADER: call_id},
            )
        )
        deltas: Final = tuple(
            itertools.chain.from_iterable(
                choice.delta.tool_calls or ()
                for choice in itertools.chain.from_iterable(chunk.choices for chunk in chunks)
            )
        )
        assert _grouped(deltas) == tuple((call.call_id, call.name, call.arguments) for call in _two(tag)), chunks
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(call_id, "success")


def test_responses_stream_still_carries_both_calls(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        client: Final = openai.OpenAI(base_url=_openai_base(gateway), api_key=gateway.key, max_retries=0)
        call_id: Final = uuid.uuid4().hex
        events: Final = tuple(
            client.responses.create(
                model=model,
                input=_prompt("together", tag),
                tools=_RESPONSES_TOOLS,
                stream=True,
                extra_headers={_CALL_ID_HEADER: call_id},
            )
        )
        done: Final = tuple(event.item for event in events if isinstance(event, ResponseOutputItemDoneEvent))
        calls: Final = tuple(
            (item.call_id, item.name, item.arguments) for item in done if isinstance(item, ResponseFunctionToolCall)
        )
        assert calls == tuple((call.call_id, call.name, call.arguments) for call in _two(tag)), events
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(call_id, "success")


def _bad_value(field: Field, variant: Variant) -> str:
    match variant:
        case "int":
            return "1000000" if field == "index" else "12345"
        case "list":
            return "[1]" if field == "index" else json.dumps(["x"])
        case "empty":
            return '""'
        case "5kb":
            if field == "arguments":
                return json.dumps(json.dumps({"command": "x" * 5120}))
            return json.dumps("1" * 5120 if field == "index" else "x" * 5120)
        case "twice":
            return ""


def _decoy(field: Field) -> str:
    match field:
        case "index":
            return "0"
        case "id":
            return '"call_decoy"'
        case "type":
            return '"decoy"'
        case "name":
            return '"Decoy"'
        case "arguments":
            return json.dumps(json.dumps({"decoy": True}))


def _field_entry(call: _Call, field: Field) -> dict[str, JsonValue]:
    entry: Final = _opening(call, call.arguments)
    function: Final = object_value(entry["function"])
    match field:
        case "index" | "id" | "type":
            return {**entry, field: _PLACEHOLDER}
        case "name" | "arguments":
            return {**entry, "function": {**function, field: _PLACEHOLDER}}


def _normal_value(call: _Call, field: Field) -> str:
    match field:
        case "index":
            return str(call.index)
        case "id":
            return json.dumps(call.call_id)
        case "type":
            return '"function"'
        case "name":
            return json.dumps(call.name)
        case "arguments":
            return json.dumps(call.arguments)


def _malformed_frames(identity: str, calls: tuple[_Call, ...], field: Field, variant: Variant) -> tuple[bytes, ...]:
    first, second = calls
    frame: Final = _tool_frame(identity, [_opening(first, first.arguments), _field_entry(second, field)], first=True)
    placeholder: Final = json.dumps(_PLACEHOLDER).encode()
    key: Final = json.dumps(field).encode()
    rendered: Final = (
        frame.replace(
            key + b": " + placeholder,
            key + b": " + _decoy(field).encode() + b", " + key + b": " + _normal_value(second, field).encode(),
        )
        if variant == "twice"
        else frame.replace(placeholder, _bad_value(field, variant).encode())
    )
    return (rendered, _finish(identity), _DONE)


def _malformed_upstream(field: Field, variant: Variant) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        shape_name, tag = user_prompt(_JSON_OBJECT.validate_json(request.body)).split(":")
        identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
        _assert_upstream_request(request, stream=True)
        if shape_name == "single_fragments":
            return _stream(_single_fragments(identity, _one(tag)))
        return _stream(_malformed_frames(identity, _two(tag), field, variant))

    return answering_model_discovery(respond)


def _outcome(response: httpx.Response) -> str:
    if response.status_code != 200:
        return f"{response.status_code} {response.text[:200]}"
    events: Final = parse_sse(response.text)
    return json.dumps(
        {
            "events": sorted(set(event_types(events))),
            "blocks": [[block.kind, block.call_id, block.name, block.content[:80]] for block in _blocks(events)],
        }
    )


def _assert_terminated(response: httpx.Response) -> None:
    if response.status_code != 200:
        assert 400 <= response.status_code < 600 and "error" in response.text, response.text
        return
    types: Final = event_types(parse_sse(response.text))
    assert types and types[-1] in ("message_stop", "error"), response.text


def _assert_still_serves(gateway: Gateway, model: str) -> None:
    assert gateway.request("GET", "/health/liveliness").status_code == 200
    tag: Final = uuid.uuid4().hex[:12]
    sent: Final = _send(gateway, model, "single_fragments", tag)
    _assert_tool_turn(sent, _one(tag))
    _assert_one_row(sent.call_id, "success")


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("field", _FIELDS)
def test_malformed_second_call_field_never_breaks_the_stream(
    gateway: Gateway, field: Field, variant: Variant, record_property: RecordProperty
) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_malformed_upstream(field, variant)) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, _prompt("together", tag), stream=True),
            headers={_CALL_ID_HEADER: uuid.uuid4().hex},
        )
        record_property("outcome", _outcome(response))
        _assert_terminated(response)
        _assert_still_serves(gateway, model)
        assert posted_targets(wire) == (_UPSTREAM_TARGET, _UPSTREAM_TARGET)


def test_calls_sharing_an_index_lose_no_argument_text(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        sent: Final = _send(gateway, model, "shared_index", tag)
        _assert_completed(sent)
        tool_text: Final = "".join(block.content for block in _blocks(sent.events) if block.kind == "tool_use")
        assert tool_text == "".join(call.arguments for call in _two(tag)), sent.events
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_one_row(sent.call_id, "success")


def test_upstream_drop_after_two_calls_ends_in_an_error_event(
    gateway: Gateway, record_property: RecordProperty
) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        sent: Final = _send(gateway, model, "drop_after_calls", tag)
        record_property(
            "outcome", json.dumps([[block.kind, block.call_id, block.content] for block in _blocks(sent.events)])
        )
        types: Final = event_types(sent.events)
        assert types[0] == "message_start" and types[-1] == "error" and "message_stop" not in types, sent.events
        assert all(
            json.loads(block.content) is not None
            for block in _blocks(sent.events)
            if block.kind == "tool_use" and block.content
        ), sent.events
        _assert_one_row(sent.call_id, "failure")
        _assert_still_serves(gateway, model)


def test_upstream_401_reaches_the_caller(gateway: Gateway) -> None:
    with (
        wire_server(
            answering_model_discovery(
                lambda _: Reply(status=401, body=b'{"error": {"message": "bad key", "type": "invalid_request_error"}}')
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _model(scenario.model, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, _prompt("together", "x"), stream=True)
        )
        assert response.status_code == 401, response.text
        assert "bad key" in response.text, response.text
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        assert posted_targets(wire)[0] == _UPSTREAM_TARGET


def test_two_choices_in_one_chunk_still_terminate(gateway: Gateway, record_property: RecordProperty) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, _prompt("two_choices", tag), stream=True)
        )
        record_property("outcome", _outcome(response))
        _assert_terminated(response)
        _assert_still_serves(gateway, model)
        assert posted_targets(wire) == (_UPSTREAM_TARGET, _UPSTREAM_TARGET)


def _cache_hits(call_id: str) -> tuple[tuple[JsonValue, bool], ...]:
    rows: Final = eventually(
        lambda: read_rows(_CACHE_ROWS, (call_id,)), lambda found: len(found) >= 1, seconds=_ROW_SECONDS
    )
    return tuple((row["status"], row["cache_hit"] == "True") for row in rows)


def test_repeated_request_replays_both_blocks_from_cache_and_logs_once_each(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        first: Final = _send(gateway, model, "together", tag)
        _assert_tool_turn(first, _two(tag))
        assert _cache_hits(first.call_id) == (("success", False),)
        twin: Final = _send(gateway, model, "together", tag)
        _assert_tool_turn(twin, _two(tag))
        assert _cache_hits(twin.call_id) == (("success", True),)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        uncached: Final = _send(gateway, model, "together", tag, cached=False)
        _assert_tool_turn(uncached, _two(tag))
        assert _cache_hits(uncached.call_id) == (("success", False),)
        assert message_id(twin.events) == message_id(first.events)
        assert message_id(uncached.events) != message_id(first.events)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)


_BURST: Final[tuple[ShapeName, ...]] = (
    "together",
    "held",
    "five_together",
    "text_then_together",
    "single_fragments",
    "nonstream",
) * 5


def _expected_text(shape: ShapeName) -> str:
    return _TEXT if shape == "text_then_together" else ""


async def _burst_one(
    client: httpx.AsyncClient, key: str, model: str, shape: ShapeName
) -> tuple[ShapeName, str, str, httpx.Response]:
    tag: Final = uuid.uuid4().hex[:12]
    call_id: Final = uuid.uuid4().hex
    response: Final = await client.post(
        "/v1/messages",
        json=_messages_body(model, _prompt(shape, tag), stream=shape != "nonstream"),
        headers={"Authorization": f"Bearer {key}", _CALL_ID_HEADER: call_id},
    )
    return shape, tag, call_id, response


def _assert_burst_result(shape: ShapeName, tag: str, call_id: str, response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    if shape == "nonstream":
        content: Final = _list(_JSON_OBJECT.validate_json(response.content)["content"])
        ids: Final = tuple(object_value(block)["id"] for block in content if object_value(block)["type"] == "tool_use")
        assert ids == tuple(call.call_id for call in _two(tag)), content
        return
    _assert_tool_turn(_Sent(call_id, parse_sse(response.text)), _calls(shape, tag), text=_expected_text(shape))


async def test_concurrent_mixed_burst_gets_the_right_blocks_per_request(gateway: Gateway) -> None:
    with wire_server(_upstream(pause=0.05)) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            results: Final = await asyncio.gather(*(_burst_one(client, gateway.key, model, shape) for shape in _BURST))
        for shape, tag, call_id, response in results:
            _assert_burst_result(shape, tag, call_id, response)
        for _, _, call_id, _ in results:
            _assert_one_row(call_id, "success")
        assert posted_targets(wire) == (_UPSTREAM_TARGET,) * len(_BURST)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_slow_upstream_keeps_held_calls_whole(gateway: Gateway) -> None:
    with wire_server(_upstream(pause=0.5)) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        tags: Final = tuple(uuid.uuid4().hex[:12] for _ in range(4))
        with ThreadPoolExecutor(max_workers=len(tags)) as pool:
            sent: Final = tuple(pool.map(partial(_send, gateway, model, "held"), tags))
        for tag, each in zip(tags, sent, strict=True):
            _assert_tool_turn(each, _two(tag))
            _assert_one_row(each.call_id, "success")
        assert posted_targets(wire) == (_UPSTREAM_TARGET,) * len(tags)


def test_client_leaving_while_a_call_is_held_keeps_the_proxy_serving(
    gateway: Gateway, record_property: RecordProperty
) -> None:
    tag: Final = uuid.uuid4().hex[:12]
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        shape_name, request_tag = user_prompt(_JSON_OBJECT.validate_json(request.body)).split(":")
        identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
        if shape_name == "single_fragments":
            return _stream(_single_fragments(identity, _one(request_tag)))
        return Reply(
            content_type="text/event-stream", chunks=_together(identity, _two(request_tag)), gate_after_first=gate
        )

    with wire_server(answering_model_discovery(respond)) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        call_id: Final = uuid.uuid4().hex
        try:
            with gateway.client.stream(
                "POST",
                "/v1/messages",
                json=_messages_body(model, _prompt("together", tag), stream=True),
                headers={"Authorization": f"Bearer {gateway.key}", _CALL_ID_HEADER: call_id},
            ) as response:
                assert response.status_code == 200, response
                seen: Final = next(line for line in response.iter_lines() if '"tool_use"' in line)
                assert f"call_tmp_{tag}" in seen, seen
        finally:
            gate.set()
        _assert_still_serves(gateway, model)
        record_property("abandoned_rows", json.dumps(_statuses(call_id)))
        record_property("upstream_disconnects", wire.disconnected.qsize())


def _owned_send(gateway: Gateway, model: str, tag: str) -> _Sent | BaseException:
    try:
        return _send(gateway, model, "held", tag)
    except (httpx.HTTPError, AssertionError) as error:
        return error


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_worker_killed_mid_burst_leaves_survivors_with_whole_blocks(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with wire_server(_upstream(pause=0.3)) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario.model, wire.url)
        with owned_proxy_process(
            gateway, tmp_path, {"INTEGRATION_UPSTREAM_URL": gateway.upstream_url}, workers=2
        ) as owned:
            workers: Final = eventually(owned.worker_pids, lambda found: len(found) == 2, seconds=90)
            eventually(
                lambda: _owned_send(owned.gateway, model, uuid.uuid4().hex[:12]),
                lambda result: isinstance(result, _Sent),
                seconds=90,
            )
            victim: Final = min(workers)
            tags: Final = tuple(uuid.uuid4().hex[:12] for _ in range(12))
            with ThreadPoolExecutor(max_workers=len(tags)) as pool:
                futures: Final[tuple[Future[_Sent | BaseException], ...]] = tuple(
                    pool.submit(_owned_send, owned.gateway, model, tag) for tag in tags
                )
                eventually(lambda: sum(future.done() for future in futures), lambda done: done >= 1, seconds=60)
                with owned.replacing_worker(victim) as kill:
                    kill()
                    survivor_tag: Final = uuid.uuid4().hex[:12]
                    survivor: Final = eventually(
                        lambda: _owned_send(owned.gateway, model, survivor_tag),
                        lambda result: isinstance(result, _Sent),
                        seconds=30,
                    )
                results: Final = tuple(future.result(timeout=180) for future in futures)
            assert isinstance(survivor, _Sent), survivor
            _assert_tool_turn(survivor, _two(survivor_tag))
            completed: Final = tuple(
                (tag, result) for tag, result in zip(tags, results, strict=True) if isinstance(result, _Sent)
            )
            record_property("completed", len(completed))
            record_property("failed", len(results) - len(completed))
            assert completed, results
            for tag, result in completed:
                _assert_tool_turn(result, _two(tag))
            eventually(
                owned.worker_pids,
                lambda found: len(found) == 2 and victim not in found,
                seconds=graceful_stop_seconds() + 60,
            )
            after_tag: Final = uuid.uuid4().hex[:12]
            after: Final = eventually(
                lambda: _owned_send(owned.gateway, model, after_tag),
                lambda result: isinstance(result, _Sent),
                seconds=60,
            )
            assert isinstance(after, _Sent), after
            _assert_tool_turn(after, _two(after_tag))
            assert len(posted_targets(wire)) >= len(completed) + 2
