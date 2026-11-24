import itertools
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.anthropic_sse import SseEvent, parse_sse
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_GPT6: Final = "gpt-6.2-sol"
_PRE_GPT6: Final = "gpt-5.5"
_API_KEY: Final = "synthetic-openai-key"
_PREFIX: Final = "litellm_encrypted_reasoning:"
_MARKER: Final = "litellm_turn_settings:"
_MIDCONVO_ITEM_TYPES: Final = frozenset({"configuration_update", "additional_tools"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType({"input_tokens": 41, "output_tokens": 5, "total_tokens": 46})
_SCHEMA_A: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
)
_SCHEMA_B: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "object",
        "properties": {"verdict": {"type": "string"}, "confidence": {"type": "number"}},
        "required": ["verdict", "confidence"],
        "additionalProperties": False,
    }
)
_CITY_PARAMETERS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)
_TOOLS: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "weather": MappingProxyType(
            {"name": "get_weather", "description": "Weather for a city", "input_schema": dict(_CITY_PARAMETERS)}
        ),
        "time": MappingProxyType(
            {"name": "get_time", "description": "Local time in a city", "input_schema": dict(_CITY_PARAMETERS)}
        ),
        "weather_in_celsius": MappingProxyType(
            {
                "name": "get_weather",
                "description": "Weather for a city, always in Celsius",
                "input_schema": dict(_CITY_PARAMETERS),
            }
        ),
    }
)

Block: TypeAlias = dict[str, JsonValue]
Content: TypeAlias = tuple[Block, ...]
Rewrite: TypeAlias = Callable[[int, int, Block], Block | None]


@dataclass(frozen=True, slots=True)
class _Turn:
    effort: str | None = "low"
    tools: tuple[str, ...] = ()
    schema: Mapping[str, JsonValue] | None = None
    format_field: Literal["output_format", "output_config"] = "output_config"
    reply: Literal["text", "tool_call"] = "text"


@dataclass(frozen=True, slots=True)
class _Settings:
    effort: JsonValue
    tools: tuple[tuple[JsonValue, JsonValue], ...]
    schema: JsonValue


@dataclass(frozen=True, slots=True)
class _Conversation:
    bodies: tuple[dict[str, JsonValue], ...]
    contents: tuple[Content, ...]
    blobs: tuple[str, ...]
    user_texts: tuple[str, ...]


def _keep(sending: int, answered: int, block: Block) -> Block | None:
    return block


def _strip_all(sending: int, answered: int, block: Block) -> Block | None:
    return None


def _strip_second_turn_when_sending_third(sending: int, answered: int, block: Block) -> Block | None:
    return None if (sending, answered) == (2, 1) else block


def _record_field(block: Block) -> str:
    return "signature" if block["type"] == "thinking" else "data"


def _with_record(block: Block, edit: Callable[[str], str]) -> Block:
    field: Final = _record_field(block)
    head, marker, rest = str(block[field]).partition(_MARKER)
    if not marker:
        return block
    record, separator, blob = rest.partition(";")
    return {**block, field: f"{head}{marker}{edit(record)}{separator}{blob}"}


def _garbled(sending: int, answered: int, block: Block) -> Block | None:
    return _with_record(block, lambda record: f"%%not-a-record%%{record[::-1]}")


def _tampered(sending: int, answered: int, block: Block) -> Block | None:
    return _with_record(block, lambda record: f"{record[:-1]}{'B' if record.endswith('A') else 'A'}")


def _blob(identity: str, number: int) -> str:
    return f"gAAAAB{identity}-encrypted-reasoning-turn-{number}"


def _output(identity: str, number: int, turn: _Turn, summarized: bool) -> tuple[Block, Block]:
    summary: Final[list[JsonValue]] = (
        [{"type": "summary_text", "text": f"Planning step {number}."}] if summarized else []
    )
    reasoning: Final[Block] = {
        "type": "reasoning",
        "id": f"rs_{identity}_{number}",
        "summary": summary,
        "encrypted_content": _blob(identity, number),
    }
    if turn.reply == "tool_call":
        return reasoning, {
            "type": "function_call",
            "id": f"fc_{identity}_{number}",
            "call_id": f"call_{identity}_{number}",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"}),
            "status": "completed",
        }
    return reasoning, {
        "type": "message",
        "id": f"msg_{identity}_{number}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": f"Answer {number}.", "annotations": []}],
    }


def _response(identity: str, number: int, backend: str, status: str, output: tuple[Block, ...]) -> Block:
    return {
        "id": f"resp_{identity}_{number}",
        "object": "response",
        "created_at": 1789788253,
        "status": status,
        "model": backend,
        "output": list(output),
        "usage": dict(_USAGE) if status == "completed" else None,
    }


def _streamed_item(index: int, item: Block) -> tuple[Block, tuple[Block, ...]]:
    def delta(kind: str, text: JsonValue, **position: JsonValue) -> Block:
        return {"type": kind, "item_id": item["id"], "output_index": index, **position, "delta": text}

    match item["type"]:
        case "reasoning":
            summaries: Final = enumerate(_JSON_OBJECTS.validate_python(item["summary"]))
            return {**item, "summary": [], "encrypted_content": None}, tuple(
                delta("response.reasoning_summary_text.delta", summary["text"], summary_index=position)
                for position, summary in summaries
            )
        case "function_call":
            return {**item, "arguments": ""}, (delta("response.function_call_arguments.delta", item["arguments"]),)
        case _:
            parts: Final = enumerate(_JSON_OBJECTS.validate_python(item["content"]))
            return {**item, "status": "in_progress", "content": []}, tuple(
                delta("response.output_text.delta", part["text"], content_index=position) for position, part in parts
            )


def _item_events(index: int, item: Block) -> tuple[Block, ...]:
    started, deltas = _streamed_item(index, item)
    return (
        {"type": "response.output_item.added", "output_index": index, "item": started},
        *deltas,
        {"type": "response.output_item.done", "output_index": index, "item": item},
    )


def _frames(identity: str, number: int, backend: str, output: tuple[Block, ...]) -> tuple[bytes, ...]:
    item_events: Final = itertools.chain.from_iterable(_item_events(index, item) for index, item in enumerate(output))
    events: Final = (
        {"type": "response.created", "response": _response(identity, number, backend, "in_progress", ())},
        *item_events,
        {"type": "response.completed", "response": _response(identity, number, backend, "completed", output)},
    )
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': sequence})}\n\n".encode()
        for sequence, event in enumerate(events)
    )


def _reply(identity: str, number: int, backend: str, output: tuple[Block, ...], stream: bool) -> Reply:
    if stream:
        return Reply(content_type="text/event-stream", chunks=_frames(identity, number, backend, output))
    return Reply(body=json.dumps(_response(identity, number, backend, "completed", output)).encode())


def _format(schema: Mapping[str, JsonValue]) -> Block:
    return {"type": "json_schema", "schema": dict(schema)}


def _request_body(model: str, turn: _Turn, messages: list[JsonValue], stream: bool) -> Block:
    output_format: Final = None if turn.schema is None else _format(turn.schema)
    output_config: Final[Block] = {
        **({} if turn.effort is None else {"effort": turn.effort}),
        **({"format": output_format} if output_format is not None and turn.format_field == "output_config" else {}),
    }
    return {
        "model": model,
        "max_tokens": 4096,
        "stream": stream,
        "thinking": {"type": "disabled"} if turn.effort is None else {"type": "adaptive"},
        "messages": messages,
        **({"output_config": output_config} if output_config else {}),
        **(
            {"output_format": output_format}
            if output_format is not None and turn.format_field == "output_format"
            else {}
        ),
        **({"tools": [dict(_TOOLS[name]) for name in turn.tools]} if turn.tools else {}),
    }


def _user_text(identity: str, index: int) -> str:
    return f"Step {index + 1} of task {identity}."


def _user_message(identity: str, index: int, answered: tuple[Content, ...]) -> Block:
    previous_calls: Final = () if index == 0 else tuple(b for b in answered[index - 1] if b["type"] == "tool_use")
    if not previous_calls:
        return {"role": "user", "content": _user_text(identity, index)}
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": call["id"], "content": "18C and sunny"} for call in previous_calls
        ],
    }


def _rewritten(sending: int, answered: int, block: Block, rewrite: Rewrite) -> Block | None:
    if block["type"] in ("thinking", "redacted_thinking"):
        return rewrite(sending, answered, block)
    return block


def _assistant_message(sending: int, answered: int, content: Content, rewrite: Rewrite) -> Block:
    blocks: Final = (_rewritten(sending, answered, block, rewrite) for block in content)
    return {"role": "assistant", "content": [block for block in blocks if block is not None]}


def _history(identity: str, answered: tuple[Content, ...], rewrite: Rewrite) -> list[JsonValue]:
    sending: Final = len(answered)
    earlier: Final = itertools.chain.from_iterable(
        (_user_message(identity, index, answered), _assistant_message(sending, index, answered[index], rewrite))
        for index in range(sending)
    )
    return [*earlier, _user_message(identity, sending, answered)]


def _deltas(events: tuple[SseEvent, ...], index: JsonValue) -> tuple[Block, ...]:
    matching: Final = (event for event in events if event.event == "content_block_delta")
    return tuple(
        _JSON_OBJECT.validate_python(event.data["delta"]) for event in matching if event.data["index"] == index
    )


def _joined(deltas: tuple[Block, ...], kind: str, field: str) -> str:
    return "".join(str(delta[field]) for delta in deltas if delta["type"] == kind)


def _assembled(block: Block, deltas: tuple[Block, ...]) -> Block:
    match block["type"]:
        case "thinking":
            return {
                "type": "thinking",
                "thinking": f"{block.get('thinking') or ''}{_joined(deltas, 'thinking_delta', 'thinking')}",
                "signature": f"{block.get('signature') or ''}{_joined(deltas, 'signature_delta', 'signature')}",
            }
        case "text":
            return {"type": "text", "text": f"{block.get('text') or ''}{_joined(deltas, 'text_delta', 'text')}"}
        case "tool_use":
            arguments: Final = _joined(deltas, "input_json_delta", "partial_json")
            return {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": _JSON_OBJECT.validate_json(arguments) if arguments else {},
            }
        case _:
            return block


def _streamed_content(text: str) -> Content:
    events: Final = parse_sse(text)
    starts: Final = tuple(event.data for event in events if event.event == "content_block_start")
    return tuple(
        _assembled(_JSON_OBJECT.validate_python(start["content_block"]), _deltas(events, start["index"]))
        for start in starts
    )


def _assistant_content(response: httpx.Response, stream: bool) -> Content:
    assert response.status_code == 200, response.text
    if stream:
        return _streamed_content(response.text)
    return _JSON_OBJECTS.validate_python(_JSON_OBJECT.validate_json(response.content)["content"])


def _each_turn(
    send: Callable[[tuple[Content, ...]], Content], count: int, answered: tuple[Content, ...]
) -> tuple[Content, ...]:
    if len(answered) == count:
        return answered
    return _each_turn(send, count, (*answered, send(answered)))


def _converse(
    gateway: Gateway,
    turns: tuple[_Turn, ...],
    *,
    backend: str = _GPT6,
    stream: bool = False,
    summarized: bool = True,
    rewrite: Rewrite = _keep,
    openai_dialect_host: bool = True,
) -> _Conversation:
    identity: Final = f"midconvo-settings-{uuid.uuid4().hex}"
    replies: Final[Iterator[Reply]] = iter(
        tuple(
            _reply(identity, number, backend, _output(identity, number, turn, summarized), stream)
            for number, turn in enumerate(turns, start=1)
        )
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        return next(replies)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{backend}",
            api_base=wire.url,
            api_key=_API_KEY,
            **({"prompt_cache_options": {"mode": "implicit"}} if openai_dialect_host else {}),
        )

        def send(answered: tuple[Content, ...]) -> Content:
            messages: Final = _history(identity, answered, rewrite)
            body: Final = _request_body(model, turns[len(answered)], messages, stream)
            return _assistant_content(gateway.request("POST", "/v1/messages", body), stream)

        contents: Final = _each_turn(send, len(turns), ())
        bodies: Final = tuple(_JSON_OBJECT.validate_json(request.body) for request in wire.drain())
    assert len(bodies) == len(turns), bodies
    return _Conversation(
        bodies=bodies,
        contents=contents,
        blobs=tuple(_blob(identity, number) for number in range(1, len(turns) + 1)),
        user_texts=tuple(_user_text(identity, index) for index in range(len(turns))),
    )


def _inputs(body: Block) -> tuple[Block, ...]:
    return _JSON_OBJECTS.validate_python(body["input"])


def _top_level(body: Block) -> Block:
    return {key: body[key] for key in ("reasoning", "text", "tools") if key in body}


def _midconvo_items(body: Block) -> tuple[Block, ...]:
    return tuple(item for item in _inputs(body) if item.get("type") in _MIDCONVO_ITEM_TYPES)


def _encrypted_contents(body: Block) -> tuple[JsonValue, ...]:
    return tuple(item.get("encrypted_content") for item in _inputs(body) if item.get("type") == "reasoning")


def _user_item(text: str) -> Block:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _turn_items(previous: Block, current: Block, blob: str) -> tuple[Block, ...]:
    before: Final = _inputs(previous)
    after: Final = _inputs(current)
    assert after[: len(before)] == before, current
    reasoning, output = after[len(before) : len(before) + 2]
    assert reasoning["type"] == "reasoning" and reasoning["encrypted_content"] == blob, current
    assert output["type"] in ("message", "function_call"), current
    return after[len(before) + 2 :]


def _update_then_user(previous: Block, current: Block, blob: str) -> tuple[Block, Block]:
    items: Final = _turn_items(previous, current, blob)
    assert len(items) == 2, f"expected one settings item right before the turn's user message, got {items}"
    return items[0], items[1]


def _effort_update(effort: str) -> Block:
    return {"type": "configuration_update", "reasoning": {"effort": effort}}


def _assert_format_update(item: Block, schema: Mapping[str, JsonValue]) -> None:
    assert item["type"] == "configuration_update" and set(item) == {"type", "text"}, item
    text_format: Final = _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_python(item["text"])["format"])
    assert (text_format["type"], text_format["schema"]) == ("json_schema", dict(schema)), item


def _function_tool(tool: Block) -> tuple[JsonValue, ...]:
    return tool["type"], tool["name"], tool.get("description"), tool.get("parameters")


def _assert_additional_tools(item: Block, names: tuple[str, ...]) -> None:
    assert (item["type"], item["role"], set(item)) == ("additional_tools", "developer", {"type", "role", "tools"}), item
    assert tuple(map(_function_tool, _JSON_OBJECTS.validate_python(item["tools"]))) == tuple(
        ("function", _TOOLS[name]["name"], _TOOLS[name]["description"], dict(_CITY_PARAMETERS)) for name in names
    ), item


def _settings(body: Block) -> _Settings:
    reasoning: Final = _JSON_OBJECT.validate_python(body.get("reasoning") or {})
    text_format: Final = _JSON_OBJECT.validate_python(
        _JSON_OBJECT.validate_python(body.get("text") or {}).get("format") or {}
    )
    tools: Final = _JSON_OBJECTS.validate_python(body.get("tools") or ())
    return _Settings(
        effort=reasoning.get("effort"),
        tools=tuple((tool.get("name"), tool.get("description")) for tool in tools),
        schema=text_format.get("schema"),
    )


def _requested(turn: _Turn) -> _Settings:
    return _Settings(
        effort=turn.effort,
        tools=tuple((_TOOLS[name]["name"], _TOOLS[name]["description"]) for name in turn.tools),
        schema=None if turn.schema is None else dict(turn.schema),
    )


def _records(content: Content) -> tuple[JsonValue, ...]:
    reasoning_blocks: Final = (block for block in content if block["type"] in ("thinking", "redacted_thinking"))
    return tuple(block[_record_field(block)] for block in reasoning_blocks)


_EFFORT_CHANGE: Final = (_Turn(effort="low"), _Turn(effort="high"), _Turn(effort="high"))


@pytest.mark.parametrize(
    ("stream", "summarized"),
    ((False, True), (False, False), (True, True)),
    ids=("non-streaming-thinking", "non-streaming-redacted-thinking", "streaming-thinking"),
)
def test_effort_change_mid_conversation_is_sent_as_a_configuration_update(
    gateway: Gateway, stream: bool, summarized: bool
) -> None:
    """regression: a GPT-6 effort change at turn 2 keeps turn 1's top-level reasoning and is replayed at turn 3."""
    conversation: Final = _converse(gateway, _EFFORT_CHANGE, stream=stream, summarized=summarized)
    first, second, third = conversation.bodies
    assert _turn_items(first, second, conversation.blobs[0]) == (
        _effort_update("high"),
        _user_item(conversation.user_texts[1]),
    ), second
    assert _turn_items(second, third, conversation.blobs[1]) == (_user_item(conversation.user_texts[2]),), third
    assert _top_level(first)["reasoning"] == {"effort": "low"}, first
    assert _top_level(second) == _top_level(first) == _top_level(third), conversation.bodies


@pytest.mark.parametrize(
    ("format_field", "before", "after"),
    (("output_format", None, _SCHEMA_A), ("output_config", _SCHEMA_A, _SCHEMA_B)),
    ids=("output-format-added", "output-config-format-changed"),
)
def test_output_format_change_mid_conversation_is_sent_as_a_configuration_update(
    gateway: Gateway,
    format_field: Literal["output_format", "output_config"],
    before: Mapping[str, JsonValue] | None,
    after: Mapping[str, JsonValue],
) -> None:
    """regression: a GPT-6 output format change at turn 2 keeps turn 1's top-level text and is replayed at turn 3."""
    turns: Final = (
        _Turn(schema=before, format_field=format_field),
        _Turn(schema=after, format_field=format_field),
        _Turn(schema=after, format_field=format_field),
    )
    conversation: Final = _converse(gateway, turns)
    first, second, third = conversation.bodies
    update, user = _update_then_user(first, second, conversation.blobs[0])
    _assert_format_update(update, after)
    assert user == _user_item(conversation.user_texts[1]), second
    assert _turn_items(second, third, conversation.blobs[1]) == (_user_item(conversation.user_texts[2]),), third
    assert _settings(first).schema == (None if before is None else dict(before)), first
    assert _top_level(second) == _top_level(first) == _top_level(third), conversation.bodies


@pytest.mark.parametrize("first_reply", ("text", "tool_call"), ids=("plain-turn", "agentic-tool-result-turn"))
def test_tool_added_mid_conversation_is_sent_as_additional_tools(
    gateway: Gateway, first_reply: Literal["text", "tool_call"]
) -> None:
    """regression: a tool added at turn 2 leaves turn 1's top-level tools and is announced once at that turn.

    A turn that only returns a tool result has no user message, so the item follows the tool output and never
    separates a call from its output.
    """
    turns: Final = (
        _Turn(tools=("weather",), reply=first_reply),
        _Turn(tools=("weather", "time")),
        _Turn(tools=("weather", "time")),
    )
    conversation: Final = _converse(gateway, turns)
    first, second, third = conversation.bodies
    if first_reply == "tool_call":
        tool_output, announced = _update_then_user(first, second, conversation.blobs[0])
        assert tool_output["type"] == "function_call_output", second
    else:
        announced, user = _update_then_user(first, second, conversation.blobs[0])
        assert user == _user_item(conversation.user_texts[1]), second
    _assert_additional_tools(announced, ("time",))
    assert _turn_items(second, third, conversation.blobs[1]) == (_user_item(conversation.user_texts[2]),), third
    assert _settings(first).tools == _requested(turns[0]).tools, first
    assert _top_level(second) == _top_level(first) == _top_level(third), conversation.bodies


def test_second_change_at_turn_three_adds_its_own_item_and_keeps_the_first(gateway: Gateway) -> None:
    """regression: an effort change at turn 2 then a tool added at turn 3 leaves both items at their own turns."""
    turns: Final = (
        _Turn(effort="low", tools=("weather",)),
        _Turn(effort="high", tools=("weather",)),
        _Turn(effort="high", tools=("weather", "time")),
    )
    conversation: Final = _converse(gateway, turns)
    first, second, third = conversation.bodies
    assert _turn_items(first, second, conversation.blobs[0]) == (
        _effort_update("high"),
        _user_item(conversation.user_texts[1]),
    ), second
    announced, user = _update_then_user(second, third, conversation.blobs[1])
    _assert_additional_tools(announced, ("time",))
    assert user == _user_item(conversation.user_texts[2]), third
    assert _top_level(second) == _top_level(first) == _top_level(third), conversation.bodies


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
def test_every_replayed_reasoning_item_carries_the_original_encrypted_content(gateway: Gateway, stream: bool) -> None:
    """guard: reasoning replayed upstream is exactly what the upstream returned, whatever the signature carries."""
    turns: Final = (
        _Turn(effort="low", tools=("weather",)),
        _Turn(effort="high", tools=("weather",)),
        _Turn(effort="high", tools=("weather", "time")),
    )
    conversation: Final = _converse(gateway, turns, stream=stream)
    assert tuple(map(_encrypted_contents, conversation.bodies)) == (
        (),
        conversation.blobs[:1],
        conversation.blobs[:2],
    ), conversation.bodies


@pytest.mark.parametrize(
    ("turns", "rewrite", "replayed"),
    (
        ((_Turn(effort="low"), _Turn(effort="high")), _strip_all, ()),
        ((_Turn(effort="low"), _Turn(effort="high")), _garbled, (0,)),
        ((_Turn(effort="low"), _Turn(effort="high")), _tampered, (0,)),
        (_EFFORT_CHANGE, _strip_second_turn_when_sending_third, (0,)),
        ((_Turn(tools=("weather", "time")), _Turn(tools=("weather",))), _keep, (0,)),
        ((_Turn(tools=("weather",)), _Turn(tools=("weather_in_celsius",))), _keep, (0,)),
        ((_Turn(effort="low"), _Turn(effort=None)), _keep, (0,)),
    ),
    ids=(
        "client-strips-thinking",
        "garbled-record",
        "tampered-record",
        "middle-turn-without-record",
        "tool-removed",
        "tool-definition-changed",
        "thinking-turned-off",
    ),
)
def test_unusable_record_or_inexpressible_change_falls_back_to_current_settings(
    gateway: Gateway, turns: tuple[_Turn, ...], rewrite: Rewrite, replayed: tuple[int, ...]
) -> None:
    """guard (fallback): the last turn writes its own settings top-level and carries no mid-conversation items."""
    conversation: Final = _converse(gateway, turns, rewrite=rewrite)
    last: Final = conversation.bodies[-1]
    assert _settings(last) == _requested(turns[-1]), last
    assert _midconvo_items(last) == (), last
    assert _encrypted_contents(last) == tuple(conversation.blobs[index] for index in replayed), last


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
def test_pre_gpt6_model_keeps_todays_shape_and_plain_signatures(gateway: Gateway, stream: bool) -> None:
    """guard: gpt-5.5 writes a changed effort top-level and returns signatures with no turn settings."""
    conversation: Final = _converse(gateway, _EFFORT_CHANGE, backend=_PRE_GPT6, stream=stream)
    assert tuple(map(_settings, conversation.bodies)) == tuple(map(_requested, _EFFORT_CHANGE)), conversation.bodies
    assert tuple(map(_midconvo_items, conversation.bodies)) == ((), (), ()), conversation.bodies
    assert tuple(map(_records, conversation.contents)) == tuple((f"{_PREFIX}{blob}",) for blob in conversation.blobs), (
        conversation.contents
    )


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
def test_unchanged_settings_keep_todays_shape_on_every_turn(gateway: Gateway, stream: bool) -> None:
    """guard: a GPT-6 conversation that never changes settings sends them top-level and adds no items."""
    turn: Final = _Turn(effort="medium", tools=("weather",), schema=_SCHEMA_A)
    conversation: Final = _converse(gateway, (turn, turn, turn), stream=stream)
    assert tuple(map(_settings, conversation.bodies)) == (_requested(turn),) * 3, conversation.bodies
    assert tuple(map(_midconvo_items, conversation.bodies)) == ((), (), ()), conversation.bodies
    first, second, third = conversation.bodies
    assert _turn_items(first, second, conversation.blobs[0]) == (_user_item(conversation.user_texts[1]),), second
    assert _turn_items(second, third, conversation.blobs[1]) == (_user_item(conversation.user_texts[2]),), third


def test_openai_compatible_host_not_marked_as_openai_dialect_keeps_todays_shape(gateway: Gateway) -> None:
    """guard: a GPT-6 deployment on a host the operator never marked as OpenAI's dialect writes changes top-level."""
    conversation: Final = _converse(gateway, _EFFORT_CHANGE, openai_dialect_host=False)
    assert tuple(map(_settings, conversation.bodies)) == tuple(map(_requested, _EFFORT_CHANGE)), conversation.bodies
    assert tuple(map(_midconvo_items, conversation.bodies)) == ((), (), ()), conversation.bodies
