from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Final

from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery
from integration._support.wire import Reply, Request
from pydantic import JsonValue


WEATHER: Final = "get_weather"
TIME: Final = "get_time"
ONLY_SEARCH: Final = "only-search"
TWO_SEARCHES: Final = "two-searches"
REJECT_REPLAY: Final = "reject-replay"
REJECTED: Final = "scripted: the replayed tool_search_output was rejected"
TOOL_SEARCH: Final[dict[str, JsonValue]] = {"type": "tool_search"}
PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
REPLAY_TYPES: Final = frozenset({"reasoning", "tool_search_call", "tool_search_output", "function_call"})


def function(name: str) -> dict[str, JsonValue]:
    return {"name": name, "description": f"Look up {name}", "parameters": PARAMETERS}


def deferred(name: str) -> dict[str, JsonValue]:
    return {"type": "function", "function": function(name), "defer_loading": True}


def chat_tools(*names: str) -> list[JsonValue]:
    return [TOOL_SEARCH, *(deferred(name) for name in names)]


def loaded(name: str) -> dict[str, JsonValue]:
    return {"type": "function", **function(name), "strict": False, "defer_loading": True}


def namespace(name: str, *members: str) -> dict[str, JsonValue]:
    return {"type": "namespace", "name": name, "description": f"{name} tools", "tools": [loaded(m) for m in members]}


def reasoning(marker: str, label: str) -> dict[str, JsonValue]:
    return {
        "id": f"rs_{label}_{marker}",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": f"enc-{label}-{marker}",
    }


def search_call(marker: str, label: str = "a") -> dict[str, JsonValue]:
    return {
        "id": f"tsc_{label}_{marker}",
        "type": "tool_search_call",
        "arguments": {"query": f"tools {label}"},
        "execution": "server",
        "status": "completed",
    }


def search_output(marker: str, tools: Sequence[JsonValue], label: str = "a") -> dict[str, JsonValue]:
    return {
        "id": f"tso_{label}_{marker}",
        "type": "tool_search_output",
        "execution": "server",
        "status": "completed",
        "tools": list(tools),
    }


def call_id(marker: str, name: str = WEATHER) -> str:
    return f"call_{name}_{marker}"


def replayed_call(marker: str, name: str = WEATHER) -> dict[str, JsonValue]:
    return {
        "type": "function_call",
        "call_id": call_id(marker, name),
        "name": name,
        "arguments": '{"city": "Paris"}',
        "namespace": name,
    }


def call_item(marker: str, name: str = WEATHER) -> dict[str, JsonValue]:
    return {**replayed_call(marker, name), "id": f"fc_{name}_{marker}", "status": "completed"}


def searched(marker: str, label: str, name: str) -> tuple[dict[str, JsonValue], ...]:
    return (reasoning(marker, label), search_call(marker, label), search_output(marker, [loaded(name)], label))


def turn_one_output(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return (*searched(marker, "a", WEATHER), reasoning(marker, "b"), call_item(marker))


def turn_two_output(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return (*searched(marker, "c", TIME), call_item(marker, TIME))


def only_search_output(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return searched(marker, "a", WEATHER)


def on_the_wire(item: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**item, "created_by": None} if item.get("type") == "tool_search_call" else dict(item)


def _message_item(identity: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response(identity: str, model: str, output: Sequence[JsonValue], status: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": model,
        "output": list(output),
        "usage": rv.USAGE,
    }


def _item_events(index: int, item: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    added: Final[dict[str, JsonValue]] = {"type": "response.output_item.added", "output_index": index, "item": {**item}}
    done: Final[dict[str, JsonValue]] = {"type": "response.output_item.done", "output_index": index, "item": {**item}}
    if item.get("type") == "function_call":
        return (
            {**added, "item": {**item, "arguments": "", "status": "in_progress"}},
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "item_id": item["id"],
                "delta": item["arguments"],
            },
            done,
        )
    if item.get("type") == "message":
        contents: Final = rv.ITEMS.validate_python(item["content"])
        return (
            {**added, "item": {**item, "content": [], "status": "in_progress"}},
            {
                "type": "response.output_text.delta",
                "output_index": index,
                "item_id": item["id"],
                "content_index": 0,
                "delta": contents[0]["text"],
            },
            done,
        )
    return (added, done)


def stream_chunks(identity: str, model: str, output: Sequence[Mapping[str, JsonValue]]) -> tuple[bytes, ...]:
    created: Final[dict[str, JsonValue]] = {
        "type": "response.created",
        "response": _response(identity, model, [], "in_progress"),
    }
    items: Final = tuple(chain.from_iterable(_item_events(index, item) for index, item in enumerate(output)))
    completed: Final[dict[str, JsonValue]] = {
        "type": "response.completed",
        "response": _response(identity, model, [dict(item) for item in output], "completed"),
    }
    events: Final = (created, *items, completed)
    return tuple(rv.sse({**event, "sequence_number": number}) for number, event in enumerate(events))


def reply(
    identity: str, model: str, output: Sequence[Mapping[str, JsonValue]], *, stream: bool, pause: float = 0
) -> Reply:
    if stream:
        return Reply(
            content_type="text/event-stream",
            chunks=stream_chunks(identity, model, output),
            pause_between_chunks=pause,
        )
    return Reply(body=json.dumps(_response(identity, model, [dict(item) for item in output], "completed")).encode())


def _count(items: Sequence[Mapping[str, JsonValue]], kind: str) -> int:
    return sum(1 for item in items if item.get("type") == kind)


def _scripted_output(text: str, marker: str, items: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    results: Final = _count(items, "function_call_output")
    if TWO_SEARCHES in text and results == 1:
        return tuple(on_the_wire(item) for item in turn_two_output(marker))
    if results or (ONLY_SEARCH in text and _count(items, "tool_search_output")):
        return (_message_item(f"{marker}-{results}", rv.answer(marker)),)
    chosen: Final = only_search_output(marker) if ONLY_SEARCH in text else turn_one_output(marker)
    return tuple(on_the_wire(item) for item in chosen)


def scripted_with_pause(pause: float) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = pcb.body_of(request)
        text: Final = request.body.decode()
        marker: Final = rv.newest_marker(text) or uuid.uuid4().hex
        items: Final = pcb.input_items(request)
        if REJECT_REPLAY in text and _count(items, "tool_search_output"):
            return rv.error(400, REJECTED, "invalid_request_error")
        return reply(
            f"resp_{marker}-{uuid.uuid4().hex}",
            string_value(body["model"]),
            [rv.JSON_OBJECT.validate_python(item) for item in _scripted_output(text, marker, items)],
            stream=body.get("stream") is True,
            pause=pause,
        )

    return answering_model_discovery(respond)


scripted: Final = scripted_with_pause(0)


def replayed(request: Request) -> list[dict[str, JsonValue]]:
    return [item for item in pcb.input_items(request) if item.get("type") in REPLAY_TYPES]


def first_message(text: str) -> dict[str, JsonValue]:
    choices: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(text)["choices"])
    return rv.JSON_OBJECT.validate_python(choices[0]["message"])


def provider_fields(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    fields: Final = value.get("provider_specific_fields")
    return rv.JSON_OBJECT.validate_python(fields) if isinstance(fields, dict) else {}


def tool_calls(message: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return rv.ITEMS.validate_python(message.get("tool_calls") or [])


def namespaces(message: Mapping[str, JsonValue]) -> list[JsonValue]:
    return [provider_fields(call).get("namespace") for call in tool_calls(message)]


def chunks_of(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        rv.JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )


def _deltas(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    choices: Final = tuple(rv.ITEMS.validate_python(chunk.get("choices") or []) for chunk in chunks)
    return tuple(rv.JSON_OBJECT.validate_python(found[0].get("delta") or {}) for found in choices if found)


def _merged(left: Mapping[str, JsonValue], right: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**left, **{key: value for key, value in right.items() if value is not None}}


@dataclass(frozen=True, slots=True)
class _CallParts:
    index: int
    id: str | None
    name: str | None
    arguments: str
    fields: Mapping[str, JsonValue]


def _call_parts(delta: Mapping[str, JsonValue]) -> tuple[_CallParts, ...]:
    calls: Final = rv.ITEMS.validate_python(delta.get("tool_calls") or [])
    functions: Final = tuple(rv.JSON_OBJECT.validate_python(call.get("function") or {}) for call in calls)
    return tuple(
        _CallParts(
            index=int(str(call.get("index") or 0)),
            id=call["id"] if isinstance(call.get("id"), str) else None,
            name=fn["name"] if isinstance(fn.get("name"), str) else None,
            arguments=str(fn.get("arguments") or ""),
            fields=_merged(provider_fields(fn), provider_fields(call)),
        )
        for call, fn in zip(calls, functions, strict=True)
    )


def _rebuilt_call(parts: Sequence[_CallParts]) -> dict[str, JsonValue]:
    fields: Final = reduce(_merged, (part.fields for part in parts), {})
    ids: Final = tuple(part.id for part in parts if part.id)
    names: Final = tuple(part.name for part in parts if part.name)
    return {
        "id": ids[0],
        "type": "function",
        "function": {"name": names[0], "arguments": "".join(part.arguments for part in parts)},
        **({"provider_specific_fields": fields} if fields else {}),
    }


def assistant_from_stream(chunks: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    deltas: Final = _deltas(chunks)
    parts: Final = tuple(chain.from_iterable(_call_parts(delta) for delta in deltas))
    indexes: Final = sorted({part.index for part in parts})
    calls: Final[list[JsonValue]] = [
        _rebuilt_call([part for part in parts if part.index == index]) for index in indexes
    ]
    fields: Final = reduce(_merged, (provider_fields(delta) for delta in deltas), {})
    reasoning_items: Final[list[JsonValue]] = list(
        chain.from_iterable(rv.ITEMS.validate_python(delta.get("reasoning_items") or []) for delta in deltas)
    )
    content: Final = "".join(str(delta.get("content") or "") for delta in deltas)
    return {
        "role": "assistant",
        "content": content or None,
        **({"tool_calls": calls} if calls else {}),
        **({"provider_specific_fields": fields} if fields else {}),
        **({"reasoning_items": reasoning_items} if reasoning_items else {}),
    }


def tool_result(marker: str, name: str = WEATHER) -> dict[str, JsonValue]:
    return {"role": "tool", "tool_call_id": call_id(marker, name), "content": "sunny"}


def spend_rows_of(model: str, response_id: str) -> list[dict[str, JsonValue]]:
    rows: Final = read_rows(
        'SELECT request_id, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)
    )
    return [row for row in rows if rv.same_response(string_value(row["request_id"]), response_id)]
