import json
import uuid
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Final

from integration._support.client import list_value, object_value
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue

MODEL: Final = "gpt-6.1-sol"
PLAN: Final = "I am starting by choosing which city to check."
PICK: Final = "My pick is Tokyo, so I will check its weather."
PATCH_NOTE: Final = "I will create hello.txt containing hi."
ANSWER: Final = "It is raining in Tokyo."

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
PATCH_TOOL: Final[dict[str, JsonValue]] = {
    "type": "custom",
    "custom": {"name": "apply_patch", "description": "Apply a V4A patch to the workspace"},
}


def _message(item_id: str, text: str, phase: str) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning(item_id: str, summary: str) -> dict[str, JsonValue]:
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": summary}],
        "encrypted_content": f"enc_{item_id}",
    }


_WEATHER_CALL: Final[dict[str, JsonValue]] = {
    "type": "function_call",
    "id": "fc_weather",
    "call_id": "call_weather",
    "name": "get_weather",
    "arguments": '{"city":"Tokyo"}',
    "status": "completed",
}
WEATHER_TURN: Final[tuple[dict[str, JsonValue], ...]] = (
    _message("msg_plan", PLAN, "commentary"),
    _reasoning("rs_pick", "Tokyo is often rainy this time of year."),
    _message("msg_pick", PICK, "commentary"),
    _WEATHER_CALL,
)
TWO_THOUGHTS_TURN: Final[tuple[dict[str, JsonValue], ...]] = (
    _reasoning("rs_scan", "Paris and Tokyo are the two candidates."),
    _reasoning("rs_pick", "Tokyo is often rainy this time of year."),
    _message("msg_pick", PICK, "commentary"),
    _WEATHER_CALL,
)
CALL_ONLY_TURN: Final[tuple[dict[str, JsonValue], ...]] = (_WEATHER_CALL,)
PATCH_TURN: Final[tuple[dict[str, JsonValue], ...]] = (
    _message("msg_patch", PATCH_NOTE, "commentary"),
    {
        "type": "custom_tool_call",
        "id": "ctc_patch",
        "call_id": "call_patch",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch",
        "status": "completed",
    },
)
WEATHER_REPLAY: Final = (
    ("message", "assistant", "commentary", PLAN),
    ("reasoning", "rs_pick"),
    ("message", "assistant", "commentary", PICK),
    ("function_call", "call_weather", "get_weather"),
    ("function_call_output", "call_weather"),
)
TWO_THOUGHTS_REPLAY: Final = (
    ("reasoning", "rs_scan"),
    ("reasoning", "rs_pick"),
    ("message", "assistant", "commentary", PICK),
    ("function_call", "call_weather", "get_weather"),
    ("function_call_output", "call_weather"),
)


def _response(identity: str, output: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1789788253,
        "status": "completed",
        "model": MODEL,
        "output": list(output),
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _event(payload: dict[str, JsonValue]) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _opened(item: dict[str, JsonValue]) -> dict[str, JsonValue]:
    match item["type"]:
        case "message":
            return {**item, "status": "in_progress", "content": []}
        case "function_call":
            return {**item, "status": "in_progress", "arguments": ""}
        case _:
            return item


def _deltas(index: int, item: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    match item["type"]:
        case "message":
            text: Final = object_value(list_value(item["content"])[0])["text"]
            return (
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        case "function_call":
            return (
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item["arguments"],
                },
            )
        case _:
            return ()


def _item_events(index: int, item: dict[str, JsonValue]) -> tuple[bytes, ...]:
    return (
        _event({"type": "response.output_item.added", "output_index": index, "item": _opened(item)}),
        *(_event(delta) for delta in _deltas(index, item)),
        _event({"type": "response.output_item.done", "output_index": index, "item": item}),
    )


def _stream(identity: str, output: Sequence[dict[str, JsonValue]]) -> tuple[bytes, ...]:
    response: Final = _response(identity, output)
    return (
        _event({"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}),
        *chain.from_iterable(_item_events(index, item) for index, item in enumerate(output)),
        _event({"type": "response.completed", "response": response}),
    )


def _answered(body: dict[str, JsonValue]) -> bool:
    return any(
        object_value(item).get("type") in {"function_call_output", "custom_tool_call_output"}
        for item in list_value(body["input"])
    )


def responses_upstream(turn_one: Sequence[dict[str, JsonValue]]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.target == "/responses", request.target
        body: Final = object_value(json.loads(request.body))
        identity: Final = uuid.uuid4().hex
        output: Final = (_message("msg_answer", ANSWER, "final_answer"),) if _answered(body) else turn_one
        if body.get("stream"):
            return Reply(content_type="text/event-stream", chunks=_stream(identity, output))
        return Reply(body=json.dumps(_response(identity, output)).encode())

    return respond


def _text(content: JsonValue) -> JsonValue:
    if isinstance(content, str):
        return content
    return "".join(str(object_value(part).get("text")) for part in list_value(content))


def _shape(item: JsonValue) -> tuple[JsonValue, ...]:
    fields: Final = object_value(item)
    match fields.get("type"):
        case "message":
            return ("message", fields.get("role"), fields.get("phase"), _text(fields.get("content")))
        case "reasoning":
            return ("reasoning", fields.get("id"))
        case "function_call" | "custom_tool_call" as kind:
            return (kind, fields.get("call_id"), fields.get("name"))
        case kind:
            return (kind, fields.get("call_id"))


def posted_inputs(wire: Wire) -> tuple[tuple[tuple[JsonValue, ...], ...], ...]:
    return tuple(
        tuple(_shape(item) for item in list_value(object_value(json.loads(request.body))["input"]))
        for request in wire.drain()
        if request.method == "POST"
    )
