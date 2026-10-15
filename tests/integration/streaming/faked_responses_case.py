import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Final

import httpx
from anthropic.types import ToolParam as AnthropicToolParam
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.responses import FunctionToolParam
from pydantic import JsonValue, TypeAdapter

API_KEY: Final = "synthetic-faked-stream-key"
AZURE_API_VERSION: Final = "2025-03-01-preview"
CALL_ID: Final = "x-litellm-call-id"
NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
FAKED_STREAM: Final[dict[str, JsonValue]] = {"supports_native_streaming": False}
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
QUESTION: Final = "What is the weather in Paris?"
ANSWER: Final = "Paris is 22 degrees Celsius with clear skies."
ARGUMENTS: Final = json.dumps({"city": "Paris", "unit": "celsius"})
SECOND_ARGUMENTS: Final = json.dumps({"city": "Lyon", "days": 3})
SQL: Final = "SELECT city FROM weather WHERE celsius > 20"
REFUSAL: Final = "I cannot share the forecast for that place."
SUMMARIES: Final = ("First, find the city.", "Then, look up its weather.")
RESPONSES_TOOLS: Final[list[JsonValue]] = [
    {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "get_forecast", "parameters": {"type": "object", "properties": {}}},
]
CHAT_TOOLS: Final[list[JsonValue]] = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_forecast", "parameters": {"type": "object", "properties": {}}}},
]
ANTHROPIC_TOOLS: Final[list[JsonValue]] = [
    {"name": "get_weather", "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_forecast", "input_schema": {"type": "object", "properties": {}}},
]

_TOOL_NAMES: Final = ("get_weather", "get_forecast")
RESPONSES_TOOL_PARAMS: Final[list[FunctionToolParam]] = [
    {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}, "strict": False}
    for name in _TOOL_NAMES
]
CHAT_TOOL_PARAMS: Final[list[ChatCompletionFunctionToolParam]] = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
    for name in _TOOL_NAMES
]
ANTHROPIC_TOOL_PARAMS: Final[list[AnthropicToolParam]] = [
    {"name": name, "input_schema": {"type": "object", "properties": {}}} for name in _TOOL_NAMES
]


def reasoning_item(identity: str, summaries: Sequence[str] = SUMMARIES) -> dict[str, JsonValue]:
    return {
        "type": "reasoning",
        "id": f"rs_{identity}",
        "summary": [{"type": "summary_text", "text": text} for text in summaries],
    }


def message_item(identity: str, text: str = ANSWER) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": f"msg_{identity}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def refusal_item(identity: str, refusal: str = REFUSAL) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": f"msg_refusal_{identity}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": refusal}],
    }


def function_call_item(
    identity: str, arguments: JsonValue = ARGUMENTS, name: str = "get_weather"
) -> dict[str, JsonValue]:
    return {
        "type": "function_call",
        "id": f"fc_{name}_{identity}",
        "call_id": f"call_{name}_{identity}",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def custom_tool_call_item(identity: str, tool_input: str = SQL) -> dict[str, JsonValue]:
    return {
        "type": "custom_tool_call",
        "id": f"ctc_{identity}",
        "call_id": f"call_sql_{identity}",
        "name": "run_sql",
        "input": tool_input,
    }


def every_item_kind(identity: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        reasoning_item(identity),
        message_item(identity),
        function_call_item(identity),
        custom_tool_call_item(identity),
        refusal_item(identity),
    )


def response_body(identity: str, output: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "faked-stream-backend",
        "output": [dict(item) for item in output],
        "usage": {"input_tokens": 12, "output_tokens": 30, "total_tokens": 42},
    }


def answering(output: Callable[[str], Sequence[Mapping[str, JsonValue]]]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        identity: Final = uuid.uuid4().hex[:16]
        return Reply(body=json.dumps(response_body(identity, output(identity))).encode())

    return respond


def faked_model(scenario: Scenario, wire: Wire, model: str = "openai/faked-stream-backend") -> str:
    return scenario.model(model=model, api_base=wire.url + "/v1", api_key=API_KEY, model_info=FAKED_STREAM)


def faked_azure_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model="azure/responses/faked-stream-deployment",
        api_base=wire.url,
        api_key=API_KEY,
        api_version=AZURE_API_VERSION,
        model_info=FAKED_STREAM,
    )


def posted_bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(request.body) for request in wire.drain() if request.method == "POST")


def assert_one_unstreamed_upstream_call(wire: Wire, target: str) -> dict[str, JsonValue]:
    received: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert [request.target for request in received] == [target], [request.target for request in received]
    body: Final = JSON_OBJECT.validate_json(received[0].body)
    assert body.get("stream") is not True, body
    return body


def data_events(lines: Iterator[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(line.removeprefix("data: ")) for line in lines if line.startswith("data: {"))


def streamed(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], key: str | None = None
) -> tuple[httpx.Headers, tuple[dict[str, JsonValue], ...]]:
    with gateway.client.stream(
        "POST",
        path,
        json={**body, "stream": True},
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}", "anthropic-version": "2023-06-01"},
        timeout=120,
    ) as response:
        assert response.status_code == 200, response.read()
        return response.headers, data_events(response.iter_lines())


def stream_events(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], key: str | None = None
) -> tuple[str, tuple[dict[str, JsonValue], ...]]:
    headers, events = streamed(gateway, path, body, key)
    return headers[CALL_ID], events


def responses_request(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": f"{QUESTION} {uuid.uuid4().hex}", "tools": RESPONSES_TOOLS, **extra}


def joined(events: Sequence[Mapping[str, JsonValue]], kind: str, output_index: int, **match: JsonValue) -> str:
    return "".join(string_value(event["delta"]) for event in events if _matches(event, kind, output_index, match))


def _matches(event: Mapping[str, JsonValue], kind: str, output_index: int, match: Mapping[str, JsonValue]) -> bool:
    return (
        event["type"] == kind
        and event.get("output_index") == output_index
        and all(event.get(name) == value for name, value in match.items())
    )


def _of_kind(events: Sequence[Mapping[str, JsonValue]], kind: str) -> Iterator[dict[str, JsonValue]]:
    for event in events:
        if event["type"] == kind:
            yield dict(event)


def items_by_index(events: Sequence[Mapping[str, JsonValue]], kind: str) -> dict[int, dict[str, JsonValue]]:
    return {int(str(event["output_index"])): object_value(event["item"]) for event in _of_kind(events, kind)}


def _announced_parts(
    events: Sequence[Mapping[str, JsonValue]], kind: str, output_index: int
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        object_value(event["part"]) for event in _of_kind(events, kind) if event["output_index"] == output_index
    )


def _rebuilt_part(
    events: Sequence[Mapping[str, JsonValue]], output_index: int, content_index: int, part: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    match part.get("type"):
        case "output_text":
            text: Final = joined(events, "response.output_text.delta", output_index, content_index=content_index)
            return {**part, "text": string_value(part["text"]) + text}
        case "refusal":
            refusal: Final = joined(events, "response.refusal.delta", output_index, content_index=content_index)
            return {**part, "refusal": string_value(part["refusal"]) + refusal}
        case _:
            return dict(part)


def _rebuilt_summary(
    events: Sequence[Mapping[str, JsonValue]], output_index: int, summary_index: int, part: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    text: Final = joined(events, "response.reasoning_summary_text.delta", output_index, summary_index=summary_index)
    return {**part, "text": string_value(part["text"]) + text}


def _rebuilt_item(
    events: Sequence[Mapping[str, JsonValue]], output_index: int, added: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    match added.get("type"):
        case "message":
            parts: Final = (
                *map(object_value, list_value(added["content"])),
                *_announced_parts(events, "response.content_part.added", output_index),
            )
            return {
                **added,
                "content": [_rebuilt_part(events, output_index, index, part) for index, part in enumerate(parts)],
            }
        case "reasoning":
            summaries: Final = (
                *map(object_value, list_value(added["summary"])),
                *_announced_parts(events, "response.reasoning_summary_part.added", output_index),
            )
            return {
                **added,
                "summary": [
                    _rebuilt_summary(events, output_index, index, part) for index, part in enumerate(summaries)
                ],
            }
        case "function_call":
            arguments: Final = joined(events, "response.function_call_arguments.delta", output_index)
            return {**added, "arguments": string_value(added["arguments"]) + arguments}
        case "custom_tool_call":
            tool_input: Final = joined(events, "response.custom_tool_call_input.delta", output_index)
            return {**added, "input": string_value(added["input"]) + tool_input}
        case "mcp_call":
            mcp_arguments: Final = joined(events, "response.mcp_call_arguments.delta", output_index)
            return {**added, "arguments": string_value(added["arguments"]) + mcp_arguments}
        case _:
            return dict(added)


def rebuilt(events: Sequence[Mapping[str, JsonValue]]) -> dict[int, dict[str, JsonValue]]:
    return {
        index: _rebuilt_item(events, index, added)
        for index, added in items_by_index(events, "response.output_item.added").items()
    }


def values(item: Mapping[str, JsonValue]) -> JsonValue:
    match item.get("type"):
        case "message":
            return [
                [string_value(part["type"]), part.get("text", part.get("refusal"))]
                for part in map(object_value, list_value(item["content"]))
            ]
        case "reasoning":
            return [part["text"] for part in map(object_value, list_value(item["summary"]))]
        case "function_call" | "mcp_call":
            return [item["name"], item["arguments"]]
        case "custom_tool_call":
            return [item["name"], item["input"]]
        case _:
            return json.dumps(item, sort_keys=True)


def completed_output(events: Sequence[Mapping[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    completed: Final = tuple(_of_kind(events, "response.completed"))
    assert len(completed) == 1, [event["type"] for event in events]
    return tuple(map(object_value, list_value(object_value(completed[0]["response"])["output"])))


def assert_each_value_arrives_once(
    events: Sequence[Mapping[str, JsonValue]], expected: Sequence[Mapping[str, JsonValue]]
) -> None:
    wanted: Final = [values(item) for item in expected]
    seen_by_client: Final = rebuilt(events)
    done: Final = items_by_index(events, "response.output_item.done")
    assert [values(seen_by_client[index]) for index in sorted(seen_by_client)] == wanted, [
        event["type"] for event in events
    ]
    assert [values(done[index]) for index in sorted(done)] == wanted
    assert [values(item) for item in completed_output(events)] == wanted


def spend_statuses(call_id: str) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return [row["status"] for row in rows]


def http_client(gateway: Gateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=120, trust_env=False)
