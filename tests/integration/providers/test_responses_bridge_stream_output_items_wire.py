import json
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Final

import openai
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "qwen3-bridge-items"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_QUESTION: Final = "What is the weather in Paris?"
_ANSWER: Final = "Paris is 22 degrees Celsius with clear skies."
_PREFACE: Final = "Checking the weather."
_REASONING: Final = "The user asks for the weather, so the weather tool applies."
_CALL_ID: Final = "call_bridge_items_1"
_ARGUMENTS: Final[dict[str, JsonValue]] = {"city": "Paris"}
_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_DISCOVERY_PROBE: Final = ("GET", "/v1/models")


def _frame(identity: str, delta: dict[str, JsonValue], finish_reason: str | None = None) -> bytes:
    usage: Final = {"usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}} if finish_reason else {}
    body: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _BACKEND,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **usage,
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _sse_reply(identity: str, deltas: Sequence[dict[str, JsonValue]], finish_reason: str) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _frame(identity, {"role": "assistant", "content": ""}),
            *(_frame(identity, delta) for delta in deltas),
            _frame(identity, {}, finish_reason),
            b"data: [DONE]\n\n",
        ),
    )


def _tool_call_delta() -> dict[str, JsonValue]:
    return {
        "tool_calls": [
            {
                "index": 0,
                "id": _CALL_ID,
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps(_ARGUMENTS)},
            }
        ]
    }


def _is_discovery_probe(request: Request) -> bool:
    return (request.method, request.target) == _DISCOVERY_PROBE


@contextmanager
def _vllm_server(respond: Callable[[Request], Reply]) -> Iterator[Wire]:
    with wire_server(
        lambda request: Reply(body=b'{"object":"list","data":[]}') if _is_discovery_probe(request) else respond(request)
    ) as wire:
        yield wire


def _bridged_vllm_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY, use_chat_completions_api=True
    )


def _only_streamed_chat(wire: Wire, *tool_names: str) -> dict[str, JsonValue]:
    received: Final = tuple(request for request in wire.drain() if not _is_discovery_probe(request))
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/chat/completions")]
    assert received[0].headers["authorization"] == f"Bearer {_API_KEY}"
    body: Final = _JSON_OBJECT.validate_json(received[0].body)
    assert body["stream"] is True
    tools: Final = body.get("tools", [])
    assert isinstance(tools, list)
    assert [object_value(object_value(tool)["function"])["name"] for tool in tools] == list(tool_names), tools
    messages: Final = body["messages"]
    assert isinstance(messages, list) and object_value(messages[-1])["content"] == _QUESTION, messages
    return body


def _spend_statuses(model: str) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return [row["status"] for row in rows]


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _raw_events(gateway: Gateway, body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    with gateway.client.stream(
        "POST",
        "/v1/responses",
        json={**body, "cache": _NO_CACHE},
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        return [
            _JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: {")
        ]


def _raw_added_items(events: Sequence[dict[str, JsonValue]]) -> Iterator[tuple[JsonValue, JsonValue]]:
    for event in events:
        if event["type"] == "response.output_item.added":
            yield object_value(event["item"])["type"], event["output_index"]


def _raw_completed_output_types(events: Sequence[dict[str, JsonValue]]) -> list[JsonValue]:
    completed: Final = [event for event in events if event["type"] == "response.completed"]
    assert len(completed) == 1, [event["type"] for event in events]
    output: Final = object_value(completed[0]["response"])["output"]
    assert isinstance(output, list)
    return [object_value(item)["type"] for item in output]


def test_openai_sdk_responses_stream_with_a_tool_call_only_reply_announces_no_message_item(gateway: Gateway) -> None:
    identity: Final = f"chatcmpl-bridge-{uuid.uuid4().hex}"
    reply: Final = _sse_reply(identity, (_tool_call_delta(),), "tool_calls")
    with _vllm_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = _bridged_vllm_model(scenario, wire)
        with _openai_client(gateway).responses.stream(
            model=model,
            input=_QUESTION,
            tools=[_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            store=False,
            extra_body={"cache": _NO_CACHE},
        ) as stream:
            events: Final = list(stream)
            final: Final = stream.get_final_response()
        added: Final = [
            (event.item.type, event.output_index) for event in events if event.type == "response.output_item.added"
        ]
        assert added == [("function_call", 0)], [event.type for event in events]
        assert [item.type for item in final.output] == ["function_call"], final.output
        call: Final = final.output[0]
        assert call.type == "function_call" and call.name == "get_weather" and call.call_id == _CALL_ID
        assert json.loads(call.arguments) == _ARGUMENTS
        assert final.output_text == ""
        created: Final = [event for event in events if event.type == "response.created"]
        assert len(created) == 1 and created[0].response.id == final.id, [event.type for event in events]
        _only_streamed_chat(wire, "get_weather")
        assert _spend_statuses(model) == ["success"]


async def test_async_openai_sdk_responses_stream_text_reply_has_one_message_item_at_index_zero(
    gateway: Gateway,
) -> None:
    identity: Final = f"chatcmpl-bridge-{uuid.uuid4().hex}"
    reply: Final = _sse_reply(identity, ({"content": _ANSWER[:17]}, {"content": _ANSWER[17:]}), "stop")
    with _vllm_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = _bridged_vllm_model(scenario, wire)
        stream: Final = await _async_openai_client(gateway).responses.create(
            model=model, input=_QUESTION, store=False, stream=True, extra_body={"cache": _NO_CACHE}
        )
        events: Final = [event async for event in stream]
        added: Final = [
            (event.item.type, event.output_index) for event in events if event.type == "response.output_item.added"
        ]
        assert added == [("message", 0)], [event.type for event in events]
        done: Final = [(event.item.type, event.output_index) for event in events if event.type == "response.output_item.done"]
        assert done == [("message", 0)], [event.type for event in events]
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == _ANSWER
        completed: Final = [event for event in events if event.type == "response.completed"]
        assert len(completed) == 1 and completed[0].response.output_text == _ANSWER
        assert [item.type for item in completed[0].response.output] == ["message"], completed[0].response.output
        created: Final = [event for event in events if event.type == "response.created"]
        assert len(created) == 1 and created[0].response.id == completed[0].response.id
        _only_streamed_chat(wire)
        assert _spend_statuses(model) == ["success"]


async def test_async_openai_sdk_responses_stream_reasoning_then_text_gets_contiguous_output_indexes(
    gateway: Gateway,
) -> None:
    identity: Final = f"chatcmpl-bridge-{uuid.uuid4().hex}"
    reply: Final = _sse_reply(identity, ({"reasoning_content": _REASONING}, {"content": _ANSWER}), "stop")
    with _vllm_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = _bridged_vllm_model(scenario, wire)
        stream: Final = await _async_openai_client(gateway).responses.create(
            model=model, input=_QUESTION, store=False, stream=True, extra_body={"cache": _NO_CACHE}
        )
        events: Final = [event async for event in stream]
        added: Final = [
            (event.item.type, event.output_index) for event in events if event.type == "response.output_item.added"
        ]
        assert added == [("reasoning", 0), ("message", 1)], [event.type for event in events]
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == _ANSWER
        completed: Final = [event for event in events if event.type == "response.completed"]
        assert len(completed) == 1 and completed[0].response.output_text == _ANSWER
        assert [item.type for item in completed[0].response.output] == ["reasoning", "message"], (
            completed[0].response.output
        )
        _only_streamed_chat(wire)
        assert _spend_statuses(model) == ["success"]


def test_raw_responses_stream_text_then_tool_call_keeps_the_message_first(gateway: Gateway) -> None:
    identity: Final = f"chatcmpl-bridge-{uuid.uuid4().hex}"
    reply: Final = _sse_reply(identity, ({"content": _PREFACE}, _tool_call_delta()), "tool_calls")
    with _vllm_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = _bridged_vllm_model(scenario, wire)
        events: Final = _raw_events(
            gateway, {"model": model, "input": _QUESTION, "tools": [_TOOL], "store": False, "stream": True}
        )
        assert list(_raw_added_items(events)) == [("message", 0), ("function_call", 1)], [e["type"] for e in events]
        assert _raw_completed_output_types(events) == ["message", "function_call"]
        deltas: Final = [event["delta"] for event in events if event["type"] == "response.output_text.delta"]
        assert "".join(str(delta) for delta in deltas) == _PREFACE, deltas
        done_calls: Final = [
            object_value(event["item"])
            for event in events
            if event["type"] == "response.output_item.done" and object_value(event["item"])["type"] == "function_call"
        ]
        assert [(item["name"], item["call_id"]) for item in done_calls] == [("get_weather", _CALL_ID)], done_calls
        _only_streamed_chat(wire, "get_weather")
        assert _spend_statuses(model) == ["success"]
