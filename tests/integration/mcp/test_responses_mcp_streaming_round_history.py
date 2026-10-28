from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final

from integration._support.client import Gateway
from integration._support.mcp import mcp_peer, register_mcp, tool_calls
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
TOOL_ARGUMENTS: Final = '{"a":2,"b":3}'
ANSWER: Final = "All three tool rounds are done."


def _event_frame(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _streamed_response(round_number: int, model: str, tool_name: str) -> Reply:
    response_id: Final = f"resp_mcp_round_{round_number}"
    call_id: Final = f"call_{round_number}"
    is_tool_call: Final = round_number <= 3
    output_item: Final[dict[str, JsonValue]] = (
        {
            "type": "function_call",
            "id": f"fc_{round_number}",
            "call_id": call_id,
            "name": tool_name,
            "arguments": TOOL_ARGUMENTS,
            "status": "completed",
        }
        if is_tool_call
        else {
            "type": "message",
            "id": f"msg_{round_number}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
        }
    )
    added_item: Final[dict[str, JsonValue]] = (
        {**output_item, "arguments": "", "status": "in_progress"}
        if is_tool_call
        else {**output_item, "content": [], "status": "in_progress"}
    )
    item_events: Final[tuple[dict[str, JsonValue], ...]] = (
        (
            {"type": "response.output_item.added", "output_index": 0, "item": added_item},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{round_number}",
                "output_index": 0,
                "delta": TOOL_ARGUMENTS,
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{round_number}",
                "output_index": 0,
                "arguments": TOOL_ARGUMENTS,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": output_item},
        )
        if is_tool_call
        else (
            {"type": "response.output_item.added", "output_index": 0, "item": added_item},
            {
                "type": "response.output_text.delta",
                "item_id": f"msg_{round_number}",
                "output_index": 0,
                "content_index": 0,
                "delta": ANSWER,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": output_item},
        )
    )
    response: Final[dict[str, JsonValue]] = {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [output_item],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.in_progress",
            "response": {**response, "status": "in_progress", "output": []},
        },
        *item_events,
        {"type": "response.completed", "response": response},
    )
    chunks: Final = tuple(
        _event_frame({**event, "sequence_number": sequence_number}) for sequence_number, event in enumerate(events)
    )
    return Reply(content_type="text/event-stream", chunks=chunks)


def _tool_call_history(body: Mapping[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    input_items: Final = body["input"]
    assert isinstance(input_items, list)
    return tuple(
        (str(item["type"]), str(item["call_id"]))
        for item in input_items
        if isinstance(item, dict) and item.get("type") in ("function_call", "function_call_output")
    )


def test_streaming_responses_mcp_follow_up_keeps_every_tool_round(gateway: Gateway) -> None:
    alias: Final = "hist" + uuid.uuid4().hex[:8]
    upstream_requests: Final[list[dict[str, JsonValue]]] = []

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.target.endswith("/responses"), request.target
        request_body: Final = JSON_OBJECT.validate_json(request.body)
        upstream_requests.append(request_body)
        return _streamed_response(len(upstream_requests), str(request_body["model"]), f"{alias}-add")

    with mcp_peer() as peer, wire_server(respond) as wire, gateway.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, alias)
        model: Final = scenario.model(model="openai/responses/gpt-4o-mini", api_base=wire.url + "/v1")
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]})
        peer.drain()
        response: Final = gateway.client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "input": "run each tool round in order",
                "stream": True,
                "tools": [
                    {
                        "type": "mcp",
                        "server_url": "litellm_proxy",
                        "server_label": "litellm",
                        "require_approval": "never",
                    }
                ],
            },
            timeout=90,
        )
        assert response.status_code == 200, response.text
        assert "response.completed" in response.text and ANSWER in response.text, response.text

    assert len(upstream_requests) == 4, upstream_requests
    assert _tool_call_history(upstream_requests[2]) == (
        ("function_call", "call_1"),
        ("function_call_output", "call_1"),
        ("function_call", "call_2"),
        ("function_call_output", "call_2"),
    )
    assert _tool_call_history(upstream_requests[3]) == (
        ("function_call", "call_1"),
        ("function_call_output", "call_1"),
        ("function_call", "call_2"),
        ("function_call_output", "call_2"),
        ("function_call", "call_3"),
        ("function_call_output", "call_3"),
    )
    third_input: Final = upstream_requests[2]["input"]
    fourth_input: Final = upstream_requests[3]["input"]
    assert isinstance(third_input, list) and isinstance(fourth_input, list)
    assert sum(1 for item in third_input if isinstance(item, dict) and item.get("role") == "user") == 1
    assert sum(1 for item in fourth_input if isinstance(item, dict) and item.get("role") == "user") == 1
    assert len(tool_calls(peer.drain())) == 3
