import json
import os
import uuid
from pathlib import Path
from typing import Final

import openai
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-azure-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOL: Final = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_RESPONSES_TOOL: Final = {**_TOOL, "strict": None}


def _spend_row(response_id: str) -> dict[str, JsonValue]:
    assert response_id, "Caller response had no id"
    rows: Final = eventually(
        lambda: read_rows(
            """
            SELECT
                request_id,
                status,
                CASE
                    WHEN left(request_id, 16) = 'resp_bGl0ZWxsbTp'
                    THEN convert_from(
                        decode(
                            translate(
                                regexp_replace(substr(request_id, 6), '_cache_hit[0-9]+[.][0-9]+$', ''),
                                '-_',
                                '+/'
                            ),
                            'base64'
                        ),
                        'UTF8'
                    )
                    ELSE request_id
                END AS decoded_request_id
            FROM "LiteLLM_SpendLogs"
            WHERE request_id=%s
               OR CASE
                    WHEN left(request_id, 16) = 'resp_bGl0ZWxsbTp'
                    THEN split_part(
                        convert_from(
                            decode(
                                translate(
                                    regexp_replace(substr(request_id, 6), '_cache_hit[0-9]+[.][0-9]+$', ''),
                                    '-_',
                                    '+/'
                                ),
                                'base64'
                            ),
                            'UTF8'
                        ),
                        ';response_id:',
                        2
                    ) = %s
                    ELSE false
                  END
            """,
            (response_id, response_id),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    decoded_request_id: Final = rows[0]["decoded_request_id"]
    assert rows[0]["request_id"] == response_id or (
        isinstance(decoded_request_id, str) and decoded_request_id.endswith(f";response_id:{response_id}")
    ), rows
    return rows[0]


def _record_audit_cell(
    row_id: str,
    node_id: str,
    response_id: str,
    target: str,
    spend_found: bool,
) -> None:
    results_dir: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    assert results_dir is not None, "INTEGRATION_RESULTS_DIR is required for audit cell evidence"
    artifact: Final = Path(results_dir) / "audit-cells.jsonl"
    record: Final = {
        "row_id": row_id,
        "node_id": node_id,
        "response_id": response_id,
        "upstream_path": target,
        "spend_row_found": spend_found,
        "leg": os.environ.get("LITAUDIT_LEG", "head"),
    }
    with artifact.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _responses_reply(identity: str) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Weather marker-{identity}.", "annotations": []}],
                },
                {
                    "type": "function_call",
                    "id": f"fc_{identity}",
                    "call_id": f"call_{identity}",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                    "status": "completed",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def _responses_stream(identity: str) -> tuple[bytes, ...]:
    text: Final = f"Weather marker-{identity}."
    events: Final = (
        {
            "type": "response.created",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "in_progress",
                "model": "gpt-6-sol",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [_RESPONSES_TOOL],
                "output": [],
            },
            "sequence_number": 0,
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "id": f"msg_{identity}",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": text,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "id": f"msg_{identity}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "sequence_number": 4,
            "item": {
                "id": f"fc_{identity}",
                "type": "function_call",
                "status": "in_progress",
                "call_id": f"call_{identity}",
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": f"fc_{identity}",
            "output_index": 1,
            "sequence_number": 5,
            "delta": '{"city":"Paris"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "sequence_number": 6,
            "item": {
                "id": f"fc_{identity}",
                "type": "function_call",
                "status": "completed",
                "call_id": f"call_{identity}",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [_RESPONSES_TOOL],
                "output": [
                    {
                        "id": f"msg_{identity}",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                    {
                        "id": f"fc_{identity}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": f"call_{identity}",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 15,
                },
            },
            "sequence_number": 7,
        },
    )
    return tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def test_b6_azure_gpt_6_native_responses_returns_message_and_function_call(gateway: Gateway) -> None:
    identity: Final = uuid.uuid4().hex
    prompt: Final = f"Weather marker-{identity}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == prompt
        assert body["tools"] == [_TOOL]
        assert body["stream"] is False
        assert "reasoning" not in body
        return Reply(body=_responses_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": prompt,
                "tools": [_TOOL],
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "B6",
            "test_b6_azure_gpt_6_native_responses_returns_message_and_function_call",
            str(body["id"]),
            target,
            True,
        )
        assert body["output"] == [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "phase": None,
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Weather marker-{identity}.",
                        "annotations": [],
                        "logprobs": None,
                    }
                ],
            },
            {
                "type": "function_call",
                "id": f"fc_{identity}",
                "call_id": f"call_{identity}",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
                "status": "completed",
                "namespace": None,
            },
        ], response.text
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]


async def test_b7_azure_gpt_6_native_responses_stream_events_pass_through(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: streamed /v1/responses spend row stores the routing-encoded id, not the id the client got (LIT-9194)"
    )
    identity: Final = uuid.uuid4().hex
    prompt: Final = f"Weather marker-{identity}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == prompt
        assert body["tools"] == [_TOOL]
        assert body["stream"] is True
        assert "reasoning" not in body
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
        )
        async with openai.AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            stream: Final = await client.responses.create(
                model=model,
                input=prompt,
                tools=[_TOOL],
                stream=True,
            )
            events: Final = tuple([event async for event in stream])
        event_dump: Final = [event.model_dump(exclude_none=True) for event in events]
        created: Final = _JSON_OBJECT.validate_python(next(event.model_dump(exclude_none=True) for event in events))
        response_id: Final = str(_JSON_OBJECT.validate_python(created["response"])["id"])
        try:
            spend: Final = _spend_row(response_id)
        except AssertionError:
            _record_audit_cell(
                "B7",
                "test_b7_azure_gpt_6_native_responses_stream_events_pass_through",
                response_id,
                target,
                False,
            )
            raise
        _record_audit_cell(
            "B7",
            "test_b7_azure_gpt_6_native_responses_stream_events_pass_through",
            response_id,
            target,
            True,
        )
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]
        assert event_dump == [
            {
                "type": "response.created",
                "model": str(model),
                "sequence_number": 0,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": 1,
                    "status": "in_progress",
                    "model": "gpt-6-sol",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [_TOOL],
                    "output": [],
                },
            },
            {
                "type": "response.output_item.added",
                "model": str(model),
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "response.output_text.delta",
                "model": str(model),
                "sequence_number": 2,
                "item_id": f"msg_{identity}",
                "output_index": 0,
                "content_index": 0,
                "delta": f"Weather marker-{identity}.",
            },
            {
                "type": "response.output_item.done",
                "model": str(model),
                "sequence_number": 3,
                "output_index": 0,
                "item": {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Weather marker-{identity}.",
                            "annotations": [],
                        }
                    ],
                },
            },
            {
                "type": "response.output_item.added",
                "model": str(model),
                "sequence_number": 4,
                "output_index": 1,
                "item": {
                    "id": f"fc_{identity}",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": f"call_{identity}",
                    "name": "get_weather",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "model": str(model),
                "sequence_number": 5,
                "item_id": f"fc_{identity}",
                "output_index": 1,
                "delta": '{"city":"Paris"}',
            },
            {
                "type": "response.output_item.done",
                "model": str(model),
                "sequence_number": 6,
                "output_index": 1,
                "item": {
                    "id": f"fc_{identity}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": f"call_{identity}",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
            },
            {
                "type": "response.completed",
                "model": str(model),
                "sequence_number": 7,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [_TOOL],
                    "output": [
                        {
                            "id": f"msg_{identity}",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": f"Weather marker-{identity}.",
                                    "annotations": [],
                                }
                            ],
                        },
                        {
                            "id": f"fc_{identity}",
                            "type": "function_call",
                            "status": "completed",
                            "call_id": f"call_{identity}",
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 15,
                        "cost": 0.0,
                    },
                },
            },
        ], event_dump
