import json
import os
import uuid
from pathlib import Path
from typing import Final

import anthropic
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-6-sol"
_API_KEY: Final = "synthetic-openai-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOL_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "include_forecast": {"type": "boolean"},
    },
    "required": ["city"],
}
_WEATHER_TOOL_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def _responses_user_input(identity: str) -> list[JsonValue]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"What is the weather in Paris? {identity}"}],
        }
    ]


def _spend_row(response_id: str) -> dict[str, JsonValue]:
    assert response_id, "Caller response had no id"
    observed: Final = eventually(
        lambda: (
            response_id,
            read_rows(
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
        ),
        lambda value: len(value[1]) == 1,
        seconds=70,
    )
    rows: Final = observed[1]
    decoded_request_id: Final = rows[0]["decoded_request_id"]
    assert rows[0]["request_id"] == response_id or (
        isinstance(decoded_request_id, str) and decoded_request_id.endswith(f";response_id:{response_id}")
    ), rows
    return rows[0]


def _record_audit_cell(row_id: str, node_id: str, response_id: str, spend_found: bool, path: str) -> None:
    results_dir: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    assert results_dir is not None, "INTEGRATION_RESULTS_DIR is required for audit cell evidence"
    artifact: Final = Path(results_dir) / "audit-cells.jsonl"
    record: Final = {
        "row_id": row_id,
        "node_id": node_id,
        "response_id": response_id,
        "upstream_path": path,
        "spend_row_found": spend_found,
        "leg": os.environ.get("LITAUDIT_LEG", "head"),
    }
    with artifact.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _responses_reply(identity: str, model_name: str = "gpt-6-sol") -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": model_name,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Let me check the weather.",
                            "annotations": [],
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
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def _responses_sse(identity: str) -> tuple[bytes, ...]:
    message_id: Final = f"msg_{identity}"
    function_id: Final = f"fc_{identity}"
    call_id: Final = f"call_{identity}"
    text: Final = "Let me check the weather."
    events: Final = (
        {
            "type": "response.created",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "in_progress",
                "model": "gpt-6-sol",
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": function_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": function_id,
            "output_index": 1,
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": function_id,
            "output_index": 1,
            "delta": '"Paris"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "id": function_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
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
                "output": [
                    {
                        "id": message_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                    {
                        "id": function_id,
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )
    return tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)


@pytest.mark.covers("providers.anthropic_messages_bridge.optional_tool_properties_stay_optional_on_the_wire")
def test_messages_tool_with_optional_properties_reaches_openai_responses_non_strict(gateway: Gateway) -> None:
    identity: Final = f"messages-optional-tool-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = json.loads(request.body)
        assert body["model"] == _BACKEND, body
        assert identity in json.dumps(body["input"]), body["input"]
        assert body["reasoning"] == {"effort": "low"}, body
        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "strict": False,
                "description": "Current weather for a city",
                "parameters": _WEATHER_TOOL_SCHEMA,
            }
        ], body["tools"]
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": _BACKEND,
                    "output": [
                        {
                            "type": "function_call",
                            "id": f"fc_{identity}",
                            "call_id": f"call_{identity}",
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 30, "output_tokens": 9, "total_tokens": 39},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "input_schema": _WEATHER_TOOL_SCHEMA,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        body: Final = response.json()
        assert body["stop_reason"] == "tool_use", response.text
        assert body["content"] == [
            {
                "type": "tool_use",
                "id": f"call_{identity}",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ], response.text


def test_b3_azure_gpt_6_anthropic_messages_sync_returns_text_and_tool_use(gateway: Gateway) -> None:
    identity: Final = f"b3-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == _responses_user_input(identity)
        assert "reasoning" not in body
        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "strict": None,
                "parameters": _WEATHER_TOOL_SCHEMA,
            }
        ]
        return Reply(body=_responses_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url).rstrip("/"),
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            message: Final = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a city.",
                        "input_schema": _WEATHER_TOOL_SCHEMA,
                    }
                ],
            )
        message_body: Final = message.model_dump(exclude_none=True)
        spend: Final = _spend_row(message.id)
        _record_audit_cell(
            "B3",
            "test_b3_azure_gpt_6_anthropic_messages_sync_returns_text_and_tool_use",
            message.id,
            True,
            target,
        )
        assert message_body["content"] == [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": f"call_{identity}",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ], message.model_dump_json()
        assert message.stop_reason == "tool_use", message.model_dump_json()
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]


async def test_b4_azure_gpt_6_anthropic_messages_async_stream_returns_tool_use(gateway: Gateway) -> None:
    identity: Final = f"b4-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["stream"] is True
        assert body["input"] == _responses_user_input(identity)
        assert "reasoning" not in body
        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "strict": None,
                "parameters": _WEATHER_TOOL_SCHEMA,
            }
        ]
        return Reply(content_type="text/event-stream", chunks=_responses_sse(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        async with anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url).rstrip("/"),
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            async with client.messages.stream(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a city.",
                        "input_schema": _WEATHER_TOOL_SCHEMA,
                    }
                ],
            ) as stream:
                events: Final = tuple([event async for event in stream])
                message: Final = await stream.get_final_message()
        message_body: Final = message.model_dump(exclude_none=True)
        response_id: Final = f"resp_{identity}"
        try:
            spend: Final = _spend_row(response_id)
        except AssertionError:
            _record_audit_cell(
                "B4",
                "test_b4_azure_gpt_6_anthropic_messages_async_stream_returns_tool_use",
                response_id,
                False,
                target,
            )
            raise
        _record_audit_cell(
            "B4",
            "test_b4_azure_gpt_6_anthropic_messages_async_stream_returns_tool_use",
            response_id,
            True,
            target,
        )
        event_dump: Final = [event.model_dump(exclude_none=True) for event in events]
        assert message_body["content"] == [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": f"call_{identity}",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ], message.model_dump_json()
        assert message.stop_reason == "tool_use", message.model_dump_json()
        assert any(event.type == "message_stop" for event in events), event_dump
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]


def test_b5_openai_custom_base_gpt_6_low_effort_messages_raw_httpx_returns_tool_use(
    gateway: Gateway,
) -> None:
    identity: Final = f"b5-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/responses"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["reasoning"] == {"effort": "low"}
        assert body["input"] == _responses_user_input(identity)
        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "strict": False,
                "parameters": _TOOL_SCHEMA,
            }
        ]
        return Reply(body=_responses_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6-sol", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a city.",
                        "input_schema": _TOOL_SCHEMA,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "B5",
            "test_b5_openai_custom_base_gpt_6_low_effort_messages_raw_httpx_returns_tool_use",
            str(body["id"]),
            True,
            target,
        )
        assert body["content"] == [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": f"call_{identity}",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ], response.text
        assert body["stop_reason"] == "tool_use", response.text
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]
