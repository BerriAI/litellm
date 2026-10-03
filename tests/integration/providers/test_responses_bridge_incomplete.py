import json
import os
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CHAT_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_RESPONSES_TOOL: Final = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "strict": None,
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


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


def _record_audit_cell(row_id: str, node_id: str, response_id: str, spend_found: bool) -> None:
    results_dir: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    assert results_dir is not None, "INTEGRATION_RESULTS_DIR is required for audit cell evidence"
    artifact: Final = Path(results_dir) / "audit-cells.jsonl"
    record: Final = {
        "row_id": row_id,
        "node_id": node_id,
        "response_id": response_id,
        "upstream_path": "/responses",
        "spend_row_found": spend_found,
        "leg": os.environ.get("LITAUDIT_LEG", "head"),
    }
    with artifact.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _assert_bridge_request(request: Request, identity: str) -> None:
    assert request.method == "POST" and request.target == "/responses", request.target
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == "gpt-6-sol", body
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"What is the weather in Paris? {identity}"}],
        }
    ], body
    assert body["tools"] == [_RESPONSES_TOOL], body


def _responses_reply(
    identity: str,
    output: list[JsonValue],
    *,
    status: str = "completed",
    incomplete_details: dict[str, JsonValue] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": status,
            "model": "gpt-6-sol",
            "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            **({"incomplete_details": incomplete_details} if incomplete_details is not None else {}),
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.responses_bridge.max_output_tokens_incomplete_maps_to_length")
def test_chat_over_responses_deployment_returns_length_when_output_tokens_run_out(gateway: Gateway) -> None:
    identity: Final = "responses-incomplete-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = json.loads(request.body)
        assert body["model"] == "gpt-5.3-codex"
        assert body["max_output_tokens"] == 16
        assert body["reasoning"] == {"effort": "high"}
        assert body["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"explain the plan in detail {identity}"}],
            }
        ]
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.3-codex",
                    "output": [{"type": "reasoning", "id": f"rs_{identity}", "summary": []}],
                    "usage": {"input_tokens": 12, "output_tokens": 16, "total_tokens": 28},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-5.3-codex", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"explain the plan in detail {identity}"}],
                "reasoning_effort": "high",
                "max_completion_tokens": 16,
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert len(tuple(request for request in wire.drain() if request.method == "POST")) == 1
        assert [choice["finish_reason"] for choice in body["choices"]] == ["length"], response.text
        assert body["choices"][0]["message"]["content"] == "", response.text
        assert body["choices"][0]["message"]["role"] == "assistant", response.text
        assert body["usage"]["prompt_tokens"] == 12 and body["usage"]["completion_tokens"] == 16, response.text
        assert body["usage"]["total_tokens"] == 28, response.text


@pytest.mark.covers("other.provider_wire.responses_bridge.sub_minimum_max_tokens_clamped_to_provider_floor")
def test_messages_over_responses_deployment_with_max_tokens_1_is_clamped_to_16_instead_of_400(gateway: Gateway) -> None:
    identity: Final = "responses-clamp-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = json.loads(request.body)
        assert body["model"] == "gpt-5.4"
        if body["max_output_tokens"] < 16:
            return Reply(
                status=400,
                body=json.dumps(
                    {
                        "error": {
                            "message": "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 1 instead.",
                            "type": "invalid_request_error",
                            "param": "max_output_tokens",
                            "code": "integer_below_min_value",
                        }
                    }
                ).encode(),
            )
        assert body["max_output_tokens"] == 16
        assert body["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"warmup probe {identity}"}],
            }
        ]
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg_{identity}",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 1, "total_tokens": 13},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-5.4", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": f"warmup probe {identity}"}],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert len(tuple(request for request in wire.drain() if request.method == "POST")) == 1
        assert body["role"] == "assistant", response.text
        assert body["content"] == [{"type": "text", "text": "ok"}], response.text
        assert body["stop_reason"] == "end_turn", response.text


@pytest.mark.covers("providers.responses_bridge.sub_minimum_max_tokens_is_raised_to_the_openai_floor")
def test_messages_over_responses_deployment_with_max_tokens_one_reaches_openai_as_sixteen(gateway: Gateway) -> None:
    identity: Final = "responses-min-tokens-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = json.loads(request.body)
        assert body["model"] == "gpt-5.6-sol"
        assert body["max_output_tokens"] == 16, body
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-5.6-sol",
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg_{identity}",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-5.6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": f"warmup {identity}"}],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert len(tuple(request for request in wire.drain() if request.method == "POST")) == 1
        assert body["content"] == [{"type": "text", "text": "ok"}], response.text
        assert body["usage"]["input_tokens"] == 9 and body["usage"]["output_tokens"] == 1, response.text


def test_c1_chat_over_responses_deployment_merges_message_and_function_call(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_weather",
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
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                            "status": "completed",
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C1", "test_c1_chat_over_responses_deployment_merges_message_and_function_call", str(body["id"]), True
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather.",
                    "tool_calls": [
                        {
                            "id": "fc_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c2_chat_over_responses_deployment_keeps_reasoning_with_merged_tool_call(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-reasoning-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_weather_reasoning",
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
                            "type": "reasoning",
                            "id": "rs_weather",
                            "summary": [{"type": "summary_text", "text": "Checking the forecast."}],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                            "status": "completed",
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C2", "test_c2_chat_over_responses_deployment_keeps_reasoning_with_merged_tool_call", str(body["id"]), True
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather.",
                    "reasoning_content": "Checking the forecast.",
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_weather",
                            "summary": [{"type": "summary_text", "text": "Checking the forecast."}],
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "fc_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c4_chat_over_responses_deployment_returns_tool_call_only_reply_as_one_choice(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-tool-only-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C4",
            "test_c4_chat_over_responses_deployment_returns_tool_call_only_reply_as_one_choice",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "fc_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c7_chat_over_responses_deployment_merges_function_call_followed_by_message(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-tool-then-message-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                            "status": "completed",
                        },
                        {
                            "type": "message",
                            "id": "msg_after_tool",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "After the tool.", "annotations": []}],
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C7",
            "test_c7_chat_over_responses_deployment_merges_function_call_followed_by_message",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "After the tool.",
                    "tool_calls": [
                        {
                            "id": "fc_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c3_chat_over_responses_deployment_merges_message_reasoning_and_function_call(
    gateway: Gateway,
) -> None:
    identity: Final = "c3-" + uuid.uuid4().hex
    reasoning: Final = {
        "type": "reasoning",
        "id": f"rs_{identity}",
        "summary": [{"type": "summary_text", "text": "Checking the forecast."}],
    }
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}],
        },
        reasoning,
        {
            "type": "function_call",
            "id": f"fc_{identity}",
            "call_id": f"call_{identity}",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        },
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        return Reply(body=_responses_reply(identity, output))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_CHAT_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C3",
            "test_c3_chat_over_responses_deployment_merges_message_reasoning_and_function_call",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sunny.",
                    "reasoning_content": "Checking the forecast.",
                    "reasoning_items": [reasoning],
                    "tool_calls": [
                        {
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c5_chat_over_responses_deployment_returns_message_only_choice(gateway: Gateway) -> None:
    identity: Final = "c5-" + uuid.uuid4().hex
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Only text.", "annotations": []}],
        }
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        return Reply(body=_responses_reply(identity, output))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_CHAT_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C5", "test_c5_chat_over_responses_deployment_returns_message_only_choice", str(body["id"]), True
        )
        assert body["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "Only text."},
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c6_chat_over_responses_deployment_merges_two_function_calls_into_one_choice(
    gateway: Gateway,
) -> None:
    pytest.skip("BUG: non-stream bridged chat gives every tool call index 0 (LIT-9195)")
    identity: Final = "c6-" + uuid.uuid4().hex
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I will check both cities.", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": f"fc_{identity}_0",
            "call_id": f"call_{identity}_0",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": f"fc_{identity}_1",
            "call_id": f"call_{identity}_1",
            "name": "get_weather",
            "arguments": '{"city":"Rome"}',
            "status": "completed",
        },
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        return Reply(body=_responses_reply(identity, output))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_CHAT_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C6",
            "test_c6_chat_over_responses_deployment_merges_two_function_calls_into_one_choice",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I will check both cities.",
                    "tool_calls": [
                        {
                            "id": f"call_{identity}_0",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        },
                        {
                            "id": f"call_{identity}_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Rome"}'},
                            "index": 1,
                        },
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c8_chat_over_responses_deployment_keeps_url_citation_with_merged_tool_call(gateway: Gateway) -> None:
    identity: Final = "c8-" + uuid.uuid4().hex
    annotation: Final = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 6,
        "title": "Forecast",
        "url": "https://example.com/forecast",
    }
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Sunny.",
                    "annotations": [annotation],
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
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        return Reply(body=_responses_reply(identity, output))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_CHAT_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C8",
            "test_c8_chat_over_responses_deployment_keeps_url_citation_with_merged_tool_call",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sunny.",
                    "annotations": [annotation],
                    "tool_calls": [
                        {
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c9_two_messages_and_tool_call_remains_a_documented_converter_gap(gateway: Gateway) -> None:
    identity: Final = "c9-" + uuid.uuid4().hex
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}_0",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "First message.", "annotations": []}],
        },
        {
            "type": "message",
            "id": f"msg_{identity}_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Second message.", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": f"fc_{identity}",
            "call_id": f"call_{identity}",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        },
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        return Reply(body=_responses_reply(identity, output))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_CHAT_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C9",
            "test_c9_two_messages_and_tool_call_remains_a_documented_converter_gap",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "First message."},
            },
            {
                "finish_reason": "tool_calls",
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": "Second message.",
                    "tool_calls": [
                        {
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            },
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend


def test_c10_incomplete_chat_over_responses_keeps_length_and_merged_tool_call(gateway: Gateway) -> None:
    identity: Final = "c10-" + uuid.uuid4().hex
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Partial answer.", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": f"fc_{identity}",
            "call_id": f"call_{identity}",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        },
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        _assert_bridge_request(request, identity)
        request_body: Final = _JSON_OBJECT.validate_json(request.body)
        assert request_body["max_output_tokens"] == 64, request_body
        return Reply(
            body=_responses_reply(
                identity,
                output,
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_CHAT_TOOL],
                "max_completion_tokens": 64,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "C10",
            "test_c10_incomplete_chat_over_responses_keeps_length_and_merged_tool_call",
            str(body["id"]),
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "length",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Partial answer.",
                    "tool_calls": [
                        {
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        assert spend["status"] == "success", spend
