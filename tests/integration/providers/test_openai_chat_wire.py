import json
import os
import uuid
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_GPT_6_MODELS: Final = ("gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol")
_API_KEY: Final = "synthetic-openai-key"
_PROMPT: Final = "Summarize this conversation in one sentence."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_WEATHER_TOOL: Final = {
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
_RESPONSES_WEATHER_TOOL: Final = {
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


def _responses_tool_reply(model_name: str, identity: str) -> bytes:
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


def _responses_user_input(identity: str) -> list[JsonValue]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"What is the weather in Paris? {identity}"}],
        }
    ]


def _responses_sse(identity: str) -> tuple[bytes, ...]:
    text: Final = "Let me check the weather."
    message_id: Final = f"msg_{identity}"
    function_id: Final = f"fc_{identity}"
    call_id: Final = f"call_{identity}"
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


def _error_response_id(response: httpx.Response) -> str:
    identity: Final = response.headers.get("x-litellm-call-id")
    assert isinstance(identity, str) and identity, "Error response had no LiteLLM call id"
    return identity


def _record_audit_cell(row_id: str, node_id: str, response_id: str | None, path: str, spend_found: bool) -> None:
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


def _completion(identity: str, content: str, model_name: str = _BACKEND) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
        }
    ).encode()


def _tool_completion(model_name: str, identity: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{identity}",
            "object": "chat.completion",
            "created": 1,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the weather.",
                        "tool_calls": [
                            {
                                "id": f"call_{identity}",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ).encode()


@pytest.mark.covers("providers.openai_chat_wire.tool_choice_without_tools_is_dropped_before_the_wire")
def test_openai_chat_tool_choice_without_tools_is_not_forwarded(gateway: Gateway) -> None:
    identity: Final = f"openai-toolless-{uuid.uuid4().hex}"
    prompt: Final = f"{_PROMPT} {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert "tool_choice" not in body, body
        assert "tools" not in body, body
        return Reply(body=_completion(identity, "One sentence."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "tool_choice": "none"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "One sentence.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_a3_azure_gpt_6_function_tool_with_reasoning_effort_none_stays_on_chat(
    gateway: Gateway, model_name: str
) -> None:
    identity: Final = f"azure-{model_name}-{uuid.uuid4().hex}"
    upstream_target: Final = f"/openai/deployments/{model_name}/chat/completions?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == upstream_target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["messages"] == [{"role": "user", "content": f"What is the weather in Paris? {identity}"}]
        assert body["tools"] == [_WEATHER_TOOL]
        assert body["reasoning_effort"] == "none"
        return Reply(body=_tool_completion(model_name, identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{model_name}",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "reasoning_effort": "none",
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(payload["id"]))
        _record_audit_cell(
            "A3",
            f"test_a3_azure_gpt_6_function_tool_with_reasoning_effort_none_stays_on_chat[{model_name}]",
            str(payload["id"]),
            upstream_target,
            True,
        )
        assert payload["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather.",
                    "tool_calls": [
                        {
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", upstream_target)]


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_a1_azure_gpt_6_function_tool_without_reasoning_effort_bridges_to_responses(
    gateway: Gateway, model_name: str
) -> None:
    identity: Final = f"azure-{model_name}-{uuid.uuid4().hex}"
    upstream_target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == upstream_target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["input"] == _responses_user_input(identity)
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        assert "reasoning" not in body
        return Reply(body=_responses_tool_reply(model_name, identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{model_name}",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "num_retries": 0,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "A1",
            f"test_a1_azure_gpt_6_function_tool_without_reasoning_effort_bridges_to_responses[{model_name}]",
            str(body["id"]),
            upstream_target,
            True,
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
                            "id": f"call_{identity}",
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
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", upstream_target)]


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_a7_openai_custom_base_gpt_6_function_tool_without_reasoning_effort_stays_on_chat(
    gateway: Gateway, model_name: str
) -> None:
    identity: Final = f"openai-{model_name}-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["messages"] == [{"role": "user", "content": f"What is the weather in Paris? {identity}"}]
        assert "reasoning_effort" not in body
        assert body["tools"] == [
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
        ]
        return Reply(body=_tool_completion(model_name, identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{model_name}", api_base=wire.url, api_key=_API_KEY)
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
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "A7",
            f"test_a7_openai_custom_base_gpt_6_function_tool_without_reasoning_effort_stays_on_chat[{model_name}]",
            str(body["id"]),
            "/chat/completions",
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_a8_openai_custom_base_gpt_6_function_tool_with_low_effort_bridges_to_responses(
    gateway: Gateway, model_name: str
) -> None:
    identity: Final = f"openai-{model_name}-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["reasoning"]["effort"] == "low"
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        assert body["input"] == _responses_user_input(identity)
        return Reply(body=_responses_tool_reply(model_name, identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{model_name}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "reasoning_effort": "low",
                "tools": [_WEATHER_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "A8",
            f"test_a8_openai_custom_base_gpt_6_function_tool_with_low_effort_bridges_to_responses[{model_name}]",
            str(body["id"]),
            "/responses",
            True,
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
                            "id": f"call_{identity}",
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


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_a2_azure_gpt_6_function_tool_with_low_effort_bridges_to_responses(gateway: Gateway, model_name: str) -> None:
    identity: Final = f"a2-{model_name}-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["reasoning"] == {"effort": "low"}
        assert body["input"] == _responses_user_input(identity)
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply(model_name, identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{model_name}",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "reasoning_effort": "low",
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "A2",
            f"test_a2_azure_gpt_6_function_tool_with_low_effort_bridges_to_responses[{model_name}]",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
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


@pytest.mark.parametrize("tools", (None, []), ids=("no-tools", "empty-tools"))
def test_a4_a5_azure_gpt_6_without_function_tools_stays_on_chat(
    gateway: Gateway, tools: list[JsonValue] | None
) -> None:
    row_id: Final = "A4" if tools is None else "A5"
    node_id: Final = (
        f"test_a4_a5_azure_gpt_6_without_function_tools_stays_on_chat[{'no-tools' if tools is None else 'empty-tools'}]"
    )
    identity: Final = f"{row_id.lower()}-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["messages"] == [{"role": "user", "content": f"No tools here. {identity}"}]
        assert body.get("tools") == tools
        return Reply(body=_completion(f"chatcmpl-{identity}", "No tool call.", "gpt-6-sol"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        request_body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [{"role": "user", "content": f"No tools here. {identity}"}],
            "cache": {"no-cache": True},
            **({"tools": tools} if tools is not None else {}),
        }
        response: Final = gateway.request("POST", "/v1/chat/completions", request_body)
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(row_id, node_id, str(body["id"]), target, True)
        assert body["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "No tool call.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], response.text


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_a6_azure_gpt_6_function_tool_with_null_effort_bridges_to_responses(gateway: Gateway, model_name: str) -> None:
    identity: Final = f"a6-{model_name}-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["input"] == _responses_user_input(identity)
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        assert "reasoning" not in body
        return Reply(body=_responses_tool_reply(model_name, identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{model_name}",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "reasoning_effort": None,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "A6",
            f"test_a6_azure_gpt_6_function_tool_with_null_effort_bridges_to_responses[{model_name}]",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text


def test_a9_azure_gpt_6_custom_tool_only_is_identical_to_base(gateway: Gateway) -> None:
    identity: Final = f"a9-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview"
    custom_tool: Final = {
        "type": "custom",
        "custom": {
            "name": "run_code",
            "description": "Run a code snippet",
            "format": {"type": "text"},
        },
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["tools"] == [custom_tool]
        assert body["messages"] == [{"role": "user", "content": f"Prepare to run code. {identity}"}]
        return Reply(body=_completion(f"chatcmpl-{identity}", "Ready to run code.", "gpt-6-sol"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Prepare to run code. {identity}"}],
                "tools": [custom_tool],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "A9",
            "test_a9_azure_gpt_6_custom_tool_only_is_identical_to_base",
            str(body["id"]),
            target,
            True,
        )
        assert spend["status"] == "success", spend
        assert body["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Ready to run code.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], response.text


def test_a10_sync_openai_sdk_gets_merged_azure_gpt_6_choice(gateway: Gateway) -> None:
    identity: Final = f"a10-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == _responses_user_input(identity)
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        with openai.OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            completion: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                tools=[_WEATHER_TOOL],
                extra_body={"cache": {"no-cache": True}},
            )
        _spend_row(completion.id)
        _record_audit_cell(
            "A10",
            "test_a10_sync_openai_sdk_gets_merged_azure_gpt_6_choice",
            completion.id,
            target,
            True,
        )
        assert completion.model_dump(exclude_none=True)["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "Let me check the weather.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": '{"city":"Paris"}', "name": "get_weather"},
                            "id": f"call_{identity}",
                            "index": 0,
                            "type": "function",
                        }
                    ],
                },
            }
        ], completion.model_dump_json()


async def test_a11_async_openai_sdk_gets_merged_azure_gpt_6_choice(gateway: Gateway) -> None:
    identity: Final = f"a11-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == _responses_user_input(identity)
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        async with openai.AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            completion: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                tools=[_WEATHER_TOOL],
                extra_body={"cache": {"no-cache": True}},
            )
        _spend_row(completion.id)
        _record_audit_cell(
            "A11",
            "test_a11_async_openai_sdk_gets_merged_azure_gpt_6_choice",
            completion.id,
            target,
            True,
        )
        assert completion.model_dump(exclude_none=True)["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "Let me check the weather.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": '{"city":"Paris"}', "name": "get_weather"},
                            "id": f"call_{identity}",
                            "index": 0,
                            "type": "function",
                        }
                    ],
                },
            }
        ], completion.model_dump_json()


def test_d1_azure_gpt_6_responses_400_is_followed_by_a_chat_request(gateway: Gateway) -> None:
    identity: Final = f"d1-gpt-6-sol-{uuid.uuid4().hex}"
    responses_target: Final = "/openai/responses?api-version=2025-04-01-preview"
    chat_target: Final = "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert request.method == "POST"
        assert body["model"] == "gpt-6-sol"
        if request.target == responses_target:
            assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
            return Reply(
                status=400, body=b'{"error":{"message":"synthetic bad request","type":"invalid_request_error"}}'
            )
        assert request.target == chat_target
        assert body["messages"] == [{"role": "user", "content": f"No tools. {identity}"}]
        assert "tools" not in body
        return Reply(body=_completion(f"chatcmpl-{identity}", "Recovered on chat.", "gpt-6-sol"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        failed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "num_retries": 0,
                "cache": {"no-cache": True},
            },
        )
        recovered: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"No tools. {identity}"}],
                "cache": {"no-cache": True},
            },
        )
        assert failed.status_code == 400, failed.text
        assert "synthetic bad request" in failed.text, failed.text
        assert recovered.status_code == 200, recovered.text
        failed_id: Final = _error_response_id(failed)
        _spend_row(failed_id)
        recovered_body: Final = _JSON_OBJECT.validate_json(recovered.content)
        _spend_row(str(recovered_body["id"]))
        _record_audit_cell(
            "D1",
            "test_d1_azure_gpt_6_responses_400_is_followed_by_a_chat_request",
            failed_id,
            responses_target,
            True,
        )
        _record_audit_cell(
            "D1",
            "test_d1_azure_gpt_6_responses_400_is_followed_by_a_chat_request",
            str(recovered_body["id"]),
            chat_target,
            True,
        )
        assert recovered_body["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Recovered on chat.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], recovered.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", responses_target),
            ("POST", chat_target),
        ]


def test_d2_azure_gpt_6_responses_500_returns_an_error_not_empty_choices(gateway: Gateway) -> None:
    identity: Final = f"d2-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(status=500, body=b'{"error":{"message":"synthetic upstream failure","type":"api_error"}}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "num_retries": 0,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 500, response.text
        assert "synthetic upstream failure" in response.text, response.text
        assert "choices" not in _JSON_OBJECT.validate_json(response.content), response.text
        response_id: Final = _error_response_id(response)
        spend: Final = _spend_row(response_id)
        _record_audit_cell(
            "D2",
            "test_d2_azure_gpt_6_responses_500_returns_an_error_not_empty_choices",
            response_id,
            target,
            True,
        )
        assert spend["status"] == "failure", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]


@pytest.mark.parametrize(
    ("effort", "case_id"),
    [
        pytest.param(5, "integer-5", id="integer-5"),
        pytest.param(["low"], "list-low", id="list-low"),
        pytest.param("", "empty-string", id="empty-string"),
        pytest.param("x" * 5000, "five-kb-string", id="five-kb-string"),
    ],
)
def test_d3_azure_gpt_6_reasoning_effort_edge_values_keep_base_status_and_route(
    gateway: Gateway,
    effort: JsonValue,
    case_id: str,
) -> None:
    identity: Final = f"d3-gpt-6-sol-{case_id}-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == _responses_user_input(identity)
        assert body["reasoning"] == {"effort": effort}
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "reasoning_effort": effort,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        spend: Final = _spend_row(str(body["id"]))
        _record_audit_cell(
            "D3",
            f"test_d3_azure_gpt_6_reasoning_effort_edge_values_keep_base_status_and_route[{case_id}]",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]


def test_d4_azure_gpt_6_malformed_function_tool_is_forwarded_to_chat(gateway: Gateway) -> None:
    identity: Final = f"d4-gpt-6-sol-{uuid.uuid4().hex}"
    malformed_tool: Final = {"type": "function"}

    def respond(request: Request) -> Reply:
        target: Final = "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview"
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["messages"] == [{"role": "user", "content": f"Malformed tool. {identity}"}]
        assert body["tools"] == [malformed_tool]
        return Reply(body=_completion(f"chatcmpl-{identity}", "Malformed tool passed through.", "gpt-6-sol"))

    with wire_server(respond) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="azure/gpt-6-sol",
                api_base=wire.url,
                api_key=_API_KEY,
                api_version="2025-04-01-preview",
            )
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"Malformed tool. {identity}"}],
                    "tools": [malformed_tool],
                    "cache": {"no-cache": True},
                },
            )
            assert response.status_code == 200, response.text
            body: Final = _JSON_OBJECT.validate_json(response.content)
            response_id: Final = str(body["id"])
            spend: Final = _spend_row(response_id)
            assert spend["status"] == "success", spend
            _record_audit_cell(
                "D4",
                "test_d4_azure_gpt_6_malformed_function_tool_is_forwarded_to_chat",
                response_id,
                "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview",
                True,
            )
            assert body["choices"] == [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Malformed tool passed through.",
                        "provider_specific_fields": {"refusal": None},
                    },
                    "provider_specific_fields": {},
                }
            ], response.text
            assert [(request.method, request.target) for request in wire.drain()] == [
                ("POST", "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview")
            ]


def test_d5_azure_gpt_6_invalid_json_arguments_are_preserved_in_the_merged_tool_call(gateway: Gateway) -> None:
    identity: Final = f"d5-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(
            body=json.dumps(
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
                            "content": [{"type": "output_text", "text": "Let me check.", "annotations": []}],
                        },
                        {
                            "type": "function_call",
                            "id": f"fc_{identity}",
                            "call_id": f"call_{identity}",
                            "name": "get_weather",
                            "arguments": '{"city"',
                            "status": "completed",
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Call the tool. {identity}"}],
                "tools": [_WEATHER_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "D5",
            "test_d5_azure_gpt_6_invalid_json_arguments_are_preserved_in_the_merged_tool_call",
            str(body["id"]),
            target,
            True,
        )
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city"'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text


def test_d6_azure_gpt_6_bad_caller_key_is_rejected_before_provider_routing(gateway: Gateway) -> None:
    identity: Final = f"d6-gpt-6-sol-{uuid.uuid4().hex}"
    with wire_server(lambda _: Reply(body=_responses_tool_reply("gpt-6-sol", identity))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="azure/gpt-6-sol",
                api_base=wire.url,
                api_key=_API_KEY,
                api_version="2025-04-01-preview",
            )
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"Use the tool. {identity}"}],
                    "tools": [_WEATHER_TOOL],
                    "cache": {"no-cache": True},
                },
                key="sk-audit-invalid-caller-key",
                headers={"x-litellm-call-id": identity},
            )
            assert response.status_code == 401, response.text
            rows: Final = read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,))
            assert rows == [], rows
            _record_audit_cell(
                "D6",
                "test_d6_azure_gpt_6_bad_caller_key_is_rejected_before_provider_routing",
                None,
                "none",
                False,
            )
            assert wire.drain() == ()


def test_d7_azure_gpt_6_identical_uncached_function_requests_are_each_logged(gateway: Gateway) -> None:
    pytest.skip("BUG: cache no-cache is ignored on chat requests bridged to Responses (LIT-9196)")
    identity: Final = f"d7-gpt-6-sol-{uuid.uuid4().hex}"
    response_identities: Final = (f"{identity}-first", f"{identity}-second")
    replies: Final = iter(response_identities)
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply("gpt-6-sol", next(replies)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        request_body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": f"Repeat this request. {identity}"}],
            "tools": [_WEATHER_TOOL],
            "cache": {"no-cache": True},
        }
        responses: Final = tuple(gateway.request("POST", "/v1/chat/completions", request_body) for _ in range(2))
        assert [response.status_code for response in responses] == [200, 200], [response.text for response in responses]
        bodies: Final = tuple(_JSON_OBJECT.validate_json(response.content) for response in responses)
        spends: Final = tuple(_spend_row(str(body["id"])) for body in bodies)
        for body in bodies:
            _record_audit_cell(
                "D7",
                "test_d7_azure_gpt_6_identical_uncached_function_requests_are_each_logged",
                str(body["id"]),
                target,
                True,
            )
        observed: Final = wire.drain()
        assert [(request.method, request.target) for request in observed] == [
            ("POST", target),
            ("POST", target),
        ]
        wire_bodies: Final = tuple(_JSON_OBJECT.validate_json(request.body) for request in observed)
        assert wire_bodies[0] == wire_bodies[1], wire_bodies
        assert [body["model"] for body in wire_bodies] == ["gpt-6-sol", "gpt-6-sol"]
        assert [body["id"] for body in bodies] == [
            f"resp_{response_identity}" for response_identity in response_identities
        ]
        assert len({spend["request_id"] for spend in spends}) == 2, spends
        for index, spend in enumerate(spends):
            assert spend["status"] == "success", (index, spend)
        assert [body["choices"] for body in bodies] == [
            [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the weather.",
                        "tool_calls": [
                            {
                                "id": f"call_{response_identity}",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                                "index": 0,
                            }
                        ],
                    },
                }
            ]
            for response_identity in response_identities
        ], [response.text for response in responses]


def test_e1_azure_gpt_6_agent_model_alias_bridges_to_responses(gateway: Gateway) -> None:
    identity: Final = f"e1-agent-model-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.post(
            "/model/new",
            {
                "model_name": "agent-model",
                "litellm_params": {
                    "model": "azure/gpt-6-sol",
                    "api_base": wire.url,
                    "api_key": _API_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {},
            },
        )
        model_info: Final = _JSON_OBJECT.validate_python(created["model_info"])
        scenario.cleanups.callback(scenario.delete_model, str(model_info["id"]))
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "agent-model",
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "E1",
            "test_e1_azure_gpt_6_agent_model_alias_bridges_to_responses",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text


def test_e2_azure_gpt_6_router_fallback_keeps_a_single_bridged_choice(gateway: Gateway) -> None:
    identity: Final = f"e2-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] in {"gpt-6-sol", "gpt-6-luna"}
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        if body["model"] == "gpt-6-sol":
            return Reply(status=500, body=b'{"error":{"message":"synthetic primary failure","type":"api_error"}}')
        return Reply(body=_responses_tool_reply("gpt-6-luna", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        fallback: Final = scenario.model(
            model="azure/gpt-6-luna",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": primary,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "fallbacks": [fallback],
                "num_retries": 0,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "E2",
            "test_e2_azure_gpt_6_router_fallback_keeps_a_single_bridged_choice",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", target),
            ("POST", target),
        ]


def test_e3_azure_gpt_6_reasoning_dict_effort_none_stays_on_chat(gateway: Gateway) -> None:
    identity: Final = f"e3-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["reasoning_effort"] == "none"
        assert body["tools"] == [_WEATHER_TOOL]
        return Reply(body=_tool_completion("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "reasoning_effort": {"effort": "none"},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "E3",
            "test_e3_azure_gpt_6_reasoning_dict_effort_none_stays_on_chat",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], response.text


def test_e4_azure_gpt_6_reasoning_dict_summary_auto_bridges_and_merges(gateway: Gateway) -> None:
    identity: Final = f"e4-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["reasoning"] == {"effort": "none", "summary": "auto"}
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(body=_responses_tool_reply("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "reasoning_effort": {"effort": "none", "summary": "auto"},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "E4",
            "test_e4_azure_gpt_6_reasoning_dict_summary_auto_bridges_and_merges",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text


def test_e5_azure_gpt_6_deployment_none_effort_keeps_unset_request_on_chat(gateway: Gateway) -> None:
    identity: Final = f"e5-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/deployments/gpt-6-sol/chat/completions?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["reasoning_effort"] == "none"
        assert body["tools"] == [_WEATHER_TOOL]
        return Reply(body=_tool_completion("gpt-6-sol", identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
            reasoning_effort="none",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        _spend_row(str(body["id"]))
        _record_audit_cell(
            "E5",
            "test_e5_azure_gpt_6_deployment_none_effort_keeps_unset_request_on_chat",
            str(body["id"]),
            target,
            True,
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
                            "id": f"call_{identity}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], response.text


def test_b1_azure_gpt_6_bridged_stream_returns_text_and_tool_call_on_one_choice(gateway: Gateway) -> None:
    identity: Final = f"azure-gpt-6-sol-stream-{uuid.uuid4().hex}"
    expected_text: Final = "Let me check the weather."
    response_id: Final = f"resp_{identity}"
    events: Final = (
        {
            "type": "response.created",
            "response": {
                "id": response_id,
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
                "id": "msg_weather",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_weather",
            "output_index": 0,
            "content_index": 0,
            "delta": "Let me check ",
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_weather",
            "output_index": 0,
            "content_index": 0,
            "delta": "the weather.",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg_weather",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": expected_text, "annotations": []}],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "status": "in_progress",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 1,
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 1,
            "delta": '"Paris"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [
                    {
                        "id": "msg_weather",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": expected_text, "annotations": []}],
                    },
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )
    stream_chunks: Final = tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/openai/responses?api-version=2025-04-01-preview"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        return Reply(content_type="text/event-stream", chunks=stream_chunks)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {gateway.key}"},
            json={
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
                "stream": True,
                "cache": {"no-cache": True},
            },
        ) as response:
            response_body: Final = response.read()
            assert response.status_code == 200, response.text
            chunks: Final = tuple(
                _JSON_OBJECT.validate_json(line.removeprefix("data: "))
                for line in response_body.decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            assert chunks, response.text
            _spend_row(response_id)
            _record_audit_cell(
                "B1",
                "test_b1_azure_gpt_6_bridged_stream_returns_text_and_tool_call_on_one_choice",
                response_id,
                "/openai/responses?api-version=2025-04-01-preview",
                True,
            )
            choices: Final = tuple(chain.from_iterable(chunk["choices"] for chunk in chunks))
            assert choices, response.text
            assert all(choice["index"] == 0 for choice in choices), response.text
            assert "".join(str(choice["delta"].get("content") or "") for choice in choices) == expected_text, (
                response.text
            )
            tool_call_chunks: Final = tuple(
                chain.from_iterable(choice["delta"].get("tool_calls", []) for choice in choices)
            )
            assert (
                "".join(str(tool_call["function"].get("name") or "") for tool_call in tool_call_chunks) == "get_weather"
            ), response.text
            assert (
                "".join(str(tool_call["function"].get("arguments") or "") for tool_call in tool_call_chunks)
                == '{"city":"Paris"}'
            ), response.text
            assert tuple(
                choice.get("finish_reason") for choice in choices if choice.get("finish_reason") is not None
            ) == ("tool_calls",), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/responses?api-version=2025-04-01-preview")
        ]


async def test_b2_async_openai_sdk_stream_returns_text_and_tool_call_on_one_choice(gateway: Gateway) -> None:
    identity: Final = f"b2-gpt-6-sol-{uuid.uuid4().hex}"
    target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["tools"] == [_RESPONSES_WEATHER_TOOL]
        return Reply(content_type="text/event-stream", chunks=_responses_sse(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        async with openai.AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                tools=[_WEATHER_TOOL],
                stream=True,
                extra_body={"cache": {"no-cache": True}},
            )
            chunks: Final = tuple([chunk async for chunk in stream])
        chunk_dump: Final = [chunk.model_dump(exclude_none=True) for chunk in chunks]
        assert chunks, f"No streaming chunks: {chunk_dump}"
        response_id: Final = chunks[0].id
        spend: Final = _spend_row(response_id)
        _record_audit_cell(
            "B2",
            "test_b2_async_openai_sdk_stream_returns_text_and_tool_call_on_one_choice",
            response_id,
            target,
            True,
        )
        choices: Final = tuple(choice for chunk in chunks for choice in chunk.choices)
        assert choices, f"No streaming choices: {chunk_dump}"
        assert all(choice.index == 0 for choice in choices), chunk_dump
        assert "".join(choice.delta.content or "" for choice in choices) == "Let me check the weather.", chunk_dump
        tool_chunks: Final = tuple(chain.from_iterable(choice.delta.tool_calls or () for choice in choices))
        assert "".join(tool_call.function.name or "" for tool_call in tool_chunks) == "get_weather", chunk_dump
        assert "".join(tool_call.function.arguments or "" for tool_call in tool_chunks) == '{"city":"Paris"}', (
            chunk_dump
        )
        assert tuple(choice.finish_reason for choice in choices if choice.finish_reason is not None) == (
            "tool_calls",
        ), chunk_dump
        assert spend["status"] == "success", spend
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]
