import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from pydantic import JsonValue
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.provider_wire.responses_bridge.max_output_tokens_incomplete_maps_to_length")
@pytest.mark.parametrize(
    ("incomplete_reason", "expected_finish_reason"),
    [
        pytest.param("max_output_tokens", "length", id="length"),
        pytest.param("content_filter", "content_filter", id="content_filter"),
    ],
)
def test_chat_over_responses_deployment_returns_length_when_output_tokens_run_out(
    gateway: Gateway, incomplete_reason: str, expected_finish_reason: str
) -> None:
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
                    "incomplete_details": {"reason": incomplete_reason},
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
        assert [choice["finish_reason"] for choice in body["choices"]] == [expected_finish_reason], response.text
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


def test_chat_over_responses_deployment_merges_message_and_function_call(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather",
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
        body: Final = response.json()
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


def test_chat_over_responses_deployment_keeps_reasoning_with_merged_tool_call(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-reasoning-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather_reasoning",
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
        body: Final = response.json()
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


def test_chat_over_responses_deployment_returns_tool_call_only_reply_as_one_choice(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-tool-only-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather_tool_only",
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
        body: Final = response.json()
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


def test_chat_over_responses_deployment_merges_function_call_followed_by_message(gateway: Gateway) -> None:
    identity: Final = "responses-bridge-tool-then-message-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather_tool_then_message",
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
        body: Final = response.json()
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


def test_chat_over_responses_deployment_merges_two_messages_and_function_call_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-two-messages-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather_two_messages",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_first",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "First message.", "annotations": []}],
                        },
                        {
                            "type": "message",
                            "id": "msg_second",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Second message.", "annotations": []}],
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
        body: Final = response.json()
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "First message.Second message.",
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


def test_chat_over_responses_deployment_routes_commentary_out_of_content(gateway: Gateway) -> None:
    identity: Final = "responses-commentary-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'

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
                            "id": "msg_commentary",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Let me answer that.",
                                    "annotations": [],
                                }
                            ],
                        },
                        {
                            "type": "message",
                            "id": "msg_answer",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": answer, "annotations": []}],
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
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                    "provider_specific_fields": {"commentary": "Let me answer that."},
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def test_chat_over_responses_deployment_stream_joins_commentary_fragments(gateway: Gateway) -> None:
    identity: Final = "responses-commentary-stream-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'
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
                "id": "msg_commentary",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "commentary",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_commentary",
            "output_index": 0,
            "content_index": 0,
            "delta": "Let me ",
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_commentary",
            "output_index": 0,
            "content_index": 0,
            "delta": "answer that.",
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "msg_answer",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "final_answer",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_answer",
            "output_index": 1,
            "content_index": 0,
            "delta": answer,
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
                        "type": "message",
                        "id": "msg_commentary",
                        "status": "completed",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "Let me answer that.", "annotations": []}],
                    },
                    {
                        "type": "message",
                        "id": "msg_answer",
                        "status": "completed",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )
    stream_chunks: Final = tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(content_type="text/event-stream", chunks=stream_chunks)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {gateway.key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "stream": True,
                "cache": {"no-cache": True},
            },
        ) as response:
            response_body: Final = response.read()
            assert response.status_code == 200, response.text
            chunks: Final = tuple(
                json.loads(line.removeprefix("data: "))
                for line in response_body.decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            deltas: Final = tuple(choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ())
            assert deltas, response_body.decode()
            assert "".join(str(delta.get("content") or "") for delta in deltas) == answer, response_body.decode()
            assert (
                "".join(
                    str(object_value(delta["provider_specific_fields"])["commentary"])
                    for delta in deltas
                    if isinstance(delta.get("provider_specific_fields"), dict)
                    and "commentary" in delta["provider_specific_fields"]
                )
                == "Let me answer that."
            ), response_body.decode()


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_chat_over_responses_deployment_guardrail_masks_commentary_email(gateway: Gateway, stream: bool) -> None:
    identity: Final = "responses-commentary-guardrail-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'
    commentary_text: Final = f"Let me check for {identity}-user@example.com first"
    guardrail_name: Final = f"mask-commentary-{uuid.uuid4().hex[:8]}"
    created: Final = gateway.client.post(
        "/guardrails",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "guardrail": {
                "guardrail_name": guardrail_name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "post_call",
                    "default_on": True,
                    "patterns": [
                        {
                            "pattern_type": "regex",
                            "pattern": r"[\w.-]+@[\w.-]+",
                            "name": "email",
                            "action": "MASK",
                        }
                    ],
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    guardrail_id: Final = created.json()["guardrail_id"]
    try:
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
                    "id": "msg_commentary",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [],
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_commentary",
                "output_index": 0,
                "content_index": 0,
                "delta": commentary_text,
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "msg_answer",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [],
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_answer",
                "output_index": 1,
                "content_index": 0,
                "delta": answer,
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
                            "type": "message",
                            "id": "msg_commentary",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [{"type": "output_text", "text": commentary_text, "annotations": []}],
                        },
                        {
                            "type": "message",
                            "id": "msg_answer",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": answer, "annotations": []}],
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            },
        )
        output_items: Final = [
            {
                "type": "message",
                "id": "msg_commentary",
                "status": "completed",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": commentary_text, "annotations": []}],
            },
            {
                "type": "message",
                "id": "msg_answer",
                "status": "completed",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            },
        ]
        stream_chunks: Final = tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)

        def respond(request: Request) -> Reply:
            if request.method == "GET" and request.target == "/v1/models":
                return Reply(body=b'{"object":"list","data":[]}')
            assert request.method == "POST" and request.target == "/responses", request.target
            if stream:
                return Reply(content_type="text/event-stream", chunks=stream_chunks)
            return Reply(
                body=json.dumps(
                    {
                        "id": f"resp_{identity}",
                        "object": "response",
                        "created_at": 1789788253,
                        "status": "completed",
                        "model": "gpt-6-sol",
                        "output": output_items,
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    }
                ).encode()
            )

        with wire_server(respond) as wire, gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
            )
            if stream:
                # flat post_call guardrails drop streamed rewrites; a post_call
                # pipeline is what delivers ended-stream rewrites to the client
                policy_name: Final = f"mask-commentary-policy-{uuid.uuid4().hex[:8]}"
                policy: Final = gateway.client.post(
                    "/policies",
                    headers={"x-litellm-api-key": gateway.key},
                    json={
                        "policy_name": policy_name,
                        "guardrails_add": [guardrail_name],
                        "pipeline": {
                            "mode": "post_call",
                            "steps": [{"guardrail": guardrail_name, "on_pass": "next", "on_fail": "block"}],
                        },
                    },
                )
                assert policy.status_code == 200, policy.text
                policy_id: Final = policy.json()["policy_id"]
                attachment: Final = gateway.client.post(
                    "/policies/attachments",
                    headers={"x-litellm-api-key": gateway.key},
                    json={"policy_name": policy_name, "models": [model]},
                )
                assert attachment.status_code == 200, attachment.text
                attachment_id: Final = attachment.json()["attachment_id"]
            try:
                if stream:
                    with gateway.client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {gateway.key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                            "stream": True,
                            "cache": {"no-cache": True},
                        },
                    ) as response:
                        response_body: Final = response.read()
                        assert response.status_code == 200, response.text
                        chunks: Final = tuple(
                            json.loads(line.removeprefix("data: "))
                            for line in response_body.decode().splitlines()
                            if line.startswith("data: ") and line != "data: [DONE]"
                        )
                        deltas: Final = tuple(
                            choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ()
                        )
                        assert "".join(str(delta.get("content") or "") for delta in deltas) == answer, (
                            response_body.decode()
                        )
                        assert (
                            "".join(
                                str(object_value(delta["provider_specific_fields"])["commentary"])
                                for delta in deltas
                                if isinstance(delta.get("provider_specific_fields"), dict)
                                and "commentary" in delta["provider_specific_fields"]
                            )
                            == "Let me check for [EMAIL_REDACTED] first"
                        ), response_body.decode()
                else:
                    response = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                            "cache": {"no-cache": True},
                        },
                    )
                    assert response.status_code == 200, response.text
                    body: Final = response.json()
                    assert body["choices"] == [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": answer,
                                "provider_specific_fields": {"commentary": "Let me check for [EMAIL_REDACTED] first"},
                            },
                        }
                    ], response.text
            finally:
                if stream:
                    gateway.client.delete(
                        f"/policies/attachments/{attachment_id}", headers={"x-litellm-api-key": gateway.key}
                    )
                    gateway.client.delete(f"/policies/{policy_id}", headers={"x-litellm-api-key": gateway.key})
    finally:
        deleted: Final = gateway.client.delete(
            f"/guardrails/{guardrail_id}", headers={"x-litellm-api-key": gateway.key}
        )
        assert deleted.status_code == 200, deleted.text


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_chat_over_responses_deployment_guardrail_masks_commentary_email_in_content_on_tool_turns(
    gateway: Gateway, stream: bool
) -> None:
    identity: Final = "responses-commentary-tool-guardrail-" + uuid.uuid4().hex
    commentary_text: Final = f"Let me check for {identity}-user@example.com first"
    masked_note: Final = "Let me check for [EMAIL_REDACTED] first"
    guardrail_name: Final = f"mask-commentary-content-{uuid.uuid4().hex[:8]}"
    created: Final = gateway.client.post(
        "/guardrails",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "guardrail": {
                "guardrail_name": guardrail_name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "post_call",
                    "default_on": True,
                    "patterns": [
                        {
                            "pattern_type": "regex",
                            "pattern": r"[\w.-]+@[\w.-]+",
                            "name": "email",
                            "action": "MASK",
                        }
                    ],
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    guardrail_id: Final = created.json()["guardrail_id"]
    policy_name: Final = f"mask-commentary-content-policy-{uuid.uuid4().hex[:8]}"
    policy: Final = gateway.client.post(
        "/policies",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "policy_name": policy_name,
            "guardrails_add": [guardrail_name],
            "pipeline": {
                "mode": "post_call",
                "steps": [{"guardrail": guardrail_name, "on_pass": "next", "on_fail": "block"}],
            },
        },
    )
    assert policy.status_code == 200, policy.text
    policy_id: Final = policy.json()["policy_id"]
    attachment_id: Final | None = None
    try:
        output_items: Final = [
            {
                "type": "message",
                "id": "msg_commentary",
                "status": "completed",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": commentary_text, "annotations": []}],
            },
            {
                "type": "function_call",
                "id": "fc_sf",
                "call_id": "call_sf",
                "name": "get_weather",
                "arguments": '{"city":"SF"}',
                "status": "completed",
            },
        ]
        stream_events: Final = (
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
                    "id": "msg_commentary",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [],
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_commentary",
                "output_index": 0,
                "content_index": 0,
                "delta": commentary_text,
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_sf",
                    "call_id": "call_sf",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                    "status": "completed",
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
                    "output": output_items,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            },
        )

        def respond(request: Request) -> Reply:
            if request.method == "GET" and request.target == "/v1/models":
                return Reply(body=b'{"object":"list","data":[]}')
            assert request.method == "POST" and request.target == "/responses", request.target
            if stream:
                sse: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in stream_events) + "data: [DONE]\n\n"
                return Reply(body=sse.encode(), headers={"content-type": "text/event-stream"})
            return Reply(
                body=json.dumps(
                    {
                        "id": f"resp_{identity}",
                        "object": "response",
                        "created_at": 1789788253,
                        "status": "completed",
                        "model": "gpt-6-sol",
                        "output": output_items,
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    }
                ).encode()
            )

        with wire_server(respond) as wire, gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
            )
            attachment: Final = gateway.client.post(
                "/policies/attachments",
                headers={"x-litellm-api-key": gateway.key},
                json={"policy_name": policy_name, "models": [model]},
            )
            assert attachment.status_code == 200, attachment.text
            attachment_id = attachment.json()["attachment_id"]
            try:
                if stream:
                    with gateway.client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {gateway.key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": f"What is the weather in SF? {identity}"}],
                            "stream": True,
                            "cache": {"no-cache": True},
                        },
                    ) as response:
                        response_body: Final = response.read()
                        assert response.status_code == 200, response.text
                        chunks: Final = tuple(
                            json.loads(line.removeprefix("data: "))
                            for line in response_body.decode().splitlines()
                            if line.startswith("data: ") and line != "data: [DONE]"
                        )
                        deltas: Final = tuple(
                            choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ()
                        )
                        assert "".join(str(delta.get("content") or "") for delta in deltas) == masked_note, (
                            response_body.decode()
                        )
                        assert all(
                            "commentary" not in delta["provider_specific_fields"]
                            for delta in deltas
                            if isinstance(delta.get("provider_specific_fields"), dict)
                        ), response_body.decode()
                        assert "example.com" not in response_body.decode(), response_body.decode()
                else:
                    response = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": f"What is the weather in SF? {identity}"}],
                            "cache": {"no-cache": True},
                        },
                    )
                    assert response.status_code == 200, response.text
                    body: Final = response.json()
                    assert body["choices"][0]["message"]["content"] == masked_note, response.text
                    assert "commentary" not in (body["choices"][0]["message"].get("provider_specific_fields") or {}), (
                        response.text
                    )
                    assert body["choices"][0]["finish_reason"] == "tool_calls", response.text
            finally:
                gateway.client.delete(
                    f"/policies/attachments/{attachment_id}", headers={"x-litellm-api-key": gateway.key}
                )
    finally:
        gateway.client.delete(f"/policies/{policy_id}", headers={"x-litellm-api-key": gateway.key})
        deleted: Final = gateway.client.delete(
            f"/guardrails/{guardrail_id}", headers={"x-litellm-api-key": gateway.key}
        )
        assert deleted.status_code == 200, deleted.text


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_chat_over_responses_deployment_keeps_commentary_in_content_on_tool_turns(
    gateway: Gateway, stream: bool
) -> None:
    identity: Final = "responses-commentary-content-" + uuid.uuid4().hex
    commentary_text: Final = "Let me check the weather."
    requests_seen: list[Mapping[str, JsonValue]] = []  # mutable-ok: upstream turn counter

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        requests_seen.append(json.loads(request.body))
        if stream:
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
                        "id": "msg_commentary",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [],
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_commentary",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": commentary_text,
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_sf",
                        "call_id": "call_sf",
                        "name": "get_weather",
                        "arguments": '{"city":"SF"}',
                        "status": "completed",
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
                                "type": "function_call",
                                "id": "fc_sf",
                                "call_id": "call_sf",
                                "name": "get_weather",
                                "arguments": '{"city":"SF"}',
                                "status": "completed",
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                },
            )
            sse: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return Reply(body=sse.encode(), headers={"content-type": "text/event-stream"})
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
                            "id": "msg_commentary",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [{"type": "output_text", "text": commentary_text, "annotations": []}],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_sf",
                            "call_id": "call_sf",
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                            "status": "completed",
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        messages: list[Mapping[str, JsonValue]] = [
            {"role": "user", "content": f"What is the weather in SF? {identity}"}
        ]  # mutable-ok: conversation grows each turn

        def call() -> Mapping[str, JsonValue]:
            if stream:
                chunks: list[Mapping[str, JsonValue]] = []  # mutable-ok: stream events accumulate
                with gateway.client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {gateway.key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "stream": True,
                        "cache": {"no-cache": True},
                    },
                ) as response:
                    response_body: Final = response.read()
                    assert response.status_code == 200, response.text
                    chunks.extend(
                        json.loads(line.removeprefix("data: "))
                        for line in response_body.decode().splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    )
                deltas: Final = tuple(choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ())
                finish_reasons: Final = tuple(
                    choice["finish_reason"]
                    for chunk in chunks
                    for choice in chunk.get("choices") or ()
                    if choice.get("finish_reason")
                )
                assert "".join(str(delta.get("content") or "") for delta in deltas) == commentary_text, (
                    response_body.decode()
                )
                assert all(
                    "commentary" not in delta["provider_specific_fields"]
                    for delta in deltas
                    if isinstance(delta.get("provider_specific_fields"), dict)
                ), response_body.decode()
                assert finish_reasons == ("tool_calls",), response_body.decode()
                tool_calls: Final = [
                    tool_call
                    for chunk in chunks
                    for choice in chunk.get("choices") or ()
                    for tool_call in (choice["delta"].get("tool_calls") or [])
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": commentary_text,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                return {}
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": messages, "tools": tools, "cache": {"no-cache": True}},
            )
            assert response.status_code == 200, response.text
            choice: Final[Mapping[str, JsonValue]] = response.json()["choices"][0]
            message: Final[Mapping[str, JsonValue]] = choice["message"]
            messages.append(
                {k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "provider_specific_fields")}
            )
            return choice

        first: Final = call()
        if not stream:
            assert first["finish_reason"] == "tool_calls"
            assert first["message"]["content"] == commentary_text
            assert "commentary" not in (first["message"].get("provider_specific_fields") or {})
            assert first["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

        messages.append({"role": "tool", "tool_call_id": "call_sf", "content": "sunny, 21C"})
        second: Final = call()
        assert len(requests_seen) == 2
        assert requests_seen[1]["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"What is the weather in SF? {identity}"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": commentary_text}],
            },
            {"type": "function_call", "call_id": "call_sf", "name": "get_weather", "arguments": '{"city":"SF"}'},
            {
                "type": "function_call_output",
                "call_id": "call_sf",
                "output": [{"type": "input_text", "text": "sunny, 21C"}],
            },
        ], json.dumps(requests_seen[1]["input"])


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_chat_over_responses_deployment_matches_native_chat_shape_across_a_long_conversation(
    gateway: Gateway, stream: bool
) -> None:
    identity: Final = "responses-long-convo-" + uuid.uuid4().hex
    turns: list[Mapping[str, JsonValue]] = []  # mutable-ok: upstream turn counter

    def sse_body(outputs: Sequence[Mapping[str, JsonValue]], turn_number: int) -> bytes:
        events: list[Mapping[str, JsonValue]] = [  # mutable-ok: ordered event list
            {
                "type": "response.created",
                "response": {
                    "id": f"resp_{identity}_{turn_number}",
                    "object": "response",
                    "created_at": 1,
                    "status": "in_progress",
                    "model": "gpt-6-sol",
                },
            }
        ]
        for output_index, item in enumerate(outputs):
            events.append({"type": "response.output_item.added", "output_index": output_index, "item": item})
            if item["type"] == "message":
                for content_index, part in enumerate(item["content"]):
                    events.append(
                        {
                            "type": "response.output_text.delta",
                            "item_id": item["id"],
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": part["text"],
                        }
                    )
        events.append(
            {
                "type": "response.completed",
                "response": {
                    "id": f"resp_{identity}_{turn_number}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": outputs,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            }
        )
        return ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        body: Final = json.loads(request.body)
        turns.append(body["input"])
        outputs: Final = {
            1: [
                {
                    "type": "message",
                    "id": "msg_c1",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Checking SF.", "annotations": []}],
                },
                {
                    "type": "function_call",
                    "id": "fc_sf",
                    "call_id": "call_sf",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                    "status": "completed",
                },
            ],
            2: [
                {
                    "type": "reasoning",
                    "id": "rs_t2",
                    "encrypted_content": "enc_t2",
                    "summary": [{"type": "summary_text", "text": "Planning both lookups."}],
                },
                {
                    "type": "message",
                    "id": "msg_c2",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Now NYC, weather and time.", "annotations": []}],
                },
                {
                    "type": "function_call",
                    "id": "fc_nyc_weather",
                    "call_id": "call_nyc_weather",
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc_nyc_time",
                    "call_id": "call_nyc_time",
                    "name": "get_time",
                    "arguments": '{"city":"NYC"}',
                    "status": "completed",
                },
            ],
            3: [
                {
                    "type": "message",
                    "id": "msg_c3",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Here is the summary.", "annotations": []}],
                },
                {
                    "type": "message",
                    "id": "msg_a3",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": f'{{"done": "{identity}"}}', "annotations": []}],
                },
            ],
        }.get(len(turns)) or [
            {
                "type": "message",
                "id": "msg_plain",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Anything else?", "annotations": []}],
            }
        ]
        if stream:
            return Reply(body=sse_body(outputs, len(turns)), headers={"content-type": "text/event-stream"})
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}_{len(turns)}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": outputs,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        tools: Final = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name}.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
            for name in ("get_weather", "get_time")
        ]
        messages: list[Mapping[str, JsonValue]] = [
            {"role": "user", "content": f"Weather in SF? {identity}"}
        ]  # mutable-ok: conversation grows each turn

        def turn(
            user_message: str | None = None, tool_results: tuple[tuple[str, str], ...] = ()
        ) -> Mapping[str, JsonValue]:
            for call_id, output in tool_results:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
            if user_message is not None:
                messages.append({"role": "user", "content": user_message})
            if stream:
                chunks: list[Mapping[str, JsonValue]] = []  # mutable-ok: stream events accumulate
                with gateway.client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {gateway.key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "stream": True,
                        "cache": {"no-cache": True},
                    },
                ) as response:
                    response_body: Final = response.read()
                    assert response.status_code == 200, response.text
                    chunks.extend(
                        json.loads(line.removeprefix("data: "))
                        for line in response_body.decode().splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    )
                deltas: Final = tuple(choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ())
                merged_tool_calls: dict[int, dict[str, JsonValue]] = {}  # mutable-ok: stream accumulator
                for delta in deltas:
                    for tool_call in delta.get("tool_calls") or ():
                        index = int(tool_call.get("index", 0))
                        entry = merged_tool_calls.setdefault(
                            index,
                            {
                                "id": None,
                                "type": "function",
                                "index": index,
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_call.get("id"):
                            entry["id"] = tool_call["id"]
                        function_part = tool_call.get("function") or {}
                        entry["function"]["name"] += str(function_part.get("name") or "")
                        entry["function"]["arguments"] += str(function_part.get("arguments") or "")
                stream_message: Final = {
                    "role": "assistant",
                    "content": "".join(str(delta.get("content") or "") for delta in deltas) or None,
                    "tool_calls": tuple(merged_tool_calls[index] for index in sorted(merged_tool_calls)) or None,
                    "provider_specific_fields": (
                        {
                            "commentary": "".join(
                                str(delta["provider_specific_fields"]["commentary"])
                                for delta in deltas
                                if isinstance(delta.get("provider_specific_fields"), dict)
                                and isinstance(delta["provider_specific_fields"].get("commentary"), str)
                            )
                        }
                        if any(
                            isinstance(delta.get("provider_specific_fields"), dict)
                            and "commentary" in delta["provider_specific_fields"]
                            for delta in deltas
                        )
                        else None
                    ),
                    "reasoning_items": next(
                        (delta["reasoning_items"] for delta in deltas if delta.get("reasoning_items")), None
                    ),
                    "finish_reason": next(
                        (
                            choice["finish_reason"]
                            for chunk in reversed(chunks)
                            for choice in chunk.get("choices") or ()
                            if choice.get("finish_reason")
                        ),
                        None,
                    ),
                }
                messages.append(
                    {
                        k: v
                        for k, v in stream_message.items()
                        if k in ("role", "content", "tool_calls", "provider_specific_fields", "reasoning_items")
                        and v is not None
                    }
                )
                return {"finish_reason": stream_message["finish_reason"], "message": stream_message}
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": messages, "tools": tools, "cache": {"no-cache": True}},
            )
            assert response.status_code == 200, response.text
            choice: Final[Mapping[str, JsonValue]] = response.json()["choices"][0]
            message: Final[Mapping[str, JsonValue]] = choice["message"]
            messages.append(
                {
                    k: v
                    for k, v in message.items()
                    if k in ("role", "content", "tool_calls", "provider_specific_fields", "reasoning_items")
                }
            )
            return choice

        first: Final = turn()
        assert first["finish_reason"] == "tool_calls"
        assert first["message"]["content"] == "Checking SF."
        assert "commentary" not in (first["message"].get("provider_specific_fields") or {})

        second: Final = turn(user_message="And NYC, weather plus time", tool_results=(("call_sf", "sunny, 21C"),))
        assert second["finish_reason"] == "tool_calls"
        assert second["message"]["content"] == "Now NYC, weather and time."
        assert "commentary" not in (second["message"].get("provider_specific_fields") or {})
        assert [tc["function"]["name"] for tc in second["message"]["tool_calls"]] == ["get_weather", "get_time"]

        third: Final = turn(
            tool_results=(("call_nyc_weather", "cloudy, 15C"), ("call_nyc_time", "14:00")),
        )
        assert third["finish_reason"] == "stop"
        assert third["message"]["content"] == f'{{"done": "{identity}"}}'
        assert third["message"]["provider_specific_fields"] == {"commentary": "Here is the summary."}

        fourth: Final = turn(user_message="Anything else?")
        assert fourth["finish_reason"] == "stop"
        assert fourth["message"]["content"] == "Anything else?"

        def content_message(role: str, text: str) -> dict[str, JsonValue]:
            return {
                "type": "message",
                "role": role,
                "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}],
            }

        def function_call(call_id: str, name: str, args: str) -> dict[str, JsonValue]:
            return {"type": "function_call", "call_id": call_id, "name": name, "arguments": args}

        def call_output(call_id: str, text: str) -> dict[str, JsonValue]:
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": [{"type": "input_text", "text": text}],
            }

        user_sf: Final = content_message("user", f"Weather in SF? {identity}")
        assert len(turns) == 4
        assert turns[0] == [user_sf]
        assert turns[1] == [
            user_sf,
            content_message("assistant", "Checking SF."),
            function_call("call_sf", "get_weather", '{"city":"SF"}'),
            call_output("call_sf", "sunny, 21C"),
            content_message("user", "And NYC, weather plus time"),
        ]
        reasoning_replay: Final = (
            [
                {
                    "type": "reasoning",
                    "id": "rs_t2",
                    "summary": [{"type": "summary_text", "text": "Planning both lookups."}],
                    "encrypted_content": "enc_t2",
                }
            ]
            if not stream
            else []
        )
        expected_turn3: Final = [
            *turns[1],
            *reasoning_replay,
            content_message("assistant", "Now NYC, weather and time."),
            function_call("call_nyc_weather", "get_weather", '{"city":"NYC"}'),
            function_call("call_nyc_time", "get_time", '{"city":"NYC"}'),
            call_output("call_nyc_weather", "cloudy, 15C"),
            call_output("call_nyc_time", "14:00"),
        ]
        # stream clients cannot see delta.reasoning_items today (the generic stream
        # handler rebuilds deltas without it), so nothing is replayed for them
        assert turns[2] == expected_turn3
        assert turns[3] == [
            *expected_turn3,
            content_message("assistant", f'{{"done": "{identity}"}}'),
            content_message("user", "Anything else?"),
        ]


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_cursor_chat_completions_route_folds_commentary_into_provider_specific_fields(
    gateway: Gateway, stream: bool
) -> None:
    identity: Final = "cursor-commentary-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'
    stream_events: Final = (
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
                "id": "msg_commentary",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "commentary",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_commentary",
            "output_index": 0,
            "content_index": 0,
            "delta": "Let me answer that.",
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "msg_answer",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "final_answer",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_answer",
            "output_index": 1,
            "content_index": 0,
            "delta": answer,
        },
        {
            "type": "response.completed",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        if stream:
            sse: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in stream_events) + "data: [DONE]\n\n"
            return Reply(body=sse.encode(), headers={"content-type": "text/event-stream"})
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
                            "id": "msg_commentary",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [{"type": "output_text", "text": "Let me answer that.", "annotations": []}],
                        },
                        {
                            "type": "message",
                            "id": "msg_answer",
                            "status": "completed",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": answer, "annotations": []}],
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    request_body: Final = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"What is the weather in Paris? {identity}"}],
            }
        ],
        "stream": stream,
    }

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        if stream:
            with gateway.client.stream(
                "POST",
                "/cursor/chat/completions",
                headers={"Authorization": f"Bearer {gateway.key}"},
                json={**request_body, "model": model},
            ) as response:
                response_body: Final = response.read()
                assert response.status_code == 200, response.text
                chunks: Final = tuple(
                    json.loads(line.removeprefix("data: "))
                    for line in response_body.decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                )
                deltas: Final = tuple(choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ())
                assert "".join(str(delta.get("content") or "") for delta in deltas) == answer, response_body.decode()
                assert (
                    "".join(
                        str(delta["provider_specific_fields"]["commentary"])
                        for delta in deltas
                        if isinstance(delta.get("provider_specific_fields"), dict)
                        and "commentary" in delta["provider_specific_fields"]
                    )
                    == "Let me answer that."
                ), response_body.decode()
        else:
            response = gateway.request("POST", "/cursor/chat/completions", {**request_body, "model": model})
            assert response.status_code == 200, response.text
            body: Final = response.json()
            assert body["choices"] == [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {"commentary": "Let me answer that."},
                    },
                }
            ], response.text


def test_chat_over_responses_deployment_end_of_stream_blank_commentary_rewrite(gateway: Gateway) -> None:
    identity: Final = "responses-commentary-blank-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'
    commentary_text: Final = "Let me check the weather first."
    guardrail_name: Final = f"blank-commentary-{uuid.uuid4().hex[:8]}"

    stream_events: Final = (
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
                "id": "msg_commentary",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "commentary",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_commentary",
            "output_index": 0,
            "content_index": 0,
            "delta": commentary_text,
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "msg_answer",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "final_answer",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_answer",
            "output_index": 1,
            "content_index": 0,
            "delta": answer,
        },
        {
            "type": "response.completed",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        if request.target == "/beta/litellm_basic_guardrail_api":
            checked: Final = json.loads(request.body).get("texts") or []
            return Reply(
                body=json.dumps(
                    {
                        "action": "GUARDRAIL_INTERVENED",
                        "texts": ["" if text == commentary_text else text for text in checked],
                    }
                ).encode()
            )
        assert request.method == "POST" and request.target == "/responses", request.target
        sse: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in stream_events) + "data: [DONE]\n\n"
        return Reply(body=sse.encode(), headers={"content-type": "text/event-stream"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.client.post(
            "/guardrails",
            headers={"x-litellm-api-key": gateway.key},
            json={
                "guardrail": {
                    "guardrail_name": guardrail_name,
                    "litellm_params": {
                        "guardrail": "generic_guardrail_api",
                        "mode": "post_call",
                        "default_on": False,
                        "api_base": wire.url,
                        "api_key": "synthetic-guardrail-key",
                    },
                }
            },
        )
        assert created.status_code == 200, created.text
        guardrail_id: Final = created.json()["guardrail_id"]
        policy_name: Final = f"blank-commentary-policy-{uuid.uuid4().hex[:8]}"
        policy: Final = gateway.client.post(
            "/policies",
            headers={"x-litellm-api-key": gateway.key},
            json={
                "policy_name": policy_name,
                "guardrails_add": [guardrail_name],
                "pipeline": {
                    "mode": "post_call",
                    "steps": [{"guardrail": guardrail_name, "on_pass": "next", "on_fail": "block"}],
                },
            },
        )
        assert policy.status_code == 200, policy.text
        policy_id: Final = policy.json()["policy_id"]
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        attachment: Final = gateway.client.post(
            "/policies/attachments",
            headers={"x-litellm-api-key": gateway.key},
            json={"policy_name": policy_name, "models": [model]},
        )
        assert attachment.status_code == 200, attachment.text
        attachment_id: Final = attachment.json()["attachment_id"]
        try:
            with gateway.client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {gateway.key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                    "stream": True,
                    "cache": {"no-cache": True},
                },
            ) as response:
                response_body: Final = response.read()
                assert response.status_code == 200, response.text
                chunks: Final = tuple(
                    json.loads(line.removeprefix("data: "))
                    for line in response_body.decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                )
                deltas: Final = tuple(choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ())
                assert "".join(str(delta.get("content") or "") for delta in deltas) == answer, response_body.decode()
                assert all(
                    delta["provider_specific_fields"]["commentary"] in (None, "")
                    for delta in deltas
                    if isinstance(delta.get("provider_specific_fields"), dict)
                    and "commentary" in delta["provider_specific_fields"]
                ), response_body.decode()
                assert commentary_text not in response_body.decode(), response_body.decode()
        finally:
            gateway.client.delete(f"/policies/attachments/{attachment_id}", headers={"x-litellm-api-key": gateway.key})
            gateway.client.delete(f"/policies/{policy_id}", headers={"x-litellm-api-key": gateway.key})
            deleted: Final = gateway.client.delete(
                f"/guardrails/{guardrail_id}", headers={"x-litellm-api-key": gateway.key}
            )
            assert deleted.status_code == 200, deleted.text


def test_chat_over_responses_deployment_conversation_switching_between_chat_and_responses(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-switching-" + uuid.uuid4().hex
    hits: list[tuple[str, Mapping[str, JsonValue]]] = []  # mutable-ok: upstream recorder

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target in ("/responses", "/chat/completions"), request.target
        body: Final = json.loads(request.body)
        hits.append((request.target, body))
        if request.target == "/chat/completions":
            return Reply(
                body=json.dumps(
                    {
                        "id": f"chatcmpl_{identity}",
                        "object": "chat.completion",
                        "created": 1789788253,
                        "model": "gpt-6-sol",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Plain chat answer."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                    }
                ).encode()
            )
        responses_count: Final = sum(1 for target, _ in hits if target == "/responses")
        outputs: Final = {
            1: [
                {
                    "type": "message",
                    "id": "msg_s1",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Checking SF.", "annotations": []}],
                },
                {
                    "type": "function_call",
                    "id": "fc_sf",
                    "call_id": "call_sf",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                    "status": "completed",
                },
            ],
            2: [
                {
                    "type": "message",
                    "id": "msg_s2",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Here is the summary.", "annotations": []}],
                },
                {
                    "type": "message",
                    "id": "msg_s3",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": f'{{"done": "{identity}"}}', "annotations": []}],
                },
            ],
        }.get(
            responses_count,
            [
                {
                    "type": "message",
                    "id": "msg_s4",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done for now.", "annotations": []}],
                }
            ],
        )
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}_{responses_count}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": outputs,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key")
        tools: Final = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ]
        messages: list[Mapping[str, JsonValue]] = [
            {"role": "user", "content": f"Weather in SF? {identity}"}
        ]  # mutable-ok: conversation grows each turn

        def call(with_tools: bool) -> Mapping[str, JsonValue]:
            payload: Final = {
                "model": model,
                "messages": messages,
                "reasoning_effort": "low",
                "cache": {"no-cache": True},
            }
            if with_tools:
                payload["tools"] = tools
            response: Final = gateway.request("POST", "/v1/chat/completions", payload)
            assert response.status_code == 200, response.text
            choice: Final[Mapping[str, JsonValue]] = response.json()["choices"][0]
            message: Final[Mapping[str, JsonValue]] = choice["message"]
            messages.append(
                {
                    k: v
                    for k, v in message.items()
                    if k in ("role", "content", "tool_calls", "provider_specific_fields", "reasoning_items")
                }
            )
            return choice

        call(with_tools=True)
        messages.append({"role": "tool", "tool_call_id": "call_sf", "content": "sunny, 21C"})
        call(with_tools=True)
        messages.append({"role": "user", "content": "Thanks, anything else?"})
        call(with_tools=False)
        messages.append({"role": "user", "content": "Wrap up."})
        call(with_tools=True)

        assert [target for target, _ in hits] == [
            "/responses",
            "/responses",
            "/chat/completions",
            "/responses",
        ], [target for target, _ in hits]
        assert len(hits) == 4


def test_chat_over_responses_deployment_guardrail_blocks_late_commentary_delta(gateway: Gateway) -> None:
    identity: Final = "responses-late-commentary-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'
    secret: Final = f"{identity}-secret-token"
    guardrail_name: Final = f"block-late-commentary-{uuid.uuid4().hex[:8]}"

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
                "id": "msg_answer",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "final_answer",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_answer",
            "output_index": 0,
            "content_index": 0,
            "delta": answer,
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "msg_commentary",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "commentary",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_commentary",
            "output_index": 1,
            "content_index": 0,
            "delta": f"note with {secret}",
        },
        {
            "type": "response.completed",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        if request.target == "/beta/litellm_basic_guardrail_api":
            checked: Final = json.loads(request.body).get("texts") or []
            if any(secret in text for text in checked):
                return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "secret in output"}).encode())
            return Reply(body=b'{"action": "NONE"}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.client.post(
            "/guardrails",
            headers={"x-litellm-api-key": gateway.key},
            json={
                "guardrail": {
                    "guardrail_name": guardrail_name,
                    "litellm_params": {
                        "guardrail": "generic_guardrail_api",
                        "mode": "post_call",
                        "default_on": True,
                        "streaming_sampling_rate": 1,
                        "api_base": wire.url,
                        "api_key": "synthetic-guardrail-key",
                    },
                }
            },
        )
        assert created.status_code == 200, created.text
        guardrail_id: Final = created.json()["guardrail_id"]
        try:
            model: Final = scenario.model(
                model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
            )
            with gateway.client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {gateway.key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                    "stream": True,
                    "cache": {"no-cache": True},
                },
            ) as response:
                response_body: Final = response.read()
                assert response.status_code == 200, response.text
                body_text: Final = response_body.decode()
                assert secret not in body_text, body_text
                assert "note with" not in body_text, body_text
        finally:
            deleted: Final = gateway.client.delete(
                f"/guardrails/{guardrail_id}", headers={"x-litellm-api-key": gateway.key}
            )
            assert deleted.status_code == 200, deleted.text


def test_chat_over_responses_deployment_incremental_diff_masks_commentary(gateway: Gateway) -> None:
    identity: Final = "responses-incr-commentary-" + uuid.uuid4().hex
    answer: Final = f'{{"city": "Paris", "marker": "{identity}"}}'
    commentary_text: Final = f"Let me check for {identity}-user@example.com first"
    guardrail_name: Final = f"incr-commentary-{uuid.uuid4().hex[:8]}"

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
                "id": "msg_commentary",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "commentary",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_commentary",
            "output_index": 0,
            "content_index": 0,
            "delta": commentary_text,
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "msg_answer",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "final_answer",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_answer",
            "output_index": 1,
            "content_index": 0,
            "delta": answer,
        },
        {
            "type": "response.completed",
            "response": {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        if request.target == "/beta/litellm_basic_guardrail_api":
            checked: Final = json.loads(request.body).get("texts") or []
            return Reply(
                body=json.dumps(
                    {
                        "action": "GUARDRAIL_INTERVENED",
                        "texts": [text.replace(f"{identity}-user@example.com", "[EMAIL_REDACTED]") for text in checked],
                    }
                ).encode()
            )
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.client.post(
            "/guardrails",
            headers={"x-litellm-api-key": gateway.key},
            json={
                "guardrail": {
                    "guardrail_name": guardrail_name,
                    "litellm_params": {
                        "guardrail": "generic_guardrail_api",
                        "mode": "post_call",
                        "default_on": True,
                        "streaming_transform_mode": "incremental_diff",
                        "api_base": wire.url,
                        "api_key": "synthetic-guardrail-key",
                    },
                }
            },
        )
        assert created.status_code == 200, created.text
        guardrail_id: Final = created.json()["guardrail_id"]
        try:
            model: Final = scenario.model(
                model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
            )
            with gateway.client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {gateway.key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                    "stream": True,
                    "cache": {"no-cache": True},
                },
            ) as response:
                response_body: Final = response.read()
                assert response.status_code == 200, response.text
                chunks: Final = tuple(
                    json.loads(line.removeprefix("data: "))
                    for line in response_body.decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                )
                deltas: Final = tuple(choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ())
                assert "".join(str(delta.get("content") or "") for delta in deltas) == answer, response_body.decode()
                assert (
                    "".join(
                        str(object_value(delta["provider_specific_fields"])["commentary"])
                        for delta in deltas
                        if isinstance(delta.get("provider_specific_fields"), dict)
                        and "commentary" in delta["provider_specific_fields"]
                    )
                    == "Let me check for [EMAIL_REDACTED] first"
                ), response_body.decode()
        finally:
            deleted: Final = gateway.client.delete(
                f"/guardrails/{guardrail_id}", headers={"x-litellm-api-key": gateway.key}
            )
            assert deleted.status_code == 200, deleted.text
