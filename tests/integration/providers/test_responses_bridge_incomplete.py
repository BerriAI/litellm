import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request, wire_server


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


@pytest.mark.covers("other.provider_wire.responses_bridge.commentary_phase_moves_to_provider_specific_fields")
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


@pytest.mark.covers("other.provider_wire.responses_bridge.streamed_commentary_joined_and_assembled")
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
                        "content": [
                            {"type": "output_text", "text": "Let me answer that.", "annotations": []}
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
            deltas: Final = tuple(
                choice["delta"] for chunk in chunks for choice in chunk.get("choices") or ()
            )
            assert deltas, response_body.decode()
            assert "".join(str(delta.get("content") or "") for delta in deltas) == answer, response_body.decode()
            assert "".join(
                str(object_value(delta["provider_specific_fields"])["commentary"])
                for delta in deltas
                if isinstance(delta.get("provider_specific_fields"), dict)
                and "commentary" in delta["provider_specific_fields"]
            ) == "Let me answer that.", response_body.decode()


@pytest.mark.covers("other.provider_wire.responses_bridge.output_guardrail_masks_commentary")
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
                            "content": [
                                {"type": "output_text", "text": commentary_text, "annotations": []}
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
                            "messages": [
                                {"role": "user", "content": f"What is the weather in Paris? {identity}"}
                            ],
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
                        assert "".join(
                            str(object_value(delta["provider_specific_fields"])["commentary"])
                            for delta in deltas
                            if isinstance(delta.get("provider_specific_fields"), dict)
                            and "commentary" in delta["provider_specific_fields"]
                        ) == "Let me check for [EMAIL_REDACTED] first", response_body.decode()
                else:
                    response = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [
                                {"role": "user", "content": f"What is the weather in Paris? {identity}"}
                            ],
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
                                "provider_specific_fields": {
                                    "commentary": "Let me check for [EMAIL_REDACTED] first"
                                },
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
