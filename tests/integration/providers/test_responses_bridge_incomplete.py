import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


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


_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
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


def _message_item(item_id: str, text: str, phase: str | None = None) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        **({} if phase is None else {"phase": phase}),
    }


_WEATHER_CALL: Final[dict[str, JsonValue]] = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city":"Paris"}',
    "status": "completed",
}

_TWO_MESSAGES_THEN_CALL: Final = (
    _message_item("msg_first", "First message."),
    _message_item("msg_second", "Second message."),
    _WEATHER_CALL,
)


def test_chat_over_responses_deployment_folds_two_messages_and_function_call_into_one_choice(
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
                    "id": "resp_two_messages",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": list(_TWO_MESSAGES_THEN_CALL),
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
                "tools": [_WEATHER_TOOL],
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
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


_COMMENTARY_THEN_FINAL_ANSWER: Final = (
    _message_item("msg_commentary", "Let me answer that.", phase="commentary"),
    _message_item("msg_final", '{"city": "Paris", "summary": "Sunny, 21C"}', phase="final_answer"),
)


def test_chat_over_responses_deployment_folds_commentary_and_final_answer_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-commentary-final-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_commentary_final",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": list(_COMMENTARY_THEN_FINAL_ANSWER),
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
                "messages": [{"role": "user", "content": f"Weather in Paris as JSON. {identity}"}],
                "tools": [_WEATHER_TOOL],
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
                    "content": 'Let me answer that.{"city": "Paris", "summary": "Sunny, 21C"}',
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


_REASONING_ITEM: Final[dict[str, JsonValue]] = {
    "type": "reasoning",
    "id": "rs_fold",
    "summary": [{"type": "summary_text", "text": "Checking the forecast."}],
}

_TIME_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the local time for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_TIME_CALL: Final[dict[str, JsonValue]] = {
    "type": "function_call",
    "id": "fc_2",
    "call_id": "call_2",
    "name": "get_time",
    "arguments": '{"city":"Paris","tz":"CET"}',
    "status": "completed",
}

_REASONING_THEN_TWO_MESSAGES_THEN_CALL: Final = (
    _REASONING_ITEM,
    _message_item("msg_first", "First message."),
    _message_item("msg_second", "Second message."),
    _WEATHER_CALL,
)

_MESSAGE_THEN_TWO_CALLS: Final = (
    _message_item("msg_only", "Checking both."),
    _WEATHER_CALL,
    _TIME_CALL,
)

_TWO_CALLS_CHAT_SHAPE: Final = [
    {
        "id": "fc_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        "index": 0,
    },
    {
        "id": "fc_2",
        "type": "function",
        "function": {"name": "get_time", "arguments": '{"city":"Paris","tz":"CET"}'},
        "index": 1,
    },
]


def _responses_body(response_id: str, output: tuple[dict[str, JsonValue], ...]) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": list(output),
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def test_chat_over_responses_deployment_folds_reasoning_two_messages_and_function_call_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-reasoning-two-messages-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(body=_responses_body("resp_reasoning_two_messages", _REASONING_THEN_TWO_MESSAGES_THEN_CALL))

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
                "tools": [_WEATHER_TOOL],
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
                    "reasoning_content": "Checking the forecast.",
                    "reasoning_items": [_REASONING_ITEM],
                    "tool_calls": _TWO_CALLS_CHAT_SHAPE[:1],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def test_chat_over_responses_deployment_folds_message_and_two_function_calls_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-two-calls-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(body=_responses_body("resp_two_calls", _MESSAGE_THEN_TWO_CALLS))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Weather and time in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL, _TIME_TOOL],
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
                    "content": "Checking both.",
                    "tool_calls": _TWO_CALLS_CHAT_SHAPE,
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def _responses_frame(event: str, payload: dict[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _message_item_frames(output_index: int, item: dict[str, JsonValue]) -> tuple[bytes, ...]:
    item_id: Final = item["id"]
    text: Final = object_value(list_value(item["content"])[0])["text"]
    return (
        _responses_frame(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "content": []},
            },
        ),
        _responses_frame(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        _responses_frame(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": output_index, "item": item},
        ),
    )


def _function_call_frames(output_index: int, item: dict[str, JsonValue]) -> tuple[bytes, ...]:
    return (
        _responses_frame(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            },
        ),
        _responses_frame(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": output_index,
                "delta": item["arguments"],
            },
        ),
        _responses_frame(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": output_index,
                "arguments": item["arguments"],
            },
        ),
        _responses_frame(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": output_index, "item": item},
        ),
    )


def _reasoning_item_frames(output_index: int, item: dict[str, JsonValue]) -> tuple[bytes, ...]:
    text: Final = object_value(list_value(item["summary"])[0])["text"]
    return (
        _responses_frame(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "summary": []},
            },
        ),
        _responses_frame(
            "response.reasoning_summary_text.delta",
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": item["id"],
                "output_index": output_index,
                "summary_index": 0,
                "delta": text,
            },
        ),
        _responses_frame(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": output_index, "item": item},
        ),
    )


def _sse_payloads(lines: list[str]) -> list[dict[str, JsonValue]]:
    return [json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]"]


def test_chat_stream_over_responses_deployment_folds_two_messages_and_function_call_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-two-messages-stream-" + uuid.uuid4().hex
    response_payload: Final[dict[str, JsonValue]] = {
        "id": "resp_two_messages_stream",
        "object": "response",
        "created_at": 1789788253,
        "model": "gpt-6-sol",
        "status": "in_progress",
        "output": [],
    }

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True, request.body
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _responses_frame("response.created", {"type": "response.created", "response": response_payload}),
                *_message_item_frames(0, _TWO_MESSAGES_THEN_CALL[0]),
                *_message_item_frames(1, _TWO_MESSAGES_THEN_CALL[1]),
                *_function_call_frames(2, _TWO_MESSAGES_THEN_CALL[2]),
                _responses_frame(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            **response_payload,
                            "status": "completed",
                            "output": list(_TWO_MESSAGES_THEN_CALL),
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                        },
                    },
                ),
            ),
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
                "tools": [_WEATHER_TOOL],
                "stream": True,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        chunks: Final = _sse_payloads(list(response.iter_lines()))
        choices: Final = [object_value(choice) for chunk in chunks for choice in list_value(chunk["choices"])]
        assert [choice["index"] for choice in choices] == [0] * len(choices), chunks
        deltas: Final = [object_value(choice["delta"]) for choice in choices]
        assert (
            "".join(str(delta["content"]) for delta in deltas if delta.get("content"))
            == "First message.Second message."
        )
        tool_calls: Final = [
            object_value(call)
            for delta in deltas
            if delta.get("tool_calls")
            for call in list_value(delta["tool_calls"])
        ]
        assert [
            object_value(call["function"])["name"] for call in tool_calls if object_value(call["function"]).get("name")
        ] == ["get_weather"], chunks
        assert (
            "".join(str(object_value(call["function"]).get("arguments") or "") for call in tool_calls)
            == '{"city":"Paris"}'
        ), chunks
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["tool_calls"], chunks
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def test_chat_stream_over_responses_deployment_folds_commentary_and_final_answer_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-commentary-final-stream-" + uuid.uuid4().hex
    response_payload: Final[dict[str, JsonValue]] = {
        "id": "resp_commentary_final_stream",
        "object": "response",
        "created_at": 1789788253,
        "model": "gpt-6-sol",
        "status": "in_progress",
        "output": [],
    }

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True, request.body
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _responses_frame("response.created", {"type": "response.created", "response": response_payload}),
                *_message_item_frames(0, _COMMENTARY_THEN_FINAL_ANSWER[0]),
                *_message_item_frames(1, _COMMENTARY_THEN_FINAL_ANSWER[1]),
                _responses_frame(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            **response_payload,
                            "status": "completed",
                            "output": list(_COMMENTARY_THEN_FINAL_ANSWER),
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                        },
                    },
                ),
            ),
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
                "messages": [{"role": "user", "content": f"Weather in Paris as JSON. {identity}"}],
                "tools": [_WEATHER_TOOL],
                "stream": True,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        chunks: Final = _sse_payloads(list(response.iter_lines()))
        choices: Final = [object_value(choice) for chunk in chunks for choice in list_value(chunk["choices"])]
        assert [choice["index"] for choice in choices] == [0] * len(choices), chunks
        deltas: Final = [object_value(choice["delta"]) for choice in choices]
        assert (
            "".join(str(delta["content"]) for delta in deltas if delta.get("content"))
            == 'Let me answer that.{"city": "Paris", "summary": "Sunny, 21C"}'
        ), chunks
        assert not [delta for delta in deltas if delta.get("tool_calls")], chunks
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["stop"], chunks
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def _stream_reply(response_id: str, output: tuple[dict[str, JsonValue], ...], frames: tuple[bytes, ...]) -> Reply:
    response_payload: Final[dict[str, JsonValue]] = {
        "id": response_id,
        "object": "response",
        "created_at": 1789788253,
        "model": "gpt-6-sol",
        "status": "in_progress",
        "output": [],
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _responses_frame("response.created", {"type": "response.created", "response": response_payload}),
            *frames,
            _responses_frame(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        **response_payload,
                        "status": "completed",
                        "output": list(output),
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                },
            ),
        ),
    )


def _stream_choices(response: httpx.Response) -> list[dict[str, JsonValue]]:
    chunks: Final = _sse_payloads(list(response.iter_lines()))
    return [object_value(choice) for chunk in chunks for choice in list_value(chunk["choices"])]


def _stream_tool_calls(deltas: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [
        object_value(call) for delta in deltas if delta.get("tool_calls") for call in list_value(delta["tool_calls"])
    ]


def test_chat_stream_over_responses_deployment_folds_reasoning_two_messages_and_function_call_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-reasoning-two-messages-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True, request.body
        return _stream_reply(
            "resp_reasoning_two_messages_stream",
            _REASONING_THEN_TWO_MESSAGES_THEN_CALL,
            (
                *_reasoning_item_frames(0, _REASONING_THEN_TWO_MESSAGES_THEN_CALL[0]),
                *_message_item_frames(1, _REASONING_THEN_TWO_MESSAGES_THEN_CALL[1]),
                *_message_item_frames(2, _REASONING_THEN_TWO_MESSAGES_THEN_CALL[2]),
                *_function_call_frames(3, _REASONING_THEN_TWO_MESSAGES_THEN_CALL[3]),
            ),
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
                "tools": [_WEATHER_TOOL],
                "stream": True,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        choices: Final = _stream_choices(response)
        assert [choice["index"] for choice in choices] == [0] * len(choices), choices
        deltas: Final = [object_value(choice["delta"]) for choice in choices]
        assert (
            "".join(str(delta["reasoning_content"]) for delta in deltas if delta.get("reasoning_content"))
            == "Checking the forecast."
        ), choices
        assert (
            "".join(str(delta["content"]) for delta in deltas if delta.get("content"))
            == "First message.Second message."
        ), choices
        tool_calls: Final = _stream_tool_calls(deltas)
        assert [
            object_value(call["function"])["name"] for call in tool_calls if object_value(call["function"]).get("name")
        ] == ["get_weather"], choices
        assert (
            "".join(str(object_value(call["function"]).get("arguments") or "") for call in tool_calls)
            == '{"city":"Paris"}'
        ), choices
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["tool_calls"], choices
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def test_chat_stream_over_responses_deployment_folds_message_and_two_function_calls_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-two-calls-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True, request.body
        return _stream_reply(
            "resp_two_calls_stream",
            _MESSAGE_THEN_TWO_CALLS,
            (
                *_message_item_frames(0, _MESSAGE_THEN_TWO_CALLS[0]),
                *_function_call_frames(1, _MESSAGE_THEN_TWO_CALLS[1]),
                *_function_call_frames(2, _MESSAGE_THEN_TWO_CALLS[2]),
            ),
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
                "messages": [{"role": "user", "content": f"Weather and time in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL, _TIME_TOOL],
                "stream": True,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        choices: Final = _stream_choices(response)
        assert [choice["index"] for choice in choices] == [0] * len(choices), choices
        deltas: Final = [object_value(choice["delta"]) for choice in choices]
        assert "".join(str(delta["content"]) for delta in deltas if delta.get("content")) == "Checking both.", choices
        tool_calls: Final = _stream_tool_calls(deltas)
        assert [
            (call["index"], object_value(call["function"])["name"])
            for call in tool_calls
            if object_value(call["function"]).get("name")
        ] == [(0, "get_weather"), (1, "get_time")], choices
        arguments_by_index: Final = {
            index: "".join(
                str(object_value(call["function"]).get("arguments") or "")
                for call in tool_calls
                if call["index"] == index
            )
            for index in (0, 1)
        }
        assert arguments_by_index == {0: '{"city":"Paris"}', 1: '{"city":"Paris","tz":"CET"}'}, choices
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["tool_calls"], choices
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
