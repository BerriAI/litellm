import json
import uuid
from typing import Final

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


def _message_item(item_id: str, text: str) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
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
