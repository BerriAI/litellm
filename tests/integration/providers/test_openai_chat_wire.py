import json
import uuid
from itertools import chain
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_GPT_6_MODELS: Final = ("gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol")
_API_KEY: Final = "synthetic-openai-key"
_PROMPT: Final = "Summarize this conversation in one sentence."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _completion(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
        }
    ).encode()


def _tool_completion(model_name: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-weather",
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
                                "id": "call_1",
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

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        assert "tool_choice" not in body, body
        assert "tools" not in body, body
        return Reply(body=_completion(identity, "One sentence."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "tool_choice": "none"},
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
def test_azure_gpt_6_function_tool_with_reasoning_effort_none_stays_on_chat(gateway: Gateway, model_name: str) -> None:
    identity: Final = f"azure-{model_name}-{uuid.uuid4().hex}"
    upstream_target: Final = f"/openai/deployments/{model_name}/chat/completions?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == upstream_target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["messages"] == [{"role": "user", "content": f"What is the weather in Paris? {identity}"}]
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
        assert body["reasoning_effort"] == "none"
        return Reply(body=_tool_completion(model_name))

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
                "reasoning_effort": "none",
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather.",
                    "tool_calls": [
                        {
                            "id": "call_1",
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
def test_azure_gpt_6_function_tool_without_reasoning_effort_bridges_to_responses(
    gateway: Gateway, model_name: str
) -> None:
    identity: Final = f"azure-{model_name}-{uuid.uuid4().hex}"
    upstream_target: Final = "/openai/responses?api-version=2025-04-01-preview"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == upstream_target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["tools"] == [
            {
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
        ]
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_weather",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": model_name,
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
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", upstream_target)]


@pytest.mark.parametrize("model_name", _GPT_6_MODELS, ids=_GPT_6_MODELS)
def test_openai_custom_base_gpt_6_function_tool_without_reasoning_effort_stays_on_chat(
    gateway: Gateway, model_name: str
) -> None:
    identity: Final = f"openai-{model_name}-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == model_name
        assert body["messages"] == [{"role": "user", "content": f"What is the weather in Paris? {identity}"}]
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
        return Reply(body=_tool_completion(model_name))

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
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather.",
                    "tool_calls": [
                        {
                            "id": "call_1",
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


def test_openai_custom_base_gpt_6_function_tool_with_low_effort_bridges_to_responses(gateway: Gateway) -> None:
    identity: Final = f"openai-gpt-6-sol-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["reasoning"]["effort"] == "low"
        assert body["tools"] == [
            {
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
        ]
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
        model: Final = scenario.model(model="openai/gpt-6-sol", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "reasoning_effort": "low",
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


def test_azure_gpt_6_bridged_stream_returns_text_and_tool_call_on_one_choice(gateway: Gateway) -> None:
    identity: Final = f"azure-gpt-6-sol-stream-{uuid.uuid4().hex}"
    expected_text: Final = "Let me check the weather."
    events: Final = (
        {
            "type": "response.created",
            "response": {
                "id": "resp_weather",
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
                "id": "resp_weather",
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
