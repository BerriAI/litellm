import json
import uuid
from collections.abc import Mapping
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, held
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

_SYSTEM_INSTRUCTION: Final = {"parts": [{"text": "use the tools"}]}
_RESPONSE_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_GENERATION_CONFIG: Final = {
    "maxOutputTokens": 50,
    "temperature": 0.1,
    "stopSequences": ["END"],
    "responseMimeType": "application/json",
    "responseSchema": _RESPONSE_SCHEMA,
}
_PREFIXES: Final = ("/v1beta/models", "/models")
_SCHEMA_KEYS: Final = ("responseSchema", "responseJsonSchema")
_IMAGE_DATA: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
_TOOLS: Final = [
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            {
                "name": "get_time",
                "parametersJsonSchema": {
                    "type": "object",
                    "properties": {"zone": {"type": "string"}},
                },
            },
        ]
    }
]

_REQUEST: Final = {
    "contents": [],
    "systemInstruction": _SYSTEM_INSTRUCTION,
    "generationConfig": _GENERATION_CONFIG,
    "tools": _TOOLS,
    "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
    "cache": {"no-cache": True},
}

_EXPECTED_TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}},
        },
    },
]

_EXPECTED_ANTHROPIC_TOOLS: Final = [
    {
        "name": "get_weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "type": "custom",
    },
    {
        "name": "get_time",
        "input_schema": {"type": "object", "properties": {"zone": {"type": "string"}}},
        "type": "custom",
    },
]

_OPENAI_REPLY: Final = {
    "id": "chatcmpl-wire1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "checking now",
                "tool_calls": [
                    {
                        "id": "call_weather1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
}

_ANTHROPIC_REPLY: Final = {
    "id": "msg_wire1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [
        {"type": "text", "text": "checking now"},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 9},
}

_EXPECTED_CANDIDATES: Final = [
    {
        "content": {
            "parts": [
                {"text": "checking now"},
                {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
            ],
            "role": "model",
        },
        "finishReason": "STOP",
        "index": 0,
        "safetyRatings": [],
    }
]
_EXPECTED_USAGE: Final = {"promptTokenCount": 20, "candidatesTokenCount": 9, "totalTokenCount": 29}


def _openai_stream_frames() -> tuple[bytes, ...]:
    def frame(chunk: dict) -> bytes:
        return f"data: {json.dumps(chunk)}\n\n".encode()

    base: Final = {
        "id": "chatcmpl-wire2",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
    }
    return (
        frame(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "checking "}, "finish_reason": None}
                ],
            }
        ),
        frame(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_weather1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"ci'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        frame(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty":"SF"}'}}]},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        frame(
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        ),
        frame({**base, "choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}}),
        b"data: [DONE]\n\n",
    )


def _expected_gemini_stream_frames(usage: Mapping[str, int] | None) -> list[dict]:
    return [
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "checking "}], "role": "model"},
                    "finishReason": None,
                    "index": 0,
                    "safetyRatings": [],
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}],
                        "role": "model",
                    },
                    "finishReason": None,
                    "index": 0,
                    "safetyRatings": [],
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            **({"usageMetadata": dict(usage)} if usage is not None else {}),
        },
    ]


def _contents(marker: str) -> list[dict]:
    return [
        {"role": "user", "parts": [{"text": f"what is the weather in SF {marker}"}]},
        {"role": "model", "parts": [{"text": "let me check"}]},
        {"role": "user", "parts": [{"text": "and the time"}]},
    ]


def _request(marker: str, schema_key: str = "responseSchema") -> dict:
    generation_config: Final = {
        **{key: value for key, value in _GENERATION_CONFIG.items() if key != "responseSchema"},
        schema_key: _RESPONSE_SCHEMA,
    }
    return {**_REQUEST, "contents": _contents(marker), "generationConfig": generation_config}


def _tool_loop_contents(marker: str, image_part: dict) -> list[dict]:
    return [
        {"role": "user", "parts": [{"text": f"what is in this photo {marker}"}, image_part]},
        {
            "role": "model",
            "parts": [{"text": "checking"}, {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}],
        },
        {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"forecast": "sunny"}}}]},
    ]


def _expected_anthropic_tool_loop_messages(marker: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"what is in this photo {marker}"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMAGE_DATA}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "call_get_weather", "name": "get_weather", "input": {"city": "SF"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_get_weather", "content": '{"forecast": "sunny"}'}],
        },
    ]


def _expected_openai_tool_loop_messages(marker: str) -> list[dict]:
    return [
        {"role": "system", "content": "use the tools"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"what is in this photo {marker}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_IMAGE_DATA}"}},
            ],
        },
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call_get_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_get_weather", "content": '{"forecast": "sunny"}'},
    ]


def _expected_openai_messages(marker: str) -> list[dict]:
    return [
        {"role": "system", "content": "use the tools"},
        {"role": "user", "content": f"what is the weather in SF {marker}"},
        {"role": "assistant", "content": "let me check"},
        {"role": "user", "content": "and the time"},
    ]


def _expected_anthropic_messages(marker: str) -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": f"what is the weather in SF {marker}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "let me check"}]},
        {"role": "user", "content": [{"type": "text", "text": "and the time"}]},
    ]


@pytest.mark.parametrize("schema_key", _SCHEMA_KEYS)
@pytest.mark.parametrize("prefix", _PREFIXES)
def test_generate_content_adapts_openai_backend_request_and_response(
    gateway: Gateway, prefix: str, schema_key: str
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "messages": _expected_openai_messages(marker),
            "max_tokens": 50,
            "temperature": 0.1,
            "stop": ["END"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": _RESPONSE_SCHEMA},
            },
            "tools": _EXPECTED_TOOLS,
            "tool_choice": "required",
        }
        return Reply(body=json.dumps(_OPENAI_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", f"{prefix}/{model}:generateContent", _request(marker, schema_key), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")]


@pytest.mark.parametrize("schema_key", _SCHEMA_KEYS)
@pytest.mark.parametrize("prefix", _PREFIXES)
def test_generate_content_adapts_anthropic_backend_request_and_response(
    gateway: Gateway, prefix: str, schema_key: str
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        assert json.loads(request.body) == {
            "model": "claude-haiku-4-5",
            "system": [{"type": "text", "text": "use the tools"}],
            "messages": _expected_anthropic_messages(marker),
            "max_tokens": 50,
            "temperature": 0.1,
            "stop_sequences": ["END"],
            "output_format": {
                "type": "json_schema",
                "schema": {**_RESPONSE_SCHEMA, "additionalProperties": False},
            },
            "tools": _EXPECTED_ANTHROPIC_TOOLS,
            "tool_choice": {"type": "any"},
        }
        return Reply(body=json.dumps(_ANTHROPIC_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="anthropic/claude-haiku-4-5", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", f"{prefix}/{model}:generateContent", _request(marker, schema_key), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/messages")]


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_stream_generate_content_adapts_openai_sse_to_gemini_frames(gateway: Gateway, prefix: str) -> None:
    frames: Final = _openai_stream_frames()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "messages": _expected_openai_messages(marker),
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": _EXPECTED_TOOLS,
            "tool_choice": "required",
        }
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            f"{prefix}/{model}:streamGenerateContent?alt=sse",
            json={
                "contents": _contents(marker),
                "systemInstruction": _SYSTEM_INSTRUCTION,
                "tools": _TOOLS,
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        text: Final = response.text
        assert "data: [DONE]" not in text, text
        frames_out: Final = [part for part in text.split("\n\n") if part]
        assert all(part.startswith("data: ") for part in frames_out), text
        parsed: Final = [json.loads(part.removeprefix("data: ")) for part in frames_out]
        assert "usageMetadata" in parsed[-1], text
        assert set(parsed[-1]["usageMetadata"]) == {
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
        }, text
        assert [
            {key: value for key, value in frame.items() if key != "usageMetadata"} for frame in parsed
        ] == _expected_gemini_stream_frames(None), text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")]


def test_generate_content_adapter_accepts_snake_case_generation_config(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: top-level snake_case generation_config is dropped on the completion-adapter path, no max_tokens, temperature, stop or response_format"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "messages": _expected_openai_messages(marker),
            "max_tokens": 50,
            "temperature": 0.1,
            "stop": ["END"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": _RESPONSE_SCHEMA},
            },
            "tools": _EXPECTED_TOOLS,
            "tool_choice": "required",
        }, request.body.decode()
        return Reply(body=json.dumps(_OPENAI_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[model])
        request_body: Final = {
            "contents": _contents(marker),
            "system_instruction": _SYSTEM_INSTRUCTION,
            "generation_config": _GENERATION_CONFIG,
            "tools": _TOOLS,
            "tool_config": {"functionCallingConfig": {"mode": "ANY"}},
            "cache": {"no-cache": True},
        }
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:generateContent", request_body, key=key)
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")], response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_stream_generate_content_adapter_reports_upstream_usage(gateway: Gateway, prefix: str, backend: str) -> None:
    frames: Final = _openai_stream_frames() if backend == "openai" else _anthropic_stream_frames()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == ("/v1/chat/completions" if backend == "openai" else "/v1/messages")
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = (
            scenario.model(model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key")
            if backend == "openai"
            else scenario.model(
                model="anthropic/claude-haiku-4-5", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            f"{prefix}/{model}:streamGenerateContent?alt=sse",
            json={
                "contents": _contents(marker),
                "systemInstruction": _SYSTEM_INSTRUCTION,
                **({"generationConfig": {"maxOutputTokens": 50}} if backend == "anthropic" else {}),
                "tools": _TOOLS,
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert "data: [DONE]" not in response.text, response.text
        frames_out: Final = [part for part in response.text.split("\n\n") if part]
        parsed: Final = [json.loads(part.removeprefix("data: ")) for part in frames_out]
        assert parsed == _expected_gemini_stream_frames(
            {"promptTokenCount": 20, "candidatesTokenCount": 9, "totalTokenCount": 29}
        ), response.text
        assert [(r.method, r.target) for r in wire.drain()] == [
            ("POST", "/v1/chat/completions" if backend == "openai" else "/v1/messages")
        ], response.text


def _anthropic_stream_frames() -> tuple[bytes, ...]:
    def frame(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    return (
        frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_wire2",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "checking "}},
        ),
        frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            },
        ),
        frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ci'}},
        ),
        frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'ty":"SF"}'},
            },
        ),
        frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        frame(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        ),
        frame("message_stop", {"type": "message_stop"}),
    )


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_stream_generate_content_adapts_anthropic_sse_to_gemini_frames(gateway: Gateway, prefix: str) -> None:
    frames: Final = _anthropic_stream_frames()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        assert json.loads(request.body) == {
            "model": "claude-haiku-4-5",
            "system": [{"type": "text", "text": "use the tools"}],
            "messages": _expected_anthropic_messages(marker),
            "max_tokens": 50,
            "tools": _EXPECTED_ANTHROPIC_TOOLS,
            "tool_choice": {"type": "any"},
            "stream": True,
        }, request.body.decode()
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="anthropic/claude-haiku-4-5", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            f"{prefix}/{model}:streamGenerateContent?alt=sse",
            json={
                "contents": _contents(marker),
                "systemInstruction": _SYSTEM_INSTRUCTION,
                "generationConfig": {"maxOutputTokens": 50},
                "tools": _TOOLS,
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        text: Final = response.text
        assert "data: [DONE]" not in text, text
        frames_out: Final = [part for part in text.split("\n\n") if part]
        assert all(part.startswith("data: ") for part in frames_out), text
        parsed: Final = [json.loads(part.removeprefix("data: ")) for part in frames_out]
        assert set(parsed[-1]["usageMetadata"]) == {
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
        }, text
        assert [
            {key: value for key, value in frame.items() if key != "usageMetadata"} for frame in parsed
        ] == _expected_gemini_stream_frames(None), text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/messages")]


def _tool_loop_request(marker: str, image_part: dict) -> dict:
    return {
        "contents": _tool_loop_contents(marker, image_part),
        "systemInstruction": _SYSTEM_INSTRUCTION,
        "tools": _TOOLS,
    }


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_generate_content_adapts_openai_tool_loop_history_and_inline_image(gateway: Gateway, prefix: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "messages": _expected_openai_tool_loop_messages(marker),
            "tools": _EXPECTED_TOOLS,
        }, request.body.decode()
        return Reply(body=json.dumps(_OPENAI_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[model])
        image_part: Final = {"inline_data": {"mime_type": "image/png", "data": _IMAGE_DATA}}
        response: Final = gateway.request(
            "POST", f"{prefix}/{model}:generateContent", _tool_loop_request(marker, image_part), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")]


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_generate_content_adapts_anthropic_tool_loop_history_and_inline_image(gateway: Gateway, prefix: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages"
        assert json.loads(request.body) == {
            "model": "claude-haiku-4-5",
            "system": [{"type": "text", "text": "use the tools"}],
            "messages": _expected_anthropic_tool_loop_messages(marker),
            "max_tokens": 64000,
            "tools": _EXPECTED_ANTHROPIC_TOOLS,
        }, request.body.decode()
        return Reply(body=json.dumps(_ANTHROPIC_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="anthropic/claude-haiku-4-5", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        key: Final = scenario.key(models=[model])
        image_part: Final = {"inline_data": {"mime_type": "image/png", "data": _IMAGE_DATA}}
        response: Final = gateway.request(
            "POST", f"{prefix}/{model}:generateContent", _tool_loop_request(marker, image_part), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/messages")]


def test_generate_content_adapter_keeps_camel_case_inline_data(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: a camelCase inlineData part (the google-genai SDK wire spelling) is silently dropped on the completion-adapter path, the image never reaches the provider"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "messages": _expected_openai_tool_loop_messages(marker),
            "tools": _EXPECTED_TOOLS,
        }, request.body.decode()
        return Reply(body=json.dumps(_OPENAI_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        marker: Final = uuid.uuid4().hex
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[model])
        image_part: Final = {"inlineData": {"mimeType": "image/png", "data": _IMAGE_DATA}}
        response: Final = gateway.request(
            "POST", f"/v1beta/models/{model}:generateContent", _tool_loop_request(marker, image_part), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")], response.text


_TEXT_BASE: Final = {
    "id": "chatcmpl-wire3",
    "object": "chat.completion.chunk",
    "created": 1700000000,
    "model": "gpt-4o-mini",
}
_TEXT_USAGE: Final = {"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17}
_ZERO_USAGE: Final = {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}


def _text_frame(chunk: Mapping[str, JsonValue], response_id: str = "chatcmpl-wire3") -> bytes:
    return f"data: {json.dumps({**_TEXT_BASE, 'id': response_id, **chunk})}\n\n".encode()


_DELTA_CHUNK: Final = {
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]
}
_FINISH_CHUNK: Final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
_USAGE_CHUNK: Final = {"choices": [], "usage": _TEXT_USAGE}
_TEXT_DELTA: Final = _text_frame(_DELTA_CHUNK)
_TEXT_FINISH: Final = _text_frame(_FINISH_CHUNK)
_TEXT_USAGE_FRAME: Final = _text_frame(_USAGE_CHUNK)
_TEXT_DONE: Final = b"data: [DONE]\n\n"


def _billed_text_frames(include_usage: bool) -> tuple[bytes, ...]:
    response_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    chunks: Final = (_DELTA_CHUNK, _FINISH_CHUNK, *((_USAGE_CHUNK,) if include_usage else ()))
    return (*(_text_frame(chunk, response_id) for chunk in chunks), _TEXT_DONE)


def _text_stream_request(marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": f"say hello {marker}"}]}], **fields}


def _gemini_frames(response_text: str) -> list[dict]:
    parts: Final = [part for part in response_text.split("\n\n") if part.strip()]
    assert all(part.startswith("data: ") for part in parts), response_text
    return [json.loads(part.removeprefix("data: ")) for part in parts]


def _stop_frames(frames: list[dict]) -> list[dict]:
    return [frame for frame in frames if any(c.get("finishReason") for c in frame.get("candidates", []))]


def _openai_text_model(scenario, wire) -> str:
    return scenario.model(model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key")


def _stream_text(gateway: Gateway, model: str, key: str, body: Mapping[str, JsonValue]):
    return gateway.client.post(
        f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
        json=dict(body),
        headers={"Authorization": f"Bearer {key}"},
    )


@pytest.mark.parametrize(
    ("caller_options", "forwarded"),
    [
        pytest.param(None, {"include_usage": True}, id="caller_sends_none"),
        pytest.param({"include_usage": True}, {"include_usage": True}, id="caller_opts_in"),
        pytest.param({"include_usage": False}, None, id="caller_opts_out"),
        pytest.param({}, None, id="caller_sends_empty_object"),
    ],
)
def test_stream_generate_content_adapter_forwards_the_callers_stream_options(
    gateway: Gateway, caller_options: dict | None, forwarded: dict | None
) -> None:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        asks: Final = body.get("stream_options", {}).get("include_usage") is True
        frames: Final = (_TEXT_DELTA, _TEXT_FINISH, *((_TEXT_USAGE_FRAME,) if asks else ()), _TEXT_DONE)
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_text_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        options: Final = {} if caller_options is None else {"stream_options": caller_options}
        response: Final = _stream_text(gateway, model, key, _text_stream_request(uuid.uuid4().hex, **options))
        assert response.status_code == 200, response.text
        frames: Final = _gemini_frames(response.text)
        stops: Final = _stop_frames(frames)
        assert len(stops) == 1 and stops[0] is frames[-1], response.text
        expected_usage: Final = (
            {"promptTokenCount": 13, "candidatesTokenCount": 4, "totalTokenCount": 17} if forwarded else _ZERO_USAGE
        )
        assert frames[-1]["usageMetadata"] == expected_usage, response.text
        received: Final = wire.drain()
        assert len(received) == 1
        assert json.loads(received[0].body).get("stream_options") == forwarded, received[0].body


_BILLED_SELECT: Final = 'SELECT prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s'


def test_stream_generate_content_adapter_without_upstream_usage_reports_the_billed_tokens(gateway: Gateway) -> None:
    def respond(_request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=_billed_text_frames(include_usage=False))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_text_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_text(gateway, model, key, _text_stream_request(uuid.uuid4().hex))
        assert response.status_code == 200, response.text
        frames: Final = _gemini_frames(response.text)
        assert frames[0]["candidates"][0]["content"]["parts"] == [{"text": "hello"}], response.text
        stops: Final = _stop_frames(frames)
        assert len(stops) == 1 and stops[0] is frames[-1], response.text
        digest: Final = sha256(key.encode()).hexdigest()
        eventually(lambda: read_rows(_BILLED_SELECT, (digest,)), lambda rows: len(rows) >= 1, seconds=70)
        rows: Final = held(lambda: read_rows(_BILLED_SELECT, (digest,)), lambda found: len(found) == 1, holding=3)
        usage: Final = frames[-1]["usageMetadata"]
        assert (usage["promptTokenCount"], usage["candidatesTokenCount"]) == (
            rows[0]["prompt_tokens"],
            rows[0]["completion_tokens"],
        ), (response.text, rows)
        assert usage["promptTokenCount"] > 0, response.text
        assert len(wire.drain()) == 1


def test_stream_generate_content_adapter_usage_before_finish_keeps_the_real_usage_last(gateway: Gateway) -> None:
    def respond(_request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream", chunks=(_TEXT_DELTA, _TEXT_USAGE_FRAME, _TEXT_FINISH, _TEXT_DONE)
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_text_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_text(gateway, model, key, _text_stream_request(uuid.uuid4().hex))
        assert response.status_code == 200, response.text
        frames: Final = _gemini_frames(response.text)
        assert len(_stop_frames(frames)) == 1, response.text
        assert [frame["usageMetadata"] for frame in frames if "usageMetadata" in frame][-1] == {
            "promptTokenCount": 13,
            "candidatesTokenCount": 4,
            "totalTokenCount": 17,
        }, response.text
        assert len(wire.drain()) == 1


def test_stream_generate_content_adapter_slow_usage_after_finish_still_arrives(gateway: Gateway) -> None:
    def respond(_request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=(_TEXT_DELTA, _TEXT_FINISH, _TEXT_USAGE_FRAME, _TEXT_DONE),
            pause_between_chunks=1.5,
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_text_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_text(gateway, model, key, _text_stream_request(uuid.uuid4().hex))
        assert response.status_code == 200, response.text
        frames: Final = _gemini_frames(response.text)
        stops: Final = _stop_frames(frames)
        assert len(stops) == 1 and stops[0] is frames[-1], response.text
        assert frames[-1]["usageMetadata"] == {
            "promptTokenCount": 13,
            "candidatesTokenCount": 4,
            "totalTokenCount": 17,
        }, response.text
        assert len(wire.drain()) == 1


def test_stream_generate_content_adapter_upstream_drop_after_finish_still_sends_stop(gateway: Gateway) -> None:
    def respond(_request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=(_TEXT_DELTA, _TEXT_FINISH, _TEXT_USAGE_FRAME, _TEXT_DONE),
            abort_after=2,
            pause_between_chunks=1.5,
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_text_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = _stream_text(gateway, model, key, _text_stream_request(uuid.uuid4().hex))
        assert response.status_code == 200, response.text
        frames: Final = _gemini_frames(response.text)
        assert frames[0]["candidates"][0]["content"]["parts"] == [{"text": "hello"}], response.text
        assert len(_stop_frames(frames)) == 1, response.text
        assert len(wire.drain()) == 1
        healthy: Final = gateway.request("GET", "/health/liveliness")
        assert healthy.status_code == 200, healthy.text


_ATTRIBUTION_SELECT: Final = (
    "SELECT request_tags, team_id, metadata->>'user_api_key_alias' AS alias, model_group, prompt_tokens,"
    ' completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s'
)
_STATUS_SELECT: Final = 'SELECT status FROM "LiteLLM_SpendLogs" WHERE api_key=%s'


@pytest.mark.parametrize("streamed", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("status", [400, 401, 429])
def test_generate_content_adapter_upstream_error_keeps_the_provider_status(
    gateway: Gateway, status: int, streamed: bool
) -> None:
    message: Final = f"scripted openai rejection {uuid.uuid4().hex}"

    def respond(_request: Request) -> Reply:
        error: Final = {"message": message, "type": "invalid_request_error", "code": str(status)}
        return Reply(status=status, body=json.dumps({"error": error}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            num_retries=0,
            max_retries=0,
        )
        key: Final = scenario.key(models=[model])
        method: Final = "streamGenerateContent?alt=sse" if streamed else "generateContent"
        response: Final = gateway.client.post(
            f"/v1beta/models/{model}:{method}",
            json=_text_stream_request(uuid.uuid4().hex),
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == status, response.text
        assert message in response.text, response.text
        digest: Final = sha256(key.encode()).hexdigest()
        eventually(lambda: read_rows(_STATUS_SELECT, (digest,)), lambda rows: len(rows) >= 1, seconds=70)
        rows: Final = held(lambda: read_rows(_STATUS_SELECT, (digest,)), lambda found: len(found) == 1, holding=3)
        assert rows[0]["status"] == "failure", rows
        assert len(wire.drain()) == 1
        healthy: Final = gateway.request("GET", "/health/liveliness")
        assert healthy.status_code == 200, healthy.text


@pytest.mark.parametrize("streamed", [False, True], ids=["generate", "stream"])
def test_generate_content_adapter_spend_row_keeps_key_team_and_tag_attribution(
    gateway: Gateway, streamed: bool
) -> None:
    def respond(request: Request) -> Reply:
        if json.loads(request.body).get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_billed_text_frames(include_usage=True))
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                    ],
                    "usage": _TEXT_USAGE,
                }
            ).encode()
        )

    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_text_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        alias: Final = f"adapter-alias-{marker}"
        key: Final = scenario.key(models=[model], team_id=team, key_alias=alias)
        method: Final = "streamGenerateContent?alt=sse" if streamed else "generateContent"
        response: Final = gateway.client.post(
            f"/v1beta/models/{model}:{method}",
            json=_text_stream_request(marker),
            headers={"Authorization": f"Bearer {key}", "x-litellm-tags": f"adapter-tag-{marker}"},
        )
        assert response.status_code == 200, response.text
        digest: Final = sha256(key.encode()).hexdigest()
        eventually(lambda: read_rows(_ATTRIBUTION_SELECT, (digest,)), lambda rows: len(rows) >= 1, seconds=70)
        rows: Final = held(lambda: read_rows(_ATTRIBUTION_SELECT, (digest,)), lambda found: len(found) == 1, holding=3)
        row: Final = rows[0]
        assert (row["team_id"], row["alias"], row["model_group"]) == (team, alias, model), row
        assert f"adapter-tag-{marker}" in json.dumps(row["request_tags"]), row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (13, 4), row
        assert len(wire.drain()) == 1
