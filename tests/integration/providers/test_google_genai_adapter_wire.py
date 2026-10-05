import json
import uuid
from typing import Final

from integration._support.client import Gateway
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
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'ty":"SF"}'}}]
                        },
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


def _contents(marker: str) -> list[dict]:
    return [
        {"role": "user", "parts": [{"text": f"what is the weather in SF {marker}"}]},
        {"role": "model", "parts": [{"text": "let me check"}]},
        {"role": "user", "parts": [{"text": "and the time"}]},
    ]


def _request(marker: str) -> dict:
    return {**_REQUEST, "contents": _contents(marker)}


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


def test_generate_content_adapts_openai_backend_request_and_response(gateway: Gateway) -> None:
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
            "POST", f"/v1beta/models/{model}:generateContent", _request(marker), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")]


def test_generate_content_adapts_anthropic_backend_request_and_response(gateway: Gateway) -> None:
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
            "tools": [
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
            ],
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
            "POST", f"/v1beta/models/{model}:generateContent", _request(marker), key=key
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["candidates"] == _EXPECTED_CANDIDATES, response.text
        assert body["usageMetadata"] == _EXPECTED_USAGE, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/messages")]


def test_stream_generate_content_adapts_openai_sse_to_gemini_frames(gateway: Gateway) -> None:
    frames: Final = _openai_stream_frames()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "messages": _expected_openai_messages(marker),
            "stream": True,
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
            f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
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
        assert parsed == [
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
                "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
            },
        ], text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1/chat/completions")]
