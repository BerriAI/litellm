import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

_GEMINI_MODEL: Final = "gemini/gemini-2.5-flash"
_GEMINI_DEPLOYMENT_KEY: Final = "synthetic-gemini-deployment-key"

_CONTENTS: Final = (
    {"role": "user", "parts": [{"text": "name a color"}]},
    {"role": "model", "parts": [{"text": "blue"}]},
    {"role": "user", "parts": [{"text": "name another"}]},
)
_SYSTEM_INSTRUCTION: Final = {"parts": [{"text": "answer with one word"}]}
_GENERATION_CONFIG_REQUEST: Final = {
    "maxOutputTokens": 64,
    "temperature": 0.2,
    "top_p": 0.9,
    "response_mime_type": "application/json",
    "responseSchema": {
        "type": "object",
        "properties": {"color": {"type": "string"}},
        "required": ["color"],
    },
}
_TOOLS: Final = (
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "description": "read the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
    },
)
_TOOL_CONFIG: Final = {"functionCallingConfig": {"mode": "AUTO"}}
_SAFETY_SETTINGS: Final = ({"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},)

_REPLY_BODY: Final = {
    "candidates": [
        {
            "content": {"parts": [{"text": "green"}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
}

_STREAM_FRAMES: Final = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"gre"}],"role":"model"},"index":0}]}\n\n',
    b'id: gemini-frame-2\ndata: {"candidates":[{"content":{"parts":[{"text":"en"}],"role":"model"},"index":0}]}\n\n',
    b"retry: 3000\n\n",
    b'data: {"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],'
    b'"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7,"totalTokenCount":18}}\n\n',
)


class _Candidate(BaseModel):
    content: dict
    finishReason: str
    index: int


class _GenerateContentResponse(BaseModel):
    candidates: tuple[_Candidate, ...]
    usageMetadata: dict


def _assert_no_virtual_key(request: Request, key: str) -> None:
    assert all(key not in value for value in request.headers.values()), dict(request.headers)


@pytest.mark.parametrize("prefix", ["/v1beta/models", "/models"])
def test_generate_content_rewrites_native_request_for_gemini_backend(gateway: Gateway, prefix: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/models/gemini-2.5-flash:generateContent"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        _assert_no_virtual_key(request, key)
        assert json.loads(request.body) == {
            "model": "gemini-2.5-flash",
            "contents": list(_CONTENTS),
            "systemInstruction": _SYSTEM_INSTRUCTION,
            "tools": list(_TOOLS),
            "toolConfig": _TOOL_CONFIG,
            "safetySettings": list(_SAFETY_SETTINGS),
            "generationConfig": {
                "maxOutputTokens": 64,
                "temperature": 0.2,
                "topP": 0.9,
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                    "required": ["color"],
                },
            },
        }
        return Reply(body=json.dumps(_REPLY_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            f"{prefix}/{model}:generateContent",
            {
                "contents": list(_CONTENTS),
                "systemInstruction": _SYSTEM_INSTRUCTION,
                "generationConfig": _GENERATION_CONFIG_REQUEST,
                "tools": list(_TOOLS),
                "toolConfig": _TOOL_CONFIG,
                "safetySettings": list(_SAFETY_SETTINGS),
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _GenerateContentResponse.model_validate_json(response.content)
        assert [candidate.model_dump() for candidate in payload.candidates] == _REPLY_BODY["candidates"], response.text
        assert payload.usageMetadata == _REPLY_BODY["usageMetadata"], response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/models/gemini-2.5-flash:generateContent")]


@pytest.mark.parametrize(
    "prefix,query,chunking",
    [
        ("/v1beta/models", "?alt=sse", "whole"),
        ("/v1beta/models", "", "whole"),
        ("/models", "?alt=sse", "whole"),
        ("/models", "", "whole"),
        ("/v1beta/models", "?alt=sse", "split"),
        ("/v1beta/models", "", "split"),
        ("/models", "?alt=sse", "split"),
        ("/models", "", "split"),
        ("/v1beta/models", "?alt=sse", "packed"),
        ("/v1beta/models", "", "packed"),
        ("/models", "?alt=sse", "packed"),
        ("/models", "", "packed"),
    ],
)
def test_stream_generate_content_relays_gemini_sse_frames(
    gateway: Gateway, prefix: str, query: str, chunking: str
) -> None:
    stream_bytes: Final = b"".join(_STREAM_FRAMES)
    json_boundary: Final = stream_bytes.index(b'"gre"') + 2
    delimiter_boundary: Final = stream_bytes.index(b"\n\n") + 1
    id_boundary: Final = stream_bytes.index(b"id: gemini-frame-2") + len(b"id:")
    chunks: Final = {
        "whole": _STREAM_FRAMES,
        "split": (
            stream_bytes[:json_boundary],
            stream_bytes[json_boundary:delimiter_boundary],
            stream_bytes[delimiter_boundary:id_boundary],
            stream_bytes[id_boundary:],
        ),
        "packed": (stream_bytes,),
    }[chunking]

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        _assert_no_virtual_key(request, key)
        assert json.loads(request.body) == {
            "model": "gemini-2.5-flash",
            "contents": list(_CONTENTS),
            "tools": None,
            "generationConfig": {"maxOutputTokens": 64},
        }
        return Reply(content_type="text/event-stream", chunks=chunks)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            f"{prefix}/{model}:streamGenerateContent{query}",
            {"contents": list(_CONTENTS), "generationConfig": {"maxOutputTokens": 64}},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        assert response.content == stream_bytes, response.text
        assert b"[DONE]" not in response.content, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [
            ("POST", "/models/gemini-2.5-flash:streamGenerateContent?alt=sse")
        ]


@pytest.mark.parametrize("prefix", ["/v1beta/models", "/models"])
def test_generate_content_accepts_snake_case_top_level_fields(gateway: Gateway, prefix: str) -> None:
    pytest.skip(
        "BUG: top-level snake_case generation_config, tool_config and safety_settings are dropped before reaching Gemini"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/models/gemini-2.5-flash:generateContent"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        _assert_no_virtual_key(request, key)
        assert json.loads(request.body) == {
            "model": "gemini-2.5-flash",
            "contents": list(_CONTENTS),
            "systemInstruction": _SYSTEM_INSTRUCTION,
            "tools": list(_TOOLS),
            "toolConfig": _TOOL_CONFIG,
            "safetySettings": list(_SAFETY_SETTINGS),
            "generationConfig": {
                "maxOutputTokens": 64,
                "temperature": 0.2,
                "topP": 0.9,
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                    "required": ["color"],
                },
            },
        }, request.body.decode()
        return Reply(body=json.dumps(_REPLY_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            f"{prefix}/{model}:generateContent",
            {
                "contents": list(_CONTENTS),
                "system_instruction": _SYSTEM_INSTRUCTION,
                "generation_config": _GENERATION_CONFIG_REQUEST,
                "tools": list(_TOOLS),
                "tool_config": _TOOL_CONFIG,
                "safety_settings": list(_SAFETY_SETTINGS),
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _GenerateContentResponse.model_validate_json(response.content)
        assert [candidate.model_dump() for candidate in payload.candidates] == _REPLY_BODY["candidates"], response.text
        assert payload.usageMetadata == _REPLY_BODY["usageMetadata"], response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/models/gemini-2.5-flash:generateContent")], (
            response.text
        )


@pytest.mark.parametrize("prefix", ["/v1beta/models", "/models"])
@pytest.mark.parametrize("spelling", ["header", "query"])
@pytest.mark.parametrize(
    "route,path",
    [
        (":generateContent", "/models/gemini-2.5-flash:generateContent"),
        (":streamGenerateContent?alt=sse", "/models/gemini-2.5-flash:streamGenerateContent?alt=sse"),
    ],
    ids=["generateContent", "streamGenerateContent"],
)
def test_generate_content_authenticates_google_key_spellings(
    gateway: Gateway, prefix: str, spelling: str, route: str, path: str
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == path
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        _assert_no_virtual_key(request, key)
        if route == ":generateContent":
            return Reply(body=json.dumps(_REPLY_BODY).encode())
        return Reply(content_type="text/event-stream", chunks=_STREAM_FRAMES)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            f"{prefix}/{model}{route}",
            json={"contents": list(_CONTENTS)},
            headers={"x-goog-api-key": key} if spelling == "header" else {},
            params={"key": key} if spelling == "query" else {},
        )
        assert response.status_code == 200, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", path)], response.text


@pytest.mark.parametrize("prefix", ["/v1beta/models", "/models"])
@pytest.mark.parametrize(
    "route",
    [":generateContent", ":streamGenerateContent?alt=sse"],
    ids=["generateContent", "streamGenerateContent"],
)
@pytest.mark.parametrize("spelling", ["header", "query"])
def test_generate_content_rejects_key_without_model_access(
    gateway: Gateway, prefix: str, route: str, spelling: str
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_REPLY_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        other_model: Final = scenario.model(
            model="gemini/gemini-2.5-flash-lite", api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY
        )
        other_only_key: Final = scenario.key(models=[other_model])
        response: Final = gateway.client.post(
            f"{prefix}/{model}{route}",
            json={"contents": list(_CONTENTS)},
            headers={"x-goog-api-key": other_only_key} if spelling == "header" else {},
            params={"key": other_only_key} if spelling == "query" else {},
        )
        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == "key_model_access_denied", response.text
        assert wire.drain() == (), response.text
