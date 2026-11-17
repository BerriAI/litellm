import itertools
import json
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

_BACKEND: Final = "gemini-3.7-flash"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_SIGNATURE: Final = "sig-4f2a"
_ARGS: Final = {"city": "Paris"}
_FUNCTIONS: Final = [
    {
        "name": "get_weather",
        "description": "Return the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _candidate(*, with_signature: bool) -> dict[str, JsonValue]:
    part: Final = {
        "functionCall": {"name": "get_weather", "args": _ARGS, "id": "fc-1"},
        **({"thoughtSignature": _SIGNATURE} if with_signature else {}),
    }
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [part]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
        "modelVersion": _BACKEND,
    }


def _model(gateway: Gateway, scenario: Scenario, wire_url: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=f"{wire_url}{_MODEL_PATH}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
    )


def _non_streaming_call(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    response: Final = gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "functions": _FUNCTIONS,
            "messages": [{"role": "user", "content": "weather?"}],
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _streaming_call(gateway: Gateway, model: str) -> tuple[dict[str, JsonValue], ...]:
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "functions": _FUNCTIONS,
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=30,
    ) as response:
        assert response.status_code == 200, response.read()
        lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
    assert lines[-1] == "data: [DONE]", lines[-3:]
    return tuple(_JSON_OBJECT.validate_json(line.removeprefix("data: ").encode()) for line in lines[:-1])


def _function_call_of(response: dict[str, JsonValue]) -> dict[str, JsonValue]:
    message: Final = response["choices"][0]["message"]
    assert isinstance(message, dict)
    call: Final = message["function_call"]
    assert isinstance(call, dict)
    return call


def test_vertex_gemini_function_call_thought_signature_is_returned_non_streaming(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == f"{_MODEL_PATH}:generateContent"
        return Reply(body=json.dumps(_candidate(with_signature=True)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url)
        call: Final = _function_call_of(_non_streaming_call(gateway, model))
        assert call["name"] == "get_weather"
        assert json.loads(str(call["arguments"])) == _ARGS
        assert call.get("provider_specific_fields") == {"thought_signature": _SIGNATURE}


def test_vertex_gemini_function_call_without_signature_has_no_provider_fields_non_streaming(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == f"{_MODEL_PATH}:generateContent"
        return Reply(body=json.dumps(_candidate(with_signature=False)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url)
        call: Final = _function_call_of(_non_streaming_call(gateway, model))
        assert call["name"] == "get_weather"
        assert json.loads(str(call["arguments"])) == _ARGS
        assert "provider_specific_fields" not in call
        assert "thought_signature" not in json.dumps(call)


def test_vertex_gemini_function_call_thought_signature_is_returned_streaming(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == f"{_MODEL_PATH}:streamGenerateContent?alt=sse"
        payload: Final = json.dumps(_candidate(with_signature=True))
        return Reply(content_type="text/event-stream", chunks=[f"data: {payload}\n\n".encode()])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url)
        chunks: Final = _streaming_call(gateway, model)
        function_calls: Final = tuple(
            choice["delta"]["function_call"]
            for chunk in chunks
            for choice in chunk.get("choices", ())
            if choice.get("delta", {}).get("function_call")
        )
        assert function_calls, "no function_call delta received"
        merged: Final = "".join(str(call.get("arguments", "")) for call in function_calls)
        assert json.loads(merged) == _ARGS
        assert function_calls[-1].get("provider_specific_fields") == {"thought_signature": _SIGNATURE}


def test_vertex_gemini_function_call_without_signature_has_no_provider_fields_streaming(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == f"{_MODEL_PATH}:streamGenerateContent?alt=sse"
        payload: Final = json.dumps(_candidate(with_signature=False))
        return Reply(content_type="text/event-stream", chunks=[f"data: {payload}\n\n".encode()])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url)
        chunks: Final = _streaming_call(gateway, model)
        function_calls: Final = tuple(
            choice["delta"]["function_call"]
            for chunk in chunks
            for choice in chunk.get("choices", ())
            if choice.get("delta", {}).get("function_call")
        )
        assert function_calls, "no function_call delta received"
        assert all("provider_specific_fields" not in call for call in function_calls)
        assert "thought_signature" not in json.dumps(function_calls)


def test_vertex_gemini_kwargs_extra_param_reaches_generation_config(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == f"{_MODEL_PATH}:generateContent"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["generationConfig"]["candidate_count"] == 3, body
        return Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "done"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
                    "modelVersion": _BACKEND,
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _model(gateway, scenario, wire.url)
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "candidate_count": 3,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "done"


_REPLAYED_SIGNATURE: Final = "cmVwbGF5ZWQtc2lnbmF0dXJl"
_SIGNED_CALL_ID: Final = f"call_7c1e__thought__{_REPLAYED_SIGNATURE}"
_TOOL_TURN_MESSAGES: Final = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": _SIGNED_CALL_ID,
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps(_ARGS)},
            }
        ],
    },
    {"role": "tool", "tool_call_id": _SIGNED_CALL_ID, "content": "sunny"},
]


def _tool_turn_call(gateway: Gateway, model: str, **extra: JsonValue) -> None:
    response: Final = gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "tools": [{"type": "function", "function": function} for function in _FUNCTIONS],
            "messages": _TOOL_TURN_MESSAGES,
            **extra,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=30,
    )
    assert response.status_code == 200, response.text


class _GeminiPart(TypedDict):
    function_call: NotRequired[ReadOnly[JsonValue]]
    thoughtSignature: NotRequired[ReadOnly[str]]


class _GeminiContent(TypedDict):
    parts: ReadOnly[tuple[_GeminiPart, ...]]


class _GeminiRequest(TypedDict):
    contents: ReadOnly[tuple[_GeminiContent, ...]]


class _ToolCall(TypedDict):
    id: ReadOnly[str]


class _ChatMessage(TypedDict):
    tool_calls: NotRequired[ReadOnly[tuple[_ToolCall, ...] | None]]
    tool_call_id: NotRequired[ReadOnly[str]]


class _ChatRequest(TypedDict):
    messages: ReadOnly[tuple[_ChatMessage, ...]]


_GEMINI_REQUEST: Final = TypeAdapter(_GeminiRequest)
_CHAT_REQUEST: Final = TypeAdapter(_ChatRequest)


def _function_call_signatures(request: Request) -> tuple[str | None, ...]:
    contents: Final = _GEMINI_REQUEST.validate_json(request.body)["contents"]
    parts: Final = itertools.chain.from_iterable(content["parts"] for content in contents)
    return tuple(part.get("thoughtSignature") for part in parts if "function_call" in part)


def _sent_tool_call_ids(request: Request) -> tuple[str, ...]:
    messages: Final = _CHAT_REQUEST.validate_json(request.body)["messages"]
    calls: Final = itertools.chain.from_iterable(message.get("tool_calls") or () for message in messages)
    return (
        *(call["id"] for call in calls),
        *(message["tool_call_id"] for message in messages if "tool_call_id" in message),
    )


def test_vertex_gemini_alias_without_gemini_in_name_replays_tool_call_thought_signature(gateway: Gateway) -> None:
    with (
        wire_server(lambda _: Reply(body=json.dumps(_candidate(with_signature=False)).encode())) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _model(gateway, scenario, wire.url)
        assert "gemini" not in model
        _tool_turn_call(gateway, model)
        (request,) = wire.drain()

    assert request.target == f"{_MODEL_PATH}:generateContent"
    assert _function_call_signatures(request) == (_REPLAYED_SIGNATURE,)


def test_openai_target_receives_tool_call_ids_without_thought_signature(gateway: Gateway) -> None:
    completion: Final = {
        "id": "chatcmpl-scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }
    with wire_server(lambda _: Reply(body=json.dumps(completion).encode())) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _tool_turn_call(gateway, model)
        (request,) = wire.drain()

    assert request.target == "/v1/chat/completions"
    assert _sent_tool_call_ids(request) == ("call_7c1e", "call_7c1e")


def test_vertex_gemini_alias_without_gemini_in_name_replays_tool_call_thought_signature_streaming(
    gateway: Gateway,
) -> None:
    payload: Final = json.dumps(_candidate(with_signature=False))
    with (
        wire_server(
            lambda _: Reply(content_type="text/event-stream", chunks=(f"data: {payload}\n\n".encode(),))
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _model(gateway, scenario, wire.url)
        _tool_turn_call(gateway, model, stream=True)
        (request,) = wire.drain()

    assert request.target == f"{_MODEL_PATH}:streamGenerateContent?alt=sse"
    assert _function_call_signatures(request) == (_REPLAYED_SIGNATURE,)


def test_fallback_from_openai_to_vertex_gemini_alias_strips_then_replays_thought_signature(gateway: Gateway) -> None:
    with (
        wire_server(lambda _: Reply(status=500, body=b'{"error": {"message": "scripted outage"}}')) as openai_wire,
        wire_server(lambda _: Reply(body=json.dumps(_candidate(with_signature=False)).encode())) as vertex_wire,
        gateway.scenario() as scenario,
    ):
        primary: Final = scenario.model(api_base=f"{openai_wire.url}/v1")
        fallback: Final = _model(gateway, scenario, vertex_wire.url)
        _tool_turn_call(gateway, primary, fallbacks=[fallback], num_retries=0)
        openai_requests: Final = openai_wire.drain()
        (vertex_request,) = vertex_wire.drain()

    assert openai_requests, "the primary deployment was never tried"
    assert {_sent_tool_call_ids(request) for request in openai_requests} == {("call_7c1e", "call_7c1e")}
    assert _function_call_signatures(vertex_request) == (_REPLAYED_SIGNATURE,)
