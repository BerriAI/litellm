import base64
import binascii
import itertools
import json
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.prompt_templates.factory import get_dummy_thought_signature

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


Provider: TypeAlias = Literal["gemini", "vertex_ai"]

_GEMINI_3_BACKEND: Final = "gemini-3.8-flash"
_GEMINI_3_MODEL_PATH: Final = (
    f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_GEMINI_3_BACKEND}"
)
_GEMINI_API_KEY: Final = "synthetic-gemini-key"
_ISSUED_SIGNATURE: Final = "CiQBjz1rX+/9kL2mQ8vT3wYpHHn5aZcR4sE7uVbN0dFgK1oMx2s="
_STRIPPED_SIGNATURE: Final = re.sub(r"[+/=]", "", _ISSUED_SIGNATURE)
_ISSUED_SIGNATURE_DECODABLE_WITHOUT_PLUS_SLASH_PADDING: Final = "CiQBjz1rX+/9kL2mQ8vT3w+pHHn5aZcR4sE7uVbN0dFgK1oMx2s="
_UNDERSCORED_SIGNATURE: Final = re.sub(r"[+/=]", "_", _ISSUED_SIGNATURE_DECODABLE_WITHOUT_PLUS_SLASH_PADDING)
_PLACEHOLDER_SIGNATURE: Final = get_dummy_thought_signature()
_SIGNATURE_KEYS: Final = ("thoughtSignature", "thought_signature")
_FUNCTION_CALL_KEYS: Final = ("functionCall", "function_call")
_SIBLING_ID: Final = "call_rome"
_ANSWER: Final = "Sunny in Paris, rainy in Rome."
_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _register(gateway: Gateway, scenario: Scenario, provider: Provider, wire: Wire) -> str:
    name: Final = f"{provider}-{_GEMINI_3_BACKEND}-{uuid.uuid4().hex}"
    if provider == "gemini":
        return scenario.model(
            name=name, model=f"gemini/{_GEMINI_3_BACKEND}", api_base=wire.url, api_key=_GEMINI_API_KEY
        )
    return scenario.model(
        name=name,
        model=f"vertex_ai/{_GEMINI_3_BACKEND}",
        api_base=f"{wire.url}{_GEMINI_3_MODEL_PATH}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
    )


def _target(provider: Provider) -> str:
    prefix: Final = f"/models/{_GEMINI_3_BACKEND}" if provider == "gemini" else _GEMINI_3_MODEL_PATH
    return f"{prefix}:generateContent"


def _parts(content: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return _JSON_OBJECTS.validate_python(content.get("parts", []))


def _contents(body: bytes) -> list[dict[str, JsonValue]]:
    return _JSON_OBJECTS.validate_python(_JSON_OBJECT.validate_json(body)["contents"])


def _all_parts(body: bytes) -> tuple[dict[str, JsonValue], ...]:
    return tuple(itertools.chain.from_iterable(map(_parts, _contents(body))))


def _signature_of(part: dict[str, JsonValue]) -> JsonValue:
    return next((part[key] for key in _SIGNATURE_KEYS if key in part), None)


def _replayed_call_signatures(body: bytes) -> list[JsonValue]:
    calls: Final = (part for part in _all_parts(body) if any(key in part for key in _FUNCTION_CALL_KEYS))
    return [_signature_of(call) for call in calls]


def _decodes_as_bytes(signature: str) -> bool:
    if len(signature) % 4:
        return False
    try:
        base64.b64decode(signature.translate(str.maketrans("-_", "+/")), validate=True)
    except binascii.Error:
        return False
    return True


def _rejection(message: str) -> Reply:
    return Reply(
        status=400,
        body=json.dumps({"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}).encode(),
    )


def _signed_parts(body: bytes) -> Iterator[tuple[str, str]]:
    for content_index, content in enumerate(_contents(body)):
        for part_index, part in enumerate(_parts(content)):
            if (signature := _signature_of(part)) is not None:
                yield f"contents[{content_index}].parts[{part_index}].thought_signature", str(signature)


def _rejected_signature(body: bytes) -> Reply | None:
    signed: Final = tuple(_signed_parts(body))
    # Wording and order of checks observed live on Vertex on 2026-11-16 (BerriAI/litellm#52001)
    undecodable: Final = next((path for path, signature in signed if not _decodes_as_bytes(signature)), None)
    if undecodable is not None:
        return _rejection(f"Invalid value at '{undecodable}' (TYPE_BYTES), Base64 decoding failed")
    accepted: Final = (_ISSUED_SIGNATURE, _PLACEHOLDER_SIGNATURE)
    if any(signature not in accepted for _, signature in signed):
        return _rejection("Invalid thought signature.")
    return None


def _answer() -> bytes:
    return json.dumps(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": _ANSWER}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 8, "totalTokenCount": 38},
            "modelVersion": _GEMINI_3_BACKEND,
        }
    ).encode()


def _gemini(provider: Provider) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _target(provider)), request.target
        rejection: Final = _rejected_signature(request.body)
        if rejection is not None:
            return rejection
        return Reply(body=_answer())

    return respond


def _call(call_id: str, city: str) -> dict[str, JsonValue]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
    }


def _replay_parallel_pair(gateway: Gateway, provider: Provider, first_id: str) -> tuple[Request, ...]:
    with wire_server(_gemini(provider)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, provider, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "tools": [{"type": "function", "function": _FUNCTIONS[0]}],
                "messages": [
                    {"role": "user", "content": "Weather in Paris and Rome?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_call(first_id, "Paris"), _call(_SIBLING_ID, "Rome")],
                    },
                    {"role": "tool", "tool_call_id": first_id, "content": "sunny"},
                    {"role": "tool", "tool_call_id": _SIBLING_ID, "content": "rainy"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == _ANSWER
        return wire.drain()


_PROVIDERS: Final[tuple[Provider, ...]] = ("vertex_ai", "gemini")


@pytest.mark.parametrize(
    "mangled_signature", [_STRIPPED_SIGNATURE, _UNDERSCORED_SIGNATURE], ids=["stripped", "underscored"]
)
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_gemini_3_tool_call_id_with_mangled_signature_replays_the_placeholder_in_one_request(
    gateway: Gateway, provider: Provider, mangled_signature: str
) -> None:
    """regression (#52001): a harness that strips or rewrites +/= in the id must not turn later turns into a 400."""
    received: Final = _replay_parallel_pair(gateway, provider, f"call_paris__thought__{mangled_signature}")
    assert len(received) == 1, received
    assert _replayed_call_signatures(received[0].body) == [_PLACEHOLDER_SIGNATURE, None]


def test_gemini_3_responses_call_id_with_mangled_signature_replays_the_placeholder(gateway: Gateway) -> None:
    """regression (#52001): the same mangled signature in a Responses API call_id."""
    call_id: Final = f"call_paris__thought__{_STRIPPED_SIGNATURE}"
    with wire_server(_gemini("vertex_ai")) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "vertex_ai", wire)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "tools": [{"type": "function", **_FUNCTIONS[0]}],
                "input": [
                    {"role": "user", "content": "Weather in Paris?"},
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                    {"type": "function_call_output", "call_id": call_id, "output": "sunny"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _ANSWER in response.text, response.text
        received: Final = wire.drain()
    assert len(received) == 1, received
    assert _replayed_call_signatures(received[0].body) == [_PLACEHOLDER_SIGNATURE]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_gemini_3_tool_call_id_with_intact_signature_replays_it_byte_for_byte(
    gateway: Gateway, provider: Provider
) -> None:
    """guard: an untouched signature, + / and = included, still reaches Gemini exactly as it issued it."""
    received: Final = _replay_parallel_pair(gateway, provider, f"call_paris__thought__{_ISSUED_SIGNATURE}")
    assert len(received) == 1, received
    assert _replayed_call_signatures(received[0].body) == [_ISSUED_SIGNATURE, None]


_FIRST_ID: Final = "call_paris"
_STREAM_TARGET: Final = f"{_GEMINI_3_MODEL_PATH}:streamGenerateContent?alt=sse"
_FUNCTION_RESPONSE_KEYS: Final = ("functionResponse", "function_response")
_EXPECTED_CALLS: Final = (
    (_FIRST_ID, "get_weather", {"city": "Paris"}),
    (_SIBLING_ID, "get_weather", {"city": "Rome"}),
)
_EXPECTED_RESULTS: Final = (
    (_FIRST_ID, "get_weather", {"content": "sunny"}),
    (_SIBLING_ID, "get_weather", {"content": "rainy"}),
)
_CHAT_TOOLS: Final = [{"type": "function", "function": _FUNCTIONS[0]}]
_RESPONSES_TOOLS: Final = [{"type": "function", **_FUNCTIONS[0]}]


class _Delta(TypedDict):
    content: NotRequired[ReadOnly[str | None]]


class _StreamChoice(TypedDict):
    delta: ReadOnly[_Delta]


class _StreamChunk(TypedDict):
    choices: NotRequired[ReadOnly[tuple[_StreamChoice, ...]]]


_STREAM_CHUNK: Final = TypeAdapter(_StreamChunk)


def _vertex_gemini_3(request: Request) -> Reply:
    if request.target != _STREAM_TARGET:
        return _gemini("vertex_ai")(request)
    rejection: Final = _rejected_signature(request.body)
    if rejection is not None:
        return rejection
    return Reply(content_type="text/event-stream", chunks=(b"data: " + _answer() + b"\n\n",))


def _send(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[httpx.Response, tuple[Request, ...]]:
    with wire_server(_vertex_gemini_3) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "vertex_ai", wire)
        response: Final = gateway.request("POST", path, {"model": model, **body})
        return response, wire.drain()


def _has_any(part: dict[str, JsonValue], keys: tuple[str, ...]) -> bool:
    return any(key in part for key in keys)


def _row(part: dict[str, JsonValue], keys: tuple[str, ...], field: str) -> tuple[JsonValue, JsonValue, JsonValue]:
    payload: Final = _JSON_OBJECT.validate_python(next(part[key] for key in keys if key in part))
    return payload.get("id"), payload.get("name"), payload.get(field)


def _merged_tool_turn_calls(request: Request) -> tuple[dict[str, JsonValue], ...]:
    assert request.target in (_target("vertex_ai"), _STREAM_TARGET), request.target
    contents: Final = _contents(request.body)
    assert [content.get("role") for content in contents] == ["user", "model", "user"], contents
    call_parts: Final = tuple(part for part in _parts(contents[1]) if _has_any(part, _FUNCTION_CALL_KEYS))
    result_parts: Final = _parts(contents[2])
    assert all(_has_any(part, _FUNCTION_RESPONSE_KEYS) for part in result_parts), result_parts
    assert tuple(_row(part, _FUNCTION_CALL_KEYS, "args") for part in call_parts) == _EXPECTED_CALLS
    assert tuple(_row(part, _FUNCTION_RESPONSE_KEYS, "response") for part in result_parts) == _EXPECTED_RESULTS
    return call_parts


def _split_tool_turn(first_id: str, reasoning: str | None) -> list[JsonValue]:
    second_call: Final = {"role": "assistant", "content": None, "tool_calls": [_call(_SIBLING_ID, "Rome")]}
    return [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": None, "tool_calls": [_call(first_id, "Paris")]},
        second_call if reasoning is None else {**second_call, "reasoning_content": reasoning},
        {"role": "tool", "tool_call_id": first_id, "content": "sunny"},
        {"role": "tool", "tool_call_id": _SIBLING_ID, "content": "rainy"},
    ]


def _streamed_text(body: str) -> str:
    payloads: Final = tuple(line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: "))
    assert payloads and payloads[-1] == "[DONE]", payloads[-3:]
    choices: Final = itertools.chain.from_iterable(
        _STREAM_CHUNK.validate_json(payload).get("choices", ()) for payload in payloads[:-1]
    )
    return "".join(choice["delta"].get("content") or "" for choice in choices)


@pytest.mark.parametrize("reasoning", [None, "Now Rome."], ids=["plain", "with_reasoning"])
def test_gemini_3_consecutive_assistant_tool_calls_reach_vertex_as_one_model_turn(
    gateway: Gateway, reasoning: str | None
) -> None:
    response, received = _send(
        gateway,
        "/v1/chat/completions",
        {"tools": _CHAT_TOOLS, "messages": _split_tool_turn(_FIRST_ID, reasoning)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == _ANSWER
    assert len(received) == 1, received
    calls: Final = _merged_tool_turn_calls(received[0])
    assert _signature_of(calls[0]) == _PLACEHOLDER_SIGNATURE


def test_gemini_3_consecutive_assistant_tool_calls_reach_vertex_as_one_model_turn_streaming(gateway: Gateway) -> None:
    response, received = _send(
        gateway,
        "/v1/chat/completions",
        {"tools": _CHAT_TOOLS, "messages": _split_tool_turn(_FIRST_ID, "Now Rome."), "stream": True},
    )
    assert response.status_code == 200, response.text
    assert _streamed_text(response.text) == _ANSWER
    assert len(received) == 1, received
    assert received[0].target == _STREAM_TARGET
    calls: Final = _merged_tool_turn_calls(received[0])
    assert _signature_of(calls[0]) == _PLACEHOLDER_SIGNATURE


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_gemini_3_responses_reasoning_between_function_calls_reaches_vertex_as_one_model_turn(
    gateway: Gateway, stream: bool
) -> None:
    response, received = _send(
        gateway,
        "/v1/responses",
        {
            "tools": _RESPONSES_TOOLS,
            "stream": stream,
            "input": [
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {
                    "type": "function_call",
                    "call_id": _FIRST_ID,
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Paris"}),
                },
                {
                    "type": "reasoning",
                    "id": "rs_rome",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Now Rome."}],
                },
                {
                    "type": "function_call",
                    "call_id": _SIBLING_ID,
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Rome"}),
                },
                {"type": "function_call_output", "call_id": _FIRST_ID, "output": "sunny"},
                {"type": "function_call_output", "call_id": _SIBLING_ID, "output": "rainy"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert _ANSWER in response.text, response.text
    assert len(received) == 1, received
    assert received[0].target == (_STREAM_TARGET if stream else _target("vertex_ai"))
    _merged_tool_turn_calls(received[0])


def test_gemini_3_consecutive_assistant_tool_calls_keep_the_signature_on_its_own_call(gateway: Gateway) -> None:
    signed_id: Final = f"{_FIRST_ID}__thought__{_ISSUED_SIGNATURE}"
    response, received = _send(
        gateway,
        "/v1/chat/completions",
        {"tools": _CHAT_TOOLS, "messages": _split_tool_turn(signed_id, None)},
    )
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    first, second = _merged_tool_turn_calls(received[0])
    assert _signature_of(first) == _ISSUED_SIGNATURE
    assert _signature_of(second) != _ISSUED_SIGNATURE


def test_gemini_3_single_assistant_message_with_both_tool_calls_reaches_vertex_as_one_model_turn(
    gateway: Gateway,
) -> None:
    response, received = _send(
        gateway,
        "/v1/chat/completions",
        {
            "tools": _CHAT_TOOLS,
            "messages": [
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call(_FIRST_ID, "Paris"), _call(_SIBLING_ID, "Rome")],
                },
                {"role": "tool", "tool_call_id": _FIRST_ID, "content": "sunny"},
                {"role": "tool", "tool_call_id": _SIBLING_ID, "content": "rainy"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == _ANSWER
    assert len(received) == 1, received
    calls: Final = _merged_tool_turn_calls(received[0])
    assert _signature_of(calls[0]) == _PLACEHOLDER_SIGNATURE


def test_gemini_3_tool_result_for_a_call_no_assistant_made_is_rejected_before_vertex(gateway: Gateway) -> None:
    response, received = _send(
        gateway,
        "/v1/chat/completions",
        {
            "tools": _CHAT_TOOLS,
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [_call(_FIRST_ID, "Paris")]},
                {"role": "tool", "tool_call_id": _FIRST_ID, "content": "sunny"},
                {"role": "tool", "tool_call_id": "call_never_issued", "content": "rainy"},
            ],
        },
    )
    assert response.status_code >= 400, response.text
    assert received == ()
