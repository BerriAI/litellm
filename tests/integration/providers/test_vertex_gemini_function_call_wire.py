import json
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

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
        assert body["generationConfig"]["top_k"] == 3, body
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
                "top_k": 3,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "done"
