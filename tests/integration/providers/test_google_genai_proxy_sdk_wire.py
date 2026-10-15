import json
from collections.abc import Callable, Mapping
from typing import Final, Literal, TypeAlias

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google import genai
from google.genai import types
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gemini-2.5-flash"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_PROMPT: Final = "Reply with only the single word: pong"
_PROVIDERS: Final[tuple[Literal["gemini", "vertex_ai"], ...]] = ("gemini", "vertex_ai")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
Provider: TypeAlias = Literal["gemini", "vertex_ai"]


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


def _register_model(gateway: Gateway, scenario: Scenario, provider: Provider, wire: Wire) -> str:
    if provider == "gemini":
        return scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=f"{wire.url}{_MODEL_PATH}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=_service_account_json(wire.url),
    )


def _target(provider: Provider, stream: bool) -> str:
    path: Final = f"/models/{_BACKEND}" if provider == "gemini" else _MODEL_PATH
    if stream:
        return f"{path}:streamGenerateContent?alt=sse"
    return f"{path}:generateContent"


def _response_body() -> Mapping[str, JsonValue]:
    return {
        "responseId": "scripted-response",
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "pong"}]},
                "index": 0,
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1, "totalTokenCount": 10},
        "modelVersion": _BACKEND,
    }


def _expected_request_body(provider: Provider) -> Mapping[str, JsonValue]:
    request: Final = {
        "model": _BACKEND,
        "contents": [{"parts": [{"text": _PROMPT}], "role": "user"}],
    }
    if provider == "gemini":
        return {
            **request,
            "tools": None,
            "generationConfig": {"temperature": 0, "topP": 0.95, "topK": 20},
        }
    return {
        **request,
        "generationConfig": {"temperature": 0, "top_p": 0.95, "top_k": 20},
    }


def _respond(provider: Provider, stream: bool) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            return Reply(
                body=b'{"access_token":"scripted-token","expires_in":3600,"token_type":"Bearer"}'
            )
        assert request.target == _target(provider, stream)
        if provider == "gemini":
            assert request.headers["x-goog-api-key"] == _API_KEY
        else:
            assert request.headers["authorization"] == "Bearer scripted-token"
        assert _JSON_OBJECT.validate_json(request.body) == _expected_request_body(provider)
        if stream:
            return Reply(
                content_type="text/event-stream",
                chunks=(f"data: {json.dumps(_response_body())}\n\n".encode(),),
            )
        return Reply(body=json.dumps(_response_body()).encode())

    return respond


def _client(gateway: Gateway) -> genai.Client:
    return genai.Client(
        api_key=gateway.key,
        http_options={"base_url": str(gateway.client.base_url)},
    )


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_google_genai_sdk_non_streaming_forwards_body_and_response(
    gateway: Gateway, provider: Provider
) -> None:
    with wire_server(_respond(provider, stream=False)) as wire, gateway.scenario() as scenario:
        model: Final = _register_model(gateway, scenario, provider, wire)
        response: Final = _client(gateway).models.generate_content(
            model=model,
            contents=types.Part.from_text(text=_PROMPT),
            config=types.GenerateContentConfig(temperature=0, top_p=0.95, top_k=20),
        )
        assert response.text == "pong"
        assert response.usage_metadata.prompt_token_count == 9
        assert response.usage_metadata.candidates_token_count == 1


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_google_genai_sdk_streaming_forwards_body_and_response(
    gateway: Gateway, provider: Provider
) -> None:
    with wire_server(_respond(provider, stream=True)) as wire, gateway.scenario() as scenario:
        model: Final = _register_model(gateway, scenario, provider, wire)
        chunks: Final = tuple(
            _client(gateway).models.generate_content_stream(
                model=model,
                contents=types.Part.from_text(text=_PROMPT),
                config=types.GenerateContentConfig(temperature=0, top_p=0.95, top_k=20),
            )
        )
        assert tuple(chunk.text for chunk in chunks) == ("pong",)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_google_genai_sdk_dict_streaming_forwards_body_and_response(
    gateway: Gateway, provider: Provider
) -> None:
    with wire_server(_respond(provider, stream=True)) as wire, gateway.scenario() as scenario:
        model: Final = _register_model(gateway, scenario, provider, wire)
        chunks: Final = tuple(
            _client(gateway).models.generate_content_stream(
                model=model,
                contents={"text": _PROMPT},
                config={"temperature": 0, "top_p": 0.95, "top_k": 20},
            )
        )
        assert tuple(chunk.text for chunk in chunks) == ("pong",)
