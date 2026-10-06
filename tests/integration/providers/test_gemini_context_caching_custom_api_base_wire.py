import json
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gemini-3.7-flash"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_VERTEX_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_VERTEX_COLLECTION: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/cachedContents"
_GEMINI_COLLECTION: Final = "/v1beta/cachedContents"
_GEMINI_MODEL_PATH: Final = f"/v1beta/models/{_BACKEND}"
_GEMINI_KEY: Final = "scripted-gemini-key"
_CACHE_NAME: Final = "cachedContents/scripted-cache"
_CACHED_TOKENS: Final = 5120
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CACHED_TEXT: Final = " ".join(
    f"Policy clause {index}: the gateway keeps this clause in the cache." for index in range(700)
)
_MESSAGES: Final = (
    {
        "role": "user",
        "content": [{"type": "text", "text": _CACHED_TEXT, "cache_control": {"type": "ephemeral"}}],
    },
    {"role": "user", "content": "How many clauses are there?"},
)


def _generate_content_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "700"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": _CACHED_TOKENS + 9,
                    "cachedContentTokenCount": _CACHED_TOKENS,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": _CACHED_TOKENS + 11,
                },
                "modelVersion": _BACKEND,
            }
        ).encode()
    )


def _google_responder(collection: str, model_path: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == collection and request.method == "GET":
            return Reply(body=b'{"cachedContents": []}')
        if request.target == collection and request.method == "POST":
            return Reply(body=json.dumps({"name": _CACHE_NAME, "model": f"models/{_BACKEND}"}).encode())
        assert request.target == f"{model_path}:generateContent", f"{request.method} {request.target}"
        return _generate_content_reply()

    return respond


def _chat(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    response: Final = gateway.client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": list(_MESSAGES), "max_tokens": 32},
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def _assert_cache_round_trip(requests: tuple[Request, ...], collection: str, model_path: str) -> None:
    assert tuple((request.method, request.target) for request in requests) == (
        ("GET", collection),
        ("POST", collection),
        ("POST", f"{model_path}:generateContent"),
    ), [(request.method, request.target) for request in requests]
    created: Final = _JSON_OBJECT.validate_json(requests[1].body)
    assert "contents" in created, created
    generated: Final = _JSON_OBJECT.validate_json(requests[2].body)
    assert generated["cachedContent"] == _CACHE_NAME, generated


def _cached_tokens_of(response: dict[str, JsonValue]) -> JsonValue:
    usage: Final = response["usage"]
    assert isinstance(usage, dict)
    details: Final = usage["prompt_tokens_details"]
    assert isinstance(details, dict)
    return details["cached_tokens"]


def test_gemini_custom_api_base_lists_and_creates_the_cached_contents_collection(gateway: Gateway) -> None:
    with (
        wire_server(_google_responder(_GEMINI_COLLECTION, _GEMINI_MODEL_PATH)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=f"{wire.url}/v1beta", api_key=_GEMINI_KEY)
        response: Final = _chat(gateway, model)
        requests: Final = wire.drain()
        _assert_cache_round_trip(requests, _GEMINI_COLLECTION, _GEMINI_MODEL_PATH)
        assert all(request.headers.get("x-goog-api-key") == _GEMINI_KEY for request in requests)
        assert _cached_tokens_of(response) == _CACHED_TOKENS


def test_vertex_model_path_api_base_lists_and_creates_the_cached_contents_collection(gateway: Gateway) -> None:
    with (
        wire_server(_google_responder(_VERTEX_COLLECTION, _VERTEX_MODEL_PATH)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"vertex_ai/{_BACKEND}",
            api_base=f"{wire.url}{_VERTEX_MODEL_PATH}",
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        )
        response: Final = _chat(gateway, model)
        requests: Final = wire.drain()
        _assert_cache_round_trip(requests, _VERTEX_COLLECTION, _VERTEX_MODEL_PATH)
        assert all(request.headers.get("authorization") == "Bearer scripted-token" for request in requests)
        assert _cached_tokens_of(response) == _CACHED_TOKENS
