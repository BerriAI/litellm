import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gemini-2.5-flash"
_API_KEY: Final = "synthetic-gemini-key"
_CACHE_NAME: Final = "cachedContents/synthetic-cache"
_CACHED_POLICY: Final = " ".join(f"policy clause {index} applies" for index in range(600))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _generate_content_reply(text: str) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {
                "promptTokenCount": 1300,
                "candidatesTokenCount": 5,
                "totalTokenCount": 1305,
                "cachedContentTokenCount": 1290,
            },
            "modelVersion": _BACKEND,
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.gemini.messages_cache_control_creates_cached_content_with_anthropic_ttl")
def test_gemini_messages_cache_control_creates_cached_content_and_generates_from_it(gateway: Gateway) -> None:
    identity: Final = f"gemini-messages-cache-{uuid.uuid4().hex}"
    user_prompt: Final = f"Summarize the policy. Request {identity}."

    def respond(request: Request) -> Reply:
        assert request.headers["x-goog-api-key"] == _API_KEY, request.headers
        if request.method == "GET":
            assert request.target == f"/models/{_BACKEND}:cachedContents", request.target
            return Reply(body=b"{}")
        assert request.method == "POST", request.method
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if request.target == f"/models/{_BACKEND}:cachedContents":
            assert isinstance(body["displayName"], str) and body["displayName"], body
            assert body == {
                "contents": [{"role": "user", "parts": [{"text": "."}]}],
                "model": f"models/{_BACKEND}",
                "displayName": body["displayName"],
                "ttl": "300s",
                "system_instruction": {"parts": [{"text": _CACHED_POLICY}]},
                "tools": None,
            }
            return Reply(body=json.dumps({"name": _CACHE_NAME, "model": f"models/{_BACKEND}"}).encode())
        assert request.target == f"/models/{_BACKEND}:generateContent", request.target
        assert body == {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"max_output_tokens": 32},
            "cachedContent": _CACHE_NAME,
        }
        return Reply(body=_generate_content_reply("The policy applies."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 32,
                "system": [
                    {"type": "text", "text": _CACHED_POLICY, "cache_control": {"type": "ephemeral", "ttl": "5m"}}
                ],
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": "The policy applies."}], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("GET", f"/models/{_BACKEND}:cachedContents"),
            ("POST", f"/models/{_BACKEND}:cachedContents"),
            ("POST", f"/models/{_BACKEND}:generateContent"),
        ]
