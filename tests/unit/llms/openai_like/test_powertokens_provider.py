"""
Tests for the PowerTokens JSON-configured provider.
"""

import json
from typing import Final

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy.public_endpoints.public_endpoints import router

POWERTOKENS_API_BASE: Final = "https://api.powertokens.ai/v1"


@pytest.fixture
def public_client() -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_powertokens_resolves_credentials_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POWERTOKENS_API_BASE", raising=False)
    monkeypatch.setenv("POWERTOKENS_API_KEY", "pt-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="powertokens/glm-5.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.2"
    assert provider == "powertokens"
    assert api_key == "pt-env-key"
    assert api_base == POWERTOKENS_API_BASE


def test_powertokens_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POWERTOKENS_API_KEY", "pt-env-key")
    monkeypatch.setenv("POWERTOKENS_API_BASE", "https://pt.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="powertokens/glm-5.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "powertokens"
    assert api_base == "https://pt.internal.example/v1"


def test_powertokens_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POWERTOKENS_API_KEY", "pt-env-key")
    monkeypatch.setenv("POWERTOKENS_API_BASE", "https://pt.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="powertokens/glm-5.2",
        custom_llm_provider=None,
        api_base="https://pt.explicit.example/v1",
        api_key="pt-explicit-key",
    )

    assert provider == "powertokens"
    assert api_key == "pt-explicit-key"
    assert api_base == "https://pt.explicit.example/v1"


def test_powertokens_chat_completion_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POWERTOKENS_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{POWERTOKENS_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_pt",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "glm-5.2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from PowerTokens"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="powertokens/glm-5.2",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="pt-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer pt-test-key"
    assert body["model"] == "glm-5.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from PowerTokens"


def test_powertokens_responses_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POWERTOKENS_API_BASE", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{POWERTOKENS_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_pt",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "glm-5.2",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_pt",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from PowerTokens", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="powertokens/glm-5.2",
            input="Say hello",
            api_key="pt-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer pt-test-key"
    assert body["model"] == "glm-5.2"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from PowerTokens"


def test_powertokens_is_served_in_add_model_form(public_client: TestClient):
    response: Final = public_client.get("/public/providers/fields")
    assert response.status_code == 200
    powertokens: Final = next(p for p in response.json() if p["litellm_provider"] == "powertokens")

    assert {field["key"]: field["required"] for field in powertokens["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
    assert powertokens["default_model_placeholder"].startswith("powertokens/")


def test_powertokens_public_endpoints_match_provider_config(public_client: TestClient):
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    response: Final = public_client.get("/public/endpoints")
    assert response.status_code == 200
    advertised: Final = {
        entry["endpoint"]
        for entry in response.json()["endpoints"]
        if any(p["slug"] == "powertokens" for p in entry["providers"])
    }
    config: Final = JSONProviderRegistry.get("powertokens")

    assert config is not None
    assert advertised
    assert advertised == {path.removeprefix("/v1") for path in config.supported_endpoints}
