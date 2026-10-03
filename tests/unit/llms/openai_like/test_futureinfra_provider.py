import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_futureinfra_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("FUTUREINFRA_API_KEY", "futureinfra-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="futureinfra/openai/gpt-4o-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "openai/gpt-4o-mini"
    assert provider == "futureinfra"
    assert api_key == "futureinfra-test-key"
    assert api_base == "https://futureinfra.ai/v1/ai"


def test_futureinfra_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("FUTUREINFRA_API_KEY", "futureinfra-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="futureinfra/openai/gpt-4o-mini",
        custom_llm_provider=None,
        api_base="https://futureinfra.internal.example/v1/ai",
        api_key="futureinfra-explicit-key",
    )

    assert provider == "futureinfra"
    assert api_key == "futureinfra-explicit-key"
    assert api_base == "https://futureinfra.internal.example/v1/ai"


def test_futureinfra_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    futureinfra = next(provider for provider in providers if provider["litellm_provider"] == "futureinfra")

    assert futureinfra["provider"] == "FUTUREINFRA"
    assert futureinfra["provider_display_name"] == "FutureInfra"
    assert futureinfra["default_model_placeholder"] == "futureinfra/openai/gpt-4o-mini"
    assert {field["key"]: field["required"] for field in futureinfra["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_futureinfra_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["futureinfra"]["endpoints"] == {
        "chat_completions": True,
        "messages": False,
        "responses": False,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
    }


def test_futureinfra_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://futureinfra.ai/v1/ai/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_futureinfra",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from FutureInfra"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="futureinfra/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="futureinfra-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://futureinfra.ai/v1/ai/chat/completions"
    assert request.headers["authorization"] == "Bearer futureinfra-test-key"
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from FutureInfra"
