import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

PHALA_API_BASE: Final = "https://inference.phala.com/v1"
PHALA_MODEL_ID: Final = "z-ai/glm-5.3"
PHALA_MODEL: Final = f"phala/{PHALA_MODEL_ID}"


@pytest.mark.parametrize("custom_provider", (None, "phala"))
def test_phala_resolves_environment_credentials(monkeypatch: pytest.MonkeyPatch, custom_provider: str | None) -> None:
    monkeypatch.setenv("PHALA_AI_API_KEY", "phala-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=PHALA_MODEL if custom_provider is None else PHALA_MODEL_ID,
        custom_llm_provider=custom_provider,
    )

    assert (model, provider, api_key, api_base) == (PHALA_MODEL_ID, "phala", "phala-env-key", PHALA_API_BASE)


def test_phala_detected_from_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHALA_AI_API_KEY", "phala-env-key")

    model, provider, api_key, api_base = get_llm_provider(model=PHALA_MODEL_ID, api_base=PHALA_API_BASE)

    assert (model, provider, api_key, api_base) == (PHALA_MODEL_ID, "phala", "phala-env-key", PHALA_API_BASE)


def test_phala_explicit_credentials_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHALA_AI_API_KEY", "phala-env-key")
    monkeypatch.setenv("PHALA_API_BASE", "https://environment.example/v1")

    model, provider, api_key, api_base = get_llm_provider(
        model=PHALA_MODEL,
        api_base="https://explicit.example/v1",
        api_key="phala-explicit-key",
    )

    assert (model, provider, api_key, api_base) == (
        PHALA_MODEL_ID,
        "phala",
        "phala-explicit-key",
        "https://explicit.example/v1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.parametrize("api_base", (PHALA_API_BASE, "https://override.example/v1"))
async def test_phala_chat_completion_request(monkeypatch: pytest.MonkeyPatch, is_async: bool, api_base: str) -> None:
    monkeypatch.setenv("PHALA_AI_API_KEY", "phala-test-key")
    monkeypatch.setenv("PHALA_API_BASE", api_base)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{api_base}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_phala",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": PHALA_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Phala"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = (
            await litellm.acompletion(
                model=PHALA_MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                reasoning_effort="low",
                reasoning={"effort": "low"},
            )
            if is_async
            else litellm.completion(
                model=PHALA_MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                reasoning_effort="low",
                reasoning={"effort": "low"},
            )
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer phala-test-key"
    assert json.loads(request.content) == {
        "model": PHALA_MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello"}],
        "reasoning_effort": "low",
        "reasoning": {"effort": "low"},
    }
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "Hello from Phala"
    assert response.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_phala_available_in_add_model() -> None:
    from litellm.proxy.public_endpoints.public_endpoints import get_provider_fields, get_supported_providers

    assert "phala" in await get_supported_providers()
    providers: Final = await get_provider_fields()
    phala: Final = next(provider for provider in providers if provider.litellm_provider == "phala")

    assert phala.provider_display_name == "Phala"
    assert phala.default_model_placeholder == PHALA_MODEL
    assert {field.key: field.required for field in phala.credential_fields} == {"api_base": False, "api_key": True}


@pytest.mark.asyncio
async def test_phala_listed_for_chat_completions() -> None:
    from litellm.proxy.public_endpoints.public_endpoints import get_supported_endpoints

    endpoints: Final = await get_supported_endpoints()
    chat: Final = next(endpoint for endpoint in endpoints.endpoints if endpoint.key == "chat_completions")

    assert any(provider.slug == "phala" and provider.display_name == "Phala" for provider in chat.providers)
