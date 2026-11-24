import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_mallard",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "thinkingmachines/inkling-2-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Mallard AI"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_mallard_is_a_registered_provider():
    assert litellm.LlmProviders.MALLARD.value == "mallard"
    assert "mallard" in litellm.provider_list
    assert "mallard" in litellm.constants.openai_compatible_providers


def test_mallard_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MALLARD_API_KEY", "mallard-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="mallard/thinkingmachines/inkling-2-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "thinkingmachines/inkling-2-mini"
    assert provider == "mallard"
    assert api_key == "mallard-test-key"
    assert api_base == "https://api.mallardai.co.uk/v1"


def test_mallard_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MALLARD_API_KEY", "mallard-env-key")
    monkeypatch.setenv("MALLARD_API_BASE", "https://mallard.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="mallard/thinkingmachines/inkling-2-mini",
        custom_llm_provider=None,
        api_base="https://mallard.internal.example/v1",
        api_key="mallard-explicit-key",
    )

    assert provider == "mallard"
    assert api_key == "mallard-explicit-key"
    assert api_base == "https://mallard.internal.example/v1"


def test_mallard_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MALLARD_API_KEY", "mallard-env-key")
    monkeypatch.setenv("MALLARD_API_BASE", "https://mallard.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="mallard/thinkingmachines/inkling-2-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "mallard"
    assert api_base == "https://mallard.env.example/v1"


def test_mallard_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MALLARD_API_KEY", "mallard-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="thinkingmachines/inkling-2-mini",
        custom_llm_provider=None,
        api_base="https://api.mallardai.co.uk/v1",
        api_key=None,
    )

    assert model == "thinkingmachines/inkling-2-mini"
    assert provider == "mallard"
    assert api_key == "mallard-env-key"
    assert api_base == "https://api.mallardai.co.uk/v1"


def test_mallard_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    mallard = next(provider for provider in providers if provider["litellm_provider"] == "mallard")

    assert mallard["provider"] == "MALLARD"
    assert mallard["provider_display_name"] == "Mallard AI"
    assert mallard["default_model_placeholder"] == "mallard/thinkingmachines/inkling-2-mini"
    assert {field["key"]: field["required"] for field in mallard["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_mallard_supported_endpoints():
    expected: Final = {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
    }
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["mallard"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["mallard"]["endpoints"] == expected


def test_mallard_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.mallardai.co.uk/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="mallard/thinkingmachines/inkling-2-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="mallard-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.mallardai.co.uk/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer mallard-test-key"
    assert body["model"] == "thinkingmachines/inkling-2-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Mallard AI"


def test_mallard_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.mallardai.co.uk/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="mallard/thinkingmachines/inkling-2-mini",
            input="Say hello",
            api_key="mallard-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer mallard-test-key"
    assert body["model"] == "thinkingmachines/inkling-2-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Mallard AI"


@pytest.mark.asyncio
async def test_mallard_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.mallardai.co.uk/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="mallard/thinkingmachines/inkling-2-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="mallard-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer mallard-test-key"
    assert body["model"] == "thinkingmachines/inkling-2-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Mallard AI"


def test_mallard_model_prices_and_limits():
    info: Final = litellm.get_model_info("mallard/thinkingmachines/inkling-2-mini")

    assert info["litellm_provider"] == "mallard"
    assert info["max_input_tokens"] == 262_144
    assert info["max_output_tokens"] == 65_536
    assert info["input_cost_per_token"] == pytest.approx(4.0104e-08)
    assert info["cache_read_input_token_cost"] == pytest.approx(1.3368e-08)
    assert info["output_cost_per_token"] == pytest.approx(2.6736e-07)
    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_reasoning"] is True


@pytest.mark.parametrize(
    "route_kwargs",
    (
        {"model": "mallard/thinkingmachines/inkling-2-mini"},
        {"model": "thinkingmachines/inkling-2-mini", "api_base": "https://api.mallardai.co.uk/v1"},
    ),
    ids=("prefixed-model", "api-base-autodetection"),
)
def test_mallard_missing_key_rejects_before_openai_fallback(monkeypatch: pytest.MonkeyPatch, route_kwargs: dict):
    monkeypatch.delenv("MALLARD_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(url__regex=r"https://api\.mallardai\.co\.uk/v1/.*").respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="MALLARD_API_KEY"):
            litellm.completion(messages=[{"role": "user", "content": "Hello"}], **route_kwargs)
    assert route.call_count == 0


def test_mallard_chat_completion_uses_authenticated_client(monkeypatch: pytest.MonkeyPatch):
    from openai import OpenAI

    monkeypatch.delenv("MALLARD_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    with OpenAI(api_key="mallard-client-key", base_url="https://api.mallardai.co.uk/v1") as client:
        with respx.mock() as upstream:
            route: Final = upstream.post("https://api.mallardai.co.uk/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
            response: Final = litellm.completion(
                model="mallard/thinkingmachines/inkling-2-mini", messages=[{"role": "user", "content": "Hello"}], client=client
            )
    assert route.calls.last.request.headers["authorization"] == "Bearer mallard-client-key"
    assert response.choices[0].message.content == "Hello from Mallard AI"


def test_mallard_http_transport_rejects_unused_sdk_credentials(monkeypatch: pytest.MonkeyPatch):
    from openai import OpenAI

    monkeypatch.delenv("MALLARD_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")
    with OpenAI(api_key="mallard-client-key", base_url="https://api.mallardai.co.uk/v1") as client:
        with respx.mock(assert_all_called=False) as upstream:
            route: Final = upstream.post("https://api.mallardai.co.uk/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
            with pytest.raises(litellm.AuthenticationError, match="MALLARD_API_KEY"):
                litellm.completion(
                    model="mallard/thinkingmachines/inkling-2-mini", messages=[{"role": "user", "content": "Hello"}], client=client
                )
    assert route.call_count == 0
