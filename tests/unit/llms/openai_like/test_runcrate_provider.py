import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_runcrate",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-ai/DeepSeek-V3.2",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Runcrate"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_runcrate_is_a_registered_provider():
    assert litellm.LlmProviders.RUNCRATE.value == "runcrate"
    assert "runcrate" in litellm.provider_list
    assert "runcrate" in litellm.constants.openai_compatible_providers


def test_runcrate_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RUNCRATE_API_KEY", "runcrate-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="runcrate/deepseek-ai/DeepSeek-V3.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-ai/DeepSeek-V3.2"
    assert provider == "runcrate"
    assert api_key == "runcrate-test-key"
    assert api_base == "https://api.runcrate.ai/v1"


def test_runcrate_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RUNCRATE_API_KEY", "runcrate-env-key")
    monkeypatch.setenv("RUNCRATE_API_BASE", "https://runcrate.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="runcrate/deepseek-ai/DeepSeek-V3.2",
        custom_llm_provider=None,
        api_base="https://runcrate.internal.example/v1",
        api_key="runcrate-explicit-key",
    )

    assert provider == "runcrate"
    assert api_key == "runcrate-explicit-key"
    assert api_base == "https://runcrate.internal.example/v1"


def test_runcrate_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RUNCRATE_API_KEY", "runcrate-env-key")
    monkeypatch.setenv("RUNCRATE_API_BASE", "https://runcrate.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="runcrate/deepseek-ai/DeepSeek-V3.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "runcrate"
    assert api_base == "https://runcrate.env.example/v1"


def test_runcrate_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RUNCRATE_API_KEY", "runcrate-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-ai/DeepSeek-V3.2",
        custom_llm_provider=None,
        api_base="https://api.runcrate.ai/v1",
        api_key=None,
    )

    assert model == "deepseek-ai/DeepSeek-V3.2"
    assert provider == "runcrate"
    assert api_key == "runcrate-env-key"
    assert api_base == "https://api.runcrate.ai/v1"


def test_runcrate_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    runcrate = next(provider for provider in providers if provider["litellm_provider"] == "runcrate")

    assert runcrate["provider"] == "RUNCRATE"
    assert runcrate["provider_display_name"] == "Runcrate"
    assert runcrate["default_model_placeholder"] == "runcrate/deepseek-ai/DeepSeek-V3.2"
    assert {field["key"]: field["required"] for field in runcrate["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_runcrate_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["runcrate"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["runcrate"]["endpoints"] == expected


def test_runcrate_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.runcrate.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="runcrate/deepseek-ai/DeepSeek-V3.2",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="runcrate-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.runcrate.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer runcrate-test-key"
    assert body["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Runcrate"


def test_runcrate_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.runcrate.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="runcrate/deepseek-ai/DeepSeek-V3.2",
            input="Say hello",
            api_key="runcrate-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer runcrate-test-key"
    assert body["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Runcrate"


@pytest.mark.asyncio
async def test_runcrate_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.runcrate.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="runcrate/deepseek-ai/DeepSeek-V3.2",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="runcrate-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer runcrate-test-key"
    assert body["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Runcrate"
