import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_uplinkswitch",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-6.1-sol",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from UplinkSwitch"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_uplinkswitch_is_a_registered_provider():
    assert litellm.LlmProviders.UPLINKSWITCH.value == "uplinkswitch"
    assert "uplinkswitch" in litellm.provider_list
    assert "uplinkswitch" in litellm.constants.openai_compatible_providers


def test_uplinkswitch_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UPLINKSWITCH_API_KEY", "uplinkswitch-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="uplinkswitch/gpt-6.1-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-6.1-sol"
    assert provider == "uplinkswitch"
    assert api_key == "uplinkswitch-test-key"
    assert api_base == "https://api.uplinkswitch.com/v1"


def test_uplinkswitch_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UPLINKSWITCH_API_KEY", "uplinkswitch-env-key")
    monkeypatch.setenv("UPLINKSWITCH_API_BASE", "https://uplinkswitch.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="uplinkswitch/gpt-6.1-sol",
        custom_llm_provider=None,
        api_base="https://uplinkswitch.internal.example/v1",
        api_key="uplinkswitch-explicit-key",
    )

    assert provider == "uplinkswitch"
    assert api_key == "uplinkswitch-explicit-key"
    assert api_base == "https://uplinkswitch.internal.example/v1"


def test_uplinkswitch_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UPLINKSWITCH_API_KEY", "uplinkswitch-env-key")
    monkeypatch.setenv("UPLINKSWITCH_API_BASE", "https://uplinkswitch.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="uplinkswitch/gpt-6.1-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "uplinkswitch"
    assert api_base == "https://uplinkswitch.env.example/v1"


def test_uplinkswitch_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UPLINKSWITCH_API_KEY", "uplinkswitch-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-6.1-sol",
        custom_llm_provider=None,
        api_base="https://api.uplinkswitch.com/v1",
        api_key=None,
    )

    assert model == "gpt-6.1-sol"
    assert provider == "uplinkswitch"
    assert api_key == "uplinkswitch-env-key"
    assert api_base == "https://api.uplinkswitch.com/v1"


def test_uplinkswitch_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    uplinkswitch = next(provider for provider in providers if provider["litellm_provider"] == "uplinkswitch")

    assert uplinkswitch["provider"] == "UPLINKSWITCH"
    assert uplinkswitch["provider_display_name"] == "UplinkSwitch"
    assert uplinkswitch["default_model_placeholder"] == "uplinkswitch/gpt-6.1-sol"
    assert {field["key"]: field["required"] for field in uplinkswitch["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_uplinkswitch_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["uplinkswitch"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["uplinkswitch"]["endpoints"] == expected


def test_uplinkswitch_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.uplinkswitch.com/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="uplinkswitch/gpt-6.1-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="uplinkswitch-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.uplinkswitch.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer uplinkswitch-test-key"
    assert body["model"] == "gpt-6.1-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from UplinkSwitch"


def test_uplinkswitch_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.uplinkswitch.com/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.responses(
            model="uplinkswitch/gpt-6.1-sol",
            input="Say hello",
            api_key="uplinkswitch-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer uplinkswitch-test-key"
    assert body["model"] == "gpt-6.1-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from UplinkSwitch"


@pytest.mark.asyncio
async def test_uplinkswitch_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.uplinkswitch.com/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="uplinkswitch/gpt-6.1-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="uplinkswitch-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer uplinkswitch-test-key"
    assert body["model"] == "gpt-6.1-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from UplinkSwitch"
