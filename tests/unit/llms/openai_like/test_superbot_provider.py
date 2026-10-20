import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

# Stand-in credentials for the mocked upstream. They are not real keys.
EXPLICIT_CREDENTIAL: Final = "explicit-credential-for-tests"
ENV_CREDENTIAL: Final = "env-credential-for-tests"

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_superbot",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "superbot",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from superbot"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_superbot_is_a_registered_provider():
    assert litellm.LlmProviders.SUPERBOT.value == "superbot-gg"
    assert "superbot-gg" in litellm.provider_list
    assert "superbot-gg" in litellm.constants.openai_compatible_providers


def test_superbot_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SUPERBOT_API_KEY", ENV_CREDENTIAL)

    model, provider, resolved_key, api_base = get_llm_provider(
        model="superbot-gg/superbot-fast",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "superbot-fast"
    assert provider == "superbot-gg"
    assert resolved_key == ENV_CREDENTIAL
    assert api_base == "https://api.superbot.gg/v1"


def test_superbot_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SUPERBOT_API_KEY", ENV_CREDENTIAL)
    monkeypatch.setenv("SUPERBOT_API_BASE", "https://superbot.env.example/v1")

    _, provider, resolved_key, api_base = get_llm_provider(
        model="superbot-gg/superbot",
        custom_llm_provider=None,
        api_base="https://superbot.internal.example/v1",
        api_key=EXPLICIT_CREDENTIAL,
    )

    assert provider == "superbot-gg"
    assert resolved_key == EXPLICIT_CREDENTIAL
    assert api_base == "https://superbot.internal.example/v1"


def test_superbot_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SUPERBOT_API_KEY", ENV_CREDENTIAL)
    monkeypatch.setenv("SUPERBOT_API_BASE", "https://superbot.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="superbot-gg/superbot",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "superbot-gg"
    assert api_base == "https://superbot.env.example/v1"


def test_superbot_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SUPERBOT_API_KEY", ENV_CREDENTIAL)

    model, provider, resolved_key, api_base = get_llm_provider(
        model="superbot",
        custom_llm_provider=None,
        api_base="https://api.superbot.gg/v1",
        api_key=None,
    )

    assert model == "superbot"
    assert provider == "superbot-gg"
    assert resolved_key == ENV_CREDENTIAL
    assert api_base == "https://api.superbot.gg/v1"


def test_superbot_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    superbot = next(provider for provider in providers if provider["litellm_provider"] == "superbot-gg")

    assert superbot["provider"] == "Superbot"
    assert superbot["provider_display_name"] == "superbot"
    assert superbot["default_model_placeholder"] == "superbot-gg/superbot"
    assert {field["key"]: field["required"] for field in superbot["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_superbot_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["superbot-gg"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["superbot-gg"]["endpoints"] == expected


@pytest.mark.parametrize(
    "model_id, context",
    [("superbot-gg/superbot", 1_000_000), ("superbot-gg/superbot-fast", 500_000)],
)
def test_superbot_models_are_priced_in_both_cost_maps(model_id: str, context: int):
    root = Path(litellm.__file__).parent.parent
    maps: Final = [
        root / "model_prices_and_context_window.json",
        Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json",
    ]
    for path in maps:
        row = json.loads(path.read_text())[model_id]
        assert row["litellm_provider"] == "superbot-gg"
        assert row["mode"] == "chat"
        assert row["max_input_tokens"] == context
        assert row["max_output_tokens"] == 65_536
        assert row["input_cost_per_token"] == pytest.approx(1.0e-6)
        assert row["output_cost_per_token"] == pytest.approx(5.0e-6)
        assert row["cache_read_input_token_cost"] == pytest.approx(5.0e-8)
        assert row["cache_creation_input_token_cost"] == pytest.approx(1.25e-6)
        assert row["supports_function_calling"] is True


def test_superbot_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.superbot.gg/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="superbot-gg/superbot",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key=EXPLICIT_CREDENTIAL,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.superbot.gg/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {EXPLICIT_CREDENTIAL}"
    assert body["model"] == "superbot"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from superbot"


def test_superbot_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.superbot.gg/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="superbot-gg/superbot",
            input="Say hello",
            api_key=EXPLICIT_CREDENTIAL,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == f"Bearer {EXPLICIT_CREDENTIAL}"
    assert body["model"] == "superbot"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from superbot"


@pytest.mark.asyncio
async def test_superbot_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.superbot.gg/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="superbot-gg/superbot",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key=EXPLICIT_CREDENTIAL,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == f"Bearer {EXPLICIT_CREDENTIAL}"
    assert body["model"] == "superbot"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from superbot"
