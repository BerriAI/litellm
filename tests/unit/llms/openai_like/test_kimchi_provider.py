"""
Tests for the Kimchi provider identity.

Kimchi exposes an OpenAI-compatible /v1/chat/completions surface at
https://llm.kimchi.dev/openai/v1. Its provider identity must resolve to a
distinct `kimchi` slug so OpenAI-specific pricing and provider-level reporting
never apply to traffic routed through Kimchi.
"""

import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


class TestKimchiProviderIdentity:
    def test_kimchi_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.KIMCHI.value == "kimchi"
        assert "kimchi" in litellm.provider_list

    def test_kimchi_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        kimchi = JSONProviderRegistry.get("kimchi")
        assert kimchi is not None
        assert kimchi.base_url == "https://llm.kimchi.dev/openai/v1"
        assert kimchi.api_key_env == "KIMCHI_API_KEY"
        assert kimchi.api_base_env == "KIMCHI_API_BASE"

    def test_kimchi_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "kimchi" in openai_compatible_providers

    def test_prefixed_model_resolves_to_kimchi_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="kimchi/kimi-k3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "kimi-k3"
        assert provider == "kimchi"
        assert api_base == "https://llm.kimchi.dev/openai/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="kimchi/kimi-k3",
            custom_llm_provider=None,
            api_base="https://kimchi.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "kimchi"
        assert api_base == "https://kimchi.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_kimchi(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("KIMCHI_API_KEY", "sk-test")

        _, provider, api_key, api_base = get_llm_provider(
            model="kimi-k3",
            custom_llm_provider=None,
            api_base="https://llm.kimchi.dev/openai/v1",
            api_key=None,
        )

        assert provider == "kimchi"
        assert api_base == "https://llm.kimchi.dev/openai/v1"
        assert api_key == "sk-test"


def test_kimchi_chat_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KIMCHI_API_KEY", "kimchi-test-key")

    with respx.mock() as upstream:
        route: Final = upstream.post("https://llm.kimchi.dev/openai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-kimchi",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "pong"},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            },
        )
        response: Final = litellm.completion(
            model="kimchi/kimi-k3",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert str(request.url) == "https://llm.kimchi.dev/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer kimchi-test-key"
    body: Final = json.loads(request.content)
    assert body["model"] == "kimi-k3"
    assert response.choices[0].message.content == "pong"


def test_kimchi_cost_map_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent

    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())

    kimchi_entries: Final = {name: cost_map[name] for name in cost_map if name.startswith("kimchi/")}
    assert kimchi_entries
    assert kimchi_entries == {name: backup[name] for name in kimchi_entries}
