"""
Tests for the Hubris provider.

Hubris (https://hubris.pw) is an OpenAI-compatible LLM gateway registered purely
via the JSON provider registry (``litellm/llms/openai_like/providers.json``);
it has no hand-written transformation class. Model ids on Hubris always use the
full ``vendor/model`` form (e.g. ``anthropic/claude-sonnet-5``), which LiteLLM
passes through verbatim after stripping the ``hubris/`` prefix.
"""

import json

import pytest

import litellm
from litellm import completion
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

API_BASE = "https://api.hubris.pw/v1"
CHAT_URL = f"{API_BASE}/chat/completions"
MODEL = "hubris/anthropic/claude-sonnet-5"
BARE_MODEL = "anthropic/claude-sonnet-5"


def _new_config():
    return create_config_class(JSONProviderRegistry.get("hubris"))()


def _completion_payload(content="Hi from Hubris!"):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": BARE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


class TestHubrisProviderRegistration:
    def test_hubris_registered_in_json_registry(self):
        assert JSONProviderRegistry.exists("hubris")

        config = JSONProviderRegistry.get("hubris")
        assert config is not None
        assert config.base_url == API_BASE
        assert config.api_key_env == "HUBRIS_API_KEY"
        assert config.api_base_env == "HUBRIS_API_BASE"

    def test_hubris_listed(self):
        assert "hubris" in JSONProviderRegistry.list_providers()

    def test_hubris_config_defaults(self):
        config = JSONProviderRegistry.get("hubris")
        assert config.base_class == "openai_gpt"
        assert config.param_mappings == {}
        assert config.constraints == {}
        assert config.special_handling == {}

    def test_hubris_advertises_responses_api(self):
        assert JSONProviderRegistry.supports_responses_api("hubris") is True


class TestHubrisProviderResolution:
    def test_get_llm_provider_resolves_hubris(self, monkeypatch):
        monkeypatch.setenv("HUBRIS_API_KEY", "sk-gw-test")

        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider(model=MODEL)
        )

        # Only the leading "hubris/" is stripped; the vendor prefix stays.
        assert model == BARE_MODEL
        assert custom_llm_provider == "hubris"
        assert dynamic_api_key == "sk-gw-test"
        assert api_base == API_BASE

    def test_get_llm_provider_uses_explicit_api_key(self, monkeypatch):
        monkeypatch.setenv("HUBRIS_API_KEY", "env-key")

        _, _, dynamic_api_key, _ = litellm.get_llm_provider(
            model=MODEL, api_key="sk-explicit"
        )
        assert dynamic_api_key == "sk-explicit"

    def test_get_llm_provider_honors_api_base_override(self):
        _, custom_llm_provider, _, api_base = litellm.get_llm_provider(
            model=MODEL, api_base="https://custom.example/v1"
        )
        assert custom_llm_provider == "hubris"
        assert api_base == "https://custom.example/v1"

    def test_get_llm_provider_honors_api_base_env(self, monkeypatch):
        monkeypatch.setenv("HUBRIS_API_BASE", "https://env.example/v1")
        monkeypatch.setenv("HUBRIS_API_KEY", "env-key")

        _, custom_llm_provider, _, api_base = litellm.get_llm_provider(model=MODEL)
        assert custom_llm_provider == "hubris"
        assert api_base == "https://env.example/v1"

class TestHubrisDynamicConfig:
    def test_custom_llm_provider(self):
        assert _new_config().custom_llm_provider == "hubris"

    def test_get_complete_url_appends_chat_completions(self):
        url = _new_config().get_complete_url(
            api_base=API_BASE,
            api_key="k",
            model=BARE_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == CHAT_URL

    def test_get_complete_url_falls_back_to_base_url(self):
        url = _new_config().get_complete_url(
            api_base=None,
            api_key="k",
            model=BARE_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == CHAT_URL

    def test_provider_info_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("HUBRIS_API_KEY", "env-key")
        base, key = _new_config()._get_openai_compatible_provider_info(None, None)
        assert base == API_BASE
        assert key == "env-key"

    def test_standard_sampling_params_pass_through(self):
        out = _new_config().map_openai_params(
            non_default_params={"temperature": 0.6, "top_p": 0.9},
            optional_params={},
            model=BARE_MODEL,
            drop_params=False,
        )
        assert out == {"temperature": 0.6, "top_p": 0.9}


class TestHubrisCompletion:
    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self, monkeypatch):
        # respx mocks the httpx transport, so the aiohttp transport must be off.
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    @pytest.mark.respx()
    def test_completion_sends_correct_url_and_body(self, respx_mock):
        route = respx_mock.post(CHAT_URL).respond(
            json=_completion_payload(), status_code=200
        )

        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": "say hi"}],
            api_key="sk-gw-test",
            api_base=API_BASE,
            temperature=0.6,
        )

        assert route.called
        request = route.calls.last.request
        assert str(request.url) == CHAT_URL
        assert request.headers["authorization"] == "Bearer sk-gw-test"
        body = json.loads(request.content)
        assert body["model"] == BARE_MODEL
        assert body["messages"] == [{"role": "user", "content": "say hi"}]
        assert body["temperature"] == 0.6
        assert response.choices[0].message.content == "Hi from Hubris!"
        assert response.usage.total_tokens == 21

    def test_streaming_yields_content_chunks(self):
        chunks = list(
            completion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-gw-test",
                stream=True,
                mock_response="Hello from Hubris",
            )
        )
        assert len(chunks) > 0
        text = "".join((c.choices[0].delta.content or "") for c in chunks)
        assert "Hello from Hubris" in text
