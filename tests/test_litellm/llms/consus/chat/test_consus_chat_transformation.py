import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.consus.chat.transformation import (
    CONSUS_API_BASE,
    ConsusChatConfig,
)

TEST_MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.fixture(autouse=True)
def _clear_consus_env(monkeypatch):
    monkeypatch.delenv("CONSUS_API_KEY", raising=False)
    monkeypatch.delenv("CONSUS_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "consus_key", None, raising=False)
    yield


class TestConsusChatConfigAuth:
    def test_validate_environment_uses_x_api_key(self):
        config = ConsusChatConfig()
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5:il2",
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base=CONSUS_API_BASE,
        )

        assert headers["x-api-key"] == "test-key"
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_raises_when_no_key(self):
        config = ConsusChatConfig()
        with pytest.raises(ValueError, match="Missing Consus API key"):
            config.validate_environment(
                headers={},
                model="claude-sonnet-4-5:il2",
                messages=TEST_MESSAGES,  # type: ignore
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=CONSUS_API_BASE,
            )

    def test_validate_environment_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("CONSUS_API_KEY", "env-key")
        config = ConsusChatConfig()
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5:il2",
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=CONSUS_API_BASE,
        )
        assert headers["x-api-key"] == "env-key"

    def test_validate_environment_resolves_from_litellm_module(self, monkeypatch):
        monkeypatch.setattr(litellm, "consus_key", "module-key", raising=False)
        config = ConsusChatConfig()
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5:il2",
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=CONSUS_API_BASE,
        )
        assert headers["x-api-key"] == "module-key"

    def test_validate_environment_arg_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CONSUS_API_KEY", "env-key")
        monkeypatch.setattr(litellm, "consus_key", "module-key", raising=False)
        config = ConsusChatConfig()
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5:il2",
            messages=TEST_MESSAGES,  # type: ignore
            optional_params={},
            litellm_params={},
            api_key="arg-key",
            api_base=CONSUS_API_BASE,
        )
        assert headers["x-api-key"] == "arg-key"


class TestConsusProviderInfo:
    def test_default_api_base(self):
        config = ConsusChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(None, "k")
        assert api_base == CONSUS_API_BASE
        assert api_base == "https://api.consus.io/v1"

    def test_explicit_api_base_wins(self):
        config = ConsusChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(
            "https://other.example/v1", "k"
        )
        assert api_base == "https://other.example/v1"

    def test_env_api_base_used_when_arg_none(self, monkeypatch):
        monkeypatch.setenv("CONSUS_API_BASE", "https://staging.consus.io/v1")
        config = ConsusChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(None, "k")
        assert api_base == "https://staging.consus.io/v1"


class TestConsusModelRouting:
    def test_get_llm_provider_strips_prefix_keeps_colon(self, monkeypatch):
        monkeypatch.setenv("CONSUS_API_KEY", "test-key")
        model, provider, api_key, api_base = litellm.get_llm_provider(
            "consus/claude-sonnet-4-5:il5+itar"
        )
        assert provider == "consus"
        # The compliance suffix `:il5+itar` must be preserved end-to-end —
        # only the leading `consus/` prefix is stripped.
        assert model == "claude-sonnet-4-5:il5+itar"
        assert api_base == "https://api.consus.io/v1"
        assert api_key == "test-key"

    def test_get_llm_provider_for_il2_models(self, monkeypatch):
        monkeypatch.setenv("CONSUS_API_KEY", "test-key")
        model, provider, _, _ = litellm.get_llm_provider("consus/claude-opus-4-6:il2")
        assert provider == "consus"
        assert model == "claude-opus-4-6:il2"

    def test_get_llm_provider_for_gemini(self, monkeypatch):
        monkeypatch.setenv("CONSUS_API_KEY", "test-key")
        model, provider, _, _ = litellm.get_llm_provider("consus/gemini-2-5-pro:il5")
        assert provider == "consus"
        assert model == "gemini-2-5-pro:il5"


class TestConsusReasoningSupport:
    """Verify that `ConsusChatConfig.get_supported_openai_params` appends
    `reasoning_effort` if and only if `litellm.supports_reasoning` reports
    true for the model — so the param survives LiteLLM's pre-call filter
    and actually reaches the Consus Gateway.

    `litellm.supports_reasoning` is mocked so the test is independent of
    whichever model registry happens to be loaded (local JSON vs. fetched
    from GitHub).
    """

    def test_reasoning_effort_appended_when_model_supports_reasoning(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            litellm,
            "supports_reasoning",
            lambda model=None, custom_llm_provider=None: True,
        )
        params = ConsusChatConfig().get_supported_openai_params(
            "claude-sonnet-4-5:il2"
        )
        assert "reasoning_effort" in params

    def test_reasoning_effort_not_appended_for_non_reasoning_model(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            litellm,
            "supports_reasoning",
            lambda model=None, custom_llm_provider=None: False,
        )
        params = ConsusChatConfig().get_supported_openai_params("gpt-4.1:il5+itar")
        assert "reasoning_effort" not in params

    def test_supports_reasoning_called_with_consus_provider(self, monkeypatch):
        captured: dict = {}

        def fake_supports_reasoning(model=None, custom_llm_provider=None):
            captured["model"] = model
            captured["provider"] = custom_llm_provider
            return False

        monkeypatch.setattr(litellm, "supports_reasoning", fake_supports_reasoning)
        ConsusChatConfig().get_supported_openai_params("claude-sonnet-4-5:il2")
        # The override must look the model up under provider "consus" so the
        # registry's `supports_reasoning: true` flag is honored.
        assert captured["provider"] == "consus"
        assert captured["model"] == "claude-sonnet-4-5:il2"

    def test_get_supported_openai_params_swallows_lookup_errors(
        self, monkeypatch
    ):
        # If the registry lookup throws (e.g. unknown model), the override
        # must fall back to the parent's param list rather than propagating.
        def boom(model=None, custom_llm_provider=None):
            raise RuntimeError("registry blew up")

        monkeypatch.setattr(litellm, "supports_reasoning", boom)
        params = ConsusChatConfig().get_supported_openai_params(
            "claude-sonnet-4-5:il2"
        )
        assert "reasoning_effort" not in params  # parent default has none
        assert isinstance(params, list)

    def test_custom_llm_provider_returns_consus(self):
        assert ConsusChatConfig().custom_llm_provider == "consus"
