import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import get_llm_provider, get_supported_openai_params
from litellm.llms.kluster_ai.chat.transformation import KlusterAIConfig


class TestKlusterAIConfig:
    def setup_method(self):
        self.config = KlusterAIConfig()

    def test_custom_llm_provider(self):
        assert self.config.custom_llm_provider == "kluster_ai"

    def test_get_api_key(self):
        assert self.config.get_api_key("test-key") == "test-key"

        with patch(
            "litellm.llms.kluster_ai.chat.transformation.get_secret_str",
            return_value="env-key",
        ):
            assert self.config.get_api_key() == "env-key"

        with patch.dict(os.environ, {"KLUSTER_AI_API_KEY": "env-key"}, clear=False):
            assert self.config.get_api_key() == "env-key"

    def test_get_api_base_precedence(self):
        # Explicit argument wins over everything.
        assert (
            self.config.get_api_base("https://custom-base.com/v1")
            == "https://custom-base.com/v1"
        )

        # KLUSTER_AI_API_BASE override.
        with patch(
            "litellm.llms.kluster_ai.chat.transformation.get_secret_str",
            return_value="https://proxy.internal/v1",
        ):
            assert self.config.get_api_base() == "https://proxy.internal/v1"

        # Falls back to the default endpoint.
        with patch(
            "litellm.llms.kluster_ai.chat.transformation.get_secret_str",
            return_value=None,
        ):
            assert self.config.get_api_base() == KlusterAIConfig.API_BASE_URL
            assert KlusterAIConfig.API_BASE_URL == "https://api.kluster.ai/v1"

    def test_get_openai_compatible_provider_info(self):
        def fake_secret(name, *args, **kwargs):
            return "sk-secret" if name == "KLUSTER_AI_API_KEY" else None

        with patch(
            "litellm.llms.kluster_ai.chat.transformation.get_secret_str",
            side_effect=fake_secret,
        ):
            api_base, api_key = self.config._get_openai_compatible_provider_info(
                api_base=None, api_key=None
            )
        assert api_base == "https://api.kluster.ai/v1"
        assert api_key == "sk-secret"

    def test_supported_params_include_tools(self):
        params = self.config.get_supported_openai_params(
            model="klusterai/Meta-Llama-3.1-405B-Instruct-Turbo"
        )
        for expected in ("temperature", "stream", "tools", "tool_choice"):
            assert expected in params


class TestKlusterAIProviderResolution:
    def test_get_llm_provider_resolves_prefixed_model(self):
        with patch.dict(os.environ, {"KLUSTER_AI_API_KEY": "sk-secret"}, clear=False):
            model, provider, api_key, api_base = get_llm_provider(
                model="kluster_ai/Meta-Llama-3.1-405B-Instruct-Turbo"
            )
        assert model == "Meta-Llama-3.1-405B-Instruct-Turbo"
        assert provider == "kluster_ai"
        assert api_key == "sk-secret"
        assert api_base == "https://api.kluster.ai/v1"

    def test_get_llm_provider_detects_provider_from_api_base(self):
        _, provider, _, _ = get_llm_provider(
            model="Meta-Llama-3.1-405B-Instruct-Turbo",
            api_base="https://api.kluster.ai/v1",
            api_key="sk-secret",
        )
        assert provider == "kluster_ai"

    def test_get_supported_openai_params_routes_to_config(self):
        params = get_supported_openai_params(
            model="Meta-Llama-3.1-405B-Instruct-Turbo",
            custom_llm_provider="kluster_ai",
        )
        assert params is not None
        assert "tools" in params

    def test_provider_registered_in_enum_and_lists(self):
        from litellm.types.utils import LlmProviders

        assert LlmProviders.KLUSTER_AI.value == "kluster_ai"
        assert "kluster_ai" in litellm.openai_compatible_providers
        assert "api.kluster.ai/v1" in litellm.openai_compatible_endpoints
