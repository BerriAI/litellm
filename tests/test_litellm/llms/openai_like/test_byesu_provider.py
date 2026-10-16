"""
Tests for the `byesu` JSON-configured OpenAI-compatible provider.

byesu is registered purely via litellm/llms/openai_like/providers.json, so these
tests exercise the JSON registry, provider routing, and the dynamically generated
config class. No network calls are made.
"""

import os
import sys

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # allow standalone execution

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import litellm  # noqa: E402
from litellm import get_llm_provider  # noqa: E402


class TestByesuProvider:
    def test_byesu_loaded_from_json(self):
        """byesu is discovered by the JSON provider registry."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("byesu")

        cfg = JSONProviderRegistry.get("byesu")
        assert cfg is not None
        assert cfg.base_url == "https://byesu.com/v1"
        assert cfg.api_key_env == "BYESU_API_KEY"
        assert cfg.api_base_env == "BYESU_API_BASE"

    def test_byesu_in_openai_compatible_providers(self):
        """byesu is registered as an OpenAI-compatible provider."""
        from litellm.constants import openai_compatible_providers

        assert "byesu" in openai_compatible_providers

    def test_byesu_get_llm_provider_routing(self):
        """`byesu/<model>` resolves to custom_llm_provider == 'byesu'."""
        model, custom_llm_provider, _, _ = get_llm_provider(model="byesu/gpt-4o-mini")
        assert custom_llm_provider == "byesu"
        assert model == "gpt-4o-mini"

    def test_byesu_api_base_and_key_resolution(self):
        """The dynamic config resolves the byesu base URL and honors overrides."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        config_class = create_config_class(JSONProviderRegistry.get("byesu"))
        config = config_class()

        # Defaults to the byesu base URL
        api_base, _ = config._get_openai_compatible_provider_info(None, "test-key")
        assert api_base == "https://byesu.com/v1"

        # Explicit overrides win
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://custom.example.com/v1", "override-key"
        )
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "override-key"

    def test_byesu_param_mapping(self):
        """max_completion_tokens is mapped to max_tokens per providers.json."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        config_class = create_config_class(JSONProviderRegistry.get("byesu"))
        config = config_class()

        result = config.map_openai_params(
            {"max_completion_tokens": 128},
            {},
            "gpt-4o-mini",
            False,
        )
        assert result.get("max_tokens") == 128
        assert "max_completion_tokens" not in result

    def test_byesu_declares_responses_api(self):
        """byesu declares /v1/responses support."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("byesu")


if __name__ == "__main__":
    t = TestByesuProvider()
    for name in dir(t):
        if name.startswith("test_"):
            getattr(t, name)()
            print(f"PASS {name}")
    print("All byesu provider tests passed.")
