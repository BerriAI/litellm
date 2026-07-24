import os
import pytest
from unittest.mock import patch
from litellm.llms.opencode_go.chat import OpenCodeGoConfig
import litellm
from litellm.utils import get_optional_params

class TestOpenCodeGoParamMapping:
    """Test OpenCode Go param mapping — fails if mapping breaks or params silently get dropped."""

    def test_supported_params_are_mapped(self):
        config = OpenCodeGoConfig()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 100, "temperature": 0.7},
            optional_params={},
            model="opencode_go/some-model",
            drop_params=False,
        )
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7

    def test_unsupported_param_is_dropped(self):
        config = OpenCodeGoConfig()
        result = config.map_openai_params(
            non_default_params={"not_a_real_param": "x"},
            optional_params={},
            model="opencode_go/some-model",
            drop_params=False,
        )
        assert "not_a_real_param" not in result

class TestOpenCodeGoProviderWiring:
    def test_get_llm_provider_from_prefix(self):
        model, provider, _, api_base = litellm.get_llm_provider(model="opencode_go/deepseek-v4-flash")
        assert model == "deepseek-v4-flash"
        assert provider == "opencode_go"
        assert api_base == "https://opencode.ai/zen/go/v1"

    def test_get_llm_provider_from_api_base(self):
        _, provider, _, _ = litellm.get_llm_provider(
            model="deepseek-v4-flash",
            api_base="https://opencode.ai/zen/go/v1",
        )
        assert provider == "opencode_go"

    def test_get_supported_openai_params(self):
        params = litellm.get_supported_openai_params(
            model="deepseek-v4-flash",
            custom_llm_provider="opencode_go",
        )
        assert "max_tokens" in params
        assert "tool_choice" in params

    def test_get_optional_params_maps_supported(self):
        result = get_optional_params(
            model="deepseek-v4-flash",
            custom_llm_provider="opencode_go",
            temperature=0.5,
            max_tokens=10,
        )
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 10

    def test_validate_environment_present(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test")
        result = litellm.validate_environment("opencode_go/deepseek-v4-flash")
        assert result["keys_in_environment"] is True

    def test_validate_environment_missing(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        result = litellm.validate_environment("opencode_go/deepseek-v4-flash")
        assert "OPENCODE_GO_API_KEY" in result["missing_keys"]

if __name__ == "__main__":
    pytest.main([__file__])
