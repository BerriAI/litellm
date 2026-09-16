import os
import pytest
from unittest.mock import patch
from litellm.llms.baseten.chat import BasetenConfig


class TestBasetenRouting:
    """Test Baseten routing logic"""

    def test_routing_logic(self):
        """Test routing between Model API and dedicated deployments"""
        config = BasetenConfig()

        # Dedicated deployment (8-character alphanumeric)
        assert (
            config.get_api_base_for_model("abcd1234")
            == "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
        )

        # Model API (non-8-character)
        assert (
            config.get_api_base_for_model("openai/gpt-oss-120b")
            == "https://inference.baseten.co/v1"
        )


class TestBasetenModelAPI:
    """Test Baseten Model API inference"""

    @patch.dict(os.environ, {"BASETEN_API_KEY": "test-key"})
    def test_model_api_inference(self):
        """Test Model API inference with basic parameters"""
        config = BasetenConfig()

        # Test parameter mapping
        non_default_params = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="openai/gpt-oss-120b",
            drop_params=False,
        )

        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

        # Test provider info
        api_base, api_key = config._get_openai_compatible_provider_info(
            None, "test-key"
        )
        assert api_base == "https://inference.baseten.co/v1"
        assert api_key == "test-key"


class TestBasetenReasoningEffort:
    """Test reasoning_effort support for Baseten reasoning models.

    Baseten's Model API accepts a top-level `reasoning_effort` parameter and
    validates it server-side (invalid values return 400), so LiteLLM should
    forward it verbatim instead of dropping it:
    https://docs.baseten.co/inference/model-apis/reasoning
    """

    def test_reasoning_effort_is_supported(self):
        """reasoning_effort is advertised as a supported param"""
        config = BasetenConfig()

        assert "reasoning_effort" in config.get_supported_openai_params(
            model="moonshotai/Kimi-K3"
        )

    def test_reasoning_effort_passthrough_with_drop_params(self):
        """reasoning_effort survives drop_params=True and is forwarded verbatim"""
        config = BasetenConfig()

        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "max_tokens": 100},
            optional_params={},
            model="deepseek-ai/DeepSeek-V4-Flash-0731",
            drop_params=True,
        )

        assert result["reasoning_effort"] == "high"
        assert result["max_tokens"] == 100

    def test_reasoning_effort_none_passthrough(self):
        """reasoning_effort="none" (disable thinking) is forwarded verbatim"""
        config = BasetenConfig()

        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="moonshotai/Kimi-K3",
            drop_params=True,
        )

        assert result["reasoning_effort"] == "none"


if __name__ == "__main__":
    pytest.main([__file__])
