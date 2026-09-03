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


class TestBasetenGLM53:
    """Test baseten/zai-org/GLM-5.3 pricing, context window, and capability registry"""

    def test_glm_5_3_model_info(self, local_model_cost_map):
        """Verify get_model_info resolves cleanly with correct pricing and context window limits"""
        import litellm

        info = litellm.get_model_info("zai-org/GLM-5.3", custom_llm_provider="baseten")
        assert info["key"] == "baseten/zai-org/GLM-5.3"
        assert info["litellm_provider"] == "baseten"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] == 1.4e-06
        assert info["cache_read_input_token_cost"] == 1.4e-07
        assert info["output_cost_per_token"] == 4.4e-06
        assert info["max_input_tokens"] == 1048576
        assert info["max_output_tokens"] == 262144
        assert info["max_tokens"] == 262144
        assert info["supports_function_calling"] is True
        assert info["supports_parallel_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_response_schema"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is False

    def test_glm_5_3_cost_per_token(self, local_model_cost_map):
        """Verify cost_per_token returns exact expected cost"""
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            "baseten/zai-org/GLM-5.3", prompt_tokens=1000, completion_tokens=500
        )
        assert prompt_cost == pytest.approx(0.0014)
        assert completion_cost == pytest.approx(0.0022)
        assert prompt_cost + completion_cost == pytest.approx(0.0036)

    def test_glm_5_3_prompt_caching_cost(self, local_model_cost_map):
        """Verify prompt caching cost calculation for cached tokens"""
        import litellm
        from litellm.types.utils import PromptTokensDetailsWrapper, Usage

        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400),
        )
        prompt_cost, completion_cost = litellm.cost_per_token(
            "baseten/zai-org/GLM-5.3", usage_object=usage
        )
        # 600 un-cached * 1.4e-06 + 400 cached * 1.4e-07
        expected_prompt_cost = (600 * 1.4e-06) + (400 * 1.4e-07)
        expected_completion_cost = 500 * 4.4e-06
        assert prompt_cost == pytest.approx(expected_prompt_cost)
        assert completion_cost == pytest.approx(expected_completion_cost)


if __name__ == "__main__":
    pytest.main([__file__])

