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
        assert config.get_api_base_for_model("abcd1234") == "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
        
        # Model API (non-8-character)
        assert config.get_api_base_for_model("openai/gpt-oss-120b") == "https://inference.baseten.co/v1"


class TestBasetenModelAPI:
    """Test Baseten Model API inference"""

    @patch.dict(os.environ, {"BASETEN_API_KEY": "test-key"})
    def test_model_api_inference(self):
        """Test Model API inference with basic parameters"""
        config = BasetenConfig()
        
        # Test parameter mapping
        non_default_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="openai/gpt-oss-120b",
            drop_params=False
        )
        
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        
        # Test provider info
        api_base, api_key = config._get_openai_compatible_provider_info(None, "test-key")
        assert api_base == "https://inference.baseten.co/v1"
        assert api_key == "test-key"


class TestBasetenTransformRequest:
    """Test Baseten transform_request with served_model_name"""

    def test_transform_request_with_served_model_name(self):
        """Test that served_model_name overrides the model in request body"""
        config = BasetenConfig()

        result = config.transform_request(
            model="wd1lndkw",
            messages=[{"role": "user", "content": "Hello!"}],
            optional_params={},
            litellm_params={"served_model_name": "baseten-hosted/zai-org/GLM-5"},
            headers={},
        )

        assert result["model"] == "baseten-hosted/zai-org/GLM-5"
        assert result["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_transform_request_without_served_model_name(self):
        """Test that model is used as-is when served_model_name is not set"""
        config = BasetenConfig()

        result = config.transform_request(
            model="wd1lndkw",
            messages=[{"role": "user", "content": "Hello!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["model"] == "wd1lndkw"

    def test_transform_request_model_api_with_served_model_name(self):
        """Test served_model_name also works for Model API models"""
        config = BasetenConfig()

        result = config.transform_request(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Hello!"}],
            optional_params={"max_tokens": 100},
            litellm_params={"served_model_name": "my-custom-name"},
            headers={},
        )

        assert result["model"] == "my-custom-name"
        assert result["max_tokens"] == 100


if __name__ == "__main__":
    pytest.main([__file__])
