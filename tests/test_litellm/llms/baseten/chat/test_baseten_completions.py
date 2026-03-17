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

    def test_model_api_transform_request(self):
        """
        Model API happy path — no served_model_name, model passes through as-is.

        Proxy config:
            model_list:
              - model_name: baseten-model
                litellm_params:
                  model: baseten/openai/gpt-oss-120b
                  api_key: your-baseten-api-key
        """
        config = BasetenConfig()

        result = config.transform_request(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Hello!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["model"] == "openai/gpt-oss-120b"


class TestBasetenTransformRequest:
    """Test Baseten transform_request for dedicated deployments"""

    def test_dedicated_deployment_with_served_model_name(self):
        """
        Customer fix: dedicated deployment ID used for URL routing,
        served_model_name sent in request body.

        Proxy config:
            model_list:
              - model_name: baseten-model
                litellm_params:
                  model: baseten/wd1lndkw
                  served_model_name: baseten-hosted/zai-org/GLM-5
                  api_key: os.environ/BASETEN_API_KEY
        """
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

    def test_dedicated_deployment_without_served_model_name(self):
        """
        Dedicated deployment without served_model_name — deployment ID passes
        through as the model name in the request body. Only works if the
        deployment's served_model_name matches the deployment ID.

        Proxy config:
            model_list:
              - model_name: baseten-model
                litellm_params:
                  model: baseten/wd1lndkw
                  api_key: os.environ/BASETEN_API_KEY
        """
        config = BasetenConfig()

        result = config.transform_request(
            model="wd1lndkw",
            messages=[{"role": "user", "content": "Hello!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["model"] == "wd1lndkw"

    def test_dedicated_deployment_api_base_routing(self):
        """
        Dedicated deployment ID correctly builds the dedicated endpoint URL.
        """
        config = BasetenConfig()

        assert config.get_api_base_for_model("wd1lndkw") == "https://model-wd1lndkw.api.baseten.co/environments/production/sync/v1"


if __name__ == "__main__":
    pytest.main([__file__])
