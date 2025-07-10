"""
Tests for AWS Bedrock GovCloud model support
"""

import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"  # Load from local file

import pytest
from unittest.mock import Mock, patch

# Import modules that need to be reloaded
import importlib
import litellm.litellm_core_utils.get_model_cost_map
import litellm

# Reload modules to pick up environment variable
importlib.reload(litellm.litellm_core_utils.get_model_cost_map)
importlib.reload(litellm)

from litellm import completion
from litellm.llms.bedrock.common_utils import BedrockModelInfo, AmazonBedrockGlobalConfig


class TestBedrockGovCloudSupport:
    """Test suite for GovCloud model support in Bedrock"""

    def test_govcloud_regions_in_config(self):
        """Test that GovCloud regions are included in the configuration"""
        config = AmazonBedrockGlobalConfig()
        us_regions = config.get_us_regions()
        
        assert "us-gov-east-1" in us_regions
        assert "us-gov-west-1" in us_regions
        
        all_regions = config.get_all_regions()
        assert "us-gov-east-1" in all_regions
        assert "us-gov-west-1" in all_regions

    def test_govcloud_models_in_model_cost(self):
        """Test that GovCloud models are present in model cost configuration"""
        from litellm import model_cost
        
        # Test Claude models in GovCloud
        assert "bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0" in model_cost
        assert "bedrock/us-gov-west-1/anthropic.claude-3-5-sonnet-20240620-v1:0" in model_cost
        assert "bedrock/us-gov-east-1/anthropic.claude-3-haiku-20240307-v1:0" in model_cost
        assert "bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0" in model_cost
        
        # Test Llama models in GovCloud
        assert "bedrock/us-gov-east-1/meta.llama3-8b-instruct-v1:0" in model_cost
        assert "bedrock/us-gov-west-1/meta.llama3-8b-instruct-v1:0" in model_cost
        assert "bedrock/us-gov-east-1/meta.llama3-70b-instruct-v1:0" in model_cost
        assert "bedrock/us-gov-west-1/meta.llama3-70b-instruct-v1:0" in model_cost
        
        # Test Titan models in GovCloud
        assert "bedrock/us-gov-east-1/amazon.titan-text-lite-v1" in model_cost
        assert "bedrock/us-gov-west-1/amazon.titan-text-lite-v1" in model_cost

    def test_govcloud_model_routing(self):
        """Test that GovCloud models are routed correctly"""
        # Test Claude model routing
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert route == "converse"
        
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0")
        assert route == "converse"
        
        # Test Llama model routing
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-east-1/meta.llama3-8b-instruct-v1:0")
        assert route == "converse"
        
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-west-1/meta.llama3-70b-instruct-v1:0")
        assert route == "converse"
        
        # Test Titan model routing (should use invoke)
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-east-1/amazon.titan-text-lite-v1")
        assert route == "invoke"

    def test_base_model_extraction(self):
        """Test that base model names are correctly extracted from GovCloud models"""
        # Test GovCloud model extraction
        base_model = BedrockModelInfo.get_base_model("bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert base_model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
        
        base_model = BedrockModelInfo.get_base_model("bedrock/us-gov-west-1/meta.llama3-8b-instruct-v1:0")
        assert base_model == "meta.llama3-8b-instruct-v1:0"

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_client')
    def test_govcloud_client_initialization(self, mock_init_client):
        """Test that Bedrock client can be initialized with GovCloud regions"""
        mock_client = Mock()
        mock_init_client.return_value = mock_client
        
        # Test that init_bedrock_client accepts GovCloud regions
        from litellm.llms.bedrock.common_utils import init_bedrock_client
        
        # This should not raise an error
        client = init_bedrock_client(
            region_name="us-gov-east-1",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_region_name="us-gov-east-1",
            aws_bedrock_runtime_endpoint=None,
            aws_session_name=None,
            aws_profile_name=None,
            aws_role_name=None,
            aws_web_identity_token=None,
            extra_headers=None,
            timeout=None,
        )
        
        assert mock_init_client.called

    def test_govcloud_model_in_bedrock_models_list(self):
        """Test that GovCloud models are NOT included in bedrock_models list (they are pricing-only)"""
        # Regional models including GovCloud should be excluded from bedrock_models list
        # They are only in model_cost for pricing purposes
        assert not any("us-gov-east-1" in model for model in litellm.bedrock_models)
        assert not any("us-gov-west-1" in model for model in litellm.bedrock_models)

    def test_govcloud_model_cost_properties(self):
        """Test that GovCloud models have proper cost configuration"""
        from litellm import model_cost
        
        # Check a specific GovCloud model has all required properties
        govcloud_model = model_cost["bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0"]
        
        assert "max_tokens" in govcloud_model
        assert "max_input_tokens" in govcloud_model
        assert "max_output_tokens" in govcloud_model
        assert "input_cost_per_token" in govcloud_model
        assert "output_cost_per_token" in govcloud_model
        assert govcloud_model["litellm_provider"] == "bedrock"
        assert govcloud_model["mode"] == "chat"

    @pytest.mark.parametrize("model_name", [
        "bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/us-gov-east-1/meta.llama3-8b-instruct-v1:0",
        "bedrock/us-gov-west-1/meta.llama3-70b-instruct-v1:0",
    ])
    def test_govcloud_converse_models(self, model_name):
        """Test that GovCloud Claude and Llama models support Converse API"""
        route = BedrockModelInfo.get_bedrock_route(model_name)
        assert route == "converse"

    @pytest.mark.parametrize("model_name", [
        "bedrock/us-gov-east-1/amazon.titan-text-lite-v1",
        "bedrock/us-gov-west-1/amazon.titan-text-express-v1",
        "bedrock/us-gov-east-1/amazon.titan-text-premier-v1:0",
    ])
    def test_govcloud_invoke_models(self, model_name):
        """Test that GovCloud Titan models use Invoke API"""
        route = BedrockModelInfo.get_bedrock_route(model_name)
        assert route == "invoke"