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
        assert "bedrock/us-gov-east-1/claude-sonnet-4-5-20250929-v1:0" in model_cost
        assert "bedrock/us-gov-west-1/claude-sonnet-4-5-20250929-v1:0" in model_cost
        
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

    def test_govcloud_model_pricing_verification(self):
        """Test that GovCloud models have correct pricing that differs from base models"""
        from litellm import model_cost
        
        # Test Claude 3.5 Sonnet pricing
        base_model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        gov_east_model = "bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0"
        gov_west_model = "bedrock/us-gov-west-1/anthropic.claude-3-5-sonnet-20240620-v1:0"
        
        # Verify base model pricing
        base_pricing = model_cost[base_model]
        assert base_pricing["input_cost_per_token"] == 3e-06  # 0.000003
        assert base_pricing["output_cost_per_token"] == 1.5e-05  # 0.000015
        
        # Verify GovCloud models have different (higher) pricing
        gov_east_pricing = model_cost[gov_east_model]
        gov_west_pricing = model_cost[gov_west_model]
        
        # GovCloud models should have 20% higher pricing than base models
        assert gov_east_pricing["input_cost_per_token"] == 3.6e-06  # 0.0000036 (20% higher)
        assert gov_east_pricing["output_cost_per_token"] == 1.8e-05  # 0.000018 (20% higher)
        assert gov_west_pricing["input_cost_per_token"] == 3.6e-06  # 0.0000036 (20% higher)
        assert gov_west_pricing["output_cost_per_token"] == 1.8e-05  # 0.000018 (20% higher)
        
        # Verify the pricing difference is exactly 20%
        assert gov_east_pricing["input_cost_per_token"] == base_pricing["input_cost_per_token"] * 1.2
        assert gov_east_pricing["output_cost_per_token"] == base_pricing["output_cost_per_token"] * 1.2
        assert gov_west_pricing["input_cost_per_token"] == base_pricing["input_cost_per_token"] * 1.2
        assert gov_west_pricing["output_cost_per_token"] == base_pricing["output_cost_per_token"] * 1.2
        
        # Test Claude 3 Haiku pricing
        base_haiku_model = "anthropic.claude-3-haiku-20240307-v1:0"
        gov_east_haiku_model = "bedrock/us-gov-east-1/anthropic.claude-3-haiku-20240307-v1:0"
        gov_west_haiku_model = "bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0"
        
        # Verify base Haiku model pricing
        base_haiku_pricing = model_cost[base_haiku_model]
        assert base_haiku_pricing["input_cost_per_token"] == 2.5e-07  # 0.00000025
        assert base_haiku_pricing["output_cost_per_token"] == 1.25e-06  # 0.00000125
        
        # Verify GovCloud Haiku models have different (higher) pricing
        gov_east_haiku_pricing = model_cost[gov_east_haiku_model]
        gov_west_haiku_pricing = model_cost[gov_west_haiku_model]
        
        # GovCloud Haiku models should have 20% higher pricing than base models
        assert gov_east_haiku_pricing["input_cost_per_token"] == 3e-07  # 0.0000003 (20% higher)
        assert gov_east_haiku_pricing["output_cost_per_token"] == 1.5e-06  # 0.0000015 (20% higher)
        assert gov_west_haiku_pricing["input_cost_per_token"] == 3e-07  # 0.0000003 (20% higher)
        assert gov_west_haiku_pricing["output_cost_per_token"] == 1.5e-06  # 0.0000015 (20% higher)
        
        # Verify the pricing difference is exactly 20%
        assert gov_east_haiku_pricing["input_cost_per_token"] == base_haiku_pricing["input_cost_per_token"] * 1.2
        assert gov_east_haiku_pricing["output_cost_per_token"] == base_haiku_pricing["output_cost_per_token"] * 1.2
        assert gov_west_haiku_pricing["input_cost_per_token"] == base_haiku_pricing["input_cost_per_token"] * 1.2
        assert gov_west_haiku_pricing["output_cost_per_token"] == base_haiku_pricing["output_cost_per_token"] * 1.2

    @patch('litellm.completion')
    def test_govcloud_completion_cost_calculation(self, mock_completion):
        """Test that completion requests use correct pricing for GovCloud models"""
        from litellm import completion_cost, Choices, Message, ModelResponse
        from litellm.utils import Usage
        
        # Mock completion response for base model
        base_model_response = ModelResponse(
            id="test-base",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="Hello", role="assistant"))],
            created=1234567890,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        base_model_response._hidden_params = {"custom_llm_provider": "bedrock", "region_name": "us-east-1"}
        
        # Mock completion response for gov model
        gov_model_response = ModelResponse(
            id="test-gov",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="Hello", role="assistant"))],
            created=1234567890,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Same base model name
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        gov_model_response._hidden_params = {"custom_llm_provider": "bedrock", "region_name": "us-gov-east-1"}
        
        # Mock completion response for gov-west model
        gov_west_model_response = ModelResponse(
            id="test-gov-west",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="Hello", role="assistant"))],
            created=1234567890,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Same base model name
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        gov_west_model_response._hidden_params = {"custom_llm_provider": "bedrock", "region_name": "us-gov-west-1"}
        
        # Test messages
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        
        # Calculate costs using the standard Bedrock format with region parameter
        base_cost = completion_cost(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            completion_response=base_model_response,
            messages=messages,
            region_name="us-east-1",  # Standard region
        )
        
        gov_east_cost = completion_cost(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            completion_response=gov_model_response,
            messages=messages,
            region_name="us-gov-east-1",  # Gov region
        )
        
        gov_west_cost = completion_cost(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            completion_response=gov_west_model_response,
            messages=messages,
            region_name="us-gov-west-1",  # Gov region
        )
        
        # Expected costs based on pricing:
        # Base model: 10 * 3e-06 + 5 * 1.5e-05 = 0.00003 + 0.000075 = 0.000105
        # Gov models: 10 * 3.6e-06 + 5 * 1.8e-05 = 0.000036 + 0.00009 = 0.000126
        expected_base_cost = 10 * 3e-06 + 5 * 1.5e-05  # 0.000105
        expected_gov_cost = 10 * 3.6e-06 + 5 * 1.8e-05  # 0.000126
        
        # Verify costs are calculated correctly
        assert abs(base_cost - expected_base_cost) < 1e-10, f"Base cost mismatch: got {base_cost}, expected {expected_base_cost}"
        assert abs(gov_east_cost - expected_gov_cost) < 1e-10, f"Gov East cost mismatch: got {gov_east_cost}, expected {expected_gov_cost}"
        assert abs(gov_west_cost - expected_gov_cost) < 1e-10, f"Gov West cost mismatch: got {gov_west_cost}, expected {expected_gov_cost}"
        
        # Verify GovCloud costs are exactly 20% higher than base cost
        assert abs(gov_east_cost - base_cost * 1.2) < 1e-10, f"Gov East cost should be 20% higher than base: got {gov_east_cost}, expected {base_cost * 1.2}"
        assert abs(gov_west_cost - base_cost * 1.2) < 1e-10, f"Gov West cost should be 20% higher than base: got {gov_west_cost}, expected {base_cost * 1.2}"
        
        # Test with different token counts
        large_response = ModelResponse(
            id="test-large",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="A longer response", role="assistant"))],
            created=1234567890,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        large_response._hidden_params = {"custom_llm_provider": "bedrock", "region_name": "us-east-1"}
        
        large_base_cost = completion_cost(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            completion_response=large_response,
            messages=messages,
            region_name="us-east-1",
        )
        
        # Create large response for gov model
        large_gov_response = ModelResponse(
            id="test-large-gov",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="A longer response", role="assistant"))],
            created=1234567890,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        large_gov_response._hidden_params = {"custom_llm_provider": "bedrock", "region_name": "us-gov-east-1"}
        
        large_gov_cost = completion_cost(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            completion_response=large_gov_response,
            messages=messages,
            region_name="us-gov-east-1",
        )
        
        # Expected costs for larger response:
        # Base model: 100 * 3e-06 + 50 * 1.5e-05 = 0.0003 + 0.00075 = 0.00105
        # Gov model: 100 * 3.6e-06 + 50 * 1.8e-05 = 0.00036 + 0.0009 = 0.00126
        expected_large_base_cost = 100 * 3e-06 + 50 * 1.5e-05  # 0.00105
        expected_large_gov_cost = 100 * 3.6e-06 + 50 * 1.8e-05  # 0.00126
        
        assert abs(large_base_cost - expected_large_base_cost) < 1e-10, f"Large base cost mismatch: got {large_base_cost}, expected {expected_large_base_cost}"
        assert abs(large_gov_cost - expected_large_gov_cost) < 1e-10, f"Large gov cost mismatch: got {large_gov_cost}, expected {expected_large_gov_cost}"
        assert abs(large_gov_cost - large_base_cost * 1.2) < 1e-10, f"Large gov cost should be 20% higher than base: got {large_gov_cost}, expected {large_base_cost * 1.2}"

    @patch('litellm.llms.custom_httpx.http_handler.HTTPHandler.post')
    def test_govcloud_completion_with_cost_tracking(self, mock_post):
        """Test that completion requests with cost tracking use correct pricing for GovCloud models"""
        from litellm import completion
        from unittest.mock import Mock
        import json
        
        # Mock the HTTP client's post method to return responses
        def mock_post_side_effect(url, headers=None, data=None, **kwargs):
            # Extract region from the URL to determine which response to return
            region = "us-east-1"  # default
            if "us-gov-east-1" in url:
                region = "us-gov-east-1"
            elif "us-gov-west-1" in url:
                region = "us-gov-west-1"
            
            # Create mock response based on region
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            
            # Create a realistic Bedrock converse response structure
            bedrock_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Hello from {region}"
                            }
                        ]
                    }
                },
                "usage": {
                    "inputTokens": 15,
                    "outputTokens": 8,
                    "totalTokens": 23
                },
                "stopReason": "end_turn"
            }
            
            mock_response.json.return_value = bedrock_response
            mock_response.text = json.dumps(bedrock_response)
            mock_response.raise_for_status = Mock()  # Don't raise exceptions
            
            return mock_response
        
        mock_post.side_effect = mock_post_side_effect
        
        # Test base model completion
        base_result = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            aws_region_name="us-east-1"
        )
        
        # Test gov-east model completion
        gov_east_result = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            aws_region_name="us-gov-east-1"
        )
        
        # Test gov-west model completion
        gov_west_result = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            aws_region_name="us-gov-west-1"
        )
        
        # Verify the mock was called correctly
        assert mock_post.call_count == 3
        
        # Verify usage information is present
        from litellm.types.utils import ModelResponse
        assert isinstance(base_result, ModelResponse)
        assert isinstance(gov_east_result, ModelResponse)
        assert isinstance(gov_west_result, ModelResponse)
        
        base_result_typed: ModelResponse = base_result
        gov_east_result_typed: ModelResponse = gov_east_result
        gov_west_result_typed: ModelResponse = gov_west_result
        
        # Verify usage information is present
        assert hasattr(base_result_typed, 'usage') and base_result_typed.usage.prompt_tokens == 15
        assert hasattr(base_result_typed, 'usage') and base_result_typed.usage.completion_tokens == 8
        assert hasattr(gov_east_result_typed, 'usage') and gov_east_result_typed.usage.prompt_tokens == 15
        assert hasattr(gov_east_result_typed, 'usage') and gov_east_result_typed.usage.completion_tokens == 8
        assert hasattr(gov_west_result_typed, 'usage') and gov_west_result_typed.usage.prompt_tokens == 15
        assert hasattr(gov_west_result_typed, 'usage') and gov_west_result_typed.usage.completion_tokens == 8
        
        # Verify cost calculation uses correct pricing for each region
        # Get costs directly from the completion response _hidden_params
        base_cost = base_result_typed._hidden_params.get("response_cost", 0.0)
        gov_east_cost = gov_east_result_typed._hidden_params.get("response_cost", 0.0)
        gov_west_cost = gov_west_result_typed._hidden_params.get("response_cost", 0.0)

        print(f"Base cost: {base_cost}")
        print(f"Gov East cost: {gov_east_cost}")
        print(f"Gov West cost: {gov_west_cost}")
        
        # Expected costs based on pricing:
        # Base model: 15 * 3e-06 + 8 * 1.5e-05 = 0.000045 + 0.00012 = 0.000165
        # Gov models: 15 * 3.6e-06 + 8 * 1.8e-05 = 0.000054 + 0.000144 = 0.000198
        expected_base_cost = 15 * 3e-06 + 8 * 1.5e-05  # 0.000165
        expected_gov_cost = 15 * 3.6e-06 + 8 * 1.8e-05  # 0.000198
        
        # Verify costs are calculated correctly
        assert abs(base_cost - expected_base_cost) < 1e-10, f"Base cost mismatch: got {base_cost}, expected {expected_base_cost}"
        assert abs(gov_east_cost - expected_gov_cost) < 1e-10, f"Gov East cost mismatch: got {gov_east_cost}, expected {expected_gov_cost}"
        assert abs(gov_west_cost - expected_gov_cost) < 1e-10, f"Gov West cost mismatch: got {gov_west_cost}, expected {expected_gov_cost}"
        
        # Verify GovCloud costs are exactly 20% higher than base cost
        assert abs(gov_east_cost - base_cost * 1.2) < 1e-10, f"Gov East cost should be 20% higher than base: got {gov_east_cost}, expected {base_cost * 1.2}"
        assert abs(gov_west_cost - base_cost * 1.2) < 1e-10, f"Gov West cost should be 20% higher than base: got {gov_west_cost}, expected {base_cost * 1.2}"
        
        # Print cost information for verification
        print(f"Base model cost: ${base_cost:.6f}")
        print(f"GovCloud East cost: ${gov_east_cost:.6f}")
        print(f"GovCloud West cost: ${gov_west_cost:.6f}")
        print(f"GovCloud cost increase: {((gov_east_cost / base_cost) - 1) * 100:.1f}%")

    def test_govcloud_cost_per_token_with_region(self):
        """Test that cost_per_token function correctly uses region-based pricing for GovCloud models"""
        from litellm import cost_per_token
        from litellm.utils import Usage
        
        # Test usage object
        usage = Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        
        # Test base model with standard region
        base_prompt_cost, base_completion_cost = cost_per_token(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt_tokens=20,
            completion_tokens=10,
            custom_llm_provider="bedrock",
            region_name="us-east-1",
        )
        
        # Test gov models with gov regions
        gov_east_prompt_cost, gov_east_completion_cost = cost_per_token(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt_tokens=20,
            completion_tokens=10,
            custom_llm_provider="bedrock",
            region_name="us-gov-east-1",
        )
        
        gov_west_prompt_cost, gov_west_completion_cost = cost_per_token(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt_tokens=20,
            completion_tokens=10,
            custom_llm_provider="bedrock",
            region_name="us-gov-west-1",
        )
        
        # Expected costs:
        # Base model: 20 * 3e-06 + 10 * 1.5e-05 = 0.00006 + 0.00015 = 0.00021
        # Gov models: 20 * 3.6e-06 + 10 * 1.8e-05 = 0.000072 + 0.00018 = 0.000252
        expected_base_prompt_cost = 20 * 3e-06  # 0.00006
        expected_base_completion_cost = 10 * 1.5e-05  # 0.00015
        expected_gov_prompt_cost = 20 * 3.6e-06  # 0.000072
        expected_gov_completion_cost = 10 * 1.8e-05  # 0.00018
        
        # Verify costs are calculated correctly
        assert abs(base_prompt_cost - expected_base_prompt_cost) < 1e-10, f"Base prompt cost mismatch: got {base_prompt_cost}, expected {expected_base_prompt_cost}"
        assert abs(base_completion_cost - expected_base_completion_cost) < 1e-10, f"Base completion cost mismatch: got {base_completion_cost}, expected {expected_base_completion_cost}"
        
        assert abs(gov_east_prompt_cost - expected_gov_prompt_cost) < 1e-10, f"Gov East prompt cost mismatch: got {gov_east_prompt_cost}, expected {expected_gov_prompt_cost}"
        assert abs(gov_east_completion_cost - expected_gov_completion_cost) < 1e-10, f"Gov East completion cost mismatch: got {gov_east_completion_cost}, expected {expected_gov_completion_cost}"
        
        assert abs(gov_west_prompt_cost - expected_gov_prompt_cost) < 1e-10, f"Gov West prompt cost mismatch: got {gov_west_prompt_cost}, expected {expected_gov_prompt_cost}"
        assert abs(gov_west_completion_cost - expected_gov_completion_cost) < 1e-10, f"Gov West completion cost mismatch: got {gov_west_completion_cost}, expected {expected_gov_completion_cost}"
        
        # Verify GovCloud costs are exactly 20% higher than base costs
        assert abs(gov_east_prompt_cost - base_prompt_cost * 1.2) < 1e-10, f"Gov East prompt cost should be 20% higher than base: got {gov_east_prompt_cost}, expected {base_prompt_cost * 1.2}"
        assert abs(gov_east_completion_cost - base_completion_cost * 1.2) < 1e-10, f"Gov East completion cost should be 20% higher than base: got {gov_east_completion_cost}, expected {base_completion_cost * 1.2}"
        assert abs(gov_west_prompt_cost - base_prompt_cost * 1.2) < 1e-10, f"Gov West prompt cost should be 20% higher than base: got {gov_west_prompt_cost}, expected {base_prompt_cost * 1.2}"
        assert abs(gov_west_completion_cost - base_completion_cost * 1.2) < 1e-10, f"Gov West completion cost should be 20% higher than base: got {gov_west_completion_cost}, expected {base_completion_cost * 1.2}"
        
        # Test total costs
        base_total_cost = base_prompt_cost + base_completion_cost
        gov_east_total_cost = gov_east_prompt_cost + gov_east_completion_cost
        gov_west_total_cost = gov_west_prompt_cost + gov_west_completion_cost
        
        expected_base_total = expected_base_prompt_cost + expected_base_completion_cost  # 0.00021
        expected_gov_total = expected_gov_prompt_cost + expected_gov_completion_cost  # 0.000252
        
        assert abs(base_total_cost - expected_base_total) < 1e-10, f"Base total cost mismatch: got {base_total_cost}, expected {expected_base_total}"
        assert abs(gov_east_total_cost - expected_gov_total) < 1e-10, f"Gov East total cost mismatch: got {gov_east_total_cost}, expected {expected_gov_total}"
        assert abs(gov_west_total_cost - expected_gov_total) < 1e-10, f"Gov West total cost mismatch: got {gov_west_total_cost}, expected {expected_gov_total}"
        assert abs(gov_east_total_cost - base_total_cost * 1.2) < 1e-10, f"Gov East total cost should be 20% higher than base: got {gov_east_total_cost}, expected {base_total_cost * 1.2}"
        assert abs(gov_west_total_cost - base_total_cost * 1.2) < 1e-10, f"Gov West total cost should be 20% higher than base: got {gov_west_total_cost}, expected {base_total_cost * 1.2}"

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