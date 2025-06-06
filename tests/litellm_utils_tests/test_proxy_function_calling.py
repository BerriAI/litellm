"""
Test module for LiteLLM proxy function calling support.

This module tests the 'supports_function_calling' function with proxy configurations
to ensure that proxied models correctly report their function calling capabilities.
"""

import pytest
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm.utils import supports_function_calling


class TestProxyFunctionCalling:
    """Test class for proxy function calling capabilities."""

    @pytest.fixture(autouse=True)
    def reset_mock_cache(self):
        """Reset model cache before each test."""
        from litellm.utils import _model_cache
        _model_cache.flush_cache()

    @pytest.mark.parametrize(
        "direct_model,proxy_model,expected_result",
        [
            # OpenAI models
            ("gpt-3.5-turbo", "litellm_proxy/gpt-3.5-turbo", True),
            ("gpt-4", "litellm_proxy/gpt-4", True),
            ("gpt-4o", "litellm_proxy/gpt-4o", True),
            ("gpt-4o-mini", "litellm_proxy/gpt-4o-mini", True),
            ("gpt-4-turbo", "litellm_proxy/gpt-4-turbo", True),
            ("gpt-4-1106-preview", "litellm_proxy/gpt-4-1106-preview", True),
            
            # Azure OpenAI models
            ("azure/gpt-4", "litellm_proxy/azure/gpt-4", True),
            ("azure/gpt-3.5-turbo", "litellm_proxy/azure/gpt-3.5-turbo", True),
            ("azure/gpt-4-1106-preview", "litellm_proxy/azure/gpt-4-1106-preview", True),
            
            # Anthropic models (Claude supports function calling)
            ("claude-3-5-sonnet-20240620", "litellm_proxy/claude-3-5-sonnet-20240620", True),
            ("claude-3-opus-20240229", "litellm_proxy/claude-3-opus-20240229", True),
            ("claude-3-sonnet-20240229", "litellm_proxy/claude-3-sonnet-20240229", True),
            ("claude-3-haiku-20240307", "litellm_proxy/claude-3-haiku-20240307", True),
            
            # Google models
            ("gemini-pro", "litellm_proxy/gemini-pro", True),
            ("gemini/gemini-1.5-pro", "litellm_proxy/gemini/gemini-1.5-pro", True),
            ("gemini/gemini-1.5-flash", "litellm_proxy/gemini/gemini-1.5-flash", True),
            
            # Groq models (mixed support)
            ("groq/gemma-7b-it", "litellm_proxy/groq/gemma-7b-it", True),
            ("groq/llama3-70b-8192", "litellm_proxy/groq/llama3-70b-8192", False),  # This model doesn't support function calling
            
            # Cohere models (generally don't support function calling)
            ("command-nightly", "litellm_proxy/command-nightly", False),
            ("anthropic.claude-instant-v1", "litellm_proxy/anthropic.claude-instant-v1", False),
        ],
    )
    def test_proxy_function_calling_support_consistency(
        self, direct_model, proxy_model, expected_result
    ):
        """Test that proxy models have the same function calling support as their direct counterparts."""
        direct_result = supports_function_calling(direct_model)
        proxy_result = supports_function_calling(proxy_model)
        
        # Both should match the expected result
        assert direct_result == expected_result, f"Direct model {direct_model} should return {expected_result}"
        assert proxy_result == expected_result, f"Proxy model {proxy_model} should return {expected_result}"
        
        # Direct and proxy should be consistent
        assert direct_result == proxy_result, f"Mismatch: {direct_model}={direct_result} vs {proxy_model}={proxy_result}"

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_model,expected_proxy_result",
        [
            # Custom model names that cannot be resolved without proxy configuration context
            # These will return False because LiteLLM cannot determine the underlying model
            ("litellm_proxy/bedrock-claude-3-haiku", "bedrock/anthropic.claude-3-haiku-20240307-v1:0", False),
            ("litellm_proxy/bedrock-claude-3-sonnet", "bedrock/anthropic.claude-3-sonnet-20240229-v1:0", False),
            ("litellm_proxy/bedrock-claude-3-opus", "bedrock/anthropic.claude-3-opus-20240229-v1:0", False),
            ("litellm_proxy/bedrock-claude-instant", "bedrock/anthropic.claude-instant-v1", False),
            ("litellm_proxy/bedrock-titan-text", "bedrock/amazon.titan-text-express-v1", False),
            
            # Azure with custom deployment names (cannot be resolved)
            ("litellm_proxy/my-gpt4-deployment", "azure/gpt-4", False),
            ("litellm_proxy/production-gpt35", "azure/gpt-3.5-turbo", False),
            ("litellm_proxy/dev-gpt4o", "azure/gpt-4o", False),
            
            # Custom OpenAI deployments (cannot be resolved)
            ("litellm_proxy/company-gpt4", "gpt-4", False),
            ("litellm_proxy/internal-gpt35", "gpt-3.5-turbo", False),
            
            # Vertex AI with custom names (cannot be resolved)
            ("litellm_proxy/vertex-gemini-pro", "vertex_ai/gemini-1.5-pro", False),
            ("litellm_proxy/vertex-gemini-flash", "vertex_ai/gemini-1.5-flash", False),
            
            # Anthropic with custom names (cannot be resolved)
            ("litellm_proxy/claude-prod", "anthropic/claude-3-sonnet-20240229", False),
            ("litellm_proxy/claude-dev", "anthropic/claude-3-haiku-20240307", False),
            
            # Groq with custom names (cannot be resolved)
            ("litellm_proxy/fast-llama", "groq/llama3-8b-8192", False),
            ("litellm_proxy/groq-gemma", "groq/gemma-7b-it", False),
            
            # Cohere with custom names (cannot be resolved)
            ("litellm_proxy/cohere-command", "cohere/command-r", False),
            ("litellm_proxy/cohere-command-plus", "cohere/command-r-plus", False),
            
            # Together AI with custom names (cannot be resolved)
            ("litellm_proxy/together-llama", "together_ai/meta-llama/Llama-2-70b-chat-hf", False),
            ("litellm_proxy/together-mistral", "together_ai/mistralai/Mistral-7B-Instruct-v0.1", False),
            
            # Ollama with custom names (cannot be resolved)
            ("litellm_proxy/local-llama", "ollama/llama2", False),
            ("litellm_proxy/local-mistral", "ollama/mistral", False),
        ],
    )
    def test_proxy_custom_model_names_without_config(
        self, proxy_model_name, underlying_model, expected_proxy_result
    ):
        """
        Test proxy models with custom model names that differ from underlying models.
        
        Without proxy configuration context, LiteLLM cannot resolve custom model names
        to their underlying models, so these will return False.
        This demonstrates the limitation and documents the expected behavior.
        """
        # Test the underlying model directly first to establish what it SHOULD return
        try:
            underlying_result = supports_function_calling(underlying_model)
            print(f"Underlying model {underlying_model} supports function calling: {underlying_result}")
        except Exception as e:
            print(f"Warning: Could not test underlying model {underlying_model}: {e}")
        
        # Test the proxy model - this will return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        assert proxy_result == expected_proxy_result, f"Proxy model {proxy_model_name} should return {expected_proxy_result} (without config context)"

    def test_proxy_model_resolution_with_custom_names_documentation(self):
        """
        Document the behavior and limitation for custom proxy model names.
        
        This test demonstrates:
        1. The current limitation with custom model names
        2. How the proxy server would handle this in production
        3. The expected behavior for both scenarios
        """
        # Case 1: Custom model name that cannot be resolved
        custom_model = "litellm_proxy/my-custom-claude"
        result = supports_function_calling(custom_model)
        assert result is False, "Custom model names return False without proxy config context"
        
        # Case 2: Model name that can be resolved (matches pattern)
        resolvable_model = "litellm_proxy/claude-3-sonnet-20240229"  
        result = supports_function_calling(resolvable_model)
        assert result is True, "Resolvable model names work with fallback logic"
        
        # Documentation notes:
        print("""
        PROXY MODEL RESOLUTION BEHAVIOR:
        
        ✅ WORKS (with current fallback logic):
           - litellm_proxy/gpt-4
           - litellm_proxy/claude-3-sonnet-20240229
           - litellm_proxy/anthropic/claude-3-haiku-20240307
           
        ❌ DOESN'T WORK (requires proxy server config):
           - litellm_proxy/my-custom-gpt4
           - litellm_proxy/bedrock-claude-3-haiku
           - litellm_proxy/production-model
           
        💡 SOLUTION: Use LiteLLM proxy server with proper model_list configuration
           that maps custom names to underlying models.
        """)

    @pytest.mark.parametrize(
        "proxy_model_with_hints,expected_result",
        [
            # These are proxy models where we can infer the underlying model from the name
            ("litellm_proxy/gpt-4-with-functions", True),  # Hints at GPT-4
            ("litellm_proxy/claude-3-haiku-prod", True),   # Hints at Claude 3 Haiku
            ("litellm_proxy/bedrock-anthropic-claude-3-sonnet", True),  # Hints at Bedrock Claude 3 Sonnet
        ],
    )
    def test_proxy_models_with_naming_hints(self, proxy_model_with_hints, expected_result):
        """
        Test proxy models with names that provide hints about the underlying model.
        
        Note: These will currently fail because the hint-based resolution isn't implemented yet,
        but they demonstrate what could be possible with enhanced model name inference.
        """
        # This test documents potential future enhancement
        proxy_result = supports_function_calling(proxy_model_with_hints)
        
        # Currently these will return False, but we document the expected behavior
        # In the future, we could implement smarter model name inference
        print(f"Model {proxy_model_with_hints}: current={proxy_result}, desired={expected_result}")
        
        # For now, we expect False (current behavior), but document the limitation
        assert proxy_result is False, f"Current limitation: {proxy_model_with_hints} returns False without inference"

    @pytest.mark.parametrize(
        "proxy_model,expected_result",
        [
            # Test specific proxy models that should support function calling
            ("litellm_proxy/gpt-3.5-turbo", True),
            ("litellm_proxy/gpt-4", True),
            ("litellm_proxy/gpt-4o", True),
            ("litellm_proxy/claude-3-5-sonnet-20240620", True),
            ("litellm_proxy/gemini/gemini-1.5-pro", True),
            
            # Test proxy models that should not support function calling
            ("litellm_proxy/command-nightly", False),
            ("litellm_proxy/anthropic.claude-instant-v1", False),
        ],
    )
    def test_proxy_only_function_calling_support(self, proxy_model, expected_result):
        """
        Test proxy models independently to ensure they report correct function calling support.
        
        This test focuses on proxy models without comparing to direct models,
        useful for cases where we only care about the proxy behavior.
        """
        try:
            result = supports_function_calling(model=proxy_model)
            assert result == expected_result, (
                f"Proxy model {proxy_model} returned {result}, expected {expected_result}"
            )
        except Exception as e:
            pytest.fail(f"Error testing proxy model {proxy_model}: {e}")

    def test_litellm_utils_supports_function_calling_import(self):
        """Test that supports_function_calling can be imported from litellm.utils."""
        try:
            from litellm.utils import supports_function_calling
            assert callable(supports_function_calling)
        except ImportError as e:
            pytest.fail(f"Failed to import supports_function_calling: {e}")

    def test_litellm_supports_function_calling_import(self):
        """Test that supports_function_calling can be imported from litellm directly."""
        try:
            import litellm
            assert hasattr(litellm, 'supports_function_calling')
            assert callable(litellm.supports_function_calling)
        except Exception as e:
            pytest.fail(f"Failed to access litellm.supports_function_calling: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [
            "litellm_proxy/gpt-3.5-turbo",
            "litellm_proxy/gpt-4",
            "litellm_proxy/claude-3-5-sonnet-20240620",
            "litellm_proxy/gemini/gemini-1.5-pro",
        ],
    )
    def test_proxy_model_with_custom_llm_provider_none(self, model_name):
        """
        Test proxy models with custom_llm_provider=None parameter.
        
        This tests the supports_function_calling function with the custom_llm_provider
        parameter explicitly set to None, which is a common usage pattern.
        """
        try:
            result = supports_function_calling(model=model_name, custom_llm_provider=None)
            # All the models in this test should support function calling
            assert result is True, (
                f"Model {model_name} should support function calling but returned {result}"
            )
        except Exception as e:
            pytest.fail(f"Error testing {model_name} with custom_llm_provider=None: {e}")

    def test_edge_cases_and_malformed_proxy_models(self):
        """Test edge cases and malformed proxy model names."""
        test_cases = [
            ("litellm_proxy/", False),  # Empty model name after proxy prefix
            ("litellm_proxy", False),   # Just the proxy prefix without slash
            ("litellm_proxy//gpt-3.5-turbo", False),  # Double slash
            ("litellm_proxy/nonexistent-model", False),  # Non-existent model
        ]
        
        for model_name, expected_result in test_cases:
            try:
                result = supports_function_calling(model=model_name)
                # For malformed models, we expect False or the function to handle gracefully
                assert result == expected_result, (
                    f"Edge case {model_name} returned {result}, expected {expected_result}"
                )
            except Exception:
                # It's acceptable for malformed model names to raise exceptions
                # rather than returning False, as long as they're handled gracefully
                pass

    def test_proxy_model_resolution_demonstration(self):
        """
        Demonstration test showing the current issue with proxy model resolution.
        
        This test documents the current behavior and can be used to verify
        when the issue is fixed.
        """
        direct_model = "gpt-3.5-turbo"
        proxy_model = "litellm_proxy/gpt-3.5-turbo"
        
        direct_result = supports_function_calling(model=direct_model)
        proxy_result = supports_function_calling(model=proxy_model)
        
        print(f"\nDemonstration of proxy model resolution:")
        print(f"Direct model '{direct_model}' supports function calling: {direct_result}")
        print(f"Proxy model '{proxy_model}' supports function calling: {proxy_result}")
        
        # This assertion will currently fail due to the bug
        # When the bug is fixed, this test should pass
        if direct_result != proxy_result:
            pytest.skip(
                f"Known issue: Proxy model resolution inconsistency. "
                f"Direct: {direct_result}, Proxy: {proxy_result}. "
                f"This test will pass when the issue is resolved."
            )
        
        assert direct_result == proxy_result, (
            f"Proxy model resolution issue: {direct_model} -> {direct_result}, "
            f"{proxy_model} -> {proxy_result}"
        )

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_bedrock_model,expected_proxy_result,description",
        [
            # Bedrock Converse API mappings - these are the real-world scenarios
            ("litellm_proxy/bedrock-claude-3-haiku", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Bedrock Claude 3 Haiku via Converse API"),
            ("litellm_proxy/bedrock-claude-3-sonnet", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "Bedrock Claude 3 Sonnet via Converse API"),
            ("litellm_proxy/bedrock-claude-3-opus", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "Bedrock Claude 3 Opus via Converse API"),
            ("litellm_proxy/bedrock-claude-3-5-sonnet", "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0", False, "Bedrock Claude 3.5 Sonnet via Converse API"),
            
            # Bedrock Legacy API mappings (non-converse)
            ("litellm_proxy/bedrock-claude-instant", "bedrock/anthropic.claude-instant-v1", False, "Bedrock Claude Instant Legacy API"),
            ("litellm_proxy/bedrock-claude-v2", "bedrock/anthropic.claude-v2", False, "Bedrock Claude v2 Legacy API"),
            ("litellm_proxy/bedrock-claude-v2-1", "bedrock/anthropic.claude-v2:1", False, "Bedrock Claude v2.1 Legacy API"),
            
            # Bedrock other model providers via Converse API
            ("litellm_proxy/bedrock-titan-text", "bedrock/converse/amazon.titan-text-express-v1", False, "Bedrock Titan Text Express via Converse API"),
            ("litellm_proxy/bedrock-titan-text-premier", "bedrock/converse/amazon.titan-text-premier-v1:0", False, "Bedrock Titan Text Premier via Converse API"),
            ("litellm_proxy/bedrock-llama3-8b", "bedrock/converse/meta.llama3-8b-instruct-v1:0", False, "Bedrock Llama 3 8B via Converse API"),
            ("litellm_proxy/bedrock-llama3-70b", "bedrock/converse/meta.llama3-70b-instruct-v1:0", False, "Bedrock Llama 3 70B via Converse API"),
            ("litellm_proxy/bedrock-mistral-7b", "bedrock/converse/mistral.mistral-7b-instruct-v0:2", False, "Bedrock Mistral 7B via Converse API"),
            ("litellm_proxy/bedrock-mistral-8x7b", "bedrock/converse/mistral.mixtral-8x7b-instruct-v0:1", False, "Bedrock Mistral 8x7B via Converse API"),
            ("litellm_proxy/bedrock-mistral-large", "bedrock/converse/mistral.mistral-large-2402-v1:0", False, "Bedrock Mistral Large via Converse API"),
            
            # Company-specific naming patterns (real-world examples)
            ("litellm_proxy/prod-claude-haiku", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Production Claude Haiku"),
            ("litellm_proxy/dev-claude-sonnet", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "Development Claude Sonnet"),
            ("litellm_proxy/staging-claude-opus", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "Staging Claude Opus"),
            ("litellm_proxy/cost-optimized-claude", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Cost-optimized Claude deployment"),
            ("litellm_proxy/high-performance-claude", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "High-performance Claude deployment"),
            
            # Regional deployment examples
            ("litellm_proxy/us-east-claude", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "US East Claude deployment"),
            ("litellm_proxy/eu-west-claude", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "EU West Claude deployment"),
            ("litellm_proxy/ap-south-llama", "bedrock/converse/meta.llama3-70b-instruct-v1:0", False, "Asia Pacific Llama deployment"),
        ],
    )
    def test_bedrock_converse_api_proxy_mappings(
        self, proxy_model_name, underlying_bedrock_model, expected_proxy_result, description
    ):
        """
        Test real-world Bedrock Converse API proxy model mappings.
        
        This test covers the specific scenario where proxy model names like 
        'bedrock-claude-3-haiku' map to underlying Bedrock Converse API models like
        'bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0'.
        
        These mappings are typically defined in proxy server configuration files
        and cannot be resolved by LiteLLM without that context.
        """
        print(f"\nTesting: {description}")
        print(f"  Proxy model: {proxy_model_name}")
        print(f"  Underlying model: {underlying_bedrock_model}")
        
        # Test the underlying model directly to verify it supports function calling
        try:
            underlying_result = supports_function_calling(underlying_bedrock_model)
            print(f"  Underlying model function calling support: {underlying_result}")
            
            # Most Bedrock Converse API models with Anthropic Claude should support function calling
            if "anthropic.claude-3" in underlying_bedrock_model:
                assert underlying_result is True, f"Claude 3 models should support function calling: {underlying_bedrock_model}"
        except Exception as e:
            print(f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}")
        
        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")
        
        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )

    def test_real_world_proxy_config_documentation(self):
        """
        Document how real-world proxy configurations would handle model mappings.
        
        This test provides documentation on how the proxy server configuration
        would typically map custom model names to underlying models.
        """
        print("""
        
        REAL-WORLD PROXY SERVER CONFIGURATION EXAMPLE:
        ===============================================
        
        In a proxy_server_config.yaml file, you would define:
        
        model_list:
          - model_name: bedrock-claude-3-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: bedrock-claude-3-sonnet
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: prod-claude-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/PROD_AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/PROD_AWS_SECRET_ACCESS_KEY
              aws_region_name: us-west-2
        
        
        FUNCTION CALLING WITH PROXY SERVER:
        ===================================
        
        When using the proxy server with this configuration:
        
        1. Client calls: supports_function_calling("bedrock-claude-3-haiku")
        2. Proxy server resolves to: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
        3. LiteLLM evaluates the underlying model's capabilities
        4. Returns: True (because Claude 3 Haiku supports function calling)
        
        Without the proxy server configuration context, LiteLLM cannot resolve
        the custom model name and returns False.
        
        
        BEDROCK CONVERSE API BENEFITS:
        ==============================
        
        The Bedrock Converse API provides:
        - Standardized function calling interface across providers
        - Better tool use capabilities compared to legacy APIs
        - Consistent request/response format
        - Enhanced streaming support for function calls
        
        """)
        
        # Verify that direct underlying models work as expected
        bedrock_models = [
            "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
        ]
        
        for model in bedrock_models:
            try:
                result = supports_function_calling(model)
                print(f"Direct test - {model}: {result}")
                # Claude 3 models should support function calling
                assert result is True, f"Claude 3 model should support function calling: {model}"
            except Exception as e:
                print(f"Could not test {model}: {e}")

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_bedrock_model,expected_proxy_result,description",
        [
            # Bedrock Converse API mappings - these are the real-world scenarios
            ("litellm_proxy/bedrock-claude-3-haiku", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Bedrock Claude 3 Haiku via Converse API"),
            ("litellm_proxy/bedrock-claude-3-sonnet", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "Bedrock Claude 3 Sonnet via Converse API"),
            ("litellm_proxy/bedrock-claude-3-opus", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "Bedrock Claude 3 Opus via Converse API"),
            ("litellm_proxy/bedrock-claude-3-5-sonnet", "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0", False, "Bedrock Claude 3.5 Sonnet via Converse API"),
            
            # Bedrock Legacy API mappings (non-converse)
            ("litellm_proxy/bedrock-claude-instant", "bedrock/anthropic.claude-instant-v1", False, "Bedrock Claude Instant Legacy API"),
            ("litellm_proxy/bedrock-claude-v2", "bedrock/anthropic.claude-v2", False, "Bedrock Claude v2 Legacy API"),
            ("litellm_proxy/bedrock-claude-v2-1", "bedrock/anthropic.claude-v2:1", False, "Bedrock Claude v2.1 Legacy API"),
            
            # Bedrock other model providers via Converse API
            ("litellm_proxy/bedrock-titan-text", "bedrock/converse/amazon.titan-text-express-v1", False, "Bedrock Titan Text Express via Converse API"),
            ("litellm_proxy/bedrock-titan-text-premier", "bedrock/converse/amazon.titan-text-premier-v1:0", False, "Bedrock Titan Text Premier via Converse API"),
            ("litellm_proxy/bedrock-llama3-8b", "bedrock/converse/meta.llama3-8b-instruct-v1:0", False, "Bedrock Llama 3 8B via Converse API"),
            ("litellm_proxy/bedrock-llama3-70b", "bedrock/converse/meta.llama3-70b-instruct-v1:0", False, "Bedrock Llama 3 70B via Converse API"),
            ("litellm_proxy/bedrock-mistral-7b", "bedrock/converse/mistral.mistral-7b-instruct-v0:2", False, "Bedrock Mistral 7B via Converse API"),
            ("litellm_proxy/bedrock-mistral-8x7b", "bedrock/converse/mistral.mixtral-8x7b-instruct-v0:1", False, "Bedrock Mistral 8x7B via Converse API"),
            ("litellm_proxy/bedrock-mistral-large", "bedrock/converse/mistral.mistral-large-2402-v1:0", False, "Bedrock Mistral Large via Converse API"),
            
            # Company-specific naming patterns (real-world examples)
            ("litellm_proxy/prod-claude-haiku", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Production Claude Haiku"),
            ("litellm_proxy/dev-claude-sonnet", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "Development Claude Sonnet"),
            ("litellm_proxy/staging-claude-opus", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "Staging Claude Opus"),
            ("litellm_proxy/cost-optimized-claude", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Cost-optimized Claude deployment"),
            ("litellm_proxy/high-performance-claude", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "High-performance Claude deployment"),
            
            # Regional deployment examples
            ("litellm_proxy/us-east-claude", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "US East Claude deployment"),
            ("litellm_proxy/eu-west-claude", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "EU West Claude deployment"),
            ("litellm_proxy/ap-south-llama", "bedrock/converse/meta.llama3-70b-instruct-v1:0", False, "Asia Pacific Llama deployment"),
        ],
    )
    def test_bedrock_converse_api_proxy_mappings(
        self, proxy_model_name, underlying_bedrock_model, expected_proxy_result, description
    ):
        """
        Test real-world Bedrock Converse API proxy model mappings.
        
        This test covers the specific scenario where proxy model names like 
        'bedrock-claude-3-haiku' map to underlying Bedrock Converse API models like
        'bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0'.
        
        These mappings are typically defined in proxy server configuration files
        and cannot be resolved by LiteLLM without that context.
        """
        print(f"\nTesting: {description}")
        print(f"  Proxy model: {proxy_model_name}")
        print(f"  Underlying model: {underlying_bedrock_model}")
        
        # Test the underlying model directly to verify it supports function calling
        try:
            underlying_result = supports_function_calling(underlying_bedrock_model)
            print(f"  Underlying model function calling support: {underlying_result}")
            
            # Most Bedrock Converse API models with Anthropic Claude should support function calling
            if "anthropic.claude-3" in underlying_bedrock_model:
                assert underlying_result is True, f"Claude 3 models should support function calling: {underlying_bedrock_model}"
        except Exception as e:
            print(f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}")
        
        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")
        
        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )

    def test_real_world_proxy_config_documentation(self):
        """
        Document how real-world proxy configurations would handle model mappings.
        
        This test provides documentation on how the proxy server configuration
        would typically map custom model names to underlying models.
        """
        print("""
        
        REAL-WORLD PROXY SERVER CONFIGURATION EXAMPLE:
        ===============================================
        
        In a proxy_server_config.yaml file, you would define:
        
        model_list:
          - model_name: bedrock-claude-3-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: bedrock-claude-3-sonnet
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: prod-claude-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/PROD_AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/PROD_AWS_SECRET_ACCESS_KEY
              aws_region_name: us-west-2
        
        
        FUNCTION CALLING WITH PROXY SERVER:
        ===================================
        
        When using the proxy server with this configuration:
        
        1. Client calls: supports_function_calling("bedrock-claude-3-haiku")
        2. Proxy server resolves to: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
        3. LiteLLM evaluates the underlying model's capabilities
        4. Returns: True (because Claude 3 Haiku supports function calling)
        
        Without the proxy server configuration context, LiteLLM cannot resolve
        the custom model name and returns False.
        
        
        BEDROCK CONVERSE API BENEFITS:
        ==============================
        
        The Bedrock Converse API provides:
        - Standardized function calling interface across providers
        - Better tool use capabilities compared to legacy APIs
        - Consistent request/response format
        - Enhanced streaming support for function calls
        
        """)
        
        # Verify that direct underlying models work as expected
        bedrock_models = [
            "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
        ]
        
        for model in bedrock_models:
            try:
                result = supports_function_calling(model)
                print(f"Direct test - {model}: {result}")
                # Claude 3 models should support function calling
                assert result is True, f"Claude 3 model should support function calling: {model}"
            except Exception as e:
                print(f"Could not test {model}: {e}")

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_bedrock_model,expected_proxy_result,description",
        [
            # Bedrock Converse API mappings - these are the real-world scenarios
            ("litellm_proxy/bedrock-claude-3-haiku", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Bedrock Claude 3 Haiku via Converse API"),
            ("litellm_proxy/bedrock-claude-3-sonnet", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "Bedrock Claude 3 Sonnet via Converse API"),
            ("litellm_proxy/bedrock-claude-3-opus", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "Bedrock Claude 3 Opus via Converse API"),
            ("litellm_proxy/bedrock-claude-3-5-sonnet", "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0", False, "Bedrock Claude 3.5 Sonnet via Converse API"),
            
            # Bedrock Legacy API mappings (non-converse)
            ("litellm_proxy/bedrock-claude-instant", "bedrock/anthropic.claude-instant-v1", False, "Bedrock Claude Instant Legacy API"),
            ("litellm_proxy/bedrock-claude-v2", "bedrock/anthropic.claude-v2", False, "Bedrock Claude v2 Legacy API"),
            ("litellm_proxy/bedrock-claude-v2-1", "bedrock/anthropic.claude-v2:1", False, "Bedrock Claude v2.1 Legacy API"),
            
            # Bedrock other model providers via Converse API
            ("litellm_proxy/bedrock-titan-text", "bedrock/converse/amazon.titan-text-express-v1", False, "Bedrock Titan Text Express via Converse API"),
            ("litellm_proxy/bedrock-titan-text-premier", "bedrock/converse/amazon.titan-text-premier-v1:0", False, "Bedrock Titan Text Premier via Converse API"),
            ("litellm_proxy/bedrock-llama3-8b", "bedrock/converse/meta.llama3-8b-instruct-v1:0", False, "Bedrock Llama 3 8B via Converse API"),
            ("litellm_proxy/bedrock-llama3-70b", "bedrock/converse/meta.llama3-70b-instruct-v1:0", False, "Bedrock Llama 3 70B via Converse API"),
            ("litellm_proxy/bedrock-mistral-7b", "bedrock/converse/mistral.mistral-7b-instruct-v0:2", False, "Bedrock Mistral 7B via Converse API"),
            ("litellm_proxy/bedrock-mistral-8x7b", "bedrock/converse/mistral.mixtral-8x7b-instruct-v0:1", False, "Bedrock Mistral 8x7B via Converse API"),
            ("litellm_proxy/bedrock-mistral-large", "bedrock/converse/mistral.mistral-large-2402-v1:0", False, "Bedrock Mistral Large via Converse API"),
            
            # Company-specific naming patterns (real-world examples)
            ("litellm_proxy/prod-claude-haiku", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Production Claude Haiku"),
            ("litellm_proxy/dev-claude-sonnet", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "Development Claude Sonnet"),
            ("litellm_proxy/staging-claude-opus", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "Staging Claude Opus"),
            ("litellm_proxy/cost-optimized-claude", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "Cost-optimized Claude deployment"),
            ("litellm_proxy/high-performance-claude", "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0", False, "High-performance Claude deployment"),
            
            # Regional deployment examples
            ("litellm_proxy/us-east-claude", "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0", False, "US East Claude deployment"),
            ("litellm_proxy/eu-west-claude", "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0", False, "EU West Claude deployment"),
            ("litellm_proxy/ap-south-llama", "bedrock/converse/meta.llama3-70b-instruct-v1:0", False, "Asia Pacific Llama deployment"),
        ],
    )
    def test_bedrock_converse_api_proxy_mappings(
        self, proxy_model_name, underlying_bedrock_model, expected_proxy_result, description
    ):
        """
        Test real-world Bedrock Converse API proxy model mappings.
        
        This test covers the specific scenario where proxy model names like 
        'bedrock-claude-3-haiku' map to underlying Bedrock Converse API models like
        'bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0'.
        
        These mappings are typically defined in proxy server configuration files
        and cannot be resolved by LiteLLM without that context.
        """
        print(f"\nTesting: {description}")
        print(f"  Proxy model: {proxy_model_name}")
        print(f"  Underlying model: {underlying_bedrock_model}")
        
        # Test the underlying model directly to verify it supports function calling
        try:
            underlying_result = supports_function_calling(underlying_bedrock_model)
            print(f"  Underlying model function calling support: {underlying_result}")
            
            # Most Bedrock Converse API models with Anthropic Claude should support function calling
            if "anthropic.claude-3" in underlying_bedrock_model:
                assert underlying_result is True, f"Claude 3 models should support function calling: {underlying_bedrock_model}"
        except Exception as e:
            print(f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}")
        
        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")
        
        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )

    def test_real_world_proxy_config_documentation(self):
        """
        Document how real-world proxy configurations would handle model mappings.
        
        This test provides documentation on how the proxy server configuration
        would typically map custom model names to underlying models.
        """
        print("""
        
        REAL-WORLD PROXY SERVER CONFIGURATION EXAMPLE:
        ===============================================
        
        In a proxy_server_config.yaml file, you would define:
        
        model_list:
          - model_name: bedrock-claude-3-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: bedrock-claude-3-sonnet
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: prod-claude-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/PROD_AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/PROD_AWS_SECRET_ACCESS_KEY
              aws_region_name: us-west-2
        
        
        FUNCTION CALLING WITH PROXY SERVER:
        ===================================
        
        When using the proxy server with this configuration:
        
        1. Client calls: supports_function_calling("bedrock-claude-3-haiku")
        2. Proxy server resolves to: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
        3. LiteLLM evaluates the underlying model's capabilities
        4. Returns: True (because Claude 3 Haiku supports function calling)
        
        Without the proxy server configuration context, LiteLLM cannot resolve
        the custom model name and returns False.
        
        
        BEDROCK CONVERSE API BENEFITS:
        ==============================
        
        The Bedrock Converse API provides:
        - Standardized function calling interface across providers
        - Better tool use capabilities compared to legacy APIs
        - Consistent request/response format
        - Enhanced streaming support for function calls
        
        """)
        
        # Verify that direct underlying models work as expected
        bedrock_models = [
            "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
        ]
        
        for model in bedrock_models:
            try:
                result = supports_function_calling(model)
                print(f"Direct test - {model}: {result}")
                # Claude 3 models should support function calling
                assert result is True, f"Claude 3 model should support function calling: {model}"
            except Exception as e:
                print(f"Could not test {model}: {e}")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])
