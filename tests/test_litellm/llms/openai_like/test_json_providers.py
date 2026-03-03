"""
Tests for JSON-based provider configuration system.
"""

import os
import sys
from unittest.mock import MagicMock, patch

try:
    import pytest
except ImportError:
    # pytest not available, will run as standalone script
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestJSONProviderLoader:
    """Test JSON provider loading and configuration"""

    def test_load_json_providers(self):
        """Test that JSON providers load correctly"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Verify publicai is loaded
        assert JSONProviderRegistry.exists("publicai")

        # Get publicai config
        publicai = JSONProviderRegistry.get("publicai")
        assert publicai is not None
        assert publicai.base_url == "https://api.publicai.co/v1"
        assert publicai.api_key_env == "PUBLICAI_API_KEY"
        assert publicai.api_base_env == "PUBLICAI_API_BASE"
        assert publicai.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_dynamic_config_generation(self):
        """Test dynamic config class creation"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Test API info resolution
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.publicai.co/v1"

        # Test with custom base
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://custom.api.com", "test-key"
        )
        assert api_base == "https://custom.api.com"
        assert api_key == "test-key"

    def test_parameter_mapping(self):
        """Test parameter mapping works"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Test parameter mapping
        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(
            non_default_params, optional_params, "gpt-4", False
        )

        # max_completion_tokens should be mapped to max_tokens
        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

        # temperature should be passed through
        assert result["temperature"] == 0.7

    def test_supported_params(self):
        """Test that config returns supported params"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Get supported params
        supported = config.get_supported_openai_params("gpt-4")

        # Should have standard OpenAI params
        assert isinstance(supported, list)
        assert len(supported) > 0

    def test_tool_params_excluded_when_function_calling_not_supported(self):
        """Test that tool-related params are excluded for models that don't support
        function calling. Regression test for https://github.com/BerriAI/litellm/issues/21125"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Mock supports_function_calling to return False
        with patch("litellm.utils.supports_function_calling", return_value=False):
            supported = config.get_supported_openai_params("some-model-without-fc")

        tool_params = ["tools", "tool_choice", "function_call", "functions", "parallel_tool_calls"]
        for param in tool_params:
            assert param not in supported, (
                f"'{param}' should not be in supported params when function calling is not supported"
            )

        # Non-tool params should still be present
        assert "temperature" in supported
        assert "max_tokens" in supported
        assert "stop" in supported

    def test_tool_params_included_when_function_calling_supported(self):
        """Test that tool-related params are included for models that support function calling."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Mock supports_function_calling to return True
        with patch("litellm.utils.supports_function_calling", return_value=True):
            supported = config.get_supported_openai_params("some-model-with-fc")

        assert "tools" in supported
        assert "tool_choice" in supported

    def test_provider_resolution(self):
        """Test that provider resolution finds JSON providers"""
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )

        model, provider, api_key, api_base = get_llm_provider(
            model="publicai/gpt-4",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-4"
        assert provider == "publicai"
        assert api_base == "https://api.publicai.co/v1"

    def test_provider_config_manager(self):
        """Test that ProviderConfigManager returns JSON-based configs"""
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="gpt-4", provider=LlmProviders.PUBLICAI
        )

        assert config is not None
        assert config.custom_llm_provider == "publicai"


class TestPublicAIIntegration:
    """Integration tests for PublicAI provider"""

    def test_publicai_completion_basic(self):
        """Test basic completion call to PublicAI"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[{"role": "user", "content": "Say 'test successful' and nothing else"}],
                max_tokens=10,
            )

            # Verify response structure
            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], "message")
            assert hasattr(response.choices[0].message, "content")
            assert response.choices[0].message.content is not None

            # Check that we got a response
            content = response.choices[0].message.content.lower()
            assert len(content) > 0

            print(f"✓ PublicAI completion successful: {response.choices[0].message.content}")

        except Exception as e:
            if pytest:
                pytest.fail(f"PublicAI completion failed: {str(e)}")
            else:
                raise

    def test_publicai_completion_with_streaming(self):
        """Test streaming completion with PublicAI"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[{"role": "user", "content": "Count to 3"}],
                max_tokens=20,
                stream=True,
            )

            # Collect chunks
            chunks = []
            for chunk in response:
                assert chunk is not None
                if hasattr(chunk.choices[0], "delta") and hasattr(
                    chunk.choices[0].delta, "content"
                ):
                    if chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)

            # Verify we got chunks
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0

            print(f"✓ PublicAI streaming successful: {full_response}")

        except Exception as e:
            if pytest:
                pytest.fail(f"PublicAI streaming failed: {str(e)}")
            else:
                raise

    def test_publicai_parameter_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            # Use max_completion_tokens (OpenAI's newer parameter)
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[{"role": "user", "content": "Hi"}],
                max_completion_tokens=5,  # This should be mapped to max_tokens
            )

            assert response is not None
            assert len(response.choices) > 0

            print("✓ Parameter mapping successful")

        except Exception as e:
            if pytest:
                pytest.fail(f"Parameter mapping test failed: {str(e)}")
            else:
                raise

    def test_publicai_content_list_conversion(self):
        """Test that content list format is converted to string"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            # Send message with content as list (should be converted to string)
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Say hello"}
                        ]
                    }
                ],
                max_tokens=10,
            )

            assert response is not None
            assert len(response.choices) > 0

            print("✓ Content list conversion successful")

        except Exception as e:
            if pytest:
                pytest.fail(f"Content list conversion test failed: {str(e)}")
            else:
                raise


if __name__ == "__main__":
    # Run basic tests
    print("Testing JSON Provider System...")
    
    test_loader = TestJSONProviderLoader()
    print("\n1. Testing JSON provider loading...")
    test_loader.test_load_json_providers()
    print("   ✓ JSON providers loaded")
    
    print("\n2. Testing dynamic config generation...")
    test_loader.test_dynamic_config_generation()
    print("   ✓ Dynamic config works")
    
    print("\n3. Testing parameter mapping...")
    test_loader.test_parameter_mapping()
    print("   ✓ Parameter mapping works")
    
    print("\n4. Testing excluded params...")
    test_loader.test_excluded_params()
    print("   ✓ Excluded params work")
    
    print("\n5. Testing provider resolution...")
    test_loader.test_provider_resolution()
    print("   ✓ Provider resolution works")
    
    print("\n6. Testing provider config manager...")
    test_loader.test_provider_config_manager()
    print("   ✓ Config manager works")
    
    print("\n" + "="*50)
    print("PublicAI Integration Tests...")
    print("="*50)
    
    test_integration = TestPublicAIIntegration()
    
    print("\n7. Testing basic completion...")
    test_integration.test_publicai_completion_basic()
    
    print("\n8. Testing streaming...")
    test_integration.test_publicai_completion_with_streaming()
    
    print("\n9. Testing parameter mapping...")
    test_integration.test_publicai_parameter_mapping()
    
    print("\n10. Testing content list conversion...")
    test_integration.test_publicai_content_list_conversion()
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
