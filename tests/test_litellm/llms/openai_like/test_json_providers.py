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

    def test_provider_detection_by_api_base(self):
        """Test that JSON providers are detected by api_base automatically"""
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Ensure providers are loaded
        JSONProviderRegistry.load()

        # Get publicai config to use its base_url
        publicai_config = JSONProviderRegistry.get("publicai")
        assert publicai_config is not None

        # Test: When only api_base is provided (matching JSON provider base_url),
        # the provider should be automatically detected
        model, provider, api_key, api_base = get_llm_provider(
            model="gpt-4",
            custom_llm_provider=None,
            api_base=publicai_config.base_url,
            api_key=None,
        )

        # Verify provider was detected correctly
        assert provider == "publicai"
        assert api_base == publicai_config.base_url
        assert model == "gpt-4"

        # Test with api_base that has trailing slash (should still match)
        model2, provider2, api_key2, api_base2 = get_llm_provider(
            model="gpt-4",
            custom_llm_provider=None,
            api_base=publicai_config.base_url + "/",
            api_key=None,
        )

        assert provider2 == "publicai"
        assert api_base2 == publicai_config.base_url + "/"

        # Test with api_base that includes a subpath (should still match)
        model3, provider3, api_key3, api_base3 = get_llm_provider(
            model="gpt-4",
            custom_llm_provider=None,
            api_base=publicai_config.base_url + "/chat",
            api_key=None,
        )

        assert provider3 == "publicai"
        assert api_base3 == publicai_config.base_url + "/chat"


class TestDynamicProviderEnum:
    """Test get_llm_provider_enum function for dynamic provider support"""

    def test_get_llm_provider_enum_with_enum_provider(self):
        """Test get_llm_provider_enum with standard enum provider"""
        from litellm.types.utils import get_llm_provider_enum, LlmProviders

        # Test with a standard enum provider
        result = get_llm_provider_enum("openai")
        assert isinstance(result, LlmProviders)
        assert result == LlmProviders.OPENAI
        assert result.value == "openai"

    def test_get_llm_provider_enum_with_json_provider(self):
        """Test get_llm_provider_enum with JSON-configured provider"""
        from litellm.types.utils import get_llm_provider_enum, LlmProviders
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Ensure providers are loaded
        JSONProviderRegistry.load()

        # Find a JSON provider that is NOT in the enum
        # Try synthetic, poe, chutes, etc. - these should be JSON-only providers
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                # Check if it's NOT in enum
                try:
                    LlmProviders(provider_name)
                    # If we get here, it's in enum, skip it
                    continue
                except ValueError:
                    # Not in enum, use this one
                    json_provider_name = provider_name
                    break

        if json_provider_name is None:
            # Fallback: use any JSON provider and check it's not an enum
            all_json_providers = JSONProviderRegistry.list_providers()
            for provider_name in all_json_providers:
                try:
                    LlmProviders(provider_name)
                    continue
                except ValueError:
                    json_provider_name = provider_name
                    break

        if json_provider_name is None:
            # Skip test if no JSON-only provider found
            if pytest:
                pytest.skip("No JSON-only provider found for testing")
            return

        # Test with a JSON provider that's not in enum
        result = get_llm_provider_enum(json_provider_name)
        assert result is not None
        assert hasattr(result, "value")
        assert result.value == json_provider_name
        assert hasattr(result, "name")
        # JSON provider should return the value when converted to string
        assert str(result) == json_provider_name

    def test_get_llm_provider_enum_with_invalid_provider(self):
        """Test get_llm_provider_enum raises ValueError for invalid provider"""
        from litellm.types.utils import get_llm_provider_enum

        # Test with invalid provider
        try:
            get_llm_provider_enum("invalid_provider_xyz")
            assert False, "Expected ValueError for invalid provider"
        except ValueError as e:
            assert "Unknown provider" in str(e)

    def test_get_llm_provider_enum_json_provider_equality(self):
        """Test JSON provider enum-like object equality"""
        from litellm.types.utils import get_llm_provider_enum, LlmProviders
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Find a JSON provider that is NOT in the enum
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                try:
                    LlmProviders(provider_name)
                    continue
                except ValueError:
                    json_provider_name = provider_name
                    break

        if json_provider_name is None:
            if pytest:
                pytest.skip("No JSON-only provider found for testing")
            return

        json_provider = get_llm_provider_enum(json_provider_name)
        enum_provider = LlmProviders.OPENAI

        # Test __eq__ method
        assert json_provider == json_provider_name
        assert json_provider != "openai"
        assert json_provider != enum_provider

        # Test hash
        assert hash(json_provider) == hash(json_provider_name)

    def test_get_llm_provider_enum_with_multiple_json_providers(self):
        """Test get_llm_provider_enum with multiple JSON providers"""
        from litellm.types.utils import get_llm_provider_enum
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Test multiple JSON providers
        providers_to_test = ["publicai", "synthetic", "poe", "chutes"]
        for provider_name in providers_to_test:
            if JSONProviderRegistry.exists(provider_name):
                result = get_llm_provider_enum(provider_name)
                assert result is not None
                assert result.value == provider_name


class TestDynamicConstantsIntegration:
    """Test dynamic addition of JSON providers to constants"""

    def test_get_json_provider_endpoints(self):
        """Test _get_json_provider_endpoints function"""
        from litellm.constants import _get_json_provider_endpoints
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()
        endpoints = _get_json_provider_endpoints()

        # Should return a list
        assert isinstance(endpoints, list)

        # Should include JSON provider endpoints
        all_providers = JSONProviderRegistry.get_all_providers()
        expected_endpoints = [config.base_url for config in all_providers.values()]
        for endpoint in expected_endpoints:
            assert endpoint in endpoints

    def test_get_json_provider_names(self):
        """Test _get_json_provider_names function"""
        from litellm.constants import _get_json_provider_names
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()
        provider_names = _get_json_provider_names()

        # Should return a list
        assert isinstance(provider_names, list)

        # Should include JSON provider names
        expected_names = JSONProviderRegistry.list_providers()
        for name in expected_names:
            assert name in provider_names

    def test_openai_compatible_endpoints_includes_json_providers(self):
        """Test that openai_compatible_endpoints includes JSON provider endpoints"""
        from litellm.constants import openai_compatible_endpoints
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()
        all_providers = JSONProviderRegistry.get_all_providers()

        # Check that JSON provider endpoints are in the list
        for config in all_providers.values():
            assert config.base_url in openai_compatible_endpoints

    def test_openai_compatible_providers_includes_json_providers(self):
        """Test that openai_compatible_providers includes JSON provider names"""
        from litellm.constants import openai_compatible_providers
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()
        json_provider_names = JSONProviderRegistry.list_providers()

        # Check that JSON provider names are in the list
        for name in json_provider_names:
            assert name in openai_compatible_providers

    def test_openai_text_completion_compatible_providers_includes_json_providers(self):
        """Test that openai_text_completion_compatible_providers includes JSON providers"""
        from litellm.constants import openai_text_completion_compatible_providers
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()
        json_provider_names = JSONProviderRegistry.list_providers()

        # JSON providers should be included in text completion compatible providers
        for name in json_provider_names:
            assert name in openai_text_completion_compatible_providers


class TestDynamicProviderErrorHandling:
    """Test error handling when using get_llm_provider_enum in various contexts"""

    def test_get_llm_provider_enum_in_batches_context(self):
        """Test that get_llm_provider_enum works correctly in batches context"""
        from litellm.types.utils import get_llm_provider_enum
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Test that get_llm_provider_enum can handle JSON providers
        # This simulates what happens in batches.main when it calls get_llm_provider_enum
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                json_provider_name = provider_name
                break

        if json_provider_name:
            # Should not raise ValueError for JSON provider
            result = get_llm_provider_enum(json_provider_name)
            assert result is not None
            assert result.value == json_provider_name

    def test_get_llm_provider_enum_in_containers_context(self):
        """Test that get_llm_provider_enum works correctly in containers context"""
        from litellm.types.utils import get_llm_provider_enum
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Test that get_llm_provider_enum can handle JSON providers
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                json_provider_name = provider_name
                break

        if json_provider_name:
            # Should not raise ValueError for JSON provider
            try:
                result = get_llm_provider_enum(json_provider_name)
                assert result is not None
            except ValueError as e:
                # Should not raise "Unknown provider" error
                assert "Unknown provider" not in str(e)

    def test_get_llm_provider_enum_in_files_context(self):
        """Test that get_llm_provider_enum works correctly in files context"""
        from litellm.types.utils import get_llm_provider_enum
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Test that get_llm_provider_enum can handle JSON providers
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                json_provider_name = provider_name
                break

        if json_provider_name:
            # Should not raise ValueError for JSON provider
            try:
                result = get_llm_provider_enum(json_provider_name)
                assert result is not None
            except ValueError as e:
                # Should not raise "Unknown provider" error
                assert "Unknown provider" not in str(e)

    def test_get_llm_provider_enum_in_images_context(self):
        """Test that get_llm_provider_enum works correctly in images context"""
        from litellm.types.utils import get_llm_provider_enum
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Test that get_llm_provider_enum can handle JSON providers
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                json_provider_name = provider_name
                break

        if json_provider_name:
            # Should not raise ValueError for JSON provider
            try:
                result = get_llm_provider_enum(json_provider_name)
                assert result is not None
            except ValueError as e:
                # Should not raise "Unknown provider" error
                assert "Unknown provider" not in str(e)

    def test_get_llm_provider_enum_in_cost_calculator_context(self):
        """Test that get_llm_provider_enum works correctly in cost_calculator context"""
        from litellm.types.utils import get_llm_provider_enum
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()

        # Test that get_llm_provider_enum can handle JSON providers
        json_provider_name = None
        for provider_name in ["synthetic", "poe", "chutes", "nano-gpt", "apertis", "llamagate"]:
            if JSONProviderRegistry.exists(provider_name):
                json_provider_name = provider_name
                break

        if json_provider_name:
            # Should not raise ValueError for JSON provider
            try:
                result = get_llm_provider_enum(json_provider_name)
                assert result is not None
            except ValueError as e:
                # Should not raise "Unknown provider" error
                assert "Unknown provider" not in str(e)


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
    
    print("\n4. Testing provider detection by api_base...")
    test_loader.test_provider_detection_by_api_base()
    print("   ✓ Provider detection by api_base works")
    
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
