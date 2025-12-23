"""
Tests for URL-based provider loading in JSON provider system.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)


class TestJSONProviderURLLoader:
    """Test URL-based provider loading"""

    def test_load_from_json_string_success(self):
        """Test successfully loading providers from JSON string"""
        # Mock custom provider data
        custom_providers = {
            "custom_provider": {
                "base_url": "https://api.custom.com/v1",
                "api_key_env": "CUSTOM_API_KEY",
                "api_base_env": "CUSTOM_API_BASE",
                "base_class": "openai_gpt",
                "param_mappings": {
                    "max_completion_tokens": "max_tokens"
                }
            }
        }

        # Reset the registry
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        # Set environment variable with JSON string
        json_string = json.dumps(custom_providers)
        with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS": json_string}):
            JSONProviderRegistry.load()

            # Verify provider was loaded
            assert JSONProviderRegistry.exists("custom_provider")
            provider = JSONProviderRegistry.get("custom_provider")
            assert provider is not None
            assert provider.base_url == "https://api.custom.com/v1"
            assert provider.api_key_env == "CUSTOM_API_KEY"

    def test_load_from_json_string_invalid(self):
        """Test handling invalid JSON string"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        
        # Reset the registry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        # Set environment variable with invalid JSON
        with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS": "not valid json {"}):
            # Should not raise, just log warning
            JSONProviderRegistry.load()
            
            # Should still be loaded
            assert JSONProviderRegistry._loaded

    def test_load_from_url_success(self):
        """Test successfully loading providers from URL"""
        # Mock custom provider data
        custom_providers = {
            "custom_provider": {
                "base_url": "https://api.custom.com/v1",
                "api_key_env": "CUSTOM_API_KEY",
                "api_base_env": "CUSTOM_API_BASE",
                "base_class": "openai_gpt",
                "param_mappings": {
                    "max_completion_tokens": "max_tokens"
                }
            }
        }

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = custom_providers
        mock_response.raise_for_status.return_value = None

        # Reset the registry
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        with patch('httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            
            # Set environment variable
            with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS_URL": "https://example.com/providers.json"}):
                JSONProviderRegistry.load()

                # Verify provider was loaded
                assert JSONProviderRegistry.exists("custom_provider")
                provider = JSONProviderRegistry.get("custom_provider")
                assert provider is not None
                assert provider.base_url == "https://api.custom.com/v1"
                assert provider.api_key_env == "CUSTOM_API_KEY"

    def test_load_from_url_http_error(self):
        """Test handling HTTP errors when loading from URL"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        
        # Reset the registry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        # Mock HTTP error
        with patch('httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = httpx.HTTPError("Connection failed")
            
            with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS_URL": "https://example.com/providers.json"}):
                # Should not raise, just log warning
                JSONProviderRegistry.load()
                
                # Should still be loaded (but with no providers)
                assert JSONProviderRegistry._loaded

    def test_load_from_url_json_decode_error(self):
        """Test handling JSON decode errors when loading from URL"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        
        # Reset the registry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None

        with patch('httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            
            with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS_URL": "https://example.com/providers.json"}):
                # Should not raise, just log warning
                JSONProviderRegistry.load()
                
                # Should still be loaded
                assert JSONProviderRegistry._loaded

    def test_load_without_url_env_var(self):
        """Test loading when no URL environment variable is set"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        
        # Reset the registry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        with patch.dict(os.environ, {}, clear=True):
            # Should load without errors
            JSONProviderRegistry.load()
            assert JSONProviderRegistry._loaded

    def test_custom_provider_overwrites_local(self):
        """Test that custom providers from URL can overwrite local providers"""
        # Mock custom provider that overwrites publicai
        custom_providers = {
            "publicai": {
                "base_url": "https://custom.publicai.com/v1",
                "api_key_env": "CUSTOM_PUBLICAI_KEY",
                "base_class": "openai_gpt"
            }
        }

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = custom_providers
        mock_response.raise_for_status.return_value = None

        # Reset the registry
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        with patch('httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            
            with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS_URL": "https://example.com/providers.json"}):
                JSONProviderRegistry.load()

                # Verify custom provider overwrote the local one
                provider = JSONProviderRegistry.get("publicai")
                assert provider is not None
                assert provider.base_url == "https://custom.publicai.com/v1"
                assert provider.api_key_env == "CUSTOM_PUBLICAI_KEY"

    def test_invalid_url_scheme(self):
        """Test that non-http/https URL schemes are rejected"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        
        # Reset the registry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        # Try with file:// scheme (should be rejected)
        with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS_URL": "file:///etc/passwd"}):
            # Should not raise, just log warning
            JSONProviderRegistry.load()
            
            # Should still be loaded
            assert JSONProviderRegistry._loaded

    def test_merge_local_and_custom_providers(self):
        """Test that local and custom providers are merged"""
        # Mock custom provider (different from existing ones)
        custom_providers = {
            "my_custom_llm": {
                "base_url": "https://api.mycustom.com/v1",
                "api_key_env": "MY_CUSTOM_API_KEY",
                "base_class": "openai_gpt"
            }
        }

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = custom_providers
        mock_response.raise_for_status.return_value = None

        # Reset the registry
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        JSONProviderRegistry._loaded = False
        JSONProviderRegistry._providers = {}

        with patch('httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            
            with patch.dict(os.environ, {"LITELLM_CUSTOM_PROVIDERS_URL": "https://example.com/providers.json"}):
                JSONProviderRegistry.load()

                # Verify both local and custom providers exist
                assert JSONProviderRegistry.exists("publicai")  # Local provider
                assert JSONProviderRegistry.exists("my_custom_llm")  # Custom provider

