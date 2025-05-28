import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig


@pytest.fixture
def mock_azure_env_vars_none():
    """Fixture to mock all Azure environment variables to None"""
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            "AZURE_TENANT_ID": None,
            "AZURE_CLIENT_ID": None,
            "AZURE_CLIENT_SECRET": None,
            "AZURE_USERNAME": None,
            "AZURE_PASSWORD": None
        }.get(key, default)
        yield mock_getenv


class TestAzureOpenAIResponsesAPIConfig:

    def setup_method(self):
        self.config = AzureOpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_validate_environment(self, mock_azure_env_vars_none):
        """Test that validate_environment correctly sets the Authorization header"""
        # Test with provided API key
        headers = {}
        litellm_params = {}
        api_key = "test_api_key"

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key=api_key,
            litellm_params=litellm_params
        )

        assert "Authorization" in result
        assert result["Authorization"] == f"Bearer {api_key}"

        # Test with empty headers
        headers = {}

        with patch("litellm.api_key", "litellm_api_key"):
            result = self.config.validate_environment(
                headers=headers,
                model=self.model,
                litellm_params=litellm_params
            )

            assert "Authorization" in result
            assert result["Authorization"] == "Bearer litellm_api_key"

        # Test with existing headers
        headers = {"Content-Type": "application/json"}

        with patch("litellm.azure_key", "azure_key"):
            with patch("litellm.api_key", None):
                result = self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    litellm_params=litellm_params
                )

                assert "Authorization" in result
                assert result["Authorization"] == "Bearer azure_key"
                assert "Content-Type" in result
                assert result["Content-Type"] == "application/json"

        # Test with environment variable
        headers = {}

        with patch("litellm.api_key", None):
            with patch("litellm.azure_key", None):
                with patch(
                    "litellm.llms.azure.responses.transformation.get_secret_str",
                    return_value="env_api_key",
                ):
                    result = self.config.validate_environment(
                        headers=headers,
                        model=self.model,
                        litellm_params=litellm_params
                    )

                    assert "Authorization" in result
                    assert result["Authorization"] == "Bearer env_api_key"
    
    def test_validate_environment_with_azure_ad_token_from_entra_id(self, mock_azure_env_vars_none):
        """Test validate_environment with Azure AD Token from Entra ID"""
        headers = {}
        litellm_params = {
            "tenant_id": "test_tenant_id",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret"
        }
        
        mock_token_provider = MagicMock(return_value="entra_id_token")
        
        with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id",
                return_value=mock_token_provider) as mock_get_token:
            result = self.config.validate_environment(
                headers=headers,
                model=self.model,
                litellm_params=litellm_params
            )
            
            mock_get_token.assert_called_once_with(
                tenant_id="test_tenant_id",
                client_id="test_client_id",
                client_secret="test_client_secret"
            )
            assert "Authorization" in result
            assert result["Authorization"] == "Bearer entra_id_token"
    
    def test_validate_environment_with_azure_username_password(self, mock_azure_env_vars_none):
        """Test validate_environment with Azure username and password"""
        headers = {}
        litellm_params = {
            "azure_username": "test_username",
            "azure_password": "test_password",
            "client_id": "test_client_id"
        }
        
        mock_token_provider = MagicMock(return_value="username_password_token")
        
        with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_username_password",
                return_value=mock_token_provider) as mock_get_token:
            result = self.config.validate_environment(
                headers=headers,
                model=self.model,
                litellm_params=litellm_params
            )
            
            mock_get_token.assert_called_once_with(
                azure_username="test_username",
                azure_password="test_password",
                client_id="test_client_id"
            )
            assert "Authorization" in result
            assert result["Authorization"] == "Bearer username_password_token"
    
    def test_validate_environment_with_oidc_token(self, mock_azure_env_vars_none):
        """Test validate_environment with OIDC token"""
        headers = {}
        litellm_params = {
            "azure_ad_token": "oidc/test_token",
            "client_id": "test_client_id",
            "tenant_id": "test_tenant_id"
        }
        
        with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_oidc",
                return_value="processed_oidc_token") as mock_get_token:
            result = self.config.validate_environment(
                headers=headers,
                model=self.model,
                litellm_params=litellm_params
            )
            
            mock_get_token.assert_called_once_with(
                azure_ad_token="oidc/test_token",
                azure_client_id="test_client_id",
                azure_tenant_id="test_tenant_id"
            )
            assert "Authorization" in result
            assert result["Authorization"] == "Bearer processed_oidc_token"
    
    def test_validate_environment_with_service_principal(self, mock_azure_env_vars_none):
        """Test validate_environment with Service Principal"""
        headers = {}
        litellm_params = {}
        
        # Create a token provider that returns a token when called
        mock_token = lambda : "service_principal_token"
        
        with patch("litellm.api_key", None):
            with patch("litellm.azure_key", None):
                with patch("litellm.enable_azure_ad_token_refresh", True):
                    # Mock the get_azure_ad_token_provider function in the module
                    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_provider",
                            return_value=mock_token) as mock_get_provider:
                        # Call the method
                        result = self.config.validate_environment(
                            headers=headers,
                            model=self.model,
                            litellm_params=litellm_params
                        )
                        
                        # Verify the token provider was retrieved
                        mock_get_provider.assert_called_once()
                        
                        # Verify the result contains the expected Authorization header
                        assert "Authorization" in result
                        assert result["Authorization"] == f"Bearer {mock_token()}"
    
    def test_validate_environment_service_principal_error(self, mock_azure_env_vars_none):
        """Test validate_environment when Service Principal raises ValueError"""
        headers = {}
        litellm_params = {}
        
        with patch("litellm.api_key", None):
            with patch("litellm.azure_key", None):
                with patch("litellm.enable_azure_ad_token_refresh", True):
                    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_provider",
                            side_effect=ValueError("Token provider error")):
                        with patch("litellm.llms.azure.responses.transformation.get_secret_str",
                                return_value="fallback_api_key"):
                            result = self.config.validate_environment(
                                headers=headers,
                                model=self.model,
                                litellm_params=litellm_params
                            )
                            
                            assert "Authorization" in result
                            assert result["Authorization"] == "Bearer fallback_api_key"
    
    def test_validate_environment_priority_order(self, mock_azure_env_vars_none):
        """Test the priority order of token sources"""
        headers = {}
        
        # Test 1: azure_ad_token has highest priority
        litellm_params = {
            "azure_ad_token": "ad_token_from_params"
        }
        api_key = "direct_api_key"
        
        # All sources available - should use azure_ad_token (highest priority)
        with patch("litellm.api_key", "litellm_api_key"):
            with patch("litellm.azure_key", "azure_key"):
                with patch("litellm.llms.azure.responses.transformation.get_secret_str",
                        return_value="env_api_key"):
                    result = self.config.validate_environment(
                        headers=headers,
                        model=self.model,
                        api_key=api_key,
                        litellm_params=litellm_params
                    )
                    
                    assert result["Authorization"] == "Bearer ad_token_from_params"
        
        # Test 2: Without azure_ad_token, api_key has next highest priority
        litellm_params = {}  # No azure_ad_token
        api_key = "direct_api_key"
        
        with patch("litellm.api_key", "litellm_api_key"):
            with patch("litellm.azure_key", "azure_key"):
                with patch("litellm.llms.azure.responses.transformation.get_secret_str",
                        return_value="env_api_key"):
                    result = self.config.validate_environment(
                        headers=headers,
                        model=self.model,
                        api_key=api_key,
                        litellm_params=litellm_params
                    )
                    
                    assert result["Authorization"] == "Bearer direct_api_key"
        
        # No direct api_key - should use litellm_api_key since no azure_ad_token in params
        with patch("litellm.api_key", "litellm_api_key"):
            with patch("litellm.azure_key", "azure_key"):
                with patch("litellm.llms.azure.responses.transformation.get_secret_str",
                        return_value="env_api_key"):
                    result = self.config.validate_environment(
                        headers=headers,
                        model=self.model,
                        litellm_params=litellm_params
                    )
                    
                    assert result["Authorization"] == "Bearer litellm_api_key"
        
        # Test the complete fallback chain
        litellm_params = {}  # No azure_ad_token
        
        # Should use litellm.api_key
        with patch("litellm.api_key", "litellm_api_key"):
            with patch("litellm.azure_key", "azure_key"):
                result = self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    litellm_params=litellm_params
                )
                
                assert result["Authorization"] == "Bearer litellm_api_key"
        
        # Should use litellm.azure_key
        with patch("litellm.api_key", None):
            with patch("litellm.azure_key", "azure_key"):
                result = self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    litellm_params=litellm_params
                )
                
                assert result["Authorization"] == "Bearer azure_key"

    def test_validate_environment_with_env_variables(self, mock_azure_env_vars_none):
        """Test validate_environment with environment variables"""
        headers = {}
        litellm_params = {}
        
        # Test with environment variables for Entra ID
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "AZURE_TENANT_ID": "env_tenant_id",
                "AZURE_CLIENT_ID": "env_client_id",
                "AZURE_CLIENT_SECRET": "env_client_secret"
            }.get(key, default)
            
            mock_token_provider = MagicMock(return_value="env_entra_id_token")
            
            with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id",
                    return_value=mock_token_provider) as mock_get_token:
                result = self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    litellm_params=litellm_params
                )
                
                mock_get_token.assert_called_once_with(
                    tenant_id="env_tenant_id",
                    client_id="env_client_id",
                    client_secret="env_client_secret"
                )
                assert result["Authorization"] == "Bearer env_entra_id_token"
        
        # Test with environment variables for username/password
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "AZURE_USERNAME": "env_username",
                "AZURE_PASSWORD": "env_password",
                "AZURE_CLIENT_ID": "env_client_id"
            }.get(key, default)
            
            mock_token_provider = MagicMock(return_value="env_username_token")
            
            with patch("litellm.api_key", None):
                with patch("litellm.azure_key", None):
                    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_username_password",
                            return_value=mock_token_provider) as mock_get_token:
                        result = self.config.validate_environment(
                            headers=headers,
                            model=self.model,
                            litellm_params=litellm_params
                        )
                        
                        mock_get_token.assert_called_once_with(
                            azure_username="env_username",
                            azure_password="env_password",
                            client_id="env_client_id"
                        )
                        assert result["Authorization"] == "Bearer env_username_token"

    def test_validate_environment_with_azure_api_key_env_vars(self, mock_azure_env_vars_none):
        """Test validate_environment with specific Azure API key environment variables"""
        headers = {}
        litellm_params = {}
        
        # Test with AZURE_OPENAI_API_KEY environment variable
        with patch("litellm.api_key", None):
            with patch("litellm.azure_key", None):
                with patch("litellm.llms.azure.responses.transformation.get_secret_str") as mock_get_secret:
                    # Set up the mock to return different values based on the input key
                    mock_get_secret.side_effect = lambda key: "azure_openai_key" if key == "AZURE_OPENAI_API_KEY" else None
                    
                    result = self.config.validate_environment(
                        headers=headers,
                        model=self.model,
                        litellm_params=litellm_params
                    )
                    
                    assert "Authorization" in result
                    assert result["Authorization"] == "Bearer azure_openai_key"
                    mock_get_secret.assert_called_with("AZURE_OPENAI_API_KEY")
        
        # Test with AZURE_API_KEY environment variable (fallback)
        with patch("litellm.api_key", None):
            with patch("litellm.azure_key", None):
                with patch("litellm.llms.azure.responses.transformation.get_secret_str") as mock_get_secret:
                    # First call returns None (for AZURE_OPENAI_API_KEY), second call returns a value (for AZURE_API_KEY)
                    mock_get_secret.side_effect = lambda key: None if key == "AZURE_OPENAI_API_KEY" else "azure_api_key"
                    
                    result = self.config.validate_environment(
                        headers=headers,
                        model=self.model,
                        litellm_params=litellm_params
                    )
                    
                    assert "Authorization" in result
                    assert result["Authorization"] == "Bearer azure_api_key"
                    # Should be called twice, once for each environment variable
                    assert mock_get_secret.call_count == 2
