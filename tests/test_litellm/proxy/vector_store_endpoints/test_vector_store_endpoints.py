import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    LiteLLM_ManagedVectorStore,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.endpoints import (
    _update_request_data_with_litellm_managed_vector_store_registry,
)
from litellm.proxy.vector_store_endpoints.utils import (
    check_vector_store_permission,
    is_allowed_to_call_vector_store_endpoint,
)
from litellm.types.utils import LlmProviders


@pytest.mark.asyncio
async def test_router_avector_store_search_passes_correct_args():
    """
    Test that router.avector_store_search() passes the correct arguments 
    to downstream litellm.vector_stores.asearch() with custom_llm_provider and query.
    """
    # Create a router
    router = litellm.Router(model_list=[])
    
    # Mock the router's _init_vector_store_api_endpoints method to avoid real API calls
    with patch.object(router, '_init_vector_store_api_endpoints') as mock_init:
        mock_init.return_value = {
            "object": "vector_store.search_results.page",
            "search_query": "test query",
            "data": []
        }
        
        # Call router's avector_store_search
        result = await router.avector_store_search(
            vector_store_id="test_store_id",
            query="test query",
            custom_llm_provider="bedrock"
        )
        
        # Verify the internal method was called with correct args
        mock_init.assert_called_once()
        call_args = mock_init.call_args
        
        # Check that the original function is passed correctly
        assert call_args[1]["vector_store_id"] == "test_store_id"
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["custom_llm_provider"] == "bedrock"


def test_update_request_data_with_litellm_managed_vector_store_registry():
    """
    Test that _update_request_data_with_litellm_managed_vector_store_registry
    correctly updates request data with vector store registry information.
    """
    # Setup test data
    data = {"existing_key": "existing_value"}
    vector_store_id = "test_store_id"
    
    # Mock vector store registry
    mock_vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "test_store_id", 
        "custom_llm_provider": "bedrock",
        "litellm_credential_name": "test_credential",
        "litellm_params": {"api_key": "test_key", "aws_region_name": "us-east-1"}
    }
    
    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = mock_vector_store
    
    # Test with vector store registry
    with patch.object(litellm, 'vector_store_registry', mock_registry):
        result = _update_request_data_with_litellm_managed_vector_store_registry(
            data=data,
            vector_store_id=vector_store_id
        )
        
        # Verify the data was updated correctly
        assert result["existing_key"] == "existing_value"  # Original data preserved
        assert result["custom_llm_provider"] == "bedrock"
        assert result["litellm_credential_name"] == "test_credential"
        assert result["api_key"] == "test_key"
        assert result["aws_region_name"] == "us-east-1"
        
        # Verify registry was called correctly
        mock_registry.get_litellm_managed_vector_store_from_registry.assert_called_once_with(
            vector_store_id="test_store_id"
        )
    
    # Test with no vector store registry
    with patch.object(litellm, 'vector_store_registry', None):
        original_data = {"existing_key": "existing_value"}
        result = _update_request_data_with_litellm_managed_vector_store_registry(
            data=original_data,
            vector_store_id=vector_store_id
        )
        
        # Verify data remains unchanged when no registry
        assert result == original_data


class TestCheckVectorStorePermission:
    """Test suite for check_vector_store_permission function."""
    
    def test_permission_allowed_in_key_metadata(self):
        """Test that permission is allowed when found in key metadata."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["read", "write"]
                }
            ]
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        assert result is True
    
    def test_permission_allowed_in_team_metadata(self):
        """Test that permission is allowed when found in team metadata."""
        team_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "team-index",
                    "index_permissions": ["write"]
                }
            ]
        }
        
        result = check_vector_store_permission(
            index_name="team-index",
            permission="write",
            key_metadata=None,
            team_metadata=team_metadata,
        )
        
        assert result is True
    
    def test_permission_denied_wrong_permission(self):
        """Test that permission is denied when index exists but wrong permission."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="write",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        assert result is False
    
    def test_permission_denied_index_not_found(self):
        """Test that permission is denied when index doesn't exist."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "other-index",
                    "index_permissions": ["read", "write"]
                }
            ]
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        assert result is False
    
    def test_permission_denied_no_metadata(self):
        """Test that permission is denied when no metadata provided."""
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=None,
            team_metadata=None,
        )
        
        assert result is False
    
    def test_permission_denied_no_allowed_indexes_field(self):
        """Test that permission is denied when metadata has no allowed_vector_store_indexes."""
        key_metadata = {
            "some_other_field": "value"
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        assert result is False
    
    def test_key_metadata_takes_precedence(self):
        """Test that key metadata is checked and returns permission successfully."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        team_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["write"]
                }
            ]
        }
        
        # Should find permission in key_metadata (checked first)
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=team_metadata,
        )
        
        assert result is True
    
    def test_team_metadata_as_fallback(self):
        """Test that team metadata is checked when key metadata doesn't have permission."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "other-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        team_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["write"]
                }
            ]
        }
        
        # Should find permission in team_metadata
        result = check_vector_store_permission(
            index_name="my-index",
            permission="write",
            key_metadata=key_metadata,
            team_metadata=team_metadata,
        )
        
        assert result is True
    
    def test_multiple_indexes_in_metadata(self):
        """Test handling multiple indexes in metadata."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "index-1",
                    "index_permissions": ["read"]
                },
                {
                    "index_name": "index-2",
                    "index_permissions": ["write"]
                },
                {
                    "index_name": "index-3",
                    "index_permissions": ["read", "write"]
                }
            ]
        }
        
        # Test each index
        assert check_vector_store_permission("index-1", "read", key_metadata, None) is True
        assert check_vector_store_permission("index-1", "write", key_metadata, None) is False
        assert check_vector_store_permission("index-2", "write", key_metadata, None) is True
        assert check_vector_store_permission("index-2", "read", key_metadata, None) is False
        assert check_vector_store_permission("index-3", "read", key_metadata, None) is True
        assert check_vector_store_permission("index-3", "write", key_metadata, None) is True
    
    def test_invalid_metadata_structure(self):
        """Test handling of invalid metadata structures."""
        # Test when allowed_vector_store_indexes is not a list
        key_metadata = {
            "allowed_vector_store_indexes": "not-a-list"
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        assert result is False
        
        # Test when index config is not a dict
        key_metadata = {
            "allowed_vector_store_indexes": [
                "not-a-dict",
                {
                    "index_name": "my-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        # Should still work because it skips invalid entries
        assert result is True
    
    def test_missing_index_permissions_field(self):
        """Test when index_permissions field is missing."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index"
                    # Missing index_permissions field
                }
            ]
        }
        
        result = check_vector_store_permission(
            index_name="my-index",
            permission="read",
            key_metadata=key_metadata,
            team_metadata=None,
        )
        
        assert result is False


class TestIsAllowedToCallVectorStoreEndpoint:
    """Test suite for is_allowed_to_call_vector_store_endpoint function."""
    
    def test_read_permission_allowed(self):
        """Test read permission is checked correctly."""
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/my-index/search"
        
        # Mock user API key with permissions
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        mock_user_api_key.team_metadata = None
        
        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")]
        }
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=mock_provider_config):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is True
    
    def test_write_permission_allowed(self):
        """Test write permission is checked correctly."""
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/v1/vector_stores/my-index/create"
        
        # Mock user API key with permissions
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["write"]
                }
            ]
        }
        mock_user_api_key.team_metadata = None
        
        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")]
        }
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=mock_provider_config):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is True
    
    def test_permission_denied_wrong_permission(self):
        """Test permission denied when user has read but tries write."""
        # Mock request for write operation
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/v1/vector_stores/my-index/create"
        
        # Mock user API key with only read permissions
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        mock_user_api_key.team_metadata = None
        
        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")]
        }
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=mock_provider_config):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is False
    
    def test_provider_config_not_found(self):
        """Test when provider config is not found."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/my-index/search"
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = {}
        mock_user_api_key.team_metadata = None
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=None):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is False
    
    def test_endpoint_not_recognized(self):
        """Test when endpoint doesn't match any read or write patterns."""
        # Mock request with unrecognized path
        mock_request = MagicMock(spec=Request)
        mock_request.method = "DELETE"
        mock_request.url.path = "/v1/vector_stores/my-index/unknown"
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "my-index",
                    "index_permissions": ["read", "write"]
                }
            ]
        }
        mock_user_api_key.team_metadata = None
        
        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")]
        }
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=mock_provider_config):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is False
    
    def test_team_metadata_permissions(self):
        """Test that team metadata permissions work."""
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/team-index/search"
        
        # Mock user API key with no key metadata but team metadata
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = None
        mock_user_api_key.team_metadata = {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "team-index",
                    "index_permissions": ["read"]
                }
            ]
        }
        
        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")]
        }
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=mock_provider_config):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="team-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is True
    
    def test_no_permissions_configured(self):
        """Test when user has no vector store permissions configured."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/my-index/search"
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.metadata = {}
        mock_user_api_key.team_metadata = {}
        
        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")]
        }
        
        with patch('litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config', return_value=mock_provider_config):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )
        
        assert result is False 