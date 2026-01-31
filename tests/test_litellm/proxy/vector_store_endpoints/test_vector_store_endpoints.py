import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from fastapi import HTTPException

import litellm
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    LiteLLM_ManagedVectorStore,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.endpoints import (
    _update_request_data_with_litellm_managed_vector_store_registry,
)
from litellm.proxy.vector_store_endpoints.management_endpoints import (
    _check_vector_store_access,
    _resolve_embedding_config,
    _resolve_embedding_config_from_db,
    _resolve_embedding_config_from_router,
    create_vector_store_in_db,
    new_vector_store,
)
from litellm.proxy.vector_store_endpoints.utils import (
    check_vector_store_permission,
    is_allowed_to_call_vector_store_endpoint,
    is_allowed_to_call_vector_store_files_endpoint,
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
    with patch.object(router, "_init_vector_store_api_endpoints") as mock_init:
        mock_init.return_value = {
            "object": "vector_store.search_results.page",
            "search_query": "test query",
            "data": [],
        }

        # Call router's avector_store_search
        result = await router.avector_store_search(
            vector_store_id="test_store_id",
            query="test query",
            custom_llm_provider="bedrock",
        )

        # Verify the internal method was called with correct args
        mock_init.assert_called_once()
        call_args = mock_init.call_args

        # Check that the original function is passed correctly
        assert call_args[1]["vector_store_id"] == "test_store_id"
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["custom_llm_provider"] == "bedrock"


@pytest.mark.asyncio
async def test_router_avector_store_file_list_passes_correct_args():
    with patch(
        "litellm.vector_store_files.main.alist",
        new=AsyncMock(return_value={"object": "list", "data": []}),
    ) as mock_alist:
        router = litellm.Router(model_list=[])

        result = await router.avector_store_file_list(
            vector_store_id="test_store_id",
            limit=5,
            order="asc",
            custom_llm_provider="openai",
        )

        assert result == {"object": "list", "data": []}
        mock_alist.assert_called_once()
        call_kwargs = mock_alist.call_args.kwargs
        assert call_kwargs["vector_store_id"] == "test_store_id"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["order"] == "asc"
        assert call_kwargs["custom_llm_provider"] == "openai"


def test_router_vector_store_file_delete_passes_correct_args():
    with patch(
        "litellm.vector_store_files.main.delete",
        return_value={"deleted": True},
    ) as mock_delete:
        router = litellm.Router(model_list=[])

        result = router.vector_store_file_delete(
            vector_store_id="test_store_id",
            file_id="file-123",
            custom_llm_provider="openai",
        )

        assert result == {"deleted": True}
        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args.kwargs
        assert call_kwargs["vector_store_id"] == "test_store_id"
        assert call_kwargs["file_id"] == "file-123"
        assert call_kwargs["custom_llm_provider"] == "openai"


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
        "litellm_params": {"api_key": "test_key", "aws_region_name": "us-east-1"},
    }

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = (
        mock_vector_store
    )

    # Test with vector store registry
    with patch.object(litellm, "vector_store_registry", mock_registry):
        result = _update_request_data_with_litellm_managed_vector_store_registry(
            data=data, vector_store_id=vector_store_id
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
    with patch.object(litellm, "vector_store_registry", None):
        original_data = {"existing_key": "existing_value"}
        result = _update_request_data_with_litellm_managed_vector_store_registry(
            data=original_data, vector_store_id=vector_store_id
        )

        # Verify data remains unchanged when no registry
        assert result == original_data


class TestCheckVectorStorePermission:
    """Test suite for check_vector_store_permission function."""

    def test_permission_allowed_in_key_metadata(self):
        """Test that permission is allowed when found in key metadata."""
        key_metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["read", "write"]}
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
                {"index_name": "team-index", "index_permissions": ["write"]}
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
                {"index_name": "my-index", "index_permissions": ["read"]}
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
                {"index_name": "other-index", "index_permissions": ["read", "write"]}
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
        key_metadata = {"some_other_field": "value"}

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
                {"index_name": "my-index", "index_permissions": ["read"]}
            ]
        }
        team_metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["write"]}
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
                {"index_name": "other-index", "index_permissions": ["read"]}
            ]
        }
        team_metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["write"]}
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
                {"index_name": "index-1", "index_permissions": ["read"]},
                {"index_name": "index-2", "index_permissions": ["write"]},
                {"index_name": "index-3", "index_permissions": ["read", "write"]},
            ]
        }

        # Test each index
        assert (
            check_vector_store_permission("index-1", "read", key_metadata, None) is True
        )
        assert (
            check_vector_store_permission("index-1", "write", key_metadata, None)
            is False
        )
        assert (
            check_vector_store_permission("index-2", "write", key_metadata, None)
            is True
        )
        assert (
            check_vector_store_permission("index-2", "read", key_metadata, None)
            is False
        )
        assert (
            check_vector_store_permission("index-3", "read", key_metadata, None) is True
        )
        assert (
            check_vector_store_permission("index-3", "write", key_metadata, None)
            is True
        )

    def test_invalid_metadata_structure(self):
        """Test handling of invalid metadata structures."""
        # Test when allowed_vector_store_indexes is not a list
        key_metadata = {"allowed_vector_store_indexes": "not-a-list"}

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
                {"index_name": "my-index", "index_permissions": ["read"]},
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
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["read"]}
            ]
        }
        mock_user_api_key.team_metadata = None

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
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
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["write"]}
            ]
        }
        mock_user_api_key.team_metadata = None

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
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
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["read"]}
            ]
        }
        mock_user_api_key.team_metadata = None

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
            with pytest.raises(HTTPException) as exc_info:
                result = is_allowed_to_call_vector_store_endpoint(
                    provider=LlmProviders.OPENAI,
                    index_name="my-index",
                    request=mock_request,
                    user_api_key_dict=mock_user_api_key,
                )

        assert exc_info.value.status_code == 403

    def test_provider_config_not_found(self):
        """Test when provider config is not found."""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/my-index/search"

        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {}
        mock_user_api_key.team_metadata = None

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=None,
        ):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )

        assert result is None

    def test_endpoint_not_recognized(self):
        """Test when endpoint doesn't match any read or write patterns."""
        # Mock request with unrecognized path
        mock_request = MagicMock(spec=Request)
        mock_request.method = "DELETE"
        mock_request.url.path = "/v1/vector_stores/my-index/unknown"

        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["read", "write"]}
            ]
        }
        mock_user_api_key.team_metadata = None

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
            result = is_allowed_to_call_vector_store_endpoint(
                provider=LlmProviders.OPENAI,
                index_name="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )

        assert result is None

    def test_team_metadata_permissions(self):
        """Test that team metadata permissions work."""
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/team-index/search"

        # Mock user API key with no key metadata but team metadata
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = None
        mock_user_api_key.team_metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "team-index", "index_permissions": ["read"]}
            ]
        }

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
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
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {}
        mock_user_api_key.team_metadata = {}

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
            with pytest.raises(HTTPException) as exc_info:
                result = is_allowed_to_call_vector_store_endpoint(
                    provider=LlmProviders.OPENAI,
                    index_name="my-index",
                    request=mock_request,
                    user_api_key_dict=mock_user_api_key,
                )

        assert exc_info.value.status_code == 403

    def test_azure_ai_permission_allowed(self):
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from fastapi import HTTPException

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.utils import LlmProviders

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/azure_ai/indexes/dall-e-4/docs/search"
        mock_user_api_key = UserAPIKeyAuth(
            token="sk-test-mock-token-404",
            key_name="sk-...plNQ",
            metadata={
                "allowed_vector_store_indexes": [
                    {"index_name": "dall-e-4", "index_permissions": ["write"]}
                ]
            },
            spend=0.015,
        )
        mock_provider_config = MagicMock()
        mock_provider_config.get_vector_store_endpoints_by_type.return_value = {
            "read": [("GET", "/docs/search")],
            "write": [("POST", "/create")],
        }

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=mock_provider_config,
        ):
            with pytest.raises(HTTPException) as exc_info:
                is_allowed_to_call_vector_store_endpoint(
                    provider=LlmProviders.AZURE_AI,
                    index_name="dall-e-4",
                    request=mock_request,
                    user_api_key_dict=mock_user_api_key,
                )

        assert exc_info.value.status_code == 403


class TestIsAllowedToCallVectorStoreFilesEndpoint:
    def _mock_provider_config(self):
        provider_config = MagicMock()
        provider_config.get_vector_store_file_endpoints_by_type.return_value = {
            "read": (("GET", "/vector_stores/{vector_store_id}/files"),),
            "write": (
                ("POST", "/vector_stores/{vector_store_id}/files"),
                ("DELETE", "/vector_stores/{vector_store_id}/files/{file_id}"),
            ),
        }
        return provider_config

    def test_allows_access_with_permissions(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/my-index/files"

        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["read", "write"]}
            ]
        }
        mock_user_api_key.team_metadata = None

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_store_files_config",
            return_value=self._mock_provider_config(),
        ):
            result = is_allowed_to_call_vector_store_files_endpoint(
                provider=LlmProviders.OPENAI,
                vector_store_id="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )

        assert result is True

    def test_raises_when_permission_missing(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/v1/vector_stores/my-index/files"

        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {
            "allowed_vector_store_indexes": [
                {"index_name": "my-index", "index_permissions": ["read"]}
            ]
        }
        mock_user_api_key.team_metadata = None

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_store_files_config",
            return_value=self._mock_provider_config(),
        ):
            with pytest.raises(HTTPException) as exc_info:
                is_allowed_to_call_vector_store_files_endpoint(
                    provider=LlmProviders.OPENAI,
                    vector_store_id="my-index",
                    request=mock_request,
                    user_api_key_dict=mock_user_api_key,
                )

        assert exc_info.value.status_code == 403

    def test_returns_none_when_provider_not_supported(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/v1/vector_stores/my-index/files"

        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.user_role = None
        mock_user_api_key.metadata = {}
        mock_user_api_key.team_metadata = {}

        with patch(
            "litellm.proxy.vector_store_endpoints.utils.ProviderConfigManager.get_provider_vector_store_files_config",
            return_value=None,
        ):
            result = is_allowed_to_call_vector_store_files_endpoint(
                provider=LlmProviders.OPENAI,
                vector_store_id="my-index",
                request=mock_request,
                user_api_key_dict=mock_user_api_key,
            )

        assert result is None


class TestVectorStoreManagementEndpointsExist:
    def test_vector_store_management_endpoints_exist_on_proxy_startup(self):
        """
        Test that all vector store management endpoints are registered on proxy app startup.
        
        Verifies the following endpoints exist in the proxy_server app:
        - POST /vector_store/new
        - GET /vector_store/list
        - POST /vector_store/delete
        - POST /vector_store/info
        - POST /vector_store/update
        """
        from litellm.proxy.proxy_server import app

        # Define expected endpoints
        expected_endpoints = [
            ("POST", "/vector_store/new"),
            ("GET", "/vector_store/list"),
            ("POST", "/vector_store/delete"),
            ("POST", "/vector_store/info"),
            ("POST", "/vector_store/update"),
        ]
        
        # Get all routes from the app
        app_routes = []
        for route in app.routes:
            methods = getattr(route, "methods", None)
            path = getattr(route, "path", None)
            if methods is not None and path is not None:
                for method in methods:
                    app_routes.append((method, path))
        
        # Verify each expected endpoint exists
        for method, path in expected_endpoints:
            assert (method, path) in app_routes, (
                f"Expected endpoint {method} {path} not found in registered routes. "
                f"Available routes: {app_routes}"
            )


@pytest.mark.asyncio
async def test_vector_store_synchronization_across_instances():
    """
    Test that vector stores are properly synchronized across multiple instances.
    
    This test simulates the scenario where:
    1. Instance 1 creates a vector store (writes to DB, updates its own cache)
    2. Instance 2 should be able to find it (via database fallback)
    3. Instance 1 deletes the vector store (removes from DB, updates its own cache)
    4. Instance 2 should not show it in the list (database is source of truth)
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from litellm.types.vector_stores import (
        LiteLLM_ManagedVectorStore,
        VectorStoreDeleteRequest,
    )
    from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

    # Simulate two instances with separate in-memory registries
    instance_1_registry = VectorStoreRegistry(vector_stores=[])
    instance_2_registry = VectorStoreRegistry(vector_stores=[])
    
    # Mock database that both instances share
    mock_db_vector_stores = []
    
    async def mock_find_unique(where):
        """Mock find_unique for checking if vector store exists"""
        vector_store_id = where.get("vector_store_id")
        for vs in mock_db_vector_stores:
            if vs.get("vector_store_id") == vector_store_id:
                # Create a simple object that dict() can convert
                class MockVectorStore:
                    def __init__(self, data):
                        for key, value in data.items():
                            setattr(self, key, value)
                        self._data = data
                    
                    def __iter__(self):
                        return iter(self._data.items())
                return MockVectorStore(vs)
        return None
    
    async def mock_find_many(order=None):
        """Mock find_many for listing vector stores"""
        # Return objects that can be converted to dict using dict()
        # The _get_vector_stores_from_db uses dict(vector_store), so we need to make it work
        result = []
        for vs in mock_db_vector_stores:
            # Create a simple object that dict() can convert
            class MockVectorStore:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
                    self._data = data
                
                def __iter__(self):
                    return iter(self._data.items())
            result.append(MockVectorStore(vs))
        return result
    
    async def mock_create(data):
        """Mock create for adding vector store to DB"""
        vector_store = data.copy()
        mock_db_vector_stores.append(vector_store)
        mock_obj = MagicMock()
        mock_obj.model_dump.return_value = vector_store
        for key, value in vector_store.items():
            setattr(mock_obj, key, value)
        return mock_obj
    
    async def mock_delete(where):
        """Mock delete for removing vector store from DB"""
        vector_store_id = where.get("vector_store_id")
        mock_db_vector_stores[:] = [
            vs for vs in mock_db_vector_stores 
            if vs.get("vector_store_id") != vector_store_id
        ]
        return None
    
    # Create mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        side_effect=mock_find_unique
    )
    mock_prisma_client.db.litellm_managedvectorstorestable.find_many = AsyncMock(
        side_effect=mock_find_many
    )
    mock_prisma_client.db.litellm_managedvectorstorestable.create = AsyncMock(
        side_effect=mock_create
    )
    mock_prisma_client.db.litellm_managedvectorstorestable.delete = AsyncMock(
        side_effect=mock_delete
    )
    
    # Test vector store data
    test_vector_store_id = "test-sync-store-001"
    test_vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": test_vector_store_id,
        "custom_llm_provider": "bedrock",
        "vector_store_name": "Test Sync Store",
        "vector_store_description": "Testing synchronization",
        "litellm_params": {
            "vector_store_id": test_vector_store_id,
            "custom_llm_provider": "bedrock",
            "region_name": "us-east-1"
        },
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    
    # Step 1: Create vector store on Instance 1
    # (Simulate what happens in new_vector_store endpoint)
    await mock_prisma_client.db.litellm_managedvectorstorestable.create(
        data=test_vector_store
    )
    instance_1_registry.add_vector_store_to_registry(vector_store=test_vector_store)
    
    # Verify it's in Instance 1's memory
    assert instance_1_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    ) is not None, "Vector store should be in Instance 1's memory"
    
    # Verify it's in the database
    db_store = await mock_prisma_client.db.litellm_managedvectorstorestable.find_unique(
        where={"vector_store_id": test_vector_store_id}
    )
    assert db_store is not None, "Vector store should be in database"
    
    # Step 2: Instance 2 should be able to find it via database fallback
    # (Simulate what happens in pop_vector_stores_to_run_with_db_fallback)
    found_store = await instance_2_registry.get_litellm_managed_vector_store_from_registry_or_db(
        vector_store_id=test_vector_store_id,
        prisma_client=mock_prisma_client
    )
    assert found_store is not None, "Instance 2 should find vector store from database"
    assert found_store.get("vector_store_id") == test_vector_store_id
    
    # Verify it's now cached in Instance 2's memory
    assert instance_2_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    ) is not None, "Vector store should now be cached in Instance 2's memory"
    
    # Step 3: Test that Instance 2 can list vector stores from database
    # (Simulate what happens in list_vector_stores endpoint - using DB as source of truth)
    vector_stores_from_db = await VectorStoreRegistry._get_vector_stores_from_db(
        prisma_client=mock_prisma_client
    )
    
    # Verify vector store appears in the database list
    vector_store_ids = [vs.get("vector_store_id") for vs in vector_stores_from_db]
    assert test_vector_store_id in vector_store_ids, (
        "Instance 2 should see vector store from database"
    )
    
    # Verify the list endpoint logic: only show DB stores (filter out stale cache)
    # This simulates what list_vector_stores does
    db_vector_store_ids = {
        vs.get("vector_store_id") 
        for vs in vector_stores_from_db 
        if vs.get("vector_store_id")
    }
    
    # Instance 2's in-memory cache should only contain stores that exist in DB
    # (This is what the list endpoint cleanup does)
    for vs in list(instance_2_registry.vector_stores):
        vs_id = vs.get("vector_store_id")
        if vs_id and vs_id not in db_vector_store_ids:
            instance_2_registry.delete_vector_store_from_registry(vector_store_id=vs_id)
    
    # After cleanup, instance 2 should still have the vector store (it's in DB)
    assert instance_2_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    ) is not None, "Instance 2 should still have vector store (it exists in DB)"
    
    # Step 4: Delete vector store on Instance 1
    # (Simulate what happens in delete_vector_store endpoint)
    await mock_prisma_client.db.litellm_managedvectorstorestable.delete(
        where={"vector_store_id": test_vector_store_id}
    )
    instance_1_registry.delete_vector_store_from_registry(
        vector_store_id=test_vector_store_id
    )
    
    # Verify it's removed from Instance 1's memory
    assert instance_1_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    ) is None, "Vector store should be removed from Instance 1's memory"
    
    # Verify it's removed from database
    db_store_after_delete = await mock_prisma_client.db.litellm_managedvectorstorestable.find_unique(
        where={"vector_store_id": test_vector_store_id}
    )
    assert db_store_after_delete is None, "Vector store should be removed from database"
    
    # Step 5: Instance 2 should NOT show it in the list (database is source of truth)
    # The list endpoint logic should clean up stale cache entries
    vector_stores_from_db_after_delete = await VectorStoreRegistry._get_vector_stores_from_db(
        prisma_client=mock_prisma_client
    )
    
    # Verify vector store does NOT appear in the database list
    vector_store_ids_after_delete = [vs.get("vector_store_id") for vs in vector_stores_from_db_after_delete]
    assert test_vector_store_id not in vector_store_ids_after_delete, (
        "Deleted vector store should not be in database"
    )
    
    # Simulate list endpoint cleanup logic
    db_vector_store_ids_after_delete = {
        vs.get("vector_store_id") 
        for vs in vector_stores_from_db_after_delete 
        if vs.get("vector_store_id")
    }
    
    # Remove any in-memory vector stores that no longer exist in database
    for vs in list(instance_2_registry.vector_stores):
        vs_id = vs.get("vector_store_id")
        if vs_id and vs_id not in db_vector_store_ids_after_delete:
            instance_2_registry.delete_vector_store_from_registry(vector_store_id=vs_id)
    
    # Verify it was removed from Instance 2's cache
    assert instance_2_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    ) is None, (
        "Deleted vector store should be removed from Instance 2's cache"
    )
    
    # Step 6: Test that using a deleted vector store fails gracefully
    # (Simulate what happens in pop_vector_stores_to_run_with_db_fallback)
    non_default_params = {"vector_store_ids": [test_vector_store_id]}
    vector_stores_to_run = await instance_2_registry.pop_vector_stores_to_run_with_db_fallback(
        non_default_params=non_default_params,
        tools=None,
        prisma_client=mock_prisma_client
    )
    
    assert len(vector_stores_to_run) == 0, (
        "Deleted vector store should not be returned when trying to use it"
    )


@pytest.mark.asyncio
async def test_vector_store_update_and_list_synchronization():
    """
    Test that vector store updates are properly synchronized across multiple instances.
    
    This test simulates the scenario where:
    1. Instance 1 creates a vector store
    2. Instance 2 caches it in memory
    3. Instance 1 updates the vector store in the database
    4. Instance 2 should see the updated data when listing (database is source of truth)
    
    This is a regression test to prevent the bug where Instance 2 would show
    stale cached data instead of the updated database version.
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

    # Simulate two instances with separate in-memory registries
    instance_1_registry = VectorStoreRegistry(vector_stores=[])
    instance_2_registry = VectorStoreRegistry(vector_stores=[])
    
    # Mock database that both instances share
    mock_db_vector_stores = []
    
    async def mock_find_many(order=None):
        """Mock find_many for listing vector stores"""
        result = []
        for vs in mock_db_vector_stores:
            class MockVectorStore:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
                    self._data = data
                
                def __iter__(self):
                    return iter(self._data.items())
            result.append(MockVectorStore(vs))
        return result
    
    async def mock_create(data):
        """Mock create for adding vector store to DB"""
        vector_store = data.copy()
        mock_db_vector_stores.append(vector_store)
        mock_obj = MagicMock()
        mock_obj.model_dump.return_value = vector_store
        return mock_obj
    
    async def mock_update(where, data):
        """Mock update for modifying vector store in DB"""
        vector_store_id = where.get("vector_store_id")
        for i, vs in enumerate(mock_db_vector_stores):
            if vs.get("vector_store_id") == vector_store_id:
                # Update the vector store
                mock_db_vector_stores[i].update(data)
                mock_obj = MagicMock()
                mock_obj.model_dump.return_value = mock_db_vector_stores[i]
                return mock_obj
        raise Exception(f"Vector store {vector_store_id} not found")
    
    # Create mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_managedvectorstorestable.find_many = AsyncMock(
        side_effect=mock_find_many
    )
    mock_prisma_client.db.litellm_managedvectorstorestable.create = AsyncMock(
        side_effect=mock_create
    )
    mock_prisma_client.db.litellm_managedvectorstorestable.update = AsyncMock(
        side_effect=mock_update
    )
    
    # Test vector store data
    test_vector_store_id = "test-update-store-001"
    original_name = "Original Name"
    updated_name = "Updated Name"
    
    test_vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": test_vector_store_id,
        "custom_llm_provider": "bedrock",
        "vector_store_name": original_name,
        "vector_store_description": "Testing update synchronization",
        "litellm_params": {
            "vector_store_id": test_vector_store_id,
            "custom_llm_provider": "bedrock",
            "region_name": "us-east-1"
        },
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    
    # Step 1: Create vector store on Instance 1
    await mock_prisma_client.db.litellm_managedvectorstorestable.create(
        data=test_vector_store
    )
    instance_1_registry.add_vector_store_to_registry(vector_store=test_vector_store)
    
    # Step 2: Instance 2 fetches and caches the vector store
    vector_stores_from_db = await VectorStoreRegistry._get_vector_stores_from_db(
        prisma_client=mock_prisma_client
    )
    for vs in vector_stores_from_db:
        if vs.get("vector_store_id") == test_vector_store_id:
            instance_2_registry.add_vector_store_to_registry(vector_store=vs)
    
    # Verify both instances have the original data
    instance_1_vs = instance_1_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    )
    instance_2_vs = instance_2_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    )
    assert instance_1_vs.get("vector_store_name") == original_name
    assert instance_2_vs.get("vector_store_name") == original_name
    
    # Step 3: Instance 1 updates the vector store in the database
    # (Simulating what happens in update_vector_store endpoint)
    update_data = {"vector_store_name": updated_name}
    await mock_prisma_client.db.litellm_managedvectorstorestable.update(
        where={"vector_store_id": test_vector_store_id},
        data=update_data
    )
    
    # Instance 1 updates its own cache
    updated_vs_instance_1 = test_vector_store.copy()
    updated_vs_instance_1["vector_store_name"] = updated_name
    instance_1_registry.update_vector_store_in_registry(
        vector_store_id=test_vector_store_id,
        updated_data=updated_vs_instance_1
    )
    
    # Verify Instance 1 has the updated data
    instance_1_vs_after_update = instance_1_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    )
    assert instance_1_vs_after_update.get("vector_store_name") == updated_name
    
    # Verify Instance 2 still has stale data in cache
    instance_2_vs_before_list = instance_2_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    )
    assert instance_2_vs_before_list.get("vector_store_name") == original_name, (
        "Instance 2 should still have stale cached data before list operation"
    )
    
    # Step 4: Instance 2 calls list endpoint (which should sync with database)
    # This simulates what list_vector_stores endpoint does
    vector_stores_from_db_after_update = await VectorStoreRegistry._get_vector_stores_from_db(
        prisma_client=mock_prisma_client
    )
    
    # Build map from database vector stores (database is source of truth)
    vector_store_map = {}
    for vector_store in vector_stores_from_db_after_update:
        vector_store_id = vector_store.get("vector_store_id")
        if vector_store_id:
            vector_store_map[vector_store_id] = vector_store
            
            # Update in-memory registry with database versions (this is the key fix)
            instance_2_registry.update_vector_store_in_registry(
                vector_store_id=vector_store_id,
                updated_data=vector_store
            )
    
    # Step 5: Verify Instance 2 now has the updated data
    instance_2_vs_after_list = instance_2_registry.get_litellm_managed_vector_store_from_registry(
        test_vector_store_id
    )
    assert instance_2_vs_after_list.get("vector_store_name") == updated_name, (
        "Instance 2 should have updated data after list operation syncs with database"
    )
    
    # Verify the list returned the correct data
    combined_vector_stores = list(vector_store_map.values())
    assert len(combined_vector_stores) == 1
    assert combined_vector_stores[0].get("vector_store_id") == test_vector_store_id
    assert combined_vector_stores[0].get("vector_store_name") == updated_name, (
        "List should return updated data from database"
    )


@pytest.mark.asyncio
async def test_resolve_embedding_config_from_db():
    """Test that _resolve_embedding_config_from_db correctly resolves embedding config from database."""
    mock_prisma_client = MagicMock()
    
    # Mock database model with litellm_params
    mock_db_model = MagicMock()
    mock_db_model.litellm_params = {
        "api_key": "test-api-key",
        "api_base": "https://api.openai.com",
        "api_version": "2024-01-01"
    }
    
    mock_prisma_client.db.litellm_proxymodeltable.find_first = AsyncMock(
        return_value=mock_db_model
    )
    
    with patch(
        "litellm.proxy.vector_store_endpoints.management_endpoints.decrypt_value_helper",
        side_effect=lambda value, key, return_original_value: value
    ):
        result = await _resolve_embedding_config_from_db(
            embedding_model="text-embedding-ada-002",
            prisma_client=mock_prisma_client
        )
    
    assert result is not None
    assert result["api_key"] == "test-api-key"
    assert result["api_base"] == "https://api.openai.com"
    assert result["api_version"] == "2024-01-01"
    mock_prisma_client.db.litellm_proxymodeltable.find_first.assert_called_once_with(
        where={"model_name": "text-embedding-ada-002"}
    )
    
    # Test with empty embedding_model
    result_empty = await _resolve_embedding_config_from_db(
        embedding_model="",
        prisma_client=mock_prisma_client
    )
    assert result_empty is None
    
    # Test with model not found
    mock_prisma_client.db.litellm_proxymodeltable.find_first = AsyncMock(
        return_value=None
    )
    result_not_found = await _resolve_embedding_config_from_db(
        embedding_model="non-existent-model",
        prisma_client=mock_prisma_client
    )
    assert result_not_found is None


@pytest.mark.asyncio
async def test_new_vector_store_auto_resolves_embedding_config():
    """Test that new_vector_store auto-resolves embedding config when embedding_model is provided but config is not."""
    import json

    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    
    mock_prisma_client = MagicMock()
    
    # Mock vector store request with embedding_model but no embedding_config
    vector_store_data: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "test-store-001",
        "custom_llm_provider": "openai",
        "litellm_params": {
            "litellm_embedding_model": "text-embedding-ada-002",
            # Note: litellm_embedding_config is not provided
        }
    }
    
    # Mock database model lookup for embedding config resolution
    mock_db_model = MagicMock()
    mock_db_model.litellm_params = {
        "api_key": "resolved-api-key",
        "api_base": "https://api.openai.com",
        "api_version": "2024-01-01"
    }
    
    # Mock user API key
    mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key.user_role = None
    mock_user_api_key.team_id = None
    mock_user_api_key.user_id = None
    
    # Mock database operations
    mock_prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=None  # Vector store doesn't exist yet
    )
    mock_prisma_client.db.litellm_proxymodeltable.find_first = AsyncMock(
        return_value=mock_db_model
    )
    
    # Track what was passed to create
    captured_create_data = {}
    
    async def mock_create(*args, **kwargs):
        captured_create_data.update(kwargs.get("data", {}))
        mock_created_vector_store = MagicMock()
        mock_created_vector_store.model_dump.return_value = {
            "vector_store_id": "test-store-001",
            "custom_llm_provider": "openai",
            "litellm_params": kwargs.get("data", {}).get("litellm_params")
        }
        return mock_created_vector_store
    
    mock_prisma_client.db.litellm_managedvectorstorestable.create = AsyncMock(
        side_effect=mock_create
    )
    
    mock_registry = MagicMock()
    mock_registry.add_vector_store_to_registry = MagicMock()
    
    # Mock router to return None (so it falls back to DB resolution)
    mock_router = MagicMock()
    mock_router.get_deployment_by_model_group_name.return_value = None
    
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.llm_router",
        mock_router
    ), patch(
        "litellm.proxy.vector_store_endpoints.management_endpoints.decrypt_value_helper",
        side_effect=lambda value, key, return_original_value: value
    ), patch.object(
        litellm, "vector_store_registry", mock_registry
    ):
        result = await new_vector_store(
            vector_store=vector_store_data,
            user_api_key_dict=mock_user_api_key
        )
    
    assert result["status"] == "success"
    # Verify that embedding config was resolved and included in the create call
    litellm_params_json = captured_create_data.get("litellm_params")
    assert litellm_params_json is not None
    litellm_params_dict = json.loads(litellm_params_json)
    assert "litellm_embedding_config" in litellm_params_dict
    assert litellm_params_dict["litellm_embedding_config"]["api_key"] == "resolved-api-key"
    assert litellm_params_dict["litellm_embedding_config"]["api_base"] == "https://api.openai.com"
    assert litellm_params_dict["litellm_embedding_config"]["api_version"] == "2024-01-01"


def test_resolve_embedding_config_from_router():
    """Test that _resolve_embedding_config_from_router correctly extracts credentials from config-defined models."""
    from litellm.types.router import Deployment, LiteLLM_Params
    
    # Create a mock router with a model
    mock_router = MagicMock()
    
    # Create a mock deployment with litellm_params
    mock_litellm_params = MagicMock(spec=LiteLLM_Params)
    mock_litellm_params.api_key = "config-api-key"
    mock_litellm_params.api_base = "https://config-api-base.com"
    mock_litellm_params.api_version = "2024-02-01"
    
    mock_deployment = MagicMock(spec=Deployment)
    mock_deployment.litellm_params = mock_litellm_params
    
    mock_router.get_deployment_by_model_group_name.return_value = mock_deployment
    
    # Test resolution
    result = _resolve_embedding_config_from_router(
        embedding_model="text-embedding-ada-002",
        llm_router=mock_router
    )
    
    assert result is not None
    assert result["api_key"] == "config-api-key"
    assert result["api_base"] == "https://config-api-base.com"
    assert result["api_version"] == "2024-02-01"
    
    mock_router.get_deployment_by_model_group_name.assert_called_once_with(
        model_group_name="text-embedding-ada-002"
    )


def test_resolve_embedding_config_from_router_with_provider_prefix():
    """Test that _resolve_embedding_config_from_router handles provider prefixes like 'azure/model-name'."""
    from litellm.types.router import Deployment, LiteLLM_Params
    
    # Create a mock router
    mock_router = MagicMock()
    
    # Create a mock deployment
    mock_litellm_params = MagicMock(spec=LiteLLM_Params)
    mock_litellm_params.api_key = "azure-api-key"
    mock_litellm_params.api_base = "https://azure-endpoint.openai.azure.com"
    mock_litellm_params.api_version = "2024-02-15"
    
    mock_deployment = MagicMock(spec=Deployment)
    mock_deployment.litellm_params = mock_litellm_params
    
    # First call with full name returns None, second call with stripped name returns deployment
    mock_router.get_deployment_by_model_group_name.side_effect = [None, mock_deployment]
    
    result = _resolve_embedding_config_from_router(
        embedding_model="azure/text-embedding-3-large",
        llm_router=mock_router
    )
    
    assert result is not None
    assert result["api_key"] == "azure-api-key"
    assert result["api_base"] == "https://azure-endpoint.openai.azure.com"
    assert result["api_version"] == "2024-02-15"
    
    # Should have tried both the full name and stripped name
    assert mock_router.get_deployment_by_model_group_name.call_count == 2


def test_resolve_embedding_config_from_router_returns_none_when_not_found():
    """Test that _resolve_embedding_config_from_router returns None when model is not in router."""
    mock_router = MagicMock()
    mock_router.get_deployment_by_model_group_name.return_value = None
    
    result = _resolve_embedding_config_from_router(
        embedding_model="nonexistent-model",
        llm_router=mock_router
    )
    
    assert result is None


def test_resolve_embedding_config_from_router_handles_os_environ():
    """Test that _resolve_embedding_config_from_router handles os.environ/ prefixed values."""
    from litellm.types.router import Deployment, LiteLLM_Params
    
    mock_router = MagicMock()
    
    mock_litellm_params = MagicMock(spec=LiteLLM_Params)
    mock_litellm_params.api_key = "os.environ/OPENAI_API_KEY"
    mock_litellm_params.api_base = "https://direct-url.com"
    mock_litellm_params.api_version = None
    
    mock_deployment = MagicMock(spec=Deployment)
    mock_deployment.litellm_params = mock_litellm_params
    
    mock_router.get_deployment_by_model_group_name.return_value = mock_deployment
    
    with patch(
        "litellm.proxy.vector_store_endpoints.management_endpoints.get_secret",
        return_value="resolved-from-env"
    ) as mock_get_secret:
        result = _resolve_embedding_config_from_router(
            embedding_model="text-embedding-ada-002",
            llm_router=mock_router
        )
    
    assert result is not None
    assert result["api_key"] == "resolved-from-env"
    assert result["api_base"] == "https://direct-url.com"
    assert "api_version" not in result
    
    mock_get_secret.assert_called_once_with("os.environ/OPENAI_API_KEY")


@pytest.mark.asyncio
async def test_resolve_embedding_config_tries_router_then_db():
    """Test that _resolve_embedding_config tries router first, then falls back to DB."""
    from litellm.types.router import Deployment, LiteLLM_Params
    
    mock_prisma_client = MagicMock()
    mock_router = MagicMock()
    
    # Router has the model
    mock_litellm_params = MagicMock(spec=LiteLLM_Params)
    mock_litellm_params.api_key = "router-api-key"
    mock_litellm_params.api_base = "https://router-api-base.com"
    mock_litellm_params.api_version = None
    
    mock_deployment = MagicMock(spec=Deployment)
    mock_deployment.litellm_params = mock_litellm_params
    
    mock_router.get_deployment_by_model_group_name.return_value = mock_deployment
    
    # DB should NOT be called since router has the model
    mock_prisma_client.db.litellm_proxymodeltable.find_first = AsyncMock()
    
    result = await _resolve_embedding_config(
        embedding_model="text-embedding-ada-002",
        prisma_client=mock_prisma_client,
        llm_router=mock_router
    )
    
    assert result is not None
    assert result["api_key"] == "router-api-key"
    
    # DB should NOT have been called since router found the model
    mock_prisma_client.db.litellm_proxymodeltable.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_embedding_config_falls_back_to_db():
    """Test that _resolve_embedding_config falls back to DB when router doesn't have the model."""
    mock_prisma_client = MagicMock()
    mock_router = MagicMock()
    
    # Router doesn't have the model
    mock_router.get_deployment_by_model_group_name.return_value = None
    
    # DB has the model
    mock_db_model = MagicMock()
    mock_db_model.litellm_params = {
        "api_key": "db-api-key",
        "api_base": "https://db-api-base.com",
    }
    mock_prisma_client.db.litellm_proxymodeltable.find_first = AsyncMock(
        return_value=mock_db_model
    )
    
    with patch(
        "litellm.proxy.vector_store_endpoints.management_endpoints.decrypt_value_helper",
        side_effect=lambda value, key, return_original_value: value
    ):
        result = await _resolve_embedding_config(
            embedding_model="text-embedding-ada-002",
            prisma_client=mock_prisma_client,
            llm_router=mock_router
        )
    
    assert result is not None
    assert result["api_key"] == "db-api-key"
    
    # DB should have been called since router didn't find the model
    mock_prisma_client.db.litellm_proxymodeltable.find_first.assert_called()


@pytest.mark.asyncio
async def test_new_vector_store_auto_resolves_from_router():
    """Test that new_vector_store auto-resolves embedding config from router when model is config-defined."""
    import json

    from litellm.types.router import Deployment, LiteLLM_Params
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    
    mock_prisma_client = MagicMock()
    
    # Mock vector store request with embedding_model but no embedding_config
    vector_store_data: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "test-store-router-001",
        "custom_llm_provider": "openai",
        "litellm_params": {
            "litellm_embedding_model": "config-embedding-model",
            # Note: litellm_embedding_config is not provided
        }
    }
    
    # Mock router with the model
    mock_router = MagicMock()
    mock_litellm_params = MagicMock(spec=LiteLLM_Params)
    mock_litellm_params.api_key = "router-resolved-api-key"
    mock_litellm_params.api_base = "https://router-resolved-base.com"
    mock_litellm_params.api_version = "2024-03-01"
    
    mock_deployment = MagicMock(spec=Deployment)
    mock_deployment.litellm_params = mock_litellm_params
    
    mock_router.get_deployment_by_model_group_name.return_value = mock_deployment
    
    # Mock user API key
    mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key.user_role = None
    mock_user_api_key.team_id = None
    mock_user_api_key.user_id = None
    
    # Mock database operations
    mock_prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=None  # Vector store doesn't exist yet
    )
    
    # Track what was passed to create
    captured_create_data = {}
    
    async def mock_create(*args, **kwargs):
        captured_create_data.update(kwargs.get("data", {}))
        mock_created_vector_store = MagicMock()
        mock_created_vector_store.model_dump.return_value = {
            "vector_store_id": "test-store-router-001",
            "custom_llm_provider": "openai",
            "litellm_params": kwargs.get("data", {}).get("litellm_params")
        }
        return mock_created_vector_store
    
    mock_prisma_client.db.litellm_managedvectorstorestable.create = AsyncMock(
        side_effect=mock_create
    )
    
    mock_registry = MagicMock()
    mock_registry.add_vector_store_to_registry = MagicMock()
    
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client
    ), patch(
        "litellm.proxy.proxy_server.llm_router",
        mock_router
    ), patch.object(
        litellm, "vector_store_registry", mock_registry
    ):
        result = await new_vector_store(
            vector_store=vector_store_data,
            user_api_key_dict=mock_user_api_key
        )
    
    assert result["status"] == "success"
    # Verify that embedding config was resolved from router and included in the create call
    litellm_params_json = captured_create_data.get("litellm_params")
    assert litellm_params_json is not None
    litellm_params_dict = json.loads(litellm_params_json)
    assert "litellm_embedding_config" in litellm_params_dict
    assert litellm_params_dict["litellm_embedding_config"]["api_key"] == "router-resolved-api-key"
    assert litellm_params_dict["litellm_embedding_config"]["api_base"] == "https://router-resolved-base.com"
    assert litellm_params_dict["litellm_embedding_config"]["api_version"] == "2024-03-01"


class TestCheckVectorStoreAccess:
    """Test suite for _check_vector_store_access function."""

    def test_access_granted_when_no_team_id(self):
        """Test that access is granted when vector store has no team_id (legacy behavior)."""
        vector_store: LiteLLM_ManagedVectorStore = {
            "vector_store_id": "test-store",
            "custom_llm_provider": "openai",
            # No team_id field
        }
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.team_id = "team-123"
        
        result = _check_vector_store_access(vector_store, mock_user_api_key)
        assert result is True

    def test_access_granted_when_team_ids_match(self):
        """Test that access is granted when user's team_id matches vector store's team_id."""
        vector_store: LiteLLM_ManagedVectorStore = {
            "vector_store_id": "test-store",
            "custom_llm_provider": "openai",
            "team_id": "team-123",
        }
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.team_id = "team-123"
        
        result = _check_vector_store_access(vector_store, mock_user_api_key)
        assert result is True

    def test_access_denied_when_team_ids_dont_match(self):
        """Test that access is denied when user's team_id doesn't match vector store's team_id."""
        vector_store: LiteLLM_ManagedVectorStore = {
            "vector_store_id": "test-store",
            "custom_llm_provider": "openai",
            "team_id": "team-123",
        }
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.team_id = "team-456"
        
        result = _check_vector_store_access(vector_store, mock_user_api_key)
        assert result is False

    def test_access_denied_when_vector_store_has_team_id_but_user_doesnt(self):
        """Test that access is denied when vector store has team_id but user doesn't."""
        vector_store: LiteLLM_ManagedVectorStore = {
            "vector_store_id": "test-store",
            "custom_llm_provider": "openai",
            "team_id": "team-123",
        }
        
        mock_user_api_key = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key.team_id = None
        
        result = _check_vector_store_access(vector_store, mock_user_api_key)
        assert result is False


@pytest.mark.asyncio
async def test_create_vector_store_in_db():
    """Test that create_vector_store_in_db correctly creates a vector store in the database."""
    from datetime import datetime, timezone
    
    mock_prisma_client = MagicMock()
    
    # Mock vector store data
    vector_store_id = "test-create-store-001"
    custom_llm_provider = "openai"
    vector_store_name = "Test Store"
    vector_store_description = "Test Description"
    vector_store_metadata = {"key": "value"}
    litellm_params = {"api_key": "test-key"}
    team_id = "team-123"
    user_id = "user-456"
    
    # Mock database operations
    mock_prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=None  # Vector store doesn't exist yet
    )
    
    created_vector_store_data = {
        "vector_store_id": vector_store_id,
        "custom_llm_provider": custom_llm_provider,
        "vector_store_name": vector_store_name,
        "vector_store_description": vector_store_description,
        "vector_store_metadata": '{"key": "value"}',
        "litellm_params": '{"api_key": "test-key"}',
        "team_id": team_id,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    
    mock_created_vector_store = MagicMock()
    mock_created_vector_store.model_dump.return_value = created_vector_store_data
    
    mock_prisma_client.db.litellm_managedvectorstorestable.create = AsyncMock(
        return_value=mock_created_vector_store
    )
    
    mock_registry = MagicMock()
    mock_registry.add_vector_store_to_registry = MagicMock()
    
    with patch.object(litellm, "vector_store_registry", mock_registry):
        result = await create_vector_store_in_db(
            vector_store_id=vector_store_id,
            custom_llm_provider=custom_llm_provider,
            prisma_client=mock_prisma_client,
            vector_store_name=vector_store_name,
            vector_store_description=vector_store_description,
            vector_store_metadata=vector_store_metadata,
            litellm_params=litellm_params,
            team_id=team_id,
            user_id=user_id,
        )
    
    # Verify the result
    assert result is not None
    assert result["vector_store_id"] == vector_store_id
    assert result["custom_llm_provider"] == custom_llm_provider
    
    # Verify database was called correctly
    mock_prisma_client.db.litellm_managedvectorstorestable.find_unique.assert_called_once_with(
        where={"vector_store_id": vector_store_id}
    )
    mock_prisma_client.db.litellm_managedvectorstorestable.create.assert_called_once()
    
    # Verify registry was updated
    mock_registry.add_vector_store_to_registry.assert_called_once()
    
    # Verify that create was called with correct data structure
    create_call_args = mock_prisma_client.db.litellm_managedvectorstorestable.create.call_args
    create_data = create_call_args.kwargs.get("data", {})
    assert create_data["vector_store_id"] == vector_store_id
    assert create_data["custom_llm_provider"] == custom_llm_provider
    assert create_data["vector_store_name"] == vector_store_name
    assert create_data["vector_store_description"] == vector_store_description
    assert create_data["team_id"] == team_id
    assert create_data["user_id"] == user_id


@pytest.mark.asyncio
async def test_create_vector_store_in_db_raises_when_exists():
    """Test that create_vector_store_in_db raises HTTPException when vector store already exists."""
    mock_prisma_client = MagicMock()
    
    vector_store_id = "existing-store"
    
    # Mock that vector store already exists
    existing_vector_store = MagicMock()
    mock_prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=existing_vector_store
    )
    
    with pytest.raises(HTTPException) as exc_info:
        await create_vector_store_in_db(
            vector_store_id=vector_store_id,
            custom_llm_provider="openai",
            prisma_client=mock_prisma_client,
        )
    
    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.detail.lower()
    
    # Verify create was not called
    mock_prisma_client.db.litellm_managedvectorstorestable.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_vector_store_in_db_raises_when_no_db():
    """Test that create_vector_store_in_db raises HTTPException when database is not connected."""
    with pytest.raises(HTTPException) as exc_info:
        await create_vector_store_in_db(
            vector_store_id="test-store",
            custom_llm_provider="openai",
            prisma_client=None,
        )
    
    assert exc_info.value.status_code == 500
    assert "database not connected" in exc_info.value.detail.lower()
