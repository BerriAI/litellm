import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    LiteLLM_ManagedVectorStore,
)
from litellm.proxy.vector_store_endpoints.endpoints import (
    _update_request_data_with_litellm_managed_vector_store_registry,
)


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
