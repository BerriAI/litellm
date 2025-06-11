import json
import os
import sys
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from datetime import datetime, timezone

from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


def test_get_credentials_for_vector_store():
    """Test that get_credentials_for_vector_store returns correct credentials"""
    # Create test vector stores
    vector_store_1 = LiteLLM_ManagedVectorStore(
        vector_store_id="test_id_1",
        custom_llm_provider="openai",
        vector_store_name="test_store_1",
        litellm_credential_name="test_creds_1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    vector_store_2 = LiteLLM_ManagedVectorStore(
        vector_store_id="test_id_2",
        custom_llm_provider="bedrockc",
        vector_store_name="test_store_2",
        litellm_credential_name="test_creds_2",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create registry with vector stores
    registry = VectorStoreRegistry([vector_store_1, vector_store_2])

    # Mock CredentialAccessor.get_credential_values
    with patch(
        "litellm.litellm_core_utils.credential_accessor.CredentialAccessor.get_credential_values"
    ) as mock_get_creds:
        mock_get_creds.return_value = {"api_key": "test_key_1", "env": "test"}

        # Test getting credentials for existing vector store
        result = registry.get_credentials_for_vector_store("test_id_1")

        assert result == {"api_key": "test_key_1", "env": "test"}
        mock_get_creds.assert_called_once_with("test_creds_1")

    # Test getting credentials for non-existent vector store
    result = registry.get_credentials_for_vector_store("non_existent_id")
    assert result == {}


def test_add_vector_store_to_registry():
    """Test that add_vector_store_to_registry adds vector store correctly when there are pre-existing stores"""
    # Create pre-existing vector stores
    existing_store_1 = LiteLLM_ManagedVectorStore(
        vector_store_id="existing_id_1",
        custom_llm_provider="openai",
        vector_store_name="existing_store_1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    existing_store_2 = LiteLLM_ManagedVectorStore(
        vector_store_id="existing_id_2",
        custom_llm_provider="openai",
        vector_store_name="existing_store_2",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create registry with pre-existing stores
    registry = VectorStoreRegistry([existing_store_1, existing_store_2])
    assert len(registry.vector_stores) == 2

    # Add a new vector store
    new_store = LiteLLM_ManagedVectorStore(
        vector_store_id="new_id",
        custom_llm_provider="bedrock",
        vector_store_name="new_store",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    registry.add_vector_store_to_registry(new_store)

    # Verify new store was added
    assert len(registry.vector_stores) == 3
    assert registry.vector_stores[2]["vector_store_id"] == "new_id"
    assert registry.vector_stores[2]["vector_store_name"] == "new_store"

    # Try to add duplicate - should not be added
    duplicate_store = LiteLLM_ManagedVectorStore(
        vector_store_id="existing_id_1",  # Same ID as existing store
        custom_llm_provider="different_provider",
        vector_store_name="duplicate_store",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    registry.add_vector_store_to_registry(duplicate_store)

    # Verify duplicate was not added
    assert len(registry.vector_stores) == 3
    # Original store should still be there unchanged
    assert registry.vector_stores[0]["vector_store_name"] == "existing_store_1"
