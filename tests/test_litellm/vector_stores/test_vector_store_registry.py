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
from unittest.mock import MagicMock, patch

import litellm
from litellm.types.vector_stores import (
    IndexCreateLiteLLMParams,
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreIndex,
)
from litellm.vector_stores.main import search
from litellm.vector_stores.vector_store_registry import (
    VectorStoreIndexRegistry,
    VectorStoreRegistry,
)


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


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


def test_search_uses_registry_credentials():
    """search() should pull credentials from vector_store_registry when available"""
    # Import the module to get the actual handler instance
    import litellm.vector_stores.main as vector_stores_main

    vector_store = LiteLLM_ManagedVectorStore(
        vector_store_id="vs1",
        custom_llm_provider="bedrock",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    registry = VectorStoreRegistry([vector_store])
    original_registry = getattr(litellm, "vector_store_registry", None)
    litellm.vector_store_registry = registry
    try:
        logger = MagicMock()
        logger._response_cost_calculator.return_value = 0

        # Mock the search response
        mock_search_response = {
            "object": "list",
            "data": [],
            "first_id": None,
            "last_id": None,
            "has_more": False,
        }

        with (
            patch.object(
                registry,
                "get_credentials_for_vector_store",
                return_value={
                    "aws_access_key_id": "ABC",
                    "aws_secret_access_key": "DEF",
                    "aws_region_name": "us-east-1",
                },
            ) as mock_get_creds,
            patch(
                "litellm.vector_stores.main.ProviderConfigManager.get_provider_vector_stores_config",
                return_value=MagicMock(),
            ),
            patch.object(
                vector_stores_main.base_llm_http_handler,
                "vector_store_search_handler",
                return_value=mock_search_response,
            ) as mock_handler,
        ):
            search(vector_store_id="vs1", query="test", litellm_logging_obj=logger)
            mock_get_creds.assert_called_once_with("vs1")
            called_params = mock_handler.call_args.kwargs["litellm_params"]
            assert getattr(called_params, "aws_access_key_id") == "ABC"
            assert getattr(called_params, "aws_secret_access_key") == "DEF"
            assert getattr(called_params, "aws_region_name") == "us-east-1"
    finally:
        litellm.vector_store_registry = original_registry


def _managed_vector_store(vector_store_id: str) -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(
        vector_store_id=vector_store_id,
        custom_llm_provider="openai",
        vector_store_name=f"store_{vector_store_id}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _managed_vector_store_index(index_name: str) -> LiteLLM_ManagedVectorStoreIndex:
    return LiteLLM_ManagedVectorStoreIndex(
        id=f"id_{index_name}",
        index_name=index_name,
        litellm_params=IndexCreateLiteLLMParams(
            vector_store_index=index_name,
            vector_store_name=f"store_{index_name}",
        ),
    )


def test_registries_do_not_share_default_storage():
    """
    A registry built without arguments must own its own storage.

    Regression test: both registries used a mutable `= []` default, so the list was created once at
    def-time and shared by every no-arg instance. Mutating one registry leaked into the next one,
    which on the proxy meant a fresh registry inherited another's vector stores (and their
    credential references).
    """
    first_registry = VectorStoreRegistry()
    first_registry.add_vector_store_to_registry(_managed_vector_store("vs_leak"))
    assert first_registry.vector_stores[0]["vector_store_id"] == "vs_leak"

    assert VectorStoreRegistry().vector_stores == []

    first_index_registry = VectorStoreIndexRegistry()
    first_index_registry.upsert_vector_store_index(_managed_vector_store_index("idx_leak"))
    assert first_index_registry.vector_store_indexes[0].index_name == "idx_leak"

    assert VectorStoreIndexRegistry().vector_store_indexes == []


def test_registries_do_not_alias_caller_list():
    """
    The constructor must copy the caller's list rather than alias it.

    Otherwise registry mutations write back into a list the caller still owns; the proxy passes
    lists freshly read from the DB, and `load_vector_stores_from_config` appends into whatever it
    was handed.
    """
    caller_vector_stores = [_managed_vector_store("vs_1")]
    registry = VectorStoreRegistry(caller_vector_stores)
    registry.add_vector_store_to_registry(_managed_vector_store("vs_2"))

    assert len(registry.vector_stores) == 2
    assert len(caller_vector_stores) == 1

    caller_indexes = [_managed_vector_store_index("idx_1")]
    index_registry = VectorStoreIndexRegistry(caller_indexes)
    index_registry.upsert_vector_store_index(_managed_vector_store_index("idx_2"))

    assert len(index_registry.vector_store_indexes) == 2
    assert len(caller_indexes) == 1


def test_delete_vector_store_index_removes_by_name():
    """
    Regression test: delete_vector_store_index compared each stored index object against the
    index-name string, which is never equal, so the delete was an unconditional no-op.
    """
    registry = VectorStoreIndexRegistry(
        [_managed_vector_store_index("idx_a"), _managed_vector_store_index("idx_b")]
    )

    registry.delete_vector_store_index("idx_a")

    assert [index.index_name for index in registry.vector_store_indexes] == ["idx_b"]
    assert registry.get_vector_store_index_by_name("idx_a") is None
    assert registry.is_vector_store_index("idx_b") is True

    registry.delete_vector_store_index("idx_does_not_exist")
    assert [index.index_name for index in registry.vector_store_indexes] == ["idx_b"]


def test_get_vector_store_ids_from_tool_calls_does_not_share_or_mutate():
    """
    Regression test: _get_vector_store_ids_from_tool_calls took `vector_store_ids: List[str] = []`
    and extended it in place, so the def-time default accumulated ids across every call that
    omitted the argument, and a caller-supplied list was mutated behind the caller's back.
    """
    registry = VectorStoreRegistry()
    tools = [{"vector_store_ids": ["vs_a"]}]

    first = registry._get_vector_store_ids_from_tool_calls(tools=tools)
    second = registry._get_vector_store_ids_from_tool_calls(tools=tools)

    assert first == ["vs_a"]
    assert second == ["vs_a"]

    caller_ids = ["vs_seed"]
    result = registry._get_vector_store_ids_from_tool_calls(tools=tools, vector_store_ids=caller_ids)

    assert result == ["vs_seed", "vs_a"]
    assert caller_ids == ["vs_seed"]
