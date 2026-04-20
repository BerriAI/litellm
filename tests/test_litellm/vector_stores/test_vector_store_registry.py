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
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.main import search
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


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
            "has_more": False
        }
        
        with patch.object(
            registry,
            "get_credentials_for_vector_store",
            return_value={"aws_access_key_id": "ABC", "aws_secret_access_key": "DEF", "aws_region_name": "us-east-1"},
        ) as mock_get_creds, patch(
            "litellm.vector_stores.main.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ), patch.object(
            vector_stores_main.base_llm_http_handler,
            "vector_store_search_handler",
            return_value=mock_search_response,
        ) as mock_handler:
            search(vector_store_id="vs1", query="test", litellm_logging_obj=logger)
            mock_get_creds.assert_called_once_with("vs1")
            called_params = mock_handler.call_args.kwargs["litellm_params"]
            assert getattr(called_params, "aws_access_key_id") == "ABC"
            assert getattr(called_params, "aws_secret_access_key") == "DEF"
            assert getattr(called_params, "aws_region_name") == "us-east-1"
    finally:
        litellm.vector_store_registry = original_registry


def test_load_vector_stores_from_config_marks_from_litellm_config():
    """
    load_vector_stores_from_config must tag each loaded entry with
    from_litellm_config=True so downstream code can distinguish
    config-declared stores from DB/runtime stores (issue #25947).
    """
    # Pass an explicit empty list — VectorStoreRegistry's __init__ has a
    # mutable default argument ([]) that is shared across instances within
    # the same process.
    registry = VectorStoreRegistry(vector_stores=[])
    registry.load_vector_stores_from_config(
        [
            {
                "vector_store_name": "demo",
                "litellm_params": {
                    "custom_llm_provider": "pg_vector",
                    "vector_store_id": "vs-demo",
                },
            }
        ]
    )

    assert len(registry.vector_stores) == 1
    loaded = registry.vector_stores[0]
    assert loaded["vector_store_id"] == "vs-demo"
    assert loaded.get("from_litellm_config") is True


def test_add_vector_store_to_registry_does_not_mark_from_litellm_config():
    """
    Programmatic inserts (DB reload, API create) must NOT be tagged
    as from_litellm_config. Only entries declared in proxy_config.yaml
    should carry the flag.
    """
    registry = VectorStoreRegistry(vector_stores=[])
    registry.add_vector_store_to_registry(
        LiteLLM_ManagedVectorStore(
            vector_store_id="vs-runtime",
            custom_llm_provider="openai",
            vector_store_name="runtime-store",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    assert len(registry.vector_stores) == 1
    assert registry.vector_stores[0].get("from_litellm_config") is not True


@pytest.mark.asyncio
async def test_pop_vector_stores_preserves_config_loaded_on_db_miss():
    """
    Request-path regression for #25947 (Greptile follow-up): a
    vector store loaded from proxy_config.yaml must still be usable
    during inference when it is missing from the DB — the DB-reconcile
    eviction that applies to runtime entries must not fire for
    config-declared ones, or they get listed in the UI but silently
    evicted on first use.
    """
    registry = VectorStoreRegistry(vector_stores=[])
    registry.load_vector_stores_from_config(
        [
            {
                "vector_store_name": "config-demo",
                "litellm_params": {
                    "custom_llm_provider": "pg_vector",
                    "vector_store_id": "vs-config-path",
                },
            }
        ]
    )

    prisma = MagicMock()
    # Any DB lookup returns "not found" — mimics a proxy running without
    # the managed vector_store table populated.
    prisma.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=None
    )

    with patch.object(
        registry,
        "get_litellm_managed_vector_store_from_registry_or_db",
        new=AsyncMock(return_value=None),
    ):
        result = await registry.pop_vector_stores_to_run_with_db_fallback(
            non_default_params={"vector_store_ids": ["vs-config-path"]},
            prisma_client=prisma,
        )

    assert len(result) == 1
    assert result[0]["vector_store_id"] == "vs-config-path"
    # Still in the registry — not evicted by the DB-miss reconcile.
    ids_after = [vs.get("vector_store_id") for vs in registry.vector_stores]
    assert "vs-config-path" in ids_after
    # DB path should not have been consulted for a config-declared store —
    # we know it will never be there, so asking wastes a round-trip.
    prisma.db.litellm_managedvectorstorestable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_pop_vector_stores_still_evicts_runtime_on_db_miss():
    """
    Regression guard: runtime-inserted entries (no from_litellm_config
    flag) must keep the old eviction semantics, otherwise stale cache
    after a runtime deletion would never be cleaned up.
    """
    from unittest.mock import AsyncMock as _AsyncMock

    registry = VectorStoreRegistry(vector_stores=[])
    registry.add_vector_store_to_registry(
        LiteLLM_ManagedVectorStore(
            vector_store_id="vs-runtime-gone",
            custom_llm_provider="openai",
            vector_store_name="runtime-gone",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    prisma = MagicMock()
    prisma.db.litellm_managedvectorstorestable.find_unique = _AsyncMock(
        return_value=None
    )

    with patch.object(
        registry,
        "get_litellm_managed_vector_store_from_registry_or_db",
        new=_AsyncMock(return_value=None),
    ):
        await registry.pop_vector_stores_to_run_with_db_fallback(
            non_default_params={"vector_store_ids": ["vs-runtime-gone"]},
            prisma_client=prisma,
        )

    # Evicted from the in-memory registry, as before the fix.
    ids_after = [vs.get("vector_store_id") for vs in registry.vector_stores]
    assert "vs-runtime-gone" not in ids_after
    prisma.db.litellm_managedvectorstorestable.find_unique.assert_called_once()
