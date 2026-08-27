import json
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient


from datetime import datetime, timezone
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# config-loaded marker bookkeeping (fixes: /vector_store/list used to prune
# config.yaml-sourced entries because they have no DB row).
# ---------------------------------------------------------------------------


def _cfg_entry(vs_id: str) -> dict:
    return {
        "vector_store_name": f"name-{vs_id}",
        "litellm_params": {
            "vector_store_id": vs_id,
            "custom_llm_provider": "bedrock",
        },
    }


def test_load_from_config_tracks_marker_ids():
    registry = VectorStoreRegistry(vector_stores=[])
    assert registry.config_loaded_vector_store_ids == set()

    registry.load_vector_stores_from_config([_cfg_entry("vs_a"), _cfg_entry("vs_b")])

    assert {vs["vector_store_id"] for vs in registry.vector_stores} == {"vs_a", "vs_b"}
    assert registry.config_loaded_vector_store_ids == {"vs_a", "vs_b"}


def test_delete_drops_config_marker():
    """An explicit delete of a config-loaded id must also clear the marker
    so a later reload can re-load the entry without being incorrectly protected."""
    registry = VectorStoreRegistry(vector_stores=[])
    registry.load_vector_stores_from_config([_cfg_entry("vs_a")])
    assert "vs_a" in registry.config_loaded_vector_store_ids

    registry.delete_vector_store_from_registry("vs_a")

    assert registry.vector_stores == []
    assert "vs_a" not in registry.config_loaded_vector_store_ids


def test_delete_of_db_only_store_does_not_touch_marker_set():
    registry = VectorStoreRegistry(vector_stores=[])
    registry.load_vector_stores_from_config([_cfg_entry("vs_config")])
    registry.add_vector_store_to_registry(
        LiteLLM_ManagedVectorStore(
            vector_store_id="vs_from_db",
            custom_llm_provider="openai",
            vector_store_name="db-store",
            litellm_credential_name=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    registry.delete_vector_store_from_registry("vs_from_db")

    assert [vs["vector_store_id"] for vs in registry.vector_stores] == ["vs_config"]
    assert registry.config_loaded_vector_store_ids == {"vs_config"}


def test_reload_removes_entries_dropped_from_config():
    """Reloading config with a store removed should evict it from both the
    registry and the marker set — otherwise a stale marker would permanently
    protect the deleted store from the list-endpoint reconciliation."""
    registry = VectorStoreRegistry(vector_stores=[])
    registry.load_vector_stores_from_config([_cfg_entry("vs_a"), _cfg_entry("vs_b")])
    assert registry.config_loaded_vector_store_ids == {"vs_a", "vs_b"}

    # Simulate config reload with vs_b removed from config.yaml.
    registry.load_vector_stores_from_config([_cfg_entry("vs_a")])

    assert registry.config_loaded_vector_store_ids == {"vs_a"}
    assert [vs["vector_store_id"] for vs in registry.vector_stores] == ["vs_a"]


def test_reload_is_idempotent_for_unchanged_config():
    """Re-loading the same config must not create duplicate registry entries."""
    registry = VectorStoreRegistry(vector_stores=[])
    registry.load_vector_stores_from_config([_cfg_entry("vs_a")])
    registry.load_vector_stores_from_config([_cfg_entry("vs_a")])

    assert [vs["vector_store_id"] for vs in registry.vector_stores] == ["vs_a"]
    assert registry.config_loaded_vector_store_ids == {"vs_a"}
