"""
Tests for the config-sourced vector store fix.

Regression: `/vector_store/list` used to prune config.yaml-loaded stores from
the in-memory registry because they have no DB row. The fix tracks
`config_loaded_vector_store_ids` on the registry and exempts them from the
"in memory but not in DB" reconciliation branch.
"""
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


def _config_entry(vs_id: str) -> dict:
    return {
        "vector_store_name": f"name-{vs_id}",
        "litellm_params": {
            "vector_store_id": vs_id,
            "custom_llm_provider": "bedrock",
        },
    }


def test_load_from_config_tracks_id():
    registry = VectorStoreRegistry()
    assert registry.config_loaded_vector_store_ids == set()

    registry.load_vector_stores_from_config([_config_entry("vs_a"), _config_entry("vs_b")])

    assert {vs["vector_store_id"] for vs in registry.vector_stores} == {"vs_a", "vs_b"}
    assert registry.config_loaded_vector_store_ids == {"vs_a", "vs_b"}


def test_delete_drops_config_marker():
    """After a config-loaded store is explicitly deleted, the marker must go
    with it so a later reload can re-create the entry without confusion."""
    registry = VectorStoreRegistry()
    registry.load_vector_stores_from_config([_config_entry("vs_a")])
    assert "vs_a" in registry.config_loaded_vector_store_ids

    registry.delete_vector_store_from_registry("vs_a")

    assert registry.vector_stores == []
    assert "vs_a" not in registry.config_loaded_vector_store_ids


def test_delete_of_db_only_store_does_not_touch_marker_set():
    registry = VectorStoreRegistry()
    registry.load_vector_stores_from_config([_config_entry("vs_config")])
    # Simulate a DB-sourced entry being added via /vector_store/new
    registry.add_vector_store_to_registry(
        {"vector_store_id": "vs_from_db", "custom_llm_provider": "openai"}
    )

    registry.delete_vector_store_from_registry("vs_from_db")

    assert [vs["vector_store_id"] for vs in registry.vector_stores] == ["vs_config"]
    # Config marker for the untouched entry is intact.
    assert registry.config_loaded_vector_store_ids == {"vs_config"}
