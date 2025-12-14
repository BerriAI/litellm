import pytest

from litellm.proxy.mcp_semantic_filter.settings import SemanticFilterVectorStoreConfig
from litellm.proxy.mcp_semantic_filter.stores.base import ToolVectorRecord
from litellm.proxy.mcp_semantic_filter.stores import faiss_store

pytestmark = pytest.mark.skipif(
    faiss_store._faiss_error is not None,
    reason="faiss backend not available",
)


def _make_store(metric: str = "ip"):
    config = SemanticFilterVectorStoreConfig(metric=metric)
    return faiss_store.FaissVectorStore(config)


def test_upsert_and_query_ip_metric():
    store = _make_store(metric="ip")
    store.upsert_records(
        [
            ToolVectorRecord("toolA", [1.0, 0.0, 0.0]),
            ToolVectorRecord("toolB", [0.5, 0.5, 0.0]),
        ]
    )

    results = store.query([1.0, 0.0, 0.0], top_k=2)
    assert [tool_id for tool_id, _ in results] == ["toolA", "toolB"]


def test_remove_records_updates_index():
    store = _make_store()
    store.upsert_records(
        [
            ToolVectorRecord("toolA", [1.0, 0.0]),
            ToolVectorRecord("toolB", [0.0, 1.0]),
        ]
    )

    store.remove_records(["toolA"])
    results = store.query([1.0, 0.0], top_k=2)
    assert [tool_id for tool_id, _ in results] == ["toolB"]


def test_dimension_mismatch_raises_value_error():
    store = _make_store()
    store.upsert_records([ToolVectorRecord("toolA", [1.0, 0.0, 0.0])])

    with pytest.raises(ValueError):
        store.upsert_records([ToolVectorRecord("toolB", [0.1, 0.2])])
