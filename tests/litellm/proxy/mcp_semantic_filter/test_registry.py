import pytest
from types import SimpleNamespace

from litellm.proxy.mcp_semantic_filter.registry import SemanticMCPFilterRegistry
from litellm.proxy.mcp_semantic_filter.settings import (
    SemanticFilterConfig,
    SemanticFilterEmbeddingConfig,
    SemanticFilterVectorStoreConfig,
)


class DummyStore:
    def __init__(self):
        self.upserted_batches = []
        self.removed_batches = []

    def replace_records(self, records):
        self.replaced = list(records)

    def upsert_records(self, records):
        self.upserted_batches.append([record.tool_id for record in records])

    def remove_records(self, tool_ids):
        if not tool_ids:
            return
        self.removed_batches.append(sorted(tool_ids))

    def query(self, vector, top_k):
        return []


class DummyManager:
    def __init__(self, tools):
        self._tools = list(tools)

    async def list_all_registered_tools(self, **kwargs):
        return list(self._tools)

    def update_tools(self, tools):
        self._tools = list(tools)


def make_tool(server_label: str, name: str, description: str = "desc"):
    return SimpleNamespace(
        name=f"{server_label}-{name}",
        description=description,
        inputSchema={"type": "object"},
    )


@pytest.fixture
def registry_setup(monkeypatch):
    store = DummyStore()
    monkeypatch.setattr(
        "litellm.proxy.mcp_semantic_filter.registry.create_vector_store",
        lambda cfg: store,
    )

    embedding_calls = []

    def fake_embedding(model, input, **params):
        embedding_calls.append(list(input))
        return {
            "data": [
                {
                    "embedding": [float(idx)],
                }
                for idx, _ in enumerate(input)
            ]
        }

    monkeypatch.setattr(
        "litellm.proxy.mcp_semantic_filter.registry.litellm.embedding",
        fake_embedding,
    )

    config = SemanticFilterConfig(
        enabled=True,
        embedding=SemanticFilterEmbeddingConfig(model="fake-embed", parameters={}),
        vector_store=SemanticFilterVectorStoreConfig(),
    )

    registry = SemanticMCPFilterRegistry()
    registry.configure(config)
    return registry, store, embedding_calls


@pytest.mark.asyncio
async def test_rebuild_skips_when_tool_state_is_unchanged(registry_setup):
    registry, store, embedding_calls = registry_setup
    manager = DummyManager(
        [make_tool("serverA", "tool1"), make_tool("serverB", "tool2")]
    )

    await registry.rebuild_index(manager)
    await registry.rebuild_index(manager)

    assert len(embedding_calls) == 1
    assert store.upserted_batches == [["serverA-tool1", "serverB-tool2"]]
    assert store.removed_batches == []


@pytest.mark.asyncio
async def test_rebuild_embeds_only_changed_or_new_tools(registry_setup):
    registry, store, embedding_calls = registry_setup
    manager = DummyManager(
        [make_tool("serverA", "tool1"), make_tool("serverB", "tool2")]
    )

    await registry.rebuild_index(manager)

    manager.update_tools(
        [
            make_tool("serverA", "tool1", description="updated"),
            make_tool("serverC", "tool3"),
        ]
    )

    await registry.rebuild_index(manager)

    assert len(embedding_calls) == 2
    assert store.upserted_batches[-1] == ["serverA-tool1", "serverC-tool3"]
    assert store.removed_batches == [["serverB-tool2"]]
