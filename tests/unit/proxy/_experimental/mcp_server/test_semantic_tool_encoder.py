"""
Unit tests for CachingToolEncoder, the LiteLLMRouterEncoder that serves
document embeddings from the shared ToolVectorStore.
"""

import asyncio
import sys
from unittest.mock import Mock

import pytest

from mcp.types import Tool as MCPTool

requires_semantic_router = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="The semantic-router extra excludes Python 3.14"
)


def _fake_embedding_router(recorded_inputs):
    """Router stand-in returning deterministic 2-dim keyword one-hots, like the filter tests."""
    from litellm.types.utils import Embedding, EmbeddingResponse

    def _vector(text):
        return [1.0, 0.0] if "ticket" in text.lower() else [0.0, 1.0]

    def mock_embedding_sync(*args, **kwargs):
        texts = kwargs["input"]
        recorded_inputs.append(list(texts))
        return EmbeddingResponse(
            data=[Embedding(embedding=_vector(t), index=i, object="embedding") for i, t in enumerate(texts)],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )

    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync(*args, **kwargs)

    router = Mock()
    router.embedding = mock_embedding_sync
    router.aembedding = mock_embedding_async
    return router


class _InMemoryToolVectorStore:
    def __init__(self, vectors=None):
        self.vectors = dict(vectors or {})
        self.get_many_calls = 0
        self.acquire_calls = 0
        self.release_calls = 0
        self.lock_result = True
        self._fill_on_call = None

    def fill_on_get_many_call(self, call_number):
        self._fill_on_call = call_number

    async def get_many(self, keys):
        self.get_many_calls += 1
        if self._fill_on_call is not None and self.get_many_calls >= self._fill_on_call:
            self.vectors.update(self._pending_fill)
        return {key: self.vectors[key] for key in keys if key in self.vectors}

    def seed_from(self, keys_vectors):
        self._pending_fill = dict(keys_vectors)

    async def put_many(self, vectors):
        self.vectors.update({k: tuple(v) for k, v in vectors.items()})

    async def try_acquire_build_lock(self):
        self.acquire_calls += 1
        return self.lock_result

    async def release_build_lock(self):
        self.release_calls += 1


def _tool(name, description):
    return MCPTool(name=name, description=description, inputSchema={"type": "object"})


def _make_filter(recorded_inputs, store=None, embedding_model="text-embedding-3-small"):
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )

    return SemanticMCPToolFilter(
        embedding_model=embedding_model,
        litellm_router_instance=_fake_embedding_router(recorded_inputs),
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
        vector_store=store,
        lock_poll_interval_s=0.0,
    )


def _make_encoder(recorded_inputs, store, embedding_identity="text-embedding-3-small"):
    from litellm.proxy._experimental.mcp_server.semantic_tool_encoder import CachingToolEncoder

    return CachingToolEncoder(
        litellm_router_instance=_fake_embedding_router(recorded_inputs),
        model_name="text-embedding-3-small",
        score_threshold=0.3,
        store=store,
        embedding_identity=embedding_identity,
    )


@requires_semantic_router
@pytest.mark.asyncio
async def test_build_with_warm_store_never_embeds_descriptions():
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        tool_vector_key,
    )

    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    identity = "text-embedding-3-small"
    warm = _InMemoryToolVectorStore(
        {
            tool_vector_key(identity, description): [0.0, 1.0]
            for _, description in ((t.name, t.description) for t in tools)
        }
    )
    recorded = []
    filter_instance = _make_filter(recorded, store=warm)

    await filter_instance._abuild_router(tools)

    assert filter_instance.tool_router is not None
    # Only the dims probe embeds; the 300-tool warm path makes zero embedding calls.
    assert recorded == [["test"]]
    assert warm.acquire_calls == 0


@requires_semantic_router
@pytest.mark.asyncio
async def test_cold_store_embeds_once_then_second_filter_embeds_nothing():
    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    store = _InMemoryToolVectorStore()

    first_recorded = []
    first = _make_filter(first_recorded, store=store)
    await first._abuild_router(tools)
    assert first.tool_router is not None
    assert [batch for batch in first_recorded if batch != ["test"]] == [["get a ticket", "weather forecast"]]

    second_recorded = []
    second = _make_filter(second_recorded, store=store)
    await second._abuild_router(tools)
    assert second.tool_router is not None
    assert second_recorded == [["test"]]


@requires_semantic_router
@pytest.mark.asyncio
async def test_edited_description_reembeds_only_that_tool():
    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    store = _InMemoryToolVectorStore()
    recorded = []
    filter_instance = _make_filter(recorded, store=store)
    await filter_instance._abuild_router(tools)

    recorded.clear()
    edited = [_tool("srv-a_get_ticket", "close a support ticket fast"), _tool("srv-b_get_weather", "weather forecast")]
    await filter_instance._abuild_router(edited)

    assert [batch for batch in recorded if batch != ["test"]] == [["close a support ticket fast"]]


@requires_semantic_router
@pytest.mark.asyncio
async def test_different_embedding_identity_reembeds_everything():
    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    store = _InMemoryToolVectorStore()

    first_recorded = []
    await _make_filter(first_recorded, store=store, embedding_model="text-embedding-3-small")._abuild_router(tools)

    second_recorded = []
    second = _make_filter(second_recorded, store=store, embedding_model="text-embedding-3-large")
    await second._abuild_router(tools)

    assert second.tool_router is not None
    assert [batch for batch in second_recorded if batch != ["test"]] == [["get a ticket", "weather forecast"]]


@requires_semantic_router
@pytest.mark.asyncio
async def test_wrong_length_cached_vector_is_reembedded_and_overwritten():
    """
    A cached vector whose length disagrees with a live dimension probe is
    treated as a miss: it is re-embedded, served at the right length, and the
    store entry is overwritten so the stale shape cannot leak into the index.
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        tool_vector_key,
    )

    key = tool_vector_key("text-embedding-3-small", "get a ticket")
    store = _InMemoryToolVectorStore({key: (9.9, 9.9, 9.9, 9.9)})
    recorded = []
    encoder = _make_encoder(recorded, store)

    vectors = await encoder.aencode_documents(["get a ticket"])

    assert vectors == [[1.0, 0.0]]
    assert [batch for batch in recorded if batch != ["test"]] == [["get a ticket"]]
    assert store.vectors[key] == (1.0, 0.0)
