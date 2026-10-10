"""
Unit tests for the Redis-shared MCP semantic tool index store.

Covers the float32 base64 round trip, degradation on Redis failure, the
cross-pod build lock, and the CachingToolEncoder serving document
embeddings from the store instead of re-embedding.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

if sys.version_info < (3, 11):  # BaseExceptionGroup is a builtin only from 3.11
    from exceptiongroup import BaseExceptionGroup


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
        self.block_put_many = False
        self.put_started = asyncio.Event()
        self.put_block = asyncio.Event()

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
        if self.block_put_many:
            self.put_started.set()
            await self.put_block.wait()
        self.vectors.update({k: tuple(v) for k, v in vectors.items()})

    async def try_acquire_build_lock(self):
        self.acquire_calls += 1
        return self.lock_result

    async def release_build_lock(self):
        self.release_calls += 1


class _RaisingRedisCache:
    async def async_batch_get_cache(self, key_list, parent_otel_span=None):
        raise ConnectionError("redis is down")

    async def async_set_cache_pipeline(self, cache_list, ttl=None, **kwargs):
        raise ConnectionError("redis is down")

    async def async_get_cache(self, key):
        raise ConnectionError("redis is down")

    async def async_set_cache(self, key, value, **kwargs):
        raise ConnectionError("redis is down")


class _DictRedisCache:
    def __init__(self):
        self.backing = {}
        self.pipeline_calls = 0

    async def async_batch_get_cache(self, key_list, parent_otel_span=None):
        return {key: self.backing[key] for key in key_list if key in self.backing}

    async def async_set_cache_pipeline(self, cache_list, ttl=None, **kwargs):
        self.pipeline_calls += 1
        self.backing.update(dict(cache_list))


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


@requires_semantic_router
@pytest.mark.asyncio
async def test_store_put_get_round_trip_within_float32_tolerance():
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        RedisToolVectorStore,
        tool_vector_key,
    )

    redis = _DictRedisCache()
    store = RedisToolVectorStore(redis, Mock())

    key = tool_vector_key("embed-a", "get a ticket")
    await store.put_many({key: [0.6000000238, -0.8123456789, 3.14]})

    hits = await store.get_many([key])
    assert key in hits
    assert list(hits[key]) == pytest.approx([0.6000000238, -0.8123456789, 3.14], abs=1e-6)


@requires_semantic_router
@pytest.mark.asyncio
async def test_store_treats_corrupt_value_as_miss():
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        RedisToolVectorStore,
        tool_vector_key,
    )

    redis = _DictRedisCache()
    key = tool_vector_key("embed-a", "get a ticket")
    redis.backing[key] = "!!! not base64 / not float32 bytes !!!"
    store = RedisToolVectorStore(redis, Mock())

    assert await store.get_many([key]) == {}


@requires_semantic_router
@pytest.mark.asyncio
async def test_raising_redis_still_builds_the_index():
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        RedisToolVectorStore,
    )
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

    redis = _RaisingRedisCache()
    recorded = []
    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    store = RedisToolVectorStore(redis, PodLockManager(redis))
    filter_instance = _make_filter(recorded, store=store)

    await filter_instance._abuild_router(tools)

    assert filter_instance.tool_router is not None
    assert [batch for batch in recorded if batch != ["test"]] == [
        ["get a ticket", "weather forecast"]
    ]


@requires_semantic_router
@pytest.mark.asyncio
async def test_build_with_warm_store_never_embeds_descriptions():
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        tool_vector_key,
    )

    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    identity = "text-embedding-3-small"
    warm = _InMemoryToolVectorStore(
        {tool_vector_key(identity, description): [0.0, 1.0] for _, description in ((t.name, t.description) for t in tools)}
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
    assert [batch for batch in first_recorded if batch != ["test"]] == [
        ["get a ticket", "weather forecast"]
    ]

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
    assert [batch for batch in second_recorded if batch != ["test"]] == [
        ["get a ticket", "weather forecast"]
    ]


@requires_semantic_router
@pytest.mark.asyncio
async def test_lock_held_by_other_pod_waits_for_shared_vectors():
    from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
        tool_vector_key,
    )

    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    store = _InMemoryToolVectorStore()
    store.lock_result = False
    identity = "text-embedding-3-small"
    # The "holder" pod's vectors appear on the third get_many poll.
    store.seed_from({tool_vector_key(identity, t.description): [0.0, 1.0] for t in tools})
    store.fill_on_get_many_call(3)

    recorded = []
    filter_instance = _make_filter(recorded, store=store)
    await filter_instance._abuild_router(tools)

    assert filter_instance.tool_router is not None
    assert store.acquire_calls >= 1
    assert store.get_many_calls >= 3
    assert recorded == [["test"]]


@requires_semantic_router
@pytest.mark.asyncio
async def test_full_build_reads_the_store_a_constant_number_of_times():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    tools = [_tool(f"srv-a_get_tool_{i}", f"do thing {i}") for i in range(20)]
    store = _InMemoryToolVectorStore()
    filter_instance = _make_filter([], store=store)

    with (
        patch.object(global_mcp_server_manager, "get_registry", return_value={"srv-a": object()}),
        patch.object(
            global_mcp_server_manager, "get_tools_for_server", new=AsyncMock(return_value=tools)
        ),
    ):
        await filter_instance.build_router_from_mcp_registry()

    assert filter_instance.tool_router is not None
    assert store.get_many_calls <= 3, (
        f"{store.get_many_calls} get_many calls for {len(tools)} tools; reads must not scale with the tool count"
    )


@requires_semantic_router
@pytest.mark.asyncio
async def test_cancelling_a_build_while_holding_the_lock_releases_it():
    tools = [_tool("srv-a_get_ticket", "get a ticket"), _tool("srv-b_get_weather", "weather forecast")]
    store = _InMemoryToolVectorStore()
    store.block_put_many = True
    filter_instance = _make_filter([], store=store)

    build_task = asyncio.create_task(filter_instance._abuild_router(tools))
    await asyncio.wait_for(store.put_started.wait(), timeout=5)
    assert store.acquire_calls == 1

    build_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(build_task, timeout=5)
    assert store.release_calls == 1
