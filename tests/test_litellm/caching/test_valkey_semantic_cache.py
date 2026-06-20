import hashlib
import os
import struct
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.valkey_semantic_cache import ValkeySemanticCache

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _make_cache(sync_client=None, async_client=None, similarity_threshold=0.8):
    return ValkeySemanticCache(
        similarity_threshold=similarity_threshold,
        index_name="test_index",
        sync_client=sync_client or MagicMock(),
        async_client=async_client or AsyncMock(),
    )


def _search_result(distance, response='{"content": "Paris"}'):
    return SimpleNamespace(
        docs=[SimpleNamespace(response=response, vector_distance=str(distance))]
    )


def test_build_valkey_url_prefers_valkey_env(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "redis-host")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_PASSWORD", "rpass")
    monkeypatch.setenv("VALKEY_HOST", "valkey-host")
    monkeypatch.setenv("VALKEY_PORT", "6380")
    monkeypatch.setenv("VALKEY_PASSWORD", "vpass")

    assert (
        ValkeySemanticCache._build_valkey_url(None, None, None)
        == "redis://:vpass@valkey-host:6380"
    )


def test_build_valkey_url_supports_passwordless(monkeypatch):
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("VALKEY_PASSWORD", raising=False)
    monkeypatch.setenv("VALKEY_HOST", "valkey-host")
    monkeypatch.setenv("VALKEY_PORT", "6380")

    assert (
        ValkeySemanticCache._build_valkey_url(None, None, None)
        == "redis://valkey-host:6380"
    )


def test_build_valkey_url_falls_back_to_redis_env(monkeypatch):
    monkeypatch.delenv("VALKEY_HOST", raising=False)
    monkeypatch.delenv("VALKEY_PORT", raising=False)
    monkeypatch.delenv("VALKEY_PASSWORD", raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis-host")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_PASSWORD", "rpass")

    assert (
        ValkeySemanticCache._build_valkey_url(None, None, None)
        == "redis://:rpass@redis-host:6379"
    )


def test_build_valkey_url_requires_host_and_port(monkeypatch):
    for var in (
        "VALKEY_HOST",
        "VALKEY_PORT",
        "VALKEY_PASSWORD",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError, match="Missing required Valkey configuration"):
        ValkeySemanticCache._build_valkey_url(None, None, None)


def test_build_valkey_url_uses_rediss_scheme_when_ssl(monkeypatch):
    monkeypatch.setenv("VALKEY_HOST", "valkey-host")
    monkeypatch.setenv("VALKEY_PORT", "6379")
    monkeypatch.setenv("VALKEY_PASSWORD", "vpass")

    assert (
        ValkeySemanticCache._build_valkey_url(None, None, None, ssl=True)
        == "rediss://:vpass@valkey-host:6379"
    )
    assert ValkeySemanticCache._build_valkey_url(
        "h", "6379", None, ssl=False
    ).startswith("redis://")


def test_init_requires_similarity_threshold():
    with pytest.raises(ValueError, match="similarity_threshold must be provided"):
        ValkeySemanticCache(sync_client=MagicMock(), async_client=AsyncMock())


def test_init_rejects_cluster_startup_nodes():
    with pytest.raises(ValueError, match="cluster-mode-enabled"):
        ValkeySemanticCache(
            similarity_threshold=0.8,
            startup_nodes=[{"host": "shard1", "port": 6379}],
        )


def test_cache_dispatch_rejects_cluster_for_valkey_semantic():
    from litellm.caching.caching import Cache
    from litellm.types.caching import LiteLLMCacheType

    with pytest.raises(ValueError, match="cluster-mode-enabled"):
        Cache(
            type=LiteLLMCacheType.VALKEY_SEMANTIC,
            host="valkey-host",
            port="6379",
            similarity_threshold=0.8,
            redis_startup_nodes=[{"host": "shard1", "port": 6379}],
        )


def test_scope_tag_is_deterministic_hex():
    tag = ValkeySemanticCache._scope_tag("model:gpt-4o::abc-123")
    assert tag == hashlib.sha256(b"model:gpt-4o::abc-123").hexdigest()
    assert len(tag) == 64
    assert ValkeySemanticCache._scope_tag("a") != ValkeySemanticCache._scope_tag("b")


def test_embedding_to_bytes_is_little_endian_float32():
    assert ValkeySemanticCache._embedding_to_bytes([1.0, 0.0]) == struct.pack(
        "<2f", 1.0, 0.0
    )


def test_set_cache_stores_scoped_doc_with_embedding(monkeypatch):
    sync_client = MagicMock()
    cache = _make_cache(sync_client=sync_client)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    cache.set_cache(
        key="cache-key",
        value={"content": "Paris"},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    sync_client.ft.return_value.create_index.assert_called_once()
    assert sync_client.hset.call_count == 1
    doc_key, kwargs = (
        sync_client.hset.call_args.args[0],
        sync_client.hset.call_args.kwargs,
    )
    mapping = kwargs["mapping"]
    scope = ValkeySemanticCache._scope_tag("cache-key")
    assert mapping[ValkeySemanticCache.CACHE_KEY_FIELD_NAME] == scope
    assert mapping["prompt"] == "What is the capital of France?"
    assert mapping["response"] == "{'content': 'Paris'}"
    assert mapping["embedding"] == struct.pack("<3f", 0.1, 0.2, 0.3)
    assert doc_key.startswith(f"test_index:{scope}:")


def test_set_cache_applies_ttl():
    sync_client = MagicMock()
    cache = _make_cache(sync_client=sync_client)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    cache.set_cache(
        key="cache-key",
        value={"content": "Paris"},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        ttl=60,
    )

    sync_client.expire.assert_called_once()
    assert sync_client.expire.call_args.args[1] == 60


def test_set_cache_skips_ttl_when_absent():
    sync_client = MagicMock()
    cache = _make_cache(sync_client=sync_client)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    cache.set_cache(
        key="cache-key",
        value={"content": "Paris"},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    sync_client.expire.assert_not_called()


def test_get_cache_returns_hit_above_threshold():
    sync_client = MagicMock()
    sync_client.ft.return_value.search.return_value = _search_result(0.1)
    cache = _make_cache(sync_client=sync_client, similarity_threshold=0.8)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    metadata = {}
    result = cache.get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "capital of France?"}],
        metadata=metadata,
    )

    assert result == {"content": "Paris"}
    assert metadata["semantic-similarity"] == pytest.approx(0.9)


def test_get_cache_misses_below_threshold():
    sync_client = MagicMock()
    sync_client.ft.return_value.search.return_value = _search_result(0.5)
    cache = _make_cache(sync_client=sync_client, similarity_threshold=0.8)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    metadata = {}
    result = cache.get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "capital of Germany?"}],
        metadata=metadata,
    )

    assert result is None
    assert metadata["semantic-similarity"] == pytest.approx(0.5)


def test_get_cache_misses_when_no_docs():
    sync_client = MagicMock()
    sync_client.ft.return_value.search.return_value = SimpleNamespace(docs=[])
    cache = _make_cache(sync_client=sync_client)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    metadata = {}
    result = cache.get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "capital of France?"}],
        metadata=metadata,
    )

    assert result is None
    assert metadata["semantic-similarity"] == 0.0


def test_get_cache_query_filters_by_scope_tag():
    sync_client = MagicMock()
    sync_client.ft.return_value.search.return_value = _search_result(0.1)
    cache = _make_cache(sync_client=sync_client)
    cache._get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    cache.get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "capital of France?"}],
        metadata={},
    )

    query = sync_client.ft.return_value.search.call_args.args[0]
    scope = ValkeySemanticCache._scope_tag("cache-key")
    assert scope in query.query_string()
    assert "KNN 1 @embedding" in query.query_string()


def _async_ft(search_distance):
    search_obj = SimpleNamespace(
        search=AsyncMock(return_value=_search_result(search_distance)),
        create_index=AsyncMock(),
    )
    return MagicMock(return_value=search_obj)


@pytest.mark.asyncio
async def test_async_set_and_get_roundtrip():
    async_client = AsyncMock()
    async_client.ft = _async_ft(0.05)
    cache = _make_cache(async_client=async_client, similarity_threshold=0.8)
    cache._get_async_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    await cache.async_set_cache(
        key="cache-key",
        value={"content": "Paris"},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        ttl=30,
    )
    async_client.hset.assert_awaited_once()
    async_client.expire.assert_awaited_once()
    assert async_client.expire.call_args.args[1] == 30

    metadata = {}
    result = await cache.async_get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "capital city of France"}],
        metadata=metadata,
    )
    assert result == {"content": "Paris"}
    assert metadata["semantic-similarity"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_async_get_cache_misses_below_threshold():
    async_client = AsyncMock()
    async_client.ft = _async_ft(0.4)
    cache = _make_cache(async_client=async_client, similarity_threshold=0.8)
    cache._get_async_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    metadata = {}
    result = await cache.async_get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "capital of Germany?"}],
        metadata=metadata,
    )
    assert result is None
    assert metadata["semantic-similarity"] == pytest.approx(0.6)


def test_ensure_index_swallows_already_exists():
    sync_client = MagicMock()
    sync_client.ft.return_value.create_index.side_effect = Exception(
        "Index test_index already exists."
    )
    cache = _make_cache(sync_client=sync_client)

    cache._ensure_index_sync(3)
    assert cache._index_dim == 3


def test_ensure_index_reraises_unexpected_error():
    sync_client = MagicMock()
    sync_client.ft.return_value.create_index.side_effect = Exception(
        "connection refused"
    )
    cache = _make_cache(sync_client=sync_client)

    with pytest.raises(Exception, match="connection refused"):
        cache._ensure_index_sync(3)


_FT_INFO_ATTRS_DIM_1536 = [
    [b"identifier", b"litellm_cache_key", b"type", b"TAG"],
    [
        b"identifier",
        b"embedding",
        b"type",
        b"VECTOR",
        b"index",
        [b"capacity", 10240, b"dimensions", 1536, b"distance_metric", b"COSINE"],
    ],
]


def test_extract_index_dim_parses_nested_ft_info():
    info = {"attributes": _FT_INFO_ATTRS_DIM_1536}
    assert ValkeySemanticCache._extract_index_dim(info) == 1536
    assert ValkeySemanticCache._extract_index_dim({"attributes": []}) is None


def test_ensure_index_raises_on_dimension_mismatch():
    sync_client = MagicMock()
    sync_client.ft.return_value.create_index.side_effect = Exception(
        "Index test_index already exists."
    )
    sync_client.ft.return_value.info.return_value = {
        "attributes": _FT_INFO_ATTRS_DIM_1536
    }
    cache = _make_cache(sync_client=sync_client)

    with pytest.raises(
        ValueError, match="already exists with embedding dimension 1536"
    ):
        cache._ensure_index_sync(768)
    assert cache._index_dim is None


def test_ensure_index_accepts_matching_existing_dimension():
    sync_client = MagicMock()
    sync_client.ft.return_value.create_index.side_effect = Exception(
        "Index test_index already exists."
    )
    sync_client.ft.return_value.info.return_value = {
        "attributes": _FT_INFO_ATTRS_DIM_1536
    }
    cache = _make_cache(sync_client=sync_client)

    cache._ensure_index_sync(1536)
    assert cache._index_dim == 1536


def test_init_builds_only_missing_client_from_url():
    sync_client = MagicMock()
    cache = ValkeySemanticCache(
        similarity_threshold=0.8,
        redis_url="redis://valkey-host:6380",
        sync_client=sync_client,
    )
    assert cache.sync_client is sync_client
    assert cache.async_client is not None and cache.async_client is not sync_client


def test_init_uses_both_injected_clients_without_connection_info(monkeypatch):
    for var in ("VALKEY_HOST", "VALKEY_PORT", "REDIS_HOST", "REDIS_PORT"):
        monkeypatch.delenv(var, raising=False)
    sync_client = MagicMock()
    async_client = AsyncMock()

    cache = ValkeySemanticCache(
        similarity_threshold=0.8,
        sync_client=sync_client,
        async_client=async_client,
    )

    assert cache.sync_client is sync_client
    assert cache.async_client is async_client


def test_cache_dispatches_valkey_semantic_type():
    from litellm.caching.caching import Cache
    from litellm.types.caching import LiteLLMCacheType

    cache = Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        host="valkey-host",
        port="6380",
        similarity_threshold=0.8,
    )

    assert isinstance(cache.cache, ValkeySemanticCache)


@pytest.mark.asyncio
async def test_index_info_uses_valkey_ft_info():
    # The /health/readiness endpoint calls _index_info() on any
    # RedisSemanticCache instance; since ValkeySemanticCache subclasses it,
    # the inherited RedisVL implementation (which reads self.llmcache) would
    # break. This override must query valkey-search FT.INFO instead.
    async_client = AsyncMock()
    info_namespace = SimpleNamespace(info=AsyncMock(return_value={"num_docs": 3}))
    async_client.ft = MagicMock(return_value=info_namespace)
    cache = _make_cache(async_client=async_client)

    result = await cache._index_info()

    assert result == {"num_docs": 3}
    async_client.ft.assert_called_once_with("test_index")


def test_importing_caching_does_not_require_redis():
    # redis is an optional dependency (extra_proxy), so the base SDK can be
    # installed without it. Selecting valkey-semantic needs redis, but merely
    # importing litellm.caching.caching must not, or `import litellm` breaks for
    # every base-SDK user. This runs in a subprocess with redis blocked so the
    # check is not polluted by redis already being imported in this session.
    code = textwrap.dedent("""
        import sys
        for name in ("redis", "redis.asyncio", "redis.commands",
                     "redis.commands.search"):
            sys.modules[name] = None
        import litellm.caching.caching  # must not import redis at module top
        from litellm.types.caching import LiteLLMCacheType
        assert LiteLLMCacheType.VALKEY_SEMANTIC == "valkey-semantic"
        print("ok")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
