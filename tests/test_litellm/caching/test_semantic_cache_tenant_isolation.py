"""
VERIA-54: Cross-tenant data leakage in semantic caching.

Both Redis and Qdrant semantic caches used to ignore tenant metadata —
two callers from different teams could retrieve each other's cached
responses by sending semantically similar prompts. These tests assert
that cross-tenant lookups are now isolated, while same-tenant requests
continue to share, and no-tenant callers (master key, direct SDK use)
fall back to the legacy shared pool.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching._tenant_scope import get_tenant_scope


@pytest.fixture
def fake_redisvl_module():
    """Inject a stub ``redisvl.extensions.llmcache`` so tests don't need
    the real redisvl install. The stub's ``SemanticCache`` is a placeholder
    that individual tests replace via ``patch`` on the real attribute."""
    real = sys.modules.get("redisvl.extensions.llmcache")
    real_parent = sys.modules.get("redisvl.extensions")
    real_grandparent = sys.modules.get("redisvl")
    if real is None:
        stub = types.ModuleType("redisvl.extensions.llmcache")
        stub.SemanticCache = MagicMock
        sys.modules["redisvl.extensions.llmcache"] = stub
        sys.modules.setdefault(
            "redisvl.extensions", types.ModuleType("redisvl.extensions")
        )
        sys.modules.setdefault("redisvl", types.ModuleType("redisvl"))
    yield
    # Restore (or leave alone if it was already real).
    if real is None:
        sys.modules.pop("redisvl.extensions.llmcache", None)
        if real_parent is None:
            sys.modules.pop("redisvl.extensions", None)
        if real_grandparent is None:
            sys.modules.pop("redisvl", None)


# ── get_tenant_scope ──────────────────────────────────────────────────────────


def test_get_tenant_scope_team_id_only():
    assert (
        get_tenant_scope({"metadata": {"user_api_key_team_id": "team-a"}}) == "team-a"
    )


def test_get_tenant_scope_combines_team_user_org():
    """Order is canonical so two callers always produce the same scope."""
    assert (
        get_tenant_scope(
            {
                "metadata": {
                    "user_api_key_team_id": "team-a",
                    "user_api_key_user_id": "user-1",
                    "user_api_key_org_id": "org-x",
                }
            }
        )
        == "team-a|user-1|org-x"
    )


def test_get_tenant_scope_none_when_no_metadata():
    assert get_tenant_scope({}) is None
    assert get_tenant_scope({"metadata": None}) is None
    assert get_tenant_scope({"metadata": {}}) is None


def test_get_tenant_scope_none_for_non_dict_metadata():
    assert get_tenant_scope({"metadata": "not-a-dict"}) is None


def test_get_tenant_scope_skips_empty_strings():
    assert (
        get_tenant_scope(
            {
                "metadata": {
                    "user_api_key_team_id": "",
                    "user_api_key_user_id": "user-1",
                }
            }
        )
        == "user-1"
    )


# ── Qdrant semantic cache ─────────────────────────────────────────────────────


def _make_qdrant_cache():
    from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

    cache = QdrantSemanticCache.__new__(QdrantSemanticCache)
    cache.qdrant_api_base = "https://qdrant.example.com"
    cache.collection_name = "test"
    cache.headers = {}
    cache.embedding_model = "text-embedding-3-small"
    cache.similarity_threshold = 0.8
    cache.sync_client = MagicMock()
    cache.async_client = MagicMock()
    return cache


def _qdrant_set_args(cache):
    """Return the JSON body the most recent ``put`` was called with."""
    return cache.sync_client.put.call_args.kwargs["json"]


def _qdrant_search_args(cache):
    return cache.sync_client.post.call_args.kwargs["json"]


def test_qdrant_set_cache_attaches_tenant_scope(monkeypatch):
    cache = _make_qdrant_cache()
    fake_embedding = MagicMock()
    fake_embedding.__getitem__.side_effect = lambda k: (
        [{"embedding": [0.1, 0.2]}] if k == "data" else None
    )
    monkeypatch.setattr(
        "litellm.embedding", lambda **kw: {"data": [{"embedding": [0.1, 0.2]}]}
    )

    cache.set_cache(
        key="cache-key",
        value="cached-response",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_team_id": "team-a"},
    )

    body = _qdrant_set_args(cache)
    payload = body["points"][0]["payload"]
    assert payload["tenant_scope"] == "team-a"


def test_qdrant_set_cache_uses_sentinel_when_no_tenant(monkeypatch):
    """No proxy metadata → the legacy shared pool."""
    cache = _make_qdrant_cache()
    monkeypatch.setattr(
        "litellm.embedding", lambda **kw: {"data": [{"embedding": [0.1, 0.2]}]}
    )

    cache.set_cache(key="k", value="v", messages=[{"role": "user", "content": "hi"}])

    body = _qdrant_set_args(cache)
    assert body["points"][0]["payload"]["tenant_scope"] == ""


def test_qdrant_get_cache_filters_by_tenant_scope(monkeypatch):
    cache = _make_qdrant_cache()
    monkeypatch.setattr(
        "litellm.embedding", lambda **kw: {"data": [{"embedding": [0.1, 0.2]}]}
    )
    cache.sync_client.post.return_value.json.return_value = {"result": []}

    cache.get_cache(
        key="cache-key",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_team_id": "team-a"},
    )

    body = _qdrant_search_args(cache)
    assert body["filter"] == {
        "must": [{"key": "tenant_scope", "match": {"value": "team-a"}}]
    }


def test_qdrant_get_cache_no_tenant_uses_sentinel_filter(monkeypatch):
    cache = _make_qdrant_cache()
    monkeypatch.setattr(
        "litellm.embedding", lambda **kw: {"data": [{"embedding": [0.1, 0.2]}]}
    )
    cache.sync_client.post.return_value.json.return_value = {"result": []}

    cache.get_cache(key="k", messages=[{"role": "user", "content": "hi"}])

    body = _qdrant_search_args(cache)
    # Sentinel ``""`` so callers without a tenant scope share their own
    # legacy pool but never see tenant entries.
    assert body["filter"]["must"][0]["match"]["value"] == ""


# ── Redis semantic cache ──────────────────────────────────────────────────────


def _make_redis_cache():
    """Build a RedisSemanticCache without touching the real Redis."""
    from litellm.caching.redis_semantic_cache import RedisSemanticCache

    cache = RedisSemanticCache.__new__(RedisSemanticCache)
    cache.similarity_threshold = 0.8
    cache.distance_threshold = 0.2
    cache.embedding_model = "text-embedding-3-small"
    cache._index_name_base = "litellm_semantic_cache_index"
    cache._redis_url = "redis://localhost:6379"
    cache._cache_vectorizer = MagicMock()
    cache._tenant_caches = {}
    # Default (no-tenant) index — callers without proxy metadata route here.
    cache.llmcache = MagicMock()
    cache.llmcache.store = MagicMock()
    cache.llmcache.check = MagicMock(return_value=[])
    cache.llmcache.astore = AsyncMock()
    cache.llmcache.acheck = AsyncMock(return_value=[])
    return cache


def test_redis_no_tenant_uses_default_index():
    """Direct-SDK callers (no proxy metadata) keep using the existing
    index — no schema migration, no data loss for upgrading deployments."""
    cache = _make_redis_cache()
    cache.set_cache(key="k", value="v", messages=[{"role": "user", "content": "hi"}])
    cache.llmcache.store.assert_called_once()
    assert cache._tenant_caches == {}


def test_redis_tenant_scoped_index_lazy_created(fake_redisvl_module):
    """First request from a tenant lazy-creates a per-tenant
    ``SemanticCache`` whose Redis index name is derived from the scope."""
    cache = _make_redis_cache()
    fake_scoped_cache = MagicMock()
    fake_scoped_cache.store = MagicMock()
    fake_scoped_cache.check = MagicMock(return_value=[])

    with patch(
        "redisvl.extensions.llmcache.SemanticCache",
        return_value=fake_scoped_cache,
    ) as mock_ctor:
        cache.set_cache(
            key="k",
            value="v",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_api_key_team_id": "team-a"},
        )

    # Tenant index name embeds the scope hash (deterministic, stable).
    call_kwargs = mock_ctor.call_args.kwargs
    assert call_kwargs["name"].startswith("litellm_semantic_cache_index:tenant:")
    assert call_kwargs["name"] != "litellm_semantic_cache_index"
    # Default cache untouched.
    cache.llmcache.store.assert_not_called()
    # Tenant cache used.
    fake_scoped_cache.store.assert_called_once()
    # Subsequent calls reuse the same instance — no constructor re-run.
    cache.set_cache(
        key="k2",
        value="v2",
        messages=[{"role": "user", "content": "hi again"}],
        metadata={"user_api_key_team_id": "team-a"},
    )
    assert mock_ctor.call_count == 1
    assert fake_scoped_cache.store.call_count == 2


def test_redis_two_tenants_get_distinct_indexes(fake_redisvl_module):
    """Cross-tenant isolation: scope ``team-a`` must never look at
    scope ``team-b``'s index."""
    cache = _make_redis_cache()

    instances = []

    def fake_ctor(**kwargs):
        m = MagicMock()
        m.store = MagicMock()
        m.check = MagicMock(return_value=[])
        m._litellm_index_name = kwargs["name"]
        instances.append(m)
        return m

    with patch("redisvl.extensions.llmcache.SemanticCache", side_effect=fake_ctor):
        cache.set_cache(
            key="k",
            value="v-a",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_api_key_team_id": "team-a"},
        )
        cache.set_cache(
            key="k",
            value="v-b",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_api_key_team_id": "team-b"},
        )

    assert len(instances) == 2
    assert instances[0]._litellm_index_name != instances[1]._litellm_index_name
    instances[0].store.assert_called_once_with("hi", "v-a")
    instances[1].store.assert_called_once_with("hi", "v-b")


def test_redis_get_cache_routes_through_tenant_index():
    """Tenant-A querying must hit team-A's index, not the default pool."""
    cache = _make_redis_cache()
    fake_scoped = MagicMock()
    fake_scoped.check = MagicMock(return_value=[])
    cache._tenant_caches["team-a"] = fake_scoped

    cache.get_cache(
        key="k",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_team_id": "team-a"},
    )

    fake_scoped.check.assert_called_once()
    cache.llmcache.check.assert_not_called()


@pytest.mark.asyncio
async def test_redis_async_set_cache_routes_through_tenant_index():
    cache = _make_redis_cache()
    fake_scoped = MagicMock()
    fake_scoped.astore = AsyncMock()
    cache._tenant_caches["team-a"] = fake_scoped
    cache._get_async_embedding = AsyncMock(return_value=[0.1, 0.2])

    await cache.async_set_cache(
        key="k",
        value="v",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_team_id": "team-a"},
    )

    fake_scoped.astore.assert_awaited_once()
    cache.llmcache.astore.assert_not_called()


@pytest.mark.asyncio
async def test_redis_async_get_cache_routes_through_tenant_index():
    cache = _make_redis_cache()
    fake_scoped = MagicMock()
    fake_scoped.acheck = AsyncMock(return_value=[])
    cache._tenant_caches["team-a"] = fake_scoped
    cache._get_async_embedding = AsyncMock(return_value=[0.1, 0.2])

    await cache.async_get_cache(
        key="k",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_team_id": "team-a"},
    )

    fake_scoped.acheck.assert_awaited_once()
    cache.llmcache.acheck.assert_not_called()
