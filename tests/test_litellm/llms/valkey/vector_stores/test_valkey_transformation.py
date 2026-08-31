import struct
import sys
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock, patch
from urllib.parse import unquote, urlsplit

import httpx
import pytest

from litellm.llms.valkey.vector_stores.transformation import (
    ValkeyVectorStoreConfig,
    _ValkeySearchParams,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class FakeSearchIndex:
    def __init__(self, result):
        self.result = result
        self.searched_query = None
        self.searched_query_params = None

    def search(self, query, query_params=None):
        self.searched_query = query
        self.searched_query_params = query_params
        return self.result


class FakeRedis:
    def __init__(self, result=None):
        self.index = FakeSearchIndex(result if result is not None else SimpleNamespace(docs=[]))
        self.ft_index_name = None

    def ft(self, index_name):
        self.ft_index_name = index_name
        return self.index


class FakeAsyncSearchIndex(FakeSearchIndex):
    async def search(self, query, query_params=None):
        self.searched_query = query
        self.searched_query_params = query_params
        return self.result


class FakeAsyncRedis(FakeRedis):
    def __init__(self, result=None):
        super().__init__(result)
        self.index = FakeAsyncSearchIndex(self.index.result)


class FakeEmbeddingFn:
    def __init__(self, embedding):
        self.embedding = embedding
        self.captured_kwargs = None

    def __call__(self, **kwargs):
        self.captured_kwargs = kwargs
        return SimpleNamespace(data=[{"embedding": self.embedding}])


class FakeAsyncEmbeddingFn(FakeEmbeddingFn):
    async def __call__(self, **kwargs):
        self.captured_kwargs = kwargs
        return SimpleNamespace(data=[{"embedding": self.embedding}])


def _doc(doc_id, distance, **fields):
    return SimpleNamespace(id=doc_id, vector_distance=str(distance), **fields)


def _search(config, client=None, query="what is litellm", optional_params=None, litellm_params=None):
    return config.execute_search_vector_store_request(
        vector_store_id="my_index",
        query=query,
        vector_store_search_optional_params=optional_params or {},
        litellm_logging_obj=MagicMock(),
        litellm_params={"litellm_embedding_model": "openai/text-embedding-3-small", **(litellm_params or {})},
    )


def test_sync_search_builds_knn_query_with_packed_vector():
    embedding_fn = FakeEmbeddingFn([0.1, 0.2, 0.3])
    client = FakeRedis()
    config = ValkeyVectorStoreConfig(sync_client=client, embedding_fn=embedding_fn)

    _search(config, optional_params={"max_num_results": 5})

    assert client.ft_index_name == "my_index"
    assert client.index.searched_query.query_string() == "*=>[KNN 5 @embedding $vec AS vector_distance]"
    args = client.index.searched_query.get_args()
    assert args[args.index("DIALECT") + 1] == 2
    assert args[args.index("LIMIT") : args.index("LIMIT") + 3] == ["LIMIT", 0, 5]
    return_args = args[args.index("RETURN") : args.index("RETURN") + 4]
    assert return_args == ["RETURN", 2, "text", "vector_distance"]
    assert client.index.searched_query_params == {"vec": struct.pack("<3f", 0.1, 0.2, 0.3)}


def test_sync_search_defaults_to_10_results():
    client = FakeRedis()
    config = ValkeyVectorStoreConfig(sync_client=client, embedding_fn=FakeEmbeddingFn([1.0]))

    _search(config)

    assert client.index.searched_query.query_string() == "*=>[KNN 10 @embedding $vec AS vector_distance]"


def test_sync_search_honors_custom_field_names():
    client = FakeRedis(result=SimpleNamespace(docs=[_doc("doc:1", 0.5, chunk="custom text")]))
    config = ValkeyVectorStoreConfig(sync_client=client, embedding_fn=FakeEmbeddingFn([1.0]))

    response = _search(
        config,
        litellm_params={"valkey_embedding_field": "emb", "valkey_text_field": "chunk"},
    )

    assert client.index.searched_query.query_string() == "*=>[KNN 10 @emb $vec AS vector_distance]"
    assert "chunk" in client.index.searched_query.get_args()
    assert response["data"][0]["content"][0]["text"] == "custom text"


def test_sync_search_maps_response_with_inverted_score_sorted_best_first():
    client = FakeRedis(
        result=SimpleNamespace(docs=[_doc("doc:2", 0.75, text="bye"), _doc("doc:1", 0.25, text="hello world")])
    )
    config = ValkeyVectorStoreConfig(sync_client=client, embedding_fn=FakeEmbeddingFn([1.0]))

    response = _search(config)

    assert response["object"] == "vector_store.search_results.page"
    assert response["search_query"] == "what is litellm"
    assert response["data"][0]["score"] == pytest.approx(0.75)
    assert response["data"][0]["content"] == [{"text": "hello world", "type": "text"}]
    assert response["data"][0]["file_id"] == "doc:1"
    assert response["data"][0]["filename"] == "doc:1"
    assert response["data"][1]["score"] == pytest.approx(0.25)
    assert response["data"][1]["file_id"] == "doc:2"


def test_sync_search_list_query_joins_all_elements():
    embedding_fn = FakeEmbeddingFn([1.0])
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=embedding_fn)

    response = _search(config, query=["first query", "second query"])

    assert embedding_fn.captured_kwargs["input"] == ["first query second query"]
    assert response["search_query"] == "first query second query"


def test_socket_timeouts_default_to_bounded_values():
    assert ValkeyVectorStoreConfig._socket_timeouts(None) == (5.0, 30.0)


def test_socket_timeouts_derive_from_numeric_request_timeout():
    assert ValkeyVectorStoreConfig._socket_timeouts(2.0) == (2.0, 2.0)
    assert ValkeyVectorStoreConfig._socket_timeouts(120.0) == (5.0, 120.0)


def test_socket_timeouts_derive_from_httpx_timeout():
    timeout = httpx.Timeout(connect=3.0, read=7.0, write=1.0, pool=1.0)

    assert ValkeyVectorStoreConfig._socket_timeouts(timeout) == (3.0, 7.0)


def test_sync_search_expands_embedding_config_into_kwargs():
    embedding_fn = FakeEmbeddingFn([1.0])
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=embedding_fn)

    _search(
        config,
        litellm_params={"litellm_embedding_config": {"api_key": "sk-test", "api_base": "https://embed.example.com"}},
    )

    assert embedding_fn.captured_kwargs == {
        "model": "openai/text-embedding-3-small",
        "input": ["what is litellm"],
        "api_key": "sk-test",
        "api_base": "https://embed.example.com",
    }


def test_sync_search_requires_embedding_model():
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=FakeEmbeddingFn([1.0]))

    with pytest.raises(ValueError, match="litellm_embedding_model is required"):
        config.execute_search_vector_store_request(
            vector_store_id="my_index",
            query="q",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={},
        )


def test_sync_search_requires_valkey_host_without_injected_client(monkeypatch):
    monkeypatch.delenv("VALKEY_HOST", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    config = ValkeyVectorStoreConfig(embedding_fn=FakeEmbeddingFn([1.0]))

    with pytest.raises(ValueError, match="valkey_host is required"):
        _search(config)


_VALKEY_ENV_VARS: Final = (
    "VALKEY_HOST",
    "VALKEY_PORT",
    "VALKEY_PASSWORD",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_PASSWORD",
)


def test_connection_url_building(monkeypatch):
    for var in _VALKEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    full: Final = _ValkeySearchParams.model_validate(
        {"valkey_host": "h", "valkey_port": 6380, "valkey_password": "p", "valkey_ssl": True}
    )
    assert full.connection_url() == "rediss://:p@h:6380"
    minimal: Final = _ValkeySearchParams.model_validate({"valkey_host": "h", "valkey_password": ""})
    assert minimal.connection_url() == "redis://h:6379"


def test_connection_url_never_borrows_gateway_credentials_from_the_environment(monkeypatch):
    monkeypatch.setenv("VALKEY_HOST", "gateway-valkey.internal")
    monkeypatch.setenv("VALKEY_PORT", "6380")
    monkeypatch.setenv("VALKEY_PASSWORD", "gateway-secret")
    monkeypatch.setenv("REDIS_HOST", "gateway-redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6381")
    monkeypatch.setenv("REDIS_PASSWORD", "gateway-redis-secret")

    caller_controlled: Final = _ValkeySearchParams.model_validate({"valkey_host": "attacker.example.com"})

    assert caller_controlled.connection_url() == "redis://attacker.example.com:6379"


def test_connection_url_percent_encodes_the_password():
    password: Final = "p@ss/w#rd%1:x"
    params: Final = _ValkeySearchParams.model_validate({"valkey_host": "h", "valkey_password": password})

    parsed: Final = urlsplit(params.connection_url())

    assert parsed.hostname == "h"
    assert parsed.port == 6379
    assert unquote(parsed.password or "") == password


def test_connection_url_accepts_string_booleans_from_the_ui_select():
    params: Final = _ValkeySearchParams.model_validate(
        {"valkey_host": "h", "valkey_port": "6380", "valkey_ssl": "true"}
    )

    assert params.connection_url() == "rediss://h:6380"
    assert _ValkeySearchParams.model_validate({"valkey_host": "h", "valkey_ssl": "false"}).connection_url() == (
        "redis://h:6379"
    )


def test_search_rejects_filters():
    embedding_fn = FakeEmbeddingFn([1.0])
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=embedding_fn)

    with pytest.raises(ValueError, match="does not support the filters parameter"):
        _search(config, optional_params={"filters": {"category": "docs"}})

    assert embedding_fn.captured_kwargs is None


@pytest.mark.asyncio
async def test_async_search_rejects_filters():
    aembedding_fn = FakeAsyncEmbeddingFn([1.0])
    config = ValkeyVectorStoreConfig(async_client=FakeAsyncRedis(), aembedding_fn=aembedding_fn)

    with pytest.raises(ValueError, match="does not support the filters parameter"):
        await config.aexecute_search_vector_store_request(
            vector_store_id="my_index",
            query="q",
            vector_store_search_optional_params={"filters": {"category": "docs"}},
            litellm_logging_obj=MagicMock(),
            litellm_params={"litellm_embedding_model": "openai/text-embedding-3-small"},
        )

    assert aembedding_fn.captured_kwargs is None


def test_search_rejects_empty_query():
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=FakeEmbeddingFn([1.0]))

    with pytest.raises(ValueError, match="query must not be empty"):
        _search(config, query=[])


@pytest.mark.parametrize("max_num_results", [0, -1, 51])
def test_search_rejects_out_of_range_max_num_results(max_num_results):
    embedding_fn = FakeEmbeddingFn([1.0])
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=embedding_fn)

    with pytest.raises(ValueError, match="max_num_results must be between 1 and 50"):
        _search(config, optional_params={"max_num_results": max_num_results})

    assert embedding_fn.captured_kwargs is None


def test_search_allows_max_num_results_at_the_upper_bound():
    client = FakeRedis()
    config = ValkeyVectorStoreConfig(sync_client=client, embedding_fn=FakeEmbeddingFn([1.0]))

    _search(config, optional_params={"max_num_results": 50})

    assert client.index.searched_query.query_string() == "*=>[KNN 50 @embedding $vec AS vector_distance]"


def test_search_treats_an_explicit_null_max_num_results_as_the_default():
    client = FakeRedis()
    config = ValkeyVectorStoreConfig(sync_client=client, embedding_fn=FakeEmbeddingFn([1.0]))

    _search(config, optional_params={"max_num_results": None})

    assert client.index.searched_query.query_string() == "*=>[KNN 10 @embedding $vec AS vector_distance]"


def test_missing_redis_dependency_raises_actionable_error():
    config = ValkeyVectorStoreConfig(sync_client=FakeRedis(), embedding_fn=FakeEmbeddingFn([1.0]))
    blocked = {name: None for name in list(sys.modules) if name == "redis" or name.startswith("redis.")}

    with patch.dict(sys.modules, blocked):
        with pytest.raises(ValueError, match="pip install redis"):
            _search(config)


@pytest.mark.asyncio
async def test_async_search_builds_knn_query_and_maps_response():
    aembedding_fn = FakeAsyncEmbeddingFn([0.5, 0.5])
    client = FakeAsyncRedis(result=SimpleNamespace(docs=[_doc("doc:9", 0.1, text="async hit")]))
    config = ValkeyVectorStoreConfig(async_client=client, aembedding_fn=aembedding_fn)

    response = await config.aexecute_search_vector_store_request(
        vector_store_id="my_index",
        query=["async query", "part two"],
        vector_store_search_optional_params={"max_num_results": 3},
        litellm_logging_obj=MagicMock(),
        litellm_params={
            "litellm_embedding_model": "openai/text-embedding-3-small",
            "litellm_embedding_config": {"api_key": "sk-async"},
        },
    )

    assert client.ft_index_name == "my_index"
    assert client.index.searched_query.query_string() == "*=>[KNN 3 @embedding $vec AS vector_distance]"
    assert client.index.searched_query_params == {"vec": struct.pack("<2f", 0.5, 0.5)}
    assert aembedding_fn.captured_kwargs == {
        "model": "openai/text-embedding-3-small",
        "input": ["async query part two"],
        "api_key": "sk-async",
    }
    assert response["search_query"] == "async query part two"
    assert response["data"][0]["score"] == pytest.approx(0.9)
    assert response["data"][0]["content"] == [{"text": "async hit", "type": "text"}]
    assert response["data"][0]["file_id"] == "doc:9"


def test_create_vector_store_is_not_supported():
    config = ValkeyVectorStoreConfig()

    with pytest.raises(NotImplementedError, match="search-only"):
        config.transform_create_vector_store_request(vector_store_create_optional_params={}, api_base="")


def test_provider_config_manager_returns_valkey_config():
    config = ProviderConfigManager.get_provider_vector_stores_config(provider=LlmProviders.VALKEY, api_type=None)

    assert isinstance(config, ValkeyVectorStoreConfig)
