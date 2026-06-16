"""
Unit tests for Milvus RAG ingestion (litellm/rag/ingestion/milvus_ingestion.py).

These tests mock the Milvus REST API via the async httpx client, so they run in
CI without a live Milvus instance.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.rag.ingestion.milvus_ingestion import MilvusRAGIngestion


def _make_ingestion(**vector_store_overrides):
    vector_store = {
        "custom_llm_provider": "milvus",
        "collection_name": "test_collection",
        "api_base": "http://localhost:19530",
        "api_key": "root:Milvus",
    }
    vector_store.update(vector_store_overrides)
    ingestion = MilvusRAGIngestion(
        ingest_options={
            "embedding": {"model": "text-embedding-3-small"},
            "vector_store": vector_store,
        }
    )
    return ingestion


def _json_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def test_requires_collection_name():
    with pytest.raises(ValueError, match="collection_name"):
        MilvusRAGIngestion(
            ingest_options={
                "vector_store": {
                    "custom_llm_provider": "milvus",
                    "api_base": "http://localhost:19530",
                }
            }
        )


def test_requires_api_base(monkeypatch):
    monkeypatch.delenv("MILVUS_API_BASE", raising=False)
    with pytest.raises(ValueError, match="API base"):
        MilvusRAGIngestion(
            ingest_options={
                "vector_store": {
                    "custom_llm_provider": "milvus",
                    "collection_name": "c",
                }
            }
        )


def test_config_defaults():
    ingestion = _make_ingestion()
    assert ingestion.collection_name == "test_collection"
    assert ingestion.api_base == "http://localhost:19530"
    assert ingestion.vector_field == "vector"
    assert ingestion.text_field == "text"
    assert ingestion.metric_type == "COSINE"
    assert ingestion.auto_create_collection is True


def test_vector_store_id_alias():
    ingestion = _make_ingestion(collection_name=None, vector_store_id="aliased")
    assert ingestion.collection_name == "aliased"


def test_headers_include_auth_when_api_key_set():
    ingestion = _make_ingestion()
    headers = ingestion._headers()
    assert headers["Authorization"] == "Bearer root:Milvus"


def test_headers_omit_auth_when_no_api_key(monkeypatch):
    monkeypatch.delenv("MILVUS_API_KEY", raising=False)
    ingestion = _make_ingestion(api_key=None)
    headers = ingestion._headers()
    assert "Authorization" not in headers


def test_server_api_key_not_sent_to_config_supplied_api_base(monkeypatch):
    monkeypatch.setenv("MILVUS_API_KEY", "server-secret")
    ingestion = _make_ingestion(api_base="https://attacker.example", api_key=None)
    assert ingestion.api_key is None
    assert "Authorization" not in ingestion._headers()


def test_server_api_key_used_only_with_server_api_base(monkeypatch):
    monkeypatch.setenv("MILVUS_API_KEY", "server-secret")
    monkeypatch.setenv("MILVUS_API_BASE", "https://milvus.internal")
    ingestion = _make_ingestion(api_base=None, api_key=None)
    assert ingestion.api_base == "https://milvus.internal"
    assert ingestion.api_key == "server-secret"
    assert ingestion._headers()["Authorization"] == "Bearer server-secret"


def test_config_supplied_api_key_used_with_config_api_base(monkeypatch):
    monkeypatch.setenv("MILVUS_API_KEY", "server-secret")
    ingestion = _make_ingestion(
        api_base="https://tenant.milvus.example", api_key="tenant-token"
    )
    assert ingestion.api_key == "tenant-token"


@pytest.mark.asyncio
async def test_store_raises_without_embeddings():
    ingestion = _make_ingestion()
    with pytest.raises(ValueError, match="No text content"):
        await ingestion.store(
            file_content=None,
            filename="doc.txt",
            content_type="text/plain",
            chunks=[],
            embeddings=None,
        )


@pytest.mark.asyncio
async def test_store_auto_creates_collection_and_inserts():
    ingestion = _make_ingestion()

    post_mock = AsyncMock()
    post_mock.side_effect = [
        _json_response({"code": 0, "data": {"has": False}}),  # has -> not exists
        _json_response({"code": 0, "data": {}}),  # create
        _json_response({"code": 0, "data": {"insertCount": 2}}),  # insert
    ]
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    vector_store_id, filename = await ingestion.store(
        file_content=None,
        filename="doc.txt",
        content_type="text/plain",
        chunks=["chunk a", "chunk b"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )

    assert vector_store_id == "test_collection"
    assert filename == "doc.txt"
    assert post_mock.call_count == 3

    called_paths = [c.args[0] for c in post_mock.call_args_list]
    assert called_paths[0].endswith("/v2/vectordb/collections/has")
    assert called_paths[1].endswith("/v2/vectordb/collections/create")
    assert called_paths[2].endswith("/v2/vectordb/entities/insert")

    # auto-create body uses detected dimension + configured fields
    create_body = post_mock.call_args_list[1].kwargs["json"]
    assert create_body["collectionName"] == "test_collection"
    assert create_body["dimension"] == 3
    assert create_body["vectorFieldName"] == "vector"
    assert create_body["metricType"] == "COSINE"

    # insert body carries vectors, chunk text, and metadata
    insert_body = post_mock.call_args_list[2].kwargs["json"]
    assert insert_body["collectionName"] == "test_collection"
    rows = insert_body["data"]
    assert len(rows) == 2
    assert rows[0]["vector"] == [0.1, 0.2, 0.3]
    assert rows[0]["text"] == "chunk a"
    assert rows[0]["chunk_index"] == 0
    assert rows[0]["filename"] == "doc.txt"


@pytest.mark.asyncio
async def test_store_skips_create_when_collection_exists():
    ingestion = _make_ingestion()

    post_mock = AsyncMock()
    post_mock.side_effect = [
        _json_response({"code": 0, "data": {"has": True}}),  # has -> exists
        _json_response({"code": 0, "data": {"insertCount": 1}}),  # insert
    ]
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    await ingestion.store(
        file_content=None,
        filename=None,
        content_type=None,
        chunks=["only chunk"],
        embeddings=[[1.0, 2.0]],
    )

    called_paths = [c.args[0] for c in post_mock.call_args_list]
    assert not any(p.endswith("/collections/create") for p in called_paths)
    assert called_paths[-1].endswith("/v2/vectordb/entities/insert")


@pytest.mark.asyncio
async def test_store_respects_auto_create_false():
    ingestion = _make_ingestion(auto_create_collection=False)

    post_mock = AsyncMock()
    post_mock.return_value = _json_response({"code": 0, "data": {"insertCount": 1}})
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    await ingestion.store(
        file_content=None,
        filename=None,
        content_type=None,
        chunks=["c"],
        embeddings=[[1.0]],
    )

    # only the insert call - no has/create probing
    assert post_mock.call_count == 1
    assert post_mock.call_args_list[0].args[0].endswith("/v2/vectordb/entities/insert")


@pytest.mark.asyncio
async def test_post_raises_on_milvus_error_code():
    ingestion = _make_ingestion(auto_create_collection=False)

    post_mock = AsyncMock()
    post_mock.return_value = _json_response(
        {"code": 1100, "message": "collection not found"}
    )
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    with pytest.raises(RuntimeError, match="collection not found"):
        await ingestion.store(
            file_content=None,
            filename=None,
            content_type=None,
            chunks=["c"],
            embeddings=[[1.0]],
        )


@pytest.mark.asyncio
async def test_db_name_and_partition_from_env_propagated(monkeypatch):
    monkeypatch.setenv("MILVUS_DB_NAME", "mydb")
    monkeypatch.setenv("MILVUS_PARTITION_NAME", "p1")
    ingestion = _make_ingestion(auto_create_collection=False)

    post_mock = AsyncMock()
    post_mock.return_value = _json_response({"code": 0, "data": {"insertCount": 1}})
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    await ingestion.store(
        file_content=None,
        filename=None,
        content_type=None,
        chunks=["c"],
        embeddings=[[1.0]],
    )

    insert_body = post_mock.call_args_list[0].kwargs["json"]
    assert insert_body["dbName"] == "mydb"
    assert insert_body["partitionName"] == "p1"


@pytest.mark.asyncio
async def test_partition_name_from_request_is_ignored(monkeypatch):
    monkeypatch.delenv("MILVUS_PARTITION_NAME", raising=False)
    ingestion = _make_ingestion(
        auto_create_collection=False, partition_name="victim_partition"
    )

    assert ingestion.partition_name is None

    post_mock = AsyncMock()
    post_mock.return_value = _json_response({"code": 0, "data": {"insertCount": 1}})
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    await ingestion.store(
        file_content=None,
        filename=None,
        content_type=None,
        chunks=["c"],
        embeddings=[[1.0]],
    )

    insert_body = post_mock.call_args_list[0].kwargs["json"]
    assert "partitionName" not in insert_body


@pytest.mark.asyncio
async def test_db_name_from_request_is_ignored(monkeypatch):
    monkeypatch.delenv("MILVUS_DB_NAME", raising=False)
    ingestion = _make_ingestion(auto_create_collection=False, db_name="victim_db")

    assert ingestion.db_name is None

    post_mock = AsyncMock()
    post_mock.return_value = _json_response({"code": 0, "data": {"insertCount": 1}})
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock

    await ingestion.store(
        file_content=None,
        filename=None,
        content_type=None,
        chunks=["c"],
        embeddings=[[1.0]],
    )

    insert_body = post_mock.call_args_list[0].kwargs["json"]
    assert "dbName" not in insert_body


@pytest.mark.asyncio
async def test_embed_returns_none_for_empty_chunks():
    ingestion = _make_ingestion()
    assert await ingestion.embed([]) is None


@pytest.mark.asyncio
async def test_embed_uses_litellm_aembedding(monkeypatch):
    ingestion = _make_ingestion()
    captured = {}

    async def fake_aembedding(model, input):
        captured["model"] = model
        captured["input"] = input
        resp = MagicMock()
        resp.data = [{"embedding": [0.1, 0.2]} for _ in input]
        return resp

    monkeypatch.setattr(
        "litellm.rag.ingestion.base_ingestion.litellm.aembedding",
        fake_aembedding,
    )

    result = await ingestion.embed(["a", "b"])
    assert result == [[0.1, 0.2], [0.1, 0.2]]
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["a", "b"]


@pytest.mark.asyncio
async def test_embed_uses_router_when_present():
    router = MagicMock()
    resp = MagicMock()
    resp.data = [{"embedding": [1.0]}]
    router.aembedding = AsyncMock(return_value=resp)
    ingestion = MilvusRAGIngestion(
        ingest_options={
            "embedding": {"model": "custom-embed"},
            "vector_store": {
                "custom_llm_provider": "milvus",
                "collection_name": "c",
                "api_base": "http://localhost:19530",
            },
        },
        router=router,
    )

    result = await ingestion.embed(["x"])
    assert result == [[1.0]]
    router.aembedding.assert_awaited_once_with(model="custom-embed", input=["x"])


@pytest.mark.asyncio
async def test_collection_exists_false_on_error():
    ingestion = _make_ingestion()
    post_mock = AsyncMock(side_effect=RuntimeError("boom"))
    ingestion.async_httpx_client = MagicMock()
    ingestion.async_httpx_client.post = post_mock
    assert await ingestion._collection_exists() is False


def test_can_auto_create_vector_store_default_true():
    assert (
        MilvusRAGIngestion.can_auto_create_vector_store(
            {"custom_llm_provider": "milvus", "collection_name": "c"}
        )
        is True
    )


def test_can_auto_create_vector_store_respects_disabled_flag():
    assert (
        MilvusRAGIngestion.can_auto_create_vector_store(
            {
                "custom_llm_provider": "milvus",
                "collection_name": "c",
                "auto_create_collection": False,
            }
        )
        is False
    )
