"""Vendor §9.17: OpenAI vector store CRUD through the gateway (LIT-4778).

Create -> list -> retrieve -> delete against a live OpenAI-backed deployment.
Negatives pin missing search query and invalid store id handling.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from vendor_contract import assert_error_or_server_known

pytestmark = pytest.mark.e2e


class VectorStoreCreateBody(BaseModel):
    name: str
    metadata: dict[str, str] | None = None


class VectorStoreObject(BaseModel):
    id: str
    object: str | None = None
    name: str | None = None
    metadata: dict[str, str] | None = None


class VectorStoreList(BaseModel):
    object: str | None = None
    data: list[VectorStoreObject] = []


class VectorStoreDeleteResponse(BaseModel):
    id: str | None = None
    object: str | None = None
    deleted: bool | None = None


class VectorStoreSearchBody(BaseModel):
    query: str | None = None
    max_num_results: int | None = None


class VectorStoreUpdateBody(BaseModel):
    name: str | None = None


def _register_openai_model(proxy: ProxyClient, resources: ResourceManager) -> str:
    model = f"e2e-vs-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return resources.key()


class TestVectorStores:
    @pytest.mark.covers("llm.vector_stores.openai.basic.nonstream.works")
    def test_create_list_retrieve_delete_lifecycle(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        key = _register_openai_model(proxy, resources)
        name = f"e2e-vector-store-{unique_marker()}"
        created = unwrap(
            proxy.transport.post(
                "/v1/vector_stores",
                headers=proxy.transport.bearer(key),
                json=VectorStoreCreateBody(
                    name=name, metadata={"project": "e2e", "env": "test"}
                ),
                response_type=VectorStoreObject,
            )
        )
        assert created.id, f"create returned no id: {created}"
        store_id = created.id

        def _delete_store() -> None:
            _ = proxy.transport.delete(
                f"/v1/vector_stores/{store_id}",
                headers=proxy.transport.bearer(key),
                json=NoBody(),
                response_type=VectorStoreDeleteResponse,
            )

        resources.defer(_delete_store)

        listed = unwrap(
            proxy.transport.get(
                "/v1/vector_stores",
                headers=proxy.transport.bearer(key),
                params=NoBody(),
                response_type=VectorStoreList,
            )
        )
        assert any(item.id == created.id for item in listed.data), (
            f"created store {created.id} missing from list: {listed}"
        )

        retrieved = unwrap(
            proxy.transport.get(
                f"/v1/vector_stores/{created.id}",
                headers=proxy.transport.bearer(key),
                params=NoBody(),
                response_type=VectorStoreObject,
            )
        )
        assert retrieved.id == created.id
        assert retrieved.object in (None, "vector_store")

        deleted = unwrap(
            proxy.transport.delete(
                f"/v1/vector_stores/{created.id}",
                headers=proxy.transport.bearer(key),
                json=NoBody(),
                response_type=VectorStoreDeleteResponse,
            )
        )
        assert deleted.deleted is True or deleted.id == created.id

    @pytest.mark.covers("llm.vector_stores.openai.input_validation.nonstream.works")
    def test_search_missing_query_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        key = _register_openai_model(proxy, resources)
        created = unwrap(
            proxy.transport.post(
                "/v1/vector_stores",
                headers=proxy.transport.bearer(key),
                json=VectorStoreCreateBody(name=f"e2e-vs-search-{unique_marker()}"),
                response_type=VectorStoreObject,
            )
        )
        store_id = created.id

        def _delete_search_store() -> None:
            _ = proxy.transport.delete(
                f"/v1/vector_stores/{store_id}",
                headers=proxy.transport.bearer(key),
                json=NoBody(),
                response_type=VectorStoreDeleteResponse,
            )

        resources.defer(_delete_search_store)
        result = proxy.transport.send(
            f"/v1/vector_stores/{created.id}/search",
            headers=proxy.transport.bearer(key),
            json=VectorStoreSearchBody(max_num_results=10),
        )
        assert_error_or_server_known(result, "vector store search missing query")

    @pytest.mark.covers("llm.vector_stores.openai.input_validation.nonstream.works")
    def test_search_empty_query_returns_error_or_empty(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        key = _register_openai_model(proxy, resources)
        created = unwrap(
            proxy.transport.post(
                "/v1/vector_stores",
                headers=proxy.transport.bearer(key),
                json=VectorStoreCreateBody(name=f"e2e-vs-empty-{unique_marker()}"),
                response_type=VectorStoreObject,
            )
        )
        store_id = created.id

        def _delete_empty_store() -> None:
            _ = proxy.transport.delete(
                f"/v1/vector_stores/{store_id}",
                headers=proxy.transport.bearer(key),
                json=NoBody(),
                response_type=VectorStoreDeleteResponse,
            )

        resources.defer(_delete_empty_store)
        result = proxy.transport.send(
            f"/v1/vector_stores/{created.id}/search",
            headers=proxy.transport.bearer(key),
            json=VectorStoreSearchBody(query="", max_num_results=10),
        )
        assert result.status_code in (200, 400, 500), (
            f"empty search query unexpected status {result.status_code}: {result.body[:300]}"
        )

    @pytest.mark.covers("llm.vector_stores.openai.input_validation.nonstream.works")
    def test_retrieve_invalid_id_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        from e2e_http import Success, UnauthorizedError, UnknownApiError

        key = _register_openai_model(proxy, resources)
        result = proxy.transport.get(
            "/v1/vector_stores/vs_does_not_exist_xyz",
            headers=proxy.transport.bearer(key),
            params=NoBody(),
            response_type=VectorStoreObject,
        )
        match result:
            case Success():
                pytest.fail("invalid vector store id must not succeed")
            case UnknownApiError(status_code=status):
                assert status in (400, 401, 404, 500), f"unexpected status {status}"
            case UnauthorizedError():
                return
            case _:
                return

    @pytest.mark.covers("llm.vector_stores.openai.input_validation.nonstream.works")
    def test_invalid_chunking_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        key = _register_openai_model(proxy, resources)

        class ChunkingCreate(BaseModel):
            name: str
            chunking_strategy: dict[str, object]

        result = proxy.transport.send(
            "/v1/vector_stores",
            headers=proxy.transport.bearer(key),
            json=ChunkingCreate(
                name=f"e2e-vs-chunk-{unique_marker()}",
                chunking_strategy={
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 50,
                        "chunk_overlap_tokens": 40,
                    },
                },
            ),
        )
        assert_error_or_server_known(result, "invalid chunking strategy")
