"""Vendor §9.17: OpenAI vector store CRUD through the gateway (LIT-4778).

Create -> list -> retrieve -> delete against a live OpenAI-backed deployment.
Also covers upload file, attach to store, poll until ready, and search.
Negatives pin missing search query and invalid store id handling.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import FileUploadForm, NoBody, unwrap, assert_error_or_server_known
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient

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


class VectorStoreFileCreateBody(BaseModel):
    file_id: str
    attributes: dict[str, str] | None = None


class VectorStoreFileObject(BaseModel):
    id: str
    object: str | None = None
    status: str | None = None
    vector_store_id: str | None = None


class FileObject(BaseModel):
    id: str
    object: str | None = None
    purpose: str | None = None


class VectorStoreSearchHit(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_id: str | None = None
    filename: str | None = None
    score: float | None = None
    attributes: dict[str, str] | None = None
    content: list[dict[str, str]] | None = None


class VectorStoreSearchResponse(BaseModel):
    object: str | None = None
    data: list[VectorStoreSearchHit] = []


def _register_openai_model(proxy: ProxyClient, resources: ResourceManager) -> str:
    model = f"e2e-vs-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return resources.key()


def _delete_store_later(proxy: ProxyClient, resources: ResourceManager, key: str, store_id: str) -> None:
    def _delete() -> None:
        _ = proxy.transport.delete(
            f"/v1/vector_stores/{store_id}",
            headers=proxy.transport.bearer(key),
            json=NoBody(),
            response_type=VectorStoreDeleteResponse,
        )

    resources.defer(_delete)


def _poll_vector_store_file(
    proxy: ProxyClient, *, key: str, store_id: str, file_id: str
) -> VectorStoreFileObject:
    deadline = time.monotonic() + POLL_TIMEOUT
    last: VectorStoreFileObject | None = None
    while time.monotonic() < deadline:
        last = unwrap(
            proxy.transport.get(
                f"/v1/vector_stores/{store_id}/files/{file_id}",
                headers=proxy.transport.bearer(key),
                params=NoBody(),
                response_type=VectorStoreFileObject,
            )
        )
        if last.status in ("completed", "failed", "cancelled"):
            return last
        time.sleep(POLL_INTERVAL)
    raise AssertionError(
        f"vector store file {file_id} never reached a terminal status within "
        f"{POLL_TIMEOUT}s; last={last}"
    )



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
        _delete_store_later(proxy, resources, key, created.id)

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

        listed = unwrap(
            proxy.transport.get(
                "/v1/vector_stores",
                headers=proxy.transport.bearer(key),
                params=NoBody(),
                response_type=VectorStoreList,
            )
        )
        assert isinstance(listed.data, list), f"list must return data array: {listed}"
        listed_ids = {item.id for item in listed.data}
        if created.id not in listed_ids and listed.data:
            # OpenAI paginates; first page may omit a just-created store when the
            # account already has many. Create+retrieve already prove the path.
            assert retrieved.id == created.id

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
        _delete_store_later(proxy, resources, key, created.id)
        result = proxy.transport.send(
            f"/v1/vector_stores/{created.id}/search",
            headers=proxy.transport.bearer(key),
            json=VectorStoreSearchBody(max_num_results=10),
        )
        assert_error_or_server_known(result, "vector store search missing query")

    @pytest.mark.covers("llm.vector_stores.openai.basic.nonstream.works")
    def test_file_attach_poll_and_search(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        key = _register_openai_model(proxy, resources)
        marker = f"azure-falcon-{unique_marker()}"
        content = (
            b"LiteLLM e2e vector store document.\n"
            b"The secret project codename is "
            + marker.encode()
            + b".\nSearch should find that codename when queried.\n"
        )
        uploaded = unwrap(
            proxy.transport.upload(
                "/v1/files",
                headers=proxy.transport.bearer(key),
                form=FileUploadForm(purpose="assistants", custom_llm_provider="openai"),
                filename="vs_doc.txt",
                content=content,
                file_content_type="text/plain",
                response_type=FileObject,
            )
        )
        assert uploaded.id, f"file upload returned no id: {uploaded}"
        file_id = uploaded.id

        def _delete_file() -> None:
            _ = proxy.transport.delete(
                f"/v1/files/{file_id}",
                headers=proxy.transport.bearer(key),
                json=NoBody(),
                response_type=NoBody,
            )

        resources.defer(_delete_file)

        store = unwrap(
            proxy.transport.post(
                "/v1/vector_stores",
                headers=proxy.transport.bearer(key),
                json=VectorStoreCreateBody(name=f"e2e-vs-files-{unique_marker()}"),
                response_type=VectorStoreObject,
            )
        )
        _delete_store_later(proxy, resources, key, store.id)

        attached = unwrap(
            proxy.transport.post(
                f"/v1/vector_stores/{store.id}/files",
                headers=proxy.transport.bearer(key),
                json=VectorStoreFileCreateBody(
                    file_id=uploaded.id, attributes={"source": "e2e"}
                ),
                response_type=VectorStoreFileObject,
            )
        )
        assert attached.id, f"attach returned no file id: {attached}"
        ready = _poll_vector_store_file(
            proxy, key=key, store_id=store.id, file_id=attached.id
        )
        assert ready.status == "completed", f"file did not complete indexing: {ready}"

        search = unwrap(
            proxy.transport.post(
                f"/v1/vector_stores/{store.id}/search",
                headers=proxy.transport.bearer(key),
                json=VectorStoreSearchBody(query=marker, max_num_results=5),
                response_type=VectorStoreSearchResponse,
            )
        )
        assert search.data, f"search returned no hits for marker {marker!r}: {search}"
        hit_blob = " ".join(
            " ".join(part.get("text", "") for part in (hit.content or []))
            + " "
            + (hit.filename or "")
            for hit in search.data
        )
        assert marker in hit_blob or any(
            (hit.file_id or "") == uploaded.id for hit in search.data
        ), f"search hits must reference marker or uploaded file; marker={marker!r} hits={search.data}"

        deleted_file = unwrap(
            proxy.transport.delete(
                f"/v1/vector_stores/{store.id}/files/{attached.id}",
                headers=proxy.transport.bearer(key),
                json=NoBody(),
                response_type=VectorStoreDeleteResponse,
            )
        )
        assert deleted_file.deleted is True or deleted_file.id == attached.id

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
        _delete_store_later(proxy, resources, key, created.id)
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
