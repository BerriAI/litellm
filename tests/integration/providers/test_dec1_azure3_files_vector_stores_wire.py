"""Azure files (model routed) and Azure vector store search on the wire.

Files: `POST /v1/files` with a form `model` naming an Azure group uploads with that deployment's credentials
(`files_endpoints.py`, `get_authorized_credentials_for_model`), through the AzureOpenAI SDK
(`litellm/llms/azure/files/handler.py`), so Azure sees `POST /openai/files?api-version=<v>` with `api-key`. The id
the caller gets back is encoded with the model (`encode_file_id_with_model`) and read/delete/content calls on that
id must reach the same deployment with the raw Azure id.

Vector stores: a store registered with `custom_llm_provider: azure` is searched through
`AzureOpenAIVectorStoreConfig` (`litellm/llms/azure/vector_stores/transformation.py`), whose base is
`BaseAzureLLM.get_base_azure_url(route="/openai/vector_stores")`: the api_version becomes the `api-version` query,
`v1`/`latest`/`preview` move the route under `/openai/v1`, and with no api_version at all no query is sent.
Auth is `api-key`, or an Entra bearer without a key (`_base_validate_azure_environment`).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_azure3_support import full_peer, multipart_fields, multipart_file
from integration.providers.dec1_azure_support import (
    API_KEY,
    DATED_API_VERSION,
    DEPLOYMENT,
    ENTRA_TOKEN,
    json_reply,
    query_of,
)

FILE_BYTES: Final = b"synthetic azure file content\n"


def _file_object(raw_id: str) -> dict[str, object]:
    return {
        "id": raw_id,
        "object": "file",
        "bytes": len(FILE_BYTES),
        "created_at": 1,
        "filename": "notes.txt",
        "purpose": "assistants",
        "status": "processed",
    }


def _files_peer(raw_id: str):  # noqa: ANN202
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "DELETE":
            return json_reply({"id": raw_id, "object": "file", "deleted": True})
        if path.endswith("/content"):
            return Reply(body=FILE_BYTES, content_type="application/octet-stream")
        if path.endswith("/files") and request.method == "GET":
            return json_reply({"object": "list", "data": []})
        return json_reply(_file_object(raw_id))  # pyright: ignore[reportArgumentType]

    return respond


def _upload(rig: object) -> dict[str, object]:
    response = rig.gateway.request_multipart(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        "/v1/files",
        {"purpose": "assistants", "model": rig.model},  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        {"file": ("notes.txt", FILE_BYTES, "text/plain")},
    )
    assert response.status_code == 200, response.text  # pyright: ignore[reportUnknownMemberType]
    return response.json()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _file_calls(rig: object) -> list[Request]:
    return [r for r in rig.wire.drain() if "/files" in urlsplit(r.target).path]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


def _azure_files(gateway: Gateway, raw: str, **params: object):  # noqa: ANN202
    return full_peer(
        gateway,
        _files_peer(raw),
        model=f"azure/{DEPLOYMENT}",
        **{"api_key": API_KEY, "api_version": DATED_API_VERSION, **params},  # pyright: ignore[reportArgumentType]
    )


def test_a_model_routed_upload_posts_multipart_to_azure_files_with_the_deployment_key(gateway: Gateway) -> None:
    """The caller uploads `notes.txt` with form `purpose: assistants` and `model: <azure group>`.

    Azure must get one `POST /openai/files?api-version=2024-10-21` with `api-key: synthetic-azure-key`, a
    multipart `purpose` field `assistants` and the file bytes unchanged, and no form `model` field.
    """
    raw: Final = f"file-{uuid.uuid4().hex}"
    with _azure_files(gateway, raw) as rig:
        _upload(rig)
        (upload,) = [r for r in _file_calls(rig) if r.method == "POST"]
        assert (urlsplit(upload.target).path, query_of(upload)) == ("/openai/files", {"api-version": DATED_API_VERSION})
        assert upload.headers.get("api-key") == API_KEY
        fields: Final = multipart_fields(upload)
        assert fields.get("purpose") == "assistants" and "model" not in fields
        assert multipart_file(upload) == FILE_BYTES


def test_the_upload_answer_carries_a_model_encoded_id_not_azures_raw_id(gateway: Gateway) -> None:
    """Azure answers with id `file-<raw>`.

    The caller's `id` must differ from Azure's raw id (it is encoded with the model), while `bytes`,
    `filename` and `purpose` are Azure's values unchanged. UNVERIFIED: the exact encoding is not pinned here.
    """
    raw: Final = f"file-{uuid.uuid4().hex}"
    with _azure_files(gateway, raw) as rig:
        created: Final = _upload(rig)
        assert created["id"] != raw
        assert (created["bytes"], created["filename"], created["purpose"]) == (
            len(FILE_BYTES),
            "notes.txt",
            "assistants",
        )


def test_retrieving_the_encoded_id_reaches_the_same_azure_deployment_with_the_raw_id(gateway: Gateway) -> None:
    """The caller reads `GET /v1/files/<encoded id>` after the upload.

    Azure must get `GET /openai/files/<raw>?api-version=2024-10-21` with the deployment key, and the caller
    must get 200.
    """
    raw: Final = f"file-{uuid.uuid4().hex}"
    with _azure_files(gateway, raw) as rig:
        created: Final = _upload(rig)
        _file_calls(rig)
        read: Final = rig.gateway.request("GET", f"/v1/files/{created['id']}")
        assert read.status_code == 200, read.text
        (get,) = _file_calls(rig)
        assert (get.method, urlsplit(get.target).path, query_of(get)) == (
            "GET",
            f"/openai/files/{raw}",
            {"api-version": DATED_API_VERSION},
        )
        assert get.headers.get("api-key") == API_KEY


def test_file_content_of_the_encoded_id_returns_azures_bytes(gateway: Gateway) -> None:
    """The caller downloads `GET /v1/files/<encoded id>/content`.

    Azure must get `GET /openai/files/<raw>/content?api-version=2024-10-21` and the caller must get exactly the
    file bytes.
    """
    raw: Final = f"file-{uuid.uuid4().hex}"
    with _azure_files(gateway, raw) as rig:
        created: Final = _upload(rig)
        _file_calls(rig)
        content: Final = rig.gateway.request("GET", f"/v1/files/{created['id']}/content")
        assert content.status_code == 200, content.text
        assert content.content == FILE_BYTES
        assert [urlsplit(r.target).path for r in _file_calls(rig)] == [f"/openai/files/{raw}/content"]


def test_deleting_the_encoded_id_deletes_the_raw_azure_file(gateway: Gateway) -> None:
    """The caller deletes `DELETE /v1/files/<encoded id>`.

    Azure must get `DELETE /openai/files/<raw>?api-version=2024-10-21` and the caller must get `deleted: true`.
    """
    raw: Final = f"file-{uuid.uuid4().hex}"
    with _azure_files(gateway, raw) as rig:
        created: Final = _upload(rig)
        _file_calls(rig)
        deleted: Final = rig.gateway.request("DELETE", f"/v1/files/{created['id']}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["deleted"] is True
        assert [(r.method, urlsplit(r.target).path) for r in _file_calls(rig)] == [("DELETE", f"/openai/files/{raw}")]


def test_a_v1_deployment_uploads_to_openai_v1_files_without_an_api_version(gateway: Gateway) -> None:
    """The Azure deployment sets `api_version: v1`.

    The files client is a plain OpenAI client on `{api_base}/openai/v1/` (`get_azure_openai_client`), so Azure
    must get `POST /openai/v1/files` with no `api-version` query.
    """
    raw: Final = f"file-{uuid.uuid4().hex}"
    with _azure_files(gateway, raw, api_version="v1") as rig:
        _upload(rig)
        (upload,) = [r for r in _file_calls(rig) if r.method == "POST"]
        assert (urlsplit(upload.target).path, query_of(upload)) == ("/openai/v1/files", {})


_SEARCH_PAGE: Final = {
    "object": "vector_store.search_results.page",
    "search_query": "q",
    "data": [],
    "has_more": False,
    "next_page": None,
}


@contextmanager
def _azure_store(gateway: Gateway, **litellm_params: object) -> Iterator[tuple[Wire, str]]:
    store_id: Final = f"vs_dec1_{uuid.uuid4().hex}"
    with wire_server(lambda _: json_reply(_SEARCH_PAGE)) as wire, gateway.scenario() as scenario:  # pyright: ignore[reportArgumentType]
        registered: Final = gateway.request(
            "POST",
            "/vector_store/new",
            {
                "vector_store_id": store_id,
                "custom_llm_provider": "azure",
                "litellm_params": {"api_base": wire.url, **litellm_params},  # pyright: ignore[reportArgumentType]
            },
        )
        assert registered.status_code == 200, registered.text
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store_id})
        yield wire, store_id


def _search(gateway: Gateway, store_id: str) -> None:
    response: Final = gateway.request("POST", f"/v1/vector_stores/{store_id}/search", {"query": "q"})
    assert response.status_code == 200, response.text


def test_an_azure_vector_store_search_posts_to_openai_vector_stores_with_the_api_version(gateway: Gateway) -> None:
    """A store registered with `custom_llm_provider: azure`, a key and `api_version: 2024-10-21`.

    Azure must get one `POST /openai/vector_stores/<id>/search?api-version=2024-10-21` with `api-key:
    synthetic-azure-key` and the JSON body `query: "q"`.
    """
    with _azure_store(gateway, api_key=API_KEY, api_version=DATED_API_VERSION) as (wire, store_id):
        _search(gateway, store_id)
        (request,) = [r for r in wire.drain() if r.method == "POST"]
        assert (urlsplit(request.target).path, query_of(request)) == (
            f"/openai/vector_stores/{store_id}/search",
            {"api-version": DATED_API_VERSION},
        )
        assert request.headers.get("api-key") == API_KEY
        assert json.loads(request.body)["query"] == "q"


def test_an_entra_only_azure_vector_store_searches_with_a_bearer(gateway: Gateway) -> None:
    """The store has no key and `azure_ad_token: synthetic-entra-access-token`.

    Azure must get `Authorization: Bearer synthetic-entra-access-token` and no `api-key`. UNVERIFIED: assumes no
    Azure key in the proxy environment.
    """
    with _azure_store(gateway, azure_ad_token=ENTRA_TOKEN, api_version=DATED_API_VERSION) as (wire, store_id):
        _search(gateway, store_id)
        (request,) = [r for r in wire.drain() if r.method == "POST"]
        assert request.headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "api-key" not in request.headers


def test_a_v1_azure_vector_store_searches_under_openai_v1_with_api_version_v1(gateway: Gateway) -> None:
    """The store sets `api_version: v1`.

    `get_base_azure_url` moves the route under `/openai/v1` and keeps the version as a query, so Azure must get
    `POST /openai/v1/vector_stores/<id>/search?api-version=v1`.
    """
    with _azure_store(gateway, api_key=API_KEY, api_version="v1") as (wire, store_id):
        _search(gateway, store_id)
        (request,) = [r for r in wire.drain() if r.method == "POST"]
        assert (urlsplit(request.target).path, query_of(request)) == (
            f"/openai/v1/vector_stores/{store_id}/search",
            {"api-version": "v1"},
        )


def test_an_azure_vector_store_without_an_api_version_still_sends_one(gateway: Gateway) -> None:
    """The store sets a key and no `api_version` (and the proxy has no `AZURE_API_VERSION`).

    Intended: the classic `/openai/vector_stores` route requires an `api-version`, so some version must be sent.
    Today `get_base_azure_url` gets `default_api_version=None` from the vector store config and sends no query
    at all, which Azure answers 404.
    """
    pytest.skip("BUG: an Azure vector store with no api_version is searched with no api-version query")
    with _azure_store(gateway, api_key=API_KEY) as (wire, store_id):
        _search(gateway, store_id)
        (request,) = [r for r in wire.drain() if r.method == "POST"]
        assert "api-version" in query_of(request)
