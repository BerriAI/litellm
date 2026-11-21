"""Contract: what `/v1/files` sends upstream when no model is named, and which credentials it uses.

A caller that uploads, reads, downloads, lists or deletes a file without `model` or `target_model_names` is on
the "provider" path. The provider comes from the URL prefix, then the `custom-llm-provider` header, then the
`custom_llm_provider` query parameter, then the form body, and defaults to `openai`. The proxy then forwards the
OpenAI Files API request for that provider: `/v1/files...` with `Authorization: Bearer` for OpenAI and
`/openai/files...?api-version=` with `api-key` for Azure.

Credentials differ by operation, and a rewrite must keep the difference. An upload takes the matching
`files_settings` entry. Every other operation never reads `files_settings`: it takes a deployment of that
provider the calling key may use, and only when there is none does it fall back to the environment
(`OPENAI_API_KEY`, `AZURE_API_KEY`). The rig gives each source its own synthetic key so the header the upstream
received names the source.

Assumed: the owned proxy boots from `dec1_resources_support.resource_config`, the managed-files hook is loaded
(it is whenever the enterprise package is importable), and the scripted upstream names an uploaded file
`file-<filename stem>`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration.providers.dec1_resources_support import (
    AMBIENT_OPENAI_KEY,
    AZURE_API_VERSION,
    AZURE_MODEL,
    AZURE_MODEL_KEY,
    FILE_CONTENT,
    FILES_SETTINGS_AZURE_KEY,
    FILES_SETTINGS_OPENAI_KEY,
    JSON_OBJECT,
    OPENAI_MODEL_KEY,
    OTHER_MODEL,
    OTHER_MODEL_KEY,
    Rig,
    batch_line,
    bearer,
    booted_rig,
    jsonl,
    multipart_field,
    multipart_filename,
    multipart_parts,
    path_of,
    query_of,
    unique,
)

_OPENAI_DEPLOYMENT_KEYS: Final = frozenset({OPENAI_MODEL_KEY, OTHER_MODEL_KEY})
_BATCH_INPUT: Final = jsonl(batch_line("gpt-4o-mini"))
_BINARY: Final = bytes(range(256)) * 4


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(tmp_path_factory.mktemp("dec1-resources-files-wire")) as booted:
        yield booted


def test_upload_without_a_provider_goes_to_openai_with_the_files_settings_key(rig: Rig) -> None:
    """A batch file is uploaded with only `purpose` and `file`.

    The upstream must receive exactly one `POST /v1/files` carrying the `files_settings` OpenAI key, not a
    deployment key and not the environment key. A rewrite that resolves credentials the same way for every
    file operation would send the wrong key here.
    """
    stem: Final = unique("default-provider")
    rig.wire.drain()
    response: Final = rig.upload(f"{stem}.jsonl", _BATCH_INPUT)
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert (request.method, path_of(request)) == ("POST", "/v1/files")
    assert bearer(request) == FILES_SETTINGS_OPENAI_KEY


def test_upload_returns_the_provider_file_object_unchanged(rig: Rig) -> None:
    """A batch file is uploaded on the provider path.

    The caller must get status 200 and the upstream's file object field for field: the raw provider id
    `file-<stem>` (no proxy encoding on this path), `object`, `bytes`, `created_at`, `filename`, `purpose` and
    `status`.
    """
    stem: Final = unique("passthrough-object")
    response: Final = rig.upload(f"{stem}.jsonl", _BATCH_INPUT)
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert {name: body[name] for name in ("id", "object", "bytes", "created_at", "filename", "purpose", "status")} == {
        "id": f"file-{stem}",
        "object": "file",
        "bytes": len(_BATCH_INPUT),
        "created_at": 1700000000,
        "filename": f"{stem}.jsonl",
        "purpose": "batch",
        "status": "processed",
    }


@pytest.mark.parametrize("purpose", ["assistants", "fine-tune", "vision", "user_data", "evals"])
def test_upload_forwards_the_bytes_and_purpose_unchanged(rig: Rig, purpose: str) -> None:
    """A 1024 byte binary file covering every byte value is uploaded with a non-batch purpose.

    The multipart body the upstream receives must hold a `purpose` part equal to what the caller sent and a
    `file` part byte-identical to the upload. Re-encoding, newline normalisation or a text decode in the new
    front door would corrupt the file silently.
    """
    stem: Final = unique(f"bytes-{purpose}")
    rig.wire.drain()
    response: Final = rig.upload(f"{stem}.bin", _BINARY, purpose=purpose, content_type="application/octet-stream")
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert multipart_field(request, "purpose") == purpose.encode()
    assert multipart_field(request, "file") == _BINARY


def test_upload_forwards_the_filename_and_only_openai_form_fields(rig: Rig) -> None:
    """A batch file named by the caller is uploaded together with the proxy-only field `custom_llm_provider`.

    The upstream multipart must carry the caller's filename on the `file` part and must not contain the proxy
    routing fields (`custom_llm_provider`, `target_model_names`, `target_storage`, `litellm_metadata`), which
    OpenAI would reject or ignore.
    """
    stem: Final = unique("filename")
    rig.wire.drain()
    response: Final = rig.upload(f"{stem}.jsonl", _BATCH_INPUT, custom_llm_provider="openai")
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert multipart_filename(request) == f"{stem}.jsonl"
    names: Final = {part.get_param("name", header="content-disposition") for part in multipart_parts(request)}
    assert names == {"file", "purpose"}


def test_upload_forwards_expires_after_as_nested_form_fields(rig: Rig) -> None:
    """A file is uploaded with `expires_after[anchor]=created_at` and `expires_after[seconds]=3600`.

    The upstream multipart must carry the same two bracketed fields with the same values, which is how the
    OpenAI SDK serialises the `expires_after` object. UNVERIFIED: the bracketed part names come from the
    OpenAI SDK multipart encoder, not from proxy code; see `route_create_file` in
    `litellm/proxy/openai_files_endpoints/files_endpoints.py`.
    """
    stem: Final = unique("expires")
    rig.wire.drain()
    response: Final = rig.upload(
        f"{stem}.jsonl", _BATCH_INPUT, **{"expires_after[anchor]": "created_at", "expires_after[seconds]": "3600"}
    )
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert multipart_field(request, "expires_after[anchor]") == b"created_at"
    assert multipart_field(request, "expires_after[seconds]") == b"3600"


@pytest.mark.parametrize("surface", ["header", "query", "form", "path"])
def test_upload_naming_azure_reaches_the_azure_files_route_with_the_azure_files_settings(
    rig: Rig, surface: str
) -> None:
    """Azure is named on one of the four surfaces: `custom-llm-provider` header, `custom_llm_provider` query
    parameter, `custom_llm_provider` form field, or the `/azure/v1/files` URL prefix.

    Each must produce one `POST /openai/files?api-version=2024-10-21` whose `api-key` header is the Azure
    `files_settings` key, and no `Authorization: Bearer` OpenAI key.
    """
    stem: Final = unique(f"azure-{surface}")
    files: Final = {"file": (f"{stem}.jsonl", _BATCH_INPUT, "application/jsonl")}
    authorization: Final = {"Authorization": f"Bearer {rig.proxy.key}"}
    rig.wire.drain()
    match surface:
        case "header":
            response = rig.proxy.client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files=files,
                headers={**authorization, "custom-llm-provider": "azure"},
            )
        case "query":
            response = rig.proxy.client.post(
                "/v1/files",
                params={"custom_llm_provider": "azure"},
                data={"purpose": "batch"},
                files=files,
                headers=authorization,
            )
        case "form":
            response = rig.proxy.client.post(
                "/v1/files",
                data={"purpose": "batch", "custom_llm_provider": "azure"},
                files=files,
                headers=authorization,
            )
        case _:
            response = rig.proxy.client.post(
                "/azure/v1/files", data={"purpose": "batch"}, files=files, headers=authorization
            )
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert (request.method, path_of(request)) == ("POST", "/openai/files")
    assert query_of(request) == {"api-version": [AZURE_API_VERSION]}
    assert request.headers.get("api-key") == FILES_SETTINGS_AZURE_KEY


def test_upload_provider_header_wins_over_the_query_parameter(rig: Rig) -> None:
    """The header says `azure` while the query parameter says `openai`.

    The header is read first, so the upload must reach the Azure route `POST /openai/files` and nothing must
    reach `/v1/files`.
    """
    stem: Final = unique("precedence")
    rig.wire.drain()
    response: Final = rig.proxy.client.post(
        "/v1/files",
        params={"custom_llm_provider": "openai"},
        data={"purpose": "batch"},
        files={"file": (f"{stem}.jsonl", _BATCH_INPUT, "application/jsonl")},
        headers={"Authorization": f"Bearer {rig.proxy.key}", "custom-llm-provider": "azure"},
    )
    assert response.status_code == 200, response.text
    assert rig.trail() == [("POST", "/openai/files")]


@pytest.mark.parametrize("prefix", ["/v1", ""])
def test_retrieve_forwards_the_raw_id_and_returns_the_provider_object(rig: Rig, prefix: str) -> None:
    """The master key reads a raw provider file id on `/v1/files/{id}` and on the unversioned alias.

    The upstream must receive one `GET /v1/files/<id>` and the caller must get the upstream's file object with
    the same raw id. The credential is a deployment key the caller may use, never the `files_settings` key,
    because only uploads read `files_settings`.
    """
    file_id: Final = unique("file-retrieve")
    rig.wire.drain()
    response: Final = rig.proxy.request("GET", f"{prefix}/files/{file_id}")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["id"], body["object"], body["purpose"]) == (file_id, "file", "batch")
    (request,) = rig.drained()
    assert (request.method, path_of(request)) == ("GET", f"/v1/files/{file_id}")
    assert bearer(request) in _OPENAI_DEPLOYMENT_KEYS


def test_retrieve_by_a_key_limited_to_one_model_uses_that_deployments_credentials(rig: Rig) -> None:
    """A virtual key whose `models` list holds only the second OpenAI deployment reads a raw file id.

    The upstream request must carry that deployment's key. The lookup is scoped to the key's model allowlist,
    so a rewrite that picks "the first OpenAI deployment" would leak another deployment's provider account.
    """
    file_id: Final = unique("file-scoped")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[OTHER_MODEL])
        rig.wire.drain()
        response: Final = rig.proxy.request("GET", f"/v1/files/{file_id}", key=key)
        assert response.status_code == 200, response.text
        (request,) = rig.drained()
        assert bearer(request) == OTHER_MODEL_KEY


def test_retrieve_by_a_key_with_no_deployment_of_the_provider_falls_back_to_the_environment(rig: Rig) -> None:
    """A virtual key allowed only the Azure deployment reads a raw file id with the default provider `openai`.

    No OpenAI deployment is usable by that key, so the upstream request must carry the environment's
    `OPENAI_API_KEY` and go to `OPENAI_BASE_URL`. The `files_settings` key must not appear.
    """
    file_id: Final = unique("file-ambient")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[AZURE_MODEL])
        rig.wire.drain()
        response: Final = rig.proxy.request("GET", f"/v1/files/{file_id}", key=key)
        assert response.status_code == 200, response.text
        (request,) = rig.drained()
        assert (request.method, path_of(request)) == ("GET", f"/v1/files/{file_id}")
        assert bearer(request) == AMBIENT_OPENAI_KEY


def test_retrieve_naming_azure_in_the_query_reaches_the_azure_route(rig: Rig) -> None:
    """The master key reads a raw file id with `?custom_llm_provider=azure`.

    The upstream must receive `GET /openai/files/<id>?api-version=2024-10-21` with the Azure deployment's
    `api-key`, and the proxy-only query parameter `custom_llm_provider` must not be forwarded.
    """
    file_id: Final = unique("file-azure")
    rig.wire.drain()
    response: Final = rig.proxy.request("GET", f"/v1/files/{file_id}", params={"custom_llm_provider": "azure"})
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert (request.method, path_of(request)) == ("GET", f"/openai/files/{file_id}")
    assert query_of(request) == {"api-version": [AZURE_API_VERSION]}
    assert request.headers.get("api-key") == AZURE_MODEL_KEY


def test_content_download_returns_the_upstream_bytes(rig: Rig) -> None:
    """The master key downloads `/v1/files/{id}/content` for a raw provider id.

    The upstream must receive one `GET /v1/files/<id>/content` and the caller must get status 200 with a body
    byte-identical to what the upstream sent.
    """
    file_id: Final = unique("file-content")
    rig.wire.drain()
    response: Final = rig.proxy.request("GET", f"/v1/files/{file_id}/content")
    assert response.status_code == 200, response.text
    assert response.content == FILE_CONTENT
    assert rig.trail() == [("GET", f"/v1/files/{file_id}/content")]


def test_content_download_of_an_uploaded_file_round_trips_every_byte(rig: Rig) -> None:
    """A 1024 byte binary file is uploaded and then downloaded through `/v1/files/{id}/content`.

    The downloaded body must equal the uploaded bytes exactly, proving neither direction decodes or re-encodes
    the payload.
    """
    stem: Final = unique("roundtrip")
    file_id: Final = rig.uploaded_id(f"{stem}.bin", _BINARY, purpose="assistants")
    assert file_id == f"file-{stem}"
    response: Final = rig.proxy.request("GET", f"/v1/files/{file_id}/content")
    assert response.status_code == 200, response.text
    assert response.content == _BINARY


def test_delete_forwards_the_raw_id_and_returns_the_deletion_object(rig: Rig) -> None:
    """The master key deletes a raw provider file id.

    The upstream must receive one `DELETE /v1/files/<id>` and the caller must get exactly
    `{"id": <id>, "object": "file", "deleted": true}` with status 200.
    """
    file_id: Final = unique("file-delete")
    rig.wire.drain()
    response: Final = rig.proxy.request("DELETE", f"/v1/files/{file_id}")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert {name: body[name] for name in ("id", "object", "deleted")} == {
        "id": file_id,
        "object": "file",
        "deleted": True,
    }
    assert rig.trail() == [("DELETE", f"/v1/files/{file_id}")]


def test_list_naming_a_provider_asks_that_provider_and_forwards_purpose(rig: Rig) -> None:
    """The master key lists files with `?custom_llm_provider=openai&purpose=batch`.

    Naming a provider sends the list upstream: one `GET /v1/files` whose query is exactly `purpose=batch`. The
    upstream answers two files the proxy never uploaded, and the caller gets `object: "list"` with an empty
    `data`, no cursor naming a filtered file, and no further page.
    Pinned current behaviour: with the managed-files hook loaded, its post-call hook filters a provider file
    page down to the files the caller created through the proxy and rebuilds the cursors from what is left
    (enterprise/litellm_enterprise/proxy/hooks/managed_files.py:2430), and that applies to the master key too.
    """
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "GET", "/v1/files", params={"custom_llm_provider": "openai", "purpose": "batch"}
    )
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["object"], body["data"]) == ("list", []), body
    assert (body.get("first_id"), body.get("last_id")) == (None, None), body
    assert body.get("has_more") is not True, body
    (request,) = rig.drained()
    assert (request.method, path_of(request)) == ("GET", "/v1/files")
    assert query_of(request) == {"purpose": ["batch"]}


def test_list_naming_a_provider_forwards_limit_and_after(rig: Rig) -> None:
    """The master key lists files with `?custom_llm_provider=openai&limit=2&after=file-cursor`.

    Intended: the upstream `GET /v1/files` carries `limit=2` and `after=file-cursor`, so a caller can page a
    provider-backed listing. Today `list_files` passes only `purpose` to `litellm.afile_list`, so both
    parameters are dropped and every page is the first page.
    """
    pytest.skip("BUG: GET /v1/files?custom_llm_provider=... drops limit and after before the upstream call")
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "GET", "/v1/files", params={"custom_llm_provider": "openai", "limit": "2", "after": "file-cursor"}
    )
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert query_of(request) == {"limit": ["2"], "after": ["file-cursor"]}


def test_list_without_a_provider_is_answered_from_the_proxy_database(rig: Rig) -> None:
    """A fresh virtual key lists files with no provider, model or `target_model_names`.

    With the managed-files hook loaded this is a listing of the caller's own managed files, read from
    `LiteLLM_ManagedFileTable`. A key that never uploaded gets status 200, `object: "list"` and an empty
    `data`, and no request reaches any provider.
    """
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.wire.drain()
        response: Final = rig.proxy.request("GET", "/v1/files", key=key)
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["object"], body["data"]) == ("list", [])
        assert rig.drained() == ()


def test_plain_file_reads_carry_no_proxy_credentials_upstream(rig: Rig) -> None:
    """A virtual key reads a raw file id.

    The upstream request must not contain the caller's virtual key or the master key in any header value, so
    the provider never sees proxy credentials.
    """
    file_id: Final = unique("file-no-leak")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.wire.drain()
        assert rig.proxy.request("GET", f"/v1/files/{file_id}", key=key).status_code == 200
        (request,) = rig.drained()
        leaked: Final = [name for name, value in request.headers.items() if key in value or rig.proxy.key in value]
        assert leaked == []
