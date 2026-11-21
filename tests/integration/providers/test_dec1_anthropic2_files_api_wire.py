"""Contract: the OpenAI-shaped `/v1/files` routes reach Anthropic's Files API through an `anthropic/` deployment.

A caller uploads a file with `POST /v1/files` naming an Anthropic deployment as `model`. The proxy takes that
deployment's key and base (`get_authorized_credentials_for_model`), calls `AnthropicFilesConfig`
(`litellm/llms/anthropic/files/transformation.py`), and answers an OpenAI file object whose id is the Anthropic id
wrapped with the model name: `"file-" + urlsafe_b64("litellm:<anthropic id>;model,<model>")` without padding
(`encode_file_id_with_model`, `openai_files_endpoints/common_utils.py:327`). Later retrieve, content and delete
calls decode that id back to the deployment and the raw Anthropic id. Upstream every Files API call carries the
deployment key as `x-api-key`, `anthropic-version: 2023-06-01` and `anthropic-beta: files-api-2025-04-14`
(`_finalize_headers`). The uploaded id is what callers later reference in a message as a `file_id` source.

The files path is not part of the Rust work yet, but it shares the deployment credential lookup that is, so a
regression there shows up here first.
"""

from __future__ import annotations

import base64
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, Gateway
from integration.providers.dec1_anthropic2_support import (
    FILE_BYTES,
    FILE_ID,
    FILES_BETA,
    files_respond,
    multipart_field,
)
from integration.providers.dec1_anthropic_support import PROVIDER_KEY, betas, deployment, peer


def _encoded(file_id: str, model: str) -> str:
    """The proxy's model-scoped file id, spelled out from `encode_file_id_with_model`."""
    return "file-" + base64.urlsafe_b64encode(f"litellm:{file_id};model,{model}".encode()).decode().rstrip("=")


def _upload(gateway: Gateway, model: str) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/files",
        {"purpose": "messages", "model": model},
        {"file": ("dec1-notes.txt", FILE_BYTES, "text/plain")},
    )


def test_an_upload_reaches_the_anthropic_files_endpoint_as_multipart(gateway: Gateway) -> None:
    """The caller uploads `dec1-notes.txt` with `purpose=messages` and `model=<anthropic deployment>`.

    The peer must see one `POST /v1/files` whose multipart body has a `file` part holding the exact bytes and a
    `purpose` part `messages` (`transform_create_file_request`). UNVERIFIED: that `model` is read from the form
    field (`extract_file_creation_params`); it may be the query string instead.
    """
    with peer(files_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False)
        response: Final = _upload(gateway, model)
        assert response.status_code == 200, response.text
        calls: Final = [call for call in provider.all() if call.method == "POST"]
        assert [call.path for call in calls] == ["/v1/files"]
        assert calls[0].headers.get("content-type", "").startswith("multipart/form-data")
        assert FILE_BYTES in calls[0].raw
        assert multipart_field(calls[0].raw, "purpose") == b"messages"


def test_an_upload_carries_the_deployment_key_version_and_files_beta(gateway: Gateway) -> None:
    """The same upload must reach Anthropic with `x-api-key: dec1-anthropic-provider-key`, `anthropic-version:
    2023-06-01` and `files-api-2025-04-14` in `anthropic-beta`, and never with the caller's proxy key.
    """
    with peer(files_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False)
        assert _upload(gateway, model).status_code == 200
        call: Final = next(call for call in provider.all() if call.method == "POST")
        assert call.headers.get("x-api-key") == PROVIDER_KEY
        assert call.headers.get("anthropic-version") == "2023-06-01"
        assert FILES_BETA in betas(call)
        assert all(gateway.key not in value for value in call.headers.values())


def test_the_upload_answer_is_an_openai_file_object_with_a_model_scoped_id(gateway: Gateway) -> None:
    """Anthropic answers `{"id": "file_011dec1synthetic", "filename": "dec1-notes.txt", "size_bytes": 25,
    "created_at": "2026-11-17T08:30:00Z", ...}`.

    The caller must get `object: "file"`, `id` equal to the model-scoped encoding of `file_011dec1synthetic`,
    `filename: "dec1-notes.txt"`, `bytes: 25` (from `size_bytes`), `purpose: "messages"`, `status: "uploaded"` and
    `created_at: 1794904200` (the UTC timestamp of `2026-11-17T08:30:00Z`, `_parse_anthropic_file`).
    """
    with peer(files_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False)
        response: Final = _upload(gateway, model)
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body.get("object") == "file"
        assert body.get("id") == _encoded(FILE_ID, model)
        assert body.get("filename") == "dec1-notes.txt"
        assert body.get("bytes") == len(FILE_BYTES)
        assert body.get("purpose") == "messages"
        assert body.get("status") == "uploaded"
        assert body.get("created_at") == 1794904200


def test_retrieving_a_model_scoped_id_gets_the_raw_anthropic_id(gateway: Gateway) -> None:
    """The caller calls `GET /v1/files/<encoded id>` with the id the upload returned.

    The proxy must decode the deployment and call `GET /v1/files/file_011dec1synthetic` on the peer with the files
    beta, and answer an OpenAI file object whose `bytes` is 25. UNVERIFIED: whether the answered `id` is the
    encoded or the raw id; only the upstream path and `bytes` are pinned.
    """
    with peer(files_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False)
        response: Final = gateway.request("GET", f"/v1/files/{_encoded(FILE_ID, model)}")
        assert response.status_code == 200, response.text
        gets: Final = [call for call in provider.all() if call.method == "GET" and call.path.startswith("/v1/files")]
        assert [call.path for call in gets] == [f"/v1/files/{FILE_ID}"]
        assert FILES_BETA in betas(gets[0])
        assert JSON_OBJECT.validate_json(response.content).get("bytes") == len(FILE_BYTES)


def test_file_content_is_downloaded_from_the_content_path(gateway: Gateway) -> None:
    """The caller calls `GET /v1/files/<encoded id>/content`.

    The peer must see `GET /v1/files/file_011dec1synthetic/content` (`transform_file_content_request`) and the
    caller must get the file's exact bytes back.
    """
    with peer(files_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False)
        response: Final = gateway.request("GET", f"/v1/files/{_encoded(FILE_ID, model)}/content")
        assert response.status_code == 200, response.text
        assert response.content == FILE_BYTES
        assert f"/v1/files/{FILE_ID}/content" in [call.path for call in provider.all()]


def test_deleting_a_model_scoped_id_deletes_the_raw_anthropic_file(gateway: Gateway) -> None:
    """The caller calls `DELETE /v1/files/<encoded id>`.

    The peer must see `DELETE /v1/files/file_011dec1synthetic`, and the caller must get `deleted: true` and
    `object: "file"` (`transform_delete_file_response` always reports a delete Anthropic accepted as deleted).
    """
    with peer(files_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False)
        response: Final = gateway.request("DELETE", f"/v1/files/{_encoded(FILE_ID, model)}")
        assert response.status_code == 200, response.text
        deletes: Final = [call.path for call in provider.all() if call.method == "DELETE"]
        assert deletes == [f"/v1/files/{FILE_ID}"]
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body.get("deleted") is True
        assert body.get("object") == "file"
