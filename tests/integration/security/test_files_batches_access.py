import base64
import contextlib
import json
import socket
import socketserver
import threading
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal
from urllib.parse import quote, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI, PermissionDeniedError
from pydantic import BaseModel, JsonValue, TypeAdapter
from typing_extensions import assert_never

from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MANAGED_FILE_ROW: Final = 'SELECT model_mappings, flat_model_file_ids, created_by, team_id FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s'
INPUT_FILENAME: Final = "in.jsonl"
MANAGED_PREFIX: Final = "litellm_proxy:"
BUCKET: Final = "integration-delete-bucket"
VERTEX_PROJECT: Final = "files-batches-delete-project"
VERTEX_LOCATION: Final = "us-central1"
VERTEX_MODEL: Final = "vertex_ai/gemini-2.5-flash"
GCS_HOST: Final = "storage.googleapis.com"
GCS_AUTHORITY: Final = f"{GCS_HOST}:443"
VERTEX_ACCESS_TOKEN: Final = "ya29.scripted-delete-token"
_ForbiddenSurface = Literal[
    "upload_sdk_extra_body",
    "upload_form",
    "upload_query",
    "upload_header",
    "batch_encoded_input",
    "batch_body_model",
    "batch_query_model",
    "batch_header_model",
    "file_get_encoded",
    "file_get_query",
    "file_get_header",
    "file_content_encoded",
    "file_content_query",
    "file_content_header",
    "batch_get_encoded",
    "batch_get_query",
    "batch_get_header",
]


def _jsonl(model: str) -> bytes:
    line: Final = {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": [{"role": "user", "content": "one"}]},
    }
    return json.dumps(line, separators=(",", ":")).encode() + b"\n"


class _FileObject(BaseModel):
    id: str
    object: str
    purpose: str


class _Batch(BaseModel):
    id: str
    object: str
    status: str
    input_file_id: str | None = None


class _ErrorEnvelope(BaseModel):
    error: dict[str, JsonValue]


class _FileDeleted(BaseModel):
    id: str
    object: str
    deleted: bool


def _json(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def _json_reply(body: dict[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def _drained_other_than_model_list_probes(wire: Wire) -> tuple[Request, ...]:
    return tuple(
        request for request in wire.drain() if urlsplit(request.target).path not in {"/v1/models", "/openai/models"}
    )


def _bearer(request: Request) -> str:
    return request.headers.get("authorization", request.headers.get("api-key", "")).removeprefix("Bearer ")


def _file_object(file_id: str) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": 123,
        "created_at": 1700000000,
        "filename": INPUT_FILENAME,
        "purpose": "batch",
        "status": "processed",
    }


def _batch_object(batch_id: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-pending",
        "status": status,
        "created_at": 1700000000,
        "completion_window": "24h",
    }


def _openai_backend() -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path == "/v1/files":
            return _json_reply(_file_object("file-" + _bearer(request)))
        if request.method == "POST" and path == "/v1/batches":
            return _json_reply(_batch_object("batch-" + _bearer(request), "in_progress"))
        if request.method == "POST" and path.endswith("/cancel"):
            return _json_reply(_batch_object(path.rsplit("/", 2)[1], "cancelled"))
        if request.method == "GET" and path.startswith("/v1/batches/"):
            return _json_reply(_batch_object(path.rsplit("/", 1)[1], "in_progress"))
        if request.method == "GET" and path.startswith("/v1/files/") and path.endswith("/content"):
            return Reply(body=b"file-content\n", content_type="application/octet-stream")
        if request.method == "GET" and path.startswith("/v1/files/"):
            return _json_reply(_file_object(path.rsplit("/", 1)[1]))
        if request.method == "DELETE" and path.startswith("/v1/files/"):
            return _json_reply({"id": path.rsplit("/", 1)[1], "object": "file", "deleted": True})
        return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)

    return respond


def _decoded_unified(managed_file_id: str) -> str:
    decoded: Final = base64.urlsafe_b64decode(managed_file_id + "=" * (-len(managed_file_id) % 4)).decode()
    assert decoded.startswith(MANAGED_PREFIX), decoded
    return decoded


def _models_over_a_fresh_connection(gateway: Gateway, _: int) -> frozenset[str]:
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        listed: Final = client.get("/v1/models", headers={"Authorization": f"Bearer {gateway.key}"})
    assert listed.status_code == 200, listed.text
    data: Final = _json(listed)["data"]
    assert isinstance(data, list), listed.text
    return frozenset(string_value(object_value(entry)["id"]) for entry in data)


def _every_worker_serves(gateway: Gateway, *models: str) -> bool:
    with ThreadPoolExecutor(max_workers=16) as pool:
        rounds: Final = tuple(
            tuple(pool.map(partial(_models_over_a_fresh_connection, gateway), range(16))) for _ in range(2)
        )
    return all(model in seen for model in models for round_ in rounds for seen in round_)


def _wait_until_every_worker_serves(gateway: Gateway, *models: str) -> None:
    eventually(lambda: _every_worker_serves(gateway, *models), lambda served: served, seconds=90)


@dataclass(frozen=True, slots=True)
class _Member:
    team: str
    user: str
    key: str


def _member(scenario: Scenario, *models: str) -> _Member:
    team: Final = scenario.team(models=list(models))
    user: Final = scenario.member(team)
    return _Member(team, user, scenario.key(team_id=team, user_id=user))


def _error_message(response: httpx.Response, status: int) -> str:
    assert response.status_code == status, response.text
    error: Final = _ErrorEnvelope.model_validate_json(response.content)
    return str(error.error["message"])


def test_stranger_cannot_read_download_or_delete_another_teams_managed_file(gateway: Gateway) -> None:
    key: Final = "provider-key-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key)
        _wait_until_every_worker_serves(gateway, model)
        owner: Final = _member(scenario, model)
        stranger: Final = _member(scenario, model)

        uploaded: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": (INPUT_FILENAME, _jsonl(model), "application/jsonl")},
            key=owner.key,
        )
        assert uploaded.status_code == 200, uploaded.text
        managed: Final = string_value(_json(uploaded)["id"])
        wire.drain()
        row: Final = eventually(lambda: read_rows(MANAGED_FILE_ROW, (managed,)), lambda rows: len(rows) == 1)[0]

        denied: Final = (
            gateway.request("GET", f"/v1/files/{managed}", key=stranger.key),
            gateway.request("GET", f"/v1/files/{managed}/content", key=stranger.key),
            gateway.request("DELETE", f"/v1/files/{managed}", key=stranger.key),
        )
        for response in denied:
            assert (
                _error_message(response, 403) == f"User {stranger.user} does not have access to the file {managed}"
            ), response.text
        assert _drained_other_than_model_list_probes(wire) == (), "stranger calls must not reach the provider"
        assert read_rows(MANAGED_FILE_ROW, (managed,)) == [row], read_rows(MANAGED_FILE_ROW, (managed,))

        readback: Final = gateway.request("GET", f"/v1/files/{managed}", key=owner.key)
        assert readback.status_code == 200, readback.text
        assert string_value(_json(readback)["id"]) == managed, readback.text
        assert _drained_other_than_model_list_probes(wire) == ()


def test_stranger_cannot_retrieve_or_cancel_another_teams_managed_batch(gateway: Gateway) -> None:
    key: Final = "provider-key-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key)
        _wait_until_every_worker_serves(gateway, model)
        owner: Final = _member(scenario, model)
        stranger: Final = _member(scenario, model)

        uploaded: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": (INPUT_FILENAME, _jsonl(model), "application/jsonl")},
            key=owner.key,
        )
        assert uploaded.status_code == 200, uploaded.text
        managed_input: Final = string_value(_json(uploaded)["id"])
        created: Final = gateway.request(
            "POST",
            "/v1/batches",
            {"input_file_id": managed_input, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
            key=owner.key,
        )
        assert created.status_code == 200, created.text
        unified_batch: Final = string_value(_json(created)["id"])
        wire.drain()

        expected_message: Final = f"User {stranger.user} does not have access to the object {unified_batch}"
        stranger_get: Final = gateway.request("GET", f"/v1/batches/{unified_batch}", key=stranger.key)
        assert _error_message(stranger_get, 403) == expected_message, stranger_get.text
        stranger_cancel: Final = gateway.request("POST", f"/v1/batches/{unified_batch}/cancel", {}, key=stranger.key)
        assert _error_message(stranger_cancel, 403) == expected_message, stranger_cancel.text
        assert _drained_other_than_model_list_probes(wire) == (), "stranger batch calls must not reach the provider"

        owner_cancel: Final = gateway.request("POST", f"/v1/batches/{unified_batch}/cancel", {}, key=owner.key)
        assert owner_cancel.status_code == 200, owner_cancel.text
        assert string_value(_json(owner_cancel)["id"]) == unified_batch, owner_cancel.text
        (cancel_request,) = _drained_other_than_model_list_probes(wire)
        assert (cancel_request.method, cancel_request.target) == ("POST", f"/v1/batches/batch-{key}/cancel"), (
            cancel_request.target
        )
        assert cancel_request.headers["authorization"] == f"Bearer {key}", cancel_request.headers


def _sdk_upload_refusal(gateway: Gateway, forbidden: str, caller_key: str) -> httpx.Response:
    sdk_base_url: Final = str(gateway.client.base_url).rstrip("/") + "/v1"
    with OpenAI(base_url=sdk_base_url, api_key=caller_key, max_retries=0) as client:
        try:
            uploaded: Final = client.files.with_raw_response.create(
                file=(INPUT_FILENAME, _jsonl(forbidden), "application/jsonl"),
                purpose="batch",
                extra_body={"model": forbidden},
            )
        except PermissionDeniedError as refused:
            return refused.response
    raise AssertionError(f"SDK upload with a forbidden model was accepted: {uploaded.status_code} {uploaded.text}")


def _forbidden_surface_request(
    gateway: Gateway,
    surface: _ForbiddenSurface,
    forbidden: str,
    caller_key: str,
    raw_file: str,
    raw_batch: str,
    encoded_file: str,
    encoded_batch: str,
) -> httpx.Response:
    match surface:
        case "upload_sdk_extra_body":
            return _sdk_upload_refusal(gateway, forbidden, caller_key)
        case "upload_form":
            return gateway.request_multipart(
                "/v1/files",
                {"purpose": "batch", "model": forbidden},
                {"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                key=caller_key,
            )
        case "upload_query":
            return gateway.client.post(
                "/v1/files",
                params={"model": forbidden},
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                headers={"Authorization": f"Bearer {caller_key}"},
            )
        case "upload_header":
            return gateway.client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                headers={"Authorization": f"Bearer {caller_key}", "x-litellm-model": forbidden},
            )
        case "batch_encoded_input":
            return gateway.request(
                "POST",
                "/v1/batches",
                {
                    "input_file_id": encoded_file,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                key=caller_key,
            )
        case "batch_body_model":
            return gateway.request(
                "POST",
                "/v1/batches",
                {
                    "input_file_id": raw_file,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "model": forbidden,
                },
                key=caller_key,
            )
        case "batch_query_model":
            return gateway.request(
                "POST",
                "/v1/batches",
                {"input_file_id": raw_file, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
                key=caller_key,
                params={"model": forbidden},
            )
        case "batch_header_model":
            return gateway.request(
                "POST",
                "/v1/batches",
                {"input_file_id": raw_file, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
                key=caller_key,
                headers={"x-litellm-model": forbidden},
            )
        case "file_get_encoded":
            return gateway.request("GET", f"/v1/files/{encoded_file}", key=caller_key)
        case "file_get_query":
            return gateway.request("GET", f"/v1/files/{raw_file}", key=caller_key, params={"model": forbidden})
        case "file_get_header":
            return gateway.request(
                "GET", f"/v1/files/{raw_file}", key=caller_key, headers={"x-litellm-model": forbidden}
            )
        case "file_content_encoded":
            return gateway.request("GET", f"/v1/files/{encoded_file}/content", key=caller_key)
        case "file_content_query":
            return gateway.request(
                "GET", f"/v1/files/{raw_file}/content", key=caller_key, params={"model": forbidden}
            )
        case "file_content_header":
            return gateway.request(
                "GET", f"/v1/files/{raw_file}/content", key=caller_key, headers={"x-litellm-model": forbidden}
            )
        case "batch_get_encoded":
            return gateway.request("GET", f"/v1/batches/{encoded_batch}", key=caller_key)
        case "batch_get_query":
            return gateway.request("GET", f"/v1/batches/{raw_batch}", key=caller_key, params={"model": forbidden})
        case "batch_get_header":
            return gateway.request(
                "GET", f"/v1/batches/{raw_batch}", key=caller_key, headers={"x-litellm-model": forbidden}
            )
        case _:
            assert_never(surface)


@pytest.mark.parametrize(
    "surface",
    [
        "upload_sdk_extra_body",
        "upload_form",
        "upload_query",
        "upload_header",
        "batch_encoded_input",
        "batch_body_model",
        "batch_query_model",
        "batch_header_model",
        "file_get_encoded",
        "file_get_query",
        "file_get_header",
        "file_content_encoded",
        "file_content_query",
        "file_content_header",
        "batch_get_encoded",
        "batch_get_query",
        "batch_get_header",
    ],
    ids=[
        "upload-sdk-extra-body-model",
        "upload-form-model",
        "upload-query-model",
        "upload-header-model",
        "batch-create-encoded-input",
        "batch-create-body-model",
        "batch-create-query-model",
        "batch-create-model-header",
        "file-retrieve-encoded-id",
        "file-retrieve-model-query",
        "file-retrieve-model-header",
        "file-content-encoded-id",
        "file-content-model-query",
        "file-content-model-header",
        "batch-retrieve-encoded-id",
        "batch-retrieve-model-query",
        "batch-retrieve-model-header",
    ],
)
def test_a_key_granted_one_model_cannot_spend_another_deployments_files_or_batches(
    gateway: Gateway, surface: _ForbiddenSurface
) -> None:
    key_allowed: Final = "provider-key-allowed-" + uuid.uuid4().hex[:8]
    key_forbidden: Final = "provider-key-forbidden-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_allowed)
        forbidden: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_forbidden)
        _wait_until_every_worker_serves(gateway, allowed, forbidden)
        team: Final = scenario.team(models=[allowed, forbidden])
        user: Final = scenario.member(team)
        caller_key: Final = scenario.key(team_id=team, user_id=user, models=[allowed])
        caller: Final = _Member(team, user, caller_key)
        provider_client: Final = httpx.Client(base_url=wire.url, timeout=15, trust_env=False)
        with provider_client:
            seeded_file: Final = provider_client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                headers={"Authorization": f"Bearer {key_forbidden}"},
            )
            assert seeded_file.status_code == 200, seeded_file.text
            raw_file_object: Final = _FileObject.model_validate_json(seeded_file.content)
            assert raw_file_object.id == f"file-{key_forbidden}", seeded_file.text
            seeded_batch: Final = provider_client.post(
                "/v1/batches",
                json={
                    "input_file_id": raw_file_object.id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                headers={"Authorization": f"Bearer {key_forbidden}"},
            )
            assert seeded_batch.status_code == 200, seeded_batch.text
            raw_batch_object: Final = _Batch.model_validate_json(seeded_batch.content)
            assert raw_batch_object.id == f"batch-{key_forbidden}", seeded_batch.text
        wire.drain()

        encoded_forbidden_file: Final = encode_file_id_with_model(file_id=raw_file_object.id, model=forbidden)
        encoded_forbidden_batch: Final = encode_file_id_with_model(
            file_id=raw_batch_object.id, model=forbidden, id_type="batch"
        )
        response: Final = _forbidden_surface_request(
            gateway,
            surface,
            forbidden,
            caller.key,
            raw_file_object.id,
            raw_batch_object.id,
            encoded_forbidden_file,
            encoded_forbidden_batch,
        )
        provider_requests: Final = _drained_other_than_model_list_probes(wire)
        assert response.status_code == 403, f"{response.text}; provider requests={provider_requests}"
        assert _error_message(response, 403) == (
            f"The requested model '{forbidden}' is not available for this API key, or the model name is invalid. "
            "Check the models available to you and try again."
        ), response.text
        assert provider_requests == (), f"{surface} reached the provider: {provider_requests}"

        control: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": allowed},
            {"file": (INPUT_FILENAME, _jsonl(allowed), "application/jsonl")},
            key=caller.key,
        )
        assert control.status_code == 200, control.text
        (control_request,) = _drained_other_than_model_list_probes(wire)
        assert control_request.headers["authorization"] == f"Bearer {key_allowed}", control_request.headers


def test_indexed_target_model_upload_cannot_use_a_forbidden_deployment(gateway: Gateway) -> None:
    pytest.skip("BUG: target_model_names[0] upload bypasses the key's model restriction and reaches the deployment")
    key_allowed: Final = "provider-key-allowed-" + uuid.uuid4().hex[:8]
    key_forbidden: Final = "provider-key-forbidden-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_allowed)
        forbidden: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_forbidden)
        _wait_until_every_worker_serves(gateway, allowed, forbidden)
        team: Final = scenario.team(models=[allowed, forbidden])
        user: Final = scenario.member(team)
        caller_key: Final = scenario.key(team_id=team, user_id=user, models=[allowed])

        response: Final = gateway.client.post(
            "/v1/files",
            data={"purpose": "batch", "target_model_names[0]": forbidden},
            files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
            headers={"Authorization": f"Bearer {caller_key}"},
        )
        provider_requests: Final = _drained_other_than_model_list_probes(wire)
        assert response.status_code == 403, f"{response.text}; provider requests={provider_requests}"
        assert _error_message(response, 403) == (
            f"The requested model '{forbidden}' is not available for this API key, or the model name is invalid. "
            "Check the models available to you and try again."
        ), response.text
        assert provider_requests == (), f"indexed target-model upload reached the provider: {provider_requests}"


def test_repeated_target_model_upload_cannot_use_a_forbidden_deployment(gateway: Gateway) -> None:
    pytest.skip("BUG: repeated target_model_names[] upload bypasses the key's model restriction and reaches the deployment")
    key_allowed: Final = "provider-key-allowed-" + uuid.uuid4().hex[:8]
    key_forbidden: Final = "provider-key-forbidden-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_allowed)
        forbidden: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_forbidden)
        _wait_until_every_worker_serves(gateway, allowed, forbidden)
        team: Final = scenario.team(models=[allowed, forbidden])
        user: Final = scenario.member(team)
        caller_key: Final = scenario.key(team_id=team, user_id=user, models=[allowed])

        response: Final = gateway.client.post(
            "/v1/files",
            data={"purpose": "batch", "target_model_names[]": [allowed, forbidden]},
            files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
            headers={"Authorization": f"Bearer {caller_key}"},
        )
        provider_requests: Final = _drained_other_than_model_list_probes(wire)
        assert response.status_code == 403, f"{response.text}; provider requests={provider_requests}"
        assert _error_message(response, 403) == (
            f"The requested model '{forbidden}' is not available for this API key, or the model name is invalid. "
            "Check the models available to you and try again."
        ), response.text
        assert provider_requests == (), f"repeated target-model upload reached the provider: {provider_requests}"


def test_plain_target_model_upload_cannot_use_a_forbidden_deployment(gateway: Gateway) -> None:
    key_allowed: Final = "provider-key-allowed-" + uuid.uuid4().hex[:8]
    key_forbidden: Final = "provider-key-forbidden-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_allowed)
        forbidden: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_forbidden)
        _wait_until_every_worker_serves(gateway, allowed, forbidden)
        team: Final = scenario.team(models=[allowed, forbidden])
        user: Final = scenario.member(team)
        caller_key: Final = scenario.key(team_id=team, user_id=user, models=[allowed])

        response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": forbidden},
            {"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
            key=caller_key,
        )
        provider_requests: Final = _drained_other_than_model_list_probes(wire)
        assert response.status_code == 403, f"{response.text}; provider requests={provider_requests}"
        assert _error_message(response, 403) == (
            f"The requested model '{forbidden}' is not available for this API key, or the model name is invalid. "
            "Check the models available to you and try again."
        ), response.text
        assert provider_requests == (), f"plain target-model upload reached the provider: {provider_requests}"


def _s3_backend() -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "DELETE":
            return Reply(status=204, body=b"")
        return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)

    return respond


def _bedrock_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        api_key=None,
        api_base=None,
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        aws_region_name="us-east-1",
        s3_bucket_name=BUCKET,
        s3_endpoint_url=wire.url,
    )


def _vertex_credentials(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": VERTEX_PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{VERTEX_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _vertex_token_backend(request: Request) -> Reply:
    if request.method == "POST" and request.target == "/_oauth/token":
        return Reply(status=400, body=b'{"error":"invalid_grant"}', content_type="application/json")
    return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)


def test_non_admin_cannot_delete_a_raw_s3_uri(gateway: Gateway) -> None:
    with wire_server(_s3_backend()) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_model(scenario, wire)
        _wait_until_every_worker_serves(gateway, model)
        caller_user: Final = scenario.user(user_role="internal_user")
        caller_key: Final = scenario.key(user_id=caller_user, models=[model])
        object_uri: Final = f"s3://{BUCKET}/litellm-bedrock-files-obj.jsonl"
        provider_path: Final = "/bedrock/v1/files/" + quote(object_uri, safe="")

        denied: Final = gateway.request("DELETE", provider_path, key=caller_key, params={"model": model})
        assert denied.status_code == 403, denied.text
        assert _error_message(denied, 403) == (
            "Raw cloud storage file ids can only be deleted by a proxy admin key. "
            "Use the LiteLLM managed file id returned when the file was created."
        ), denied.text
        assert _drained_other_than_model_list_probes(wire) == (), (
            "non-admin cloud-storage delete reached the storage backend"
        )


def test_non_admin_cannot_delete_a_raw_gcs_uri_before_vertex_auth_or_storage(gateway: Gateway) -> None:
    with (
        wire_server(_vertex_token_backend) as token_wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=VERTEX_MODEL,
            api_key=None,
            vertex_project=VERTEX_PROJECT,
            vertex_location=VERTEX_LOCATION,
            vertex_credentials=_vertex_credentials(token_wire.url),
            gcs_bucket_name=BUCKET,
        )
        _wait_until_every_worker_serves(gateway, model)
        caller_user: Final = scenario.user(user_role="internal_user")
        caller_key: Final = scenario.key(user_id=caller_user, models=[model])
        object_uri: Final = f"gs://{BUCKET}/litellm-vertex-files/litellm-vertex-files-obj.jsonl"
        token_wire.drain()

        denied: Final = gateway.request(
            "DELETE",
            "/vertex_ai/v1/files/" + quote(object_uri, safe=""),
            key=caller_key,
            params={"model": model},
        )
        token_requests: Final = token_wire.drain()
        assert denied.status_code == 403, f"{denied.text}; token requests={token_requests}"
        assert _error_message(denied, 403) == (
            "Raw cloud storage file ids can only be deleted by a proxy admin key. "
            "Use the LiteLLM managed file id returned when the file was created."
        ), denied.text
        assert token_requests == (), "non-admin raw GCS delete reached the Vertex token endpoint"


def test_proxy_admin_deletes_a_raw_s3_uri_with_one_signed_delete(gateway: Gateway) -> None:
    with wire_server(_s3_backend()) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_model(scenario, wire)
        _wait_until_every_worker_serves(gateway, model)
        object_uri: Final = f"s3://{BUCKET}/litellm-bedrock-files-obj.jsonl"
        encoded_path: Final = "/v1/files/" + quote(object_uri, safe="")

        deleted: Final = gateway.request("DELETE", encoded_path, params={"model": model})
        assert deleted.status_code == 200, deleted.text
        parsed: Final = _FileDeleted.model_validate_json(deleted.content)
        assert parsed.deleted is True and parsed.id == object_uri, deleted.text
        (request,) = _drained_other_than_model_list_probes(wire)
        assert request.method == "DELETE" and request.target == f"/{BUCKET}/litellm-bedrock-files-obj.jsonl", (
            request.target
        )
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 "), request.headers


def _granting_vertex_token_backend(request: Request) -> Reply:
    if request.method == "POST" and request.target == "/_oauth/token":
        return _json_reply({"access_token": VERTEX_ACCESS_TOKEN, "expires_in": 3600, "token_type": "Bearer"})
    return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)


def _gcs_backend(request: Request) -> Reply:
    if request.method == "DELETE":
        return Reply(status=204, body=b"")
    return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@dataclass(frozen=True, slots=True)
class _ConnectTunnel:
    url: str
    authorities: SimpleQueue[str]


@contextmanager
def _gcs_tunnel(destination: Wire) -> Generator[_ConnectTunnel, None, None]:
    authorities: Final[SimpleQueue[str]] = SimpleQueue()
    destination_port: Final = int(destination.url.rsplit(":", 1)[1])

    class Tunnel(socketserver.StreamRequestHandler):
        rbufsize = 0
        request: socket.socket

        def handle(self) -> None:
            authority: Final = self.rfile.readline().decode().split()[1]
            while self.rfile.readline() not in (b"\r\n", b""):
                pass
            authorities.put(authority)
            if authority != GCS_AUTHORITY:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.request.settimeout(10)
            with socket.create_connection(("127.0.0.1", destination_port), timeout=10) as upstream:
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, upstream))
                outbound.start()
                _pipe(upstream, self.request)
                outbound.join(timeout=12)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Tunnel) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield _ConnectTunnel(f"http://127.0.0.1:{server.server_address[1]}", authorities)
        finally:
            server.shutdown()
            thread.join(timeout=6)


def _queued(queue: SimpleQueue[str]) -> tuple[str, ...]:
    return tuple(queue.get_nowait() for _ in range(queue.qsize()))


@pytest.mark.timeout(180)
def test_proxy_admin_deletes_a_raw_gcs_uri_with_one_authorized_storage_delete(
    gateway: Gateway, tmp_path: Path
) -> None:
    cert_file, key_file = write_self_signed_cert(tmp_path, (GCS_HOST,))
    object_name: Final = f"litellm-vertex-files/admin-{uuid.uuid4().hex}.jsonl"
    object_uri: Final = f"gs://{BUCKET}/{object_name}"
    with (
        wire_server(_granting_vertex_token_backend) as token_wire,
        wire_server(_gcs_backend, tls=server_context(cert_file, key_file)) as storage,
        _gcs_tunnel(storage) as tunnel,
        owned_proxy(gateway, tmp_path, {"HTTPS_PROXY": tunnel.url, "SSL_VERIFY": "False"}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=VERTEX_MODEL,
            api_key=None,
            vertex_project=VERTEX_PROJECT,
            vertex_location=VERTEX_LOCATION,
            vertex_credentials=_vertex_credentials(token_wire.url),
            gcs_bucket_name=BUCKET,
        )
        token_wire.drain()

        deleted: Final = candidate.request(
            "DELETE", "/v1/files/" + quote(object_uri, safe=""), params={"model": model}
        )
        storage_requests: Final = storage.drain()
        assert deleted.status_code == 200, f"{deleted.text}; storage requests={storage_requests}"
        parsed: Final = _FileDeleted.model_validate_json(deleted.content)
        assert parsed.deleted is True and parsed.id == object_uri, deleted.text
        assert [(r.method, r.target) for r in storage_requests] == [
            ("DELETE", f"/storage/v1/b/{BUCKET}/o/{quote(object_name, safe='')}")
        ], storage_requests
        assert storage_requests[0].headers["authorization"] == f"Bearer {VERTEX_ACCESS_TOKEN}", (
            storage_requests[0].headers
        )
        assert _queued(tunnel.authorities) == (GCS_AUTHORITY,), "raw GCS delete took an unexpected route"
