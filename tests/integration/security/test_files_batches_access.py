import base64
import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Final
from urllib.parse import quote, urlsplit

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MANAGED_FILE_ROW: Final = 'SELECT model_mappings, flat_model_file_ids, created_by, team_id FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s'
INPUT_FILENAME: Final = "in.jsonl"
MANAGED_PREFIX: Final = "litellm_proxy:"
BUCKET: Final = "integration-delete-bucket"


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


@pytest.mark.parametrize(
    "surface",
    [
        "upload_form",
        "upload_query",
        "upload_header",
        "batch_encoded_input",
        "batch_body_model",
        "file_get_encoded",
        "file_content_encoded",
        "batch_get_encoded",
    ],
    ids=[
        "upload-form-model",
        "upload-query-model",
        "upload-header-model",
        "batch-create-encoded-input",
        "batch-create-body-model",
        "file-retrieve-encoded-id",
        "file-content-encoded-id",
        "batch-retrieve-encoded-id",
    ],
)
def test_a_key_granted_one_model_cannot_spend_another_deployments_files_or_batches(
    gateway: Gateway, surface: str
) -> None:
    key_allowed: Final = "provider-key-allowed-" + uuid.uuid4().hex[:8]
    key_forbidden: Final = "provider-key-forbidden-" + uuid.uuid4().hex[:8]
    with wire_server(_openai_backend()) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_allowed)
        forbidden: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_forbidden)
        _wait_until_every_worker_serves(gateway, allowed, forbidden)
        caller: Final = _member(scenario, allowed)
        wire.drain()

        encoded_forbidden_file: Final = encode_file_id_with_model(file_id="file-raw-x", model=forbidden)
        encoded_forbidden_batch: Final = encode_file_id_with_model(
            file_id="batch_raw_x", model=forbidden, id_type="batch"
        )
        if surface == "upload_form":
            response: Final = gateway.request_multipart(
                "/v1/files",
                {"purpose": "batch", "model": forbidden},
                {"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                key=caller.key,
            )
        elif surface == "upload_query":
            response = gateway.client.post(
                "/v1/files",
                params={"model": forbidden},
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                headers={"Authorization": f"Bearer {caller.key}"},
            )
        elif surface == "upload_header":
            response = gateway.client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, _jsonl(forbidden), "application/jsonl")},
                headers={"Authorization": f"Bearer {caller.key}", "x-litellm-model": forbidden},
            )
        elif surface == "batch_encoded_input":
            response = gateway.request(
                "POST",
                "/v1/batches",
                {
                    "input_file_id": encoded_forbidden_file,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                key=caller.key,
            )
        elif surface == "batch_body_model":
            response = gateway.request(
                "POST",
                "/v1/batches",
                {
                    "input_file_id": "file-raw-x",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "model": forbidden,
                },
                key=caller.key,
            )
        elif surface == "file_get_encoded":
            response = gateway.request("GET", f"/v1/files/{encoded_forbidden_file}", key=caller.key)
        elif surface == "file_content_encoded":
            response = gateway.request("GET", f"/v1/files/{encoded_forbidden_file}/content", key=caller.key)
        else:
            response = gateway.request("GET", f"/v1/batches/{encoded_forbidden_batch}", key=caller.key)
        assert response.status_code == 403, response.text
        assert _error_message(response, 403) == (
            f"The requested model '{forbidden}' is not available for this API key, or the model name is invalid. "
            "Check the models available to you and try again."
        ), response.text
        assert _drained_other_than_model_list_probes(wire) == (), (
            f"{surface} reached the provider: {response.status_code}"
        )

        control: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": allowed},
            {"file": (INPUT_FILENAME, _jsonl(allowed), "application/jsonl")},
            key=caller.key,
        )
        assert control.status_code == 200, control.text
        (control_request,) = _drained_other_than_model_list_probes(wire)
        assert control_request.headers["authorization"] == f"Bearer {key_allowed}", control_request.headers


def _s3_backend() -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "DELETE":
            return Reply(status=204, body=b"")
        return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)

    return respond


@pytest.mark.parametrize("scheme", ["s3", "gs"], ids=["s3-uri", "gs-uri"])
def test_non_admin_cannot_delete_a_raw_cloud_storage_uri_but_admin_can(gateway: Gateway, scheme: str) -> None:
    with wire_server(_s3_backend()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            api_key=None,
            api_base=None,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name="us-east-1",
            s3_bucket_name=BUCKET,
            s3_endpoint_url=wire.url,
        )
        _wait_until_every_worker_serves(gateway, model)
        caller_user: Final = scenario.user(user_role="internal_user")
        caller_key: Final = scenario.key(user_id=caller_user, models=[model])
        object_uri: Final = f"{scheme}://{BUCKET}/litellm-bedrock-files-obj.jsonl"
        encoded_path: Final = "/v1/files/" + quote(object_uri, safe="")
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

        if scheme == "s3":
            deleted: Final = gateway.request("DELETE", encoded_path, params={"model": model})
            assert deleted.status_code == 200, deleted.text
            parsed: Final = _FileDeleted.model_validate_json(deleted.content)
            assert parsed.deleted is True and parsed.id == object_uri, deleted.text
            (request,) = _drained_other_than_model_list_probes(wire)
            assert request.method == "DELETE" and request.target == f"/{BUCKET}/litellm-bedrock-files-obj.jsonl", (
                request.target
            )
            assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 "), request.headers
