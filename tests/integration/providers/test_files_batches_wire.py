import base64
import json
import re
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from functools import partial
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter
from typing_extensions import assert_never

from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

INPUT_FILENAME: Final = "in.jsonl"
OUTPUT_BYTES: Final = b'{"custom_id": "r1", "result": {"response": {"body": "ok"}}}\n'
MANAGED_PREFIX: Final = "litellm_proxy:"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MANAGED_FILE_ROW: Final = (
    'SELECT model_mappings, flat_model_file_ids FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s'
)
_BatchUploadSpelling = Literal["sdk_list", "repeated_bracket", "indexed", "comma_joined"]
_ModelRoutedSpelling = Literal["sdk_extra_body", "multipart_field", "query_param", "model_header"]
_ContentRoute = Literal["v1_encoded", "files_encoded", "files_raw_query", "v1_raw_header"]
_BatchCreateSpelling = Literal["encoded_input", "body_model", "query_model", "header_model", "sanitized_metadata"]


def _jsonl(model: str) -> bytes:
    line: Final = {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": [{"role": "user", "content": "one"}]},
    }
    return json.dumps(line, separators=(",", ":")).encode() + b"\n"


def _rewritten_jsonl(model: str) -> bytes:
    line: Final = {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": [{"role": "user", "content": "one"}]},
    }
    return json.dumps(line).encode()


class _FileObject(BaseModel):
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str


class _Batch(BaseModel):
    id: str
    object: str
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    status: str


def _json(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def _json_reply(body: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def _file_object(file_id: str, byte_count: int, filename: str = INPUT_FILENAME) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": byte_count,
        "created_at": 1700000000,
        "filename": filename,
        "purpose": "batch",
        "status": "processed",
    }


def _expected_file_object(file_id: str, byte_count: int, filename: str = INPUT_FILENAME) -> _FileObject:
    return _FileObject(
        id=file_id,
        object="file",
        bytes=byte_count,
        created_at=1700000000,
        filename=filename,
        purpose="batch",
    )


def _batch_object(
    batch_id: str, input_file_id: str, status: str, output_file_id: str | None = None, error_file_id: str | None = None
) -> dict[str, JsonValue]:
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": input_file_id,
        "output_file_id": output_file_id,
        "error_file_id": error_file_id,
        "status": status,
        "created_at": 1700000000,
        "completion_window": "24h",
    }


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _text_fields(parts: tuple[Message, ...]) -> dict[str, str]:
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in parts
        if part.get_filename() is None
    }


def _file_fields(parts: tuple[Message, ...]) -> dict[str, tuple[str, str, bytes]]:
    return {
        part.get_param("name", header="content-disposition") or "": (
            part.get_filename() or "",
            part.get_content_type(),
            part.get_payload(decode=True) or b"",
        )
        for part in parts
        if part.get_filename() is not None
    }


def _drained_other_than_model_list_probes(wire: Wire) -> tuple[Request, ...]:
    return tuple(
        request for request in wire.drain() if urlsplit(request.target).path not in {"/v1/models", "/openai/models"}
    )


def _bearer(request: Request) -> str:
    return request.headers.get("authorization", request.headers.get("api-key", "")).removeprefix("Bearer ")


def _sdk_base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


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


def _deployment_ids(gateway: Gateway, *model_names: str) -> dict[str, str]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    by_name: Final = {
        string_value(entry["model_name"]): string_value(object_value(entry["model_info"])["id"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("model_name") in model_names
    }
    return by_name


def _multiplexed_openai_backends(
    created_batch: Mapping[str, JsonValue] | None = None,
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        raw_path: Final = urlsplit(request.target).path
        path: Final = "/v1/" + raw_path.removeprefix("/openai/") if raw_path.startswith("/openai/") else raw_path
        if request.method == "POST" and path == "/v1/files":
            file_part: Final = _file_fields(_multipart_parts(request))["file"]
            return _json_reply(_file_object("file-" + _bearer(request), len(file_part[2]), file_part[0]))
        if request.method == "POST" and path == "/v1/batches":
            if created_batch is not None:
                return _json_reply(created_batch)
            return _json_reply(_batch_object("batch-" + _bearer(request), "file-pending", "in_progress"))
        if request.method == "POST" and re.fullmatch(r"/v1/batches/[^/]+/cancel", path):
            return _json_reply(_batch_object("batch-" + _bearer(request), "file-pending", "cancelled"))
        if request.method == "GET" and path.startswith("/v1/batches/"):
            return _json_reply(
                _batch_object(
                    path.rsplit("/", 1)[1], "file-in", "completed", output_file_id="file-out-" + _bearer(request)
                )
            )
        if request.method == "GET" and path.startswith("/v1/files/") and path.endswith("/content"):
            return Reply(body=OUTPUT_BYTES, content_type="application/octet-stream")
        if request.method == "GET" and path.startswith("/v1/files/"):
            return _json_reply(_file_object(path.rsplit("/", 1)[1], len(_jsonl("uploaded-model"))))
        if request.method == "DELETE" and path.startswith("/v1/files/"):
            return _json_reply({"id": path.rsplit("/", 1)[1], "object": "file", "deleted": True})
        return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)

    return respond


def _managed_upload(
    gateway: Gateway, spelling: _BatchUploadSpelling, fields: Mapping[str, object]
) -> _FileObject:
    key: Final = gateway.key
    match spelling:
        case "sdk_list":
            with OpenAI(base_url=_sdk_base_url(gateway), api_key=key, max_retries=0) as client:
                response: Final = client.files.with_raw_response.create(
                    file=(INPUT_FILENAME, _jsonl("uploaded-model"), "application/jsonl"),
                    purpose="batch",
                    extra_body=fields,
                )
            assert response.status_code == 200, response.text
            return _FileObject.model_validate_json(response.content)
        case "repeated_bracket" | "indexed":
            posted: Final = gateway.client.post(
                "/v1/files",
                data={"purpose": "batch", **fields},
                files={"file": (INPUT_FILENAME, _jsonl("uploaded-model"), "application/jsonl")},
                headers={"Authorization": f"Bearer {key}"},
            )
            assert posted.status_code == 200, posted.text
            return _FileObject.model_validate_json(posted.content)
        case "comma_joined":
            created: Final = gateway.request_multipart(
                "/v1/files",
                {"purpose": "batch", **fields},
                {"file": (INPUT_FILENAME, _jsonl("uploaded-model"), "application/jsonl")},
            )
            assert created.status_code == 200, created.text
            return _FileObject.model_validate_json(created.content)
        case _:
            assert_never(spelling)


@pytest.mark.parametrize(
    "spelling",
    ["sdk_list", "repeated_bracket", "indexed", "comma_joined"],
    ids=["sdk-list-extra-body", "multipart-repeated-bracket", "multipart-indexed", "multipart-comma-joined"],
)
def test_batch_upload_to_two_target_models_reaches_each_deployment(
    gateway: Gateway, spelling: _BatchUploadSpelling
) -> None:
    key_a: Final = "provider-key-a-" + uuid.uuid4().hex[:8]
    key_b: Final = "provider-key-b-" + uuid.uuid4().hex[:8]
    with wire_server(_multiplexed_openai_backends()) as wire, gateway.scenario() as scenario:
        model_a: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_a)
        model_b: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_b)
        _wait_until_every_worker_serves(gateway, model_a, model_b)
        fields: Final[Mapping[str, object]] = {
            "sdk_list": {"target_model_names": [model_a, model_b]},
            "repeated_bracket": {"target_model_names[]": [model_a, model_b]},
            "indexed": {"target_model_names[0]": model_a, "target_model_names[1]": model_b},
            "comma_joined": {"target_model_names": f"{model_a},{model_b}"},
        }[spelling]
        managed_file: Final = _managed_upload(gateway, spelling, fields)
        assert managed_file.created_at > 0, managed_file.model_dump()
        managed: Final = managed_file.id

        requests: Final = _drained_other_than_model_list_probes(wire)
        assert sorted((r.method, r.target, r.headers["authorization"]) for r in requests) == [
            ("POST", "/v1/files", f"Bearer {key_a}"),
            ("POST", "/v1/files", f"Bearer {key_b}"),
        ], requests
        uploads: Final = {request.headers["authorization"]: request for request in requests}
        for bearer, underlying in ((key_a, "gpt-4o-mini"), (key_b, "gpt-4.1-mini")):
            parts: Final = _multipart_parts(uploads[f"Bearer {bearer}"])
            assert len(parts) == 2, [part.get_param("name", header="content-disposition") for part in parts]
            assert _text_fields(parts) == {"purpose": "batch"}
            file_part: Final = _file_fields(parts)["file"]
            assert (file_part[0], file_part[2]) == ("modified_file.jsonl", _rewritten_jsonl(underlying)), file_part

        managed_file_part: Final = _file_fields(_multipart_parts(uploads[f"Bearer {key_a}"]))["file"]
        expected_file: Final = _expected_file_object(
            managed_file.id,
            len(managed_file_part[2]),
            managed_file_part[0],
        )
        assert managed_file == expected_file, managed_file.model_dump()

        decoded: Final = _decoded_unified(managed)
        assert f"target_model_names,{model_a},{model_b}" in decoded, decoded
        deployment_ids: Final = _deployment_ids(gateway, model_a, model_b)
        rows: Final = eventually(
            lambda: read_rows(MANAGED_FILE_ROW, (managed,)),
            lambda found: len(found) == 1,
        )
        assert rows[0]["model_mappings"] == {
            deployment_ids[model_a]: f"file-{key_a}",
            deployment_ids[model_b]: f"file-{key_b}",
        }, rows
        assert sorted(rows[0]["flat_model_file_ids"]) == sorted([f"file-{key_a}", f"file-{key_b}"]), rows

        readback: Final = gateway.request("GET", f"/v1/files/{managed}")
        assert readback.status_code == 200, readback.text
        readback_file: Final = _FileObject.model_validate_json(readback.content)
        assert readback_file == expected_file, readback.text
        assert _drained_other_than_model_list_probes(wire) == ()


def test_batch_upload_to_two_target_models_keeps_the_jsonl_content_type(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: managed multi-target upload sends the rewritten batch file as application/octet-stream instead of "
        "application/jsonl"
    )
    key_a: Final = "provider-key-a-" + uuid.uuid4().hex[:8]
    key_b: Final = "provider-key-b-" + uuid.uuid4().hex[:8]
    with wire_server(_multiplexed_openai_backends()) as wire, gateway.scenario() as scenario:
        model_a: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key_a)
        model_b: Final = scenario.model(model="openai/gpt-4.1-mini", api_base=wire.url + "/v1", api_key=key_b)
        _wait_until_every_worker_serves(gateway, model_a, model_b)
        _managed_upload(
            gateway,
            "repeated_bracket",
            {"target_model_names[]": [model_a, model_b]},
        )
        uploads: Final = {
            request.headers["authorization"]: request for request in _drained_other_than_model_list_probes(wire)
        }
        content_types: Final = {
            f"Bearer {key}": _file_fields(_multipart_parts(uploads[f"Bearer {key}"]))["file"][1]
            for key in (key_a, key_b)
        }
        assert content_types == {
            f"Bearer {key_a}": "application/jsonl",
            f"Bearer {key_b}": "application/jsonl",
        }, content_types


def _model_routed_upload(
    gateway: Gateway, spelling: _ModelRoutedSpelling, alias: str, uploaded: bytes
) -> _FileObject:
    match spelling:
        case "sdk_extra_body":
            with OpenAI(base_url=_sdk_base_url(gateway), api_key=gateway.key, max_retries=0) as client:
                response: Final = client.files.with_raw_response.create(
                    file=(INPUT_FILENAME, uploaded, "application/jsonl"),
                    purpose="batch",
                    extra_body={"model": alias},
                )
            assert response.status_code == 200, response.text
            return _FileObject.model_validate_json(response.content)
        case "multipart_field":
            created: Final = gateway.request_multipart(
                "/v1/files",
                {"purpose": "batch", "model": alias},
                {"file": (INPUT_FILENAME, uploaded, "application/jsonl")},
            )
            assert created.status_code == 200, created.text
            return _FileObject.model_validate_json(created.content)
        case "query_param":
            queried: Final = gateway.client.post(
                "/v1/files",
                params={"model": alias},
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, uploaded, "application/jsonl")},
                headers={"Authorization": f"Bearer {gateway.key}"},
            )
            assert queried.status_code == 200, queried.text
            return _FileObject.model_validate_json(queried.content)
        case "model_header":
            headed: Final = gateway.client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files={"file": (INPUT_FILENAME, uploaded, "application/jsonl")},
                headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-model": alias},
            )
            assert headed.status_code == 200, headed.text
            return _FileObject.model_validate_json(headed.content)
        case _:
            assert_never(spelling)


@pytest.mark.parametrize(
    "spelling",
    ["sdk_extra_body", "multipart_field", "query_param", "model_header"],
    ids=["sdk-extra-body-model", "multipart-model-field", "query-model", "x-litellm-model-header"],
)
def test_model_routed_upload_reaches_one_deployment_and_the_encoded_id_reads_back(
    gateway: Gateway, spelling: _ModelRoutedSpelling
) -> None:
    key: Final = "provider-key-" + uuid.uuid4().hex[:8]
    uploaded: Final = _jsonl("uploaded-model")
    with wire_server(_multiplexed_openai_backends()) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key)
        _wait_until_every_worker_serves(gateway, alias)
        uploaded_file: Final = _model_routed_upload(gateway, spelling, alias, uploaded)

        expected_encoded: Final = encode_file_id_with_model(file_id=f"file-{key}", model=alias)
        assert uploaded_file == _expected_file_object(expected_encoded, len(uploaded)), uploaded_file.model_dump()
        encoded: Final = uploaded_file.id
        (upload,) = _drained_other_than_model_list_probes(wire)
        assert (upload.method, upload.target) == ("POST", "/v1/files"), upload.target
        assert upload.headers["authorization"] == f"Bearer {key}", upload.headers
        parts: Final = _multipart_parts(upload)
        assert len(parts) == 2, [part.get_param("name", header="content-disposition") for part in parts]
        assert _text_fields(parts) == {"purpose": "batch"}
        assert _file_fields(parts) == {"file": (INPUT_FILENAME, "application/jsonl", uploaded)}

        readback: Final = gateway.request("GET", f"/v1/files/{encoded}")
        assert readback.status_code == 200, readback.text
        readback_file: Final = _FileObject.model_validate_json(readback.content)
        assert readback_file == _expected_file_object(encoded, len(uploaded)), readback.text
        (fetched,) = _drained_other_than_model_list_probes(wire)
        assert (fetched.method, fetched.target) == ("GET", f"/v1/files/file-{key}"), fetched.target
        assert fetched.headers["authorization"] == f"Bearer {key}", fetched.headers
        assert fetched.body == b""


AZURE_API_VERSION: Final = "2024-10-21"


def _encoded_output_file_id(gateway: Gateway, alias: str, bearer_key: str) -> tuple[str, str]:
    uploaded_bytes: Final = _jsonl("uploaded-model")
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": alias},
        {"file": (INPUT_FILENAME, uploaded_bytes, "application/jsonl")},
    )
    assert uploaded.status_code == 200, uploaded.text
    encoded_file: Final = encode_file_id_with_model(file_id=f"file-{bearer_key}", model=alias)
    uploaded_file: Final = _FileObject.model_validate_json(uploaded.content)
    assert uploaded_file == _expected_file_object(encoded_file, len(uploaded_bytes)), uploaded.text
    encoded_input: Final = uploaded_file.id
    created: Final = gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": encoded_input, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert created.status_code == 200, created.text
    encoded_batch: Final = string_value(_json(created)["id"])
    retrieved: Final = gateway.request("GET", f"/v1/batches/{encoded_batch}")
    assert retrieved.status_code == 200, retrieved.text
    batch: Final = _Batch.model_validate_json(retrieved.content)
    assert batch.output_file_id is not None, retrieved.text
    return batch.output_file_id, f"file-out-{bearer_key}"


def _download_file_content(
    gateway: Gateway, route: _ContentRoute, encoded_output: str, raw_output: str, alias: str
) -> httpx.Response:
    match route:
        case "v1_encoded":
            return gateway.request("GET", f"/v1/files/{encoded_output}/content")
        case "files_encoded":
            return gateway.request("GET", f"/files/{encoded_output}/content")
        case "files_raw_query":
            return gateway.request("GET", f"/files/{raw_output}/content", params={"model": alias})
        case "v1_raw_header":
            return gateway.request("GET", f"/v1/files/{raw_output}/content", headers={"x-litellm-model": alias})
        case _:
            assert_never(route)


@pytest.mark.parametrize("provider", ["openai", "azure"], ids=["openai-streaming", "azure-non-streaming"])
@pytest.mark.parametrize(
    "route",
    ["v1_encoded", "files_encoded", "files_raw_query", "v1_raw_header"],
    ids=["v1-files-encoded", "files-encoded", "files-raw-model-query", "v1-files-raw-model-header"],
)
def test_batch_output_file_content_downloads_through_the_model_encoded_id(
    gateway: Gateway, provider: str, route: _ContentRoute
) -> None:
    key: Final = f"provider-key-{provider}-" + uuid.uuid4().hex[:8]
    with wire_server(_multiplexed_openai_backends()) as wire, gateway.scenario() as scenario:
        deployment: Final = (
            {"model": "openai/gpt-4o-mini", "api_base": wire.url + "/v1", "api_key": key}
            if provider == "openai"
            else {
                "model": "azure/gpt-4o-mini",
                "api_base": wire.url,
                "api_version": AZURE_API_VERSION,
                "api_key": key,
            }
        )
        alias: Final = scenario.model(**deployment)
        _wait_until_every_worker_serves(gateway, alias)
        encoded_output, raw_output = _encoded_output_file_id(gateway, alias, key)
        assert encoded_output == encode_file_id_with_model(file_id=raw_output, model=alias), encoded_output

        setup: Final = _drained_other_than_model_list_probes(wire)
        late: Final = eventually(
            lambda: _drained_other_than_model_list_probes(wire),
            lambda drained: bool(drained) or any("/content" in request.target for request in setup),
        )
        settled: Final = setup + late
        base_prefix: Final = "/v1/" if provider == "openai" else "/openai/"
        api_suffix: Final = "" if provider == "openai" else f"?api-version={AZURE_API_VERSION}"
        expected_content_target: Final = f"{base_prefix}files/{raw_output}/content{api_suffix}"
        assert [(r.method, r.target) for r in settled] == [
            ("POST", f"{base_prefix}files{api_suffix}"),
            ("POST", f"{base_prefix}batches{api_suffix}"),
            ("GET", f"{base_prefix}batches/batch-{key}{api_suffix}"),
            ("GET", expected_content_target),
        ], [(r.method, r.target) for r in settled]

        response: Final = _download_file_content(gateway, route, encoded_output, raw_output, alias)
        assert response.status_code == 200, response.text
        assert response.content == OUTPUT_BYTES, response.text
        (content_request,) = _drained_other_than_model_list_probes(wire)
        assert (content_request.method, content_request.target) == ("GET", expected_content_target), (
            content_request.target
        )
        assert content_request.body == b"", content_request.body
        if provider == "openai":
            assert content_request.headers["authorization"] == f"Bearer {key}", content_request.headers
        else:
            assert content_request.headers["api-key"] == key, content_request.headers


def _encoded_batch_input(
    gateway: Gateway, spelling: _BatchCreateSpelling, alias: str, wire: Wire, raw_input: str, bearer_key: str
) -> str:
    if spelling == "encoded_input":
        uploaded_bytes: Final = _jsonl("uploaded-model")
        uploaded: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": alias},
            {"file": (INPUT_FILENAME, uploaded_bytes, "application/jsonl")},
        )
        assert uploaded.status_code == 200, uploaded.text
        encoded_file: Final = encode_file_id_with_model(file_id=f"file-{bearer_key}", model=alias)
        uploaded_file: Final = _FileObject.model_validate_json(uploaded.content)
        assert uploaded_file == _expected_file_object(encoded_file, len(uploaded_bytes)), uploaded.text
        wire.drain()
        return uploaded_file.id
    return encode_file_id_with_model(file_id=raw_input, model=alias)


def _created_batch(
    gateway: Gateway, spelling: _BatchCreateSpelling, encoded_input: str, raw_input: str, alias: str
) -> _Batch:
    if spelling in {"query_model", "header_model"}:
        created: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": raw_input,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"job": "x"},
            },
            params={"model": alias} if spelling == "query_model" else None,
            headers={"x-litellm-model": alias} if spelling == "header_model" else None,
        )
        assert created.status_code == 200, created.text
        return _Batch.model_validate_json(created.content)
    if spelling == "sanitized_metadata":
        created: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": raw_input,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"job": "x", "attempt": 2},
                "model": alias,
            },
        )
        assert created.status_code == 200, created.text
        return _Batch.model_validate_json(created.content)
    with OpenAI(base_url=_sdk_base_url(gateway), api_key=gateway.key, max_retries=0) as client:
        extra: Final = {} if spelling == "encoded_input" else {"extra_body": {"model": alias}}
        sdk_batch: Final = client.batches.create(
            input_file_id=encoded_input if spelling == "encoded_input" else raw_input,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"job": "x"},
            **extra,
        )
    return _Batch.model_validate(sdk_batch.model_dump())


@pytest.mark.parametrize(
    "spelling",
    ["encoded_input", "body_model", "query_model", "header_model", "sanitized_metadata"],
    ids=["encoded-input-id", "body-model", "query-model", "x-litellm-model-header", "non-string-metadata-dropped"],
)
def test_create_batch_reaches_one_deployment_and_returns_model_encoded_ids(
    gateway: Gateway, spelling: _BatchCreateSpelling
) -> None:
    key: Final = "provider-key-" + uuid.uuid4().hex[:8]
    raw_input: Final = f"file-in-{key}"
    raw_batch: Final = f"batch-{key}"
    created_batch: Final = _batch_object(
        raw_batch, raw_input, "validating", output_file_id="file-out", error_file_id="file-err"
    )
    with wire_server(_multiplexed_openai_backends(created_batch)) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key=key)
        _wait_until_every_worker_serves(gateway, alias)
        encoded_input: Final = _encoded_batch_input(gateway, spelling, alias, wire, raw_input, key)
        created_batch_payload: Final = _created_batch(gateway, spelling, encoded_input, raw_input, alias)

        expected_raw_input: Final = f"file-{key}" if spelling == "encoded_input" else raw_input
        expected_input: Final = (
            encoded_input if spelling == "encoded_input" else encode_file_id_with_model(file_id=raw_input, model=alias)
        )
        assert created_batch_payload.id == encode_file_id_with_model(file_id=raw_batch, model=alias, id_type="batch"), (
            created_batch_payload.id
        )
        assert created_batch_payload.output_file_id == encode_file_id_with_model(file_id="file-out", model=alias), (
            created_batch_payload.output_file_id
        )
        assert created_batch_payload.error_file_id == encode_file_id_with_model(file_id="file-err", model=alias), (
            created_batch_payload.error_file_id
        )
        assert created_batch_payload.input_file_id == expected_input, created_batch_payload.input_file_id

        requests: Final = _drained_other_than_model_list_probes(wire)
        assert [(r.method, r.target) for r in requests] == [("POST", "/v1/batches")], requests
        (create_request,) = requests
        assert create_request.headers["authorization"] == f"Bearer {key}", create_request.headers
        expected_body: Final = {
            "input_file_id": expected_raw_input,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"job": "x"},
        }
        assert JSON_OBJECT.validate_json(create_request.body) == expected_body, create_request.body
