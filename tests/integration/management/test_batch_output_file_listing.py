from __future__ import annotations

import contextlib
import datetime
import json
import socket
import socketserver
import ssl
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

from litellm.litellm_core_utils.cloud_storage_security import BEDROCK_MANAGED_S3_OUTPUT_PREFIX

OUTPUT_BYTES: Final = 4096
ERROR_BYTES: Final = 512
BEDROCK_MODEL: Final = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_MODEL_ID: Final = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION: Final = "us-east-1"
BEDROCK_AUTHORITY: Final = f"bedrock.{BEDROCK_REGION}.amazonaws.com:443"
BEDROCK_BUCKET: Final = "integration-batch-listing-bucket"
BEDROCK_ROLE_ARN: Final = "arn:aws:iam::123456789012:role/integration-batch-role"
BEDROCK_JOB_ARN_PREFIX: Final = f"arn:aws:bedrock:{BEDROCK_REGION}:123456789012:model-invocation-job/"
BEDROCK_LAST_MODIFIED: Final = "Thu, 02 Oct 2025 12:00:00 GMT"
BEDROCK_OUTPUT_CONTENT: Final = b'{"recordId":"req-1","modelOutput":{}}\n'
_BATCH_PROCESSED_SQL: Final = 'SELECT batch_processed FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id=%s'


@dataclass(frozen=True, slots=True)
class _OpenAIBatch:
    model: str
    owner_key: str
    unrelated_key: str | None
    batch_id: str
    input_file_id: str
    scenario: ScenarioHandle


def _output_content(model: str) -> str:
    return (
        json.dumps(
            {
                "id": "batch_req_$REQUEST_ID",
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "request_id": "$REQUEST_ID",
                    "body": {
                        "id": "chatcmpl-$REQUEST_ID",
                        "object": "chat.completion",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    },
                },
                "error": None,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def _error_content() -> str:
    return (
        json.dumps(
            {
                "id": "batch_req_$REQUEST_ID",
                "custom_id": "req-2",
                "response": {"status_code": 400, "body": {"error": {"message": "rejected"}}},
                "error": {"code": "bad_request", "message": "rejected"},
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def _batch_routes(
    model: str,
    *,
    output_file: bool = True,
    metadata_fails: bool = False,
) -> RoutedResponse:
    completed: Final[dict[str, JsonValue]] = {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": "completed",
        "output_file_id": "file-out-$REQUEST_ID" if output_file else None,
        "error_file_id": "file-err-$REQUEST_ID",
        "created_at": 1,
        "in_progress_at": 1,
        "completed_at": 1,
        "expires_at": 1,
        "request_counts": {"total": 1 if not output_file else 2, "completed": 1 if output_file else 0, "failed": 1},
        "metadata": None,
    }
    output_metadata: Final = JsonResponse(
        content_type="application/json",
        body=(
            {"error": "provider metadata unavailable"}
            if metadata_fails
            else {
                "id": "file-out-$REQUEST_ID",
                "object": "file",
                "purpose": "batch_output",
                "bytes": OUTPUT_BYTES,
                "created_at": 1,
                "filename": "output.jsonl",
                "status": "processed",
            }
        ),
        status=500 if metadata_fails else 200,
    )
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /files": JsonResponse(
                content_type="application/json",
                body={
                    "id": "file-in-$REQUEST_ID",
                    "object": "file",
                    "purpose": "batch",
                    "bytes": 100,
                    "created_at": 1,
                    "filename": "input.jsonl",
                    "status": "processed",
                },
            ),
            "POST /batches": JsonResponse(
                content_type="application/json",
                body={**completed, "status": "validating", "output_file_id": None, "error_file_id": None},
            ),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(content_type="application/json", body=completed),
            "GET /files/file-out-$REQUEST_ID": output_metadata,
            "GET /files/file-err-$REQUEST_ID": JsonResponse(
                content_type="application/json",
                body={
                    "id": "file-err-$REQUEST_ID",
                    "object": "file",
                    "purpose": "batch_output",
                    "bytes": ERROR_BYTES,
                    "created_at": 1,
                    "filename": "errors.jsonl",
                    "status": "processed",
                },
            ),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl",
                body=_output_content(model),
            ),
            "GET /files/file-err-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl",
                body=_error_content(),
            ),
        },
    )


def _create_batch(
    scenario: Scenario,
    routes: RoutedResponse,
    *,
    unrelated_user: bool = False,
) -> _OpenAIBatch:
    handle: Final = register_scenario(f"batch-output-listing-{uuid.uuid4().hex}", routes)
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(api_base=handle.api_base())
    owner_id: Final = scenario.user(user_role="internal_user")
    owner_key: Final = scenario.key(user_id=owner_id, models=[model])
    unrelated_key: Final = (
        scenario.key(user_id=scenario.user(user_role="internal_user"), models=[model]) if unrelated_user else None
    )
    input_content: Final = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        }
    ).encode()
    uploaded: Final = scenario.gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model},
        {"file": ("input.jsonl", input_content, "application/jsonl")},
        key=owner_key,
    )
    assert uploaded.status_code == 200, uploaded.text
    input_file_id: Final = string_value(JSON_OBJECT.validate_json(uploaded.content)["id"])
    created: Final = scenario.gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=owner_key,
    )
    assert created.status_code == 200, created.text
    batch_id: Final = string_value(JSON_OBJECT.validate_json(created.content)["id"])
    return _OpenAIBatch(model, owner_key, unrelated_key, batch_id, input_file_id, handle)


def _retrieve_batch(gateway: Gateway, batch: _OpenAIBatch) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", f"/v1/batches/{batch.batch_id}", key=batch.owner_key)
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _list_files(
    gateway: Gateway,
    key: str,
    *,
    purpose: str | None = None,
) -> tuple[dict[str, JsonValue], ...]:
    response: Final = gateway.request(
        "GET",
        "/v1/files",
        key=key,
        params={"purpose": purpose} if purpose is not None else None,
    )
    assert response.status_code == 200, response.text
    values: Final = JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(values, list), response.text
    return tuple(object_value(value) for value in values)


def _metadata_hit_count(gateway: Gateway, scenario: ScenarioHandle) -> int:
    response: Final = httpx.get(
        f"{gateway.upstream_url}/__observations",
        timeout=15,
        trust_env=False,
    )
    assert response.status_code == 200, response.text
    requests: Final = JSON_OBJECT.validate_json(response.content)["requests"]
    assert isinstance(requests, list), response.text
    return sum(1 for request in requests if _is_metadata_hit(request, scenario.scenario_id))


def _is_metadata_hit(request: JsonValue, scenario_id: str) -> bool:
    if not isinstance(request, dict):
        return False
    path: Final = request.get("path")
    return request.get("method") == "GET" and isinstance(path, str) and path.endswith(f"/files/file-out-{scenario_id}")


def test_output_file_lists_after_owner_retrieves_completed_batch(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_batch(scenario, _batch_routes(model="gpt-4o-mini"))
        retrieved: Final = _retrieve_batch(gateway, batch)
        assert retrieved["status"] == "completed", retrieved
        output_id: Final = string_value(retrieved["output_file_id"])
        error_id: Final = string_value(retrieved["error_file_id"])
        files: Final = _list_files(gateway, batch.owner_key)
        output: Final = next((file for file in files if file.get("id") == output_id), None)
        assert output is not None, f"Completed batch output {output_id} is absent from GET /v1/files"
        assert (output["purpose"], output["bytes"]) == ("batch_output", OUTPUT_BYTES), output
        output_files: Final = _list_files(gateway, batch.owner_key, purpose="batch_output")
        assert {string_value(file["id"]) for file in output_files} == {output_id, error_id}, output_files
        assert all(file["purpose"] == "batch_output" for file in output_files), output_files
        input_files: Final = _list_files(gateway, batch.owner_key, purpose="batch")
        assert tuple(string_value(file["id"]) for file in input_files) == (batch.input_file_id,), input_files


def test_poller_registers_listable_output_files_without_a_batch_retrieve(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_batch(
            scenario,
            _batch_routes(model="gpt-4o-mini"),
            unrelated_user=True,
        )
        assert batch.unrelated_key is not None
        output_files: Final = eventually(
            lambda: _list_files(gateway, batch.owner_key, purpose="batch_output"),
            lambda values: len(values) == 2,
            seconds=60,
        )
        assert {file["purpose"] for file in output_files} == {"batch_output"}, output_files
        unrelated_files: Final = _list_files(gateway, batch.unrelated_key, purpose="batch_output")
        assert unrelated_files == (), unrelated_files
        retrieved: Final = _retrieve_batch(gateway, batch)
        output_id: Final = string_value(retrieved["output_file_id"])
        error_id: Final = string_value(retrieved["error_file_id"])
        assert {string_value(file["id"]) for file in output_files} == {output_id, error_id}, output_files


def test_error_file_only_batch_still_lists_its_error_file(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_batch(
            scenario,
            _batch_routes(model="gpt-4o-mini", output_file=False),
        )
        error_files: Final = eventually(
            lambda: _list_files(gateway, batch.owner_key, purpose="batch_output"),
            lambda values: len(values) == 1,
            seconds=60,
        )
        retrieved: Final = _retrieve_batch(gateway, batch)
        assert retrieved["output_file_id"] is None, retrieved
        error_id: Final = string_value(retrieved["error_file_id"])
        assert tuple(string_value(file["id"]) for file in error_files) == (error_id,), error_files
        assert error_files[0]["purpose"] == "batch_output", error_files


def test_output_file_lists_with_basic_details_when_provider_file_lookup_fails(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_batch(
            scenario,
            _batch_routes(model="gpt-4o-mini", metadata_fails=True),
        )
        retrieved: Final = _retrieve_batch(gateway, batch)
        output_id: Final = string_value(retrieved["output_file_id"])
        output: Final = next(
            (file for file in _list_files(gateway, batch.owner_key) if file.get("id") == output_id),
            None,
        )
        assert output is not None, f"Completed batch output {output_id} is absent after provider metadata failure"
        assert (output["purpose"], output["filename"]) == ("batch_output", f"file-out-{batch.scenario.scenario_id}"), (
            output
        )
        assert "litellm_details_fallback" not in output, output


def test_fallback_output_file_details_refresh_once_provider_recovers(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_batch(
            scenario,
            _batch_routes(model="gpt-4o-mini", metadata_fails=True),
            unrelated_user=True,
        )
        retrieved: Final = _retrieve_batch(gateway, batch)
        output_id: Final = string_value(retrieved["output_file_id"])
        listed_files: Final = _list_files(gateway, batch.owner_key, purpose="batch_output")
        basic_output: Final = next((file for file in listed_files if file.get("id") == output_id), None)
        assert basic_output is not None, f"Completed batch output {output_id} is absent after provider metadata failure"
        assert (basic_output["purpose"], basic_output["filename"]) == (
            "batch_output",
            f"file-out-{batch.scenario.scenario_id}",
        ), basic_output
        assert basic_output["bytes"] != OUTPUT_BYTES, basic_output
        assert "litellm_details_fallback" not in basic_output, basic_output

        processed_batch_rows: Final = eventually(
            lambda: read_rows(_BATCH_PROCESSED_SQL, (string_value(retrieved["id"]),)),
            lambda rows: len(rows) == 1 and rows[0].get("batch_processed") is True,
            seconds=60,
        )
        assert processed_batch_rows[0]["batch_processed"] is True, processed_batch_rows
        metadata_hits_before_recovery: Final = _metadata_hit_count(gateway, batch.scenario)
        assert metadata_hits_before_recovery >= 1, "The provider metadata route was not called before recovery"
        register_scenario(
            batch.scenario.scenario_id,
            _batch_routes(model=batch.model),
            control_url=batch.scenario.control_url,
        )
        details_response: Final = gateway.request("GET", f"/v1/files/{output_id}", key=batch.owner_key)
        assert details_response.status_code == 200, details_response.text
        details: Final = JSON_OBJECT.validate_json(details_response.content)
        assert details["bytes"] == OUTPUT_BYTES, details
        assert (details["filename"], details["purpose"]) == ("output.jsonl", "batch_output"), details
        assert "litellm_details_fallback" not in details, details
        assert _metadata_hit_count(gateway, batch.scenario) == 1

        refreshed_files: Final = _list_files(gateway, batch.owner_key, purpose="batch_output")
        refreshed_output: Final = next((file for file in refreshed_files if file.get("id") == output_id), None)
        assert refreshed_output is not None, f"Refreshed batch output {output_id} is absent from the list"
        assert (refreshed_output["bytes"], refreshed_output["filename"]) == (OUTPUT_BYTES, "output.jsonl"), (
            refreshed_output
        )
        assert "litellm_details_fallback" not in refreshed_output, refreshed_output

        assert batch.unrelated_key is not None
        unrelated_details: Final = gateway.request("GET", f"/v1/files/{output_id}", key=batch.unrelated_key)
        assert unrelated_details.status_code == 403, unrelated_details.text


def _tls_context(directory: Path) -> ssl.SSLContext:
    key: Final = ec.generate_private_key(ec.SECP256R1())
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, BEDROCK_AUTHORITY.split(":")[0])])
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    certificate_file: Final = directory / "bedrock.pem"
    key_file: Final = directory / "bedrock.key"
    certificate_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate_file, key_file)
    return context


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@dataclass(frozen=True, slots=True)
class _ConnectProxy:
    url: str
    authorities: SimpleQueue[str]


@contextmanager
def _bedrock_tunnel(destination: Wire) -> Generator[_ConnectProxy, None, None]:
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
            if authority != BEDROCK_AUTHORITY:
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
            yield _ConnectProxy(f"http://127.0.0.1:{server.server_address[1]}", authorities)
        finally:
            server.shutdown()
            thread.join(timeout=6)


@dataclass(frozen=True, slots=True)
class _BedrockControlPlane:
    job_locations: SimpleQueue[tuple[str, str, str]]
    job_id: str
    job_arn: str

    def __call__(self, request: Request) -> Reply:
        if request.method == "POST" and request.target == "/model-invocation-job":
            body: Final = JSON_OBJECT.validate_json(request.body)
            input_config: Final = object_value(object_value(body["inputDataConfig"])["s3InputDataConfig"])
            output_config: Final = object_value(object_value(body["outputDataConfig"])["s3OutputDataConfig"])
            job_name: Final = string_value(body["jobName"])
            self.job_locations.put(
                (string_value(input_config["s3Uri"]), string_value(output_config["s3Uri"]), job_name)
            )
            return Reply(body=json.dumps({"jobArn": self.job_arn}).encode())
        if request.method == "GET" and request.target.endswith(self.job_id):
            input_uri, output_uri, retrieved_job_name = self.job_locations.get()
            self.job_locations.put((input_uri, output_uri, retrieved_job_name))
            return Reply(
                body=json.dumps(
                    {
                        "jobArn": self.job_arn,
                        "jobName": retrieved_job_name,
                        "modelId": BEDROCK_MODEL_ID,
                        "status": "Completed",
                        "submitTime": 1700000000,
                        "lastModifiedTime": 1700000001,
                        "endTime": 1700000002,
                        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": input_uri}},
                        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_uri}},
                        "totalRecordCount": 1,
                        "successRecordCount": 1,
                        "errorRecordCount": 0,
                    }
                ).encode()
            )
        return Reply(status=404, body=b'{"message":"not scripted"}')


def _bedrock_s3_peer(request: Request) -> Reply:
    if request.method == "PUT" and request.target.startswith(f"/{BEDROCK_BUCKET}/"):
        return Reply(body=b"")
    if request.method == "GET" and request.target.startswith(f"/{BEDROCK_BUCKET}/{BEDROCK_MANAGED_S3_OUTPUT_PREFIX}"):
        if request.headers.get("range") == "bytes=0-0":
            return Reply(
                status=206,
                body=BEDROCK_OUTPUT_CONTENT[:1],
                headers={
                    "Content-Range": f"bytes 0-0/{len(BEDROCK_OUTPUT_CONTENT)}",
                    "Last-Modified": BEDROCK_LAST_MODIFIED,
                },
            )
        return Reply(status=200, body=BEDROCK_OUTPUT_CONTENT, headers={"Last-Modified": BEDROCK_LAST_MODIFIED})
    return Reply(status=404, body=b'{"message":"not scripted"}')


def test_bedrock_batch_output_lists_and_retrieves_details(gateway: Gateway, tmp_path: Path) -> None:
    job_id: Final = f"integration-batch-listing-{uuid.uuid4().hex}"
    job_arn: Final = BEDROCK_JOB_ARN_PREFIX + job_id
    environment: Final = {
        "AWS_CA_BUNDLE": str(tmp_path / "bedrock.pem"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "SSL_VERIFY": "False",
    }
    with (
        wire_server(_bedrock_s3_peer) as s3,
        wire_server(_BedrockControlPlane(SimpleQueue(), job_id, job_arn), tls=_tls_context(tmp_path)) as bedrock,
        _bedrock_tunnel(bedrock) as tunnel,
        owned_proxy(gateway, tmp_path, {**environment, "HTTPS_PROXY": tunnel.url}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=BEDROCK_MODEL,
            api_key=None,
            api_base=None,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name=BEDROCK_REGION,
            s3_bucket_name=BEDROCK_BUCKET,
            s3_endpoint_url=s3.url,
            aws_batch_role_arn=BEDROCK_ROLE_ARN,
        )
        owner_id: Final = scenario.user(user_role="internal_user")
        owner_key: Final = scenario.key(user_id=owner_id, models=[model])
        input_line: Final = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        }
        uploaded: Final = candidate.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("input.jsonl", (json.dumps(input_line) + "\n").encode(), "application/jsonl")},
            key=owner_key,
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = candidate.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            key=owner_key,
        )
        assert created.status_code == 200, created.text
        batch_id: Final = string_value(JSON_OBJECT.validate_json(created.content)["id"])
        retrieved: Final = candidate.request("GET", f"/v1/batches/{batch_id}", key=owner_key)
        assert retrieved.status_code == 200, retrieved.text
        batch_object: Final = JSON_OBJECT.validate_json(retrieved.content)
        assert batch_object["status"] == "completed", retrieved.text
        output_id: Final = string_value(batch_object["output_file_id"])
        output_files: Final = eventually(
            lambda: _list_files(candidate, owner_key, purpose="batch_output"),
            lambda values: any(file.get("id") == output_id for file in values),
            seconds=30,
        )
        listed_output: Final = next(file for file in output_files if file.get("id") == output_id)
        details: Final = candidate.request("GET", f"/v1/files/{output_id}", key=owner_key)
        assert details.status_code == 200, details.text
        detail_object: Final = JSON_OBJECT.validate_json(details.content)
        assert (detail_object["id"], detail_object["bytes"], detail_object["purpose"]) == (
            output_id,
            len(BEDROCK_OUTPUT_CONTENT),
            "batch_output",
        ), detail_object
        assert listed_output["id"] == output_id, listed_output
        s3_requests: Final = s3.drain()
        uploads: Final = tuple(request for request in s3_requests if request.method == "PUT")
        assert len(uploads) == 1, f"Expected one input S3 upload, saw {[request.target for request in uploads]}"
        ranged_metadata: Final = tuple(
            request
            for request in s3_requests
            if request.method == "GET" and request.headers.get("range") == "bytes=0-0"
        )
        assert ranged_metadata, "Bedrock file metadata retrieval did not issue a ranged S3 GET"
        assert all(
            request.headers.get("authorization", "").startswith("AWS4-HMAC-SHA256 ") for request in ranged_metadata
        ), ranged_metadata
        authorities: Final = tuple(tunnel.authorities.get_nowait() for _ in range(tunnel.authorities.qsize()))
        assert BEDROCK_AUTHORITY in authorities, authorities
        bedrock_requests: Final = bedrock.drain()
        assert any(request.method == "POST" for request in bedrock_requests), bedrock_requests
        assert any(request.method == "GET" for request in bedrock_requests), bedrock_requests


def test_repeated_batch_retrieve_does_not_refetch_saved_file_details(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_batch(scenario, _batch_routes(model="gpt-4o-mini"))
        first: Final = _retrieve_batch(gateway, batch)
        output_id: Final = string_value(first["output_file_id"])
        output_before: Final = next(
            (file for file in _list_files(gateway, batch.owner_key) if file.get("id") == output_id),
            None,
        )
        assert output_before is not None, f"Completed batch output {output_id} is absent from GET /v1/files"
        hits_before: Final = _metadata_hit_count(gateway, batch.scenario)
        assert hits_before >= 1, "The provider file metadata route was not called for the batch output"
        second: Final = _retrieve_batch(gateway, batch)
        third: Final = _retrieve_batch(gateway, batch)
        assert second["status"] == third["status"] == "completed", (second, third)
        output_after: Final = next(
            (file for file in _list_files(gateway, batch.owner_key) if file.get("id") == output_id),
            None,
        )
        assert output_after == output_before, output_after
        additional_hits: Final = _metadata_hit_count(gateway, batch.scenario)
        assert additional_hits == 0, f"Repeated batch retrieval fetched metadata {additional_hits} more times"
