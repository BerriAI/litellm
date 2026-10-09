import json
import socket
import threading
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, Scenario, gateway_from_environment, string_value
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

PROJECT: Final = "scripted-utc-project"
LOCATION: Final = "us-central1"
BUCKET: Final = "scripted-utc-bucket"
GCS_AUTHORITY: Final = "storage.googleapis.com:443"
JOBS_PATH: Final = f"projects/{PROJECT}/locations/{LOCATION}/batchPredictionJobs"
OUTPUT_PREFIX: Final = f"gs://{BUCKET}/litellm-vertex-files/publishers/google/models/gemini-2.5-flash"
VEO_MODEL: Final = "veo-3.1-generate-001"
VEO_OPERATION: Final = f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{VEO_MODEL}/operations/op-1"

UPLOAD_TIME_CREATED: Final = "2026-10-08T21:15:30.250Z"
UPLOAD_EPOCH: Final = 1791494130
BATCH_CREATE_TIME: Final = "2026-10-08T21:16:05.123456Z"
WHOLE_SECOND_BATCH_CREATE_TIME: Final = "2026-10-08T21:16:05Z"
WHOLE_SECOND_MODEL: Final = "gemini-2.5-flash-lite"
WHOLE_SECOND_JOB_SUFFIX: Final = "1"
BATCH_EPOCH: Final = 1791494165
OUTPUT_TIME_CREATED: Final = "2026-10-08T23:41:07.902Z"
OUTPUT_EPOCH: Final = 1791502867
OUTPUT_BYTES: Final = 2068
VEO_CREATE_TIME: Final = "2026-10-08T22:00:00.500000Z"
VEO_EPOCH: Final = 1791496800

HOST_ZONE: Final = "America/Los_Angeles"


def _gcs_object(name: str, size: int, time_created: str) -> bytes:
    return json.dumps(
        {
            "kind": "storage#object",
            "id": f"{BUCKET}/{name}/1791494130250000",
            "name": name,
            "bucket": BUCKET,
            "size": str(size),
            "contentType": "application/jsonl",
            "timeCreated": time_created,
            "updated": time_created,
        }
    ).encode()


def _batch_job(job_id: str, state: str, output_directory: str | None) -> bytes:
    create_time: Final = (
        WHOLE_SECOND_BATCH_CREATE_TIME if job_id.endswith(WHOLE_SECOND_JOB_SUFFIX) else BATCH_CREATE_TIME
    )
    return json.dumps(
        {
            "name": f"{JOBS_PATH}/{job_id}",
            "displayName": "litellm-vertex-batch-scripted",
            "model": "publishers/google/models/gemini-2.5-flash",
            "state": state,
            "outputInfo": {"gcsOutputDirectory": output_directory} if output_directory else None,
            "createTime": create_time,
            "updateTime": create_time,
        }
    ).encode()


def vertex_peer(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method == "POST" and path == "/_oauth/token":
        return Reply(body=b'{"access_token":"scripted-token","expires_in":3600,"token_type":"Bearer"}')
    if request.method == "POST" and path == f"/upload/storage/v1/b/{BUCKET}/o":
        name: Final = parse_qs(urlsplit(request.target).query)["name"][0]
        return Reply(body=_gcs_object(name, len(request.body), UPLOAD_TIME_CREATED))
    if request.method == "POST" and path.endswith("/batchPredictionJobs"):
        suffix: Final = WHOLE_SECOND_JOB_SUFFIX if WHOLE_SECOND_MODEL.encode() in request.body else "0"
        return Reply(body=_batch_job(str(uuid.uuid4().int)[:18] + suffix, "JOB_STATE_PENDING", None))
    if request.method == "GET" and f"/{JOBS_PATH}/" in path:
        job_id: Final = path.rsplit("/", 1)[-1]
        return Reply(body=_batch_job(job_id, "JOB_STATE_SUCCEEDED", f"{OUTPUT_PREFIX}/prediction-model-{job_id}"))
    if request.method == "GET" and path.startswith(f"/storage/v1/b/{BUCKET}/o/"):
        object_name: Final = path.removeprefix(f"/storage/v1/b/{BUCKET}/o/").replace("%2F", "/")
        return Reply(body=_gcs_object(object_name, OUTPUT_BYTES, OUTPUT_TIME_CREATED))
    if request.method == "POST" and path.endswith(f"/{VEO_MODEL}:predictLongRunning"):
        return Reply(body=json.dumps({"name": VEO_OPERATION}).encode())
    if request.method == "POST" and path.endswith(f"/{VEO_MODEL}:fetchPredictOperation"):
        return Reply(
            body=json.dumps(
                {"name": VEO_OPERATION, "done": False, "metadata": {"createTime": VEO_CREATE_TIME}}
            ).encode()
        )
    return Reply(status=404, body=b'{"error":"not scripted"}')


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@contextmanager
def gcs_tunnel(destination: Wire) -> Generator[str, None, None]:
    """HTTPS_PROXY that sends CONNECT storage.googleapis.com:443 to the TLS peer and refuses every other host."""
    destination_port: Final = urlsplit(destination.url).port

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30

        def do_CONNECT(self) -> None:
            if self.path != GCS_AUTHORITY:
                self.send_error(403)
                return
            self.send_response(200, "Connection Established")
            self.end_headers()
            with socket.create_connection(("127.0.0.1", destination_port), timeout=10) as upstream:
                outbound: Final = threading.Thread(target=_pipe, args=(self.connection, upstream), daemon=True)
                outbound.start()
                _pipe(upstream, self.connection)
                outbound.join(timeout=30)

        def log_message(self, format: str, *args: object) -> None:
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True

    with Server(("127.0.0.1", 0), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=6)
            server.server_close()


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    api: Wire
    gcs: Wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("vertex-utc")
    certificate, key = write_self_signed_cert(directory, names=("storage.googleapis.com",))
    with (
        gateway_from_environment() as environment,
        wire_server(vertex_peer) as api,
        wire_server(vertex_peer, tls=server_context(certificate, key)) as gcs,
        gcs_tunnel(gcs) as tunnel,
        owned_proxy(
            environment,
            directory,
            {
                "TZ": HOST_ZONE,
                "HTTPS_PROXY": tunnel,
                "https_proxy": tunnel,
                "SSL_CERT_FILE": str(certificate),
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
            config=_config_allowing(directory, api),
        ) as candidate,
    ):
        yield Rig(candidate, api, gcs)


def _config_allowing(directory: Path, api: Wire) -> Path:
    shipped: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration: Final = {
        **shipped,
        "litellm_settings": {**shipped["litellm_settings"], "user_url_allowed_hosts": [urlsplit(api.url).netloc]},
    }
    path: Final = directory / "vertex-utc.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


def _vertex_model(rig: Rig, scenario: Scenario, model: str) -> str:
    return scenario.model(
        model=model,
        api_key=None,
        api_base=rig.api.url,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=service_account_json(PROJECT, rig.api.url),
        gcs_bucket_name=BUCKET,
    )


def _object(response_content: bytes) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response_content)


def test_managed_vertex_batch_and_output_file_created_at_are_utc_epochs(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _vertex_model(rig, scenario, "vertex_ai/gemini-2.5-flash")
        owner_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), models=[model])
        line: Final = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        }
        uploaded: Final = rig.gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("input.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
            key=owner_key,
        )
        assert uploaded.status_code == 200, uploaded.text
        upload_object: Final = _object(uploaded.content)

        created: Final = rig.gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(upload_object["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            key=owner_key,
        )
        assert created.status_code == 200, created.text
        batch: Final = _object(created.content)

        retrieved: Final = rig.gateway.request("GET", f"/v1/batches/{string_value(batch['id'])}", key=owner_key)
        assert retrieved.status_code == 200, retrieved.text
        completed: Final = _object(retrieved.content)
        assert completed["status"] == "completed", completed

        output_id: Final = string_value(completed["output_file_id"])
        details: Final = rig.gateway.request("GET", f"/v1/files/{output_id}", key=owner_key)
        assert details.status_code == 200, details.text
        output: Final = _object(details.content)
        assert (output["id"], output["bytes"]) == (output_id, OUTPUT_BYTES), output
        gcs_reads: Final = tuple(
            request for request in rig.gcs.drain() if request.method == "GET" and "/storage/v1/b/" in request.target
        )
        assert gcs_reads, "The output file details never came from the GCS object metadata"

        listed: Final = rig.gateway.request("GET", "/v1/files", key=owner_key)
        assert listed.status_code == 200, listed.text
        listed_files: Final = _object(listed.content)["data"]
        assert isinstance(listed_files, list), listed.text
        listed_output: Final = next(
            (
                JSON_OBJECT.validate_python(file)
                for file in listed_files
                if isinstance(file, dict) and file.get("id") == output_id
            ),
            None,
        )
        assert listed_output is not None, f"Output file {output_id} is absent from GET /v1/files: {listed.text}"

        assert {
            "POST /v1/files": upload_object["created_at"],
            "POST /v1/batches": batch["created_at"],
            "GET /v1/batches/{id}": completed["created_at"],
            "GET /v1/files/{output}": output["created_at"],
            "GET /v1/files": listed_output["created_at"],
        } == {
            "POST /v1/files": UPLOAD_EPOCH,
            "POST /v1/batches": BATCH_EPOCH,
            "GET /v1/batches/{id}": BATCH_EPOCH,
            "GET /v1/files/{output}": OUTPUT_EPOCH,
            "GET /v1/files": OUTPUT_EPOCH,
        }


def test_managed_vertex_batch_with_whole_second_create_time_is_created_and_retrieved(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _vertex_model(rig, scenario, f"vertex_ai/{WHOLE_SECOND_MODEL}")
        owner_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), models=[model])
        line: Final = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        }
        uploaded: Final = rig.gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("input.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
            key=owner_key,
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = rig.gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(_object(uploaded.content)["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            key=owner_key,
        )
        assert created.status_code == 200, created.text
        batch: Final = _object(created.content)
        retrieved: Final = rig.gateway.request("GET", f"/v1/batches/{string_value(batch['id'])}", key=owner_key)
        assert retrieved.status_code == 200, retrieved.text
        assert (batch["created_at"], _object(retrieved.content)["created_at"]) == (BATCH_EPOCH, BATCH_EPOCH)


def test_veo_video_status_created_at_is_a_utc_epoch(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _vertex_model(rig, scenario, f"vertex_ai/{VEO_MODEL}")
        created: Final = rig.gateway.request("POST", "/v1/videos", {"model": model, "prompt": "a scripted cat"})
        assert created.status_code == 200, created.text
        video_id: Final = string_value(_object(created.content)["id"])
        status: Final = rig.gateway.request("GET", f"/v1/videos/{video_id}")
        assert status.status_code == 200, status.text
        video: Final = _object(status.content)
        assert (video["status"], video["created_at"]) == ("processing", VEO_EPOCH), video
