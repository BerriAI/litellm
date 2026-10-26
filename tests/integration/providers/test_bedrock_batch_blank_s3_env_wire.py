import contextlib
import datetime
import json
import socket
import socketserver
import ssl
import threading
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel

MODEL_ID: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION: Final = "us-east-1"
BEDROCK_AUTHORITY: Final = f"bedrock.{REGION}.amazonaws.com:443"
BUCKET: Final = "integration-blank-s3-bucket"
ROLE_ARN: Final = "arn:aws:iam::123456789012:role/integration-batch-role"
JOB_ARN_PREFIX: Final = f"arn:aws:bedrock:{REGION}:123456789012:model-invocation-job/"
KMS_KEY: Final = f"arn:aws:kms:{REGION}:123456789012:key/integration-batch-key"
BUCKET_OWNER: Final = "123456789012"
SSE_HEADER_PREFIX: Final = "x-amz-server-side-encryption"


@dataclass(frozen=True, slots=True)
class ConnectProxy:
    url: str
    authorities: SimpleQueue[str]


class _DataConfig(BaseModel):
    s3InputDataConfig: dict[str, str]


class _OutputConfig(BaseModel):
    s3OutputDataConfig: dict[str, str]


class _CreateJob(BaseModel):
    modelId: str
    roleArn: str
    inputDataConfig: _DataConfig
    outputDataConfig: _OutputConfig


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


@contextmanager
def bedrock_tunnel(destination: Wire) -> Generator[ConnectProxy, None, None]:
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
            yield ConnectProxy(f"http://127.0.0.1:{server.server_address[1]}", authorities)
        finally:
            server.shutdown()
            thread.join(timeout=6)


def s3_peer(request: Request) -> Reply:
    assert request.method == "PUT" and request.target.startswith(f"/{BUCKET}/"), request.target
    return Reply(body=b"")


def bedrock_peer(request: Request) -> Reply:
    if request.method == "POST" and request.target == "/model-invocation-job":
        return Reply(body=json.dumps({"jobArn": JOB_ARN_PREFIX + uuid.uuid4().hex}).encode())
    return Reply(status=404, body=b'{"message": "not scripted"}')


def _without_uri(config: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in config.items() if name != "s3Uri"}


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("kms_key", "bucket_owner", "sse_headers", "input_fields", "output_fields"),
    [
        pytest.param("", "", {}, {}, {}, id="blank"),
        pytest.param(
            KMS_KEY,
            BUCKET_OWNER,
            {SSE_HEADER_PREFIX: "aws:kms", f"{SSE_HEADER_PREFIX}-aws-kms-key-id": KMS_KEY},
            {"s3BucketOwner": BUCKET_OWNER},
            {"s3BucketOwner": BUCKET_OWNER, "s3EncryptionKeyId": KMS_KEY},
            id="set",
        ),
    ],
)
def test_unified_bedrock_batch_sends_s3_env_settings_only_when_they_are_non_blank(
    gateway: Gateway,
    tmp_path: Path,
    kms_key: str,
    bucket_owner: str,
    sse_headers: Mapping[str, str],
    input_fields: Mapping[str, str],
    output_fields: Mapping[str, str],
) -> None:
    environment: Final = {
        "AWS_S3_ENCRYPTION_KEY_ID": kms_key,
        "AWS_S3_BUCKET_OWNER": bucket_owner,
        "SSL_VERIFY": "False",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    with (
        wire_server(s3_peer) as s3,
        wire_server(bedrock_peer, tls=_tls_context(tmp_path)) as bedrock,
        bedrock_tunnel(bedrock) as tunnel,
        owned_proxy(gateway, tmp_path, {**environment, "HTTPS_PROXY": tunnel.url}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"bedrock/{MODEL_ID}",
            api_key=None,
            api_base=None,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name=REGION,
            s3_bucket_name=BUCKET,
            s3_endpoint_url=s3.url,
            aws_batch_role_arn=ROLE_ARN,
        )
        line: Final = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        }
        uploaded: Final = candidate.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("in.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = candidate.request(
            "POST",
            "/v1/batches",
            {"input_file_id": uploaded.json()["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        )
        assert created.status_code == 200, created.text
        assert created.json()["object"] == "batch" and created.json()["status"] == "validating", created.text

        puts: Final = s3.drain()
        assert len(puts) == 1, [put.target for put in puts]
        assert {
            name: value for name, value in puts[0].headers.items() if name.startswith(SSE_HEADER_PREFIX)
        } == sse_headers

        assert BEDROCK_AUTHORITY in {tunnel.authorities.get_nowait() for _ in range(tunnel.authorities.qsize())}
        jobs: Final = tuple(request for request in bedrock.drain() if request.method == "POST")
        assert len(jobs) == 1, [job.target for job in jobs]
        job: Final = _CreateJob.model_validate_json(jobs[0].body)
        assert job.modelId == MODEL_ID and job.roleArn == ROLE_ARN
        assert job.inputDataConfig.s3InputDataConfig["s3Uri"] == f"s3:/{puts[0].target}"
        assert _without_uri(job.inputDataConfig.s3InputDataConfig) == input_fields
        assert _without_uri(job.outputDataConfig.s3OutputDataConfig) == output_fields
