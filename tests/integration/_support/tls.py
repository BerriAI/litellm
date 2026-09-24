from __future__ import annotations

import datetime
import ipaddress
import json
import ssl
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def write_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=7))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file: Final = cert_dir / "cert.pem"
    key_file: Final = cert_dir / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


@dataclass(frozen=True, slots=True)
class TlsPeerRequest:
    path: str
    user: str | None
    cipher: str


def _completion_body(identifier: str, model: str) -> dict[str, JsonValue]:
    return {
        "id": identifier,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "tls ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _response_body(identifier: str, model: str) -> dict[str, JsonValue]:
    return {
        "id": identifier,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "temperature": 1.0,
        "truncation": "disabled",
        "store": True,
        "metadata": {},
        "output": [
            {
                "type": "message",
                "id": "msg_x",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "tls ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    }


def _chunk_body(
    identifier: str, model: str, content: str, role: str | None, finish_reason: str | None
) -> dict[str, JsonValue]:
    delta: dict[str, JsonValue] = {"content": content}
    if role is not None:
        delta["role"] = role
    return {
        "id": identifier,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class _CipherPeer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], owner: TlsPeer) -> None:
        super().__init__(address, _CipherPeerHandler)
        self.owner = owner


def _negotiated_cipher(handler: BaseHTTPRequestHandler) -> str:
    info: Final = cast(ssl.SSLSocket, handler.connection).cipher()
    return info[0] if info else ""


class _CipherPeerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        peer: Final = cast(_CipherPeer, self.server).owner
        body: Final = JSON_OBJECT.validate_json(self.rfile.read(int(self.headers["Content-Length"])))
        user_value: Final = body.get("user")
        user: Final = user_value if isinstance(user_value, str) else None
        peer.record(TlsPeerRequest(path=self.path, user=user, cipher=_negotiated_cipher(self)))
        if user is not None and "slow" in user:
            time.sleep(5)
        identifier: Final = f"chatcmpl-{user or ''}"
        model_value: Final = body.get("model")
        model: Final = model_value if isinstance(model_value, str) else "gpt-4o-mini"
        if self.path.endswith("/responses"):
            response_id: Final = f"resp_{user or ''}"
            responses_payload: Final = json.dumps(_response_body(response_id, model)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(responses_payload)))
            self.end_headers()
            self.wfile.write(responses_payload)
            return
        if body.get("stream") is True:
            chunks: Final = (
                f"data: {json.dumps(_chunk_body(identifier, model, 'tls', 'assistant', None))}\n\n"
                + f"data: {json.dumps(_chunk_body(identifier, model, ' ok', None, 'stop'))}\n\n"
                + "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(chunks)))
            self.end_headers()
            self.wfile.write(chunks)
            return
        payload: Final = json.dumps(_completion_body(identifier, model)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class TlsPeer:
    def __init__(self, cipher: str, cert_file: Path, key_file: Path) -> None:
        self.cipher = cipher
        self.cert_file = cert_file
        self.key_file = key_file
        self.port = 0
        self._server: _CipherPeer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._requests: list[TlsPeerRequest] = []

    @property
    def url(self) -> str:
        return f"https://127.0.0.1:{self.port}"

    def record(self, request: TlsPeerRequest) -> None:
        with self._lock:
            self._requests.append(request)

    def received(self) -> tuple[TlsPeerRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    def start(self) -> None:
        server: Final = _CipherPeer(("127.0.0.1", self.port), self)
        context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(self.cert_file), str(self.key_file))
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers(self.cipher)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        self.port = server.server_address[1]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None


@contextmanager
def tls_peer(cipher: str, cert_file: Path, key_file: Path) -> Generator[TlsPeer, None, None]:
    peer: Final = TlsPeer(cipher, cert_file, key_file)
    peer.start()
    try:
        yield peer
    finally:
        peer.stop()
