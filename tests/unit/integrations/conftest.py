import functools
import http.server
import ipaddress
import os
import queue
import ssl
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest


@dataclass(frozen=True, slots=True)
class TlsSink:
    url: str
    certificate_path: str
    received: "queue.Queue[str]"


class _RecordingOtelHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args: object, received: "queue.Queue[str]", **kwargs: object) -> None:
        self._received: Final = received
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        length: Final = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self._received.put(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def write_self_signed_cert(directory: Path, stem: str) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path: Final = directory / f"{stem}.crt"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path: Final = directory / f"{stem}.key"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


@pytest.fixture(autouse=True)
def restore_process_environment() -> Iterator[None]:
    original: Final = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def tls_sink(tmp_path: Path) -> Iterator[TlsSink]:
    certificate_path, key_path = write_self_signed_cert(tmp_path, "sink")
    received: queue.Queue[str] = queue.Queue()
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate_path), str(key_path))
    server: Final = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_RecordingOtelHandler, received=received)
    )
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield TlsSink(
        url=f"https://127.0.0.1:{server.server_port}",
        certificate_path=str(certificate_path),
        received=received,
    )
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
