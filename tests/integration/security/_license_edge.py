from __future__ import annotations

import ssl
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Final

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from integration._support.ports import allocate_port

_LICENSE_HOST: Final = "license.litellm.ai"
_SYNTHETIC_LICENSE: Final = "integration-owned-license-edge"


def _certificate(directory: Path) -> tuple[Path, Path, Path]:
    issued: Final = datetime.now(UTC)
    issuer: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Integration license edge CA")])
    authority_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    authority: Final = (
        x509.CertificateBuilder()
        .subject_name(issuer)
        .issuer_name(issuer)
        .public_key(authority_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(issued - timedelta(minutes=5))
        .not_valid_after(issued + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(authority_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(authority_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .sign(authority_key, hashes.SHA256())
    )
    leaf_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf: Final = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _LICENSE_HOST)]))
        .issuer_name(issuer)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(issued - timedelta(minutes=5))
        .not_valid_after(issued + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(authority_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(True, False, True, False, False, False, False, False, False), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(_LICENSE_HOST)]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(authority_key, hashes.SHA256())
    )
    ca: Final = directory / "ca.pem"
    certificate: Final = directory / "license.pem"
    key: Final = directory / "license.key"
    ca.write_bytes(authority.public_bytes(serialization.Encoding.PEM))
    certificate.write_bytes(
        leaf.public_bytes(serialization.Encoding.PEM) + authority.public_bytes(serialization.Encoding.PEM)
    )
    key.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key.chmod(0o600)
    return ca, certificate, key


class _LicenseRequest(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        accepted: Final = (
            self.headers.get("host", "").lower() in (_LICENSE_HOST, f"{_LICENSE_HOST}:443")
            and self.path == f"/verify_license/{_SYNTHETIC_LICENSE}"
        )
        body: Final = b'{"verify":true}' if accepted else b'{"verify":false}'
        self.send_response(200 if accepted else 404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def license_edge(directory: Path) -> Iterator[Mapping[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="license-edge-", dir=directory) as temporary:
        ca, certificate, key = _certificate(Path(temporary))
        context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        port: Final = allocate_port()

        class ConnectRequest(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 5

            def do_CONNECT(self) -> None:
                if self.path != f"{_LICENSE_HOST}:443":
                    self.send_error(403, "Only the owned license destination is supported")
                    return
                self.send_response(200, "Connection Established")
                self.end_headers()
                self.connection.settimeout(5)
                try:
                    with context.wrap_socket(self.connection, server_side=True) as tunnel:
                        _LicenseRequest(tunnel, self.client_address, self.server)
                except (ssl.SSLError, OSError):
                    pass
                finally:
                    self.close_connection = True

            def do_GET(self) -> None:
                self.send_error(403, "Only CONNECT to the owned license destination is supported")

            def log_message(self, format: str, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = False
            block_on_close = True

        with Server(("127.0.0.1", port), ConnectRequest) as server:
            thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
            thread.start()
            proxy: Final = f"http://127.0.0.1:{port}"
            bypass: Final = "127.0.0.1,localhost,::1"
            try:
                yield MappingProxyType(
                    {
                        "HTTPS_PROXY": proxy,
                        "HTTP_PROXY": proxy,
                        "ALL_PROXY": proxy,
                        "https_proxy": proxy,
                        "http_proxy": proxy,
                        "all_proxy": proxy,
                        "NO_PROXY": bypass,
                        "no_proxy": bypass,
                        "SSL_VERIFY": "True",
                        "SSL_CERT_FILE": str(ca),
                        "LITELLM_LICENSE": _SYNTHETIC_LICENSE,
                    }
                )
            finally:
                server.shutdown()
                thread.join(timeout=6)
                assert not thread.is_alive(), "Owned license proxy survived cleanup"
