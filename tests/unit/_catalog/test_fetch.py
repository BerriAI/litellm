import contextlib
import datetime
import ipaddress
import ssl
import threading
import types
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

import litellm
from litellm._catalog import CATALOG, Catalog, Installed, Remote, Unavailable


def _self_signed(directory: Path) -> tuple[Path, Path]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return cert_path, key_path


@contextlib.contextmanager
def _https_serving(directory: Path, body: bytes) -> Generator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    cert_path, key_path = _self_signed(directory)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{httpd.server_address[1]}/model_prices.json"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize(("ssl_verify", "installed"), [(False, True), (True, False)])
def test_the_fetch_uses_the_http_settings_provider_calls_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ssl_verify: bool, installed: bool
):
    monkeypatch.setattr(litellm, "ssl_verify", ssl_verify)
    module = types.ModuleType("fake_litellm")
    module.model_cost = {}
    catalog = Catalog.bound_to(module, "model_cost")
    body = Path(litellm.__file__).resolve().parents[1].joinpath("model_prices_and_context_window.json").read_bytes()

    with _https_serving(tmp_path, body) as url:
        result = catalog.reload(Remote(url))

    assert isinstance(result, Installed if installed else Unavailable), getattr(result, "reason", None)


def test_a_reload_of_an_unparseable_url_is_unavailable():
    assert isinstance(CATALOG.fetch(Remote("not a url"), timeout=1, attempts=1), Unavailable)
