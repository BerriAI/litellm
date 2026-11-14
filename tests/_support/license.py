import base64
import functools
import json
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from litellm.proxy.auth.litellm_license import Entitlements, LicenseVerifier


def sign_license(private_key: RSAPrivateKey, claims: Mapping[str, object]) -> str:
    message: Final = json.dumps(dict(claims)).encode()
    signature: Final = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(message + b"." + signature).decode()


def license_verifier(private_key: RSAPrivateKey, license_server: str | None = None) -> LicenseVerifier:
    pem: Final = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    return LicenseVerifier(timeout_seconds=5, public_key_pem=pem, license_server=license_server)


@contextmanager
def serve_license_server(reply: Callable[[str], tuple[int, bytes]]) -> Generator[tuple[str, list[str]]]:
    """Yields the server URL and the paths it was asked for, in order"""
    requested: Final[list[str]] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requested.append(self.path)
            status, body = reply(self.path)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requested
    finally:
        server.shutdown()
        server.server_close()


@functools.cache
def _test_signer() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def enterprise_entitlements(**claims: object) -> Entitlements:
    """Entitlements of a license signed by a throwaway key, expiring in 2999 unless overridden"""
    signer: Final = _test_signer()
    return license_verifier(signer).entitlements(sign_license(signer, {"expiration_date": "2999-01-01", **claims}))
