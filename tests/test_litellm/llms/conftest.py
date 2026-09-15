import asyncio
import datetime
import ipaddress
import socket
import threading
import time

import pytest


def _write_self_signed_cert(cert_dir):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "localhost")]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=7))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


async def _asgi_app(scope, receive, send):
    if scope["type"] != "http":
        return
    while True:
        message = await receive()
        if message["type"] == "http.request" and not message.get("more_body"):
            break
        if message["type"] == "http.disconnect":
            return
    if scope["path"] == "/stream":
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for index in range(3):
            await send(
                {
                    "type": "http.response.body",
                    "body": f"data: chunk-{index}\n\n".encode(),
                    "more_body": True,
                }
            )
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"ok": true}'})


@pytest.fixture(scope="module")
def http2_tls_server(tmp_path_factory):
    """Hypercorn TLS server on an ephemeral port that negotiates h2 or http/1.1 via ALPN."""
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    cert_dir = tmp_path_factory.mktemp("h2certs")
    cert_file, key_file = _write_self_signed_cert(cert_dir)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    shutdown = threading.Event()

    def _serve() -> None:
        loop = asyncio.new_event_loop()
        config = Config()
        config.bind = [f"127.0.0.1:{port}"]
        config.certfile = str(cert_file)
        config.keyfile = str(key_file)
        config.alpn_protocols = ["h2", "http/1.1"]
        loop.run_until_complete(serve(_asgi_app, config, shutdown_trigger=lambda: asyncio.to_thread(shutdown.wait)))
        loop.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("hypercorn test server did not start")

    yield f"https://127.0.0.1:{port}"

    shutdown.set()
    thread.join(timeout=10)
