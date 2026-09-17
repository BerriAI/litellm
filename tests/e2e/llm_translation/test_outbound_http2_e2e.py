"""Outbound HTTP/2 negotiation for LiteLLM-built httpx clients.

Spins up a local hypercorn TLS server that offers h2 and http/1.1 over ALPN and
drives the real AsyncHTTPHandler / HTTPHandler at it, so the negotiated protocol
on the wire is the assertion. No running proxy or provider credentials needed,
which is why these tests carry no `e2e` marker (same shape as the markerless
harness checks under tests/e2e/load/).
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from hypercorn.asyncio import (
    serve,  # pyright: ignore[reportUnknownVariableType]  # hypercorn's serve signature passes through untyped worker hooks
)
from hypercorn.config import Config
from hypercorn.typing import (
    ASGIReceiveCallable,
    ASGISendCallable,
    HTTPResponseBodyEvent,
    HTTPResponseStartEvent,
    Scope,
)

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def _write_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
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


async def _asgi_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    if scope["type"] != "http":
        return
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        if message["type"] == "http.request" and not message["more_body"]:
            break
    if scope["path"] == "/stream":
        await send(
            HTTPResponseStartEvent(
                type="http.response.start", status=200, headers=[(b"content-type", b"text/event-stream")]
            )
        )
        for index in range(3):
            await send(
                HTTPResponseBodyEvent(
                    type="http.response.body", body=f"data: chunk-{index}\n\n".encode(), more_body=True
                )
            )
        await send(HTTPResponseBodyEvent(type="http.response.body", body=b"", more_body=False))
        return
    await send(
        HTTPResponseStartEvent(type="http.response.start", status=200, headers=[(b"content-type", b"application/json")])
    )
    await send(HTTPResponseBodyEvent(type="http.response.body", body=b'{"ok": true}', more_body=False))


@pytest.fixture(scope="module")
def http2_tls_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    cert_dir: Final = tmp_path_factory.mktemp("h2certs")
    cert_file, key_file = _write_self_signed_cert(cert_dir)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: Final = cast(int, sock.getsockname()[1])

    shutdown: Final = threading.Event()

    def _serve() -> None:
        loop: Final = asyncio.new_event_loop()
        config: Final = Config()
        config.bind = [f"127.0.0.1:{port}"]
        config.certfile = str(cert_file)
        config.keyfile = str(key_file)
        config.alpn_protocols = ["h2", "http/1.1"]
        loop.run_until_complete(serve(_asgi_app, config, shutdown_trigger=lambda: asyncio.to_thread(shutdown.wait)))
        loop.close()

    thread: Final = threading.Thread(target=_serve, daemon=True)
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


def _async_exchange(base_url: str) -> tuple[str, str, bytes]:
    async def _run() -> tuple[str, str, bytes]:
        handler: Final = AsyncHTTPHandler(ssl_verify=False)
        try:
            response: Final = await handler.client.post(f"{base_url}/echo", json={"ping": "pong"})
            post_version: Final = response.http_version
            async with handler.client.stream("POST", f"{base_url}/stream", json={}) as stream_response:
                stream_version: Final = stream_response.http_version
                body: Final = b"".join([chunk async for chunk in stream_response.aiter_bytes()])
            return post_version, stream_version, body
        finally:
            await handler.close()

    return asyncio.run(_run())


def _sync_exchange(base_url: str) -> tuple[str, str, bytes]:
    handler: Final = HTTPHandler(ssl_verify=False)
    try:
        response: Final = handler.client.post(f"{base_url}/echo", json={"ping": "pong"})
        post_version: Final = response.http_version
        with handler.client.stream("POST", f"{base_url}/stream", json={}) as stream_response:
            stream_version: Final = stream_response.http_version
            body: Final = b"".join(stream_response.iter_bytes())
        return post_version, stream_version, body
    finally:
        handler.close()


class TestOutboundHttp2:
    @pytest.mark.parametrize("use_http2, expected_version", [(True, "HTTP/2"), (False, "HTTP/1.1")])
    def test_async_handler_negotiates_http2_only_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        http2_tls_server: str,
        use_http2: bool,
        expected_version: str,
    ) -> None:
        monkeypatch.setattr(litellm, "http2", use_http2)
        monkeypatch.delenv("LITELLM_HTTP2", raising=False)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.setattr(litellm, "force_ipv4", False)

        post_version, stream_version, body = _async_exchange(http2_tls_server)

        assert post_version == expected_version
        assert stream_version == expected_version
        assert b"data: chunk-0" in body

    @pytest.mark.parametrize("use_http2, expected_version", [(True, "HTTP/2"), (False, "HTTP/1.1")])
    def test_sync_handler_negotiates_http2_only_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        http2_tls_server: str,
        use_http2: bool,
        expected_version: str,
    ) -> None:
        monkeypatch.setattr(litellm, "http2", use_http2)
        monkeypatch.delenv("LITELLM_HTTP2", raising=False)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.setattr(litellm, "force_ipv4", False)

        post_version, stream_version, body = _sync_exchange(http2_tls_server)

        assert post_version == expected_version
        assert stream_version == expected_version
        assert b"data: chunk-0" in body
