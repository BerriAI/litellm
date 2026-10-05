from __future__ import annotations

import asyncio
import datetime
import ipaddress
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
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
from hypercorn.typing import ASGIReceiveCallable, ASGISendCallable, HTTPResponseBodyEvent, HTTPResponseStartEvent, Scope

STREAM_CHUNKS: Final = 3


@dataclass(frozen=True, slots=True)
class Observed:
    post_version: str
    post_peer_version: str
    stream_version: str
    stream_body: bytes


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


async def _peer(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    if scope["type"] != "http":
        return
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        if message["type"] == "http.request" and not message["more_body"]:
            break
    version: Final = scope["http_version"]
    if scope["path"] == "/stream":
        await send(
            HTTPResponseStartEvent(
                type="http.response.start", status=200, headers=[(b"content-type", b"text/event-stream")]
            )
        )
        for index in range(STREAM_CHUNKS):
            await send(
                HTTPResponseBodyEvent(
                    type="http.response.body", body=f"data: {version}-{index}\n\n".encode(), more_body=True
                )
            )
        await send(HTTPResponseBodyEvent(type="http.response.body", body=b"", more_body=False))
        return
    await send(
        HTTPResponseStartEvent(type="http.response.start", status=200, headers=[(b"content-type", b"application/json")])
    )
    await send(
        HTTPResponseBodyEvent(
            type="http.response.body", body=json.dumps({"http_version": version}).encode(), more_body=False
        )
    )


@pytest.fixture(scope="module")
def http2_tls_peer(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    cert_file, key_file = _write_self_signed_cert(tmp_path_factory.mktemp("h2certs"))
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
        loop.run_until_complete(serve(_peer, config, shutdown_trigger=lambda: asyncio.to_thread(shutdown.wait)))
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
        pytest.fail("hypercorn peer did not start")
    yield f"https://127.0.0.1:{port}"
    shutdown.set()
    thread.join(timeout=10)


def _async_exchange(base_url: str) -> Observed:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    async def _run() -> Observed:
        handler: Final = AsyncHTTPHandler(ssl_verify=False)
        try:
            response: Final = await handler.client.post(f"{base_url}/echo", json={"ping": "pong"})
            async with handler.client.stream("POST", f"{base_url}/stream", json={}) as stream_response:
                return Observed(
                    post_version=response.http_version,
                    post_peer_version=response.json()["http_version"],
                    stream_version=stream_response.http_version,
                    stream_body=b"".join([chunk async for chunk in stream_response.aiter_bytes()]),
                )
        finally:
            await handler.close()

    return asyncio.run(_run())


def _sync_exchange(base_url: str) -> Observed:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler: Final = HTTPHandler(ssl_verify=False)
    try:
        response: Final = handler.client.post(f"{base_url}/echo", json={"ping": "pong"})
        with handler.client.stream("POST", f"{base_url}/stream", json={}) as stream_response:
            return Observed(
                post_version=response.http_version,
                post_peer_version=response.json()["http_version"],
                stream_version=stream_response.http_version,
                stream_body=b"".join(stream_response.iter_bytes()),
            )
    finally:
        handler.close()


def _set_http2(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    if enabled:
        monkeypatch.setenv("LITELLM_HTTP2", "True")
    else:
        monkeypatch.delenv("LITELLM_HTTP2", raising=False)


def _assert_negotiated(observed: Observed, enabled: bool) -> None:
    client_version, peer_version = ("HTTP/2", "2") if enabled else ("HTTP/1.1", "1.1")
    assert observed.post_version == client_version
    assert observed.post_peer_version == peer_version
    assert observed.stream_version == client_version
    expected_stream: Final = b"".join(f"data: {peer_version}-{index}\n\n".encode() for index in range(STREAM_CHUNKS))
    assert observed.stream_body == expected_stream


@pytest.mark.covers("other.sdk_wire.http2.async_handler_negotiates_h2_only_when_enabled")
def test_async_handler_negotiates_http2_only_when_enabled(monkeypatch: pytest.MonkeyPatch, http2_tls_peer: str) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    for enabled in (False, True):
        _set_http2(monkeypatch, enabled)
        _assert_negotiated(_async_exchange(http2_tls_peer), enabled)


@pytest.mark.covers("other.sdk_wire.http2.sync_handler_negotiates_h2_only_when_enabled")
def test_sync_handler_negotiates_http2_only_when_enabled(monkeypatch: pytest.MonkeyPatch, http2_tls_peer: str) -> None:
    for enabled in (False, True):
        _set_http2(monkeypatch, enabled)
        _assert_negotiated(_sync_exchange(http2_tls_peer), enabled)
