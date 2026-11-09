from __future__ import annotations

import asyncio
import contextlib
import datetime
import functools
import ipaddress
import json
import queue
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import StreamEnded
from httpx._config import DEFAULT_LIMITS
from hypercorn.asyncio import (
    serve,  # pyright: ignore[reportUnknownVariableType]  # hypercorn's serve signature passes through untyped worker hooks
)
from hypercorn.config import Config
from hypercorn.typing import ASGIReceiveCallable, ASGISendCallable, HTTPResponseBodyEvent, HTTPResponseStartEvent, Scope
from integration._support.ports import allocate_port
from integration._support.vertex import service_account_json

STREAM_CHUNKS: Final = 3
HELD_SEARCHES: Final = 150
HOLD_SECONDS: Final = 10.0
PROBED_HEADERS: Final = frozenset({b"te", b"keep-alive", b"connection", b"proxy-connection", b"upgrade", b"x-probe"})
VERTEX_SERVING_CONFIG_PATH: Final = (
    "/v1/projects/p/locations/global/collections/default_collection/engines/e/servingConfigs/default_search"
)


@dataclass(frozen=True, slots=True)
class PeerLog:
    held_arrivals: queue.SimpleQueue[int] = field(default_factory=queue.SimpleQueue)
    all_held_arrived: asyncio.Event = field(default_factory=asyncio.Event)
    token_requests: queue.SimpleQueue[str] = field(default_factory=queue.SimpleQueue)
    counted_client_ports: queue.SimpleQueue[int] = field(default_factory=queue.SimpleQueue)


@dataclass(frozen=True, slots=True)
class TlsPeer:
    url: str
    cert_file: Path
    key_file: Path
    log: PeerLog


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


async def _peer(log: PeerLog, scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    if scope["type"] != "http":
        return
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        if message["type"] == "http.request" and not message["more_body"]:
            break
    version: Final = scope["http_version"]
    if scope["path"] == "/hold:search":
        client: Final = scope["client"]
        await _send_json(send, {"results": [{"id": await _hold_until_all_arrive(log, client[1] if client else 0)}]})
        return
    if scope["path"].endswith("/_oauth/token"):
        log.token_requests.put(scope["path"])
        await _send_json(send, {"access_token": "scripted", "expires_in": 3600, "token_type": "Bearer"})
        return
    if scope["path"] == "/counted:search":
        counted_client: Final = scope["client"]
        log.counted_client_ports.put(counted_client[1] if counted_client else 0)
        await _send_json(send, {"results": [{"id": "counted"}]})
        return
    if scope["path"].startswith("/headers/") and scope["path"].endswith(":search"):
        seen: Final = sorted(name.decode() for name, _ in scope["headers"] if name in PROBED_HEADERS)
        await _send_json(send, {"results": [{"id": f"http/{version}"}, *({"id": name} for name in seen)]})
        return
    if scope["path"].endswith(":search"):
        await _send_json(send, {"results": [{"id": f"http/{version}"}]})
        return
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
    await _send_json(send, {"http_version": version})


async def _hold_until_all_arrive(log: PeerLog, client_port: int) -> str:
    log.held_arrivals.put(client_port)
    if log.held_arrivals.qsize() >= HELD_SEARCHES:
        log.all_held_arrived.set()
    waiter: Final = asyncio.ensure_future(log.all_held_arrived.wait())
    await asyncio.wait((waiter,), timeout=HOLD_SECONDS)
    waiter.cancel()
    return "held-together" if log.all_held_arrived.is_set() else "released-alone"


async def _send_json(send: ASGISendCallable, body: dict[str, object]) -> None:
    await send(
        HTTPResponseStartEvent(type="http.response.start", status=200, headers=[(b"content-type", b"application/json")])
    )
    await send(HTTPResponseBodyEvent(type="http.response.body", body=json.dumps(body).encode(), more_body=False))


@pytest.fixture
def http2_tls_peer(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TlsPeer]:
    cert_file, key_file = _write_self_signed_cert(tmp_path_factory.mktemp("h2certs"))
    port: Final = allocate_port()
    shutdown: Final = threading.Event()
    log: Final = PeerLog()

    def _serve() -> None:
        loop: Final = asyncio.new_event_loop()
        config: Final = Config()
        config.bind = [f"127.0.0.1:{port}"]
        config.certfile = str(cert_file)
        config.keyfile = str(key_file)
        config.alpn_protocols = ["h2", "http/1.1"]
        loop.run_until_complete(
            serve(functools.partial(_peer, log), config, shutdown_trigger=lambda: asyncio.to_thread(shutdown.wait))
        )
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
    yield TlsPeer(url=f"https://127.0.0.1:{port}", cert_file=cert_file, key_file=key_file, log=log)
    shutdown.set()
    thread.join(timeout=10)


def _async_exchange(base_url: str, ssl_verify: bool | None = False) -> Observed:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    async def _run() -> Observed:
        handler: Final = AsyncHTTPHandler(ssl_verify=ssl_verify)
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


def _sync_exchange(base_url: str, ssl_verify: bool | None = False) -> Observed:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler: Final = HTTPHandler(ssl_verify=ssl_verify)
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
def test_async_handler_negotiates_http2_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    for enabled in (False, True):
        _set_http2(monkeypatch, enabled)
        _assert_negotiated(_async_exchange(http2_tls_peer.url), enabled)


@pytest.mark.covers("other.sdk_wire.http2.sync_handler_negotiates_h2_only_when_enabled")
def test_sync_handler_negotiates_http2_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    for enabled in (False, True):
        _set_http2(monkeypatch, enabled)
        _assert_negotiated(_sync_exchange(http2_tls_peer.url), enabled)


def _trust_peer_without_global_http2(monkeypatch: pytest.MonkeyPatch, peer: TlsPeer) -> None:
    import litellm

    _set_http2(monkeypatch, False)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)
    monkeypatch.setenv("SSL_VERIFY", str(peer.cert_file))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(peer.cert_file))
    litellm.in_memory_llm_clients_cache.flush_cache()


def _vertex_search_file_ids(
    peer: TlsPeer, search_url: str | None = None, ssl_verify: ssl.SSLContext | None = None
) -> list[str | None]:
    import litellm

    response: Final = litellm.vector_stores.search(
        vector_store_id="default_search",
        query="ping",
        custom_llm_provider="vertex_ai/search_api",
        api_base=f"{search_url or peer.url}{VERTEX_SERVING_CONFIG_PATH}",
        vertex_credentials=service_account_json("p", peer.url),
        vertex_project="p",
        ssl_verify=ssl_verify,
    )
    assert isinstance(response, dict)
    return [result.get("file_id") for result in response.get("data") or ()]


async def _avertex_search_file_ids(
    peer: TlsPeer,
    search_url: str | None = None,
    ssl_verify: ssl.SSLContext | None = None,
    extra_headers: dict[str, object] | None = None,
) -> list[str | None]:
    import litellm

    response: Final = await litellm.vector_stores.asearch(
        vector_store_id="default_search",
        query="ping",
        custom_llm_provider="vertex_ai/search_api",
        api_base=f"{search_url or peer.url}{VERTEX_SERVING_CONFIG_PATH}",
        vertex_credentials=service_account_json("p", peer.url),
        vertex_project="p",
        ssl_verify=ssl_verify,
        extra_headers=extra_headers,
    )
    return [result.get("file_id") for result in response.get("data") or ()]


def test_sync_vertex_search_stays_on_http1_without_global_flag(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    assert _vertex_search_file_ids(http2_tls_peer) == ["http/1.1"]


def test_async_vertex_search_negotiates_http2_without_global_flag(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    assert asyncio.run(_avertex_search_file_ids(http2_tls_peer)) == ["http/2"]


def test_default_async_client_stays_http1_after_vertex_search_used_http2(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)

    async def _run() -> tuple[list[str | None], str]:
        file_ids: Final = await _avertex_search_file_ids(http2_tls_peer)
        handler: Final = AsyncHTTPHandler(ssl_verify=str(http2_tls_peer.cert_file))
        try:
            response: Final = await handler.post(f"{http2_tls_peer.url}/echo", json={"ping": "pong"})
            return file_ids, response.json()["http_version"]
        finally:
            await handler.close()

    assert asyncio.run(_run()) == (["http/2"], "1.1")


def test_vertex_search_fetches_one_token_for_many_searches(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    import litellm

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    credentials: Final = service_account_json("p", f"{http2_tls_peer.url}/token-reuse")
    search: Final = MappingProxyType(
        {
            "vector_store_id": "default_search",
            "query": "ping",
            "custom_llm_provider": "vertex_ai/search_api",
            "api_base": f"{http2_tls_peer.url}{VERTEX_SERVING_CONFIG_PATH}",
            "vertex_credentials": credentials,
            "vertex_project": "p",
        }
    )

    async def _searches() -> None:
        await litellm.vector_stores.asearch(**search)
        await asyncio.gather(*(litellm.vector_stores.asearch(**search) for _ in range(5)))

    asyncio.run(_searches())
    litellm.vector_stores.search(**search)

    token_requests: Final = http2_tls_peer.log.token_requests
    token_paths: Final = tuple(iter(lambda: None if token_requests.empty() else token_requests.get(), None))
    assert token_paths.count("/token-reuse/_oauth/token") == 1


def test_vertex_search_over_forced_ipv4_trusts_the_configured_ca(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    import litellm

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    monkeypatch.setattr(litellm, "force_ipv4", True)

    assert _vertex_search_file_ids(http2_tls_peer) == ["http/1.1"]
    assert asyncio.run(_avertex_search_file_ids(http2_tls_peer)) == ["http/2"]


@pytest.mark.parametrize("off", ["False", "0", "no", " OFF "])
def test_explicit_http2_off_keeps_vertex_search_on_http1(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer, off: str
) -> None:
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    monkeypatch.setenv("LITELLM_HTTP2", off)
    assert asyncio.run(_avertex_search_file_ids(http2_tls_peer)) == ["http/1.1"]


@pytest.mark.parametrize("off", [False, "false", 0])
def test_http2_setting_off_keeps_vertex_search_on_http1(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer, off: bool | int | str
) -> None:
    import litellm

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    monkeypatch.setattr(litellm, "http2", off)
    assert asyncio.run(_avertex_search_file_ids(http2_tls_peer)) == ["http/1.1"]


@pytest.mark.parametrize(
    ("te", "forwarded"), [("gzip", ["http/2", "x-probe"]), ("trailers", ["http/2", "te", "x-probe"])]
)
def test_async_vertex_search_over_http2_sends_no_connection_specific_headers(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer, te: str, forwarded: list[str]
) -> None:
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    received: Final = asyncio.run(
        _avertex_search_file_ids(
            http2_tls_peer,
            search_url=f"{http2_tls_peer.url}/headers",
            extra_headers={
                "TE": te,
                "Keep-Alive": "timeout=5",
                "Connection": "keep-alive",
                "Proxy-Connection": "keep-alive",
                "Upgrade": "h2c",
                "X-Probe": "kept",
            },
        )
    )
    assert received == forwarded


def _alpn_offered_by(context: ssl.SSLContext, peer: TlsPeer) -> str | None:
    host, port = peer.url.removeprefix("https://").split(":")
    with (
        socket.create_connection((host, int(port)), timeout=5) as raw,
        context.wrap_socket(raw, server_hostname=host) as tls,
    ):
        return tls.selected_alpn_protocol()


def test_vertex_search_leaves_a_caller_ssl_context_on_http1(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    sync_context: Final = ssl.create_default_context(cafile=str(http2_tls_peer.cert_file))
    async_context: Final = ssl.create_default_context(cafile=str(http2_tls_peer.cert_file))

    assert _vertex_search_file_ids(http2_tls_peer, ssl_verify=sync_context) == ["http/1.1"]
    assert asyncio.run(_avertex_search_file_ids(http2_tls_peer, ssl_verify=async_context)) == ["http/1.1"]
    assert _alpn_offered_by(sync_context, http2_tls_peer) != "h2"
    assert _alpn_offered_by(async_context, http2_tls_peer) != "h2"


def _route_https_through_a_dead_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{allocate_port()}")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRUST_ENV", "True")


async def _async_peer_http_version(peer: TlsPeer, http2: bool) -> str:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    handler: Final = AsyncHTTPHandler(ssl_verify=str(peer.cert_file), http2=http2)
    try:
        response: Final = await handler.client.post(f"{peer.url}/echo", json={"ping": "pong"})
        return response.http_version
    finally:
        await handler.close()


def test_http2_client_skips_env_proxies_when_aiohttp_trust_env_is_disabled(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    import litellm

    _set_http2(monkeypatch, False)
    _route_https_through_a_dead_proxy(monkeypatch)

    for force_ipv4 in (False, True):
        monkeypatch.setattr(litellm, "force_ipv4", force_ipv4)
        assert asyncio.run(_async_peer_http_version(http2_tls_peer, http2=True)) == "HTTP/2"


def test_global_http2_toggle_keeps_env_proxies_when_aiohttp_trust_env_is_disabled(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    import litellm

    _set_http2(monkeypatch, True)
    _route_https_through_a_dead_proxy(monkeypatch)

    for force_ipv4 in (False, True):
        monkeypatch.setattr(litellm, "force_ipv4", force_ipv4)
        with pytest.raises(httpx.ConnectError):
            asyncio.run(_async_peer_http_version(http2_tls_peer, http2=False))


def _answer_or_drain(connection: H2Connection, stream_id: int, drain: bool) -> None:
    if drain:
        connection.close_connection(last_stream_id=stream_id)
        return
    connection.send_headers(stream_id, [(":status", "200"), ("content-type", "application/json")])
    connection.send_data(stream_id, json.dumps({"results": [{"id": "answered"}]}).encode(), end_stream=True)


def _reply_to_ended_streams(connection: H2Connection, tls: ssl.SSLSocket, data: bytes, drain: bool) -> bool:
    ended: Final = tuple(event.stream_id for event in connection.receive_data(data) if isinstance(event, StreamEnded))
    for stream_id in ended:
        _answer_or_drain(connection, stream_id, drain)
    tls.sendall(connection.data_to_send())
    return bool(ended)


def _serve_h2(context: ssl.SSLContext, raw: socket.socket, drain: bool) -> None:
    with contextlib.suppress(OSError), raw, context.wrap_socket(raw, server_side=True) as tls:
        connection: Final = H2Connection(config=H2Configuration(client_side=False))
        connection.initiate_connection()
        tls.sendall(connection.data_to_send())
        while data := tls.recv(65535):
            if _reply_to_ended_streams(connection, tls, data, drain) and drain:
                return


def _read_http1_request(tls: ssl.SSLSocket) -> bytes:
    with tls.makefile("rb") as stream:
        head: Final = tuple(iter(stream.readline, b"\r\n"))
        length: Final = next(
            (int(line.split(b":", 1)[1]) for line in head if line.lower().startswith(b"content-length:")), 0
        )
        return b"".join(head) + stream.read(length)


def _hang_up_after_request(context: ssl.SSLContext, raw: socket.socket, connection_number: int) -> None:
    with contextlib.suppress(OSError), raw, context.wrap_socket(raw, server_side=True) as tls:
        _ = _read_http1_request(tls)


def _drain_every_other_connection(context: ssl.SSLContext, raw: socket.socket, connection_number: int) -> None:
    _serve_h2(context, raw, drain=connection_number % 2 == 1)


def _answer_every_connection(context: ssl.SSLContext, raw: socket.socket, connection_number: int) -> None:
    _serve_h2(context, raw, drain=False)


@contextmanager
def _raw_tls_peer(
    peer: TlsPeer,
    alpn: str,
    serve_connection: Callable[[ssl.SSLContext, socket.socket, int], None],
    client_ca: Path | None = None,
) -> Generator[tuple[str, queue.SimpleQueue[int]]]:
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(peer.cert_file, peer.key_file)
    context.set_alpn_protocols([alpn])
    if client_ca is not None:
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(client_ca)
    stop: Final = threading.Event()
    accepted: Final = queue.SimpleQueue[int]()

    with socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(0.1)

        def _accept_one() -> None:
            raw: Final = listener.accept()[0]
            accepted.put(accepted.qsize() + 1)
            threading.Thread(target=serve_connection, args=(context, raw, accepted.qsize()), daemon=True).start()

        def _accept() -> None:
            while not stop.is_set():
                with contextlib.suppress(TimeoutError):
                    _accept_one()

        acceptor: Final = threading.Thread(target=_accept, daemon=True)
        acceptor.start()
        try:
            yield f"https://127.0.0.1:{listener.getsockname()[1]}", accepted
        finally:
            stop.set()
            acceptor.join(timeout=5)


def test_vertex_search_retries_a_connection_the_server_drains_with_goaway(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)

    with _raw_tls_peer(http2_tls_peer, "h2", _drain_every_other_connection) as (search_url, _):
        assert asyncio.run(_avertex_search_file_ids(http2_tls_peer, search_url=search_url)) == ["answered"]
        _set_http2(monkeypatch, True)
        assert _vertex_search_file_ids(http2_tls_peer, search_url=search_url) == ["answered"]


def test_vertex_search_sends_once_when_an_http1_server_hangs_up(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    import litellm

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    monkeypatch.setenv("LITELLM_HTTP2", "False")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    with _raw_tls_peer(http2_tls_peer, "http/1.1", _hang_up_after_request) as (search_url, accepted):
        with pytest.raises(litellm.InternalServerError, match="Server disconnected"):
            _vertex_search_file_ids(http2_tls_peer, search_url=search_url)
        assert accepted.qsize() == 1
        with pytest.raises(litellm.InternalServerError, match="Server disconnected"):
            asyncio.run(_avertex_search_file_ids(http2_tls_peer, search_url=search_url))
        assert accepted.qsize() == 2


@pytest.mark.parametrize("force_ipv4", [False, True])
def test_vertex_search_keeps_its_http2_connection_across_an_idle_gap(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer, force_ipv4: bool
) -> None:
    import litellm
    from litellm.constants import AIOHTTP_KEEPALIVE_TIMEOUT

    idle_gap: Final = DEFAULT_LIMITS.keepalive_expiry + 1
    assert idle_gap < AIOHTTP_KEEPALIVE_TIMEOUT
    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    monkeypatch.setattr(litellm, "force_ipv4", force_ipv4)

    async def _search_twice(search_url: str) -> list[str | None]:
        first: Final = await _avertex_search_file_ids(http2_tls_peer, search_url=search_url)
        await asyncio.sleep(idle_gap)
        return [*first, *await _avertex_search_file_ids(http2_tls_peer, search_url=search_url)]

    with _raw_tls_peer(http2_tls_peer, "h2", _answer_every_connection) as (search_url, accepted):
        assert asyncio.run(_search_twice(search_url)) == ["answered", "answered"]
        assert accepted.qsize() == 1


def test_async_http2_handler_opens_another_connection_once_the_streams_run_out(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer
) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)

    async def _burst() -> list[str]:
        handler: Final = AsyncHTTPHandler(ssl_verify=str(http2_tls_peer.cert_file), http2=True)
        try:
            responses: Final = await asyncio.gather(
                *(handler.post(f"{http2_tls_peer.url}/hold:search", json={}) for _ in range(HELD_SEARCHES))
            )
            return [response.json()["results"][0]["id"] for response in responses]
        finally:
            await handler.close()

    search_ids: Final = asyncio.run(_burst())
    client_ports: Final = frozenset(
        http2_tls_peer.log.held_arrivals.get_nowait() for _ in range(http2_tls_peer.log.held_arrivals.qsize())
    )

    assert search_ids == ["held-together"] * HELD_SEARCHES
    assert len(client_ports) == 2


def test_http2_transport_opens_no_more_connections_than_its_cap(http2_tls_peer: TlsPeer) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    async def _burst() -> frozenset[str]:
        transport: Final = AsyncHTTPHandler.create_async_transport(
            ssl_context=ssl.create_default_context(cafile=str(http2_tls_peer.cert_file)),
            http2=True,
            limits=httpx.Limits(max_connections=1),
        )
        async with httpx.AsyncClient(transport=transport) as client:
            responses: Final = await asyncio.gather(
                *(client.post(f"{http2_tls_peer.url}/counted:search", json={}) for _ in range(HELD_SEARCHES))
            )
        return frozenset(response.json()["results"][0]["id"] for response in responses)

    assert asyncio.run(_burst()) == {"counted"}
    counted_client_ports: Final = http2_tls_peer.log.counted_client_ports
    client_ports: Final = frozenset(
        iter(lambda: None if counted_client_ports.empty() else counted_client_ports.get(), None)
    )
    assert len(client_ports) == 1


def test_forced_ipv4_handlers_honour_ssl_verify_false(monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer) -> None:
    import litellm

    monkeypatch.setattr(litellm, "force_ipv4", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    for enabled in (False, True):
        _set_http2(monkeypatch, enabled)
        _assert_negotiated(_async_exchange(http2_tls_peer.url), enabled)
        _assert_negotiated(_sync_exchange(http2_tls_peer.url), enabled)


def _ca_dir_holding(cert_file: Path, ca_dir: Path) -> Path:
    subject_hash: Final = subprocess.run(
        ["openssl", "x509", "-hash", "-noout", "-in", str(cert_file)], capture_output=True, text=True, check=True
    ).stdout.strip()
    (ca_dir / f"{subject_hash}.0").write_bytes(cert_file.read_bytes())
    return ca_dir


def test_forced_ipv4_handlers_trust_a_ca_found_only_in_ssl_cert_dir(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer, tmp_path: Path
) -> None:
    import litellm

    monkeypatch.setattr(litellm, "force_ipv4", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.delenv("SSL_VERIFY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("SSL_CERT_DIR", str(_ca_dir_holding(http2_tls_peer.cert_file, tmp_path)))
    for enabled in (False, True):
        _set_http2(monkeypatch, enabled)
        _assert_negotiated(_async_exchange(http2_tls_peer.url, ssl_verify=None), enabled)
        _assert_negotiated(_sync_exchange(http2_tls_peer.url, ssl_verify=None), enabled)


def test_vertex_search_over_forced_ipv4_presents_the_client_certificate(
    monkeypatch: pytest.MonkeyPatch, http2_tls_peer: TlsPeer, tmp_path: Path
) -> None:
    import litellm

    _trust_peer_without_global_http2(monkeypatch, http2_tls_peer)
    monkeypatch.setattr(litellm, "force_ipv4", True)
    client_pem: Final = tmp_path / "client.pem"
    client_pem.write_bytes(http2_tls_peer.cert_file.read_bytes() + http2_tls_peer.key_file.read_bytes())
    monkeypatch.setenv("SSL_CERTIFICATE", str(client_pem))

    with _raw_tls_peer(http2_tls_peer, "h2", _answer_every_connection, client_ca=http2_tls_peer.cert_file) as (
        search_url,
        _,
    ):
        assert asyncio.run(_avertex_search_file_ids(http2_tls_peer, search_url=search_url)) == ["answered"]
        _set_http2(monkeypatch, True)
        assert _vertex_search_file_ids(http2_tls_peer, search_url=search_url) == ["answered"]


@pytest.mark.parametrize("proxied", [False, True], ids=["direct", "behind-http-proxy"])
def test_an_evicted_http2_handler_is_not_closed_mid_request(monkeypatch: pytest.MonkeyPatch, proxied: bool) -> None:
    from litellm.caching.evicted_client_closer import EvictedClientCloser
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)

    async def _evict_mid_request() -> None:
        arrived: Final = asyncio.Event()
        release: Final = asyncio.Event()

        async def _upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.read(4096)
            writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n")
            await writer.drain()
            arrived.set()
            await release.wait()
            writer.write(b"0\r\n\r\n")
            await writer.drain()

        server: Final = await asyncio.start_server(_upstream, "127.0.0.1", 0)
        port: Final[int] = server.sockets[0].getsockname()[1]
        if proxied:
            monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{port}")
        closer: Final = EvictedClientCloser(grace_seconds=0.0)
        handler: Final = AsyncHTTPHandler(http2=True)
        evicted: Final = handler.client
        closer.mark_owned(handler)
        closer.schedule(handler)
        request: Final = asyncio.create_task(
            handler.get("http://upstream.invalid/" if proxied else f"http://127.0.0.1:{port}/")
        )

        await arrived.wait()
        closer.reap()
        await asyncio.sleep(0)
        assert evicted.is_closed is False, "closed a handler that was serving a request"

        release.set()
        assert (await request).text == "hello"
        closer.reap()
        await asyncio.sleep(0)
        assert evicted.is_closed is True
        assert handler.client.is_closed is False, "a held handler must self-heal after its evicted client is closed"
        server.close()

    asyncio.run(_evict_mid_request())
