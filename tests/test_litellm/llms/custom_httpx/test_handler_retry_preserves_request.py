"""Regression tests for ``AsyncHTTPHandler``'s connection-error retry.

The retry fires when a pooled keep-alive connection turns out to be dead: the
client picks a connection that still looks healthy, writes a request onto it,
and never hears back. These tests drive that from a real loopback server
(``_StaleKeepAliveServer``) rather than by monkeypatching ``client.send``, so
they exercise the transport, the connection pool, and the retry together.

What they pin down:

* a streaming response returned by the retry stays readable to the last byte;
* the retry replays the *original* request -- same method, same body, whether
  the body came from ``content=`` or ``files=``;
* a non-2xx served on the retry path is still routed through litellm's
  credential masking;
* handler-level TLS settings survive the retry.

Both transports are covered. litellm's default is aiohttp; httpcore is the
opt-in (``litellm.disable_aiohttp_transport = True``).

``tests/test_litellm/readme.md`` says this directory holds mocked tests, and a
mock is what this bug cannot be caught with: it needs a live connection pool
holding a keep-alive connection the upstream has already dropped, and a mocked
transport has no pool of its own to poison. The in-tree precedent for a real
loopback server here is
``tests/test_litellm/llms/anthropic/chat/test_anthropic_chat_handler.py``. The
server below binds ``127.0.0.1:0``, so it reaches no network, needs no API key
or credentials, and dies with the test that started it.
"""

import asyncio
import datetime
import io
import ipaddress
import os
import socket
import ssl
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

# litellm retries on ``(httpx.RemoteProtocolError, httpx.ConnectError)``, and the
# two transports map a dead pooled connection onto different httpx errors:
#
#   httpcore  FIN after the request is read -> httpx.RemoteProtocolError  (caught)
#             RST                           -> httpx.ReadError            (not caught)
#   aiohttp   RST -> aiohttp.ClientOSError  -> httpx.ConnectError         (caught)
#             FIN -> ServerDisconnectedError-> httpx.ReadError            (not caught)
#
# So each transport is driven with the drop shape that actually reaches the
# retry. (aiohttp additionally retries dropped keep-alives itself, but only for
# the methods it considers idempotent -- PUT and DELETE, not POST or PATCH.)
TRANSPORT_PARAMS = [
    pytest.param("aiohttp", "rst", id="aiohttp"),
    pytest.param("httpcore", "fin", id="httpcore"),
]


class _SeenRequest(NamedTuple):
    method: str
    target: str
    body: bytes


class _StaleKeepAliveServer:
    """A loopback HTTP server that poisons its own keep-alive connections.

    The first request on any TCP connection is answered normally. A *second*
    request on the same connection is read in full -- so a test can inspect
    exactly what the client re-sent -- and then the connection is dropped
    without a response, either with a FIN (``drop="fin"``) or an RST
    (``drop="rst"``).

    Every request the server reads, answered or not, lands in ``requests``.
    """

    def __init__(
        self,
        drop: str,
        stream_frames: int = 0,
        tls: Optional[Tuple[str, str]] = None,
    ):
        self.drop = drop
        self.stream_frames = stream_frames
        self.tls = tls
        self.status = 200
        self.requests: List[_SeenRequest] = []
        self._server: Optional[asyncio.AbstractServer] = None
        self.url = ""

    async def __aenter__(self) -> "_StaleKeepAliveServer":
        ssl_ctx = None
        if self.tls is not None:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=self.tls[0], keyfile=self.tls[1])
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0, ssl=ssl_ctx)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"{'https' if self.tls else 'http'}://127.0.0.1:{port}/v1/things"
        return self

    async def __aexit__(self, *exc_info) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer) -> None:
        served = 0
        try:
            while True:
                request = await self._read_request(reader)
                if request is None:
                    return
                self.requests.append(request)
                served += 1
                if served == 1:
                    await self._respond(writer)
                else:
                    self._drop(writer)
                    return
        except (ConnectionError, asyncio.IncompleteReadError, ssl.SSLError):
            return
        finally:
            writer.close()

    @staticmethod
    async def _read_request(reader) -> Optional[_SeenRequest]:
        head = b""
        while b"\r\n\r\n" not in head:
            byte = await reader.read(1)
            if not byte:
                return None
            head += byte
        lines = head.split(b"\r\n")
        method, target, _ = lines[0].split(b" ", 2)
        headers = {}
        for line in lines[1:]:
            if b":" in line:
                name, _, value = line.partition(b":")
                headers[name.strip().lower()] = value.strip()

        if headers.get(b"transfer-encoding", b"").lower() == b"chunked":
            body = b""
            while True:
                size = int((await reader.readuntil(b"\r\n")).strip(), 16)
                if size == 0:
                    await reader.readuntil(b"\r\n")
                    break
                body += await reader.readexactly(size)
                await reader.readexactly(2)
        else:
            length = int(headers.get(b"content-length", b"0"))
            body = await reader.readexactly(length) if length else b""

        return _SeenRequest(method.decode(), target.decode(), body)

    async def _respond(self, writer) -> None:
        if self.stream_frames:
            writer.write(
                b"HTTP/1.1 %d OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n" % self.status
            )
            await writer.drain()
            for index in range(self.stream_frames):
                frame = b"data: frame-%d\n\n" % index
                writer.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0.01)
            writer.write(b"0\r\n\r\n")
        else:
            body = b"ok" if self.status == 200 else b"unauthorized"
            writer.write(
                b"HTTP/1.1 %d STATUS\r\nContent-Type: text/plain\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (self.status, len(body), body)
            )
        await writer.drain()

    def _drop(self, writer) -> None:
        if self.drop == "rst":
            sock = writer.get_extra_info("socket")
            # "ii" is the Linux and macOS `struct linger` (two C ints); Windows
            # wants two shorts. CI is ubuntu-only, so this stays as it is.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        writer.close()


class _NonSeekableReader:
    """A read-once file-like object, as a pipe or a socket wrapper would be."""

    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def seek(self, *args, **kwargs) -> int:
        raise io.UnsupportedOperation("seek")


@pytest.fixture
def restore_transport_globals():
    """litellm picks its transport from module globals; put them back afterwards."""
    saved = (litellm.disable_aiohttp_transport, litellm.force_ipv4)
    yield
    litellm.disable_aiohttp_transport, litellm.force_ipv4 = saved


def _make_handler(transport: str, **kwargs) -> AsyncHTTPHandler:
    litellm.disable_aiohttp_transport = transport == "httpcore"
    litellm.force_ipv4 = False
    return AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0), **kwargs)


def _track_sends(handler: AsyncHTTPHandler) -> List[str]:
    """Record the method of every request handed to ``httpx.AsyncClient.send``.

    That is the layer litellm retries at, and counting it is the only way to
    tell litellm's retry from aiohttp's. aiohttp retries a dropped keep-alive
    itself, one layer below httpx, for the methods it treats as idempotent: the
    server sees a third request either way, so a server-side count cannot
    separate them, but only litellm's retry is a second ``send``. httpx runs its
    ``request`` event hook once per ``send``, so this list counts sends.
    """
    sends: List[str] = []

    async def record(request: httpx.Request) -> None:
        sends.append(request.method)

    handler.client.event_hooks["request"].append(record)
    return sends


def _assert_retried(server: _StaleKeepAliveServer, sends: List[str]) -> None:
    """Fail unless litellm's retry ran, so a green test cannot be an empty one.

    Most of these cases still pass when the retry never fires -- the request
    simply succeeds on its first attempt and there is nothing left to get wrong.
    An httpx, httpcore or aiohttp upgrade that maps a dead pooled connection
    onto an error the retry clause does not catch would quietly turn them into
    that, so each one says out loud that the retry happened.
    """
    assert len(sends) == 3, (
        f"litellm's retry did not fire: httpx sent {len(sends)} request(s), "
        f"expected 3 (warm-up, poisoned attempt, retry)"
    )
    assert len(server.requests) == 3, (
        f"the server saw {len(server.requests)} request(s), expected 3 (warm-up, poisoned attempt, retry)"
    )


async def _warm_pool(handler: AsyncHTTPHandler, url: str) -> None:
    """Leave exactly one idle keep-alive connection in the pool.

    Wall-clock sensitive, and nothing here extends the connection's life. httpx
    defaults ``Limits.keepalive_expiry`` to 5.0s and ``create_client`` does not
    override it, so on httpcore a pause longer than that between this warm-up
    and the request under test drops the idle connection: no poisoned
    connection is picked, litellm's retry never fires, and ``_assert_retried``
    fails. (aiohttp holds its own connector open for
    ``AIOHTTP_KEEPALIVE_TIMEOUT``, 120s by default, and tolerates far more.) So
    keep the code between the warm-up and the request under test short, and on
    an unexplained failure suspect a stalled runner before the retry itself.
    """
    response = await handler.post(url, data=b"{}")
    assert response.status_code == 200


def _self_signed_cert(directory) -> Tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
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
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = str(directory / "cert.pem")
    key_path = str(directory / "key.pem")
    with open(cert_path, "wb") as handle:
        handle.write(certificate.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as handle:
        handle.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    return cert_path, key_path


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_keeps_the_streaming_body_readable(transport, drop, restore_transport_globals):
    """The response the retry hands back must still stream to the last byte.

    Closing a single-use retry client in a ``finally`` returns the streaming
    response with its transport already torn down, so the body dies almost at
    once: on either transport at most one of the six frames arrives before
    ``aiter_bytes()`` raises ``httpx.ReadError``, and sometimes none does.
    """
    async with _StaleKeepAliveServer(drop=drop, stream_frames=6) as server:
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            response = await handler.post(server.url, data=b"{}", stream=True)
            frames = 0
            async for chunk in response.aiter_bytes():
                frames += chunk.count(b"data: frame-")

            assert frames == 6
            _assert_retried(server, sends)
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_preserves_the_http_method(method, transport, drop, restore_transport_globals):
    """A retried PUT must reach the upstream as a PUT, not as a POST."""
    if transport == "aiohttp" and method in ("put", "delete"):
        pytest.skip(
            "aiohttp retries a dropped keep-alive itself for the methods it treats as "
            "idempotent, one layer below httpx, so litellm's retry never runs here and "
            "the case would pass with or without this change"
        )

    async with _StaleKeepAliveServer(drop=drop) as server:
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            response = await getattr(handler, method)(server.url, data=b'{"n": 1}')

            assert response.status_code == 200
            _assert_retried(server, sends)
            dropped, retried = server.requests[-2:]
            assert (dropped.method, retried.method) == (method.upper(), method.upper())
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_preserves_a_content_body(transport, drop, restore_transport_globals):
    """``content=`` must survive the retry rather than being dropped."""
    payload = b'{"prompt": "' + b"x" * 256 + b'"}'
    async with _StaleKeepAliveServer(drop=drop) as server:
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            response = await handler.post(server.url, content=payload)

            assert response.status_code == 200
            _assert_retried(server, sends)
            dropped, retried = server.requests[-2:]
            assert dropped.body == payload
            assert retried.body == payload
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_of_a_one_shot_content_iterator_fails_loudly(transport, drop, restore_transport_globals):
    """An unreplayable ``content=`` iterator must raise, not send nothing.

    The retry replays the original ``httpx.Request``, and a request whose body
    is a one-shot async iterator has already spent it, so httpx refuses with
    ``StreamConsumed``. Failing is the honest outcome: the old retry built a
    fresh request that was never passed ``content=`` at all, so it sent a
    zero-byte body and returned the upstream's 200 as if the payload had landed.

    The two transports surface it differently. httpcore lets httpx's own
    ``StreamConsumed`` out; aiohttp catches it while writing the body and
    re-raises it as a connection error, which httpx maps to ``NetworkError``
    carrying the same text.
    """

    async def one_shot():
        yield b'{"prompt": "sent exactly once"}'

    expected = httpx.StreamConsumed if transport == "httpcore" else httpx.NetworkError
    async with _StaleKeepAliveServer(drop=drop) as server:
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            with pytest.raises(expected) as raised:
                await handler.post(server.url, content=one_shot())

            assert "the content has already been streamed" in str(raised.value)
            assert len(sends) == 3, f"litellm's retry did not fire: httpx sent {len(sends)} request(s)"
            # The retry raised before anything reached the wire, so the upstream
            # saw only the warm-up and the attempt that hit the poisoned
            # connection: no empty-bodied third request quietly answered 200.
            assert len(server.requests) == 2
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_preserves_a_multipart_upload(transport, drop, restore_transport_globals):
    """A retried ``files=`` upload must carry the file, not an empty body.

    Silently uploading zero bytes and returning 200 is worse than failing: the
    caller has no way to tell the upload did not happen.
    """
    files = {"file": ("audio.wav", b"RIFF" + b"\x00" * 2048, "audio/wav")}
    async with _StaleKeepAliveServer(drop=drop) as server:
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            response = await handler.post(server.url, files=files)

            assert response.status_code == 200
            _assert_retried(server, sends)
            dropped, retried = server.requests[-2:]
            assert len(dropped.body) > 2048
            assert len(retried.body) == len(dropped.body)
            assert retried.body == dropped.body
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_of_a_non_seekable_upload_replays_an_empty_part(transport, drop, restore_transport_globals):
    """A ``files=`` part read from a non-seekable handle replays empty.

    A limitation of httpx, not of this change, and the reason the test above
    uses bytes. ``FileField.render_data`` rewinds with ``seek(0)`` and swallows
    ``io.UnsupportedOperation``, so on the replay it reads an exhausted handle,
    yields nothing, and the retry sends the multipart envelope around a
    zero-length part, which the upstream answers 200. Bytes, str and seekable
    handles all replay faithfully. Pinned so a change in httpx is noticed.
    """
    payload = b"RIFF" + b"\x00" * 2048
    files = {"file": ("audio.wav", _NonSeekableReader(payload), "audio/wav")}
    async with _StaleKeepAliveServer(drop=drop) as server:
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            response = await handler.post(server.url, files=files)

            assert response.status_code == 200
            _assert_retried(server, sends)
            dropped, retried = server.requests[-2:]
            assert payload in dropped.body
            assert payload not in retried.body
            assert len(retried.body) == len(dropped.body) - len(payload)
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", TRANSPORT_PARAMS)
async def test_retry_path_error_still_masks_the_api_key(transport, drop, restore_transport_globals):
    """A non-2xx served on the retry must go through litellm's masking.

    ``raise_for_status()`` has to stay in the caller's outer ``try`` for that:
    an ``HTTPStatusError`` raised from inside the retry's own ``except`` clause
    cannot reach the sibling ``except httpx.HTTPStatusError`` handler, so the
    full URL -- API key and all -- ends up in the exception text.
    """
    secret = "sk-not-a-real-key-0123456789"
    async with _StaleKeepAliveServer(drop=drop) as server:
        url = f"{server.url}?key={secret}"
        handler = _make_handler(transport)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, url)
            server.status = 401

            with pytest.raises(httpx.HTTPStatusError) as raised:
                await handler.post(url, data=b"{}")

            message = str(raised.value)
            assert "key=[REDACTED_API_KEY]" in message
            assert secret not in message
            _assert_retried(server, sends)
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop", [pytest.param("httpcore", "fin", id="httpcore")])
async def test_retry_preserves_handler_ssl_verify(transport, drop, tmp_path, restore_transport_globals):
    """The retry must keep the handler's own TLS settings.

    Building a second client for the retry without forwarding ``ssl_verify``
    silently swaps the handler's configuration for litellm's global default, so
    a handler that deliberately trusts a private (here, self-signed) certificate
    fails the moment it retries.

    httpcore only: over TLS, aiohttp reports a dropped pooled connection as
    ``ServerDisconnectedError`` whichever way the peer goes away (the SSL layer
    absorbs the reset), and that maps to ``httpx.ReadError``, which the retry
    clause does not catch. The retry is unreachable there, so there is nothing
    to assert.
    """
    cert, key = _self_signed_cert(tmp_path)
    async with _StaleKeepAliveServer(drop=drop, tls=(cert, key)) as server:
        handler = _make_handler(transport, ssl_verify=False)
        sends = _track_sends(handler)
        try:
            await _warm_pool(handler, server.url)

            response = await handler.post(server.url, data=b"{}")

            assert response.status_code == 200
            _assert_retried(server, sends)
        finally:
            await handler.close()
