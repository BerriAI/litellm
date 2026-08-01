"""``AsyncHTTPHandler`` must not replay a request whose connection died.

``post``, ``put``, ``patch`` and ``delete`` each used to catch
``(httpx.RemoteProtocolError, httpx.ConnectError)`` and re-send the request. A
dead pooled keep-alive produces those errors, but so does an upstream that read
the request in full, ran the work, and then died before answering. The two are
the same exception, so a replay could re-run billed work just as readily as it
could rescue a connection that never carried the request, and nothing available
to the client separates the two. The replay carried no header delta, no log
line and no counter, so neither the operator nor the upstream could tell it
from a genuine second call.

RFC 9110 section 9.2.2 describes exactly that heuristic, calls it riskier, and
then says a proxy MUST NOT automatically retry non-idempotent requests. aiohttp,
which is litellm's default transport, implements the retry one layer below httpx
and deliberately scopes POST and PATCH out of it
(``IDEMPOTENT_METHODS`` in ``aiohttp/client.py``). Above this layer the Router
already retries at its default ``num_retries``, where the retry is counted,
configurable, and free to pick a different deployment.

So the specification these tests pin is one line: **one call into a handler
method issues exactly one httpx send.** Counting sends rather than server-side
requests is what makes that checkable, because aiohttp retries PUT and DELETE
underneath httpx and the server therefore sees a second request either way.

The tests drive a real loopback server (``_StaleKeepAliveServer``) rather than
monkeypatching ``client.send``, because a mocked transport has no connection
pool of its own to poison, and the pool is the thing under test. The server
binds ``127.0.0.1:0``, so it reaches no network and needs no credentials.
"""

import asyncio
import os
import socket
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

# The two transports map a dead pooled connection onto different httpx errors,
# so each is driven with the drop shape that reaches the code path at issue:
#
#   httpcore  FIN after the request is read -> httpx.RemoteProtocolError
#             RST                           -> httpx.ReadError
#   aiohttp   RST -> aiohttp.ClientOSError   -> httpx.ConnectError
#             FIN -> ServerDisconnectedError -> httpx.ReadError
#
# The deleted retry caught the first error of each pair, which is why those are
# the shapes that would show a replay if it came back.
TRANSPORT_PARAMS = [
    pytest.param("aiohttp", "rst", httpx.ConnectError, id="aiohttp"),
    pytest.param("httpcore", "fin", httpx.RemoteProtocolError, id="httpcore"),
]

# aiohttp treats these as idempotent and retries them itself, below httpx.
AIOHTTP_RETRIES_THESE = ["put", "delete"]
# aiohttp declines to retry these, so nothing but litellm could re-send them.
AIOHTTP_DECLINES_THESE = ["post", "patch"]

# Distinct from the warm-up body, so a replay is countable even when the method
# under test is the POST that warmed the pool.
BODY = b'{"n": 1}'


class _SeenRequest(NamedTuple):
    method: str
    target: str
    body: bytes
    # 1 for the first request on a TCP connection, 2 for the one that gets
    # dropped. A request that arrives as 1 went out on a fresh connection.
    position: int = 0


class _StaleKeepAliveServer:
    """A loopback HTTP server that poisons its own keep-alive connections.

    The first request on any TCP connection is answered normally. A *second*
    request on the same connection is read in full, so a test can see exactly
    what the client sent, and then the connection is dropped without a
    response, either with a FIN (``drop="fin"``) or an RST (``drop="rst"``).

    Reading the second request in full before dropping it is the point: it is
    the shape where a retry replays work the upstream already did.

    Every request the server reads, answered or not, lands in ``requests``.
    """

    def __init__(self, drop: str, stream_frames: int = 0):
        self.drop = drop
        self.stream_frames = stream_frames
        self.requests: List[_SeenRequest] = []
        self._server: Optional[asyncio.AbstractServer] = None
        self.url = ""

    async def __aenter__(self) -> "_StaleKeepAliveServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1/things"
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
                served += 1
                self.requests.append(request._replace(position=served))
                if served == 1:
                    await self._respond(writer)
                else:
                    self._drop(writer)
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
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
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
            await writer.drain()
            for index in range(self.stream_frames):
                frame = b"data: frame-%d\n\n" % index
                writer.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0.01)
            writer.write(b"0\r\n\r\n")
        else:
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()

    def _drop(self, writer) -> None:
        if self.drop == "rst":
            sock = writer.get_extra_info("socket")
            # "ii" is the Linux and macOS `struct linger` (two C ints); Windows
            # wants two shorts. CI is ubuntu-only, so this stays as it is.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        writer.close()


@pytest.fixture
def restore_transport_globals():
    """litellm picks its transport from module globals; put them back afterwards."""
    saved = (litellm.disable_aiohttp_transport, litellm.force_ipv4)
    yield
    litellm.disable_aiohttp_transport, litellm.force_ipv4 = saved


def _make_handler(transport: str) -> Tuple[AsyncHTTPHandler, List[str]]:
    """Build a handler that records the method of every request httpx sends.

    Counting sends is the only way to tell a litellm retry from aiohttp's own.
    aiohttp retries a dropped keep-alive one layer below httpx for the methods
    it treats as idempotent, so the server sees a second request either way and
    a server-side count cannot separate them. Only a litellm retry is a second
    ``send``, and httpx runs its ``request`` event hook once per ``send``.

    The hook goes in at construction rather than onto ``handler.client``
    afterwards, so it covers every client the handler uses. A re-send issued on
    a second, short-lived client, which is how the retry was written before it
    was removed, therefore still shows up in this count.
    """
    litellm.disable_aiohttp_transport = transport == "httpcore"
    litellm.force_ipv4 = False

    sends: List[str] = []

    async def record(request: httpx.Request) -> None:
        sends.append(request.method)

    handler = AsyncHTTPHandler(
        timeout=httpx.Timeout(10.0, connect=5.0),
        event_hooks={"request": [record]},
    )
    return handler, sends


async def _warm_pool(handler: AsyncHTTPHandler, url: str) -> None:
    """Leave exactly one idle keep-alive connection in the pool.

    Wall-clock sensitive, and nothing here extends the connection's life. httpx
    defaults ``Limits.keepalive_expiry`` to 5.0s and ``create_client`` does not
    override it, so on httpcore a pause longer than that between this warm-up
    and the request under test drops the idle connection: the request then goes
    out on a fresh connection, succeeds, and the test no longer exercises a
    poisoned one. ``_assert_pool_was_poisoned`` catches that rather than letting
    the case pass empty. Keep the code between the warm-up and the request under
    test short, and on an unexplained failure suspect a stalled runner first.
    (aiohttp holds its connector open for ``AIOHTTP_KEEPALIVE_TIMEOUT``, 120s by
    default, and tolerates far more.)
    """
    response = await handler.post(url, data=b"{}")
    assert response.status_code == 200


def _assert_pool_was_poisoned(server: _StaleKeepAliveServer) -> None:
    """Fail unless the request under test actually landed on the reused connection.

    A request count cannot show this: a fresh connection produces a second
    server-side request just as a reused one does. Only the request's position
    on its own connection separates them, and it is position 2 that gets
    dropped. Without this check a case could go green having exercised a
    connection that was never at risk of being replayed.
    """
    assert any(r.position == 2 for r in server.requests), (
        "no request reached the poisoned keep-alive connection: positions "
        f"{[r.position for r in server.requests]}. The request under test went out on a "
        "fresh connection, so nothing was exercised."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", AIOHTTP_DECLINES_THESE)
@pytest.mark.parametrize("transport,drop,expected", TRANSPORT_PARAMS)
async def test_a_dead_connection_is_not_replayed(method, transport, drop, expected, restore_transport_globals):
    """The upstream reads the request, dies, and litellm lets the failure through.

    The server has already consumed the body at this point, so a replay would
    be a second execution of work the upstream may have completed. One send in,
    one request at the server, and the transport error reaches the caller.
    """
    async with _StaleKeepAliveServer(drop=drop) as server:
        handler, sends = _make_handler(transport)
        try:
            await _warm_pool(handler, server.url)

            with pytest.raises(expected):
                await getattr(handler, method)(server.url, data=BODY)

            _assert_pool_was_poisoned(server)
            assert len(sends) == 2, f"litellm re-sent the request: {sends}"
            # The body the upstream may already have acted on reached it once.
            assert sum(r.body == BODY for r in server.requests) == 1
            assert server.requests[-1].method == method.upper()
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,drop,expected", TRANSPORT_PARAMS)
async def test_a_dead_connection_on_a_streaming_post_is_not_replayed(
    transport, drop, expected, restore_transport_globals
):
    """Streaming takes the same path, and used to hide the replay best.

    The retry returned a stream over a client it then closed, so the body died
    partway through and the second billed generation was invisible to the
    caller. With no retry the connection failure is what the caller sees.
    """
    async with _StaleKeepAliveServer(drop=drop, stream_frames=6) as server:
        handler, sends = _make_handler(transport)
        try:
            await _warm_pool(handler, server.url)

            with pytest.raises(expected):
                response = await handler.post(server.url, data=BODY, stream=True)
                async for _ in response.aiter_bytes():
                    pass

            _assert_pool_was_poisoned(server)
            assert len(sends) == 2, f"litellm re-sent the request: {sends}"
        finally:
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", AIOHTTP_RETRIES_THESE)
@pytest.mark.parametrize("transport,drop,expected", TRANSPORT_PARAMS)
async def test_litellm_adds_no_retry_where_the_transport_has_its_own(
    method, transport, drop, expected, restore_transport_globals
):
    """PUT and DELETE are aiohttp's to retry, and litellm does not add a second one.

    aiohttp retries these below httpx, so on that transport the call can still
    succeed and the server can still see a second request. Neither is litellm's
    doing, and the send count says so on both transports.

    The two aiohttp cases are a control rather than a guard: aiohttp recovers
    before any exception reaches litellm, so they hold whether or not litellm
    has a retry of its own. The httpcore cases are the ones that fail if it
    comes back.
    """
    async with _StaleKeepAliveServer(drop=drop) as server:
        handler, sends = _make_handler(transport)
        try:
            await _warm_pool(handler, server.url)

            try:
                await getattr(handler, method)(server.url, data=BODY)
            except httpx.TransportError:
                pass

            _assert_pool_was_poisoned(server)
            assert len(sends) == 2, f"litellm re-sent the request: {sends}"
        finally:
            await handler.close()
