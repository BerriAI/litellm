"""
Collecting an HTTP handler must not abort a response that is still on the wire.

``HTTPHandler`` and ``AsyncHTTPHandler`` close their client from ``__del__``.
Closing a client tears down the connection pool, which aborts every response
still streaming through it. ``_handler_may_close_client`` already withholds the
close from a client someone else holds, but a streaming response holds the
connection it is reading from and never the client, so the refcount it reads
says "sole referrer" for exactly the client that is busiest. The handler is
routinely collectable at that moment: a provider's streaming call returns the
response and drops the handler, and ``get_async_httpx_client`` caches handlers
behind a one-hour TTL and then lets them go.

The fix anchors the handler to the streaming response, so these tests turn on
*when* the handler is collected rather than on whether it is: pinned while the
body can still arrive, released once the caller is done with the response.

Nothing here re-tests the shapes ``_handler_may_close_client`` covers -- a
borrowed ``handler.client``, a caller-supplied client, an evicted-but-held
client. Those are pinned in ``tests/test_litellm/llms/custom_httpx/
test_http_handler.py``. What is uncovered there is the in-flight response, so no
test here may keep the client in a local: that inflates the very refcount under
test, and the test then passes on a broken handler. They hold weak references
instead, which the refcount does not count.

These live here rather than under ``tests/test_litellm/`` because they need a
real connection pool: a mocked transport goes on yielding chunks after its
client is closed, so the very teardown under test is what a mock cannot
reproduce. The server is a hermetic, credential-free ``ThreadingHTTPServer`` on
an ephemeral loopback port, and needs no network access beyond it.

Related: https://github.com/BerriAI/litellm/issues/24929
"""

import asyncio
import gc
import threading
import time
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import LlmProviders

FRAME_COUNT = 6
# Generous: the server emits all frames in ~0.3s. A client whose pool was torn
# down mid-stream can stall silently instead of raising, so reads are bounded.
READ_TIMEOUT_SECONDS = 15.0
RELEASE_TIMEOUT_SECONDS = 3.0

BOTH_TRANSPORTS = pytest.mark.parametrize("disable_aiohttp_transport", [False, True], ids=["aiohttp", "httpcore"])

STILL_PINNED = "the handler was released while its response could still read"
NOT_RELEASED = "the handler outlived the response that was holding it"


class _ChunkedSSEServer:
    """In-process HTTP/1.1 server that answers every request with chunked SSE frames."""

    def __init__(self, frame_count: int = FRAME_COUNT, frame_delay: float = 0.05) -> None:
        self.frame_count = frame_count
        self.frame_delay = frame_delay
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for index in range(parent.frame_count):
                        frame = f"data: frame-{index}\n\n".encode()
                        self.wfile.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                        self.wfile.flush()
                        time.sleep(parent.frame_delay)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_GET = _stream
            do_POST = _stream

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/stream"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()


def _select_transport(monkeypatch, disable_aiohttp_transport: bool) -> None:
    monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", disable_aiohttp_transport)
    monkeypatch.setattr(litellm, "force_ipv4", False)


async def _read_frames(response: httpx.Response) -> int:
    """Count SSE frames, collecting garbage between chunks so a finalizer has every chance to fire.

    The body is joined before counting: a chunk boundary can fall inside the
    marker, which a per-chunk count would miss.
    """
    chunks = []
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        gc.collect()
    return b"".join(chunks).count(b"data: frame-")


async def _wait_until(is_done, failure: str) -> None:
    deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_done():
            return
        await asyncio.sleep(0.05)
    pytest.fail(failure)


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_async_stream_survives_handler_collection(monkeypatch, disable_aiohttp_transport):
    """A response still streaming keeps working after its handler goes out of scope.

    The caller holds the response and nothing else, which is what a provider's
    streaming path is left with once ``post(..., stream=True)`` has returned.
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)

    with _ChunkedSSEServer() as server:
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        response = await handler.post(server.url, stream=True)

        ref = weakref.ref(handler)
        del handler
        gc.collect()
        await asyncio.sleep(0)  # let any close the finalizer scheduled run

        assert ref() is not None, STILL_PINNED
        assert await asyncio.wait_for(_read_frames(response), timeout=READ_TIMEOUT_SECONDS) == FRAME_COUNT

        del response
        gc.collect()
        assert ref() is None, NOT_RELEASED


def test_sync_stream_survives_handler_collection(monkeypatch):
    """The sync handler closes inline from its finalizer, so a stream must hold it off.

    litellm/main.py builds a sync handler only for non-streaming calls, commented
    "Keep this here, otherwise, the httpx.client closes and streaming is
    impossible" -- a workaround for this finalizer rather than a fix for it.
    """
    monkeypatch.setattr(litellm, "force_ipv4", False)

    with _ChunkedSSEServer() as server:
        handler = HTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        response = handler.post(server.url, stream=True)

        ref = weakref.ref(handler)
        del handler
        gc.collect()
        assert ref() is not None, STILL_PINNED

        # Joined before counting, as in ``_read_frames``.
        chunks = []
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            gc.collect()
        assert b"".join(chunks).count(b"data: frame-") == FRAME_COUNT

        del response
        gc.collect()
        assert ref() is None, NOT_RELEASED


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_an_abandoned_stream_still_releases_its_handler(monkeypatch, disable_aiohttp_transport):
    """A caller that drops a stream unread must not pin the handler for good.

    Tying the handler to the response's own lifetime is what bounds this. No
    deadline, and no poll of the connection's state, can tell an abandoned body
    from one the upstream is merely slow to finish: httpx leaves the connection
    checked out until the response is read or closed, and a legitimate stream is
    bounded only by how long the upstream keeps sending.
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)

    with _ChunkedSSEServer() as server:
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        client_ref = weakref.ref(handler.client)
        response = await handler.post(server.url, stream=True)

        ref = weakref.ref(handler)
        del handler, response
        gc.collect()

        assert ref() is None, NOT_RELEASED
        await _wait_until(
            lambda: client_ref() is None or client_ref().is_closed,
            "the client outlived the abandoned stream without being closed",
        )


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_the_pool_is_released_once_the_stream_it_carried_ends(monkeypatch, disable_aiohttp_transport):
    """Holding the finalizer off must defer the close, not drop it.

    Otherwise a collected handler leaks its pool for every streaming request it
    was carrying, and on aiohttp warns "Unclosed client session" when the
    collector eventually takes it. The pool and the session are children of the
    client, so keeping one here does not inflate the refcount the finalizer
    reads, the way keeping the client would.
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)

    with _ChunkedSSEServer() as server:
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        transport = handler.client._transport
        if disable_aiohttp_transport:
            pool = transport._pool

            def is_released() -> bool:
                return pool.connections == []
        else:
            session = transport._get_valid_client_session()

            def is_released() -> bool:
                return session.closed

        response = await handler.post(server.url, stream=True)

        del handler, transport
        gc.collect()
        assert not is_released(), "the pool was torn down while it was still carrying a body"

        assert await asyncio.wait_for(_read_frames(response), timeout=READ_TIMEOUT_SECONDS) == FRAME_COUNT
        del response
        gc.collect()

        await _wait_until(is_released, "the pool outlived the stream it carried, unclosed")


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_a_non_streaming_response_does_not_pin_its_handler(monkeypatch, disable_aiohttp_transport):
    """Only a body that can still arrive holds the handler.

    A non-streaming response has been read in full by the time ``post`` returns,
    so pinning the handler to it would delay every client close behind whatever
    the caller goes on to do with the response.
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)

    with _ChunkedSSEServer(frame_count=1, frame_delay=0.0) as server:
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        response = await handler.post(server.url)
        assert response.status_code == 200

        ref = weakref.ref(handler)
        del handler
        gc.collect()

        assert ref() is None, "a fully-read response pinned its handler"


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_cached_handler_eviction_does_not_abort_an_in_flight_stream(monkeypatch, disable_aiohttp_transport):
    """Evicting a cached handler mid-stream leaves the stream alone.

    ``get_async_httpx_client`` caches handlers for an hour. When that TTL
    expires the cache drops the only reference to a handler whose client is
    still streaming -- the production shape of #24929.
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    with _ChunkedSSEServer() as server:
        handler = get_async_httpx_client(llm_provider=LlmProviders.OPENAI)
        response = await handler.post(server.url, stream=True)

        # An hour passes: the TTL expires and the cache lets the handler go.
        ref = weakref.ref(handler)
        litellm.in_memory_llm_clients_cache.flush_cache()
        del handler
        gc.collect()

        assert ref() is not None, STILL_PINNED
        assert await asyncio.wait_for(_read_frames(response), timeout=READ_TIMEOUT_SECONDS) == FRAME_COUNT

        del response
        gc.collect()
        assert ref() is None, NOT_RELEASED
