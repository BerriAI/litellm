"""
Garbage-collecting an HTTP handler must not close the httpx client it holds.

``HTTPHandler`` and ``AsyncHTTPHandler`` used to close their client from
``__del__``. Closing an httpx client tears down the connection pool that every
in-flight response is streaming through, and it permanently invalidates the
client for future requests -- including for callers who only ever borrowed
``handler.client``. Since nothing in a response's reference graph points back
at the handler, and since litellm caches handlers behind a one-hour TTL, the
handler routinely became collectable while its client was still in use.

``LLMClientCache`` documents the invariant this broke: evicted clients "may
still be in use by in-flight requests", so they are left to normal garbage
collection rather than closed eagerly. A finalizer that closes on collection
defeats exactly that.

Each test below is one shape the finalizers broke; all of them fail if either
``__del__`` comes back. Async cases run on both transports, because litellm
defaults to aiohttp and only uses httpcore when aiohttp is disabled.

These live here rather than under ``tests/test_litellm/`` because they need a
real connection pool: a mocked transport goes on yielding chunks after its
client is closed, so the very teardown under test is what a mock cannot
reproduce. The server is a hermetic, credential-free ``ThreadingHTTPServer``
on an ephemeral loopback port, and needs no network access beyond it.

Related: https://github.com/BerriAI/litellm/issues/24929
"""

import asyncio
import gc
import os
import sys
import threading
import time
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

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

BOTH_TRANSPORTS = pytest.mark.parametrize("disable_aiohttp_transport", [False, True], ids=["aiohttp", "httpcore"])

# Every test here rests on the handler actually being collected at the ``del``.
# If something ever pins it, the test would pass while guarding nothing, so each
# one checks the premise. The check comes last: a reintroduced finalizer also
# fails it, by resurrecting the handler into the task it creates for ``close()``,
# and the transport error is the more useful thing to see first.
HANDLER_NOT_COLLECTED = "handler was not collected; this test no longer exercises the finalizer path"


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


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_async_stream_survives_handler_collection(monkeypatch, disable_aiohttp_transport):
    """A response being streamed keeps working after its handler is collected."""
    _select_transport(monkeypatch, disable_aiohttp_transport)

    with _ChunkedSSEServer() as server:
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        client = handler.client
        try:
            response = await client.send(client.build_request("GET", server.url), stream=True)

            # The handler loses its last reference while the body is still streaming.
            ref = weakref.ref(handler)
            del handler
            gc.collect()

            frames = await asyncio.wait_for(_read_frames(response), timeout=READ_TIMEOUT_SECONDS)
            assert frames == FRAME_COUNT
            assert client.is_closed is False
            assert ref() is None, HANDLER_NOT_COLLECTED
        finally:
            await client.aclose()


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_borrowed_async_client_outlives_its_handler(monkeypatch, disable_aiohttp_transport):
    """A caller that keeps only ``handler.client`` can still send requests once the handler is gone.

    This is the shape at litellm/a2a_protocol/main.py (``httpx_client =
    _async_handler.client``, handed to the a2a SDK) and at
    litellm/proxy/pass_through_endpoints/pass_through_endpoints.py (``async_client
    = async_client_obj.client``). Both take the handler from
    ``get_async_httpx_client``, so the cache pins it for
    ``_DEFAULT_TTL_FOR_HTTPX_CLIENTS`` (one hour) and then lets it go on eviction,
    at which point it is collected while the borrowed client is still serving a
    longer-lived consumer: ``create_a2a_client`` hands its client to the a2a SDK
    and documents it as "create client once, reuse for multiple requests".
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)

    with _ChunkedSSEServer(frame_count=1, frame_delay=0.0) as server:
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        client = handler.client
        try:
            ref = weakref.ref(handler)
            del handler
            gc.collect()
            # A finalizer would close the client from a task, so let the loop turn.
            await asyncio.sleep(0.05)

            assert client.is_closed is False
            response = await client.get(server.url)
            assert response.status_code == 200
            assert ref() is None, HANDLER_NOT_COLLECTED
        finally:
            await client.aclose()


def test_sync_handler_collection_does_not_close_a_caller_owned_client(monkeypatch):
    """A throwaway handler wrapped around someone else's client must not close it.

    litellm/llms/azure/azure.py builds ``HTTPHandler(client=litellm.client_session)``
    for a single image generation and drops it. With a finalizer, that one call
    left the user's shared session closed for the rest of the process.
    """
    monkeypatch.setattr(litellm, "force_ipv4", False)

    with _ChunkedSSEServer(frame_count=1, frame_delay=0.0) as server:
        caller_client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
        monkeypatch.setattr(litellm, "client_session", caller_client)
        try:
            handler = HTTPHandler(client=litellm.client_session)
            ref = weakref.ref(handler)
            del handler
            gc.collect()

            assert caller_client.is_closed is False
            assert caller_client.get(server.url).status_code == 200
            assert ref() is None, HANDLER_NOT_COLLECTED
        finally:
            caller_client.close()


def test_sync_stream_survives_handler_collection(monkeypatch):
    """A sync response being streamed keeps working after its handler is collected.

    litellm/main.py builds a sync handler only for non-streaming calls, commented
    "Keep this here, otherwise, the httpx.client closes and streaming is
    impossible" -- a workaround for this finalizer rather than a fix for it.
    """
    monkeypatch.setattr(litellm, "force_ipv4", False)

    with _ChunkedSSEServer() as server:
        handler = HTTPHandler(timeout=httpx.Timeout(10.0, connect=5.0))
        client = handler.client
        try:
            response = client.send(client.build_request("GET", server.url), stream=True)

            # The handler loses its last reference while the body is still streaming.
            ref = weakref.ref(handler)
            del handler
            gc.collect()

            # Joined before counting, as in ``_read_frames``.
            chunks = []
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                gc.collect()

            assert b"".join(chunks).count(b"data: frame-") == FRAME_COUNT
            assert client.is_closed is False
            assert ref() is None, HANDLER_NOT_COLLECTED
        finally:
            client.close()


@pytest.mark.asyncio
@BOTH_TRANSPORTS
async def test_cached_handler_eviction_does_not_abort_an_in_flight_stream(monkeypatch, disable_aiohttp_transport):
    """Evicting a cached handler mid-stream leaves the stream alone.

    ``get_async_httpx_client`` caches handlers for an hour. When that TTL
    expires the cache drops the only reference to a handler whose client is
    still streaming -- the production shape of #24929, and the case
    ``LLMClientCache`` documents as "may still be in use by in-flight requests".
    """
    _select_transport(monkeypatch, disable_aiohttp_transport)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    with _ChunkedSSEServer() as server:
        handler = get_async_httpx_client(llm_provider=LlmProviders.OPENAI)
        client = handler.client
        try:
            response = await client.send(client.build_request("GET", server.url), stream=True)

            # An hour passes: the TTL expires and the cache lets the handler go.
            ref = weakref.ref(handler)
            litellm.in_memory_llm_clients_cache.flush_cache()
            del handler
            gc.collect()

            frames = await asyncio.wait_for(_read_frames(response), timeout=READ_TIMEOUT_SECONDS)
            assert frames == FRAME_COUNT
            assert client.is_closed is False

            # And the evicted client is still usable for the next request.
            assert (await client.get(server.url)).status_code == 200
            assert ref() is None, HANDLER_NOT_COLLECTED
        finally:
            await client.aclose()
