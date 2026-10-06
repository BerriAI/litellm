"""
Tests for LLMClientCache.

The cache intentionally does NOT close clients on eviction because evicted
clients may still be referenced by in-flight requests.  Closing them eagerly
causes ``RuntimeError: Cannot send a request, as the client has been closed.``

See: https://github.com/BerriAI/litellm/pull/22247
"""

import asyncio
import gc
import importlib
import os
import threading
import time
import warnings
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import litellm
from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, get_async_httpx_client
from litellm.types.utils import LlmProviders
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from tests.fake_openai_endpoint import ensure_fake_openai_endpoint


class MockAsyncClient:
    """Mock async HTTP client with an async close method."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class MockSyncClient:
    """Mock sync HTTP client with a sync close method."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_remove_key_does_not_close_async_client():
    """
    Evicting an async client from LLMClientCache must NOT close it because
    an in-flight request may still hold a reference to the client.

    Regression test for production 'client has been closed' crashes.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0  # expired

    cache._remove_key("test-key")
    # Give the event loop a chance to run any background tasks
    await asyncio.sleep(0.1)

    # Client must NOT be closed — it may still be in use
    assert mock_client.closed is False
    assert "test-key" not in cache.cache_dict
    assert "test-key" not in cache.ttl_dict


def test_remove_key_does_not_close_sync_client():
    """
    Evicting a sync client from the cache must NOT close it.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockSyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    cache._remove_key("test-key")

    assert mock_client.closed is False
    assert "test-key" not in cache.cache_dict


@pytest.mark.asyncio
async def test_eviction_does_not_close_async_clients():
    """
    When the cache is full and an entry is evicted, the evicted async client
    must remain open and must not produce 'coroutine was never awaited' warnings.
    """
    cache = LLMClientCache(max_size_in_memory=2, default_ttl=1)

    clients = []
    for i in range(2):
        client = MockAsyncClient()
        clients.append(client)
        cache.set_cache(f"key-{i}", client)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        # This should trigger eviction of one of the existing entries
        cache.set_cache("key-new", "new-value")
        await asyncio.sleep(0.1)

    coroutine_warnings = [w for w in caught_warnings if "coroutine" in str(w.message).lower()]
    assert len(coroutine_warnings) == 0, f"Got unawaited coroutine warnings: {coroutine_warnings}"

    # Evicted clients must NOT be closed
    for client in clients:
        assert client.closed is False


@pytest.mark.asyncio
async def test_eviction_no_unawaited_coroutine_warning():
    """
    Evicting an async client from LLMClientCache must not produce
    'coroutine was never awaited' warnings.

    Regression test for https://github.com/BerriAI/litellm/issues/22128
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0  # expired

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cache._remove_key("test-key")
        await asyncio.sleep(0.1)

    coroutine_warnings = [w for w in caught_warnings if "coroutine" in str(w.message).lower()]
    assert len(coroutine_warnings) == 0, f"Got unawaited coroutine warnings: {coroutine_warnings}"


def test_remove_key_no_event_loop():
    """
    _remove_key works correctly even when there's no running event loop.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    # Should not raise even though there's no running event loop
    cache._remove_key("test-key")
    assert "test-key" not in cache.cache_dict


class _FakeClock:
    """Hand-advanced monotonic clock, so grace windows need no real waiting."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_evicted_litellm_owned_client_is_closed_once_the_grace_window_elapses():
    """
    Eviction only drops the cache's reference. The SDK clients are reference
    cycles, so without an explicit close the client keeps its connection pool
    open until a generational collection runs.
    """
    clock = _FakeClock()
    cache = LLMClientCache(
        max_size_in_memory=2,
        evicted_client_closer=EvictedClientCloser(grace_seconds=60.0, clock=clock),
    )

    client = MockAsyncClient()
    cache.set_cache("client-key", client, litellm_owned_client=True, ttl=600)

    cache.ttl_dict = {key: 0 for key in cache.ttl_dict}
    cache.expiration_heap = [(0, key) for _, key in cache.expiration_heap]
    cache.evict_cache()
    await asyncio.sleep(0.1)
    assert client.closed is False, "an in-flight request may still hold the client"

    clock.advance(61.0)
    cache.get_cache("any-key")
    await asyncio.sleep(0.1)

    assert client.closed is True


@pytest.mark.asyncio
async def test_evicted_caller_supplied_client_is_never_closed():
    """litellm does not own a client the caller passed in, so it must stay open."""
    clock = _FakeClock()
    cache = LLMClientCache(
        max_size_in_memory=2,
        evicted_client_closer=EvictedClientCloser(grace_seconds=60.0, clock=clock),
    )

    client = MockAsyncClient()
    cache.set_cache("client-key", client, ttl=600)

    cache.ttl_dict = {key: 0 for key in cache.ttl_dict}
    cache.expiration_heap = [(0, key) for _, key in cache.expiration_heap]
    cache.evict_cache()

    clock.advance(3600.0)
    cache.get_cache("any-key")
    await asyncio.sleep(0.1)

    assert client.closed is False


def test_remove_key_removes_plain_values():
    """
    _remove_key correctly removes non-client values (strings, dicts, etc.).
    """
    cache = LLMClientCache(max_size_in_memory=5)

    cache.cache_dict["str-key"] = "hello"
    cache.ttl_dict["str-key"] = 0
    cache.cache_dict["dict-key"] = {"foo": "bar"}
    cache.ttl_dict["dict-key"] = 0

    cache._remove_key("str-key")
    cache._remove_key("dict-key")

    assert "str-key" not in cache.cache_dict
    assert "dict-key" not in cache.cache_dict


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def fake_openai_endpoint():
    ensure_fake_openai_endpoint()
    yield


@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}


@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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
