import asyncio
import importlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
import httpx
import pytest

import litellm
from litellm.exceptions import Timeout as LitellmTimeout
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.model_cost_table import publish
from litellm.llms.base_llm.auth.types import TokenExchangeSpec
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    MaskedHTTPStatusError,
    get_httpx_client,
)
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


async def _read_http_request(reader: asyncio.StreamReader) -> None:
    raw = b""
    while b"\r\n\r\n" not in raw:
        chunk = await reader.read(1024)
        if not chunk:
            return
        raw += chunk
    head, _, body = raw.partition(b"\r\n\r\n")
    content_length = next(
        (int(line.split(b":", 1)[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")),
        0,
    )
    while len(body) < content_length:
        body += await reader.read(content_length - len(body))


@pytest.mark.asyncio
async def test_init_held_async_handler_survives_external_client_close():
    handler = AsyncHTTPHandler(timeout=42.5)
    held_client = handler.client
    await held_client.aclose()
    assert held_client.is_closed

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_http_request(reader)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        response = await handler.post(f"http://127.0.0.1:{port}/v1/compress", json={"messages": []})
    finally:
        server.close()
        await server.wait_closed()

    assert response.status_code == 200
    assert handler.client is not held_client
    assert handler.client.timeout == httpx.Timeout(42.5)
    await handler.close()


@pytest.mark.asyncio
async def test_init_held_async_handler_survives_evicted_client_close():
    from litellm.caching.evicted_client_closer import EvictedClientCloser
    from litellm.caching.llm_caching_handler import LLMClientCache

    cache = LLMClientCache(evicted_client_closer=EvictedClientCloser(grace_seconds=0))
    handler = AsyncHTTPHandler(timeout=42.5)
    held_client = handler.client
    cache.set_cache("init-held-handler", handler, litellm_owned_client=True, ttl=0)
    await asyncio.sleep(0.02)
    assert cache.get_cache("init-held-handler") is None
    await asyncio.sleep(0.05)
    assert held_client.is_closed

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_http_request(reader)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        response = await handler.post(f"http://127.0.0.1:{port}/v1/compress", json={"messages": []})
    finally:
        server.close()
        await server.wait_closed()

    assert response.status_code == 200
    assert handler.client is not held_client
    assert handler.client.timeout == httpx.Timeout(42.5)
    await handler.close()


def test_init_held_sync_handler_recreates_closed_client():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class OkRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    handler = HTTPHandler(timeout=7)
    held_client = handler.client
    held_client.close()

    server = HTTPServer(("127.0.0.1", 0), OkRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = handler.get(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status_code == 200
    assert handler.client is not held_client
    assert handler.client.timeout == httpx.Timeout(7)
    handler.close()


_WIF_SPEC: Final = TokenExchangeSpec(
    token_url="https://api.anthropic.com/v1/oauth/token",
    assertion_ref="oidc/env/ANTHROPIC_IDENTITY_TOKEN",
    assertion_field="assertion",
    static_body={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "federation_rule_id": "fdrl_1"},
    body_encoding="json",
    request_headers={},
    cache_key_identity=("fdrl_1", "org-1", "", ""),
    timeout_seconds=2.0,
)


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


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
    publish(litellm.model_cost)


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


_SERVER_DELAY_S = 5

_PER_REQUEST_TIMEOUT_S = 1.0

_CLIENT_DEFAULT_TIMEOUT_S = 60.0


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(_SERVER_DELAY_S)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        except OSError:
            pass

    def log_message(self, *args):
        pass


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_post_delay_exceeds_per_request_timeout_raises():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address

    handler = get_httpx_client(params={"timeout": _CLIENT_DEFAULT_TIMEOUT_S})
    try:
        with pytest.raises(LitellmTimeout):
            handler.post(
                f"http://{host}:{port}/delay",
                headers={"content-type": "application/json"},
                data=json.dumps({"model": "claude", "messages": []}),
                timeout=_PER_REQUEST_TIMEOUT_S,
            )
    except MaskedHTTPStatusError as e:
        pytest.skip(f"httpbin.org unavailable: {e}")
    finally:
        handler.close()
        server.shutdown()
        server.server_close()


_TIMEOUT_CLIENT_LIMITS: Final = httpx.Timeout(42.0, connect=7.0)
