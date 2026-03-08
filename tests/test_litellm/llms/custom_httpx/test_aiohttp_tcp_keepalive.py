"""
Tests for TCP keepalive fix and ClientTimeout.total fix.

Issue: azure_api_type: responses uses aiohttp transport which has ~60s socket
idle timeout, ignoring timeout parameter.

Root causes fixed:
1. aiohttp TCPConnector had no TCP keepalive, so Kubernetes/OpenShift load
   balancers silently closed idle TCP connections after ~60s during long
   Azure reasoning requests.
2. ClientTimeout was created without total=None, meaning aiohttp's default
   5-minute cap could silently override user-configured timeouts >300s.
"""
import os
import socket
import sys
from unittest.mock import patch

import aiohttp
import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    _AIOHTTP_SUPPORTS_SOCKET_FACTORY,
)


# ---------------------------------------------------------------------------
# Test 1 – TCP keepalive socket factory sets SO_KEEPALIVE on created sockets
# ---------------------------------------------------------------------------


def test_tcp_keepalive_socket_factory_sets_so_keepalive():
    """
    _make_tcp_keepalive_socket_factory() must produce a factory that enables
    SO_KEEPALIVE on every socket it creates. This is the OS-level mechanism
    that prevents Kubernetes/OpenShift load balancers from silently dropping
    idle TCP connections during long Azure reasoning requests.
    """
    factory = AsyncHTTPHandler._make_tcp_keepalive_socket_factory()

    # Build an addr_info tuple matching what getaddrinfo() returns
    addr_info = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))
    sock = factory(addr_info)

    try:
        keepalive = sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        # Non-zero means enabled; exact value is platform-specific (Linux=1, macOS=8)
        assert keepalive != 0, "SO_KEEPALIVE must be enabled on every socket"
    finally:
        sock.close()


def test_tcp_keepalive_socket_factory_sets_keepidle_on_linux():
    """
    On Linux (Kubernetes nodes), TCP_KEEPIDLE must be set so that keepalive
    probes start after the configured idle period, not the OS default (often
    2 hours) — which would be longer than the K8s LB idle timeout of ~60s.
    """
    if not hasattr(socket, "TCP_KEEPIDLE"):
        pytest.skip("TCP_KEEPIDLE not available on this platform")

    idle_secs = 30
    factory = AsyncHTTPHandler._make_tcp_keepalive_socket_factory(idle=idle_secs)

    addr_info = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))
    sock = factory(addr_info)

    try:
        actual = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)
        assert actual == idle_secs
    finally:
        sock.close()


def test_tcp_keepalive_socket_factory_sets_keepintvl_and_keepcnt():
    """
    TCP_KEEPINTVL (probe interval) and TCP_KEEPCNT (max failed probes) must
    also be set so the OS knows when to give up and close the connection.
    """
    if not hasattr(socket, "TCP_KEEPINTVL") or not hasattr(socket, "TCP_KEEPCNT"):
        pytest.skip("TCP_KEEPINTVL/TCP_KEEPCNT not available on this platform")

    factory = AsyncHTTPHandler._make_tcp_keepalive_socket_factory(
        idle=30, interval=10, count=5
    )

    addr_info = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))
    sock = factory(addr_info)

    try:
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 10
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 5
    finally:
        sock.close()


def test_tcp_keepalive_socket_factory_does_not_raise_on_unsupported_platform():
    """
    The factory must not raise even when TCP_KEEPIDLE / TCP_KEEPINTVL /
    TCP_KEEPCNT are unavailable (e.g. Windows or older kernels).
    SO_KEEPALIVE is still set; the fine-grained options fail silently.
    """
    factory = AsyncHTTPHandler._make_tcp_keepalive_socket_factory()

    addr_info = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))

    with patch.object(socket.socket, "setsockopt", side_effect=OSError("not supported")):
        # Must not raise
        sock = factory(addr_info)
        sock.close()


# ---------------------------------------------------------------------------
# Test 2 – TCPConnector is wired with socket_factory when keepalive is on
# ---------------------------------------------------------------------------


def test_create_aiohttp_transport_uses_socket_factory_when_keepalive_enabled():
    """
    When AIOHTTP_TCP_KEEPALIVE is True (default), _create_aiohttp_transport
    must call _make_tcp_keepalive_socket_factory() so the returned factory is
    wired into the TCPConnector. The connector is created lazily (inside a
    lambda), so we verify the factory builder was invoked rather than patching
    TCPConnector.__init__ directly.

    _AIOHTTP_SUPPORTS_SOCKET_FACTORY is forced to True so the test is not
    spuriously skipped on aiohttp 3.10/3.11 environments.
    """
    with patch(
        "litellm.llms.custom_httpx.http_handler.AIOHTTP_TCP_KEEPALIVE", True
    ), patch(
        "litellm.llms.custom_httpx.http_handler._AIOHTTP_SUPPORTS_SOCKET_FACTORY", True
    ), patch.object(
        AsyncHTTPHandler,
        "_make_tcp_keepalive_socket_factory",
        wraps=AsyncHTTPHandler._make_tcp_keepalive_socket_factory,
    ) as mock_factory_builder:
        AsyncHTTPHandler._create_aiohttp_transport()

    assert mock_factory_builder.call_count == 1, (
        "_make_tcp_keepalive_socket_factory must be called when AIOHTTP_TCP_KEEPALIVE=True"
    )


def test_create_aiohttp_transport_no_socket_factory_when_keepalive_disabled():
    """
    When AIOHTTP_TCP_KEEPALIVE is False, _make_tcp_keepalive_socket_factory
    must NOT be called so aiohttp uses its default socket creation.
    """
    with patch(
        "litellm.llms.custom_httpx.http_handler.AIOHTTP_TCP_KEEPALIVE", False
    ), patch.object(
        AsyncHTTPHandler,
        "_make_tcp_keepalive_socket_factory",
    ) as mock_factory_builder:
        AsyncHTTPHandler._create_aiohttp_transport()

    assert mock_factory_builder.call_count == 0, (
        "_make_tcp_keepalive_socket_factory must NOT be called when AIOHTTP_TCP_KEEPALIVE=False"
    )


# ---------------------------------------------------------------------------
# Test 3 – ClientTimeout.total is None so aiohttp's 300s default is removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_timeout_total_is_none_for_user_configured_timeout():
    """
    When an explicit read timeout is configured, ClientTimeout.total must be
    None so aiohttp's implicit 300 s cap does not silently override the user's
    value (e.g. timeout=600 for heavy Azure reasoning).  Only sock_read
    controls how long to wait for data in that case.
    """
    captured = {}

    class FakeSession:
        closed = False
        _loop = None

        def request(self, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, size):
                            yield b""

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())

    request = httpx.Request("POST", "https://example.com/")
    request.extensions["timeout"] = {
        "connect": 5.0,
        "read": 600.0,  # user configured timeout=600
        "write": 600.0,
        "pool": 5.0,
    }

    await transport.handle_async_request(request)

    assert "timeout" in captured
    client_timeout: aiohttp.ClientTimeout = captured["timeout"]
    assert client_timeout.total is None, (
        "ClientTimeout.total must be None to avoid aiohttp's 300s default cap "
        "overriding the user's configured timeout"
    )
    assert client_timeout.sock_read == 600.0, (
        "sock_read must be set to the user's configured timeout value"
    )


@pytest.mark.asyncio
async def test_client_timeout_total_none_when_read_timeout_is_set():
    """
    When sock_read is set, ClientTimeout.total must be None so aiohttp's
    implicit 300 s cap does not prematurely cut streaming responses whose
    total duration exceeds the per-chunk sock_read budget.
    """
    captured = {}

    class FakeSession:
        closed = False
        _loop = None

        def request(self, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, _size):
                            for chunk in [b"chunk0\n", b"chunk1\n", b"chunk2\n"]:
                                yield chunk

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore

    request = httpx.Request("GET", "https://example.com/stream")
    # Simulate a short sock_read timeout — total=None must not cap the stream
    request.extensions["timeout"] = {
        "connect": 5.0,
        "read": 0.15,
        "pool": 5.0,
    }

    await transport.handle_async_request(request)

    assert "timeout" in captured
    client_timeout: aiohttp.ClientTimeout = captured["timeout"]
    assert client_timeout.total is None, (
        "ClientTimeout.total must be None when sock_read is set so streaming "
        "responses are not capped by aiohttp's implicit 300 s default"
    )
    assert client_timeout.sock_read == 0.15


@pytest.mark.asyncio
async def test_client_timeout_total_is_300_when_no_read_timeout():
    """
    When no explicit read timeout is provided (sock_read=None), total must
    fall back to 300 s to preserve the backwards-compatible safety net against
    hung servers — removing it unconditionally would be a breaking change.
    """
    captured = {}

    class FakeSession:
        closed = False
        _loop = None

        def request(self, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, _size):
                            yield b""

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore

    request = httpx.Request("POST", "https://example.com/")
    # No read timeout — simulates a caller that omits timeout entirely
    request.extensions["timeout"] = {}

    await transport.handle_async_request(request)

    client_timeout: aiohttp.ClientTimeout = captured["timeout"]
    assert client_timeout.total == 300, (
        "ClientTimeout.total must be 300 s when no read timeout is configured "
        "to preserve the safety net against hung servers"
    )
    assert client_timeout.sock_read is None


@pytest.mark.asyncio
async def test_client_timeout_total_is_300_when_opt_out_flag_set():
    """
    When AIOHTTP_KEEP_TOTAL_TIMEOUT_CAP=true, the old 300 s total cap must be
    restored even when an explicit read timeout is configured, allowing users to
    opt back into the previous behaviour without upgrading.
    """
    captured = {}

    class FakeSession:
        closed = False
        _loop = None

        def request(self, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, _size):
                            yield b""

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore

    request = httpx.Request("POST", "https://example.com/")
    request.extensions["timeout"] = {"connect": 5.0, "read": 600.0, "pool": 5.0}

    with patch(
        "litellm.llms.custom_httpx.aiohttp_transport.AIOHTTP_KEEP_TOTAL_TIMEOUT_CAP",
        True,
    ):
        await transport.handle_async_request(request)

    client_timeout: aiohttp.ClientTimeout = captured["timeout"]
    assert client_timeout.total == 300, (
        "ClientTimeout.total must be 300 s when AIOHTTP_KEEP_TOTAL_TIMEOUT_CAP=true"
    )


@pytest.mark.asyncio
async def test_tcp_connector_accepts_socket_factory_on_supported_aiohttp():
    """
    On aiohttp>=3.12 (where socket_factory is supported), TCPConnector must be
    instantiable with the factory returned by _make_tcp_keepalive_socket_factory
    without raising TypeError.  On older versions the test is skipped.

    This is the regression guard the reviewer requested: the tests that check
    whether _make_tcp_keepalive_socket_factory is *called* do not catch a
    TypeError raised when TCPConnector is actually *instantiated* with the kwarg.
    """
    if not _AIOHTTP_SUPPORTS_SOCKET_FACTORY:
        pytest.skip("socket_factory not supported in this aiohttp version (<3.12)")

    factory = AsyncHTTPHandler._make_tcp_keepalive_socket_factory()
    # Must not raise TypeError
    connector = aiohttp.TCPConnector(socket_factory=factory)
    await connector.close()
