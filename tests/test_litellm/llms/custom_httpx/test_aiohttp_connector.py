import collections
from time import monotonic

import pytest
from aiohttp.client_proto import ResponseHandler
from aiohttp.client_reqrep import ConnectionKey

from litellm.llms.custom_httpx.aiohttp_connector import HardenedTCPConnector


class _FakeTransport:
    """Minimal transport stand-in: stays 'open' until close()/abort()."""

    def __init__(self):
        self._closing = False

    def is_closing(self):
        return self._closing

    def close(self):
        self._closing = True

    def abort(self):
        self._closing = True

    def get_extra_info(self, name, default=None):
        return default


def _make_key(host="api.openai.com"):
    return ConnectionKey(
        host=host,
        port=443,
        is_ssl=True,
        ssl=True,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )


def _make_pooled_proto(loop):
    proto = ResponseHandler(loop=loop)
    proto.transport = _FakeTransport()
    return proto


def _pool(connector, key, proto):
    connector._conns[key] = collections.deque([(proto, monotonic())])


@pytest.mark.asyncio
async def test_get_drops_connection_poisoned_after_pooling():
    """
    Regression for aiohttp 3.14.x connection-pool poisoning (litellm #33820).

    A connection is only pooled while should_close is False. If a stray
    sock_read timer later fires on the idle pooled connection it stamps a
    timeout exception and flips should_close True *without* closing the
    transport. Vanilla aiohttp _get would still hand it out (it only checks
    is_connected + keepalive window). HardenedTCPConnector must refuse it.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    connector = HardenedTCPConnector(keepalive_timeout=120)
    try:
        key = _make_key()
        proto = _make_pooled_proto(loop)

        assert proto.should_close is False
        _pool(connector, key, proto)

        # Emulate the post-release sock_read timer firing on the pooled conn.
        proto._on_read_timeout()
        assert proto.should_close is True
        assert proto.is_connected() is True  # transport was NOT closed

        conn = await connector._get(key, [])
        assert conn is None, "poisoned connection must not be reused"
        assert proto.is_connected() is False, "poisoned connection must be closed"
    finally:
        connector._conns.clear()


@pytest.mark.asyncio
async def test_get_reuses_healthy_connection():
    """A clean pooled connection is still reused - the fix must not break keepalive."""
    import asyncio

    loop = asyncio.get_running_loop()
    connector = HardenedTCPConnector(keepalive_timeout=120)
    try:
        key = _make_key()
        proto = _make_pooled_proto(loop)
        _pool(connector, key, proto)

        conn = await connector._get(key, [])
        assert conn is not None
        assert conn.protocol is proto
    finally:
        connector._conns.clear()


@pytest.mark.asyncio
async def test_get_skips_poisoned_and_returns_next_healthy():
    """With a poisoned then a healthy conn queued, _get skips past the poison."""
    import asyncio

    loop = asyncio.get_running_loop()
    connector = HardenedTCPConnector(keepalive_timeout=120)
    try:
        key = _make_key()
        poisoned = _make_pooled_proto(loop)
        healthy = _make_pooled_proto(loop)
        connector._conns[key] = collections.deque(
            [(poisoned, monotonic()), (healthy, monotonic())]
        )

        poisoned._on_read_timeout()
        assert poisoned.should_close is True

        conn = await connector._get(key, [])
        assert conn is not None
        assert conn.protocol is healthy
        assert poisoned.is_connected() is False
    finally:
        connector._conns.clear()


def test_http_handler_uses_hardened_connector():
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    assert http_handler_module.HardenedTCPConnector is HardenedTCPConnector
