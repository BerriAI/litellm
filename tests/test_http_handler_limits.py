"""
Tests for httpx connection pool limits (fixes #13220).

Verifies that httpx clients are created with bounded connection pools
to prevent file descriptor exhaustion under parallel load.
"""

import os
from unittest.mock import patch

import httpx


def _get_pool_limits(client):
    """Extract pool limits from an httpx client via private internals.

    httpx does not expose pool configuration through a public API, so we
    access ``_transport._pool._max_connections`` directly.  If httpx
    changes its internals this will raise ``AttributeError`` — that is
    intentional so CI catches the breakage rather than silently passing.
    """
    pool = client._transport._pool
    return pool._max_connections, pool._max_keepalive_connections


def test_sync_client_has_connection_limits():
    """HTTPHandler creates httpx.Client with pool limits."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler = HTTPHandler(timeout=httpx.Timeout(timeout=10.0))
    max_conn, max_keepalive = _get_pool_limits(handler.client)
    assert max_conn == 100
    assert max_keepalive == 20
    handler.close()


def test_async_client_has_connection_limits_with_httpx_transport():
    """AsyncHTTPHandler uses pool limits when httpx transport is active."""
    import litellm

    original = getattr(litellm, "disable_aiohttp_transport", False)
    try:
        litellm.disable_aiohttp_transport = True
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        handler = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=10.0))
        max_conn, max_keepalive = _get_pool_limits(handler.client)
        assert max_conn == 100
        assert max_keepalive == 20
    finally:
        litellm.disable_aiohttp_transport = original


def test_limits_configurable_via_env_max_connections():
    """LITELLM_HTTP_MAX_CONNECTIONS env var controls max_connections."""
    import importlib

    import litellm.constants

    with patch.dict(os.environ, {"LITELLM_HTTP_MAX_CONNECTIONS": "50"}):
        importlib.reload(litellm.constants)
        from litellm.constants import HTTPX_MAX_CONNECTIONS

        assert HTTPX_MAX_CONNECTIONS == 50

    os.environ.pop("LITELLM_HTTP_MAX_CONNECTIONS", None)
    importlib.reload(litellm.constants)


def test_limits_configurable_via_env_max_keepalive():
    """LITELLM_HTTP_MAX_KEEPALIVE env var controls max_keepalive_connections."""
    import importlib

    import litellm.constants

    with patch.dict(os.environ, {"LITELLM_HTTP_MAX_KEEPALIVE": "10"}):
        importlib.reload(litellm.constants)
        from litellm.constants import HTTPX_MAX_KEEPALIVE_CONNECTIONS

        assert HTTPX_MAX_KEEPALIVE_CONNECTIONS == 10

    os.environ.pop("LITELLM_HTTP_MAX_KEEPALIVE", None)
    importlib.reload(litellm.constants)


def test_default_constants_values():
    """Default constants match expected values."""
    from litellm.constants import HTTPX_MAX_CONNECTIONS, HTTPX_MAX_KEEPALIVE_CONNECTIONS

    assert HTTPX_MAX_CONNECTIONS == 100
    assert HTTPX_MAX_KEEPALIVE_CONNECTIONS == 20


def test_invalid_env_values_fallback_to_defaults():
    """Non-integer env values fall back to defaults instead of crashing."""
    import importlib

    import litellm.constants

    with patch.dict(
        os.environ,
        {"LITELLM_HTTP_MAX_CONNECTIONS": "not_a_number", "LITELLM_HTTP_MAX_KEEPALIVE": ""},
    ):
        importlib.reload(litellm.constants)
        assert litellm.constants.HTTPX_MAX_CONNECTIONS == 100
        assert litellm.constants.HTTPX_MAX_KEEPALIVE_CONNECTIONS == 20

    os.environ.pop("LITELLM_HTTP_MAX_CONNECTIONS", None)
    os.environ.pop("LITELLM_HTTP_MAX_KEEPALIVE", None)
    importlib.reload(litellm.constants)


def test_multiple_sync_handlers_all_enforce_limits():
    """Multiple HTTPHandler instances all enforce pool limits."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handlers = [HTTPHandler(timeout=httpx.Timeout(timeout=10.0)) for _ in range(3)]
    for handler in handlers:
        max_conn, max_keepalive = _get_pool_limits(handler.client)
        assert max_conn == 100
        assert max_keepalive == 20
        handler.close()


def test_get_httpx_client_returns_handler_with_limits():
    """_get_httpx_client returns an HTTPHandler with pool limits."""
    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

    client = _get_httpx_client()
    max_conn, max_keepalive = _get_pool_limits(client.client)
    assert max_conn == 100
    assert max_keepalive == 20


def test_sync_client_custom_env_limits():
    """HTTPHandler respects custom env var limits at creation time."""
    import importlib

    import litellm.constants
    import litellm.llms.custom_httpx.http_handler as handler_mod

    with patch.dict(
        os.environ,
        {"LITELLM_HTTP_MAX_CONNECTIONS": "200", "LITELLM_HTTP_MAX_KEEPALIVE": "40"},
    ):
        importlib.reload(litellm.constants)
        importlib.reload(handler_mod)
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        h = HTTPHandler(timeout=httpx.Timeout(timeout=10.0))
        max_conn, max_keepalive = _get_pool_limits(h.client)
        assert max_conn == 200
        assert max_keepalive == 40
        h.close()

    os.environ.pop("LITELLM_HTTP_MAX_CONNECTIONS", None)
    os.environ.pop("LITELLM_HTTP_MAX_KEEPALIVE", None)
    importlib.reload(litellm.constants)
    importlib.reload(handler_mod)


def test_invalid_env_values_fallback_to_defaults_non_numeric():
    """Non-integer env values fall back to defaults instead of crashing."""
    import importlib

    import litellm.constants

    with patch.dict(
        os.environ,
        {"LITELLM_HTTP_MAX_CONNECTIONS": "not_a_number", "LITELLM_HTTP_MAX_KEEPALIVE": ""},
    ):
        importlib.reload(litellm.constants)
        from litellm.constants import HTTPX_MAX_CONNECTIONS, HTTPX_MAX_KEEPALIVE_CONNECTIONS

        assert HTTPX_MAX_CONNECTIONS == 100
        assert HTTPX_MAX_KEEPALIVE_CONNECTIONS == 20

    os.environ.pop("LITELLM_HTTP_MAX_CONNECTIONS", None)
    os.environ.pop("LITELLM_HTTP_MAX_KEEPALIVE", None)
    importlib.reload(litellm.constants)


def test_force_ipv4_transport_has_pool_limits():
    """When force_ipv4=True, custom transport must carry the same pool limits."""
    import litellm
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    original = getattr(litellm, "force_ipv4", False)
    try:
        litellm.force_ipv4 = True
        handler = HTTPHandler(timeout=httpx.Timeout(timeout=10.0))
        max_conn, max_keepalive = _get_pool_limits(handler.client)
        assert max_conn == 100
        assert max_keepalive == 20
        handler.close()
    finally:
        litellm.force_ipv4 = original
