"""
Tests for opt-in HTTP/2 support on outbound LLM requests.

HTTP/2 is gated behind `litellm.enable_http2` (or LITELLM_ENABLE_HTTP2). It
forces the httpx transport (aiohttp cannot speak HTTP/2) and passes http2=True
to the httpx sync/async clients. Default (off) behavior must be unchanged.
"""

import asyncio
import os
import sys

import httpx
import pytest
from aiohttp import ClientSession

sys.path.insert(0, os.path.abspath("../../../.."))
import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_http2_limits,
    _get_httpx_client,
    _http2_cache_key_suffix,
    _should_enable_http2,
    get_async_httpx_client,
)


@pytest.fixture(autouse=True)
def _restore_http2_globals():
    """Snapshot and restore all globals these tests mutate, plus env + cache."""
    saved = {
        "enable_http2": litellm.enable_http2,
        "http2_max_connections": litellm.http2_max_connections,
        "http2_max_keepalive_connections": litellm.http2_max_keepalive_connections,
        "force_ipv4": litellm.force_ipv4,
        "disable_aiohttp_transport": litellm.disable_aiohttp_transport,
    }
    saved_envs = {
        k: os.environ.get(k)
        for k in (
            "LITELLM_ENABLE_HTTP2",
            "LITELLM_HTTP2_MAX_CONNECTIONS",
            "LITELLM_HTTP2_MAX_KEEPALIVE_CONNECTIONS",
        )
    }
    saved_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    yield
    for k, v in saved.items():
        setattr(litellm, k, v)
    for k, v in saved_envs.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    litellm.in_memory_llm_clients_cache = saved_cache


def _pool_http2(client) -> bool:
    """Return the resolved http2 flag on a constructed httpx client's pool."""
    return getattr(client._transport._pool, "_http2", None)


# ---------------------------------------------------------------------------
# _should_enable_http2 — the single decision source
# ---------------------------------------------------------------------------
class TestShouldEnableHttp2:
    def test_default_is_off(self):
        assert litellm.enable_http2 is False
        assert _should_enable_http2() is False

    def test_bool_true(self):
        litellm.enable_http2 = True
        assert _should_enable_http2() is True

    @pytest.mark.parametrize(
        "value,expected",
        [
            # litellm.str_to_bool only recognizes "true"/"false" (case-insensitive);
            # anything else is treated as falsy here (consistent with other flags).
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("1", False),
            ("0", False),
            ("", False),
        ],
    )
    def test_string_values_from_config_yaml(self, value, expected):
        # config.yaml may pass a quoted string straight through setattr(litellm, ...)
        litellm.enable_http2 = value
        assert _should_enable_http2() is expected

    def test_env_var(self):
        litellm.enable_http2 = False
        os.environ["LITELLM_ENABLE_HTTP2"] = "True"
        assert _should_enable_http2() is True
        os.environ["LITELLM_ENABLE_HTTP2"] = "False"
        assert _should_enable_http2() is False

    def test_global_takes_priority_over_env(self):
        litellm.enable_http2 = True
        os.environ["LITELLM_ENABLE_HTTP2"] = "False"
        assert _should_enable_http2() is True


# ---------------------------------------------------------------------------
# Transport selection — http2 must route off aiohttp
# ---------------------------------------------------------------------------
class TestTransportSelection:
    def test_default_off_uses_aiohttp(self):
        litellm.enable_http2 = False
        assert AsyncHTTPHandler._should_use_aiohttp_transport() is True
        handler = AsyncHTTPHandler()
        assert "Aiohttp" in type(handler.client._transport).__name__

    def test_http2_on_disables_aiohttp(self):
        litellm.enable_http2 = True
        assert AsyncHTTPHandler._should_use_aiohttp_transport() is False

    def test_http2_short_circuit_comes_before_disable_flag(self):
        # Even with aiohttp not explicitly disabled, http2 wins.
        litellm.enable_http2 = True
        litellm.disable_aiohttp_transport = False
        assert AsyncHTTPHandler._should_use_aiohttp_transport() is False

    def test_disable_aiohttp_without_http2_still_works(self):
        litellm.enable_http2 = False
        litellm.disable_aiohttp_transport = True
        handler = AsyncHTTPHandler()
        assert "Aiohttp" not in type(handler.client._transport).__name__
        # and http2 is NOT enabled on that plain httpx client
        assert _pool_http2(handler.client) is False


# ---------------------------------------------------------------------------
# Client construction — http2 actually reaches the pool
# ---------------------------------------------------------------------------
class TestClientConstruction:
    def test_async_client_http2_enabled(self):
        litellm.enable_http2 = True
        handler = AsyncHTTPHandler()
        assert _pool_http2(handler.client) is True

    def test_async_client_http2_disabled_by_default(self):
        litellm.enable_http2 = False
        handler = AsyncHTTPHandler()
        # default path is aiohttp; no httpx _pool to inspect, assert transport type
        assert "Aiohttp" in type(handler.client._transport).__name__

    def test_sync_client_http2_enabled(self):
        litellm.enable_http2 = True
        handler = HTTPHandler()
        assert _pool_http2(handler.client) is True

    def test_sync_client_http2_disabled_by_default(self):
        litellm.enable_http2 = False
        handler = HTTPHandler()
        assert _pool_http2(handler.client) is False


# ---------------------------------------------------------------------------
# force_ipv4 + http2 — the explicit-transport edge case
# ---------------------------------------------------------------------------
class TestForceIpv4WithHttp2:
    def test_async_transport_carries_http2_and_local_address(self):
        litellm.force_ipv4 = True
        transport = AsyncHTTPHandler._create_httpx_transport(http2=True)
        assert isinstance(transport, httpx.AsyncHTTPTransport)
        assert getattr(transport._pool, "_http2", None) is True

    def test_sync_transport_carries_http2(self):
        litellm.force_ipv4 = True
        handler = HTTPHandler.__new__(HTTPHandler)
        transport = handler._create_sync_transport(http2=True)
        assert isinstance(transport, httpx.HTTPTransport)
        assert getattr(transport._pool, "_http2", None) is True

    def test_async_client_force_ipv4_plus_http2_end_to_end(self):
        litellm.force_ipv4 = True
        litellm.enable_http2 = True
        handler = AsyncHTTPHandler()
        assert _pool_http2(handler.client) is True

    def test_force_ipv4_plus_http2_limits_applied_async(self):
        # Regression: httpx ignores AsyncClient(limits=) when an explicit transport
        # is passed, so the force_ipv4 transport must carry the limits itself.
        litellm.force_ipv4 = True
        litellm.enable_http2 = True
        litellm.http2_max_connections = 7
        handler = AsyncHTTPHandler()
        assert handler.client._transport._pool._max_connections == 7

    def test_force_ipv4_plus_http2_limits_applied_sync(self):
        litellm.force_ipv4 = True
        litellm.enable_http2 = True
        litellm.http2_max_connections = 9
        handler = HTTPHandler()
        assert handler.client._transport._pool._max_connections == 9

    def test_force_ipv4_without_http2_has_no_http2(self):
        litellm.force_ipv4 = True
        transport = AsyncHTTPHandler._create_httpx_transport(http2=False)
        assert getattr(transport._pool, "_http2", None) is False

    def test_user_sync_transport_takes_priority_over_http2(self):
        # A user-supplied litellm.sync_transport must be returned as-is.
        litellm.force_ipv4 = False
        sentinel = httpx.HTTPTransport()
        litellm.sync_transport = sentinel
        try:
            handler = HTTPHandler.__new__(HTTPHandler)
            assert handler._create_sync_transport(http2=True) is sentinel
        finally:
            litellm.sync_transport = None

    def test_user_sync_transport_with_http2_emits_warning(self):
        # When http2 is enabled but a custom sync_transport swallows it, the user
        # must be warned rather than silently downgraded to HTTP/1.1.
        from unittest.mock import patch

        litellm.force_ipv4 = False
        litellm.enable_http2 = True
        sentinel = httpx.HTTPTransport()
        litellm.sync_transport = sentinel
        try:
            with patch(
                "litellm.llms.custom_httpx.http_handler.verbose_logger"
            ) as mock_logger:
                handler = HTTPHandler()
                assert handler.client._transport is sentinel
                assert mock_logger.warning.called
        finally:
            litellm.sync_transport = None


# ---------------------------------------------------------------------------
# shared_session priority — must not be silently dropped
# ---------------------------------------------------------------------------
class TestSharedSessionPriority:
    def test_shared_session_overrides_http2(self):
        litellm.enable_http2 = True

        async def _build():
            session = ClientSession()
            try:
                handler = AsyncHTTPHandler(shared_session=session)
                return type(handler.client._transport).__name__
            finally:
                await session.close()

        tname = asyncio.run(_build())
        assert "Aiohttp" in tname

    def test_shared_session_emits_warning(self):
        from unittest.mock import patch

        litellm.enable_http2 = True

        async def _build():
            session = ClientSession()
            try:
                with patch(
                    "litellm.llms.custom_httpx.http_handler.verbose_logger"
                ) as mock_logger:
                    AsyncHTTPHandler(shared_session=session)
                    assert mock_logger.warning.called
            finally:
                await session.close()

        asyncio.run(_build())

    def test_disable_aiohttp_with_shared_session_matches_original(self):
        # Zero-regression: in the original (pre-HTTP/2) code, disable_aiohttp_transport
        # made _should_use_aiohttp_transport() return False -> httpx transport, even
        # with a shared_session present (the session is simply not used). Our refactor
        # must preserve that exact behavior when http2 is OFF.
        litellm.enable_http2 = False
        litellm.disable_aiohttp_transport = True

        async def _build():
            session = ClientSession()
            try:
                handler = AsyncHTTPHandler(shared_session=session)
                return type(handler.client._transport).__name__
            finally:
                await session.close()

        tname = asyncio.run(_build())
        assert "Aiohttp" not in tname

    def test_disable_aiohttp_without_shared_session_uses_httpx(self):
        # And without a shared_session, disable_aiohttp_transport must still pick
        # httpx (HTTP/1.1) — unchanged from before the HTTP/2 feature.
        litellm.enable_http2 = False
        litellm.disable_aiohttp_transport = True
        handler = AsyncHTTPHandler()
        assert "Aiohttp" not in type(handler.client._transport).__name__

    def test_default_with_shared_session_uses_aiohttp(self):
        # Default config (aiohttp enabled) + shared_session -> aiohttp transport.
        litellm.enable_http2 = False
        litellm.disable_aiohttp_transport = False

        async def _build():
            session = ClientSession()
            try:
                handler = AsyncHTTPHandler(shared_session=session)
                return type(handler.client._transport).__name__
            finally:
                await session.close()

        assert "Aiohttp" in asyncio.run(_build())


# ---------------------------------------------------------------------------
# Connection pool limits
# ---------------------------------------------------------------------------
class TestHttp2Limits:
    def test_no_limits_by_default(self):
        litellm.http2_max_connections = None
        litellm.http2_max_keepalive_connections = None
        assert _get_http2_limits() is None

    def test_max_connections_only(self):
        litellm.http2_max_connections = 64
        litellm.http2_max_keepalive_connections = None
        limits = _get_http2_limits()
        assert limits is not None
        assert limits.max_connections == 64

    def test_both_limits(self):
        litellm.http2_max_connections = 64
        litellm.http2_max_keepalive_connections = 16
        limits = _get_http2_limits()
        assert limits.max_connections == 64
        assert limits.max_keepalive_connections == 16

    def test_limits_applied_to_async_client(self):
        litellm.enable_http2 = True
        litellm.http2_max_connections = 33
        handler = AsyncHTTPHandler()
        assert handler.client._transport._pool._max_connections == 33

    def test_limits_from_env_vars(self):
        litellm.http2_max_connections = None
        litellm.http2_max_keepalive_connections = None
        os.environ["LITELLM_HTTP2_MAX_CONNECTIONS"] = "77"
        os.environ["LITELLM_HTTP2_MAX_KEEPALIVE_CONNECTIONS"] = "12"
        limits = _get_http2_limits()
        assert limits.max_connections == 77
        assert limits.max_keepalive_connections == 12

    def test_global_takes_priority_over_env_for_limits(self):
        litellm.http2_max_connections = 5
        os.environ["LITELLM_HTTP2_MAX_CONNECTIONS"] = "999"
        assert _get_http2_limits().max_connections == 5

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_invalid_limit_values_raise(self, bad):
        litellm.http2_max_connections = bad
        with pytest.raises(ValueError, match="positive integer"):
            _get_http2_limits()

    def test_non_integer_limit_raises(self):
        litellm.http2_max_connections = "lots"
        with pytest.raises(ValueError, match="positive integer"):
            _get_http2_limits()

    def test_bool_limit_rejected(self):
        # bool is a subclass of int — must be rejected explicitly.
        litellm.http2_max_connections = True
        with pytest.raises(ValueError, match="positive integer"):
            _get_http2_limits()

    def test_invalid_env_limit_raises(self):
        litellm.http2_max_connections = None
        os.environ["LITELLM_HTTP2_MAX_CONNECTIONS"] = "not-a-number"
        with pytest.raises(ValueError, match="positive integer"):
            _get_http2_limits()


# ---------------------------------------------------------------------------
# Client cache isolation
# ---------------------------------------------------------------------------
class TestCacheIsolation:
    def test_async_cache_key_differs_for_http2(self):
        from litellm.types.llms.custom_http import httpxSpecialProvider

        # reset cache
        litellm.in_memory_llm_clients_cache = None

        litellm.enable_http2 = False
        h1 = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        litellm.enable_http2 = True
        h2 = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        # Different protocol clients must not be the same cached instance
        assert h1 is not h2
        assert _pool_http2(h2.client) is True

    def test_sync_cache_key_differs_for_http2(self):
        litellm.in_memory_llm_clients_cache = None

        litellm.enable_http2 = False
        c1 = _get_httpx_client()
        litellm.enable_http2 = True
        c2 = _get_httpx_client()
        assert c1 is not c2
        assert _pool_http2(c2.client) is True

    def test_suffix_empty_when_off(self):
        litellm.enable_http2 = False
        assert _http2_cache_key_suffix() == ""

    def test_suffix_raises_on_invalid_limits(self):
        # Consistency: invalid limits must fail fast at cache-key time, the same
        # way client construction would — not produce a key for an unbuildable client.
        litellm.enable_http2 = True
        litellm.http2_max_connections = -5
        with pytest.raises(ValueError, match="positive integer"):
            _http2_cache_key_suffix()

    def test_suffix_encodes_limits(self):
        litellm.enable_http2 = True
        litellm.http2_max_connections = None
        litellm.http2_max_keepalive_connections = None
        plain = _http2_cache_key_suffix()
        litellm.http2_max_connections = 50
        with_limits = _http2_cache_key_suffix()
        # Changing the limits must change the cache key so a stale client with a
        # different pool size is never reused.
        assert plain != with_limits
        assert "50" in with_limits

    def test_cache_returns_new_client_when_limits_change(self):
        litellm.in_memory_llm_clients_cache = None
        litellm.enable_http2 = True

        litellm.http2_max_connections = 10
        c1 = _get_httpx_client()
        litellm.http2_max_connections = 20
        c2 = _get_httpx_client()
        assert c1 is not c2
        assert c2.client._transport._pool._max_connections == 20


# ---------------------------------------------------------------------------
# Missing h2 package -> actionable error
# ---------------------------------------------------------------------------
class TestMissingH2:
    def test_clear_error_when_h2_missing(self):
        import builtins

        from litellm.llms.custom_httpx import http_handler

        litellm.enable_http2 = True
        # force the one-time availability check to run again
        saved = http_handler._HTTP2_AVAILABLE
        http_handler._HTTP2_AVAILABLE = None

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "h2":
                raise ImportError("no h2")
            return real_import(name, *args, **kwargs)

        from unittest.mock import patch

        try:
            with patch("builtins.__import__", side_effect=_fake_import):
                with pytest.raises(ImportError, match="h2"):
                    AsyncHTTPHandler()
        finally:
            # restore so later tests (and a real h2 install) aren't poisoned by
            # the cached False
            http_handler._HTTP2_AVAILABLE = saved

    def test_availability_result_is_cached(self):
        # After a successful check, the flag is True and no re-import happens.
        from litellm.llms.custom_httpx import http_handler

        http_handler._HTTP2_AVAILABLE = None
        http_handler._verify_http2_available()
        assert http_handler._HTTP2_AVAILABLE is True
        # second call is a no-op (returns without raising)
        http_handler._verify_http2_available()
