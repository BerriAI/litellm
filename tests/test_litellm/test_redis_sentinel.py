"""
Tests for Redis Sentinel SSL and password support.
"""

import ssl
from unittest.mock import MagicMock, patch

from litellm._redis import (
    _build_sentinel_master_kwargs,
    _get_cached_redis_ssl_context,
    _get_redis_ssl_context,
    _make_shared_ssl_connection_class,
    _remove_sentinel_kwargs,
    _redis_ssl_context_cache,
    _redis_ssl_class_cache,
)


class TestRemoveSentinelKwargs:
    def test_removes_sentinel_keys(self):
        kwargs = {
            "host": "localhost",
            "sentinel_nodes": [("host", 26379)],
            "sentinel_password": "pass",
            "service_name": "mymaster",
            "password": "redis_pass",
        }
        result = _remove_sentinel_kwargs(kwargs)
        assert "sentinel_nodes" not in result
        assert "sentinel_password" not in result
        assert "service_name" not in result
        assert result["host"] == "localhost"
        assert result["password"] == "redis_pass"

    def test_does_not_modify_original(self):
        kwargs = {"sentinel_nodes": [("host", 26379)], "host": "localhost"}
        _remove_sentinel_kwargs(kwargs)
        assert "sentinel_nodes" in kwargs


class TestBuildSentinelMasterKwargs:
    def test_forwards_password_and_ssl(self):
        kwargs = {
            "password": "secret",
            "ssl": True,
            "ssl_cert_reqs": "none",
            "db": 0,
            "sentinel_nodes": [("h", 26379)],
        }
        result = _build_sentinel_master_kwargs(kwargs)
        assert result["password"] == "secret"
        assert result["ssl"] is True
        assert result["ssl_cert_reqs"] == "none"
        assert result["db"] == 0
        assert "sentinel_nodes" not in result

    def test_excludes_max_connections(self):
        kwargs = {"max_connections": 100, "password": "secret"}
        result = _build_sentinel_master_kwargs(kwargs)
        assert "max_connections" not in result

    def test_sets_default_socket_timeout(self):
        result = _build_sentinel_master_kwargs({})
        assert "socket_timeout" in result

    def test_sentinel_password_fallback_when_no_password(self):
        kwargs = {"sentinel_password": "sent_pass"}
        result = _build_sentinel_master_kwargs(kwargs)
        assert result["password"] == "sent_pass"

    def test_explicit_password_not_overridden_by_sentinel_password(self):
        kwargs = {"password": "master_pass", "sentinel_password": "sent_pass"}
        result = _build_sentinel_master_kwargs(kwargs)
        assert result["password"] == "master_pass"


class TestGetRedisSslContext:
    def test_default_verifies_certs(self):
        ctx = _get_redis_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_none_disables_verification(self):
        ctx = _get_redis_ssl_context(ssl_cert_reqs="none")
        assert ctx.verify_mode == ssl.CERT_NONE
        assert ctx.check_hostname is False

    def test_optional_mode(self):
        ctx = _get_redis_ssl_context(ssl_cert_reqs="optional")
        assert ctx.verify_mode == ssl.CERT_OPTIONAL
        assert ctx.check_hostname is False


class TestCachedSslContext:
    def setup_method(self):
        _redis_ssl_context_cache.clear()
        _redis_ssl_class_cache.clear()

    def test_returns_same_context(self):
        ctx1 = _get_cached_redis_ssl_context("none")
        ctx2 = _get_cached_redis_ssl_context("none")
        assert ctx1 is ctx2

    def test_different_reqs_different_context(self):
        ctx_none = _get_cached_redis_ssl_context("none")
        ctx_req = _get_cached_redis_ssl_context(None)
        assert ctx_none is not ctx_req


class TestMakeSharedSslConnectionClass:
    def test_overrides_wrap_socket(self):
        base_cls = type(
            "FakeSSL",
            (),
            {
                "__init__": lambda self, *a, **kw: None,
            },
        )
        cls = _make_shared_ssl_connection_class(base_cls, "none")
        assert cls.__name__.startswith("Shared")
        assert hasattr(cls, "_wrap_socket_with_ssl")

    def test_shared_ctx_reused_for_vanilla_ssl(self):
        """Vanilla SSL (no client certs) should use the shared context."""
        shared_ctx = _get_cached_redis_ssl_context("none")
        mock_sock = MagicMock()

        base_cls = type(
            "FakeSSL",
            (),
            {
                "__init__": lambda self, *a, **kw: None,
                "_wrap_socket_with_ssl": lambda self, sock: None,
            },
        )
        cls = _make_shared_ssl_connection_class(base_cls, "none")
        instance = cls()
        instance.host = "redis.example.com"
        # No certfile/keyfile/ca — vanilla case
        with patch.object(
            shared_ctx, "wrap_socket", return_value=mock_sock
        ) as mock_wrap:
            instance._wrap_socket_with_ssl(MagicMock())
            mock_wrap.assert_called_once()
            assert mock_wrap.call_args[1]["server_hostname"] == "redis.example.com"

    def test_falls_back_to_super_with_client_certs(self):
        """Client certs should trigger fallback to original redis-py method."""
        parent_called = []

        class FakeSSL:
            def __init__(self, *a, **kw):
                pass

            def _wrap_socket_with_ssl(self, sock):
                parent_called.append(True)
                return sock

        cls = _make_shared_ssl_connection_class(FakeSSL, "none")
        instance = cls()
        instance.certfile = "/path/to/cert.pem"
        instance.keyfile = None
        instance._wrap_socket_with_ssl(MagicMock())
        assert len(parent_called) == 1


class TestInitRedisSentinelSsl:
    @patch("litellm._redis.redis.Sentinel")
    def test_sentinel_receives_ssl_kwargs(self, mock_sentinel_cls):
        """Sentinel constructor should get sentinel_kwargs with ssl and password."""
        mock_sentinel = MagicMock()
        mock_sentinel_cls.return_value = mock_sentinel
        mock_sentinel.sentinels = []
        mock_sentinel.master_for.return_value = MagicMock()

        from litellm._redis import _init_redis_sentinel

        kwargs = {
            "sentinel_nodes": [("host1", 26379)],
            "sentinel_password": "sent_pass",
            "service_name": "mymaster",
            "password": "redis_pass",
            "ssl": True,
            "ssl_cert_reqs": "none",
        }
        _init_redis_sentinel(kwargs)

        # Check sentinel_kwargs was passed
        sentinel_call_kwargs = mock_sentinel_cls.call_args[1]
        assert "sentinel_kwargs" in sentinel_call_kwargs
        sk = sentinel_call_kwargs["sentinel_kwargs"]
        assert sk["password"] == "sent_pass"
        assert sk["ssl"] is True

        # Check master_for got password and connection_class (not raw ssl=True)
        master_call_kwargs = mock_sentinel.master_for.call_args[1]
        assert master_call_kwargs["password"] == "redis_pass"
        assert "ssl" not in master_call_kwargs  # replaced by connection_class
        assert "connection_class" in master_call_kwargs

    @patch("litellm._redis.async_redis.Sentinel")
    def test_async_sentinel_receives_ssl_kwargs(self, mock_sentinel_cls):
        """Async sentinel should also get sentinel_kwargs with ssl and password."""
        mock_sentinel = MagicMock()
        mock_sentinel_cls.return_value = mock_sentinel
        mock_sentinel.sentinels = []
        mock_sentinel.master_for.return_value = MagicMock()

        from litellm._redis import _init_async_redis_sentinel

        kwargs = {
            "sentinel_nodes": [("host1", 26379)],
            "sentinel_password": "sent_pass",
            "service_name": "mymaster",
            "password": "redis_pass",
            "ssl": True,
            "ssl_cert_reqs": "none",
        }
        _init_async_redis_sentinel(kwargs)

        sentinel_call_kwargs = mock_sentinel_cls.call_args[1]
        assert "sentinel_kwargs" in sentinel_call_kwargs
        sk = sentinel_call_kwargs["sentinel_kwargs"]
        assert sk["password"] == "sent_pass"
        assert sk["ssl"] is True
