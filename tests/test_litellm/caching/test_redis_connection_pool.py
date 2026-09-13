from importlib import import_module
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm._redis import (
    _coerce_redis_kwargs_types,
    _get_redis_client_logic,
    _get_redis_env_kwarg_mapping,
    get_redis_async_client,
    get_redis_connection_pool,
)


def test_url_config_uses_passed_pool():
    """When connection_pool is provided with a URL config, the client
    should use the passed pool — not create a new one via from_url()."""
    mock_pool = MagicMock()

    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {"url": "redis://localhost:6379/0"}

        client = get_redis_async_client(connection_pool=mock_pool)

    assert client.connection_pool is mock_pool


def test_url_config_falls_back_to_from_url_without_pool():
    """When no connection_pool is provided, URL config should still
    use from_url() as before."""
    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {"url": "redis://localhost:6379/0"}

        client = get_redis_async_client()

    # from_url creates its own pool — just verify it's not None
    assert client.connection_pool is not None


def test_max_connections_url_config(monkeypatch):
    """max_connections should be respected when using URL-based config."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "10")

    pool = get_redis_connection_pool()

    assert pool.max_connections == 10


def test_max_connections_url_config_string_value(monkeypatch):
    """max_connections provided as a string (from env var) should be
    cast to int."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "25")

    pool = get_redis_connection_pool()

    assert pool.max_connections == 25


def test_max_connections_url_config_invalid_value(monkeypatch):
    """Invalid max_connections from an env var should be silently dropped,
    falling back to the pool default (50 for BlockingConnectionPool)."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "not_a_number")

    pool = get_redis_connection_pool()

    # BlockingConnectionPool default is 50
    assert pool.max_connections == 50


def test_max_connections_url_config_none_value():
    """max_connections=None should be silently ignored."""
    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "url": "redis://localhost:6379/0",
            "max_connections": None,
        }

        pool = get_redis_connection_pool()

    assert pool.max_connections == 50


def _make_redis_cache():
    """Create a RedisCache with all external I/O mocked out."""
    mock_sync_client = MagicMock()
    mock_async_pool = AsyncMock()
    patches = [
        patch("litellm._redis.get_redis_client", return_value=mock_sync_client),
        patch("litellm._redis.get_redis_connection_pool", return_value=mock_async_pool),
        patch.object(import_module("litellm.caching.redis_cache").RedisCache, "_setup_health_pings"),
    ]
    for p in patches:
        p.start()

    from litellm.caching.redis_cache import RedisCache

    cache = RedisCache(host="localhost", port=6379)

    for p in patches:
        p.stop()

    return cache, mock_sync_client, mock_async_pool


@pytest.mark.asyncio
async def test_disconnect_closes_sync_client():
    """disconnect() should close both the async pool and the sync client."""
    cache, mock_sync_client, mock_async_pool = _make_redis_cache()
    await cache.disconnect()

    mock_async_pool.disconnect.assert_awaited_once_with(inuse_connections=True)
    mock_sync_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_idempotent():
    """Calling disconnect() twice should not raise."""
    cache, mock_sync_client, mock_async_pool = _make_redis_cache()
    mock_sync_client.close.side_effect = [None, RuntimeError("already closed")]

    await cache.disconnect()
    await cache.disconnect()  # should not raise


def test_coerce_redis_kwargs_types_int():
    """String values for int-typed Redis params are coerced to int."""
    result = _coerce_redis_kwargs_types({"health_check_interval": "30", "port": "6380", "db": "1"})
    assert result["health_check_interval"] == 30
    assert isinstance(result["health_check_interval"], int)
    assert result["port"] == 6380
    assert result["db"] == 1


def test_coerce_redis_kwargs_types_bool():
    """String values for bool-typed Redis params are coerced to bool."""
    result = _coerce_redis_kwargs_types({"ssl": "true", "decode_responses": "false"})
    assert result["ssl"] is True
    assert result["decode_responses"] is False


def test_coerce_redis_kwargs_types_none_default_numeric():
    """String values for known None-default numeric params are coerced."""
    result = _coerce_redis_kwargs_types({"max_connections": "20", "socket_timeout": "5.5"})
    assert result["max_connections"] == 20
    assert isinstance(result["max_connections"], int)
    assert result["socket_timeout"] == 5.5
    assert isinstance(result["socket_timeout"], float)


def _redis_signature_pre_8x(
    socket_timeout=None,
    socket_connect_timeout=None,
    max_connections=None,
    health_check_interval=0,
):
    """Stand-in for the redis-py <= 7.x Redis signature, where the timeout defaults are None."""


def _redis_signature_8x(
    socket_timeout=5,
    socket_connect_timeout=5,
    max_connections=None,
    health_check_interval=0,
):
    """Stand-in for the redis-py 8.x Redis signature, where the timeout defaults became int 5."""


@pytest.mark.parametrize(
    "client",
    [_redis_signature_pre_8x, _redis_signature_8x],
    ids=["redis-py<=7.x", "redis-py-8.x"],
)
def test_coerce_fractional_socket_timeout_survives_signature_default_change(client):
    """redis-py 8.x changed socket_timeout's default from None to int 5. Deriving the
    target type from the signature default made int("5.5") raise, so the key was dropped
    and REDIS_SOCKET_TIMEOUT=5.5 silently disappeared on 8.x."""
    result = _coerce_redis_kwargs_types(
        {"socket_timeout": "5.5", "socket_connect_timeout": "2.5", "max_connections": "20"},
        client=client,
    )

    assert result["socket_timeout"] == pytest.approx(5.5)
    assert isinstance(result["socket_timeout"], float)
    assert result["socket_connect_timeout"] == pytest.approx(2.5)
    assert isinstance(result["socket_connect_timeout"], float)
    assert result["max_connections"] == 20
    assert isinstance(result["max_connections"], int)


def test_coerce_invalid_socket_timeout_is_still_dropped():
    """Garbage must not survive the explicit-type path; Redis falls back to its own default."""
    result = _coerce_redis_kwargs_types({"socket_timeout": "not_a_number"}, client=_redis_signature_8x)

    assert "socket_timeout" not in result


def test_coerce_redis_kwargs_types_invalid_drops_key():
    """A string that cannot be coerced to the expected numeric type is dropped."""
    result = _coerce_redis_kwargs_types({"health_check_interval": "not_a_number"})
    assert "health_check_interval" not in result


def test_coerce_redis_kwargs_types_non_string_unchanged():
    """Non-string values pass through without modification."""
    result = _coerce_redis_kwargs_types({"health_check_interval": 30, "ssl": True})
    assert result["health_check_interval"] == 30
    assert result["ssl"] is True


def test_health_check_interval_from_env_is_int(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "30")

    pool = get_redis_connection_pool()

    assert pool is not None
    interval = pool.connection_kwargs.get("health_check_interval")
    assert interval == 30
    assert isinstance(interval, int), f"Expected int, got {type(interval)}: {interval!r}"


def _signature_without_defaults(testkey):
    """Stand-in for a client whose parameter declares no default at all."""


def _signature_with_float_default(myparam=1.0):
    """Stand-in for a client whose parameter declares a float default."""


def test_coerce_redis_kwargs_types_empty_default_param_unchanged():
    """String params whose signature entry has no default (inspect.Parameter.empty) are left as-is."""
    result = _coerce_redis_kwargs_types({"testkey": "some_value"}, client=_signature_without_defaults)

    assert result["testkey"] == "some_value"
    assert isinstance(result["testkey"], str)


def test_coerce_redis_kwargs_types_float_valid():
    """String values for params whose signature default is a float are coerced to float."""
    result = _coerce_redis_kwargs_types({"myparam": "3.14"}, client=_signature_with_float_default)

    assert result["myparam"] == pytest.approx(3.14)
    assert isinstance(result["myparam"], float)


def test_coerce_redis_kwargs_types_float_invalid_drops_key():
    """An unconvertible string for a float-default param is dropped from the result."""
    result = _coerce_redis_kwargs_types({"myparam": "not_a_float"}, client=_signature_with_float_default)

    assert "myparam" not in result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("false", False), ("true", True), ("0", False), ("1", True)],
)
def test_coerce_socket_keepalive_string(raw, expected):
    """socket_keepalive's signature default is None, so it needs an explicit bool
    coercion: a leftover "false" string is truthy and enables keepalive."""
    result = _coerce_redis_kwargs_types({"socket_keepalive": raw})

    assert result["socket_keepalive"] is expected


def test_get_redis_client_logic_coerces_cluster_only_kwargs(monkeypatch):
    """Cluster-only kwargs (absent from redis.Redis's signature) must still be
    coerced when routing to a cluster, or Helm-stringified values reach
    RedisCluster as strings."""
    for envvar in (*_get_redis_env_kwarg_mapping(), "REDIS_CLUSTER_NODES", "REDIS_SENTINEL_NODES"):
        monkeypatch.delenv(envvar, raising=False)

    result = _get_redis_client_logic(
        startup_nodes='[{"host": "localhost", "port": 7000}]',
        cluster_error_retry_attempts="5",
        require_full_coverage="false",
        health_check_interval="30",
    )

    assert result["cluster_error_retry_attempts"] == 5
    assert isinstance(result["cluster_error_retry_attempts"], int)
    assert result["require_full_coverage"] is False
    assert result["health_check_interval"] == 30
    assert isinstance(result["health_check_interval"], int)


def test_get_redis_client_logic_raises_without_host_or_url(monkeypatch):
    """_get_redis_client_logic raises ValueError when neither host nor url is provided."""
    for envvar in (*_get_redis_env_kwarg_mapping(), "REDIS_CLUSTER_NODES", "REDIS_SENTINEL_NODES"):
        monkeypatch.delenv(envvar, raising=False)

    with pytest.raises(ValueError, match="Either 'host' or 'url' must be specified for redis"):
        _get_redis_client_logic()
