"""
Unit tests for cache settings management endpoints
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.proxy._types import LitellmTableNames, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.cache_settings_endpoints import (
    _CACHE_SENSITIVE_FIELDS,
    CacheSettingsManager,
    CacheSettingsUpdateRequest,
    CacheTestRequest,
    _resolve_cache_url_precedence,
    get_cache_settings,
    test_cache_connection,
    update_cache_settings,
)
from litellm.types.management_endpoints.cache_settings_endpoints import (
    CACHE_SETTINGS_FIELDS,
)


@pytest.mark.asyncio
async def test_test_cache_connection_calls_cache_test_connection_with_params():
    """
    Test that test_cache_connection endpoint calls cache.test_connection()
    with the correct parameters from the request body.
    """
    # Mock cache settings from request
    cache_settings = {
        "type": "redis",
        "host": "test-redis-host",
        "port": "6379",
        "password": "test-password",
    }

    request = CacheTestRequest(cache_settings=cache_settings)
    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test", user_id="test-user")

    # Mock Cache class and its test_connection method
    mock_cache_instance = MagicMock()
    mock_cache_instance.cache = MagicMock()
    mock_cache_instance.cache.test_connection = AsyncMock(
        return_value={
            "status": "success",
            "message": "Redis connection test successful",
        }
    )

    # Patch Cache class at the import location (litellm module)
    with patch("litellm.Cache") as mock_cache_class:
        mock_cache_class.return_value = mock_cache_instance

        # Call the endpoint
        result = await test_cache_connection(request=request, user_api_key_dict=user_api_key_dict)

        # Verify Cache was instantiated with correct params
        mock_cache_class.assert_called_once_with(**cache_settings)

        # Verify test_connection was called on the cache instance
        mock_cache_instance.cache.test_connection.assert_called_once()

        # Verify response
        assert result.status == "success"
        assert result.message == "Redis connection test successful"
        assert result.error is None


def test_cache_settings_fields_expose_url_and_db():
    """The dynamic UI form is driven by CACHE_SETTINGS_FIELDS; url + db must be
    present (with the right types) so the Redis URL and logical database index
    are configurable from the Admin UI."""
    by_name = {f.field_name: f for f in CACHE_SETTINGS_FIELDS}

    assert "url" in by_name
    assert "db" in by_name
    # db is a logical database index → integer
    assert by_name["db"].field_type == "Integer"
    # Both are common connection fields, shown for every Redis type
    assert by_name["url"].redis_type is None
    assert by_name["db"].redis_type is None


class TestResolveCacheUrlPrecedence:
    """url wins over the discrete host/port/db/password fields."""

    def test_url_overrides_discrete_connection_fields(self):
        settings = {
            "type": "redis",
            "url": "redis://user:pw@host:6379/1",
            "host": "host",
            "port": "6379",
            "db": 1,
            "username": "user",
            "password": "pw",
            "namespace": "ns",
            "ttl": 60,
        }

        result = _resolve_cache_url_precedence(settings)

        assert result["url"] == "redis://user:pw@host:6379/1"
        assert "host" not in result
        assert "port" not in result
        assert "db" not in result
        # username and password are both encodable in the url, so the discrete
        # copies must not ride along and override it
        assert "username" not in result
        assert "password" not in result
        # Non-connection fields survive
        assert result["type"] == "redis"
        assert result["namespace"] == "ns"
        assert result["ttl"] == 60

    def test_no_url_returns_copy_unchanged(self):
        settings = {"type": "redis", "host": "host", "port": "6379", "db": 1}

        result = _resolve_cache_url_precedence(settings)

        assert result == settings
        assert result is not settings

    def test_blank_url_does_not_strip_discrete_fields(self):
        settings = {"type": "redis", "url": "   ", "host": "host", "db": 2}

        result = _resolve_cache_url_precedence(settings)

        assert result["host"] == "host"
        assert result["db"] == 2

    def test_cluster_mode_keeps_discrete_fields(self):
        settings = {
            "type": "redis",
            "url": "redis://host:6379",
            "redis_startup_nodes": [{"host": "127.0.0.1", "port": "7001"}],
            "host": "host",
            "password": "pw",
        }

        result = _resolve_cache_url_precedence(settings)

        assert result["host"] == "host"
        assert result["password"] == "pw"


@pytest.mark.asyncio
async def test_test_cache_connection_url_takes_precedence_over_discrete_fields():
    """When url + discrete fields are both sent, the tested Cache instance is
    built from the url alone (host/port/db/password dropped)."""
    cache_settings = {
        "type": "redis",
        "url": "redis://:pw@host:6379/1",
        "host": "ignored-host",
        "port": "6379",
        "db": 1,
        "password": "pw",
    }

    request = CacheTestRequest(cache_settings=cache_settings)
    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test", user_id="test-user")

    mock_cache_instance = MagicMock()
    mock_cache_instance.cache = MagicMock()
    mock_cache_instance.cache.test_connection = AsyncMock(return_value={"status": "success", "message": "ok"})

    with patch("litellm.Cache") as mock_cache_class:
        mock_cache_class.return_value = mock_cache_instance

        result = await test_cache_connection(request=request, user_api_key_dict=user_api_key_dict)

        called_kwargs = mock_cache_class.call_args.kwargs
        assert called_kwargs["url"] == "redis://:pw@host:6379/1"
        assert "host" not in called_kwargs
        assert "port" not in called_kwargs
        assert "db" not in called_kwargs
        assert "password" not in called_kwargs
        assert result.status == "success"


@pytest.mark.asyncio
async def test_update_cache_settings_persists_url_precedence(monkeypatch):
    """The persisted (source-of-truth) row and the reinitialized cache both use
    the url-resolved settings, so a stored config never carries a contradictory
    host+url pair."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_cacheconfig.upsert = AsyncMock()

    proxy_config = MagicMock()
    proxy_config._encrypt_env_variables = MagicMock(
        side_effect=lambda environment_variables: dict(environment_variables)
    )
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))
    proxy_config._init_cache = MagicMock()
    proxy_config.switch_on_llm_response_caching = MagicMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        await update_cache_settings(
            request=CacheSettingsUpdateRequest(
                cache_settings={
                    "type": "redis",
                    "url": "redis://:pw@host:6379/1",
                    "host": "ignored-host",
                    "port": "6379",
                    "db": 1,
                    "password": "pw",
                    "namespace": "ns",
                }
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    persisted = proxy_config._encrypt_env_variables.call_args.kwargs["environment_variables"]
    assert persisted["url"] == "redis://:pw@host:6379/1"
    assert persisted["namespace"] == "ns"
    assert "host" not in persisted
    assert "port" not in persisted
    assert "db" not in persisted
    assert "password" not in persisted

    init_params = proxy_config._init_cache.call_args.kwargs["cache_params"]
    assert "host" not in init_params
    assert init_params["url"] == "redis://:pw@host:6379/1"


def test_url_is_a_masked_field():
    """A Redis URL can carry an inline password, so it must be masked on read
    alongside the discrete password fields."""
    assert "url" in _CACHE_SENSITIVE_FIELDS


@pytest.mark.asyncio
async def test_get_cache_settings_masks_password_bearing_url():
    """GET /cache/settings must not leak an inline url password in plaintext,
    while non-credential fields (e.g. namespace) come back untouched."""
    stored_url = "redis://:supersecretpassword@host:6379/1"
    stored_settings = {"type": "redis", "url": stored_url, "namespace": "ns"}

    cache_row = MagicMock()
    cache_row.cache_settings = json.dumps(stored_settings)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=cache_row)

    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
    ):
        response = await get_cache_settings(user_api_key_dict=_admin_auth())

    returned_url = response.current_values["url"]
    assert returned_url != stored_url
    assert "supersecretpassword" not in returned_url
    # non-credential field is not masked
    assert response.current_values["namespace"] == "ns"


class TestCacheSettingsManager:
    """Tests for CacheSettingsManager class"""

    def test_cache_params_equal_identical_params(self):
        """
        Test that _cache_params_equal returns True for identical params.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
            "password": "test-password",
        }
        params2 = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
            "password": "test-password",
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is True

    def test_cache_params_equal_different_params(self):
        """
        Test that _cache_params_equal returns False for different params.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
        }
        params2 = {
            "type": "redis",
            "host": "different-host",
            "port": "6379",
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is False

    def test_cache_params_equal_filters_redis_type(self):
        """
        Test that _cache_params_equal filters out redis_type field.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "redis_type": "node",  # Should be ignored
        }
        params2 = {
            "type": "redis",
            "host": "localhost",
            "redis_type": "cluster",  # Different value, but should be ignored
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is True

    def test_cache_params_equal_filters_none_values(self):
        """
        Test that _cache_params_equal filters out None values.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "port": None,
        }
        params2 = {
            "type": "redis",
            "host": "localhost",
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is True

    def test_update_cache_params(self):
        """
        Test that update_cache_params updates the tracked cache params.
        """
        # Reset the class variable
        CacheSettingsManager._last_cache_params = None

        cache_params = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
        }

        CacheSettingsManager.update_cache_params(cache_params)

        assert CacheSettingsManager._last_cache_params == cache_params
        # Verify it's a copy, not a reference
        assert CacheSettingsManager._last_cache_params is not cache_params

    @pytest.mark.asyncio
    async def test_init_cache_settings_in_db_initializes_when_params_changed(self):
        """
        Test that init_cache_settings_in_db initializes cache when params change.
        """
        # Reset the class variable
        CacheSettingsManager._last_cache_params = None

        # Mock prisma client
        mock_prisma_client = MagicMock()
        mock_cache_config = MagicMock()
        mock_cache_config.cache_settings = '{"type": "redis", "host": "localhost", "port": "6379"}'
        mock_prisma_client.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=mock_cache_config)

        # Mock proxy_config
        mock_proxy_config = MagicMock()
        mock_proxy_config._decrypt_db_variables = MagicMock(
            return_value={
                "type": "redis",
                "host": "localhost",
                "port": "6379",
            }
        )
        mock_proxy_config._init_cache = MagicMock()
        mock_proxy_config.switch_on_llm_response_caching = MagicMock()

        # Call the method
        await CacheSettingsManager.init_cache_settings_in_db(
            prisma_client=mock_prisma_client, proxy_config=mock_proxy_config
        )

        # Verify cache was initialized
        mock_proxy_config._init_cache.assert_called_once()
        mock_proxy_config.switch_on_llm_response_caching.assert_called_once()

        # Verify params were stored
        assert CacheSettingsManager._last_cache_params is not None
        assert "type" in CacheSettingsManager._last_cache_params
        assert "redis_type" not in CacheSettingsManager._last_cache_params

    @pytest.mark.asyncio
    async def test_init_cache_settings_in_db_skips_when_params_unchanged(self):
        """
        Test that init_cache_settings_in_db skips initialization when params unchanged.
        """
        # Set existing params
        existing_params = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
        }
        CacheSettingsManager._last_cache_params = existing_params.copy()

        # Mock prisma client
        mock_prisma_client = MagicMock()
        mock_cache_config = MagicMock()
        mock_cache_config.cache_settings = '{"type": "redis", "host": "localhost", "port": "6379"}'
        mock_prisma_client.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=mock_cache_config)

        # Mock proxy_config
        mock_proxy_config = MagicMock()
        mock_proxy_config._decrypt_db_variables = MagicMock(
            return_value={
                "type": "redis",
                "host": "localhost",
                "port": "6379",
            }
        )
        mock_proxy_config._init_cache = MagicMock()
        mock_proxy_config.switch_on_llm_response_caching = MagicMock()

        # Call the method
        await CacheSettingsManager.init_cache_settings_in_db(
            prisma_client=mock_prisma_client, proxy_config=mock_proxy_config
        )

        # Verify cache was NOT initialized (params unchanged)
        mock_proxy_config._init_cache.assert_not_called()
        mock_proxy_config.switch_on_llm_response_caching.assert_not_called()

    @pytest.mark.asyncio
    async def test_init_cache_settings_in_db_retries_on_transport_error(self):
        """`CacheSettingsManager.init_cache_settings_in_db` self-heals across one
        ClientNotConnectedError via call_with_db_reconnect_retry."""
        import prisma

        invocations: list = []

        async def _flaky_find_unique(**kwargs):
            invocations.append(None)
            if len(invocations) == 1:
                raise prisma.errors.ClientNotConnectedError()
            return None  # No config → function returns early after retry.

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_cacheconfig.find_unique = AsyncMock(side_effect=_flaky_find_unique)
        mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
        mock_prisma_client._db_auth_reconnect_timeout_seconds = 2.0
        mock_prisma_client._db_auth_reconnect_lock_timeout_seconds = 0.1
        mock_proxy_config = MagicMock()

        await CacheSettingsManager.init_cache_settings_in_db(
            prisma_client=mock_prisma_client, proxy_config=mock_proxy_config
        )

        assert len(invocations) == 2
        mock_prisma_client.attempt_db_reconnect.assert_awaited_once()
        reconnect_kwargs = mock_prisma_client.attempt_db_reconnect.await_args.kwargs
        assert reconnect_kwargs["reason"] == "init_cache_settings_in_db_lookup_failure"


# ── Audit-log emission for /cache/settings ────────────────────────────────────


def _admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


@pytest.mark.asyncio
async def test_update_cache_settings_emits_audit_log_when_enabled(monkeypatch):
    """Cache config carries Redis credentials; mutation must emit an
    audit-log row when ``store_audit_logs`` is on, with values redacted."""
    monkeypatch.setattr(litellm, "store_audit_logs", True)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_cacheconfig.upsert = AsyncMock()

    proxy_config = MagicMock()
    proxy_config._encrypt_env_variables = MagicMock(
        side_effect=lambda environment_variables: dict(environment_variables)
    )
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))
    proxy_config._init_cache = MagicMock()
    proxy_config.switch_on_llm_response_caching = MagicMock()

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_config",
            proxy_config,
        ),
        patch(
            "litellm.proxy.proxy_server.store_model_in_db",
            True,
        ),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await update_cache_settings(
            request=CacheSettingsUpdateRequest(
                cache_settings={
                    "type": "redis",
                    "host": "redis.example.com",
                    "password": "super-secret-redis-pw",
                }
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        # asyncio.create_task fires the coroutine eagerly; await one tick to let
        # the audit-log emit run before the test exits.
        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert log.table_name == LitellmTableNames.CACHE_CONFIG_TABLE_NAME
    assert log.object_id == "cache_config"
    assert log.action == "created"  # no existing row → create

    after = json.loads(log.updated_values)
    # Field names are preserved so an auditor can see what changed.
    assert set(after["settings"].keys()) == {"type", "host", "password"}
    # Plaintext values must NOT appear in the serialized row.
    assert "super-secret-redis-pw" not in log.updated_values
    assert "redis.example.com" not in log.updated_values


@pytest.mark.asyncio
async def test_update_cache_settings_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_cacheconfig.upsert = AsyncMock()

    proxy_config = MagicMock()
    proxy_config._encrypt_env_variables = MagicMock(
        side_effect=lambda environment_variables: dict(environment_variables)
    )
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))
    proxy_config._init_cache = MagicMock()
    proxy_config.switch_on_llm_response_caching = MagicMock()

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_config",
            proxy_config,
        ),
        patch(
            "litellm.proxy.proxy_server.store_model_in_db",
            True,
        ),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await update_cache_settings(
            request=CacheSettingsUpdateRequest(cache_settings={"type": "redis", "host": "redis.example.com"}),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    assert audit_calls == []
