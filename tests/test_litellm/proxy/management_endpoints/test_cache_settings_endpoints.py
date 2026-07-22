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
    _REDACTED_VALUE,
    CacheSettingsManager,
    CacheSettingsUpdateRequest,
    CacheTestRequest,
    _merge_over_saved,
    _overlay_environment,
    _parse_stored_settings,
    _redact_credentials,
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


class TestParseStoredSettings:
    """The stored blob arrives as a JSON string or a parsed dict; both must
    normalize to a dict so the secret-preservation read never silently drops it."""

    def test_parses_a_json_string(self):
        assert _parse_stored_settings('{"host": "h", "password": "pw"}') == {"host": "h", "password": "pw"}

    def test_passes_a_dict_through(self):
        assert _parse_stored_settings({"host": "h", "password": "pw"}) == {"host": "h", "password": "pw"}

    def test_non_mapping_becomes_empty(self):
        assert _parse_stored_settings(None) == {}
        assert _parse_stored_settings("[1, 2]") == {}


class TestMergeOverSaved:
    """The secret-preservation contract behind the redacted-resubmit fix."""

    def test_redacted_secret_restores_stored_value(self):
        # same connection target, an unrelated field edited: the stored secret
        # is restored behind the redacted resubmit
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "samehost", "namespace": "new", "password": _REDACTED_VALUE},
            saved={"type": "redis", "host": "samehost", "password": "realpw"},
        )
        assert merged["namespace"] == "new"
        assert merged["password"] == "realpw"

    def test_stored_secret_not_replayed_to_a_different_target(self):
        # credential replay guard: omitting the password while pointing at a new
        # host must NOT resurrect the stored secret (it would be sent elsewhere)
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "attacker.example.com", "password": _REDACTED_VALUE},
            saved={"type": "redis", "host": "real-redis", "password": "realpw"},
        )
        assert "password" not in merged

    def test_omitted_secret_restores_stored_value(self):
        # same host (target unchanged), password field omitted entirely
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "samehost", "namespace": "n"},
            saved={"type": "redis", "host": "samehost", "password": "realpw"},
        )
        assert merged["password"] == "realpw"

    def test_sentinel_password_not_replayed_to_different_sentinel_nodes(self):
        # sentinel target change with an omitted sentinel_password must not
        # resurrect the stored one and send it to the caller's sentinels
        merged = _merge_over_saved(
            incoming={"type": "redis", "sentinel_nodes": [["attacker", 26379]], "service_name": "mymaster"},
            saved={
                "type": "redis",
                "sentinel_nodes": [["real", 26379]],
                "service_name": "mymaster",
                "sentinel_password": "realsp",
            },
        )
        assert "sentinel_password" not in merged

    def test_sentinel_password_preserved_when_sentinel_target_unchanged(self):
        merged = _merge_over_saved(
            incoming={"type": "redis", "sentinel_nodes": [["real", 26379]], "service_name": "mymaster"},
            saved={
                "type": "redis",
                "sentinel_nodes": [["real", 26379]],
                "service_name": "mymaster",
                "sentinel_password": "realsp",
            },
        )
        assert merged["sentinel_password"] == "realsp"

    def test_password_not_replayed_to_different_cluster_nodes(self):
        merged = _merge_over_saved(
            incoming={"type": "redis", "redis_startup_nodes": [{"host": "attacker", "port": "7001"}]},
            saved={
                "type": "redis",
                "redis_startup_nodes": [{"host": "real", "port": "7001"}],
                "password": "realpw",
            },
        )
        assert "password" not in merged

    def test_equivalent_target_representations_still_preserve_secret(self):
        # the client sends port as a string, storage holds it as an int: the
        # target is unchanged, so the untouched password must not be dropped
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "h", "port": "6379", "password": _REDACTED_VALUE},
            saved={"type": "redis", "host": "h", "port": 6379, "password": "realpw"},
        )
        assert merged["password"] == "realpw"

    def test_explicit_empty_string_clears_the_secret(self):
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "h", "password": ""},
            saved={"type": "redis", "host": "h", "password": "realpw"},
        )
        assert merged.get("password") == ""

    def test_explicit_null_clears_the_secret(self):
        # an explicit null is a clear, not an omission, so it must not restore
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "h", "password": None},
            saved={"type": "redis", "host": "h", "password": "realpw"},
        )
        assert merged.get("password") is None

    def test_secret_not_reused_when_a_pinned_target_field_is_omitted(self):
        # omitting the host (a pinned target) means the request does not describe
        # the stored target, so the stored secret must not be restored (and thus
        # cannot be sent to whatever host the incomplete request resolves to)
        merged = _merge_over_saved(
            incoming={"type": "redis", "port": "6379"},
            saved={"type": "redis", "host": "real", "port": 6379, "password": "realpw"},
        )
        assert "password" not in merged

    def test_redacted_secret_with_no_stored_value_is_dropped(self):
        # env-sourced secret: nothing stored to restore, so the marker must not
        # be persisted; the environment stays the source at runtime
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "h", "password": _REDACTED_VALUE},
            saved={},
        )
        assert "password" not in merged

    def test_new_secret_value_wins(self):
        merged = _merge_over_saved(
            incoming={"password": "brandnewpw"},
            saved={"password": "realpw"},
        )
        assert merged["password"] == "brandnewpw"

    def test_switching_from_url_to_host_port_drops_stored_url(self):
        # admin migrates a url-mode cache to discrete host/port: the stored url
        # must not be resurrected (url precedence would then discard host/port)
        merged = _merge_over_saved(
            incoming={"type": "redis", "host": "newhost", "port": "6379"},
            saved={"type": "redis", "url": "redis://:pw@oldhost:6379/0"},
        )
        assert "url" not in merged
        assert merged["host"] == "newhost"
        assert merged["port"] == "6379"

    def test_untouched_url_is_preserved_without_a_discrete_target(self):
        # a url-mode save that touches nothing keeps the stored url
        merged = _merge_over_saved(
            incoming={"type": "redis", "namespace": "ns"},
            saved={"type": "redis", "url": "redis://:pw@host:6379/0"},
        )
        assert merged["url"] == "redis://:pw@host:6379/0"


def test_overlay_environment_fills_unset_connection_fields(monkeypatch):
    """A cache with no stored connection resolves REDIS_* env for the UI."""
    for var in ("REDIS_URL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "env-password")

    effective = _overlay_environment({})

    assert effective["host"] == "redis.internal"
    assert effective["port"] == "6380"
    assert effective["password"] == "env-password"
    assert effective["type"] == "redis"


def test_overlay_environment_stored_value_wins(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "env-host")
    effective = _overlay_environment({"type": "redis", "host": "stored-host"})
    assert effective["host"] == "stored-host"


@pytest.mark.asyncio
async def test_get_cache_settings_falls_back_to_redis_env(monkeypatch):
    """A cache configured purely through REDIS_* env vars shows its effective
    connection instead of a blank page, with the password redacted."""
    for var in ("REDIS_URL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "env-password")

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=None)

    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
    ):
        response = await get_cache_settings(user_api_key_dict=_admin_auth())

    values = response.current_values
    assert values["host"] == "redis.internal"
    assert values["port"] == "6380"
    assert values["type"] == "redis"
    # the env password is surfaced as configured, not leaked in plaintext
    assert values["password"] == _REDACTED_VALUE


@pytest.mark.asyncio
async def test_get_cache_settings_redacts_password_with_marker(monkeypatch):
    for var in ("REDIS_URL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    cache_row = MagicMock()
    cache_row.cache_settings = json.dumps(
        {"type": "redis", "host": "h", "password": "supersecret", "namespace": "ns"}
    )
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=cache_row)
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
    ):
        response = await get_cache_settings(user_api_key_dict=_admin_auth())

    assert response.current_values["password"] == _REDACTED_VALUE
    assert response.current_values["namespace"] == "ns"


@pytest.mark.asyncio
async def test_get_cache_settings_url_mode_hides_env_discrete_fields(monkeypatch):
    """A url-mode stored config must not surface env-overlaid host/port.

    Otherwise a no-op save would submit the env host and, via url precedence,
    silently switch the cache off its configured url.
    """
    for var in ("REDIS_URL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REDIS_HOST", "env-host")
    monkeypatch.setenv("REDIS_PORT", "6380")

    cache_row = MagicMock()
    cache_row.cache_settings = {"type": "redis", "url": "redis://:pw@stored-host:6379/0"}
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=cache_row)
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
    ):
        response = await get_cache_settings(user_api_key_dict=_admin_auth())

    values = response.current_values
    assert values["url"] == _REDACTED_VALUE
    # the env host/port must not leak in and shadow the url
    assert "host" not in values
    assert "port" not in values


def _mock_proxy_config_identity_crypto():
    proxy_config = MagicMock()
    proxy_config._encrypt_env_variables = MagicMock(
        side_effect=lambda environment_variables: dict(environment_variables)
    )
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))
    proxy_config._init_cache = MagicMock()
    proxy_config.switch_on_llm_response_caching = MagicMock()
    return proxy_config


@pytest.mark.asyncio
async def test_update_preserves_stored_password_on_redacted_resubmit(monkeypatch):
    """Editing an unrelated field and re-submitting the redacted password must
    keep the stored secret, not persist the marker over a working password."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)

    existing = MagicMock()
    # prisma returns the Json column as an already-parsed dict, not a JSON
    # string; a reader that json.loads unconditionally would drop the whole row
    existing.cache_settings = {"type": "redis", "host": "oldhost", "password": "realpw"}
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=existing)
    mock_prisma.db.litellm_cacheconfig.upsert = AsyncMock()
    proxy_config = _mock_proxy_config_identity_crypto()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        result = await update_cache_settings(
            request=CacheSettingsUpdateRequest(
                # same host (the target is unchanged), an unrelated field edited
                cache_settings={"type": "redis", "host": "oldhost", "namespace": "edited", "password": _REDACTED_VALUE}
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    persisted = proxy_config._encrypt_env_variables.call_args.kwargs["environment_variables"]
    assert persisted["host"] == "oldhost"
    assert persisted["namespace"] == "edited"
    assert persisted["password"] == "realpw"
    # the response never echoes the plaintext secret back either
    assert result["settings"]["password"] == _REDACTED_VALUE


@pytest.mark.asyncio
async def test_update_drops_env_sourced_redacted_secret(monkeypatch):
    """With no stored row, a re-submitted redacted secret is env-sourced; the
    marker must not be persisted so the environment stays the source."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_cacheconfig.upsert = AsyncMock()
    proxy_config = _mock_proxy_config_identity_crypto()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        await update_cache_settings(
            request=CacheSettingsUpdateRequest(
                cache_settings={"type": "redis", "host": "h", "password": _REDACTED_VALUE}
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    persisted = proxy_config._encrypt_env_variables.call_args.kwargs["environment_variables"]
    assert "password" not in persisted


@pytest.mark.asyncio
async def test_update_applies_new_password(monkeypatch):
    """A real new secret value replaces the stored one."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)

    existing = MagicMock()
    existing.cache_settings = json.dumps({"type": "redis", "host": "h", "password": "oldpw"})
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=existing)
    mock_prisma.db.litellm_cacheconfig.upsert = AsyncMock()
    proxy_config = _mock_proxy_config_identity_crypto()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        await update_cache_settings(
            request=CacheSettingsUpdateRequest(
                cache_settings={"type": "redis", "host": "h", "password": "brandnewpw"}
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    persisted = proxy_config._encrypt_env_variables.call_args.kwargs["environment_variables"]
    assert persisted["password"] == "brandnewpw"


@pytest.mark.asyncio
async def test_test_cache_connection_survives_saved_lookup_failure(monkeypatch):
    """A failed saved-settings lookup must not block the connection test.

    The test endpoint reads the stored row to resolve a redacted credential, but
    that read can raise (a misconfigured or unavailable client), and it must fall
    back to the submitted settings rather than abort — otherwise a shared client
    left in an odd state by another test would break every connection test.
    """
    monkeypatch.setattr(litellm, "store_audit_logs", False)

    # a client whose find_unique is not awaitable, so the saved read raises
    bad_prisma = MagicMock()
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    cache_instance = MagicMock()
    cache_instance.cache = MagicMock()
    cache_instance.cache.test_connection = AsyncMock(return_value={"status": "success", "message": "ok"})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", bad_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.Cache") as mock_cache_class,
    ):
        mock_cache_class.return_value = cache_instance
        result = await test_cache_connection(
            request=CacheTestRequest(cache_settings={"type": "redis", "host": "h", "port": "6379", "password": "pw"}),
            user_api_key_dict=_admin_auth(),
        )

    mock_cache_class.assert_called_once()
    assert result.status == "success"


@pytest.mark.asyncio
async def test_get_cache_settings_does_not_surface_non_display_env_credentials(monkeypatch):
    """The env overlay must not leak credential kwargs the UI does not manage.

    _redis_kwargs_from_environment resolves every redis.Redis kwarg, including
    secrets like azure_client_secret; only cache display fields may be surfaced,
    so a non-admin reading /cache/settings never retrieves such a credential.
    """
    for var in ("REDIS_URL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_AZURE_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_AZURE_CLIENT_SECRET", "super-azure-secret")

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=None)
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
    ):
        response = await get_cache_settings(user_api_key_dict=_admin_auth())

    values = response.current_values
    assert values.get("host") == "redis.internal"
    # the non-display credential must not appear in the response at all
    assert "azure_client_secret" not in values
    assert "super-azure-secret" not in values.values()


@pytest.mark.asyncio
async def test_test_cache_connection_does_not_log_plaintext_credentials(monkeypatch, caplog):
    """The connection test must not write the resolved plaintext secret to logs.

    _merge_over_saved substitutes the stored password for a redacted resubmit, so
    the settings dict carries the real secret; the debug log must redact it.
    """
    import logging

    existing = MagicMock()
    existing.cache_settings = {"type": "redis", "host": "h", "port": "6379", "password": "realredispw"}
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=existing)
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    cache_instance = MagicMock()
    cache_instance.cache = MagicMock()
    cache_instance.cache.test_connection = AsyncMock(return_value={"status": "success", "message": "ok"})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.Cache") as mock_cache_class,
        caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"),
    ):
        mock_cache_class.return_value = cache_instance
        # resubmit the redacted marker; the merge resolves it to the stored secret
        await test_cache_connection(
            request=CacheTestRequest(
                cache_settings={"type": "redis", "host": "h", "port": "6379", "password": _REDACTED_VALUE}
            ),
            user_api_key_dict=_admin_auth(),
        )

    # the real password was used to build the client but never written to the log
    assert mock_cache_class.call_args.kwargs["password"] == "realredispw"
    assert "realredispw" not in caplog.text


@pytest.mark.asyncio
async def test_test_cache_connection_does_not_replay_saved_password_to_new_host(monkeypatch):
    """Credential-replay guard on the connection test.

    A caller that submits a different host while omitting the password must not
    have the stored password restored and sent to the caller-chosen host.
    """
    existing = MagicMock()
    existing.cache_settings = {"type": "redis", "host": "real-redis", "port": "6379", "password": "realredispw"}
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=existing)
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables = MagicMock(side_effect=lambda variables_dict: dict(variables_dict))

    cache_instance = MagicMock()
    cache_instance.cache = MagicMock()
    cache_instance.cache.test_connection = AsyncMock(return_value={"status": "success", "message": "ok"})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
        patch("litellm.Cache") as mock_cache_class,
    ):
        mock_cache_class.return_value = cache_instance
        await test_cache_connection(
            request=CacheTestRequest(
                cache_settings={"type": "redis", "host": "attacker.example.com", "port": "6379"}
            ),
            user_api_key_dict=_admin_auth(),
        )

    called_kwargs = mock_cache_class.call_args.kwargs
    # the stored password is NOT sent to the attacker-chosen host
    assert called_kwargs.get("password") != "realredispw"
    assert "password" not in called_kwargs
