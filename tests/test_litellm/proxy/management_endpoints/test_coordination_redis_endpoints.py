"""
Unit tests for coordination Redis settings management endpoints
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.caching.caching import RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.proxy._types import LitellmTableNames, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.coordination_redis_endpoints import (
    _REDACTED_VALUE,
    CoordinationRedisSettingsRequest,
    get_coordination_redis_settings,
    check_coordination_redis_connection,
    update_coordination_redis_settings,
)
from litellm.types.management_endpoints.coordination_redis_endpoints import (
    COORDINATION_REDIS_SETTINGS_FIELDS,
)

_SAVED_SETTINGS = {
    "host": "coord-redis.example.com",
    "port": 6379,
    "password": "super-secret-redis-pw",
    "url": "redis://:super-secret-redis-pw@coord-redis.example.com:6379",
    "sentinel_password": "super-secret-sentinel-pw",
}


def _admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _prisma_with_general_settings(general_settings: dict | None) -> MagicMock:
    """A prisma client whose LiteLLM_Config `general_settings` row holds ``general_settings``."""
    row = None
    if general_settings is not None:
        row = MagicMock()
        row.param_value = json.dumps(general_settings)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=row)
    mock_prisma.db.litellm_config.upsert = AsyncMock()
    return mock_prisma


def _proxy_config(file_general_settings: dict | None = None) -> MagicMock:
    proxy_config = MagicMock()
    proxy_config.get_config_state = MagicMock(
        return_value={"general_settings": file_general_settings or {}},
    )
    return proxy_config


# ── GET /coordination_redis/settings ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_redacts_every_credential_field():
    """password, sentinel_password and the (password-bearing) url never leave the
    server in plaintext; non-credential fields come back untouched."""
    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            _prisma_with_general_settings({"coordination_redis": _SAVED_SETTINGS}),
        ),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    serialized = json.dumps(response.model_dump())
    assert "super-secret-redis-pw" not in serialized
    assert "super-secret-sentinel-pw" not in serialized

    assert response.values["password"] == _REDACTED_VALUE
    assert response.values["sentinel_password"] == _REDACTED_VALUE
    assert response.values["url"] == _REDACTED_VALUE
    assert response.values["host"] == "coord-redis.example.com"
    assert response.values["port"] == 6379

    # field metadata is hydrated with the same redacted values
    by_name = {field.field_name: field for field in response.fields}
    assert by_name["password"].field_value == _REDACTED_VALUE
    assert by_name["host"].field_value == "coord-redis.example.com"


@pytest.mark.asyncio
async def test_get_source_is_coordination_redis_when_block_present(monkeypatch):
    """An explicit block wins even when a Redis cache backend and REDIS_* env both exist."""
    monkeypatch.setattr(litellm, "cache", MagicMock(cache=MagicMock(spec=RedisCache)))
    monkeypatch.setenv("REDIS_HOST", "env-redis")

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            _prisma_with_general_settings({"coordination_redis": _SAVED_SETTINGS}),
        ),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    assert response.source == "coordination_redis"


@pytest.mark.asyncio
async def test_get_source_reads_block_from_yaml_config_when_db_row_absent(monkeypatch):
    """A block set in config.yaml (not the DB) still reports source=coordination_redis."""
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.delenv("REDIS_HOST", raising=False)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings(None)),
        patch(
            "litellm.proxy.proxy_server.proxy_config",
            _proxy_config({"coordination_redis": {"host": "yaml-redis"}}),
        ),
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    assert response.source == "coordination_redis"
    assert response.values["host"] == "yaml-redis"


@pytest.mark.parametrize("cache_backend_cls", [RedisCache, RedisClusterCache])
@pytest.mark.asyncio
async def test_get_source_is_cache_backend_when_no_block(monkeypatch, cache_backend_cls):
    """With no explicit block, a plain-Redis response-cache backend is borrowed —
    which beats the REDIS_* env fallback."""
    monkeypatch.setattr(litellm, "cache", MagicMock(cache=MagicMock(spec=cache_backend_cls)))
    monkeypatch.setenv("REDIS_HOST", "env-redis")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    assert response.source == "cache_backend"
    assert response.values == {}


@pytest.mark.asyncio
async def test_get_source_is_environment_when_no_block_and_non_redis_cache(monkeypatch):
    """A non-Redis cache backend falls through to the REDIS_* env fallback."""
    monkeypatch.setattr(litellm, "cache", MagicMock(cache=MagicMock()))
    monkeypatch.setenv("REDIS_HOST", "env-redis")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    assert response.source == "environment"


@pytest.mark.asyncio
async def test_get_source_is_none_when_nothing_configured(monkeypatch):
    monkeypatch.setattr(litellm, "cache", None)
    for env_var in ("REDIS_HOST", "REDIS_URL", "REDIS_CLUSTER_NODES", "REDIS_SENTINEL_NODES"):
        monkeypatch.delenv(env_var, raising=False)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    assert response.source is None


@pytest.mark.asyncio
async def test_get_source_does_not_build_a_client(monkeypatch):
    """The env-fallback probe is read-only: no Redis client is constructed on GET."""
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setenv("REDIS_HOST", "env-redis")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server._build_redis_usage_cache") as mock_build,
    ):
        response = await get_coordination_redis_settings(user_api_key_dict=_admin_auth())

    assert response.source == "environment"
    mock_build.assert_not_called()


@pytest.mark.asyncio
async def test_get_rejects_non_admin():
    with pytest.raises(HTTPException) as exc_info:
        await get_coordination_redis_settings(
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_role=LitellmUserRoles.INTERNAL_USER)
        )
    assert exc_info.value.status_code == 403


def test_fields_cover_every_coordination_redis_param():
    """The declarative field list drives the Admin UI form; it must stay in sync
    with the model the backend validates against."""
    from litellm.proxy._types import CoordinationRedisParams

    assert {field.field_name for field in COORDINATION_REDIS_SETTINGS_FIELDS} == set(
        CoordinationRedisParams.model_fields.keys()
    )

    by_name = {field.field_name: field for field in COORDINATION_REDIS_SETTINGS_FIELDS}
    assert by_name["startup_nodes"].section == "cluster"
    assert by_name["sentinel_nodes"].section == "sentinel"
    assert by_name["host"].section == "connection"


# ── POST /coordination_redis/settings ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_rejects_settings_without_a_connection_target(monkeypatch):
    """A block with no host/url/startup_nodes/sentinel_nodes would blow up at
    startup; reject it at write time and persist nothing."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _prisma_with_general_settings({})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_coordination_redis_settings(
                request=CoordinationRedisSettingsRequest(settings={"ssl": True, "service_name": "mymaster"}),
                user_api_key_dict=_admin_auth(),
                litellm_changed_by=None,
            )

    assert exc_info.value.status_code == 400
    mock_prisma.db.litellm_config.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_persists_into_the_general_settings_config_row(monkeypatch):
    """Settings land under `general_settings.coordination_redis` in LiteLLM_Config
    (the row startup merges over the yaml config), and sibling general_settings
    keys survive the write."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _prisma_with_general_settings({"master_key": "sk-1234"})
    invalidated: list[str] = []

    async def _capture_invalidate(param_name: str) -> None:
        invalidated.append(param_name)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.coordination_redis_endpoints.invalidate_config_param",
            new=_capture_invalidate,
        ),
    ):
        response = await update_coordination_redis_settings(
            request=CoordinationRedisSettingsRequest(
                settings={"host": "coord-redis.example.com", "port": 6379, "password": "pw"}
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    upsert_kwargs = mock_prisma.db.litellm_config.upsert.call_args.kwargs
    assert upsert_kwargs["where"] == {"param_name": "general_settings"}
    persisted = json.loads(upsert_kwargs["data"]["update"]["param_value"])
    assert persisted["coordination_redis"] == {
        "host": "coord-redis.example.com",
        "port": 6379,
        "password": "pw",
    }
    assert persisted["master_key"] == "sk-1234"
    assert invalidated == ["general_settings"]

    # the response echoes the saved settings back redacted
    assert response["settings"]["password"] == _REDACTED_VALUE
    assert response["settings"]["host"] == "coord-redis.example.com"


@pytest.mark.asyncio
async def test_update_persists_os_environ_refs_verbatim(monkeypatch):
    """`os.environ/VAR` refs are resolved only to validate; the ref itself is what
    gets stored, so the credential never lands in the DB."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    monkeypatch.setenv("MY_REDIS_HOST", "resolved-host")
    mock_prisma = _prisma_with_general_settings({})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.coordination_redis_endpoints.invalidate_config_param",
            new=AsyncMock(),
        ),
    ):
        await update_coordination_redis_settings(
            request=CoordinationRedisSettingsRequest(settings={"host": "os.environ/MY_REDIS_HOST"}),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    persisted = json.loads(mock_prisma.db.litellm_config.upsert.call_args.kwargs["data"]["update"]["param_value"])
    assert persisted["coordination_redis"] == {"host": "os.environ/MY_REDIS_HOST"}


@pytest.mark.asyncio
async def test_update_keeps_saved_credential_when_client_echoes_the_redaction_marker(monkeypatch):
    """The UI reads settings back redacted; re-submitting them must not persist
    `***REDACTED***` as the password."""
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _prisma_with_general_settings({"coordination_redis": _SAVED_SETTINGS})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.coordination_redis_endpoints.invalidate_config_param",
            new=AsyncMock(),
        ),
    ):
        await update_coordination_redis_settings(
            request=CoordinationRedisSettingsRequest(
                settings={"host": "new-host", "port": 6380, "password": _REDACTED_VALUE}
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    persisted = json.loads(mock_prisma.db.litellm_config.upsert.call_args.kwargs["data"]["update"]["param_value"])
    assert persisted["coordination_redis"]["password"] == "super-secret-redis-pw"
    assert persisted["coordination_redis"]["host"] == "new-host"


@pytest.mark.asyncio
async def test_update_emits_audit_log_with_values_redacted(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _prisma_with_general_settings({})
    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.coordination_redis_endpoints.invalidate_config_param",
            new=AsyncMock(),
        ),
        patch("litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update", new=capture),
    ):
        await update_coordination_redis_settings(
            request=CoordinationRedisSettingsRequest(
                settings={"host": "coord-redis.example.com", "password": "super-secret-redis-pw"}
            ),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert log.table_name == LitellmTableNames.CONFIG_TABLE_NAME
    assert log.object_id == "coordination_redis"
    assert log.action == "created"  # no prior block → create

    after = json.loads(log.updated_values)
    assert set(after["settings"].keys()) == {"host", "password"}
    assert "super-secret-redis-pw" not in log.updated_values
    assert "coord-redis.example.com" not in log.updated_values


@pytest.mark.asyncio
async def test_update_audit_action_is_updated_when_a_block_already_exists(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _prisma_with_general_settings({"coordination_redis": {"host": "old-host"}})
    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.coordination_redis_endpoints.invalidate_config_param",
            new=AsyncMock(),
        ),
        patch("litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update", new=capture),
    ):
        await update_coordination_redis_settings(
            request=CoordinationRedisSettingsRequest(settings={"host": "new-host"}),
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        for _ in range(3):
            await asyncio.sleep(0)

    assert audit_calls[0].action == "updated"
    assert json.loads(audit_calls[0].before_value)["settings"] == {"host": _REDACTED_VALUE}


@pytest.mark.asyncio
async def test_update_rejects_non_admin():
    with pytest.raises(HTTPException) as exc_info:
        await update_coordination_redis_settings(
            request=CoordinationRedisSettingsRequest(settings={"host": "coord-redis.example.com"}),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_role=LitellmUserRoles.INTERNAL_USER),
            litellm_changed_by=None,
        )
    assert exc_info.value.status_code == 403


# ── POST /coordination_redis/settings/test ────────────────────────────────────


@pytest.mark.asyncio
async def test_connection_test_returns_healthy_on_successful_ping():
    mock_client = MagicMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.disconnect = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server._build_redis_usage_cache", return_value=mock_client) as mock_build,
    ):
        response = await check_coordination_redis_connection(
            request=CoordinationRedisSettingsRequest(
                settings={"host": "coord-redis.example.com", "port": 6379, "password": "pw"}
            ),
            user_api_key_dict=_admin_auth(),
        )

    assert response.status == "healthy"
    assert response.error is None
    assert mock_build.call_args.args[0] == {"host": "coord-redis.example.com", "port": 6379, "password": "pw"}
    mock_client.ping.assert_awaited_once()
    mock_client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_test_reports_unhealthy_without_leaking_the_password():
    """Redis client errors echo the connection url back; the password must be
    scrubbed out of the error the admin sees."""
    mock_client = MagicMock()
    mock_client.ping = AsyncMock(
        side_effect=ConnectionError("Error connecting to redis://:super-secret-redis-pw@coord-redis.example.com:6379")
    )
    mock_client.disconnect = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server._build_redis_usage_cache", return_value=mock_client),
    ):
        response = await check_coordination_redis_connection(
            request=CoordinationRedisSettingsRequest(
                settings={
                    "host": "coord-redis.example.com",
                    "url": "redis://:super-secret-redis-pw@coord-redis.example.com:6379",
                    "password": "super-secret-redis-pw",
                }
            ),
            user_api_key_dict=_admin_auth(),
        )

    assert response.status == "unhealthy"
    assert response.error is not None
    assert "super-secret-redis-pw" not in response.error
    assert _REDACTED_VALUE in response.error
    mock_client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_test_uses_the_saved_password_for_a_redacted_field():
    """An admin re-testing settings read back from GET sends `***REDACTED***`;
    the saved credential is what actually gets dialed."""
    mock_client = MagicMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.disconnect = AsyncMock()

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            _prisma_with_general_settings({"coordination_redis": _SAVED_SETTINGS}),
        ),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server._build_redis_usage_cache", return_value=mock_client) as mock_build,
    ):
        response = await check_coordination_redis_connection(
            request=CoordinationRedisSettingsRequest(
                settings={"host": "coord-redis.example.com", "password": _REDACTED_VALUE}
            ),
            user_api_key_dict=_admin_auth(),
        )

    assert response.status == "healthy"
    assert mock_build.call_args.args[0]["password"] == "super-secret-redis-pw"


@pytest.mark.asyncio
async def test_connection_test_times_out_instead_of_hanging():
    async def _never_returns():
        await asyncio.sleep(60)

    mock_client = MagicMock()
    mock_client.ping = MagicMock(side_effect=lambda: _never_returns())
    mock_client.disconnect = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
        patch("litellm.proxy.proxy_server._build_redis_usage_cache", return_value=mock_client),
        patch(
            "litellm.proxy.management_endpoints.coordination_redis_endpoints._PING_TIMEOUT_SECONDS",
            0.01,
        ),
    ):
        response = await check_coordination_redis_connection(
            request=CoordinationRedisSettingsRequest(settings={"host": "unreachable"}),
            user_api_key_dict=_admin_auth(),
        )

    assert response.status == "unhealthy"
    assert "timed out" in (response.error or "")


@pytest.mark.asyncio
async def test_connection_test_rejects_settings_without_a_connection_target():
    with (
        patch("litellm.proxy.proxy_server.prisma_client", _prisma_with_general_settings({})),
        patch("litellm.proxy.proxy_server.proxy_config", _proxy_config()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await check_coordination_redis_connection(
                request=CoordinationRedisSettingsRequest(settings={"ssl": True}),
                user_api_key_dict=_admin_auth(),
            )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_connection_test_rejects_non_admin():
    with pytest.raises(HTTPException) as exc_info:
        await check_coordination_redis_connection(
            request=CoordinationRedisSettingsRequest(settings={"host": "coord-redis.example.com"}),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_role=LitellmUserRoles.INTERNAL_USER),
        )
    assert exc_info.value.status_code == 403
