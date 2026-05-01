"""
Regression tests for team callback endpoint access control and audit logging.

The team callback endpoints mutate or expose callback credentials. They must
enforce target-team management access and, when audit logging is enabled, emit
redacted audit rows for callback mutations.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy._types import (
    AddTeamCallback,
    LitellmTableNames,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_callback_endpoints import (
    add_team_callbacks,
    disable_team_logging,
    get_team_callbacks,
)


def _team_row(
    *,
    team_id: str = "team-victim",
    metadata: dict | None = None,
    admin_user_id: str = "victim_admin",
    organization_id: str = "org-victim",
) -> MagicMock:
    row = MagicMock()
    row.team_id = team_id
    row.metadata = metadata or {}
    row.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "victim-team",
        "members_with_roles": [
            {"role": "admin", "user_id": admin_user_id},
        ],
        "organization_id": organization_id,
        "metadata": row.metadata,
    }
    return row


def _patch_prisma(existing_team: MagicMock):
    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock(return_value=existing_team)

    updated_row = MagicMock()
    updated_row.team_id = existing_team.team_id
    mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=updated_row)
    return mock_prisma


def _admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


@pytest.fixture
def unauthorized_caller():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="random_authenticated_user",
        api_key="sk-random",
    )


@pytest.fixture
def patched_prisma():
    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_client,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_org_admin_for_team",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        mock_client.get_data = AsyncMock(return_value=_team_row())
        mock_client.db.litellm_teamtable.update = AsyncMock()
        yield mock_client


@pytest.mark.asyncio
async def test_add_team_callbacks_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={
            "langfuse_public_key": "pk-attacker",
            "langfuse_secret_key": "sk-attacker",
        },
    )
    with pytest.raises(HTTPException) as exc:
        await add_team_callbacks(
            data=data,
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403
    patched_prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_disable_team_logging_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    with pytest.raises(HTTPException) as exc:
        await disable_team_logging(
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403
    patched_prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_get_team_callbacks_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    with pytest.raises(HTTPException) as exc:
        await get_team_callbacks(
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_proxy_admin_can_add_team_callbacks(patched_prisma):
    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={
            "langfuse_public_key": "pk-admin",
            "langfuse_secret_key": "sk-admin",
        },
    )
    await add_team_callbacks(
        data=data,
        http_request=Mock(spec=Request),
        team_id="team-victim",
        user_api_key_dict=_admin_auth(),
    )
    patched_prisma.db.litellm_teamtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_admin_of_target_team_can_add_callbacks(patched_prisma):
    patched_prisma.get_data = AsyncMock(
        return_value=_team_row(admin_user_id="team_admin_user")
    )

    team_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="team_admin_user",
        api_key="sk-team-admin",
    )
    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={
            "langfuse_public_key": "pk-team",
            "langfuse_secret_key": "sk-team",
        },
    )
    await add_team_callbacks(
        data=data,
        http_request=Mock(spec=Request),
        team_id="team-victim",
        user_api_key_dict=team_admin,
    )
    patched_prisma.db.litellm_teamtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_disable_team_logging_emits_audit_log_when_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(
        _team_row(
            team_id="team-1",
            metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": [],
                }
            },
        )
    )

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert log.table_name == LitellmTableNames.TEAM_TABLE_NAME
    assert log.object_id == "team-1"
    assert log.action == "updated"
    assert log.changed_by == "admin-user"

    before = json.loads(log.before_value)
    after = json.loads(log.updated_values)
    assert before["metadata"]["callback_settings"]["success_callback"] == ["langfuse"]
    assert after["metadata"]["callback_settings"]["success_callback"] == []
    assert after["metadata"]["callback_settings"]["failure_callback"] == []


@pytest.mark.asyncio
async def test_disable_team_logging_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _patch_prisma(
        _team_row(
            team_id="team-1",
            metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": [],
                }
            },
        )
    )

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    assert audit_calls == []


@pytest.mark.asyncio
async def test_add_team_callbacks_emits_audit_log_when_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={"logging": []}))

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langfuse",
                callback_type="success",
                callback_vars={
                    "langfuse_public_key": "pk",
                    "langfuse_secret_key": "sk",
                },
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by="ops-on-call",
        )
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert log.table_name == LitellmTableNames.TEAM_TABLE_NAME
    assert log.object_id == "team-1"
    assert log.action == "updated"
    assert log.changed_by == "ops-on-call"

    before = json.loads(log.before_value)
    after = json.loads(log.updated_values)
    assert before["metadata"]["logging"] == []
    assert len(after["metadata"]["logging"]) == 1
    assert after["metadata"]["logging"][0]["callback_name"] == "langfuse"

    callback_vars = after["metadata"]["logging"][0]["callback_vars"]
    assert callback_vars["langfuse_public_key"] != "pk"
    assert callback_vars["langfuse_secret_key"] != "sk"
    assert "langfuse_public_key" in callback_vars
    assert "langfuse_secret_key" in callback_vars
    assert "sk" not in log.updated_values.replace("sk-", "")
    assert "pk" not in (log.updated_values.replace("pk-", "").replace("public_key", ""))


@pytest.mark.asyncio
async def test_disable_team_logging_redacts_existing_callback_secrets(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(
        _team_row(
            team_id="team-1",
            metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": [],
                    "callback_vars": {
                        "langfuse_public_key": "pk-real",
                        "langfuse_secret_key": "sk-real-secret",
                    },
                }
            },
        )
    )

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await disable_team_logging(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    assert len(audit_calls) == 1
    log = audit_calls[0]
    assert "sk-real-secret" not in log.before_value
    assert "sk-real-secret" not in log.updated_values
    assert "pk-real" not in log.before_value
    assert "pk-real" not in log.updated_values


@pytest.mark.asyncio
async def test_add_team_callbacks_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _patch_prisma(_team_row(team_id="team-1", metadata={"logging": []}))

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ),
    ):
        await add_team_callbacks(
            data=AddTeamCallback(
                callback_name="langfuse",
                callback_type="success",
                callback_vars={
                    "langfuse_public_key": "pk",
                    "langfuse_secret_key": "sk",
                },
            ),
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=_admin_auth(),
            litellm_changed_by=None,
        )

    assert audit_calls == []
