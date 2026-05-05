"""
Audit-log emission for the team-callback admin endpoints.

The endpoints in ``team_callback_endpoints.py`` mutate a team's logging
callbacks (``add_team_callbacks``) or zero them out entirely
(``disable_team_logging``).  Both are admin-only mutations, and the
disable variant is itself a logging-control action, so when the operator
has Enterprise audit logging enabled (``litellm.store_audit_logs = True``)
each call must emit a row that captures who did it and what the metadata
looked like before/after.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import (
    AddTeamCallback,
    LitellmTableNames,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_callback_endpoints import (
    add_team_callbacks,
    disable_team_logging,
)


def _admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="admin-user",
        user_role="proxy_admin",
    )


def _existing_team_row(metadata: dict) -> MagicMock:
    row = MagicMock()
    row.team_id = "team-1"
    row.metadata = metadata
    return row


def _patch_prisma(existing_metadata: dict):
    """Build a context-manager that patches the proxy's ``prisma_client``
    to return ``existing_metadata`` from ``get_data`` and a stub team row
    from ``litellm_teamtable.update``."""
    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock(return_value=_existing_team_row(existing_metadata))

    updated_row = MagicMock()
    updated_row.team_id = "team-1"
    mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=updated_row)
    return mock_prisma


@pytest.mark.asyncio
async def test_disable_team_logging_emits_audit_log_when_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma = _patch_prisma(
        {
            "callback_settings": {
                "success_callback": ["langfuse"],
                "failure_callback": [],
            }
        }
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
        # asyncio.create_task fires the coroutine eagerly; await one tick to let
        # the audit-log emit run before the test exits.
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
    # Before: the team's pre-existing success_callback survives in the snapshot.
    assert before["metadata"]["callback_settings"]["success_callback"] == ["langfuse"]
    # After: callbacks zeroed out by the endpoint.
    assert after["metadata"]["callback_settings"]["success_callback"] == []
    assert after["metadata"]["callback_settings"]["failure_callback"] == []


@pytest.mark.asyncio
async def test_disable_team_logging_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _patch_prisma(
        {
            "callback_settings": {
                "success_callback": ["langfuse"],
                "failure_callback": [],
            }
        }
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
    mock_prisma = _patch_prisma({"logging": []})

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
    # ``litellm_changed_by`` header takes precedence over the auth user_id.
    assert log.changed_by == "ops-on-call"

    before = json.loads(log.before_value)
    after = json.loads(log.updated_values)
    assert before["metadata"]["logging"] == []
    assert len(after["metadata"]["logging"]) == 1
    assert after["metadata"]["logging"][0]["callback_name"] == "langfuse"

    # Callback secrets MUST NOT leak into the audit log payload.
    callback_vars = after["metadata"]["logging"][0]["callback_vars"]
    assert callback_vars["langfuse_public_key"] != "pk"
    assert callback_vars["langfuse_secret_key"] != "sk"
    # Key names are preserved so the auditor can see which fields changed.
    assert "langfuse_public_key" in callback_vars
    assert "langfuse_secret_key" in callback_vars
    # And no plaintext secret should appear anywhere in the serialized row.
    assert "sk" not in log.updated_values.replace("sk-", "")  # crude leak check
    assert "pk" not in (log.updated_values.replace("pk-", "").replace("public_key", ""))


@pytest.mark.asyncio
async def test_disable_team_logging_redacts_existing_callback_secrets(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    # Existing team has populated callback_vars containing secrets — redaction
    # must apply to the BEFORE snapshot too.
    mock_prisma = _patch_prisma(
        {
            "callback_settings": {
                "success_callback": ["langfuse"],
                "failure_callback": [],
                "callback_vars": {
                    "langfuse_public_key": "pk-real",
                    "langfuse_secret_key": "sk-real-secret",
                },
            }
        }
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
    # The pre-existing secret_key value must NOT appear in the serialized
    # before_value or updated_values.
    assert "sk-real-secret" not in log.before_value
    assert "sk-real-secret" not in log.updated_values
    assert "pk-real" not in log.before_value
    assert "pk-real" not in log.updated_values


@pytest.mark.asyncio
async def test_add_team_callbacks_no_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    mock_prisma = _patch_prisma({"logging": []})

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
