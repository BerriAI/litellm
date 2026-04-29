"""
Regression tests for the IDOR fix on team callback endpoints
(GHSA-xxv2-fprq-9x93).

The three endpoints below previously authenticated the caller but never
checked whether the caller could manage the target team — any
authenticated key holder could write callback credentials to any team,
disable any team's logging, or read back another team's stored
third-party API credentials (Langfuse / Langsmith / GCS).

The fix routes each handler through ``_verify_team_access``, which
enforces the proxy-admin / org-admin / team-admin hierarchy used by
sibling endpoints in ``team_endpoints.py``.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import (
    AddTeamCallback,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.team_callback_endpoints import (
    add_team_callbacks,
    disable_team_logging,
    get_team_callbacks,
)


def _other_team_existing_row():
    """Return a mock team row owned by someone other than the test caller."""
    row = MagicMock()
    row.model_dump.return_value = {
        "team_id": "team-victim",
        "team_alias": "victim-team",
        "members_with_roles": [
            {"role": "admin", "user_id": "victim_admin"},
        ],
        "organization_id": "org-victim",
    }
    row.metadata = {}
    return row


@pytest.fixture
def unauthorized_caller():
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="random_authenticated_user",
        api_key="sk-random",
    )


@pytest.fixture
def patched_prisma():
    """
    Patch the proxy_server.prisma_client used inside each handler with a
    mock that returns a victim-team row from get_data().
    """
    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
        ) as mock_client,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_org_admin_for_team",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        mock_client.get_data = AsyncMock(return_value=_other_team_existing_row())
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
    # The unauthorized caller must NOT have written to the victim team.
    patched_prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_disable_team_logging_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    # The endpoint catches HTTPException and re-wraps it as ProxyException
    # with the original status code preserved.
    from litellm.proxy._types import ProxyException

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await disable_team_logging(
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    code = getattr(exc.value, "status_code", None) or getattr(exc.value, "code", None)
    assert int(code) == 403
    patched_prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
async def test_get_team_callbacks_rejects_unauthorized_caller(
    patched_prisma, unauthorized_caller
):
    # The endpoint catches generic Exception and re-wraps as ProxyException;
    # an HTTPException raised by the access guard surfaces as a 403
    # ProxyException — both shapes are acceptable failure modes, what
    # matters is that the caller does NOT receive the team's callback data.
    from litellm.proxy._types import ProxyException

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await get_team_callbacks(
            http_request=Mock(spec=Request),
            team_id="team-victim",
            user_api_key_dict=unauthorized_caller,
        )
    code = getattr(exc.value, "status_code", None) or getattr(exc.value, "code", None)
    assert int(code) == 403


@pytest.mark.asyncio
async def test_proxy_admin_can_add_team_callbacks(patched_prisma):
    """
    A proxy admin should pass the access guard and reach the DB write.
    Sanity check that the guard didn't over-rotate.
    """
    proxy_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin",
        api_key="sk-admin",
    )
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
        user_api_key_dict=proxy_admin,
    )
    patched_prisma.db.litellm_teamtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_admin_of_target_team_can_add_callbacks(patched_prisma):
    """
    A team admin OF THE TARGET team should pass the access guard.
    """
    # Override the victim row so the caller IS the team admin.
    row = MagicMock()
    row.model_dump.return_value = {
        "team_id": "team-victim",
        "team_alias": "victim-team",
        "members_with_roles": [
            {"role": "admin", "user_id": "team_admin_user"},
        ],
        "organization_id": "org-victim",
    }
    row.metadata = {}
    patched_prisma.get_data = AsyncMock(return_value=row)

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
