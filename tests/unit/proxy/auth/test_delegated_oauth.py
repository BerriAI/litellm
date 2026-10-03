from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import SessionPrincipal
from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, LiteLLM_UserTable, Member
from litellm.proxy.auth.delegated_oauth import delegated_identity
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    db: Final = MagicMock()
    db.writer_db.litellm_usertable.find_unique = AsyncMock(
        return_value=LiteLLM_UserTable(user_id="admin", user_role="proxy_admin", teams=[])
    )
    db.db = db.writer_db
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", db)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_custom_auth", None)
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize("role,active,status", [("internal_user", True, 403), ("proxy_admin", False, 401)])
async def test_delegated_identity_rechecks_current_admin(
    database: MagicMock, role: str, active: bool, status: int
) -> None:
    principal: Final = SessionPrincipal(user_id="admin", client_id="app", audience="proxy_api")
    assert (await delegated_identity(principal)).user_role == "proxy_admin"
    database.writer_db.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id="admin", user_role=role, metadata={"scim_active": active}
    )
    with pytest.raises(HTTPException) as error:
        await delegated_identity(principal)
    assert error.value.status_code == status


@pytest.mark.asyncio
async def test_delegated_identity_uses_current_team_limits_and_roster(database: MagicMock) -> None:
    database.writer_db.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id="admin", user_role="proxy_admin", teams=["team"], rpm_limit=13
    )
    database.writer_db.litellm_teamtable.find_unique = AsyncMock(
        return_value=LiteLLM_TeamTable(
            team_id="team", models=["allowed-model"], rpm_limit=10,
            members_with_roles=[Member(user_id="admin", role="user")],
        )
    )
    database.writer_db.litellm_teammembership.find_unique = AsyncMock(
        return_value=LiteLLM_TeamMembership(
            user_id="admin", team_id="team", litellm_budget_table=LiteLLM_BudgetTable(rpm_limit=3)
        )
    )
    principal: Final = SessionPrincipal(user_id="admin", client_id="app", audience="proxy_api", team_id="team")
    first: Final = await delegated_identity(principal)
    assert (first.user_rpm_limit, first.team_rpm_limit, first.team_member_rpm_limit) == (13, 10, 3)
    assert first.team_models == ["allowed-model"]
    database.writer_db.litellm_teammembership.find_unique.return_value = LiteLLM_TeamMembership(
        user_id="admin", team_id="team", litellm_budget_table=LiteLLM_BudgetTable(rpm_limit=1)
    )
    assert (await delegated_identity(principal)).team_member_rpm_limit == 1
    database.writer_db.litellm_teamtable.find_unique.return_value = LiteLLM_TeamTable(team_id="team")
    with pytest.raises(HTTPException) as error:
        await delegated_identity(principal)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_delegated_identity_fails_closed_when_database_is_unavailable(database: MagicMock) -> None:
    database.writer_db.litellm_usertable.find_unique.side_effect = httpx.ConnectError("Database unavailable")
    with pytest.raises(HTTPException) as error:
        await delegated_identity(SessionPrincipal(user_id="admin", client_id="app"))
    assert error.value.status_code == 503
