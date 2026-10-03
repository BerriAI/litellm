from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import SessionPrincipal
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    Member,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import common_checks
from litellm.proxy.auth.delegated_oauth import delegated_identity
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging


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
            team_id="team",
            models=["allowed-model"],
            rpm_limit=10,
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


async def _common_checks(token: UserAPIKeyAuth, team: LiteLLM_TeamTable | None = None) -> bool:
    logging: Final = MagicMock(spec=ProxyLogging)
    logging.budget_alerts = AsyncMock()
    return await common_checks(
        request_body={"model": "gpt-4"},
        team_object=team,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route="/v1/chat/completions",
        llm_router=None,
        proxy_logging_obj=logging,
        valid_token=token,
        request=Request(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [], "query_string": b""}
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
async def test_common_checks_reread_organization_budget_for_delegated_identity(
    database: MagicMock, fresh: bool
) -> None:
    from litellm.proxy import proxy_server

    cached: Final = LiteLLM_OrganizationTable(
        organization_id="org",
        budget_id="budget",
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )
    await proxy_server.user_api_key_cache.async_set_cache(key="org_id:org:with_budget", value=cached)
    database.db.litellm_organizationtable.find_unique = AsyncMock(
        return_value=cached.model_copy(update={"litellm_budget_table": LiteLLM_BudgetTable(max_budget=0.0)})
    )
    token: Final = UserAPIKeyAuth(token="token", user_id="admin", org_id="org")
    token.requires_fresh_policy = fresh
    if not fresh:
        assert await _common_checks(token) is True
        return
    with pytest.raises(litellm.BudgetExceededError):
        await _common_checks(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
async def test_common_checks_reread_team_access_groups_for_delegated_identity(database: MagicMock, fresh: bool) -> None:
    from litellm.proxy import proxy_server

    cached: Final = LiteLLM_AccessGroupTable(
        access_group_id="group", access_group_name="group", access_model_names=["gpt-4"]
    )
    await proxy_server.user_api_key_cache.async_set_cache(key="access_group_id:group", value=cached)
    database.writer_db.litellm_accessgrouptable.find_unique = AsyncMock(
        return_value=cached.model_copy(update={"access_model_names": []})
    )
    database.writer_db.litellm_teammembership.find_unique = AsyncMock(return_value=None)
    team: Final = LiteLLM_TeamTable(team_id="team", models=["other-model"], access_group_ids=["group"])
    token: Final = UserAPIKeyAuth(token="token", user_id="admin", team_id="team")
    token.requires_fresh_policy = fresh
    if not fresh:
        assert await _common_checks(token, team) is True
        return
    with pytest.raises(ProxyException) as error:
        await _common_checks(token, team)
    assert error.value.type == ProxyErrorTypes.team_model_access_denied
