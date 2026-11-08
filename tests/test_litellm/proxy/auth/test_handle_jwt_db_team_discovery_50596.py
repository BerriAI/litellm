import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable
from litellm.proxy.auth.handle_jwt import JWTAuthManager


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_prefers_entitled_team_when_model_is_none():
    """
    Regression test for #50596 / #50598:
    When requested_model is None (e.g. /v1/models or /v1/mcp/tools discovery)
    and a user belongs to an organizational team with models=['no-default-models']
    listed before an access-bearing team, _resolve_db_team_fallback should prefer
    the team with model / access_group_ids / object_permission entitlements.
    """
    dept_team = LiteLLM_TeamTable(
        team_id="dept-team",
        models=["no-default-models"],
        access_group_ids=[],
        blocked=False,
    )
    access_team = LiteLLM_TeamTable(
        team_id="access-team",
        models=["no-default-models"],
        access_group_ids=["ag-models-and-mcp"],
        blocked=False,
    )

    async def fake_get_team_object(team_id, **kwargs):
        return dept_team if team_id == "dept-team" else access_team

    user_object = LiteLLM_UserTable(
        user_id="user-1",
        teams=["dept-team", "access-team"],
    )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            side_effect=fake_get_team_object,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new=AsyncMock(return_value=None),
        ),
        patch.object(
            JWTAuthManager,
            "_is_team_route_allowed",
            return_value=True,
        ),
    ):
        team_id, team_obj, _ = await JWTAuthManager._resolve_db_team_fallback(
            user_object=user_object,
            user_id="user-1",
            requested_model=None,
            route="/v1/models",
            jwt_handler=MagicMock(),
            enforce_team_based_model_access=True,
            team_id_upsert=False,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert team_id == "access-team"
    assert team_obj is not None
    assert team_obj.team_id == "access-team"
