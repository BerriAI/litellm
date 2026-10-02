from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.log_visibility import LogVisibility, log_visibility


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "expected"),
    (
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), LogVisibility(all_teams=True)),
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), LogVisibility(all_teams=True)),
        (
            UserAPIKeyAuth(user_id="user", token="key", team_id="unpermitted"),
            LogVisibility(user_id="user", team_ids=("permitted",)),
        ),
        (UserAPIKeyAuth(token="key", team_id="unpermitted"), LogVisibility(api_key_hash="key")),
    ),
)
async def test_log_visibility_uses_user_and_permitted_teams_instead_of_key_team_membership(
    auth: UserAPIKeyAuth,
    expected: LogVisibility,
) -> None:
    async def permitted_teams(caller: UserAPIKeyAuth) -> tuple[str, ...]:
        assert caller is auth
        return ("permitted",)

    assert await log_visibility(auth, permitted_teams) == expected


@pytest.mark.asyncio
async def test_missing_team_permissions_preserve_only_authenticated_user_visibility() -> None:
    async def no_teams(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    auth: Final = UserAPIKeyAuth(user_id="user", token="key", team_id="team")
    assert await log_visibility(auth, no_teams) == LogVisibility(user_id=auth.user_id)


@pytest.mark.asyncio
async def test_team_membership_without_authenticated_identity_does_not_grant_log_access() -> None:
    with pytest.raises(HTTPException) as error:
        await log_visibility(UserAPIKeyAuth(team_id="team"))
    assert error.value.status_code == 403
