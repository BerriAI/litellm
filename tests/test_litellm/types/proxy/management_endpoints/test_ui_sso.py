import pytest
from pydantic import ValidationError

from litellm.proxy._types import KeyManagementRoutes, TeamMemberPermissions
from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams

GRANTABLE_PERMISSIONS = sorted(
    permission.value
    for permission in TeamMemberPermissions
    if permission not in (TeamMemberPermissions.KEY_INFO, TeamMemberPermissions.KEY_HEALTH)
)

DEAD_KEY_MANAGEMENT_ROUTES = sorted(
    {route.value for route in KeyManagementRoutes} - {permission.value for permission in TeamMemberPermissions}
)


def test_every_grantable_permission_is_a_key_management_route():
    key_management_values = {route.value for route in KeyManagementRoutes}
    assert {permission.value for permission in TeamMemberPermissions} <= key_management_values


def test_grantable_permissions_are_exactly_the_enforced_set():
    assert GRANTABLE_PERMISSIONS == [
        "/key/access_group_assignment",
        "/key/delete",
        "/key/generate",
        "/key/list",
        "/key/regenerate",
        "/key/service-account/generate",
        "/key/update",
        "/spend/logs",
        "/team/daily/activity",
    ]


def test_default_team_params_accept_every_grantable_permission():
    params = DefaultTeamSSOParams(team_member_permissions=GRANTABLE_PERMISSIONS)
    assert [permission.value for permission in params.team_member_permissions or []] == GRANTABLE_PERMISSIONS


def test_default_team_params_accept_the_always_included_baseline_pair():
    params = DefaultTeamSSOParams(team_member_permissions=["/key/info", "/key/health"])
    assert [permission.value for permission in params.team_member_permissions or []] == ["/key/info", "/key/health"]


@pytest.mark.parametrize("dead_route", DEAD_KEY_MANAGEMENT_ROUTES)
def test_default_team_params_reject_grants_that_are_never_enforced(dead_route):
    with pytest.raises(ValidationError):
        DefaultTeamSSOParams(team_member_permissions=[dead_route])


def test_dead_routes_cover_the_known_unenforced_grants():
    assert DEAD_KEY_MANAGEMENT_ROUTES == [
        "/key/aliases",
        "/key/block",
        "/key/bulk_update",
        "/key/unblock",
        "/key/{key_id}/regenerate",
        "/key/{key_id}/reset_spend",
        "/spend/logs/v2",
        "/team/key/bulk_update",
    ]
