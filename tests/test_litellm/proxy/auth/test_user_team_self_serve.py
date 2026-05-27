"""
Unit tests for LIT-3254: opt-in self-service team creation via
`general_settings.allow_user_team_creation`.

These tests pin the gate inside RouteChecks.non_proxy_admin_allowed_routes_check
without spinning up the full proxy. The behavior we verify:

1. Flag absent  -> internal user POSTing /team/new is BLOCKED
2. Flag = False -> internal user POSTing /team/new is BLOCKED
3. Flag = True  -> internal user POSTing /team/new is ALLOWED
4. Flag = True  -> internal user POSTing /team/delete still BLOCKED
5. Flag = True  -> PROXY_ADMIN_VIEW_ONLY POSTing /team/new still BLOCKED
6. Helper is safe under non-dict general_settings (early init)
"""
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks


def _make_user_and_token(role: str = LitellmUserRoles.INTERNAL_USER.value):
    valid_token = UserAPIKeyAuth(
        api_key="sk-test",
        user_id="user-internal-1",
        user_role=role,
    )
    user_obj = LiteLLM_UserTable(
        user_id="user-internal-1",
        user_role=role,
        user_email="u@example.com",
        spend=0.0,
        models=[],
        teams=[],
        max_budget=None,
    )
    return user_obj, valid_token


def _run(route, general_settings, role=LitellmUserRoles.INTERNAL_USER.value, method="POST", request_data=None):
    user_obj, valid_token = _make_user_and_token(role=role)
    request = MagicMock()
    request.query_params = {}
    request.method = method
    with patch(
        "litellm.proxy.proxy_server.general_settings", general_settings
    ), patch("litellm.proxy.proxy_server.premium_user", True):
        return RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=role,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data=request_data if request_data is not None else {},
        )


def test_internal_user_team_new_blocked_when_flag_absent():
    with pytest.raises(Exception) as exc:
        _run("/team/new", general_settings={})
    assert "Only proxy admin" in str(exc.value)


def test_internal_user_team_new_blocked_when_flag_false():
    with pytest.raises(Exception) as exc:
        _run("/team/new", general_settings={"allow_user_team_creation": False})
    assert "Only proxy admin" in str(exc.value)


def test_internal_user_team_new_allowed_when_flag_true():
    rv = _run("/team/new", general_settings={"allow_user_team_creation": True})
    assert rv is None


@pytest.mark.parametrize(
    "route",
    [
        "/team/delete",
        "/team/update",
        "/team/block",
        "/user/new",
        "/user/delete",
        "/user/bulk_update",
    ],
)
def test_flag_does_not_leak_to_other_admin_routes(route):
    """
    The flag is intentionally scoped to /team/new only. Any other
    management/admin route must still raise for an INTERNAL_USER even
    when the flag is True.
    """
    with pytest.raises(Exception) as exc:
        _run(route, general_settings={"allow_user_team_creation": True})
    assert "Only proxy admin" in str(exc.value)


def test_view_only_admin_team_new_still_blocked_when_flag_true():
    """
    PROXY_ADMIN_VIEW_ONLY follows a no-writes-ever rule regardless of
    `allow_user_team_creation`. /team/new is explicitly in the viewers
    write denylist; the new flag must not relax that.
    """
    with pytest.raises(Exception):
        _run(
            "/team/new",
            general_settings={"allow_user_team_creation": True},
            role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
        )


def test_internal_user_view_only_team_new_still_blocked_when_flag_true():
    """
    INTERNAL_USER_VIEW_ONLY must NEVER be able to create teams regardless of
    `allow_user_team_creation`. The new elif branch in
    `non_proxy_admin_allowed_routes_check` is scoped to
    `LitellmUserRoles.INTERNAL_USER.value` only; this test guards against a
    future refactor that accidentally widens the role match to cover
    view-only internal users.
    """
    with pytest.raises(Exception) as exc:
        _run(
            "/team/new",
            general_settings={"allow_user_team_creation": True},
            role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
        )
    assert "Only proxy admin" in str(exc.value)


def test_helper_returns_false_when_general_settings_is_not_a_dict():
    with patch("litellm.proxy.proxy_server.general_settings", None):
        assert RouteChecks._user_team_self_serve_enabled() is False
    with patch("litellm.proxy.proxy_server.general_settings", "not-a-dict"):
        assert RouteChecks._user_team_self_serve_enabled() is False


def test_helper_returns_true_when_flag_is_truthy():
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_user_team_creation": True},
    ):
        assert RouteChecks._user_team_self_serve_enabled() is True


def test_helper_returns_false_when_flag_missing():
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        assert RouteChecks._user_team_self_serve_enabled() is False


# -- Org-scoped self-service must NOT bypass the org-admin path -------------


def test_internal_user_team_new_with_organization_id_still_blocked_when_flag_true():
    """
    Veria flagged a cross-organization bypass on the first revision of
    LIT-3254: without this guard, an internal user with no organization
    memberships could POST `/team/new` with any known `organization_id` and
    silently create a team they admin inside that organization, because
    `team_endpoints.new_team` only checks that the org row exists.

    Fix: the self-service branch is restricted to standalone teams. Any
    request that carries an `organization_id` falls through to the existing
    `_user_is_org_admin` branch and is rejected if the caller is not an
    org-admin of that org.
    """
    with pytest.raises(Exception) as exc:
        _run(
            "/team/new",
            general_settings={"allow_user_team_creation": True},
            request_data={"organization_id": "org-abc-123"},
        )
    assert "Only proxy admin" in str(exc.value)


def test_internal_user_team_new_without_organization_id_still_allowed_when_flag_true():
    """
    Sanity: the new guard does not regress the standalone-team happy path.
    `request_data` carries no `organization_id`, so the gate still allows it.
    """
    rv = _run(
        "/team/new",
        general_settings={"allow_user_team_creation": True},
        request_data={"team_alias": "my-self-served-team"},
    )
    assert rv is None


def test_internal_user_team_new_with_explicit_none_organization_id_still_allowed():
    """
    `request_data={"organization_id": None}` is the same as omitting the
    field for our purposes — should still go through the new branch.
    """
    rv = _run(
        "/team/new",
        general_settings={"allow_user_team_creation": True},
        request_data={"organization_id": None},
    )
    assert rv is None
