import jwt
import pytest

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.management_endpoints.ui_sso import MicrosoftSSOHandler


def _id_token(**claims) -> str:
    """Build a signed id_token carrying the given claims."""
    payload = {
        "sub": "user123",
        "email": "user@company.com",
        "aud": "litellm-app",
        "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
        "exp": 9999999999,
        **claims,
    }
    return jwt.encode(payload, "secret", algorithm="HS256")


def test_extracts_proxy_admin_role_from_jwt():
    """Ensure supported app roles like 'proxy_admin' are extracted from the id_token."""
    token = _id_token(app_roles=["proxy_admin"])

    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    assert roles == ["proxy_admin"]


def test_extracts_app_roles_from_roles_claim():
    """Entra emits app role values in the `roles` claim; both spellings are read."""
    token = _id_token(roles=["internal_user"])

    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    assert roles == ["internal_user"]


@pytest.mark.parametrize(
    "app_roles, expected",
    [
        (["proxy_admin"], LitellmUserRoles.PROXY_ADMIN),
        (["proxy_admin_viewer"], LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        (["internal_user"], LitellmUserRoles.INTERNAL_USER),
        (["internal_user_viewer"], LitellmUserRoles.INTERNAL_USER_VIEW_ONLY),
        # Case-insensitive, matching get_litellm_user_role.
        (["PROXY_ADMIN_VIEWER"], LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        # Roles outside the privilege hierarchy still resolve.
        (["org_admin"], LitellmUserRoles.ORG_ADMIN),
    ],
)
def test_maps_single_app_role(app_roles, expected):
    """A lone app role maps to its LitellmUserRoles equivalent."""
    assert MicrosoftSSOHandler.get_user_role_from_app_roles(app_roles) == expected


@pytest.mark.parametrize(
    "app_roles",
    [
        ["internal_user", "proxy_admin_viewer"],
        ["proxy_admin_viewer", "internal_user"],
    ],
)
def test_highest_privilege_role_wins_regardless_of_claim_order(app_roles):
    """
    A user in one group mapped to `internal_user` and another mapped to
    `proxy_admin_viewer` gets the higher privilege role either way.

    Entra does not guarantee the ordering of the `roles` claim, so the resolved
    role must not depend on it.
    """
    assert MicrosoftSSOHandler.get_user_role_from_app_roles(app_roles) == (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)


@pytest.mark.parametrize(
    "app_roles",
    [
        ["internal_user", "proxy_admin_viewer", "proxy_admin"],
        ["proxy_admin", "proxy_admin_viewer", "internal_user"],
        ["proxy_admin_viewer", "internal_user", "proxy_admin"],
    ],
)
def test_proxy_admin_beats_every_other_role(app_roles):
    """proxy_admin outranks every other role in the hierarchy, in any claim order."""
    assert MicrosoftSSOHandler.get_user_role_from_app_roles(app_roles) == LitellmUserRoles.PROXY_ADMIN


def test_unrecognised_app_roles_are_ignored():
    """App roles that are not LitellmUserRoles values do not shadow ones that are."""
    app_roles = ["Some.Custom.Role", "msiam_access", "internal_user"]

    assert MicrosoftSSOHandler.get_user_role_from_app_roles(app_roles) == LitellmUserRoles.INTERNAL_USER


@pytest.mark.parametrize("app_roles", [None, [], ["msiam_access"], ["User"]])
def test_returns_none_when_no_role_resolves(app_roles):
    """
    Returning None lets the caller keep the user's stored role or apply
    default_internal_user_params, rather than forcing a role.
    """
    assert MicrosoftSSOHandler.get_user_role_from_app_roles(app_roles) is None


def test_no_role_claim_yields_no_app_roles():
    """An id_token with no role claim produces no app roles, and so no role."""
    token = _id_token()

    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    assert roles == []
    assert MicrosoftSSOHandler.get_user_role_from_app_roles(roles) is None


def test_end_to_end_from_id_token_to_role():
    """The id_token -> role path resolves the highest privilege role."""
    token = _id_token(roles=["internal_user", "proxy_admin_viewer"])

    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    assert MicrosoftSSOHandler.get_user_role_from_app_roles(roles) == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
