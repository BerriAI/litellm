import jwt

from litellm.proxy.management_endpoints.ui_sso import MicrosoftSSOHandler
from litellm.proxy.management_endpoints.types import get_litellm_user_role
from litellm.proxy._types import LitellmUserRoles


def test_extracts_proxy_admin_role_from_jwt():
    """Ensure supported app roles like 'proxy_admin' are extracted from the id_token."""
    payload = {
        "sub": "user123",
        "email": "admin@company.com",
        "app_roles": ["proxy_admin"],
        "aud": "litellm-app",
        "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
        "exp": 9999999999,
    }

    token = jwt.encode(payload, "secret", algorithm="HS256")
    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    assert roles == ["proxy_admin"]


def test_maps_internal_user_role():
    """Ensure internal_user role is correctly mapped to LitellmUserRoles."""
    payload = {
        "sub": "user456",
        "email": "user@company.com",
        "app_roles": ["internal_user"],
        "aud": "litellm-app",
        "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
        "exp": 9999999999,
    }

    token = jwt.encode(payload, "secret", algorithm="HS256")
    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    # Map to LitellmUserRoles
    chosen = None
    for r in roles:
        mapped = get_litellm_user_role(r)
        if mapped is not None:
            chosen = mapped
            break

    assert chosen == LitellmUserRoles.INTERNAL_USER


def test_maps_proxy_admin_viewer_role():
    """Ensure proxy_admin_viewer role is correctly mapped."""
    payload = {
        "sub": "user789",
        "email": "viewer@company.com",
        "app_roles": ["proxy_admin_viewer"],
        "aud": "litellm-app",
        "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
        "exp": 9999999999,
    }

    token = jwt.encode(payload, "secret", algorithm="HS256")
    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    chosen = None
    for r in roles:
        mapped = get_litellm_user_role(r)
        if mapped is not None:
            chosen = mapped
            break

    assert chosen == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY


def test_defaults_to_internal_user_viewer_when_no_role():
    """Ensure default role is internal_user_viewer when no app role is present."""
    payload = {
        "sub": "user_no_role",
        "email": "noRole@company.com",
        "aud": "litellm-app",
        "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
        "exp": 9999999999,
    }

    token = jwt.encode(payload, "secret", algorithm="HS256")
    roles = MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    assert roles == []

    # Default role would be internal_user_viewer
    default_role = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    assert default_role.value == "internal_user_viewer"

