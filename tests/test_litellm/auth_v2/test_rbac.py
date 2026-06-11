from __future__ import annotations

from fastapi.security import SecurityScopes

from litellm.auth_v2.models import AuthMethod, Principal, PrincipalType
from litellm.auth_v2.rbac import Role, has_any_role, has_required_scopes


def _principal(*, scopes=None, roles=None) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject="u1",
        auth_method=AuthMethod.OIDC,
        scopes=scopes or [],
        roles=roles or [],
    )


def test_required_scopes_is_subset_check():
    principal = _principal(scopes=["models:read", "chat:write", "scim:write"])
    assert has_required_scopes(SecurityScopes(["models:read"]), principal)
    assert has_required_scopes(SecurityScopes(["models:read", "chat:write"]), principal)


def test_missing_required_scope_fails():
    principal = _principal(scopes=["models:read"])
    assert not has_required_scopes(SecurityScopes(["chat:write"]), principal)


def test_empty_required_scopes_always_passes():
    assert has_required_scopes(SecurityScopes([]), _principal())


def test_has_any_role_matches_one_of_allowed():
    principal = _principal(roles=[Role.TEAM_MEMBER, Role.ORG_VIEWER])
    assert has_any_role(principal, (Role.ORG_VIEWER, Role.PLATFORM_ADMIN))


def test_has_any_role_rejects_when_no_overlap():
    principal = _principal(roles=[Role.TEAM_MEMBER])
    assert not has_any_role(principal, (Role.PLATFORM_ADMIN, Role.ORG_ADMIN))


def test_has_any_role_false_when_principal_has_no_roles():
    assert not has_any_role(_principal(), (Role.PLATFORM_ADMIN,))
