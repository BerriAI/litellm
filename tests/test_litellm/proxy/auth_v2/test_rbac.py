from __future__ import annotations

import pytest
from fastapi.security import SecurityScopes

from litellm.proxy.auth_v2.authorization import RBACEngine, Role
from litellm.proxy.auth_v2.models import AuthMethod, Principal, PrincipalType


def _principal(*, scopes=None, roles=None) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject="u1",
        auth_method=AuthMethod.OIDC,
        scopes=scopes or [],
        roles=roles or [],
    )


# --------------------------------------------------------------------------- #
# Scopes stay a plain SecurityScopes subset check (not Casbin)
# --------------------------------------------------------------------------- #


def test_required_scopes_is_subset_check():
    principal = _principal(scopes=["models:read", "chat:write", "scim:write"])
    assert principal.has_required_scopes(SecurityScopes(["models:read"]))
    assert principal.has_required_scopes(SecurityScopes(["models:read", "chat:write"]))


def test_missing_required_scope_fails():
    principal = _principal(scopes=["models:read"])
    assert not principal.has_required_scopes(SecurityScopes(["chat:write"]))


def test_empty_required_scopes_always_passes():
    assert _principal().has_required_scopes(SecurityScopes([]))


# --------------------------------------------------------------------------- #
# RBACEngine.has_role honors the role hierarchy (Casbin g-rules)
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine() -> RBACEngine:
    return RBACEngine()


@pytest.mark.parametrize(
    "held,gate",
    [
        (Role.PLATFORM_ADMIN, Role.ORG_ADMIN),
        (Role.PLATFORM_ADMIN, Role.ORG_VIEWER),
        (Role.PLATFORM_ADMIN, Role.TEAM_ADMIN),
        (Role.PLATFORM_ADMIN, Role.TEAM_MEMBER),
        (Role.PLATFORM_ADMIN, Role.PLATFORM_VIEWER),
        (Role.ORG_ADMIN, Role.ORG_VIEWER),
        (Role.ORG_ADMIN, Role.ORG_ADMIN),  # exact match
        (Role.TEAM_ADMIN, Role.TEAM_MEMBER),
    ],
)
def test_has_role_inherits_down_the_hierarchy(engine, held, gate):
    assert engine.has_any_role(_principal(roles=[held]), (gate,))


@pytest.mark.parametrize(
    "held,gate",
    [
        (Role.ORG_ADMIN, Role.TEAM_MEMBER),  # sideways, no inheritance edge
        (Role.TEAM_MEMBER, Role.ORG_ADMIN),  # lower cannot reach higher
        (Role.ORG_VIEWER, Role.ORG_ADMIN),
    ],
)
def test_has_role_does_not_climb_the_hierarchy(engine, held, gate):
    assert not engine.has_any_role(_principal(roles=[held]), (gate,))


def test_has_role_false_without_roles(engine):
    assert not engine.has_any_role(_principal(), (Role.TEAM_MEMBER,))


# --------------------------------------------------------------------------- #
# RBACEngine.enforce against the default policy
# --------------------------------------------------------------------------- #


def test_platform_admin_enforces_any_object_and_action(engine):
    assert engine.enforce(_principal(roles=[Role.PLATFORM_ADMIN]), "/anything", "POST")
    # keyMatch: "/*" / "/scim/v2/*" span path separators, so deep paths are covered
    assert engine.enforce(
        _principal(roles=[Role.PLATFORM_ADMIN]), "/scim/v2/Users", "DELETE"
    )
    assert engine.enforce(
        _principal(roles=[Role.PLATFORM_ADMIN]), "/scim/v2/Groups", "POST"
    )


def test_platform_viewer_is_read_only(engine):
    viewer = _principal(roles=[Role.PLATFORM_VIEWER])
    assert engine.enforce(viewer, "/anything", "GET")
    assert not engine.enforce(viewer, "/anything", "POST")


def test_org_viewer_has_no_write_grant(engine):
    assert not engine.enforce(_principal(roles=[Role.ORG_VIEWER]), "/widgets", "POST")


def test_enforce_false_without_roles(engine):
    assert not engine.enforce(_principal(), "/anything", "GET")


# --------------------------------------------------------------------------- #
# Operator CSV policy fully replaces the in-code defaults
# --------------------------------------------------------------------------- #


def test_csv_policy_overrides_defaults(tmp_path):
    policy = tmp_path / "policy.csv"
    policy.write_text("p, platform_viewer, /reports, POST\n")
    engine = RBACEngine(policy_path=str(policy))

    # the operator rule is honored
    assert engine.enforce(_principal(roles=[Role.PLATFORM_VIEWER]), "/reports", "POST")
    # the built-in platform_admin "/*" grant is gone, not merged
    assert not engine.enforce(
        _principal(roles=[Role.PLATFORM_ADMIN]), "/reports", "POST"
    )
    assert not engine.enforce(
        _principal(roles=[Role.PLATFORM_ADMIN]), "/anything", "GET"
    )


def test_act_matcher_is_anchored(tmp_path):
    # a "GET" policy must not grant a superstring act like "GETX" (regexMatch ^(...)$)
    policy = tmp_path / "policy.csv"
    policy.write_text("p, platform_viewer, /x, GET\n")
    engine = RBACEngine(policy_path=str(policy))
    viewer = _principal(roles=[Role.PLATFORM_VIEWER])
    assert engine.enforce(viewer, "/x", "GET")
    assert not engine.enforce(viewer, "/x", "GETX")


# --------------------------------------------------------------------------- #
# filter_claim_roles: the shared allowlist gate for JWT, OIDC-login and SAML
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "roles,allowed,allow_platform,expected",
    [
        # default deny: a self-asserted role grants nothing
        (["platform_admin", "org_admin"], [], False, []),
        # allowlist filters; platform role excluded even if listed without the gate
        (["platform_admin", "org_admin"], ["org_admin"], False, ["org_admin"]),
        (["platform_admin"], ["platform_admin"], False, []),
        # platform role only survives with the explicit gate
        (["platform_admin"], ["platform_admin"], True, ["platform_admin"]),
    ],
)
def test_filter_claim_roles(roles, allowed, allow_platform, expected):
    from litellm.proxy.auth_v2.authorization import filter_claim_roles

    assert filter_claim_roles(roles, allowed, allow_platform) == expected


# --------------------------------------------------------------------------- #
# Object matcher spans path separators (keyMatch): multi-segment authorization
# --------------------------------------------------------------------------- #


def test_enforce_matches_multi_segment_paths(engine):
    # "/*" now spans separators, so nested routes are covered by the default policy
    assert engine.enforce(
        _principal(roles=[Role.PLATFORM_VIEWER]), "/api/v1/models", "GET"
    )
    assert engine.enforce(
        _principal(roles=[Role.PLATFORM_ADMIN]), "/api/v1/x/y", "POST"
    )


def test_enforce_denies_multi_segment_when_unauthorized(engine):
    # viewer is GET-only and org_viewer has no write grant, even on nested paths;
    # the act anchor still rejects a superstring verb
    assert not engine.enforce(
        _principal(roles=[Role.ORG_VIEWER]), "/api/v1/models", "POST"
    )
    assert not engine.enforce(
        _principal(roles=[Role.PLATFORM_VIEWER]), "/api/v1/models", "POST"
    )
    assert not engine.enforce(
        _principal(roles=[Role.PLATFORM_VIEWER]), "/api/v1/models", "GETX"
    )


def test_operator_csv_object_pattern_spans_segments(tmp_path):
    policy = tmp_path / "policy.csv"
    policy.write_text("p, org_viewer, /api/*, GET\n")
    engine = RBACEngine(policy_path=str(policy))
    viewer = _principal(roles=[Role.ORG_VIEWER])
    assert engine.enforce(viewer, "/api/v1/models", "GET")
    assert not engine.enforce(viewer, "/api/v1/models", "GETX")
