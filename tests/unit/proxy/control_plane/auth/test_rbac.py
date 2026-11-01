import pytest

from litellm.proxy.control_plane.auth.rbac import (
    ADMIN_ROLE,
    DEFAULT_ROLES,
    EDITOR_ROLE,
    USAGE_VIEWER_ROLE,
    VIEWER_ROLE,
    DataplaneBinding,
    RoleDefinition,
    RolePolicy,
    Scope,
    UnknownParent,
    UnknownScope,
    build_policy,
    build_principal,
    clamp_scopes,
    default_policy,
    parse_role_definition,
    parse_scopes,
)


def _principal(policy: RolePolicy, roles: frozenset[str], ids: frozenset[str] = frozenset()):
    return build_principal(policy, "u", None, roles, None, ids, "oidc")


@pytest.mark.parametrize("role", list(DEFAULT_ROLES))
@pytest.mark.parametrize("scope", list(Scope))
def test_every_default_role_scope_cell_matches_the_policy_table(role: str, scope: Scope) -> None:
    assert _principal(default_policy(), frozenset({role})).has_scope(scope) is (scope in DEFAULT_ROLES[role])


def test_default_tiers_only_admin_manages_access_only_editor_writes_and_usage_viewer_sees_only_usage() -> None:
    assert DEFAULT_ROLES[ADMIN_ROLE] == frozenset(Scope)
    assert {Scope.DATAPLANES_WRITE, Scope.ROLES_WRITE, Scope.AUDIT_READ}.isdisjoint(DEFAULT_ROLES[EDITOR_ROLE])
    assert Scope.PROXY_WRITE in DEFAULT_ROLES[EDITOR_ROLE] and Scope.PROXY_WRITE not in DEFAULT_ROLES[VIEWER_ROLE]
    assert DEFAULT_ROLES[USAGE_VIEWER_ROLE] == frozenset({Scope.USAGE_READ})


def test_roles_union_their_grants() -> None:
    principal = _principal(default_policy(), frozenset({USAGE_VIEWER_ROLE, VIEWER_ROLE}))
    assert principal.scopes == DEFAULT_ROLES[VIEWER_ROLE]


def test_claimed_scopes_narrow_but_never_widen_the_role_grant() -> None:
    granted = DEFAULT_ROLES[VIEWER_ROLE]
    assert clamp_scopes(granted, frozenset({Scope.USAGE_READ, Scope.PROXY_WRITE})) == frozenset({Scope.USAGE_READ})
    assert clamp_scopes(granted, None) == granted


def test_a_scope_claim_naming_only_unknown_scopes_grants_nothing() -> None:
    """A present claim is always honored, so a token scoped to foreign scopes cannot fall back to the role grant."""
    assert clamp_scopes(DEFAULT_ROLES[ADMIN_ROLE], frozenset()) == frozenset()
    assert clamp_scopes(DEFAULT_ROLES[ADMIN_ROLE], parse_scopes(("openid", "profile"))) == frozenset()


def test_custom_role_overrides_a_default_and_can_inherit() -> None:
    policy = build_policy(
        {
            "viewer": RoleDefinition(scopes=frozenset({Scope.FLEET_READ}), inherits=frozenset()),
            "finops": RoleDefinition(scopes=frozenset({Scope.AUDIT_READ}), inherits=frozenset({"usage_viewer"})),
            "lead": RoleDefinition(scopes=frozenset(), inherits=frozenset({"finops", "editor"})),
        }
    )
    assert isinstance(policy, RolePolicy)
    assert policy.scopes_for("viewer") == frozenset({Scope.FLEET_READ})
    assert policy.scopes_for("finops") == frozenset({Scope.AUDIT_READ, Scope.USAGE_READ})
    assert policy.scopes_for("lead") == DEFAULT_ROLES[EDITOR_ROLE] | {Scope.AUDIT_READ, Scope.USAGE_READ}
    assert policy.scopes_for("admin") == frozenset(Scope)


def test_defaults_can_be_switched_off_so_only_custom_roles_exist() -> None:
    policy = build_policy({"ops": RoleDefinition(frozenset({Scope.USAGE_READ}), frozenset())}, include_defaults=False)
    assert isinstance(policy, RolePolicy)
    assert policy.names == frozenset({"ops"})
    assert policy.known(("admin", "ops")) == frozenset({"ops"})


def test_inheriting_an_unknown_role_is_an_error_value() -> None:
    policy = build_policy({"x": RoleDefinition(frozenset(), frozenset({"nope"}))})
    assert policy == UnknownParent(role="x", parent="nope")


def test_inheritance_cycles_terminate() -> None:
    policy = build_policy(
        {
            "a": RoleDefinition(frozenset({Scope.USAGE_READ}), frozenset({"b"})),
            "b": RoleDefinition(frozenset({Scope.FLEET_READ}), frozenset({"a"})),
        }
    )
    assert isinstance(policy, RolePolicy)
    assert policy.scopes_for("a") == policy.scopes_for("b") == frozenset({Scope.USAGE_READ, Scope.FLEET_READ})


def test_role_definition_rejects_unknown_scopes() -> None:
    assert parse_role_definition("r", ("usage:read", "nope"), ()) == UnknownScope(role="r", scope="nope")
    parsed = parse_role_definition("r", ("usage:read",), ("viewer",))
    assert parsed == RoleDefinition(scopes=frozenset({Scope.USAGE_READ}), inherits=frozenset({"viewer"}))


@pytest.mark.parametrize("ids", [frozenset(), frozenset({"*"}), frozenset({"*", "eu"})])
def test_no_binding_or_star_is_fleet_wide(ids: frozenset[str]) -> None:
    principal = _principal(default_policy(), frozenset({VIEWER_ROLE}), ids)
    assert principal.dataplanes == DataplaneBinding(all=True, ids=frozenset())
    assert principal.dataplanes.allows("anything")


def test_explicit_binding_limits_dataplanes_regardless_of_role() -> None:
    principal = _principal(default_policy(), frozenset({ADMIN_ROLE}), frozenset({"eu"}))
    assert principal.dataplanes.allows("eu") and not principal.dataplanes.allows("us")


@pytest.mark.parametrize(
    ("binding", "requested", "expected"),
    [
        (DataplaneBinding(True, frozenset()), None, None),
        (DataplaneBinding(True, frozenset()), frozenset({"a"}), frozenset({"a"})),
        (DataplaneBinding(False, frozenset({"a", "b"})), None, frozenset({"a", "b"})),
        (DataplaneBinding(False, frozenset({"a", "b"})), frozenset({"b", "c"}), frozenset({"b"})),
    ],
)
def test_restrict_intersects_the_request_with_the_binding(
    binding: DataplaneBinding, requested: frozenset[str] | None, expected: frozenset[str] | None
) -> None:
    assert binding.restrict(requested) == expected


def test_unknown_scopes_are_dropped_not_raised() -> None:
    assert parse_scopes(("usage:read", "openid", "profile")) == frozenset({Scope.USAGE_READ})
    assert parse_scopes(None) is None
