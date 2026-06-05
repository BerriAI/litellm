from litellm.proxy.auth.v2.enforcer import CasbinEnforcer

READER_POLICY = ["role:model_reader", "*", "model:*", "read", "allow"]
ADMIN_POLICY = ["role:proxy_admin", "*", "*", "*", "allow"]
TEAM_POLICY = ["role:team_eng", "team:eng", "model:*", "write", "allow"]


def _enforcer(policies, groupings):
    return CasbinEnforcer(policies, groupings)


def test_role_grants_only_its_action():
    e = _enforcer([READER_POLICY], [["user:alice", "role:model_reader"]])
    assert e.enforce("user:alice", "*", "model:gpt4", "read") is True
    # The whole point of granular RBAC: read does NOT imply write.
    assert e.enforce("user:alice", "*", "model:gpt4", "write") is False
    assert e.enforce("user:alice", "*", "model:gpt4", "delete") is False


def test_subject_without_role_is_denied():
    e = _enforcer([READER_POLICY], [["user:alice", "role:model_reader"]])
    assert e.enforce("user:bob", "*", "model:gpt4", "read") is False


def test_wildcard_admin_can_do_everything():
    e = _enforcer([ADMIN_POLICY], [["user:root", "role:proxy_admin"]])
    for action in ("read", "write", "delete"):
        assert e.enforce("user:root", "*", "model:anything", action) is True


def test_specific_resource_id_is_scoped():
    policy = ["role:gpt_owner", "*", "model:gpt-4o", "write", "allow"]
    e = _enforcer([policy], [["user:carol", "role:gpt_owner"]])
    # Permitted on the exact id...
    assert e.enforce("user:carol", "*", "model:gpt-4o", "write") is True
    # ...denied on a different id (per-resource granularity).
    assert e.enforce("user:carol", "*", "model:claude", "write") is False


def test_domain_scoped_policy_only_applies_in_its_domain():
    e = _enforcer([TEAM_POLICY], [["user:dan", "role:team_eng"]])
    assert e.enforce("user:dan", "team:eng", "model:gpt4", "write") is True
    # Same subject+role, wrong domain -> denied.
    assert e.enforce("user:dan", "team:sales", "model:gpt4", "write") is False


def test_explicit_deny_overrides_allow():
    e = _enforcer(
        [
            ["role:model_reader", "*", "model:*", "read", "allow"],
            ["role:model_reader", "*", "model:secret", "read", "deny"],
        ],
        [["user:eve", "role:model_reader"]],
    )
    assert e.enforce("user:eve", "*", "model:public", "read") is True
    assert e.enforce("user:eve", "*", "model:secret", "read") is False


def test_empty_policy_denies_everything():
    e = _enforcer([], [])
    assert e.enforce("user:anyone", "*", "model:gpt4", "read") is False


def test_model_call_permission_via_role_with_wildcard():
    # Calling a model is the `call` action on `model:<id>`, granted by a role.
    e = _enforcer(
        [["role:gpt_caller", "*", "model:gpt-*", "call", "allow"]],
        [["user:alice", "role:gpt_caller"]],
    )
    assert e.enforce("user:alice", "*", "model:gpt-4o", "call") is True
    assert e.enforce("user:alice", "*", "model:gpt-4o-mini", "call") is True
    # A model outside the wildcard is denied.
    assert e.enforce("user:alice", "*", "model:claude-3", "call") is False


def test_model_call_can_be_granted_directly_to_a_subject():
    # No role needed: grant the `call` permission straight to the key/user.
    e = _enforcer(
        [["user:svc-key", "*", "model:o1", "call", "allow"]],
        [],
    )
    assert e.enforce("user:svc-key", "*", "model:o1", "call") is True
    assert e.enforce("user:svc-key", "*", "model:o3", "call") is False


def test_model_call_denied_without_any_policy():
    # Clean slate: with no grant, a key cannot call any model.
    e = _enforcer([], [])
    assert e.enforce("user:anyone", "*", "model:gpt-4o", "call") is False


def test_proxy_admin_can_call_any_model():
    e = _enforcer([ADMIN_POLICY], [["user:root", "role:proxy_admin"]])
    assert e.enforce("user:root", "*", "model:anything", "call") is True


def test_resource_grouping_grants_access_to_a_named_group():
    e = CasbinEnforcer(
        policies=[["role:grp_mgr", "*", "group:prod", "write", "allow"]],
        groupings=[["user:al", "role:grp_mgr"]],
        resource_groupings=[
            ["model:gpt-4o", "group:prod"],
            ["model:claude", "group:prod"],
        ],
    )
    assert e.enforce("user:al", "*", "model:gpt-4o", "write") is True
    assert e.enforce("user:al", "*", "model:claude", "write") is True
    # A model not in the group is denied even with the same action.
    assert e.enforce("user:al", "*", "model:o1", "write") is False


def test_direct_object_matching_still_works_with_grouping_enabled():
    # No g2 rules: keyMatch / exact behavior must be unchanged.
    e = _enforcer([READER_POLICY], [["user:alice", "role:model_reader"]])
    assert e.enforce("user:alice", "*", "model:gpt4", "read") is True
    assert e.enforce("user:alice", "*", "model:gpt4", "write") is False


def test_domain_scoped_role_applies_only_in_its_domain():
    e = CasbinEnforcer(
        policies=[["role:team_admin", "*", "team:*", "manage", "allow"]],
        groupings=[],
        resource_groupings=None,
        domain_groupings=[["user:dana", "role:team_admin", "team:eng"]],
    )
    assert e.enforce("user:dana", "team:eng", "team:eng", "manage") is True
    # Same user+role, different domain -> denied.
    assert e.enforce("user:dana", "team:sales", "team:sales", "manage") is False


def test_global_and_domain_roles_coexist():
    e = CasbinEnforcer(
        policies=[
            ["role:reader", "*", "model:*", "read", "allow"],
            ["role:team_admin", "*", "team:*", "manage", "allow"],
        ],
        groupings=[["user:gina", "role:reader"]],
        domain_groupings=[["user:gina", "role:team_admin", "team:eng"]],
    )
    # Global reader role works anywhere.
    assert e.enforce("user:gina", "team:sales", "model:x", "read") is True
    # Domain-scoped admin role only in team:eng.
    assert e.enforce("user:gina", "team:eng", "team:eng", "manage") is True
    assert e.enforce("user:gina", "team:sales", "team:sales", "manage") is False
