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
