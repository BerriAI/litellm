import pytest

from litellm.proxy.auth.v2.authz.policy_admin import (
    PolicyValidationError,
    make_assignment_rule,
    make_permission_rule,
    normalize_object,
)


def test_permission_rule_shape():
    rule = make_permission_rule("model_reader", "model", "read")
    assert rule == ["p", "role:model_reader", "*", "model:*", "read", "allow"]


def test_permission_rule_with_resource_id_and_domain():
    rule = make_permission_rule(
        "gpt_owner", "model", "write", domain="team:eng", resource_id="gpt-4o"
    )
    assert rule == ["p", "role:gpt_owner", "team:eng", "model:gpt-4o", "write", "allow"]


def test_role_prefix_is_idempotent():
    assert make_permission_rule("role:x", "model", "read")[1] == "role:x"


def test_deny_effect_is_supported():
    assert make_permission_rule("r", "model", "read", effect="deny")[-1] == "deny"


def test_invalid_action_is_rejected():
    with pytest.raises(PolicyValidationError):
        make_permission_rule("r", "model", "execute")


def test_invalid_effect_is_rejected():
    with pytest.raises(PolicyValidationError):
        make_permission_rule("r", "model", "read", effect="maybe")


def test_empty_resource_is_rejected():
    with pytest.raises(PolicyValidationError):
        make_permission_rule("r", "", "read")


def test_empty_role_is_rejected():
    with pytest.raises(PolicyValidationError):
        make_permission_rule("  ", "model", "read")


def test_normalize_object_wildcard_default():
    assert normalize_object("team", None) == "team:*"
    assert normalize_object("team", "eng") == "team:eng"


def test_assignment_rule_for_user_and_team():
    assert make_assignment_rule("user", "u1", "admin") == ["g", "user:u1", "role:admin"]
    assert make_assignment_rule("team", "eng", "role:x") == ["g", "team:eng", "role:x"]


def test_global_assignment_when_domain_is_wildcard_or_absent():
    assert make_assignment_rule("user", "u1", "admin", domain=None)[0] == "g"
    assert make_assignment_rule("user", "u1", "admin", domain="*")[0] == "g"


def test_domain_scoped_assignment_is_a_g3_rule():
    rule = make_assignment_rule("user", "u1", "team_admin", domain="team:eng")
    assert rule == ["g3", "user:u1", "role:team_admin", "team:eng"]


def test_assignment_rejects_unknown_subject_type():
    with pytest.raises(PolicyValidationError):
        make_assignment_rule("org", "o1", "admin")


def test_assignment_requires_subject_id():
    with pytest.raises(PolicyValidationError):
        make_assignment_rule("user", "", "admin")
