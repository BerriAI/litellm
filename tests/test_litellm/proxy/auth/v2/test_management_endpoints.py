import pytest
from fastapi import HTTPException

import litellm.proxy.proxy_server as ps
from litellm.proxy.auth.v2.management_endpoints import (
    _require_auth_v2_enabled,
    row_to_rule,
    rule_to_row_data,
)


def test_policy_admin_surface_is_404_when_auth_v2_disabled(monkeypatch):
    # The router is registered unconditionally, so the per-request guard must hide
    # it on v1 deployments (authz itself is casbin's job once v2 is on).
    monkeypatch.setattr(ps, "general_settings", {}, raising=False)
    with pytest.raises(HTTPException) as exc:
        _require_auth_v2_enabled()
    assert exc.value.status_code == 404


def test_policy_admin_surface_is_available_when_auth_v2_enabled(monkeypatch):
    monkeypatch.setattr(ps, "general_settings", {"auth_version": "v2"}, raising=False)
    _require_auth_v2_enabled()  # must not raise


class _Row:
    def __init__(self, ptype, *values):
        self.ptype = ptype
        for i in range(6):
            setattr(self, f"v{i}", values[i] if i < len(values) else None)


def test_permission_rule_to_row():
    data = rule_to_row_data(["p", "role:x", "*", "model:*", "read", "allow"])
    assert data == {
        "ptype": "p",
        "v0": "role:x",
        "v1": "*",
        "v2": "model:*",
        "v3": "read",
        "v4": "allow",
    }


def test_assignment_rule_to_row():
    data = rule_to_row_data(["g", "user:u1", "role:x"])
    assert data == {"ptype": "g", "v0": "user:u1", "v1": "role:x"}


def test_row_to_rule_trims_empty_columns():
    row = _Row("g", "user:u1", "role:x")
    assert row_to_rule(row) == ["g", "user:u1", "role:x"]


def test_round_trip_permission_rule():
    rule = ["p", "role:x", "team:eng", "model:gpt-4o", "write", "allow"]
    row = _Row(*([rule[0]] + rule[1:]))
    assert row_to_rule(row) == rule
