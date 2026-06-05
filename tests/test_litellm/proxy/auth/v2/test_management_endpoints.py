from litellm.proxy.auth.v2.management_endpoints import rule_to_row_data, row_to_rule


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
