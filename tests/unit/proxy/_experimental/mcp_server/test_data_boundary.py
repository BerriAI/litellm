import pytest

from litellm.proxy._experimental.mcp_server.data_boundary import (
    find_data_boundary_violation,
    violation_unless_any_source_admits,
)


@pytest.mark.parametrize(
    "policies",
    ((), (("key", None),), (("key", []), ("team", None))),
)
def test_unset_or_empty_policy_admits_any_server(policies):
    assert find_data_boundary_violation("crm", "us", policies) is None
    assert find_data_boundary_violation("crm", None, policies) is None


def test_server_inside_allowed_boundary_is_admitted_case_insensitively():
    assert find_data_boundary_violation("docs", " EU ", (("key", ["eu", "us"]),)) is None


def test_cross_boundary_server_reports_server_boundary_and_allowed_set():
    violation = find_data_boundary_violation("crm", "US-East", (("key", ["eu", " EU", "eu-west"]),))

    assert violation is not None
    assert dict(violation.to_detail()) == {
        "error": (
            "MCP data boundary violation: server 'crm' is in data boundary 'us-east', "
            "but the key policy only permits data boundaries ['eu', 'eu-west']. "
            "Contact proxy admin to change the data boundary policy."
        ),
        "code": "mcp_data_boundary_violation",
        "server_name": "crm",
        "server_data_boundary": "us-east",
        "allowed_data_boundaries": ("eu", "eu-west"),
        "policy_source": "key",
    }


@pytest.mark.parametrize("server_boundary", (None, "", "   "))
def test_unlabeled_server_is_outside_every_configured_boundary(server_boundary):
    violation = find_data_boundary_violation("web", server_boundary, (("key", ["eu"]),))

    assert violation is not None
    assert violation.server_data_boundary is None
    assert "server 'web' is in no declared data boundary" in violation.reason


def test_team_policy_still_applies_when_key_policy_admits_the_server():
    violation = find_data_boundary_violation("crm", "us", (("key", ["us", "eu"]), ("team", ["eu"])))

    assert violation is not None
    assert violation.policy_source == "team"
    assert violation.allowed_data_boundaries == ("eu",)


def test_any_admitting_source_admits_otherwise_first_violation_is_reported():
    first = find_data_boundary_violation("crm", "us", (("team", ["eu"]),))
    second = find_data_boundary_violation("crm", "us", (("user", ["apac"]),))

    assert violation_unless_any_source_admits([]) is None
    assert violation_unless_any_source_admits([first, None]) is None
    assert violation_unless_any_source_admits([first, second]) is first
