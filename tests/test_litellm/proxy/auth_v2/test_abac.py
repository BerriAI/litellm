from __future__ import annotations

import pytest

from litellm.proxy.auth_v2.authorization import ABACEngine, ProtectedResource, Role
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Principal,
    PrincipalType,
    TeamIdentity,
)

# The two policies from the design: a role-restricted model allowlist on
# /v1/messages, and a team-restricted MCP tool allowlist on the github server.
POLICIES = """
policies:
  - sub_rule: "'manager' in r_sub.roles"
    obj_rule: "r_obj.endpoint == '/v1/messages' and r_obj.model in ['claude-sonnet-4-6','gpt-4o']"
    act: "POST"
  - sub_rule: "'eng' in r_sub.teams"
    obj_rule: "r_obj.mcp_server == 'github' and r_obj.mcp_tool in ['search','read_file']"
    act: "POST|GET"
"""


def _principal(*, roles=None, teams=None, claims=None, scopes=None) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject="u1",
        auth_method=AuthMethod.OIDC,
        roles=roles or [],
        teams=teams or [],
        claims=claims or {},
        scopes=scopes or [],
    )


def _manager() -> Principal:
    return _principal(claims={"roles": ["manager"]})


def _eng() -> Principal:
    return _principal(teams=[TeamIdentity(id="t1", name="eng")])


def _engine_from(tmp_path, policy_text: str) -> ABACEngine:
    path = tmp_path / "abac.yaml"
    path.write_text(policy_text)
    return ABACEngine(policy_path=str(path))


@pytest.fixture
def engine(tmp_path) -> ABACEngine:
    return _engine_from(tmp_path, POLICIES)


# --------------------------------------------------------------------------- #
# Model allowlist policy: each attribute is an independent gate
# --------------------------------------------------------------------------- #


def test_manager_allowed_model_is_permitted(engine):
    assert engine.decide(
        _manager(),
        ProtectedResource(endpoint="/v1/messages", model="gpt-4o", method="POST"),
    )


def test_manager_unlisted_model_is_denied(engine):
    # set membership, not just "manager is allowed"; also the exact case that the
    # CSV FileAdapter silently *allowed* (quotes leaking into the eval'd rule)
    assert not engine.decide(
        _manager(),
        ProtectedResource(
            endpoint="/v1/messages", model="claude-opus-4-8", method="POST"
        ),
    )


def test_non_manager_is_denied_allowed_model(engine):
    assert not engine.decide(
        _principal(),
        ProtectedResource(endpoint="/v1/messages", model="gpt-4o", method="POST"),
    )


def test_manager_wrong_action_is_denied(engine):
    assert not engine.decide(
        _manager(),
        ProtectedResource(endpoint="/v1/messages", model="gpt-4o", method="GET"),
    )


def test_manager_wrong_endpoint_is_denied(engine):
    assert not engine.decide(
        _manager(),
        ProtectedResource(
            endpoint="/v1/chat/completions", model="gpt-4o", method="POST"
        ),
    )


def test_manager_without_model_is_denied(engine):
    # a model-restricted rule must not match a request that carries no model
    assert not engine.decide(
        _manager(), ProtectedResource(endpoint="/v1/messages", method="POST")
    )


# --------------------------------------------------------------------------- #
# MCP tool allowlist policy: team membership x server x tool x action
# --------------------------------------------------------------------------- #


def test_eng_listed_tool_is_permitted(engine):
    assert engine.decide(
        _eng(),
        ProtectedResource(mcp_server="github", mcp_tool="search", method="POST"),
    )
    assert engine.decide(
        _eng(),
        ProtectedResource(mcp_server="github", mcp_tool="read_file", method="GET"),
    )


def test_eng_unlisted_tool_is_denied(engine):
    assert not engine.decide(
        _eng(),
        ProtectedResource(mcp_server="github", mcp_tool="delete_repo", method="POST"),
    )


def test_eng_wrong_server_is_denied(engine):
    assert not engine.decide(
        _eng(),
        ProtectedResource(mcp_server="gitlab", mcp_tool="search", method="POST"),
    )


def test_non_eng_team_is_denied(engine):
    sales = _principal(teams=[TeamIdentity(id="t2", name="sales")])
    assert not engine.decide(
        sales,
        ProtectedResource(mcp_server="github", mcp_tool="search", method="POST"),
    )


def test_unrelated_policy_row_does_not_poison_decision(engine):
    # the manager request must be allowed even though the eng (team) policy row
    # is also evaluated against a principal with no teams
    assert engine.decide(
        _manager(),
        ProtectedResource(endpoint="/v1/messages", model="gpt-4o", method="POST"),
    )


# --------------------------------------------------------------------------- #
# Default deny and claim-access safety
# --------------------------------------------------------------------------- #


def test_no_policy_denies_everything():
    engine = ABACEngine()
    assert not engine.decide(
        _manager(),
        ProtectedResource(endpoint="/v1/messages", model="gpt-4o", method="POST"),
    )


def test_missing_claim_does_not_raise_and_denies(tmp_path):
    policy = """
policies:
  - sub_rule: "r_sub.claims['department'] == 'eng'"
    obj_rule: "r_obj.endpoint == '/reports'"
    act: "GET"
"""
    engine = _engine_from(tmp_path, policy)
    assert engine.decide(
        _principal(claims={"department": "eng"}),
        ProtectedResource(endpoint="/reports", method="GET"),
    )
    # principal lacking the claim: must deny, not raise
    assert not engine.decide(
        _principal(), ProtectedResource(endpoint="/reports", method="GET")
    )


def test_malformed_policy_entry_fails_fast(tmp_path):
    bad = tmp_path / "abac.yaml"
    bad.write_text('policies:\n  - sub_rule: "1 == 1"\n    act: GET\n')
    with pytest.raises(ValueError):
        ABACEngine(policy_path=str(bad))


def test_rule_evaluation_error_fails_closed(tmp_path):
    # an operator referencing an attribute the resource does not carry must deny,
    # not surface the exception
    policy = """
policies:
  - sub_rule: "True"
    obj_rule: "r_obj.nonexistent == 1"
    act: "GET"
"""
    engine = _engine_from(tmp_path, policy)
    assert not engine.decide(
        _principal(), ProtectedResource(endpoint="/x", method="GET")
    )


# --------------------------------------------------------------------------- #
# Authorizer protocol compatibility (path/method enforce, flat has_any_role)
# --------------------------------------------------------------------------- #


def test_enforce_path_method_policy(tmp_path):
    policy = """
policies:
  - sub_rule: "'admin' in r_sub.roles"
    obj_rule: "r_obj.endpoint == '/health'"
    act: "GET"
"""
    engine = _engine_from(tmp_path, policy)
    admin = _principal(claims={"roles": ["admin"]})
    assert engine.enforce(admin, "/health", "GET")
    assert not engine.enforce(admin, "/health", "POST")
    assert not engine.enforce(admin, "/secrets", "GET")
    assert not engine.enforce(_principal(), "/health", "GET")


def test_has_any_role_is_flat_membership(tmp_path):
    engine = _engine_from(tmp_path, POLICIES)
    assert engine.has_any_role(_principal(roles=[Role.ORG_ADMIN]), (Role.ORG_ADMIN,))
    # no hierarchy: platform_admin does not imply org_admin here (use RBACEngine)
    assert not engine.has_any_role(
        _principal(roles=[Role.PLATFORM_ADMIN]), (Role.ORG_ADMIN,)
    )
    assert not engine.has_any_role(_principal(), (Role.TEAM_MEMBER,))
