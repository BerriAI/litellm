"""
LIT-3254 — Unit tests for `_maybe_inherit_caller_limits_for_self_served_team`,
the helper that hardens self-service team creation against the Veria #2
bypass (a non-admin creating an unlimited team and then minting team keys
that bypass their personal models/tpm/rpm caps).
"""
from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
from litellm.proxy.management_endpoints.team_endpoints import (
    _maybe_inherit_caller_limits_for_self_served_team,
)


def _caller(**kwargs):
    return UserAPIKeyAuth(api_key="sk-test", user_id="u1", **kwargs)


# -- Inheritance happy paths ------------------------------------------------


def test_inherits_caller_models_when_team_models_unset():
    caller = _caller(models=["gpt-4o-mini"])
    data = NewTeamRequest(team_alias="t")
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.models == ["gpt-4o-mini"]


def test_inherits_caller_models_when_team_models_explicitly_empty():
    """Empty list silently means "all models" on the team — the bypass."""
    caller = _caller(models=["gpt-4o-mini", "gpt-4o"])
    data = NewTeamRequest(team_alias="t", models=[])
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert sorted(data.models) == ["gpt-4o", "gpt-4o-mini"]


def test_inherits_tpm_limit_when_team_unset():
    caller = _caller(tpm_limit=1000)
    data = NewTeamRequest(team_alias="t")
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.tpm_limit == 1000


def test_inherits_rpm_limit_when_team_unset():
    caller = _caller(rpm_limit=10)
    data = NewTeamRequest(team_alias="t")
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.rpm_limit == 10


# -- Explicit team values are NOT overwritten -------------------------------


def test_explicit_team_models_subset_unchanged():
    """Caller has [a,b], team requests [a]. We must not widen to [a,b]."""
    caller = _caller(models=["a", "b"])
    data = NewTeamRequest(team_alias="t", models=["a"])
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.models == ["a"]


def test_explicit_tpm_unchanged():
    caller = _caller(tpm_limit=5000)
    data = NewTeamRequest(team_alias="t", tpm_limit=1000)
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.tpm_limit == 1000


def test_explicit_rpm_unchanged():
    caller = _caller(rpm_limit=100)
    data = NewTeamRequest(team_alias="t", rpm_limit=50)
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.rpm_limit == 50


# -- Unrestricted callers stay unrestricted ---------------------------------


def test_unrestricted_caller_leaves_team_unrestricted():
    """Caller has no model restriction and no tpm/rpm cap — team stays open."""
    caller = _caller(models=[], tpm_limit=None, rpm_limit=None)
    data = NewTeamRequest(team_alias="t")
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.models in (None, [])
    assert data.tpm_limit is None
    assert data.rpm_limit is None


def test_caller_with_only_one_restriction_only_inherits_that():
    """Caller restricts models but not tpm — only models should be inherited."""
    caller = _caller(models=["gpt-4o-mini"], tpm_limit=None, rpm_limit=None)
    data = NewTeamRequest(team_alias="t")
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.models == ["gpt-4o-mini"]
    assert data.tpm_limit is None
    assert data.rpm_limit is None


# -- max_budget is intentionally NOT auto-inherited ------------------------


def test_max_budget_is_not_inherited():
    """
    Documented carve-out: we deliberately do NOT auto-populate `max_budget`
    from the caller. The existing `_check_user_team_limits` already rejects
    a *too-large* explicit `max_budget`. Auto-inheriting a None caller
    budget would silently produce an unlimited-budget team; admins should
    set per-team budgets via `default_team_settings` or by editing the
    team. This test guards the carve-out.
    """
    caller = _caller(max_budget=100.0)
    data = NewTeamRequest(team_alias="t")
    _maybe_inherit_caller_limits_for_self_served_team(data, caller)
    assert data.max_budget is None  # left alone on purpose
