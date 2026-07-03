import pytest

from litellm.proxy.hooks.model_max_budget_limiter import resolve_effective_model_max_budget
from litellm.proxy.management_endpoints.key_management_endpoints import validate_model_max_budget


def test_resolve_effective_model_max_budget_prefers_key_overrides() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "30d"},
        "claude-haiku-4-5": {"budget_limit": 30.0, "time_period": "30d"},
    }
    key_budget = {
        "claude-sonnet-4-6": {"budget_limit": 5.0, "time_period": "7d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=key_budget,
        team_model_max_budget=team_budget,
    )

    assert merged is not None
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 5.0
    assert merged["claude-haiku-4-5"]["budget_limit"] == 30.0


def test_resolve_effective_model_max_budget_member_overrides_team() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }
    member_budget = {
        "claude-sonnet-4-6": {"budget_limit": 50.0, "time_period": "7d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=None,
        team_model_max_budget=team_budget,
        team_member_model_max_budget=member_budget,
    )

    assert merged is not None
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 50.0
    assert merged["claude-sonnet-4-6"]["time_period"] == "7d"


def test_resolve_effective_model_max_budget_three_layer_merge() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
        "claude-haiku-4-5": {"budget_limit": 10.0, "time_period": "1d"},
    }
    member_budget = {
        "claude-sonnet-4-6": {"budget_limit": 40.0, "time_period": "7d"},
    }
    key_budget = {
        "claude-sonnet-4-6": {"budget_limit": 100.0, "time_period": "30d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=key_budget,
        team_model_max_budget=team_budget,
        team_member_model_max_budget=member_budget,
    )

    assert merged is not None
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 100.0
    assert merged["claude-haiku-4-5"]["budget_limit"] == 10.0


def test_resolve_effective_model_max_budget_uses_team_when_key_empty() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "30d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=None,
        team_model_max_budget=team_budget,
    )

    assert merged == team_budget


def test_validate_model_max_budget_without_enterprise_license() -> None:
    validate_model_max_budget(
        {
            "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "30d"},
        }
    )
