import pytest

from litellm.proxy.utils import _is_valid_team_configs


def normalize(value):
    return value


def test_is_valid_team_configs_happy_path_allowed_model_mutates_config():
    team_config = {"models": ["gpt-4o", "gpt-4o-mini"], "max_budget": 100.0}
    request_data = {"model": "gpt-4o"}
    snapshot = {
        "result": _is_valid_team_configs(
            team_id="team-1",
            team_config=team_config,
            request_data=request_data,
        ),
        "models_popped": "models" not in team_config,
        "remaining_keys": sorted(team_config.keys()),
    }
    assert snapshot == {
        "result": None,
        "models_popped": True,
        "remaining_keys": ["max_budget"],
    }


def test_is_valid_team_configs_no_models_key_is_noop():
    team_config = {"max_budget": 100.0, "tpm_limit": 1000}
    request_data = {"model": "anything"}
    snapshot = {
        "result": _is_valid_team_configs(
            team_id="team-1",
            team_config=team_config,
            request_data=request_data,
        ),
        "team_config": team_config,
        "request_data": request_data,
    }
    assert snapshot == {
        "result": None,
        "team_config": {"max_budget": 100.0, "tpm_limit": 1000},
        "request_data": {"model": "anything"},
    }


def test_is_valid_team_configs_short_circuits_when_team_id_none():
    team_config = {"models": ["only-this"]}
    snapshot = {
        "result": _is_valid_team_configs(
            team_id=None,
            team_config=team_config,
            request_data={"model": "anything-else"},
        ),
        "team_config_unchanged": team_config,
        "models_key_preserved": "models" in team_config,
    }
    assert snapshot == {
        "result": None,
        "team_config_unchanged": {"models": ["only-this"]},
        "models_key_preserved": True,
    }


def test_is_valid_team_configs_raises_on_model_not_in_team_models():
    team_config = {"models": ["gpt-4o"]}
    request_data = {"model": "claude-haiku"}
    with pytest.raises(Exception) as exc_info:
        _is_valid_team_configs(
            team_id="team-1",
            team_config=team_config,
            request_data=request_data,
        )
    assert "Invalid model for team team-1" in str(exc_info.value)
    assert "claude-haiku" in str(exc_info.value)
