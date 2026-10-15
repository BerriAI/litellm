from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value
from pydantic import JsonValue


def _chat(gateway: Gateway, key: str, model: str) -> int:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hello"}]}, key=key
    )
    if response.status_code == 429:
        assert object_value(object_value(response.json())["error"])["type"] == "model_throttled", response.text
    else:
        assert response.status_code == 200, response.text
        assert object_value(response.json())["choices"], response.text
    return response.status_code


def _key_info(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    return object_value(gateway.get("/key/info", {"key": key})["info"])


def test_spend_crosses_thresholds_then_reset_restores_access(gateway: Gateway, scenario: Scenario) -> None:
    premium: Final = scenario.model(
        model="openai/gpt-6-luna", input_cost_per_token=0.0015625, output_cost_per_token=0.0015625
    )
    standard: Final = scenario.model(
        model="openai/gpt-6-luna", input_cost_per_token=0.0015625, output_cost_per_token=0.0015625
    )
    economy: Final = scenario.model(
        model="openai/gpt-6-luna", input_cost_per_token=0.0015625, output_cost_per_token=0.0015625
    )
    config: Final[dict[str, JsonValue]] = {
        "thresholds": [{"spend": 0.0625, "models": [premium]}, {"spend": 0.125, "models": [standard]}]
    }
    key: Final = scenario.key(models=[premium, standard, economy], max_budget=0.1875, model_throttling=config)
    assert _key_info(gateway, key)["model_throttling"] == config
    assert _chat(gateway, key, premium) == 200
    eventually(lambda: _key_info(gateway, key)["spend"], lambda value: value == 0.0625, seconds=70)
    assert _chat(gateway, key, premium) == 429
    assert _chat(gateway, key, standard) == 200
    eventually(lambda: _key_info(gateway, key)["spend"], lambda value: value == 0.125, seconds=70)
    assert _chat(gateway, key, premium) == 429
    assert _chat(gateway, key, standard) == 429
    assert _chat(gateway, key, economy) == 200
    eventually(lambda: _key_info(gateway, key)["spend"], lambda value: value == 0.1875, seconds=70)
    denied: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": economy, "messages": [{"role": "user", "content": "hello"}]}, key=key
    )
    assert denied.status_code == 422, denied.text
    assert "budget" in denied.text.lower(), denied.text
    gateway.post(f"/key/{key}/reset_spend", {"reset_to": 0})
    assert _chat(gateway, key, premium) == 200


@pytest.mark.parametrize("scope", ("key", "team", "user", "budget"))
def test_scope_rules_survive_readback_and_can_be_disabled(gateway: Gateway, scenario: Scenario, scope: str) -> None:
    premium: Final = scenario.model(
        model="openai/gpt-6-luna", input_cost_per_token=0.0015625, output_cost_per_token=0.0015625
    )
    economy: Final = scenario.model(model="openai/gpt-6-luna", input_cost_per_token=0, output_cost_per_token=0)
    config: Final[dict[str, JsonValue]] = {"thresholds": [{"spend": 0.0625, "models": [premium]}]}
    if scope == "team":
        team = scenario.team(models=[premium, economy], max_budget=1, model_throttling=config)
        key = scenario.key(team_id=team, models=[premium, economy])
        path = "/team/update"
        identity = {"team_id": team}
    elif scope == "user":
        user = scenario.user(models=[premium, economy], max_budget=1, model_throttling=config)
        key = scenario.key(user_id=user, models=[premium, economy])
        path = "/user/update"
        identity = {"user_id": user}
    elif scope == "budget":
        budget = scenario.budget(max_budget=1, model_throttling=config)
        key = scenario.key(models=[premium, economy], budget_id=budget)
        path = "/key/update"
        identity = {"key": key, "budget_id": None}
    else:
        key = scenario.key(models=[premium, economy], max_budget=1, model_throttling=config)
        path = "/key/update"
        identity = {"key": key}
    assert _chat(gateway, key, premium) == 200
    eventually(lambda: _key_info(gateway, key)["spend"], lambda value: value == 0.0625, seconds=70)
    assert _chat(gateway, key, premium) == 429
    assert _chat(gateway, key, economy) == 200
    gateway.post(path, {**identity, "model_throttling": None})
    assert _chat(gateway, key, premium) == 200
