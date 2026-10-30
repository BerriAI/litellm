"""A team's own wildcard deployment serves that team's keys even when a proxy-wide wildcard of the
same pattern exists, and the proxy-wide one serves every other key (#48490)."""

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario

pytestmark: Final = pytest.mark.timeout(120)


def _wildcard(scenario: Scenario, pattern: str, *, team_id: str | None) -> str:
    identity: Final = f"wildcard-{uuid.uuid4().hex}"
    scenario.gateway.post(
        "/model/new",
        {
            "model_name": pattern,
            "litellm_params": {
                "model": "openai/*",
                "api_key": f"provider-key-{identity}",
                "api_base": f"{scenario.gateway.upstream_url}/v1",
            },
            "model_info": {"id": identity, **({"team_id": team_id} if team_id is not None else {})},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _served_by(gateway: Gateway, key: str, model: str) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "wildcard precedence control"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    return response.headers["x-litellm-model-id"]


def test_a_teams_wildcard_serves_its_keys_before_a_proxy_wide_wildcard_of_the_same_pattern(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        byok_team: Final = scenario.team(models=[])
        other_team: Final = scenario.team(models=[])
        pattern: Final = f"byok-{uuid.uuid4().hex}/*"
        team_wildcard: Final = _wildcard(scenario, pattern, team_id=byok_team)
        shared_wildcard: Final = _wildcard(scenario, pattern, team_id=None)
        model: Final = pattern.replace("*", "gpt-4o-mini")

        served: Final = {
            "byok team key": _served_by(gateway, scenario.key(team_id=byok_team), model),
            "other team key": _served_by(gateway, scenario.key(team_id=other_team), model),
            "key without a team": _served_by(gateway, scenario.key(), model),
        }

        assert served == {
            "byok team key": team_wildcard,
            "other team key": shared_wildcard,
            "key without a team": shared_wildcard,
        }
