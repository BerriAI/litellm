import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows

TEAM_BUDGET: Final = 0.06


@dataclass(frozen=True, slots=True)
class ExhaustedTeam:
    scenario: Scenario
    upstream: httpx.Client
    model: str
    team_id: str
    key: str


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": f"team budget {uuid.uuid4().hex}"}],
        },
        key=key,
    )


@pytest.fixture
def exhausted(gateway: Gateway) -> Iterator[ExhaustedTeam]:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_id: Final = scenario.team(models=[model], max_budget=TEAM_BUDGET)
        key: Final = scenario.key(team_id=team_id, models=[model], max_budget=1.0)
        first: Final = _chat(gateway, model, key)
        assert first.status_code == 200, first.text
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team_id,)),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= TEAM_BUDGET,
            seconds=70,
        )
        eventually(lambda: _chat(gateway, model, key), lambda response: response.status_code != 200, seconds=30)
        upstream.get("/__observations").raise_for_status()
        yield ExhaustedTeam(scenario, upstream, model, team_id, key)


def test_the_team_budget_blocks_a_key_whose_own_budget_has_room(gateway: Gateway, exhausted: ExhaustedTeam) -> None:
    denied: Final = _chat(gateway, exhausted.model, exhausted.key)
    assert denied.status_code == 422, denied.text
    error: Final = object_value(denied.json()["error"])
    assert error["type"] == "budget_exceeded"
    assert f"Budget has been exceeded! Team={exhausted.team_id}" in str(error["message"])
    assert exhausted.upstream.get("/__observations").json()["requests"] == []


def test_raising_an_exhausted_team_budget_restores_serving(gateway: Gateway, exhausted: ExhaustedTeam) -> None:
    denied: Final = _chat(gateway, exhausted.model, exhausted.key)
    assert denied.status_code == 422, denied.text
    gateway.post("/team/update", {"team_id": exhausted.team_id, "max_budget": 1.0})
    served: Final = tuple(_chat(gateway, exhausted.model, exhausted.key) for _ in range(3))
    assert [response.status_code for response in served] == [200, 200, 200], [response.text for response in served]
    assert len(exhausted.upstream.get("/__observations").json()["requests"]) == 3
