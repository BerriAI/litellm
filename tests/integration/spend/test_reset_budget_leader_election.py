import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from redis import Redis

RESET_LEASE_KEY: Final = "cronjob_lock:reset_budget_job"
PEER_POD_LEASE: Final = json.dumps("integration-peer-pod-holding-the-reset-lease")
FAST_RESET_TICK: Final = MappingProxyType(
    {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2"}
)


def _team_spend(team: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
    assert len(rows) == 1, rows
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float)), rows
    return float(spend)


def _make_team_budget_due(team: str, spend: float) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            "UPDATE \"LiteLLM_TeamTable\" SET spend = %s, budget_reset_at = now() - interval '1 day' "
            "WHERE team_id = %s",
            (spend, team),
        )


@pytest.mark.covers("spend.budget_reset.one_pod_sweeps_per_tick")
def test_reset_sweep_skips_ticks_while_another_pod_holds_the_lease_and_resumes_after_release(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        gateway.scenario() as scenario,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        model: Final = scenario.model(input_cost_per_token=0.0, output_cost_per_token=0.0)
        team: Final = scenario.team(max_budget=1.0, budget_duration="30d")
        key: Final = scenario.key(team_id=team)
        _make_team_budget_due(team, spend=0.5)
        assert _team_spend(team) == 0.5
        eventually(
            lambda: cache.set(RESET_LEASE_KEY, PEER_POD_LEASE, ex=120, nx=True),
            lambda claimed: claimed is True,
            seconds=60,
        )
        try:
            with owned_proxy(gateway, tmp_path, FAST_RESET_TICK) as replica:
                response: Final = replica.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "classification request"}]},
                    key=key,
                )
                assert response.status_code == 200, response.text
                held: Final = eventually(
                    lambda: _team_spend(team), lambda spend: spend != 0.5, seconds=8, return_last_on_timeout=True
                )
                assert held == 0.5, f"team {team} was swept while another pod held the reset lease: spend={held}"
                assert cache.get(RESET_LEASE_KEY) == PEER_POD_LEASE.encode()
                cache.delete(RESET_LEASE_KEY)
                swept: Final = eventually(lambda: _team_spend(team), lambda spend: spend == 0.0, seconds=15)
                assert swept == 0.0
                eventually(lambda: cache.get(RESET_LEASE_KEY), lambda value: value is None, seconds=15)
        finally:
            cache.delete(RESET_LEASE_KEY)
