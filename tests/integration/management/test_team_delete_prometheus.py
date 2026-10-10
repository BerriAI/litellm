"""H7: the Prometheus team members gauge follows ``/team/member_add`` and ``/team/delete``.

An owned single-worker proxy registers the ``prometheus`` callback, so ``GET /metrics/`` serves the
in-process registry (one worker, so no ``PROMETHEUS_MULTIPROC_DIR``). A team with an alias takes three
users in one bulk ``/team/member_add``; the ``litellm_team_members_metric`` series carrying that team's
id then reads 3.0. ``/team/delete`` re-emits the gauge with an empty roster instead of dropping the
series, so the same series afterwards reads 0.0.

``disable_auto_add_proxy_admin_to_teams`` is on for the owned proxy: a master-key ``/team/new``
otherwise seeds the roster with ``default_user_id`` and the gauge would read 4.0 after three adds.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process

METRIC: Final = "litellm_team_members_metric"
METRICS_ROUTE: Final = "/metrics/"
MEMBERS: Final = 3
TEAM_SQL: Final = 'SELECT team_id, members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s'


def _prometheus_config(tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["callbacks"] = ["prometheus"]
    config["general_settings"]["disable_auto_add_proxy_admin_to_teams"] = True
    path: Final = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _labels(text: str) -> dict[str, str]:
    """``team="a",team_alias="b"`` to ``{"team": "a", "team_alias": "b"}``; ids and aliases carry no commas or quotes."""
    return {name: value.strip('"') for name, _, value in (pair.partition("=") for pair in text.split(","))}


def _team_members_series(scrape: str, team_id: str) -> tuple[dict[str, str], float] | None:
    """The one ``litellm_team_members_metric`` sample whose ``team`` label is ``team_id``, as (labels, value)."""
    samples: Final = tuple(
        (labels, float(value))
        for line in scrape.splitlines()
        if line.startswith(METRIC + "{")
        for label_text, _, value in (line[len(METRIC) + 1 :].partition("} "),)
        for labels in (_labels(label_text),)
        if labels.get("team") == team_id
    )
    assert len(samples) <= 1, f"{METRIC} exported more than one series for team {team_id}: {samples}"
    return samples[0] if samples else None


def _scrape(candidate: Gateway) -> str:
    response: Final = candidate.request("GET", METRICS_ROUTE)
    assert response.status_code == 200, f"GET {METRICS_ROUTE}: {response.status_code} {response.text}"
    return response.text


def _user(candidate: Gateway, scenario: Scenario) -> str:
    """An internal user created through ``candidate``; its removal is registered on the shared rig."""
    user_id: Final = f"integration-h7-{uuid.uuid4().hex}"
    candidate.post("/user/new", {"user_id": user_id, "auto_create_key": False, "user_role": "internal_user"})
    scenario.cleanups.callback(scenario.delete_user, user_id)
    return user_id


def _delete_team_if_present(candidate: Gateway, team_id: str) -> None:
    if read_rows(TEAM_SQL, (team_id,)):
        candidate.post("/team/delete", {"team_ids": [team_id]})
    assert read_rows(TEAM_SQL, (team_id,)) == []


@pytest.mark.timeout(240)  # owned proxy boot (prisma db push + readiness) takes 20-40 s
def test_team_members_gauge_reads_roster_size_then_zero_after_delete(gateway: Gateway, tmp_path: Path) -> None:
    with (
        gateway.scenario() as scenario,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"]},
            config=_prometheus_config(tmp_path),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned,
    ):
        candidate: Final = owned.gateway
        alias: Final = f"integration-h7-{uuid.uuid4().hex}"
        team_id: Final = string_value(candidate.post("/team/new", {"team_alias": alias})["team_id"])
        scenario.cleanups.callback(_delete_team_if_present, gateway, team_id)
        users: Final = tuple(_user(candidate, scenario) for _ in range(MEMBERS))
        candidate.post(
            "/team/member_add",
            {"team_id": team_id, "member": [{"role": "user", "user_id": user_id} for user_id in users]},
        )
        rows: Final = read_rows(TEAM_SQL, (team_id,))
        assert len(rows) == 1, rows
        roster: Final = rows[0]["members_with_roles"]
        assert isinstance(roster, list), roster
        assert sorted(string_value(object_value(member)["user_id"]) for member in roster) == sorted(users), roster

        before: Final = eventually(
            lambda: _team_members_series(_scrape(candidate), team_id),
            lambda sample: sample is not None,
            seconds=30,
        )
        assert before == ({"team": team_id, "team_alias": alias}, 3.0), before

        assert candidate.post("/team/delete", {"team_ids": [team_id]}) == {"deleted_teams": [team_id]}
        assert read_rows(TEAM_SQL, (team_id,)) == []
        after: Final = eventually(
            lambda: _team_members_series(_scrape(candidate), team_id),
            lambda sample: sample is not None and sample[1] == 0.0,
            seconds=30,
        )
        assert after == ({"team": team_id, "team_alias": alias}, 0.0), after
