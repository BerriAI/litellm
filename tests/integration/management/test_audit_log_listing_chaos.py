"""A burst of team and key updates that loses a worker, or the whole proxy, still leaves audit rows GET /audit lists.

Each leg fires 30 updates at once on an owned two-worker proxy, 15 teams and 15 keys. The test holds the first
team's row lock from its own transaction, waits until that team's update is queued behind it inside Postgres,
breaks the proxy, and only then releases the lock, so the outage always overlaps an update in flight. An update
the outage dropped is sent once more after recovery. Every row the burst and the retries left must be an object,
no request may leave more than one row, and the page must list exactly the rows Postgres holds.
"""

import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import psycopg
import pytest
from tests.integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process

RecordProperty = Callable[[str, object], None]

PER_KIND: Final = 15
WORKERS: Final = 2
UPDATE_TIMEOUT_SECONDS: Final = 60
STORED_SQL: Final = """
SELECT id, jsonb_typeof(before_value) AS before_type FROM "LiteLLM_AuditLog"
WHERE object_id = %s AND action = 'updated' ORDER BY id
"""
LOCK_TEAM_ROW_SQL: Final = 'SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s FOR UPDATE'
BLOCKED_BY_SQL: Final = """
SELECT count(*)::int AS waiting FROM pg_stat_activity WHERE %s::int = ANY(pg_blocking_pids(pid))
"""


@dataclass(frozen=True, slots=True)
class Update:
    path: str
    body: dict[str, str | int]
    object_id: str


@dataclass(frozen=True, slots=True)
class Outcome:
    update: Update
    status: int | None
    detail: str

    @property
    def dropped(self) -> bool:
        return self.status is None or self.status >= 500


def _overrides() -> dict[str, str]:
    return {"DATABASE_URL": os.environ["DATABASE_URL"], "LITELLM_STORE_AUDIT_LOGS": "true"}


@contextmanager
def _audited_proxy(gateway: Gateway, directory: Path) -> Iterator[Gateway]:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    with owned_proxy_process(
        gateway, directory, _overrides(), remove_environment=("DATABASE_URL_READ_REPLICA",), workers=WORKERS
    ) as owned:
        yield owned


def _fleet(scenario: Scenario) -> tuple[Update, ...]:
    teams: Final = tuple(scenario.team() for _ in range(PER_KIND))
    tokens: Final = tuple(sha256(scenario.key().encode()).hexdigest() for _ in range(PER_KIND))
    return (
        *(Update("/team/update", {"team_id": team, "max_budget": 7}, team) for team in teams),
        *(Update("/key/update", {"key": token, "max_budget": 3}, token) for token in tokens),
    )


def _send(target: Gateway, update: Update) -> Outcome:
    try:
        response: Final = target.client.request(
            "POST",
            update.path,
            json=update.body,
            headers={"Authorization": f"Bearer {target.key}"},
            timeout=UPDATE_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as error:
        return Outcome(update, None, f"{type(error).__name__}: {error}"[:200])
    return Outcome(update, response.status_code, response.text[:200])


def _burst(pool: ThreadPoolExecutor, target: Gateway, fleet: Sequence[Update]) -> tuple[Future[Outcome], ...]:
    return tuple(pool.submit(_send, target, update) for update in fleet)


def _outcomes(burst: Sequence[Future[Outcome]]) -> tuple[Outcome, ...]:
    return tuple(future.result(timeout=UPDATE_TIMEOUT_SECONDS + 30) for future in burst)


@contextmanager
def _update_of_team_held_in_postgres(team: str) -> Iterator[Callable[[], None]]:
    """Lock the team's row from a test-owned transaction; the yielded wait returns once an update is queued
    behind it. Leaving the block commits, which lets that update through."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
        holder.execute(LOCK_TEAM_ROW_SQL, (team,))
        backend: Final = str(holder.info.backend_pid)
        yield lambda: eventually(
            lambda: read_rows(BLOCKED_BY_SQL, (backend,))[0]["waiting"], lambda waiting: waiting != 0, seconds=30
        )


def _stored(update: Update) -> list[dict[str, object]]:
    return [dict(row) for row in read_rows(STORED_SQL, (update.object_id,))]


def _listed_ids(target: Gateway, update: Update) -> list[str]:
    page: Final = target.request(
        "GET", "/audit", params={"object_id": update.object_id, "action": "updated", "page_size": "100"}
    )
    assert page.status_code == 200, f"{update.object_id}: {page.status_code} {page.text}"
    return sorted(str(object_value(log)["id"]) for log in list_value(object_value(page.json())["audit_logs"]))


def _assert_every_row_is_listed_once_as_an_object(
    target: Gateway, outcomes: Sequence[Outcome], record_property: RecordProperty
) -> None:
    """Send each dropped update once more through ``target``, then check every update's rows."""
    assert [outcome.detail for outcome in outcomes if not outcome.dropped and outcome.status != 200] == []
    dropped: Final = tuple(outcome.update for outcome in outcomes if outcome.dropped)
    for update in dropped:
        assert _send(target, update).status == 200, update
    answered: Final = tuple(outcome.update for outcome in outcomes if outcome.status == 200)
    retried_rows: Final = [
        eventually(lambda update=update: _stored(update), lambda rows: len(rows) != 0, seconds=30)
        for update in dropped
    ]
    rows: Final = {outcome.update.object_id: _stored(outcome.update) for outcome in outcomes}
    record_property("dropped_by_the_outage", len(dropped))
    record_property("answered_200_without_a_row", sum(not rows[update.object_id] for update in answered))
    assert [len(found) for found in retried_rows if len(found) > 2] == [], retried_rows
    assert [update.object_id for update in answered if len(rows[update.object_id]) > 1] == []
    assert [row for found in rows.values() for row in found if row["before_type"] != "object"] == []
    assert sum(len(found) for found in rows.values()) >= len(dropped) + 1, rows
    for outcome in outcomes:
        assert _listed_ids(target, outcome.update) == [str(row["id"]) for row in rows[outcome.update.object_id]]


@pytest.mark.timeout(300)
def test_a_worker_killed_mid_burst_leaves_every_update_row_listed_once_as_an_object(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with (
        gateway.scenario() as scenario,
        _audited_proxy(gateway, tmp_path) as owned,
        ThreadPoolExecutor(2 * PER_KIND) as pool,
    ):
        fleet: Final = _fleet(scenario)
        victim: Final = min(owned.worker_pids())
        with owned.replacing_worker(victim) as kill:
            with _update_of_team_held_in_postgres(fleet[0].object_id) as wait_until_queued:
                burst: Final = _burst(pool, owned.gateway, fleet)
                wait_until_queued()
                kill()
                with httpx.Client(base_url=str(owned.gateway.client.base_url), timeout=15, trust_env=False) as fresh:
                    survivor: Final = fresh.get("/health/readiness")
                assert survivor.status_code == 200, survivor.text
            outcomes: Final = _outcomes(burst)
        assert victim not in owned.worker_pids()

        _assert_every_row_is_listed_once_as_an_object(owned.gateway, outcomes, record_property)


@pytest.mark.timeout(420)
def test_a_proxy_stopped_mid_burst_and_restarted_lists_every_update_row_once_as_an_object(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with gateway.scenario() as scenario, ThreadPoolExecutor(2 * PER_KIND) as pool:
        fleet: Final = _fleet(scenario)
        with _audited_proxy(gateway, tmp_path) as doomed:
            with _update_of_team_held_in_postgres(fleet[0].object_id) as wait_until_queued:
                burst: Final = _burst(pool, doomed.gateway, fleet)
                wait_until_queued()
                doomed.begin_shutdown()
            doomed.process.wait(timeout=120)
            outcomes: Final = _outcomes(burst)

        with _audited_proxy(gateway, tmp_path) as restarted:
            _assert_every_row_is_listed_once_as_an_object(restarted.gateway, outcomes, record_property)
