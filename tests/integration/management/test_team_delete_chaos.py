"""Chaos rows for ``/team/delete`` on an owned two-worker proxy: C1 worker kill, C2 Redis outage, C3 proxy restart.

Each leg creates 24 teams through the owned proxy (two internal users per team in one bulk
``/team/member_add``, plus one team key), then deletes all 24 in a 24-thread burst while the
infrastructure fails once the third delete has answered:

- C1 SIGKILLs one uvicorn worker child; the survivor still answers ``/health/readiness`` and uvicorn
  respawns the worker.
- C2 shuts the owned Redis down; ``/cache/ping`` reports it, the deletes keep answering 200 because
  cache eviction and the invalidation broadcast are best-effort, then Redis comes back.
- C3 SIGTERMs the owned proxy root and a fresh proxy starts on the same database.

After recovery the burst outcomes (status or transport error per team) are recorded, every team whose
row survived is deleted once more, and the invariants must hold for every team: no ``LiteLLM_TeamTable``
row, no ``LiteLLM_TeamMembership`` row, no ``LiteLLM_UserTable.teams`` entry naming it, its key gone
from ``LiteLLM_VerificationToken``, and one ``LiteLLM_DeletedTeamTable`` row per attempt that reached
the tombstone write. Both legs commit that tombstone before the locked transaction that removes the
team, so an attempt that died in between leaves a tombstone for a live team and the retry adds a
second; that count is pinned as observed (pre-existing, outside this PR's diff, ticketed in the audit
report) and the affected teams are recorded as ``double_tombstones``. Teams found half-deleted before
the retry are recorded as ``partial_states_before_retry`` and named in any failure.

Nothing sleeps, and only processes the test started are signalled.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    string_value,
)
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.redis_process import owned_redis

RecordProperty = Callable[[str, object], None]

TEAMS: Final = 24
MEMBERS_PER_TEAM: Final = 2
CHAOS_AFTER_ANSWERS: Final = 3
WORKERS: Final = 2
DELETE_TIMEOUT_SECONDS: Final = 60
REMOVE_FROM_ENVIRONMENT: Final = ("DATABASE_URL_READ_REPLICA",)

TEAM_SQL: Final = 'SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s'
TOMBSTONE_SQL: Final = 'SELECT id FROM "LiteLLM_DeletedTeamTable" WHERE team_id = %s'
MEMBERSHIP_SQL: Final = 'SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s'
REFERENCING_USERS_SQL: Final = 'SELECT user_id FROM "LiteLLM_UserTable" WHERE %s = ANY(teams)'
TOKEN_SQL: Final = 'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s'


@dataclass(frozen=True, slots=True)
class Team:
    team_id: str
    members: tuple[str, ...]
    hashed_key: str


@dataclass(frozen=True, slots=True)
class Outcome:
    """One burst delete: the HTTP status, or ``None`` with the transport error's class and message."""

    team_id: str
    status: int | None
    detail: str

    @property
    def label(self) -> str:
        return str(self.status) if self.status is not None else self.detail.split(":", 1)[0]

    @property
    def answered_or_dropped(self) -> bool:
        """200, a 5xx from a dying process, or a transport error; a 4xx would mean a wrong delete."""
        return self.status is None or self.status == 200 or self.status >= 500


@dataclass(frozen=True, slots=True)
class TeamState:
    team_id: str
    row_present: bool
    tombstones: int
    memberships: tuple[str, ...]
    referencing_users: tuple[str, ...]
    key_present: bool

    @property
    def clean(self) -> bool:
        """Row, memberships, ``teams`` references and key all gone; tombstones are counted per attempt."""
        return not self.row_present and not self.memberships and not self.referencing_users and not self.key_present

    @property
    def untouched(self) -> bool:
        return self.row_present and self.tombstones == 0 and self.key_present

    @property
    def partial(self) -> bool:
        return not (self.clean and self.tombstones == 1) and not self.untouched

    def describe(self) -> str:
        return (
            f"{self.team_id}: row={'present' if self.row_present else 'gone'} tombstones={self.tombstones} "
            f"memberships={len(self.memberships)} referencing_users={len(self.referencing_users)} "
            f"key={'present' if self.key_present else 'gone'}"
        )


def _state(team: Team) -> TeamState:
    return TeamState(
        team.team_id,
        row_present=bool(read_rows(TEAM_SQL, (team.team_id,))),
        tombstones=len(read_rows(TOMBSTONE_SQL, (team.team_id,))),
        memberships=tuple(string_value(row["user_id"]) for row in read_rows(MEMBERSHIP_SQL, (team.team_id,))),
        referencing_users=tuple(
            string_value(row["user_id"]) for row in read_rows(REFERENCING_USERS_SQL, (team.team_id,))
        ),
        key_present=bool(read_rows(TOKEN_SQL, (team.hashed_key,))),
    )


def _states(fleet: Sequence[Team]) -> tuple[TeamState, ...]:
    return tuple(_state(team) for team in fleet)


def _overrides() -> dict[str, str]:
    return {"DATABASE_URL": os.environ["DATABASE_URL"]}


def _user(candidate: Gateway, scenario: Scenario) -> str:
    """An internal user created through ``candidate``; its removal is registered on the shared rig."""
    user_id: Final = f"integration-chaos-{uuid.uuid4().hex}"
    candidate.post("/user/new", {"user_id": user_id, "auto_create_key": False, "user_role": "internal_user"})
    scenario.cleanups.callback(scenario.delete_user, user_id)
    return user_id


def _delete_team_if_present(candidate: Gateway, team_id: str) -> None:
    if read_rows(TEAM_SQL, (team_id,)):
        candidate.post("/team/delete", {"team_ids": [team_id]})
    assert read_rows(TEAM_SQL, (team_id,)) == []


def _team(candidate: Gateway, scenario: Scenario, index: int) -> Team:
    alias: Final = f"integration-chaos-{index:02d}-{uuid.uuid4().hex}"
    team_id: Final = string_value(candidate.post("/team/new", {"team_alias": alias})["team_id"])
    scenario.cleanups.callback(_delete_team_if_present, scenario.gateway, team_id)
    members: Final = tuple(_user(candidate, scenario) for _ in range(MEMBERS_PER_TEAM))
    candidate.post(
        "/team/member_add",
        {"team_id": team_id, "member": [{"role": "user", "user_id": user_id} for user_id in members]},
    )
    key: Final = string_value(candidate.post("/key/generate", {"team_id": team_id, "key_alias": alias})["key"])
    scenario.cleanups.callback(delete_key_if_present, scenario.gateway, key)
    return Team(team_id, members, sha256(key.encode()).hexdigest())


def _fleet(candidate: Gateway, scenario: Scenario) -> tuple[Team, ...]:
    """24 teams created through ``candidate``, each verified intact: row, key, both members' membership
    rows and ``teams`` entries present, so the invariants after the burst have something to remove.

    A master-key ``/team/new`` also seats ``default_user_id`` as an admin (roster entry, membership row and
    ``teams`` entry), so the checks are supersets. Cleanup is registered on the shared rig; the team
    callback only acts when a run fails before its delete.
    """
    fleet: Final = tuple(_team(candidate, scenario, index) for index in range(TEAMS))
    for team, state in zip(fleet, _states(fleet)):
        assert state.untouched, state.describe()
        assert set(state.memberships) >= set(team.members), state.describe()
        assert set(state.referencing_users) >= set(team.members), state.describe()
    return fleet


class Burst:
    """One ``/team/delete`` per team on ``target``, all submitted at once; ``chaos_point`` is set once the
    third delete has answered (or failed), so the leg breaks the infrastructure mid-burst."""

    def __init__(self, target: Gateway) -> None:
        self._target: Final = target
        self._lock: Final = threading.Lock()
        self._answers = 0  # rebind-ok: counter behind _lock
        self._futures: tuple[Future[Outcome], ...] = ()
        self.chaos_point: Final = threading.Event()

    def start(self, pool: ThreadPoolExecutor, fleet: Sequence[Team]) -> None:
        assert not self._futures, "burst already started"
        self._futures = tuple(pool.submit(self._delete, team) for team in fleet)
        assert self.chaos_point.wait(DELETE_TIMEOUT_SECONDS), (
            f"fewer than {CHAOS_AFTER_ANSWERS} deletes answered within {DELETE_TIMEOUT_SECONDS}s"
        )

    def _delete(self, team: Team) -> Outcome:
        try:
            response: Final = self._target.client.request(
                "POST",
                "/team/delete",
                json={"team_ids": [team.team_id]},
                headers={"Authorization": f"Bearer {self._target.key}"},
                timeout=DELETE_TIMEOUT_SECONDS,
            )
            outcome = Outcome(team.team_id, response.status_code, response.text[:200])
        except httpx.HTTPError as error:  # a killed worker or a stopped proxy drops the in-flight request
            outcome = Outcome(team.team_id, None, f"{type(error).__name__}: {error}"[:200])
        with self._lock:
            self._answers += 1
            if self._answers >= CHAOS_AFTER_ANSWERS:
                self.chaos_point.set()
        return outcome

    def outcomes(self) -> tuple[Outcome, ...]:
        return tuple(future.result(timeout=DELETE_TIMEOUT_SECONDS + 30) for future in self._futures)


def _record_burst(record_property: RecordProperty, outcomes: Sequence[Outcome], observed: Sequence[TeamState]) -> None:
    """Record the status split and the half-deleted teams seen before the retry."""
    split: Final = Counter(outcome.label for outcome in outcomes)
    record_property("status_split", dict(sorted(split.items())))
    record_property("partial_states_before_retry", [state.describe() for state in observed if state.partial])
    record_property("rows_present_before_retry", sum(state.row_present for state in observed))


def _retry_survivors(target: Gateway, fleet: Sequence[Team], observed: Sequence[TeamState]) -> tuple[str, ...]:
    """Delete once more, through ``target``, every team whose row survived the burst; each must answer 200."""
    survivors: Final = tuple(team.team_id for team, state in zip(fleet, observed) if state.row_present)
    for team_id in survivors:
        assert target.post("/team/delete", {"team_ids": [team_id]}) == {"deleted_teams": [team_id]}
    return survivors


def _expected_tombstones(before: TeamState, retried: bool) -> int:
    """One ``LiteLLM_DeletedTeamTable`` row per attempt that reached the tombstone write.

    Both legs commit the tombstone before the locked transaction that removes the team, so a burst
    attempt that died in between left one (``before.tombstones``, 0 or 1) for a team whose row
    survived, and the retry adds one more. Pinned as observed: pre-existing on the merge base,
    outside this PR's diff, ticketed in the audit report.
    """
    assert before.tombstones <= 1, before.describe()
    return before.tombstones + (1 if retried else 0)


def _assert_every_team_fully_deleted(
    record_property: RecordProperty,
    before_retry: Sequence[TeamState],
    final: Sequence[TeamState],
    retried: Sequence[str],
) -> None:
    """Every team: row, memberships, ``teams`` references and key gone; tombstones one per attempt."""
    expected: Final = {state.team_id: _expected_tombstones(state, state.team_id in retried) for state in before_retry}
    record_property("double_tombstones", sorted(team_id for team_id, count in expected.items() if count == 2))
    violations: Final = tuple(
        f"{state.describe()} expected tombstones={expected[state.team_id]}"
        for state in final
        if not state.clean or state.tombstones != expected[state.team_id] or expected[state.team_id] == 0
    )
    assert not violations, (
        f"{len(violations)} of {len(final)} teams are not fully deleted after the retry:\n  "
        + "\n  ".join(violations)
        + f"\nhalf-deleted before the retry ({sum(state.partial for state in before_retry)}):\n  "
        + "\n  ".join(state.describe() for state in before_retry if state.partial)
        + f"\nretried ({len(retried)}): {sorted(retried)}"
    )


def _workers(root: psutil.Process) -> tuple[psutil.Process, ...]:
    """uvicorn's worker children of the owned proxy root, spawned through ``multiprocessing.spawn``.

    The root's other child is the multiprocessing resource tracker; each worker's prisma query engine
    is a grandchild. A worker that just died shows as a zombie whose cmdline raises, so it is left out.
    """
    workers: Final = []
    for child in root.children():
        try:
            cmdline = child.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any("multiprocessing.spawn" in part for part in cmdline):
            workers.append(child)
    return tuple(sorted(workers, key=lambda process: process.pid))


def _cache_ping(target: Gateway) -> httpx.Response:
    return target.request("GET", "/cache/ping")


def _cache_status(response: httpx.Response) -> str:
    assert response.status_code == 200, f"/cache/ping: {response.status_code} {response.text}"
    return string_value(JSON_OBJECT.validate_json(response.content)["status"])


@pytest.mark.timeout(240)  # owned two-worker proxy boot plus a 24-team fleet and its cleanup
def test_worker_killed_mid_burst_leaves_every_team_fully_deleted_after_retry(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with (
        gateway.scenario() as scenario,
        owned_proxy_process(
            gateway, tmp_path, _overrides(), remove_environment=REMOVE_FROM_ENVIRONMENT, workers=WORKERS
        ) as owned,
        ThreadPoolExecutor(TEAMS) as pool,
    ):
        root: Final = psutil.Process(owned.process.pid)
        fleet: Final = _fleet(owned.gateway, scenario)
        burst: Final = Burst(owned.gateway)
        burst.start(pool, fleet)

        before: Final = _workers(root)
        assert len(before) == WORKERS, [process.pid for process in before]
        victim: Final = before[0]
        victim.kill()  # SIGKILL: the worker cannot finish its in-flight deletes
        victim.wait(timeout=10)
        with httpx.Client(base_url=str(owned.gateway.client.base_url), timeout=15, trust_env=False) as fresh:
            readiness: Final = fresh.get("/health/readiness")
        assert readiness.status_code == 200, (
            f"/health/readiness with worker {victim.pid} dead: {readiness.status_code} {readiness.text}"
        )

        outcomes: Final = burst.outcomes()
        respawned: Final = eventually(
            lambda: tuple(process.pid for process in _workers(root)),
            lambda pids: len(pids) == WORKERS and victim.pid not in pids,
            seconds=60,
        )
        record_property(
            "worker_pids", {"before": [process.pid for process in before], "killed": victim.pid, "after": respawned}
        )
        observed: Final = _states(fleet)
        _record_burst(record_property, outcomes, observed)
        assert all(outcome.answered_or_dropped for outcome in outcomes), [
            (outcome.team_id, outcome.status, outcome.detail) for outcome in outcomes if not outcome.answered_or_dropped
        ]
        retried: Final = _retry_survivors(owned.gateway, fleet, observed)
        _assert_every_team_fully_deleted(record_property, observed, _states(fleet), retried)


@pytest.mark.timeout(240)  # owned Redis, owned two-worker proxy boot, 24-team fleet, Redis restart
def test_redis_stopped_mid_burst_keeps_deletes_answering_200(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with (
        gateway.scenario() as scenario,
        owned_redis(tmp_path) as coordination,
        owned_proxy_process(
            gateway,
            tmp_path,
            {
                **_overrides(),
                "REDIS_HOST": coordination.host,
                "REDIS_PORT": str(coordination.port),
                # The breaker opens during the outage; the default 60 s before it probes again would
                # keep /cache/ping (whose set_cache runs under the breaker) at 503 long after restart.
                "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "5",
            },
            remove_environment=REMOVE_FROM_ENVIRONMENT,
            workers=WORKERS,
        ) as owned,
        ThreadPoolExecutor(TEAMS) as pool,
    ):
        fleet: Final = _fleet(owned.gateway, scenario)
        assert _cache_status(_cache_ping(owned.gateway)) == "healthy"
        burst: Final = Burst(owned.gateway)
        burst.start(pool, fleet)

        coordination.stop()
        down: Final = _cache_ping(owned.gateway)
        assert down.status_code == 503, f"/cache/ping with Redis stopped: {down.status_code} {down.text}"
        assert "Service Unhealthy" in down.text, down.text

        outcomes: Final = burst.outcomes()
        coordination.start()
        recovered: Final = eventually(lambda: _cache_ping(owned.gateway), lambda r: r.status_code == 200, seconds=60)
        assert _cache_status(recovered) == "healthy"

        observed: Final = _states(fleet)
        _record_burst(record_property, outcomes, observed)
        assert all(outcome.status == 200 for outcome in outcomes), (
            "deletes not answered 200 while Redis was down: "
            + str([(outcome.team_id, outcome.status, outcome.detail) for outcome in outcomes if outcome.status != 200])
            + f"; split {dict(Counter(outcome.label for outcome in outcomes))}"
        )
        retried: Final = _retry_survivors(owned.gateway, fleet, observed)
        _assert_every_team_fully_deleted(record_property, observed, _states(fleet), retried)


@pytest.mark.timeout(240)  # two owned two-worker proxy boots (before and after SIGTERM) plus a 24-team fleet
def test_proxy_terminated_mid_burst_then_restarted_leaves_every_team_fully_deleted(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with gateway.scenario() as scenario, ThreadPoolExecutor(TEAMS) as pool:
        with owned_proxy_process(
            gateway, tmp_path, _overrides(), remove_environment=REMOVE_FROM_ENVIRONMENT, workers=WORKERS
        ) as doomed:
            fleet: Final = _fleet(doomed.gateway, scenario)
            burst: Final = Burst(doomed.gateway)
            burst.start(pool, fleet)
            doomed.process.terminate()  # SIGTERM: uvicorn stops accepting, drains, and exits
            doomed.process.wait(timeout=120)
            outcomes: Final = burst.outcomes()

        at_restart: Final = _states(fleet)
        _record_burst(record_property, outcomes, at_restart)
        assert all(outcome.answered_or_dropped for outcome in outcomes), [
            (outcome.team_id, outcome.status, outcome.detail) for outcome in outcomes if not outcome.answered_or_dropped
        ]
        with owned_proxy_process(
            gateway, tmp_path, _overrides(), remove_environment=REMOVE_FROM_ENVIRONMENT, workers=WORKERS
        ) as fresh:
            retried: Final = _retry_survivors(fresh.gateway, fleet, at_restart)
            final: Final = _states(fleet)
        _assert_every_team_fully_deleted(record_property, at_restart, final, retried)
