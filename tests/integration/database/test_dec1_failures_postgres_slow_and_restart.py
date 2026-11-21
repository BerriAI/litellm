"""A Postgres that hangs, is slow, has a one-connection pool, or restarts under a serving proxy.

Contract (litellm/proxy/db/db_lookup_gate.py, litellm/constants.py `PROXY_DB_LOOKUP_*`,
litellm/proxy/health_endpoints/_health_endpoints.py, litellm/proxy/db/exception_handler.py
`call_with_db_reconnect_retry`):

* A pre-request database lookup has a deadline, `PROXY_DB_LOOKUP_DEADLINE_SECONDS` (default 10, 3 here). A
  lookup that does not answer in time fails the request with 503 `no_db_connection` instead of hanging it.
* A deadline hit marks the database "stalled" for `PROXY_DB_LOOKUP_STALL_WINDOW_SECONDS` (default 30, 8
  here): readiness answers 503 with `db: stalled` even though a fresh ping succeeds, then goes back to
  `connected` once the window has passed with no further hit.
* The readiness probe itself is bounded (2 seconds per database round trip, 4 seconds overall), so a hung
  database turns the probe into a 503, not a timeout.
* Requests that need no database (cached key, master key, liveliness) are not slowed by a hung database.
* After Postgres restarts, which drops every pooled connection, the proxy reconnects by itself: no 401 for a
  valid key, no lost or doubled spend, no proxy restart, no growing connection count.

"Hangs" means the relay keeps connections open and forwards nothing. "Restart" means the relay resets every
live connection once. Timing assertions compare against the configured deadline or a measured control.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, string_value
from tests.integration._support.timing import assert_cut_short, assert_enforced_within, assert_not_delayed_by, timed
from tests.integration.database.dec1_failures_rig import (
    CHAT,
    DB_UNAVAILABLE_ERROR,
    Rig,
    await_connected,
    chat,
    digest,
    error_of,
    failures_rig,
    marker,
    readiness,
    served_ids,
    spend_rows,
    statuses,
)

pytestmark: Final = pytest.mark.timeout(900)

DEADLINE_SECONDS: Final = 3.0
STALL_WINDOW_SECONDS: Final = 8.0
PROBE_DEADLINE_SECONDS: Final = 4.0
SLACK_SECONDS: Final = 6.0
ENVIRONMENT: Final = {
    "PROXY_DB_LOOKUP_DEADLINE_SECONDS": str(DEADLINE_SECONDS),
    "PROXY_DB_LOOKUP_STALL_WINDOW_SECONDS": str(STALL_WINDOW_SECONDS),
}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with failures_rig(tmp_path_factory.mktemp("dec1-failures-postgres-slow"), environment=ENVIRONMENT) as booted:
        yield booted


@pytest.fixture(scope="module")
def narrow(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """A proxy whose database pool holds a single connection."""
    with failures_rig(
        tmp_path_factory.mktemp("dec1-failures-postgres-pool"),
        environment=ENVIRONMENT,
        general_settings={"database_connection_pool_limit": 1},
    ) as booted:
        yield booted


@contextmanager
def _hung(target: Rig) -> Iterator[None]:
    """Postgres hangs inside the block; afterwards it answers and the proxy reports it connected."""
    target.database.stall()
    try:
        yield
    finally:
        target.database.open()
        await_connected(target.proxy)


def _warm_key(target: Rig, **fields: JsonValue) -> str:
    created: Final = target.proxy.post("/key/generate", fields)
    key: Final = string_value(created["key"])
    assert chat(target.proxy, key=key).status_code == 200
    return key


def test_an_uncached_key_is_answered_503_at_the_lookup_deadline_when_postgres_hangs(rig: Rig) -> None:
    """Postgres hangs and a valid, never used key sends a chat completion. The proxy must answer 503 with
    exactly the `no_db_connection` error object, no sooner than the 3 second lookup deadline and within a few
    seconds of it, and the provider must not be called: a hung database must not hang callers."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        text: Final = marker()
        with _hung(rig):
            answered: Final = timed(lambda: chat(rig.proxy, key=key, text=text))
            assert answered.value.status_code == 503, answered.value.text
            assert answered.value.json() == {"error": dict(DB_UNAVAILABLE_ERROR)}, answered.value.text
            assert_enforced_within(
                answered.seconds, configured=DEADLINE_SECONDS, slack=SLACK_SECONDS, not_before=DEADLINE_SECONDS - 0.5
            )
            assert rig.provider.count(text) == 0


def test_concurrent_uncached_keys_all_hit_the_deadline_together_not_one_after_another(rig: Rig) -> None:
    """Postgres hangs and five never used keys send a chat completion at the same time. All five must be
    answered 503, and the whole burst must finish within one lookup deadline plus slack, not five deadlines:
    stalled lookups must not queue behind each other."""
    with rig.proxy.scenario() as scenario:
        keys: Final = [scenario.key() for _ in range(5)]
        with _hung(rig), ThreadPoolExecutor(max_workers=5) as pool:
            burst: Final = timed(lambda: list(pool.map(lambda key: chat(rig.proxy, key=key), keys)))
            assert statuses(burst.value) == (503,) * 5
            assert_enforced_within(burst.seconds, configured=DEADLINE_SECONDS, slack=SLACK_SECONDS)


def test_a_key_refused_at_the_deadline_is_served_once_postgres_answers_again(rig: Rig) -> None:
    """A never used key gets 503 while Postgres hangs. After Postgres answers again the same key must be
    served with 200: a deadline hit must not be remembered as an invalid key."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        with _hung(rig):
            assert chat(rig.proxy, key=key).status_code == 503
        assert chat(rig.proxy, key=key).status_code == 200


def test_a_cached_key_is_not_slowed_by_a_hung_postgres(rig: Rig) -> None:
    """A cached key sends three chat completions while Postgres answers and three while it hangs. All must be
    200, and the slowest hung-database call must take less than the slowest healthy call plus the 3 second
    lookup deadline: the request path of a cached key must not wait on the database."""
    key: Final = _warm_key(rig)
    control: Final = [timed(lambda: chat(rig.proxy, key=key)) for _ in range(3)]
    with _hung(rig):
        observed: Final = [timed(lambda: chat(rig.proxy, key=key)) for _ in range(3)]
    assert statuses([call.value for call in (*control, *observed)]) == (200,) * 6
    assert_not_delayed_by(
        (call.seconds for call in observed),
        control=(call.seconds for call in control),
        injected_delay=DEADLINE_SECONDS,
    )


def test_liveliness_answers_immediately_while_postgres_hangs(rig: Rig) -> None:
    """Postgres hangs. `GET /health/liveliness` must answer 200 `"I'm alive!"` in less than the lookup
    deadline: the liveness probe must never touch the database."""
    with _hung(rig):
        alive: Final = timed(lambda: rig.proxy.client.get("/health/liveliness"))
        assert (alive.value.status_code, alive.value.json()) == (200, "I'm alive!")
        assert_cut_short(alive.seconds, injected_delay=DEADLINE_SECONDS)


def test_the_readiness_probe_is_bounded_and_503_while_postgres_hangs(rig: Rig) -> None:
    """Postgres hangs. Once the 15 second health cache has expired, `GET /health/readiness` must answer 503
    with `db: disconnected`, and a probe must take no longer than its 4 second overall deadline plus slack:
    a default Kubernetes probe timeout of 5 seconds must see an answer, not a timeout."""
    with _hung(rig):
        eventually(lambda: readiness(rig.proxy), lambda seen: seen[0] == 503, seconds=60)
        probe: Final = timed(lambda: readiness(rig.proxy))
        assert probe.value == (503, {"status": "healthy", "db": "disconnected"})
        assert_enforced_within(probe.seconds, configured=PROBE_DEADLINE_SECONDS, slack=2.0)


def test_readiness_reports_stalled_after_a_deadline_hit_even_though_the_ping_succeeds(rig: Rig) -> None:
    """A lookup hits its deadline while Postgres hangs, and Postgres then answers again. Inside the stall
    window `GET /health/readiness` must answer 503 with exactly `{"status": "healthy", "db": "stalled"}`:
    the ping uses a fresh connection, the stalled lookups are what requests wait on."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.database.stall()
        try:
            assert chat(rig.proxy, key=key).status_code == 503
        finally:
            rig.database.open()
        seen: Final = eventually(
            lambda: readiness(rig.proxy), lambda probe: probe[1]["db"] != "disconnected", seconds=30
        )
        assert seen == (503, {"status": "healthy", "db": "stalled"})
        await_connected(rig.proxy)


def test_the_stalled_state_clears_by_itself_after_the_stall_window(rig: Rig) -> None:
    """A lookup hits its deadline, Postgres recovers, and no further lookup stalls. After the 8 second stall
    window `GET /health/readiness` must answer 200 `connected` again with no restart and no admin action."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.database.stall()
        try:
            assert chat(rig.proxy, key=key).status_code == 503
        finally:
            rig.database.open()
        await_connected(rig.proxy, seconds=STALL_WINDOW_SECONDS + 60)
        assert chat(rig.proxy, key=key).status_code == 200


def test_spend_for_requests_served_while_postgres_hangs_lands_once_after_it_answers(rig: Rig) -> None:
    """A cached key is served two chat completions while Postgres hangs. After Postgres answers again the key
    must have exactly one spend row per served response id: a write that was stuck is not sent twice."""
    key: Final = _warm_key(rig)
    with _hung(rig):
        ids: Final = served_ids([chat(rig.proxy, key=key) for _ in range(2)])
    rows: Final = spend_rows(rig.proxy, key, 3)
    landed: Final = [string_value(row["request_id"]) for row in rows]
    assert all(landed.count(identity) == 1 for identity in ids), landed
    assert len(spend_rows(rig.proxy, key, 3)) == 3


def test_a_slow_postgres_that_answers_inside_the_deadline_serves_an_uncached_key(rig: Rig) -> None:
    """Every database packet is delayed by 100 ms, well inside the 3 second lookup deadline, and a never used
    key sends a chat completion. It must be served with 200: slow is not down."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.database.slow(0.1)
        try:
            served: Final = chat(rig.proxy, key=key)
        finally:
            rig.database.open()
        assert served.status_code == 200, served.text


def test_a_management_write_that_waited_on_a_hung_postgres_is_applied_exactly_once(rig: Rig) -> None:
    """The admin posts `/team/new` with a chosen alias while Postgres hangs, and Postgres answers again while
    the call is still waiting. Whatever status the call gets, afterwards the team must exist at most once,
    and if the call answered 200 it must exist exactly once: a retried write must not duplicate a team."""
    alias: Final = marker()
    with ThreadPoolExecutor(max_workers=1) as pool:
        rig.database.stall()
        try:
            pending: Final = pool.submit(rig.proxy.request, "POST", "/team/new", {"team_alias": alias})
            eventually(lambda: rig.database.live_connections(), lambda live: live >= 1, seconds=10)
        finally:
            rig.database.open()
        answered: Final = pending.result(timeout=60)
    await_connected(rig.proxy)
    teams: Final = rig.proxy.rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias = %s', (alias,))
    assert len(teams) <= 1, teams
    assert answered.status_code != 200 or len(teams) == 1, (answered.status_code, teams)
    for team in teams:
        rig.proxy.post("/team/delete", {"team_ids": [string_value(team["team_id"])]})


def test_a_burst_of_uncached_keys_on_a_one_connection_pool_gets_200_or_503_never_401_or_500(narrow: Rig) -> None:
    """The pool holds one connection and every database packet is delayed by 300 ms. Eight never used keys
    send a chat completion at once. Every answer must be either 200 or the 503 `no_db_connection`: a lookup
    that could not get a connection in time is an availability error, never an auth error or a crash."""
    with narrow.proxy.scenario() as scenario:
        keys: Final = [scenario.key() for _ in range(8)]
        narrow.database.slow(0.3)
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                answers: Final = list(pool.map(lambda key: chat(narrow.proxy, key=key), keys))
        finally:
            narrow.database.open()
        assert set(statuses(answers)) <= {200, 503}, statuses(answers)
        assert all(error_of(answer)["type"] == "no_db_connection" for answer in answers if answer.status_code == 503)


def test_every_key_of_the_burst_is_served_once_the_pool_is_free_again(narrow: Rig) -> None:
    """After the same burst on the one-connection pool, with the delay removed, each of the eight keys sends
    another chat completion. All eight must be 200: pool starvation must not poison a key or the pool."""
    with narrow.proxy.scenario() as scenario:
        keys: Final = [scenario.key() for _ in range(8)]
        narrow.database.slow(0.3)
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda key: chat(narrow.proxy, key=key), keys))
        finally:
            narrow.database.open()
        await_connected(narrow.proxy)
        assert statuses([chat(narrow.proxy, key=key) for key in keys]) == (200,) * 8


def test_a_cached_key_is_served_while_the_one_connection_pool_is_busy(narrow: Rig) -> None:
    """The pool holds one connection and Postgres hangs, so the pool is fully occupied by a stuck lookup of
    another key. A cached key sends a chat completion in the meantime. It must be served with 200 in less
    than the lookup deadline: traffic that needs no database must not queue for a pool slot."""
    key: Final = _warm_key(narrow)
    with narrow.proxy.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as pool:
        stuck_key: Final = scenario.key()
        with _hung(narrow):
            stuck: Final = pool.submit(chat, narrow.proxy, CHAT, key=stuck_key)
            served: Final = timed(lambda: chat(narrow.proxy, key=key))
            assert served.value.status_code == 200, served.value.text
            assert_cut_short(served.seconds, injected_delay=DEADLINE_SECONDS)
            assert stuck.result(timeout=60).status_code == 503


def _served_or_unavailable(target: Rig, key: str) -> httpx.Response:
    return chat(target.proxy, key=key)


def test_a_cached_key_keeps_being_served_across_a_postgres_restart(rig: Rig) -> None:
    """A cached key sends a chat completion, Postgres restarts (every pooled connection is reset), and the key
    sends another. Both must be 200."""
    key: Final = _warm_key(rig)
    rig.database.restart()
    assert chat(rig.proxy, key=key).status_code == 200
    await_connected(rig.proxy)


def test_an_uncached_key_is_never_answered_401_after_a_postgres_restart(rig: Rig) -> None:
    """Postgres restarts and a valid, never used key then sends chat completions until one is served. Every
    answer on the way must be 200 or 503, never 401, and a 200 must arrive without restarting the proxy:
    a dead pooled connection is an availability problem, not an invalid key."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.database.restart()
        seen: Final[list[int]] = []

        def attempt() -> int:
            seen.append(_served_or_unavailable(rig, key).status_code)
            return seen[-1]

        eventually(attempt, lambda status: status == 200, seconds=90)
        assert set(seen) <= {200, 503}, seen
    await_connected(rig.proxy)


def test_a_management_write_succeeds_after_a_postgres_restart_without_restarting_the_proxy(rig: Rig) -> None:
    """Postgres restarts and the admin then posts `/key/generate` with a chosen key value until it succeeds.
    It must succeed with no proxy restart, and the key must exist exactly once however many attempts failed
    first."""
    wanted: Final = f"sk-{marker()}"
    workers_before: Final = rig.owned.worker_pids()
    rig.database.restart()
    eventually(
        lambda: rig.proxy.request("POST", "/key/generate", {"key": wanted}).status_code,
        lambda status: status == 200,
        seconds=90,
    )
    rows: Final = rig.proxy.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest(wanted),))
    assert len(rows) == 1, rows
    assert rig.owned.worker_pids() == workers_before
    rig.proxy.post("/key/delete", {"keys": [wanted]})


def test_spend_queued_before_a_postgres_restart_lands_exactly_once(rig: Rig) -> None:
    """A cached key is served three chat completions and Postgres restarts right after, before the one second
    spend flush. The key must end with exactly one spend row per response id, counting its warm-up row four
    rows in total: a flush cut off by the restart is retried without duplicating rows."""
    key: Final = _warm_key(rig)
    ids: Final = served_ids([chat(rig.proxy, key=key) for _ in range(3)])
    rig.database.restart()
    rows: Final = spend_rows(rig.proxy, key, 4)
    landed: Final = [string_value(row["request_id"]) for row in rows]
    assert all(landed.count(identity) == 1 for identity in ids), landed
    await_connected(rig.proxy)
    assert len(spend_rows(rig.proxy, key, 4)) == 4


def test_a_request_in_flight_at_the_provider_survives_a_postgres_restart(rig: Rig) -> None:
    """A cached key's chat completion is held at the provider, Postgres restarts while it is in flight, and
    the provider then answers. The caller must get its 200, and the request's spend row must land once."""
    key: Final = _warm_key(rig)
    text: Final = marker()
    rig.provider.hold()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending: Final = pool.submit(chat, rig.proxy, CHAT, key=key, text=text)
            eventually(lambda: rig.provider.count(text), lambda count: count == 1, seconds=30)
            rig.database.restart()
            rig.provider.release()
            answered: Final = pending.result(timeout=60)
    finally:
        rig.provider.release()
    identity: Final = served_ids([answered])[0]
    rows: Final = spend_rows(rig.proxy, key, 2)
    assert [string_value(row["request_id"]) for row in rows].count(identity) == 1, rows


def test_repeated_postgres_restarts_do_not_grow_the_connection_count(rig: Rig) -> None:
    """The number of sessions on the proxy's database is counted, Postgres restarts three times with a new
    key served after each restart, and the sessions are counted again. The count must come back to no more
    than it was before: reconnecting must replace connections, not leak them."""
    count_sessions: Final = "SELECT count(*) AS sessions FROM pg_stat_activity WHERE datname = current_database()"
    await_connected(rig.proxy)
    before: Final = int(str(rig.proxy.rows(count_sessions, ())[0]["sessions"]))
    with rig.proxy.scenario() as scenario:
        for _ in range(3):
            key = scenario.key()
            rig.database.restart()
            eventually(lambda key=key: chat(rig.proxy, key=key).status_code, lambda status: status == 200, seconds=90)
        await_connected(rig.proxy)
        after: Final = eventually(
            lambda: int(str(rig.proxy.rows(count_sessions, ())[0]["sessions"])),
            lambda sessions: sessions <= before,
            seconds=60,
            return_last_on_timeout=True,
        )
        assert after <= before, f"{after} sessions after three restarts, {before} before"
