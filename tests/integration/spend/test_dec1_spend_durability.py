"""Contract: spend is not lost when a proxy stops, and several proxies on one database agree on spend.

A proxy does not write spend totals at the moment a request is answered. It records the request, and a background
job later adds the cost to the key, team, user and daily tables. These tests pin that a clean stop (SIGTERM) never
loses a served request, that a restart never bills it twice, and that a budget that was used up stays used up for
a freshly started proxy.

The second half uses two proxies that share one Postgres and one Redis: the worker's proxy plus either the
harness `peer` proxy or a proxy the test boots. Spend made through either one must add up on the shared key and
team, and a budget exhausted through one must be refused by the other. The last tests turn on
`general_settings.use_redis_transaction_buffer`, the setting operators use for multi-instance deployments, and
pin that the totals are the same with it on.

One served chat call costs 0.02 (10 prompt tokens at 0.001 plus 5 completion tokens at 0.002).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from redis import Redis

from tests.integration._support.client import Gateway, eventually, held, object_value
from tests.integration._support.process import owned_proxy
from tests.integration.spend.dec1_spend_support import (
    CHAT,
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    ask,
    chat_model,
    daily_row,
    hashed,
    number,
    provider_calls,
    proxy_config,
    refusal,
    rows_of_key,
    scripted_upstream,
    served,
    served_id,
    spend_of,
    spend_reaches,
    stream,
)

pytestmark: Final = pytest.mark.shares_worker_state

SLOW_FLUSH: Final = {"proxy_batch_write_at": 3600}
REDIS_BUFFER: Final = {"use_redis_transaction_buffer": True}


@contextmanager
def _second_proxy(gateway: Gateway, directory: Path, settings: dict[str, int | bool]) -> Iterator[Gateway]:
    """A proxy on the worker's database and Redis with `settings` merged into `general_settings`."""
    directory.mkdir(parents=True, exist_ok=True)
    with owned_proxy(gateway, directory, {}, config=proxy_config(directory, general_settings=settings)) as owned:
        yield owned


def _spend_stays(gateway: Gateway, key: str, amount: float) -> float | None:
    return held(
        lambda: spend_of(gateway, "key", hashed(key)),
        lambda spend: spend is not None and spend == pytest.approx(amount),
        holding=5,
        seconds=LANDING_SECONDS,
    )


def test_request_row_survives_a_stop_before_the_flush_timer(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy whose spend flush timer is one hour serves one chat call and is then stopped cleanly. Afterwards
    exactly one `LiteLLM_SpendLogs` row exists for the key, with the response id and spend 0.02. A rewrite that
    keeps rows in memory until a timer fires would lose this row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, SLOW_FLUSH) as owned:
            request_id: Final = served_id(ask(owned, model, key))
        rows: Final = rows_of_key(gateway, key, 1)
        assert rows[0]["request_id"] == request_id
        assert number(rows[0]["spend"]) == pytest.approx(0.02)


def test_key_spend_survives_a_stop_before_the_flush_timer(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with a one hour flush timer serves one call and is stopped. The key's stored spend then reaches
    0.02 and stays 0.02: the cost is neither lost nor applied twice by the proxies still running.
    UNVERIFIED: whether the stopping proxy or the surviving worker proxy applies the pending ledger entry."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, SLOW_FLUSH) as owned:
            served(ask(owned, model, key))
        assert spend_reaches(gateway, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)
        assert _spend_stays(gateway, key, 0.02) == pytest.approx(0.02)


def test_team_and_user_spend_survive_a_stop(gateway: Gateway, tmp_path: Path) -> None:
    """A team key owned by a user makes one call through a proxy that is then stopped. The team's spend and the
    user's spend both reach 0.02, so the stop flushes every entity, not only the key."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        with _second_proxy(gateway, tmp_path, SLOW_FLUSH) as owned:
            served(ask(owned, model, key))
        assert spend_reaches(gateway, "team", team, CHAT_COST) == pytest.approx(0.02)
        assert spend_reaches(gateway, "user", user, CHAT_COST) == pytest.approx(0.02)


def test_daily_row_survives_a_stop(gateway: Gateway, tmp_path: Path) -> None:
    """One call through a proxy that is then stopped leaves one `LiteLLM_DailyUserSpend` row for the key with
    `api_requests` 1, `successful_requests` 1, 10 prompt tokens, 5 completion tokens and spend 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, SLOW_FLUSH) as owned:
            served(ask(owned, model, key))
        row: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (row["api_requests"], row["successful_requests"]) == (1, 1), row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (10, 5), row
        assert number(row["spend"]) == pytest.approx(0.02)


def test_streamed_call_survives_a_stop(gateway: Gateway, tmp_path: Path) -> None:
    """A streamed chat call is read to the end and the proxy is stopped right after. The key still has one
    request row and spend 0.02, so cost that is only known when the stream ends is also flushed on stop."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, SLOW_FLUSH) as owned:
            lines: Final = stream(owned, CHAT.path, CHAT.body(model, streaming=True), key)
            assert "data: [DONE]" in lines
        assert len(rows_of_key(gateway, key, 1)) == 1
        assert spend_reaches(gateway, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)


def test_three_calls_before_a_stop_are_each_billed_once(gateway: Gateway, tmp_path: Path) -> None:
    """Three calls go through a proxy that is then stopped. Three request rows exist and the key's spend settles
    at 0.06 and stays there for five seconds."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, SLOW_FLUSH) as owned:
            ids: Final = {served_id(ask(owned, model, key)) for _ in range(3)}
        assert {row["request_id"] for row in rows_of_key(gateway, key, 3)} == ids
        assert spend_reaches(gateway, "key", hashed(key), 0.06) == pytest.approx(0.06)
        assert _spend_stays(gateway, key, 0.06) == pytest.approx(0.06)


def test_restart_does_not_bill_a_served_call_again(gateway: Gateway, tmp_path: Path) -> None:
    """One call is served, the proxy is stopped, and a second proxy is started on the same database. While the
    second proxy runs, the key's spend stays 0.02 and the key keeps exactly one request row. A rewrite that
    replays unacknowledged work on start would show 0.04."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path / "first", SLOW_FLUSH) as owned:
            served(ask(owned, model, key))
        with _second_proxy(gateway, tmp_path / "second", SLOW_FLUSH) as restarted:
            assert spend_reaches(restarted, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)
            assert _spend_stays(restarted, key, 0.02) == pytest.approx(0.02)
        assert len(rows_of_key(gateway, key, 1)) == 1


def test_key_over_budget_is_still_refused_after_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget` 0.0000001 spends 0.02 through one proxy, which is then stopped. A freshly started
    proxy refuses the key's next call with HTTP 422 and error type `budget_exceeded`, and the provider sees no
    second request. The new process has empty memory, so this only holds if the spend was durably stored."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        with _second_proxy(gateway, tmp_path / "first", SLOW_FLUSH) as owned:
            served(ask(owned, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        with _second_proxy(gateway, tmp_path / "second", SLOW_FLUSH) as restarted:
            message: Final = refusal(ask(restarted, model, key))
        assert "udget" in message, message
        assert len(provider_calls(wire)) == 1


def test_spend_from_two_instances_adds_up_on_the_key(gateway: Gateway, peer: Gateway) -> None:
    """The same key makes one call through each of two proxies that share a database. The key's stored spend
    settles at 0.04 and two request rows exist, one per response id."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ids: Final = {served_id(ask(gateway, model, key)), served_id(ask(peer, model, key))}
        assert {row["request_id"] for row in rows_of_key(gateway, key, 2)} == ids
        assert spend_reaches(gateway, "key", hashed(key), 0.04) == pytest.approx(0.04)
        assert _spend_stays(gateway, key, 0.04) == pytest.approx(0.04)


def test_spend_from_two_instances_adds_up_on_the_team(gateway: Gateway, peer: Gateway) -> None:
    """Two keys of one team each make one call, one through each proxy. The team's spend settles at 0.04 while
    each key shows 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        first: Final = scenario.key(team_id=team, models=[model])
        second: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, first))
        served(ask(peer, model, second))
        assert spend_reaches(gateway, "team", team, 0.04) == pytest.approx(0.04)
        assert spend_reaches(gateway, "key", hashed(first), CHAT_COST) == pytest.approx(0.02)
        assert spend_reaches(gateway, "key", hashed(second), CHAT_COST) == pytest.approx(0.02)


def test_two_instances_share_one_daily_row(gateway: Gateway, peer: Gateway) -> None:
    """One call through each proxy with the same key and model ends in a single `LiteLLM_DailyUserSpend` row
    with `api_requests` 2 and spend 0.04, not one row per proxy."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        served(ask(peer, model, key))
        row: Final = eventually(
            lambda: daily_row(gateway, "LiteLLM_DailyUserSpend", key),
            lambda found: found["api_requests"] == 2,
            seconds=LANDING_SECONDS,
        )
        assert number(row["spend"]) == pytest.approx(0.04)


def test_budget_used_up_on_one_instance_is_refused_by_the_other(gateway: Gateway, peer: Gateway) -> None:
    """A key with a tiny `max_budget` spends 0.02 through the first proxy. Once that spend is stored, the second
    proxy refuses the key with HTTP 422 `budget_exceeded` and the provider sees only the first request."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refused: Final = eventually(
            lambda: ask(peer, model, key), lambda response: response.status_code == 422, seconds=LANDING_SECONDS
        )
        assert "udget" in refusal(refused)
        assert len(provider_calls(wire)) == 1


def test_team_budget_used_up_on_one_instance_is_refused_by_the_other(gateway: Gateway, peer: Gateway) -> None:
    """A team with a tiny `max_budget` spends 0.02 through the first proxy with one key. A different key of the
    same team is then refused by the second proxy with a message that names `Team=<team id>`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=TINY_BUDGET)
        first: Final = scenario.key(team_id=team, models=[model])
        second: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, first))
        spend_reaches(gateway, "team", team, CHAT_COST)
        refused: Final = eventually(
            lambda: ask(peer, model, second), lambda response: response.status_code == 422, seconds=LANDING_SECONDS
        )
        assert team in refusal(refused)


def test_key_info_on_the_other_instance_shows_the_spend(gateway: Gateway, peer: Gateway) -> None:
    """A call is made through the first proxy. `GET /key/info` on the second proxy then reports `info.spend`
    0.02 for that key, so the admin UI shows the same number whichever instance answers."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        info: Final = eventually(
            lambda: object_value(peer.get(f"/key/info?key={key}")["info"]),
            lambda found: number(found["spend"]) >= 0.02 - 1e-9,
            seconds=LANDING_SECONDS,
        )
        assert number(info["spend"]) == pytest.approx(0.02)


def test_shared_redis_counter_counts_both_instances(gateway: Gateway, peer: Gateway) -> None:
    """One call through each proxy with the same key. The Redis counter `spend:key:<sha256 of the key>` that
    both proxies read for budget checks reaches 0.04.
    UNVERIFIED: the counter name comes from litellm-rust/crates/redis-store/src/spend_counters/keys.rs."""
    with (
        scripted_upstream() as wire,
        gateway.scenario() as scenario,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        served(ask(peer, model, key))
        counted: Final = eventually(
            lambda: cache.get(f"spend:key:{hashed(key)}"),
            lambda value: value is not None and float(value) >= 0.04 - 1e-9,
            seconds=LANDING_SECONDS,
        )
        assert float(counted) == pytest.approx(0.04)


def test_redis_buffer_proxy_charges_a_call_once(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with `use_redis_transaction_buffer: true` serves one call. The key's spend reaches 0.02, stays
    0.02 for five seconds, and one request row exists. The buffer must not turn one call into two charges."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, REDIS_BUFFER) as owned:
            served(ask(owned, model, key))
            assert spend_reaches(owned, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)
            assert _spend_stays(owned, key, 0.02) == pytest.approx(0.02)
            assert len(rows_of_key(owned, key, 1)) == 1


def test_redis_buffer_proxies_add_up_on_key_team_and_daily_row(gateway: Gateway, tmp_path: Path) -> None:
    """Two proxies, both with `use_redis_transaction_buffer: true`, each serve one call for the same team key.
    The key's spend and the team's spend settle at 0.04, two request rows exist, and the key's
    `LiteLLM_DailyUserSpend` row shows `api_requests` 2 with spend 0.04.
    UNVERIFIED: with the spend ledger on, the Redis buffer may no longer sit on this path at all."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        with (
            _second_proxy(gateway, tmp_path / "first", REDIS_BUFFER) as first,
            _second_proxy(gateway, tmp_path / "second", REDIS_BUFFER) as second,
        ):
            served(ask(first, model, key))
            served(ask(second, model, key))
            assert spend_reaches(first, "key", hashed(key), 0.04) == pytest.approx(0.04)
            assert spend_reaches(second, "team", team, 0.04) == pytest.approx(0.04)
            assert _spend_stays(first, key, 0.04) == pytest.approx(0.04)
            assert len(rows_of_key(first, key, 2)) == 2
            row: Final = eventually(
                lambda: daily_row(first, "LiteLLM_DailyUserSpend", key),
                lambda found: found["api_requests"] == 2,
                seconds=LANDING_SECONDS,
            )
            assert number(row["spend"]) == pytest.approx(0.04)


def test_redis_buffer_proxy_flushes_on_stop(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with the Redis buffer on and a one hour flush timer serves one call and is stopped. The key's
    spend still reaches 0.02 and one request row exists: cost parked in Redis is not stranded by the stop."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        with _second_proxy(gateway, tmp_path, {**SLOW_FLUSH, **REDIS_BUFFER}) as owned:
            served(ask(owned, model, key))
        assert len(rows_of_key(gateway, key, 1)) == 1
        assert spend_reaches(gateway, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)
        assert _spend_stays(gateway, key, 0.02) == pytest.approx(0.02)
