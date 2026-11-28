"""Contract: holds across several workers of one proxy, and the reseed of a lost Redis counter from Postgres.

Workers. Every worker of one proxy takes its holds in the same Redis counters, atomically, so a burst of concurrent
calls spread over the workers admits exactly as many as fit the budget. The in-process caches of the workers may
disagree; the counter may not. (Two separate proxies are pinned in `test_dec1_budgets_two_instances.py`.)

Reseed. A spend counter that Redis lost (TTL, eviction, flush) is rebuilt before it is used
(`litellm/proxy/db/spend_counter_reseed.py::SpendCounterReseed.coalesced`): the authoritative spend is read from its
table (`LiteLLM_VerificationToken`, `LiteLLM_TeamTable`, `LiteLLM_UserTable`, `LiteLLM_OrganizationTable`,
`LiteLLM_ProjectTable`, `LiteLLM_TeamMembership`) and written with SET NX, so concurrent seeders on several pods
agree and an existing counter is never overwritten by the table. Key and team window counters have no row; they are
rebuilt from `LiteLLM_BudgetWindowSpend` or the spend logs since the window start (`coalesced_window`).

Price: input free, output 0.004 a token, one served call 0.02 (`hold_model`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.database import write_rows
from tests.integration.spend.dec1_budgets2_support import owned
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
    background,
    budget_message,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    masked,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import hashed, scripted_upstream, served, spend_reaches

pytestmark: Final = pytest.mark.shares_worker_state

WORKERS: Final = 4
BURST: Final = 12


@pytest.fixture(scope="module")
def multi_worker_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """An owned proxy with four uvicorn workers on the shared database and Redis."""
    with (
        gateway_from_environment() as shared,
        owned(shared, tmp_path_factory.mktemp("dec1-budgets2-workers"), workers=WORKERS) as proxy,
    ):
        yield proxy


@pytest.mark.timeout(300)
def test_a_burst_over_several_workers_admits_exactly_what_fits_the_budget(multi_worker_proxy: Gateway) -> None:
    """A key with `max_budget` 0.1 sends 12 concurrent held calls with `max_tokens: 5` (hold 0.02 each) to a proxy
    with four workers, on fresh connections so they spread over the workers. Exactly 5 reach the provider and 7 are
    refused with HTTP 422; while the 5 are in flight the key counter reads exactly 0.1. A worker-local hold would
    admit up to 5 per worker."""
    proxy: Final = multi_worker_proxy
    with held_provider() as provider, proxy.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.1)
        futures: Final = [run(lambda: call(proxy, model, key, hold=True, max_tokens=5)) for _ in range(BURST)]
        provider.wait_for(5)
        refusals: Final = eventually(
            lambda: [future for future in futures if future.done()],
            lambda done: len(done) >= BURST - 5,
            seconds=60,
        )
        assert counter(cache, key_counter(key)) == pytest.approx(0.1)
        provider.release()
        statuses: Final = sorted(future.result(timeout=60).status_code for future in futures)
        assert statuses == [200] * 5 + [422] * 7
        assert provider.arrivals == 5
        assert len(refusals) == BURST - 5


@pytest.mark.timeout(300)
def test_after_a_multi_worker_burst_the_counter_and_stored_spend_agree(multi_worker_proxy: Gateway) -> None:
    """A key with `max_budget` 1.0 sends 8 concurrent calls with `max_tokens: 5` to the four worker proxy. All are
    served; the key counter settles at exactly 0.16 and the stored key spend reaches exactly 0.16. Settlements from
    different workers neither lose nor double an increment."""
    proxy: Final = multi_worker_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        futures: Final = [run(lambda: call(proxy, model, key, max_tokens=5)) for _ in range(8)]
        assert [future.result(timeout=60).status_code for future in futures] == [200] * 8
        assert counter_settles(cache, key_counter(key), 8 * CHAT_COST) == pytest.approx(8 * CHAT_COST)
        assert spend_reaches(proxy, "key", hashed(key), 8 * CHAT_COST) == pytest.approx(8 * CHAT_COST)


@pytest.mark.timeout(300)
def test_every_worker_refuses_a_key_once_its_budget_is_spent(multi_worker_proxy: Gateway) -> None:
    """A key with alias and `max_budget` 0.02 spends it with one call on the four worker proxy. Eight more calls, each
    on a new connection so they land on different workers, are all refused with HTTP 422 and exactly `Budget has
    been exceeded! Key=<alias> (sk-...<last 4>) Current cost: 0.02, Max budget: 0.02`. A worker whose own cache still
    shows spend 0 must not admit it."""
    proxy: Final = multi_worker_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        alias: Final = f"dec1-b2-workers-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(models=[model], max_budget=CHAT_COST, key_alias=alias)
        served(call(proxy, model, key, max_tokens=5))
        counter_settles(cache, key_counter(key), CHAT_COST)
        messages: Final = [
            refused(
                proxy.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": f"again {n}"}], "max_tokens": 5},
                    key=key,
                    headers={"connection": "close"},
                )
            )
            for n in range(8)
        ]
        assert set(messages) == {budget_message(f"Key={alias} ({masked(key)})", "0.02", "0.02")}


@pytest.mark.timeout(300)
def test_a_team_budget_burst_over_several_workers_admits_exactly_what_fits(multi_worker_proxy: Gateway) -> None:
    """A team with `max_budget` 0.06 and one key sends 6 concurrent held calls with `max_tokens: 5` (0.02 each) on
    the four worker proxy. Exactly 3 are served and 3 refused with HTTP 422; the team counter `spend:team:<team_id>`
    reads 0.06 while the 3 are in flight."""
    proxy: Final = multi_worker_proxy
    with held_provider() as provider, proxy.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        team_id: Final = scenario.team(models=[model], max_budget=0.06)
        key: Final = scenario.key(team_id=team_id, models=[model])
        futures: Final = [run(lambda: call(proxy, model, key, hold=True, max_tokens=5)) for _ in range(6)]
        provider.wait_for(3)
        eventually(lambda: sum(future.done() for future in futures), lambda done: done >= 3, seconds=60)
        assert counter(cache, f"spend:team:{team_id}") == pytest.approx(0.06)
        provider.release()
        assert sorted(future.result(timeout=60).status_code for future in futures) == [200] * 3 + [422] * 3


@pytest.mark.timeout(240)
def test_a_lost_key_counter_is_reseeded_from_the_stored_key_spend(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 makes one call; its stored spend reaches 0.02. The test deletes `spend:key:<hash>`
    from Redis. A held call with `max_tokens: 10` then reads 0.06 on the recreated counter (0.02 from
    `LiteLLM_VerificationToken.spend` plus the 0.04 hold). A port that restarts a lost counter at 0 would read 0.04."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        cache.delete(key_counter(key))
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.06)
        provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(240)
def test_a_lost_key_counter_reseeded_at_the_budget_refuses_the_next_call(gateway: Gateway) -> None:
    """A key with alias and `max_budget` 1.0 has its stored spend set to 1.0 directly in `LiteLLM_VerificationToken`
    and its Redis counter deleted. The next call is refused with HTTP 422 and exactly `Budget has been exceeded!
    Key=<alias> (sk-...<last 4>) Current cost: 1.0, Max budget: 1.0`: the counter was rebuilt from the table, not
    from the stale cached key object."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        alias: Final = f"dec1-b2-reseed-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(models=[model], max_budget=1.0, key_alias=alias)
        write_rows('UPDATE "LiteLLM_VerificationToken" SET spend = 1.0 WHERE token = %s', (hashed(key),))
        cache.delete(key_counter(key))
        assert refused(call(gateway, model, key, max_tokens=5)) == budget_message(
            f"Key={alias} ({masked(key)})", "1.0", "1.0"
        )


@pytest.mark.timeout(240)
def test_an_existing_counter_is_not_overwritten_by_the_table(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 makes one call (counter 0.02). The test then sets `LiteLLM_VerificationToken.spend`
    to 0.5 without touching Redis. A held call with `max_tokens: 10` reads 0.06 on the counter, not 0.54: the table
    is read only when the counter is missing, and the seed is SET NX."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        write_rows('UPDATE "LiteLLM_VerificationToken" SET spend = 0.5 WHERE token = %s', (hashed(key),))
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.06)
        provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(240)
def test_a_lost_team_counter_is_reseeded_from_the_team_table(gateway: Gateway) -> None:
    """A team with `max_budget` 1.0 has its stored spend set to 1.5 in `LiteLLM_TeamTable` and `spend:team:<team_id>`
    deleted. A call from its key is refused with HTTP 422 and a message starting `Budget has been exceeded!
    Team=<team_id> Current cost: 1.5`. UNVERIFIED: whether the team auth check or the admission engine refuses first
    (they word the current cost differently; see the budgets round 1 index)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team_id: Final = scenario.team(models=[model], max_budget=1.0)
        key: Final = scenario.key(team_id=team_id, models=[model])
        write_rows('UPDATE "LiteLLM_TeamTable" SET spend = 1.5 WHERE team_id = %s', (team_id,))
        cache.delete(f"spend:team:{team_id}")
        message: Final = refused(call(gateway, model, key, max_tokens=5))
        assert message.startswith(f"Budget has been exceeded! Team={team_id} Current cost: 1.5"), message


@pytest.mark.timeout(240)
def test_a_lost_user_counter_is_reseeded_from_the_user_table(gateway: Gateway) -> None:
    """A user with `max_budget` 1.0 owns a personal key. The user's stored spend is set to 1.0 in `LiteLLM_UserTable`
    and `spend:user:<user_id>` deleted. The next call is refused with HTTP 422 and `error.type` `budget_exceeded`.
    UNVERIFIED: the exact user refusal wording (`checks.rs` user branch) was not pinned here."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user_id: Final = scenario.user(user_role="internal_user", max_budget=1.0)
        key: Final = scenario.key(user_id=user_id, models=[model])
        write_rows('UPDATE "LiteLLM_UserTable" SET spend = 1.0 WHERE user_id = %s', (user_id,))
        cache.delete(f"spend:user:{user_id}")
        assert user_id in refused(call(gateway, model, key, max_tokens=5))


@pytest.mark.timeout(240)
def test_a_lost_key_window_counter_is_rebuilt_from_the_window_spend_not_the_key_row(gateway: Gateway) -> None:
    """A key with `budget_limits: [{budget_duration: 1d, max_budget: 1.0}]` makes one call; then
    `LiteLLM_VerificationToken.spend` is set to 0.7 and `spend:key:<hash>:window:1d` deleted. The next call settles the
    rebuilt window counter at 0.04 (0.02 recorded in the window plus 0.02), not 0.72: window counters are reseeded
    from `LiteLLM_BudgetWindowSpend` or the spend logs since the window start, never from the key's lifetime spend."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}])
        window: Final = f"spend:key:{hashed(key)}:window:1d"
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, window, CHAT_COST)
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        eventually(
            lambda: gateway.rows('SELECT 1 AS found FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)),
            lambda rows: len(rows) == 1,
            seconds=LANDING_SECONDS,
        )
        write_rows('UPDATE "LiteLLM_VerificationToken" SET spend = 0.7 WHERE token = %s', (hashed(key),))
        cache.delete(window)
        served(call(gateway, model, key, max_tokens=5))
        assert counter_settles(cache, window, 2 * CHAT_COST) == pytest.approx(2 * CHAT_COST)
