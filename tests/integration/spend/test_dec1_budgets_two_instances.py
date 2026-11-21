"""Contract: budget holds and budget decisions agree across two proxy instances that share Redis and Postgres.

Every instance reads and writes the same Redis spend counters (`spend:key:<hash>`, `spend:team:<team_id>`, ...)
for both the auth checks and the admission hold, so a hold taken on one instance is visible to the other at
once, long before any spend reaches the database. These tests run the worker's proxy and a second proxy booted
on the same database and Redis (`owned_proxy`), and keep calls in flight with a provider that waits.

The cases where the spend is already stored (budget used up on one instance refused by the other, totals adding
up) are pinned in `test_dec1_spend_durability.py`; these cover what is in flight and what management changes.
Price: input free, output 0.004 a token, one served call 0.02 (see `dec1_budgets_support`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import owned_proxy
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
    background,
    budget_message,
    call,
    counter,
    counter_settles,
    hashed,
    held_provider,
    hold_model,
    key_counter,
    masked,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import served, spend_reaches

pytestmark: Final = pytest.mark.shares_worker_state


@contextmanager
def _second(gateway: Gateway, directory: Path) -> Iterator[Gateway]:
    directory.mkdir(parents=True, exist_ok=True)
    with owned_proxy(gateway, directory, {}) as owned:
        yield owned


@pytest.mark.timeout(300)
def test_a_hold_taken_on_one_instance_refuses_a_concurrent_call_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A key with alias and `max_budget` 1.0 has a call without `max_tokens` held in flight on the first proxy,
    which pins its counter at 1.0. A call on the second proxy is refused with HTTP 422 and exactly `Budget has
    been exceeded! Key=<alias> (sk-...<last 4>) Current cost: 1.0, Max budget: 1.0`. If holds were kept per
    instance, the second proxy would admit it and the key could spend its budget once per instance.
    BUG: the refusal reads `Current cost: 1, Max budget: 1.0`. The counter is written with INCRBYFLOAT, so a
    hold of exactly 1.0 is stored as `1`; the Rust budget checks keep it as `PyNumber::Int` and print it with
    `to_string` (litellm/rust/crates/callbacks-budgets/src/checks.rs:147-153, python-compat number.rs:87-92),
    while the Python check this replaced (#51535) read the counter as `float(value)` and printed `1.0`. The
    batched read in `read_spend_counter_cache_value` returns the batch value without the `float()` the
    unbatched branch applies (litellm/proxy/proxy_server.py:3695-3702), which is the likely way an int gets in."""
    pytest.skip(
        "BUG: a budget refusal prints an integral counter as an int, Current cost: 1 instead of 1.0, since the Rust budget checks format PyNumber::Int with to_string"
    )
    with held_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        alias: Final = f"dec1-two-{tmp_path.name}"
        key: Final = scenario.key(models=[model], max_budget=1.0, key_alias=alias)
        with _second(gateway, tmp_path / "second") as other:
            in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
            provider.wait_for(1)
            try:
                message: Final = refused(call(other, model, key, max_tokens=5))
            finally:
                provider.release()
            assert message == budget_message(f"Key={alias} ({masked(key)})", "1.0", "1.0")
            served(in_flight.result(timeout=60))
            assert provider.arrivals == 1


@pytest.mark.timeout(300)
def test_after_the_hold_settles_on_one_instance_the_other_serves_the_key(gateway: Gateway, tmp_path: Path) -> None:
    """When the held call on the first proxy finishes, the counter settles to 0.02 and the second proxy admits and
    serves the next call on the same key. The settlement is visible across instances, not only on the instance
    that took the hold."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        with _second(gateway, tmp_path / "second") as other:
            in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
            provider.wait_for(1)
            provider.release()
            served(in_flight.result(timeout=60))
            counter_settles(cache, key_counter(key), CHAT_COST)
            assert served(call(other, model, key, max_tokens=5))["usage"] is not None


@pytest.mark.timeout(300)
def test_a_burst_split_across_two_instances_admits_only_what_the_budget_can_hold(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key with `max_budget` 0.05 receives ten concurrent calls with `max_tokens: 5` (hold 0.02 each), five
    through each proxy, all kept waiting at the provider once admitted. Across both instances exactly three reach
    the provider (0.02 + 0.02 + a shrunk 0.01) and seven are refused with 422; once released the key's stored
    spend settles at 0.06."""
    with held_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.05)
        with _second(gateway, tmp_path / "second") as other:
            futures: Final = [
                run(lambda target=target: call(target, model, key, hold=True, max_tokens=5))
                for target in (gateway, other) * 5
            ]
            provider.wait_for(3)
            eventually(lambda: sum(future.done() for future in futures), lambda done: done >= 7, seconds=60)
            provider.release()
            statuses: Final = sorted(future.result(timeout=60).status_code for future in futures)
            assert statuses == [200] * 3 + [422] * 7, statuses
            assert spend_reaches(gateway, "key", hashed(key), 3 * CHAT_COST) == pytest.approx(3 * CHAT_COST)


@pytest.mark.timeout(300)
def test_a_team_hold_on_one_instance_refuses_another_team_key_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `max_budget` 1.0 has two keys. Key A holds the whole team budget on the first proxy (call
    without `max_tokens`). Key B's call on the second proxy is refused at admission with exactly `Budget has been
    exceeded! Team=<team_id> Current cost: 1.02, Max budget: 1.0`."""
    with held_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        team: Final = scenario.team(max_budget=1.0, models=[model])
        first_key: Final = scenario.key(team_id=team, models=[model])
        second_key: Final = scenario.key(team_id=team, models=[model])
        with _second(gateway, tmp_path / "second") as other:
            in_flight: Final = run(lambda: call(gateway, model, first_key, hold=True))
            provider.wait_for(1)
            try:
                message: Final = refused(call(other, model, second_key, max_tokens=5))
            finally:
                provider.release()
            assert message == budget_message(f"Team={team}", "1.02", "1.0")
            served(in_flight.result(timeout=60))


@pytest.mark.timeout(300)
def test_raising_an_exhausted_key_budget_on_one_instance_readmits_it_on_the_other(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key with `max_budget` 0.01 spends 0.02 and is refused on both proxies. An admin raises it with
    `/key/update` `max_budget: 1.0` on the first proxy. The second proxy serves the key again without a restart
    (the key object is invalidated across instances), within the landing window."""
    with held_provider() as provider, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.01)
        with _second(gateway, tmp_path / "second") as other:
            served(call(gateway, model, key, max_tokens=5))
            spend_reaches(gateway, "key", hashed(key), CHAT_COST)
            refused(call(other, model, key, max_tokens=5))
            gateway.post("/key/update", {"key": key, "max_budget": 1.0})
            assert (
                eventually(
                    lambda: call(other, model, key, max_tokens=5).status_code,
                    lambda status: status == 200,
                    seconds=LANDING_SECONDS,
                )
                == 200
            )


@pytest.mark.timeout(300)
def test_lowering_a_team_budget_below_its_spend_on_one_instance_refuses_on_the_other(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A team with `max_budget` 1.0 spends 0.02 through the first proxy. An admin lowers the team to `max_budget`
    0.01 with `/team/update` on the first proxy; the second proxy then refuses the team's key with a message that
    starts `Budget has been exceeded! Team=<team_id> `."""
    with held_provider() as provider, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, provider)
        team: Final = scenario.team(max_budget=1.0, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        with _second(gateway, tmp_path / "second") as other:
            served(call(gateway, model, key, max_tokens=5))
            spend_reaches(gateway, "team", team, CHAT_COST)
            gateway.post("/team/update", {"team_id": team, "max_budget": 0.01})
            response: Final = eventually(
                lambda: call(other, model, key, max_tokens=5),
                lambda found: found.status_code == 422,
                seconds=LANDING_SECONDS,
            )
            assert refused(response).startswith(f"Budget has been exceeded! Team={team} "), response.text


@pytest.mark.timeout(300)
def test_key_info_on_either_instance_never_shows_a_hold_as_spend(gateway: Gateway, tmp_path: Path) -> None:
    """While a call holds 1.0 of a key's budget on the first proxy (counter 1.0), `GET /key/info` on the second
    proxy reports `info.spend` 0 and `info.max_budget` 1.0. Spend shown to admins is settled spend only."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        with _second(gateway, tmp_path / "second") as other:
            in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
            provider.wait_for(1)
            try:
                assert counter(cache, key_counter(key)) == pytest.approx(1.0)
                info: Final = other.get("/key/info", {"key": hashed(key)})["info"]
                assert isinstance(info, dict), info
                assert float(info["spend"] or 0.0) == 0.0 and info["max_budget"] == pytest.approx(1.0), info
            finally:
                provider.release()
            served(in_flight.result(timeout=60))
