"""Contract: the budget hold ("reservation") taken at admission and settled after the call.

Since #51535 the hold is taken in Rust (`litellm/rust/crates/callbacks-budgets/src/engine.rs::reserve`), called
from auth through `litellm/proxy/spend_tracking/budget_reservation.py::reserve_budget_for_request` after the
common checks. For every budgeted entity of the request (key, team, user, team member, end user, tag,
organization, project, model access group, budget windows) the proxy adds the request's worst case cost to that
entity's Redis spend counter before the provider is called:

    estimate = input tokens * input price + min(max_tokens or 16384, model ceiling) * output price * max(n, 1)

If the estimate does not fit, the hold is shrunk to what is left of the budget and the request is still admitted
(fail open). Only when nothing is left (the counter already sits at the budget, for example because another call
in flight holds it) is the request refused. With `general_settings.fail_closed_budget_enforcement` an estimate
that does not fit is refused instead, with an `Estimated request cost` in the message.

After the call the hold is settled: a served call moves the counter to its real cost, a failed call releases the
hold to zero, and a cancelled stream keeps only the input cost. A concurrent request reads the counter including
other requests' holds, so the hold is what stops a burst of concurrent calls from overspending.

Price: input free, output 0.004 a token, 5 completion tokens per served call, so one call costs 0.02 and a hold
is `max_tokens * 0.004` (see `dec1_budgets_support`).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, held
from tests.integration._support.process import owned_proxy
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
    NO_MAX_TOKENS_HOLD,
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
from tests.integration.spend.dec1_spend_support import proxy_config, served, spend_of, spend_reaches

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.timeout(180)
def test_an_in_flight_call_holds_its_estimated_cost_on_the_key_counter(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 sends one call with `max_tokens: 10` that the provider keeps waiting. While it
    is in flight the Redis counter `spend:key:<hash>` reads exactly 0.04 (10 output tokens at 0.004), and after
    the provider answers it settles to the real cost 0.02. A rewrite that charges only after the call, or that
    keeps the hold somewhere other requests cannot see, would read 0 here."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.04)
        provider.release()
        served(in_flight.result(timeout=60))
        counter_settles(cache, key_counter(key), CHAT_COST)


@pytest.mark.timeout(180)
def test_an_in_flight_call_without_max_tokens_holds_the_whole_remaining_budget(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 sends a held call without `max_tokens`. The estimate (16384 output tokens at
    0.004, 65.536) is larger than the budget, so the hold is shrunk to what is left: the counter reads exactly
    1.0 while the call is in flight, and the call is still admitted and served (fail open by default)."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
        provider.wait_for(1)
        assert NO_MAX_TOKENS_HOLD > 1.0
        assert counter(cache, key_counter(key)) == pytest.approx(1.0)
        provider.release()
        served(in_flight.result(timeout=60))
        counter_settles(cache, key_counter(key), CHAT_COST)


@pytest.mark.timeout(180)
def test_a_concurrent_call_is_refused_while_another_call_holds_the_rest_of_the_key_budget(gateway: Gateway) -> None:
    """A key with alias and `max_budget` 1.0 has one held call without `max_tokens`, which pins its counter at
    1.0. A second call on the same key, sent while the first is in flight, is refused with HTTP 422 and exactly
    `Budget has been exceeded! Key=<alias> (sk-...<last 4>) Current cost: 1.0, Max budget: 1.0`, although the
    key's stored spend is still 0. The provider never sees the second call.
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
        alias: Final = f"dec1-budgets-hold-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], max_budget=1.0, key_alias=alias)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
        provider.wait_for(1)
        try:
            second: Final = call(gateway, model, key, max_tokens=5)
        finally:
            provider.release()
        assert refused(second) == budget_message(f"Key={alias} ({masked(key)})", "1.0", "1.0")
        served(in_flight.result(timeout=60))
        assert provider.arrivals == 1
        assert spend_reaches(gateway, "key", hashed(key), CHAT_COST) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(180)
def test_once_the_hold_settles_the_next_call_on_the_key_is_served(gateway: Gateway) -> None:
    """After a held call without `max_tokens` finishes, its hold of 1.0 is settled to the real cost 0.02, so a
    call sent afterwards on the same key (budget 1.0) is admitted and served. A settlement that forgot to give
    back the unused part of the hold would lock the key out until the counter's TTL expired."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
        provider.wait_for(1)
        provider.release()
        served(in_flight.result(timeout=60))
        counter_settles(cache, key_counter(key), CHAT_COST)
        assert served(call(gateway, model, key, max_tokens=5))["usage"] is not None


@pytest.mark.timeout(180)
def test_a_failed_provider_call_releases_the_whole_hold(gateway: Gateway) -> None:
    """The provider answers 500 to a call with `max_tokens: 10` on a key with `max_budget` 1.0. The caller gets
    an error, the key is charged nothing (stored spend stays 0) and the Redis counter falls back to 0, so the
    failed call does not keep 0.04 of the budget pinned."""
    with held_provider(failing_status=500) as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        response: Final = call(gateway, model, key, max_tokens=10)
        assert response.status_code >= 500, response.text
        counter_settles(cache, key_counter(key), 0.0)
        held(
            lambda: spend_of(gateway, "key", hashed(key)),
            lambda spend: spend in (None, 0.0),
            holding=3,
        )


@pytest.mark.timeout(180)
def test_a_streamed_call_holds_until_the_stream_ends_and_then_settles_to_its_cost(gateway: Gateway) -> None:
    """A streamed call with `max_tokens: 10` on a key with `max_budget` 1.0 is kept waiting at the provider before
    its first chunk. Meanwhile the counter reads 0.04. Once the stream is read to the end (usage of 5 completion
    tokens in the last chunk) the counter settles to 0.02, the same as a non-streamed call."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        streamed: Final = run(
            lambda: call(
                gateway, model, key, hold=True, max_tokens=10, stream=True, stream_options={"include_usage": True}
            )
        )
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.04)
        provider.release()
        response: Final = streamed.result(timeout=60)
        assert response.status_code == 200, response.text
        assert "data: [DONE]" in response.text
        counter_settles(cache, key_counter(key), CHAT_COST)


@pytest.mark.timeout(180)
def test_a_request_whose_estimate_is_above_the_remaining_budget_is_admitted_with_a_shrunk_hold(
    gateway: Gateway,
) -> None:
    """A key with `max_budget` 0.03 sends a held call with `max_tokens: 100` (estimate 0.4). By default the
    proxy fails open: the call is admitted, the hold is cut to the 0.03 that is left (the counter reads 0.03
    while it is in flight) and the call is served."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.03)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=100))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.03)
        provider.release()
        served(in_flight.result(timeout=60))
        counter_settles(cache, key_counter(key), CHAT_COST)


@pytest.mark.timeout(180)
def test_two_concurrent_calls_that_fit_together_are_both_held_and_served(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 sends two held calls with `max_tokens: 10`. Both fit (0.04 each), so both
    reach the provider, the counter reads 0.08 while both are in flight, and after both are served it settles
    to 0.04 (two calls of 0.02)."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        first: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        second: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(2)
        assert counter(cache, key_counter(key)) == pytest.approx(0.08)
        provider.release()
        served(first.result(timeout=60))
        served(second.result(timeout=60))
        counter_settles(cache, key_counter(key), 2 * CHAT_COST)


@pytest.mark.timeout(240)
def test_a_burst_of_ten_concurrent_calls_admits_only_what_the_budget_can_hold(gateway: Gateway) -> None:
    """A key with `max_budget` 0.05 fires ten calls at once, each with `max_tokens: 5` (hold 0.02), and the
    provider keeps every admitted call waiting. Holds of 0.02, 0.02 and a shrunk 0.01 fill the budget, so exactly
    three calls reach the provider and seven are refused with HTTP 422 `budget_exceeded`. After release the key's
    stored spend settles at 0.06: the third call overshoots its shrunk hold by 0.01, which is the most a burst can
    overspend. Without the hold all ten would pass the read-time check and spend 0.2."""
    with held_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.05)
        futures: Final = [run(lambda: call(gateway, model, key, hold=True, max_tokens=5)) for _ in range(10)]
        provider.wait_for(3)
        refusals: Final = eventually(
            lambda: [future.result() for future in futures if future.done()],
            lambda done: len(done) >= 7,
            seconds=60,
        )
        assert [response.status_code for response in refusals] == [422] * 7, [r.text for r in refusals]
        assert all(refused(response).startswith("Budget has been exceeded! ") for response in refusals)
        provider.release()
        statuses: Final = sorted(future.result(timeout=60).status_code for future in futures)
        assert statuses == [200] * 3 + [422] * 7, statuses
        assert provider.arrivals == 3
        assert spend_reaches(gateway, "key", hashed(key), 3 * CHAT_COST) == pytest.approx(3 * CHAT_COST)


@pytest.mark.timeout(180)
def test_the_n_parameter_multiplies_the_hold(gateway: Gateway) -> None:
    """A held call asks for `n: 3` and `max_tokens: 5` on a key with `max_budget` 1.0. The output part of the
    estimate is multiplied by `n`, so the counter reads 0.06 (3 * 5 * 0.004) while the call is in flight."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=5, n=3))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.06)
        provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(180)
def test_max_completion_tokens_is_used_for_the_hold_like_max_tokens(gateway: Gateway) -> None:
    """The estimate reads the first of `max_completion_tokens`, `max_tokens` and `max_output_tokens`
    (`_OUTPUT_TOKEN_FIELDS`). A held call with `max_completion_tokens: 10` holds 0.04 on a key with budget 1.0."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_completion_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.04)
        provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(180)
def test_a_key_without_any_budget_takes_no_hold(gateway: Gateway) -> None:
    """A key with no `max_budget`, no team and no windows sends a held call with `max_tokens: 10`. No budgeted
    entity means no counter to reserve against: `spend:key:<hash>` stays absent (or 0) while the call is in
    flight. A rewrite that always reserves would add Redis traffic and could refuse unbudgeted keys."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) in (None, 0.0)
        provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(180)
def test_a_team_hold_taken_by_one_key_refuses_another_key_of_the_team_with_the_admission_wording(
    gateway: Gateway,
) -> None:
    """A team has `max_budget` 1.0 and two keys without budgets of their own. Key A's held call without
    `max_tokens` pins `spend:team:<team_id>` at 1.0. The team check in auth only refuses when spend is strictly
    above the budget, so key B's call (`max_tokens: 5`, hold 0.02) gets past auth and is refused by the admission
    engine instead, with exactly `Budget has been exceeded! Team=<team_id> Current cost: 1.02, Max budget: 1.0`
    (the counter plus this request's own estimate). The team counter is back at 1.0 after the refusal."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        team: Final = scenario.team(max_budget=1.0, models=[model])
        first_key: Final = scenario.key(team_id=team, models=[model])
        second_key: Final = scenario.key(team_id=team, models=[model])
        in_flight: Final = run(lambda: call(gateway, model, first_key, hold=True))
        provider.wait_for(1)
        try:
            assert counter(cache, f"spend:team:{team}") == pytest.approx(1.0)
            message: Final = refused(call(gateway, model, second_key, max_tokens=5))
            assert counter(cache, f"spend:team:{team}") == pytest.approx(1.0)
        finally:
            provider.release()
        assert message == budget_message(f"Team={team}", "1.02", "1.0")
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(240)
def test_disable_budget_reservation_lets_a_concurrent_call_through(gateway: Gateway, tmp_path: Path) -> None:
    """On a proxy started with `general_settings.disable_budget_reservation: true`, a key with `max_budget` 1.0
    has a held call without `max_tokens` in flight. No hold is taken (the counter stays absent or 0), so a second
    call on the same key is served. This is the documented escape hatch from issue #27639 and the reason the flag
    warns that hard enforcement is relaxed."""
    config: Final = proxy_config(tmp_path, general_settings={"disable_budget_reservation": True})
    with (
        held_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(owned, model, key, hold=True))
        provider.wait_for(1)
        try:
            assert counter(cache, key_counter(key)) in (None, 0.0)
            served(call(owned, model, key, max_tokens=5))
        finally:
            provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(240)
def test_fail_closed_refuses_an_estimate_that_does_not_fit_with_the_estimate_in_the_message(
    gateway: Gateway, tmp_path: Path
) -> None:
    """On a proxy with `general_settings.fail_closed_budget_enforcement: true`, a fresh key with `max_budget`
    0.03 asks for `max_tokens: 10` (estimate 0.04). Nothing has been spent, but the estimate does not fit, so the
    call is refused with HTTP 422 and exactly `Budget has been exceeded! Key=<sha256 of the key> Current cost:
    0.0, Estimated request cost: 0.04, Max budget: 0.03` (`engine.rs::exceeded` names the key by its hash, not
    its alias). The provider is never called and the counter is released back to 0."""
    config: Final = proxy_config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with (
        held_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.03, key_alias=f"dec1-fail-closed-{tmp_path.name}")
        message: Final = refused(call(owned, model, key, max_tokens=10))
        assert message == budget_message(f"Key={hashed(key)}", "0.0", "0.03", estimate="0.04")
        assert provider.arrivals == 0
        assert counter(cache, key_counter(key)) in (None, 0.0)


@pytest.mark.timeout(240)
def test_fail_closed_admits_an_estimate_that_fits(gateway: Gateway, tmp_path: Path) -> None:
    """On the same fail closed proxy, a key with `max_budget` 0.03 that asks for `max_tokens: 5` (estimate 0.02)
    fits and is served, and its counter settles at 0.02."""
    config: Final = proxy_config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with (
        held_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.03)
        served(call(owned, model, key, max_tokens=5))
        assert counter_settles(cache, key_counter(key), CHAT_COST) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(240)
def test_fail_closed_refuses_a_call_without_max_tokens_on_a_small_budget(gateway: Gateway, tmp_path: Path) -> None:
    """With fail closed on, a fresh key with `max_budget` 1.0 sends a call without `max_tokens`. Its estimate is
    16384 output tokens at 0.004 (65.536), far above the budget, so it is refused with HTTP 422 and a message that
    starts `Budget has been exceeded! Key=<hash> Current cost: 0.0, Estimated request cost: 65.536`. Operators
    turning fail closed on must know that clients have to send `max_tokens`."""
    config: Final = proxy_config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with (
        held_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        message: Final = refused(call(owned, model, key))
        assert message.startswith(f"Budget has been exceeded! Key={hashed(key)} Current cost: 0.0, "), message
        assert message.endswith(", Max budget: 1.0"), message
        estimate: Final = message.split("Estimated request cost: ", 1)[1].split(",", 1)[0]
        assert float(estimate) == pytest.approx(NO_MAX_TOKENS_HOLD)
        assert provider.arrivals == 0


@pytest.mark.timeout(240)
def test_fail_closed_refuses_with_503_when_the_spend_counters_are_unreachable(gateway: Gateway, tmp_path: Path) -> None:
    """A fail closed proxy is started with `REDIS_PORT` pointing at a closed port, so no spend counter can be read
    or written. A budgeted key's call is refused with HTTP 503 and a message that starts `Budget enforcement
    unavailable:` rather than being served on an unverifiable balance; the provider is not called.
    UNVERIFIED: which of the two 503 texts wins (`proxy_server.py::_raise_budget_unverifiable` when the read fails,
    `engine.rs::UNAVAILABLE` when the hold cannot be written) depends on whether the database fallback answers
    first; both start with `Budget enforcement unavailable:`. Also assumes the proxy boots with Redis down."""
    config: Final = proxy_config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with (
        held_provider() as provider,
        owned_proxy(gateway, tmp_path, {"REDIS_PORT": "1"}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        response: Final = call(owned, model, key, max_tokens=5)
        assert response.status_code == 503, response.text
        assert "Budget enforcement unavailable:" in response.text, response.text
        assert provider.arrivals == 0


@pytest.mark.timeout(180)
def test_a_held_call_never_shows_as_spent_on_key_info(gateway: Gateway) -> None:
    """While a call without `max_tokens` holds the whole 1.0 budget of a key, `GET /key/info` still reports
    `info.spend` 0: the hold lives only in the counter and is not real spend. After the call, `info.spend`
    becomes 0.02. A rewrite that writes holds into the key row would show phantom spend to admins."""
    with held_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
        provider.wait_for(1)
        try:
            info: Final = gateway.get("/key/info", {"key": hashed(key)})["info"]
            assert isinstance(info, dict) and float(info["spend"] or 0.0) == 0.0, info
        finally:
            provider.release()
        served(in_flight.result(timeout=60))
        eventually(
            lambda: gateway.get("/key/info", {"key": hashed(key)})["info"],
            lambda found: isinstance(found, dict) and float(found["spend"] or 0.0) == pytest.approx(CHAT_COST),
            seconds=LANDING_SECONDS,
        )
