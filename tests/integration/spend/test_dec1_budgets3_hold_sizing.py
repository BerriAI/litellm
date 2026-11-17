"""Contract: how big the admission hold is for output caps, model ceilings, reasoning prices and budget windows.

The hold is `input tokens * input price + output tokens * multiplier * max(output price, reasoning price)`
(`_max_cost_for_cost_info`, `litellm/proxy/spend_tracking/budget_reservation.py:1215-1275`). Output tokens are the
first of `max_completion_tokens`, `max_tokens`, `max_output_tokens` the body carries (`_OUTPUT_TOKEN_FIELDS`,
line 1490), else 16384, clamped to the model's `max_output_tokens` (`_estimate_output_tokens`, 1468-1488). A
hold of 0 is not taken at all (`reserve_budget_for_request`, 288-291). Rounds 1 and 2 pinned `max_tokens`, `n`,
the 16384 fallback, tiers and the Messages ceiling; this file pins the field precedence, the chat ceiling, the
flat reasoning rate, a zero cap, and holds on budget windows when the entity has no lifetime budget.

Price unless stated: input free, 0.004 per output token (`hold_model`).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.spend.dec1_budgets3_support import (
    COST,
    HOLD,
    TOKENS,
    key_window_counter,
    team_counter,
    team_window_counter,
)
from tests.integration.spend.dec1_budgets_support import (
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
)
from tests.integration.spend.dec1_spend_support import served

pytestmark: Final = pytest.mark.shares_worker_state


def test_max_completion_tokens_wins_over_max_tokens_when_both_are_sent(gateway: Gateway) -> None:
    """A key with `max_budget` 10 holds a call carrying `max_completion_tokens: 5` and `max_tokens: 50`. The key
    counter reads 0.02 (5 * 0.004) while it is in flight, not 0.2: `max_completion_tokens` is first in
    `_OUTPUT_TOKEN_FIELDS`."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=10.0)
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=50, max_completion_tokens=5))
        provider.wait_for(1)
        held: Final = counter(cache, key_counter(key))
        provider.release()
        served(pending.result(timeout=60))
        assert held == pytest.approx(0.02)


def test_a_chat_max_tokens_above_the_model_ceiling_holds_only_the_ceiling(gateway: Gateway) -> None:
    """A key with `max_budget` 1000 holds a chat call with `max_tokens: 100000` on gpt-4o-mini, whose cost map
    `max_output_tokens` is 16384. The key counter reads 16384 * 0.004 = 65.536 while it is in flight, not 400.0:
    a caller cannot pin a team's budget with an absurd cap."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1000.0)
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=100000))
        provider.wait_for(1)
        held: Final = counter(cache, key_counter(key))
        provider.release()
        pending.result(timeout=60)
        assert held == pytest.approx(65.536)


def test_a_flat_reasoning_price_above_the_output_price_sizes_the_hold(gateway: Gateway) -> None:
    """A deployment prices output at 0.004 and `output_cost_per_reasoning_token` at 0.01 a token. A held call
    with `max_tokens: 10` holds 0.1 (10 * 0.01) on the key counter: every output token is held at the dearer of
    the two rates because the reasoning share is unknown before the call (`budget_reservation.py:1264-1270`)."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider, output_cost_per_reasoning_token=0.01)
        key: Final = scenario.key(models=[model], max_budget=10.0)
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        held: Final = counter(cache, key_counter(key))
        provider.release()
        served(pending.result(timeout=60))
        assert held == pytest.approx(0.1)


def test_a_zero_output_cap_on_a_free_input_model_takes_no_hold(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 holds a call with `max_tokens: 0`. Input is free, so the estimate is 0 and no
    hold is written: the key counter does not exist while the call is in flight
    (`reservation_cost <= 0` returns None, `budget_reservation.py:288-291`). The answer still settles 0.02."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=0))
        provider.wait_for(1)
        held: Final = counter(cache, key_counter(key))
        provider.release()
        pending.result(timeout=60)
        assert held is None


def test_a_key_with_only_a_window_holds_on_the_window_counter(gateway: Gateway) -> None:
    """A key has no `max_budget` but a `1d` window of 1.0. A held call with `max_tokens: 10` writes 0.04 to
    `spend:key:<hash>:window:1d` and nothing to `spend:key:<hash>`: windows are planned on their own
    (`_get_budget_limit_counters`, `budget_reservation.py:887-925`), the lifetime counter only with a budget."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}])
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        held: Final = (counter(cache, key_window_counter(key, "1d")), counter(cache, key_counter(key)))
        provider.release()
        served(pending.result(timeout=60))
        assert held[0] == pytest.approx(HOLD)
        assert held[1] is None


def test_a_team_window_and_the_team_budget_hold_the_same_estimate(gateway: Gateway) -> None:
    """A team has `max_budget` 1.0 and a `1d` window of 1.0. A held team call with `max_tokens: 10` writes 0.04 to
    both `spend:team:<id>` and `spend:team:<id>:window:1d`, and after the answer both settle to 0.02."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        team: Final = scenario.team(
            models=[model], max_budget=1.0, budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}]
        )
        key: Final = scenario.key(team_id=team, models=[model])
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        held: Final = (counter(cache, team_counter(team)), counter(cache, team_window_counter(team, "1d")))
        provider.release()
        served(pending.result(timeout=60))
        assert held == (pytest.approx(HOLD), pytest.approx(HOLD))
        counter_settles(cache, team_counter(team), COST)
        counter_settles(cache, team_window_counter(team, "1d"), COST)


def test_a_window_with_less_room_than_the_lifetime_budget_shrinks_the_hold(gateway: Gateway) -> None:
    """A key has `max_budget` 1.0 and a `1d` window of 0.03. A held call with `max_tokens: 10` (0.04) is shrunk
    by the window, planned right after the lifetime counter: both `spend:key:<hash>` and its `:window:1d` counter
    read 0.03 while the call is in flight."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(
            models=[model], max_budget=1.0, budget_limits=[{"budget_duration": "1d", "max_budget": 0.03}]
        )
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        held: Final = (counter(cache, key_counter(key)), counter(cache, key_window_counter(key, "1d")))
        provider.release()
        served(pending.result(timeout=60))
        assert held == (pytest.approx(0.03), pytest.approx(0.03))


def test_a_streamed_call_holds_the_same_estimate_as_a_plain_one(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 holds a `stream: true` call with `max_tokens: 10` and
    `stream_options.include_usage`. The key counter reads exactly 0.04, the same as the non-streamed call: the
    estimate ignores `stream` and `stream_options` (`_max_cost_for_cost_info` reads neither)."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        pending: Final = run(
            lambda: call(
                gateway, model, key, hold=True, max_tokens=TOKENS, stream=True, stream_options={"include_usage": True}
            )
        )
        provider.wait_for(1)
        held: Final = counter(cache, key_counter(key))
        provider.release()
        pending.result(timeout=60)
        assert held == pytest.approx(HOLD)
