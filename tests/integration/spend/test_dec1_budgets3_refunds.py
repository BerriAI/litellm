"""Contract: every path that ends a request gives back the part of the budget hold it did not spend.

The hold is taken at auth (`_reserve_budget_after_common_checks`, `user_api_key_auth.py:4472-4515`), before the
response cache, the rate limiter's pre-call hook, router retries and fallbacks run. So each of those has to end
in a settlement:

- a served call (also a cache hit or a fallback) settles every entry to the logged `response_cost`
  (`proxy_track_cost_callback.py:1389`, `engine.rs::settle`); a cache hit costs 0 (`litellm_logging.py:2429`),
- a failed call settles to 0 through the failure path or `release_or_invalidate_budget_reservation`
  (`budget_reservation.py:445-469`),
- a reservation no logging callback took ownership of is released by `BudgetReservationReleaseMiddleware`
  (`proxy_server.py:3110`, `release_unbound_budget_reservation`),
- a request that is refused before a hold is planned (unknown model, no access) never takes one.

Round 1 pinned the provider 500 release for the key counter alone. This file covers every level, the other end
paths, and the paths that must take no hold. Price: input free, 0.004 per output token, 5 tokens per answer.
"""

from __future__ import annotations

import contextlib
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.caching.dec1_cache_support import CACHE_KEY_HEADER
from tests.integration.caching.response_cache_case import await_entry, redis_store
from tests.integration.spend.dec1_budgets3_support import COST, TOKENS, flaky_provider, levels
from tests.integration.spend.dec1_budgets_support import (
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
)
from tests.integration.spend.dec1_spend_support import (
    chat_model,
    failing,
    hashed,
    provider_calls,
    scripted_upstream,
    served,
    served_id,
    spend_of,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state

ROOMY: Final = 5.0


def test_a_provider_failure_releases_the_hold_on_every_level(gateway: Gateway) -> None:
    """A key under a budgeted team, organization, project, tag and customer (5.0 each) calls a provider that
    answers 500. Afterwards every one of `spend:key`, `spend:team`, `spend:org`, `spend:project`, `spend:tag` and
    `spend:end_user` reads 0 (or is absent): the release settles each entry of the one reservation to 0, not only
    the key's."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        tree: Final = levels(
            gateway, scenario, model, key_budget=ROOMY, team_budget=ROOMY, org_budget=ROOMY,
            project_budget=ROOMY, tag_budget=ROOMY, customer_budget=ROOMY,
        )  # fmt: skip
        response: Final = tree.call()
        assert response.status_code >= 500, response.text
        names: Final = tree.counters()
        for level in ("key", "team", "org", "project", "tag", "end_user"):
            counter_settles(cache, names[level], 0.0)


def test_a_provider_400_releases_the_hold_and_charges_nothing(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 calls a provider that answers 400. The caller gets a 400, the key counter
    returns to 0 and the key's stored spend stays 0: a request the provider rejected as invalid costs nothing."""
    with wire_server(failing(400)) as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        response: Final = call(gateway, model, key, max_tokens=TOKENS)
        assert response.status_code == 400, response.text
        counter_settles(cache, key_counter(key), 0.0)
        assert (spend_of(gateway, "key", hashed(key)) or 0.0) == 0.0


def test_a_cache_hit_settles_its_hold_to_zero(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 sends the same prompt twice. The first call reaches the provider and costs
    0.02; once its cache entry exists the second is answered from the cache (it carries `x-litellm-cache-key`).
    The hit still took a 0.04 hold at auth, and it settles to 0, so the key counter ends at exactly 0.02 and the
    provider saw one call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        text: Final = "dec1 budgets3 cached prompt"
        first: Final = call(gateway, model, key, max_tokens=TOKENS, messages=[{"role": "user", "content": text}])
        await_entry(redis_store(), served_id(first))
        hit: Final = call(gateway, model, key, max_tokens=TOKENS, messages=[{"role": "user", "content": text}])
        served(hit)
        assert hit.headers.get(CACHE_KEY_HEADER), dict(hit.headers)
        counter_settles(cache, key_counter(key), COST)
        assert len(provider_calls(wire)) == 1


def test_a_cache_hit_on_a_nearly_spent_key_is_admitted_with_a_shrunk_hold_and_costs_nothing(gateway: Gateway) -> None:
    """A key with `max_budget` 0.03 spends 0.02 on a first call, leaving 0.01. The identical second request is
    a cache hit: auth admits it with the hold shrunk to 0.01 (the 0.04 estimate does not fit), it is answered
    from the cache, and the counter settles back to 0.02, so the key can still make a third, uncached call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.03)
        text: Final = "dec1 budgets3 nearly spent prompt"
        first: Final = call(gateway, model, key, max_tokens=TOKENS, messages=[{"role": "user", "content": text}])
        await_entry(redis_store(), served_id(first))
        counter_settles(cache, key_counter(key), COST)
        assert served(call(gateway, model, key, max_tokens=TOKENS, messages=[{"role": "user", "content": text}]))
        counter_settles(cache, key_counter(key), COST)
        assert served(call(gateway, model, key, max_tokens=TOKENS))


def test_a_tpm_refusal_after_the_hold_releases_it(gateway: Gateway) -> None:
    """A key has `max_budget` 1.0 and `tpm_limit: 20` and sends `max_tokens: 500`. Auth holds the estimate
    (2.0, shrunk to the 1.0 budget), then the limiter's pre-call hook refuses the call with 429 for tokens. The
    provider is never called and the key counter returns to 0, so a rate limit refusal does not also eat the
    budget. UNVERIFIED: which of the failure hook or the release middleware gives the hold back."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, tpm_limit=20)
        response: Final = call(gateway, model, key, max_tokens=500)
        assert response.status_code == 429, response.text
        counter_settles(cache, key_counter(key), 0.0)
        assert provider_calls(wire) == ()


def test_a_retried_call_settles_to_one_answer_not_one_per_attempt(gateway: Gateway) -> None:
    """The provider fails the first attempt with 500 and answers the second. A key with `max_budget` 1.0 calls
    with `num_retries: 1`. The call is served and the key counter settles to 0.02, the cost of the one answer:
    the hold is settled once, after the router finished retrying."""
    with flaky_provider(failures=1) as provider, gateway.scenario() as scenario, redis_client() as cache:
        assert provider.wire is not None
        model: Final = hold_model(scenario, provider.wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        assert served(call(gateway, model, key, max_tokens=TOKENS, num_retries=1))
        counter_settles(cache, key_counter(key), COST)
        assert provider.seen == 2


def test_a_fallback_answer_settles_to_the_fallback_models_price(gateway: Gateway) -> None:
    """The primary model's provider fails with 500; the request names a fallback model on a working provider
    whose output costs 0.008 a token. The answer comes from the fallback (5 tokens, 0.04) and the key counter
    settles to 0.04: the hold estimated on the primary's price is settled to what was actually charged."""
    with wire_server(failing(500)) as broken, scripted_upstream() as working, gateway.scenario() as scenario:
        primary: Final = hold_model(scenario, broken)
        fallback: Final = chat_model(scenario, working, input_cost_per_token=0.0, output_cost_per_token=0.008)
        key: Final = scenario.key(models=[primary, fallback], max_budget=1.0)
        with redis_client() as cache:
            assert served(call(gateway, primary, key, max_tokens=TOKENS, fallbacks=[fallback]))
            assert counter_settles(cache, key_counter(key), 0.04) == pytest.approx(0.04)


def test_a_call_for_a_model_the_key_may_not_use_takes_no_hold(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 may only call model A and asks for model B. It is refused 403
    `key_model_access_denied` with the generic message of `litellm/proxy/auth/model_access_denied.py:6`, before any
    reservation, so no hold is written: the key counter is either absent or the 0.0 the budget read seeds from the
    stored spend (`SpendCounterReseed.coalesced`, litellm/proxy/db/spend_counter_reseed.py:313), never the 0.04 hold."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        allowed: Final = hold_model(scenario, wire)
        other: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[allowed], max_budget=1.0)
        response: Final = call(gateway, other, key, max_tokens=TOKENS)
        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == "key_model_access_denied"
        assert counter(cache, key_counter(key)) in (None, 0.0)


def test_a_call_for_an_unknown_model_takes_no_hold(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 and no model restriction asks for a model name no deployment serves. No price
    can be found, `estimate_request_max_cost` returns None and no hold is written (`budget_reservation.py:282-291`).
    The key counter still exists at exactly 0.0 after the refusal: the budget check read it before the router
    answered and, on a Redis miss, seeded it from the key's stored spend (`_read_spend_counter_estimate`,
    litellm/proxy/proxy_server.py:3739-3745, `spend_counter_reseed.py:313`)."""
    with gateway.scenario() as scenario, redis_client() as cache:
        key: Final = scenario.key(max_budget=1.0)
        response: Final = call(gateway, "dec1-budgets3-no-such-model", key, max_tokens=TOKENS)
        assert response.status_code in (400, 401, 404), response.text
        assert counter(cache, key_counter(key)) == 0.0


def test_a_non_streamed_call_the_client_abandons_does_not_pin_the_key(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 sends a non-streamed call with `max_tokens: 10` that the provider holds, and the
    client gives up after one second. Once the provider is released the key counter is at most 0.02 (the call
    either completed and settled, or was cancelled and released): the 0.04 hold must not stay on the counter.
    UNVERIFIED: whether a non-streamed handler is cancelled on disconnect (`release_cancelled_budget_reservation`,
    input cost 0 here) or runs to completion (cost 0.02) is not pinned; both are accepted."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        with contextlib.suppress(httpx.TimeoutException):
            gateway.client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "max_tokens": TOKENS,
                    "messages": [{"role": "user", "content": "dec1-budgets-hold abandoned"}],
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=1.0,
            )
        provider.wait_for(1)
        provider.release()
        settled: Final = eventually(
            lambda: counter(cache, key_counter(key)),
            lambda value: any(abs((value or 0.0) - amount) < 1e-9 for amount in (0.0, COST)),
            seconds=70,
        )
        assert (settled or 0.0) <= COST


def test_a_served_call_charges_the_same_amount_to_stored_spend_and_the_counter(gateway: Gateway) -> None:
    """A team key under a team with `max_budget` 1.0 makes one call. The team counter settles to 0.02 and the
    team's stored `spend` reaches 0.02: the settled hold and the database write agree, so a counter reseeded from
    the table later (`spend_counter_reseed.py`) starts where the counter left off."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=1.0)
        key: Final = scenario.key(team_id=team, models=[model])
        assert served(call(gateway, model, key, max_tokens=TOKENS))
        counter_settles(cache, f"spend:team:{team}", COST)
        assert spend_reaches(gateway, "team", team, COST) == pytest.approx(COST)
