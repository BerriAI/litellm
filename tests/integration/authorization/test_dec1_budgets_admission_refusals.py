"""Contract: which requests the budget gate at auth refuses, waives or throttles.

Budget checks run in auth, before routing, guardrails and pre_call hooks (`user_api_key_auth.py`,
`_reserve_budget_after_common_checks`, and `auth_checks.py::request_skips_budget_checks`). Some requests skip
them entirely:

    model discovery routes (`/v1/models`)            never budget checked
    a model whose input and output price are both 0 an exhausted key may still call it (`is_model_cost_zero`)

A key that opted in with `metadata.throttle_on_budget_exceeded: true` and has an RPM or TPM limit is not
refused when it exceeds its own `max_budget` while `litellm_settings.budget_exceeded_throttle_percentage` is set
(0 < pct <= 1). It keeps serving at that percentage of its limits instead (`auth/budget_throttle.py`). Without a
limit to scale, or without the global percentage, it is hard refused as usual.

One served call costs 0.02 (`chat_model` price); 0.0000001 budgets are used up by one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    TINY_BUDGET,
    ask,
    chat_model,
    hashed,
    provider_calls,
    proxy_config,
    refusal,
    scripted_upstream,
    served,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state

_THROTTLE: Final = {"budget_exceeded_throttle_percentage": 0.5}


def test_an_exhausted_key_is_served_on_a_model_whose_prices_are_both_zero(gateway: Gateway) -> None:
    """A key may call a priced model and a model created with `input_cost_per_token: 0` and
    `output_cost_per_token: 0`. After the key's tiny `max_budget` is used up on the priced model it is refused
    there with 422, but a call to the free model is still served and reaches the provider: budget checks are
    skipped for a model that can never cost anything (`request_skips_budget_checks`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        priced: Final = chat_model(scenario, wire)
        free: Final = chat_model(scenario, wire, input_cost_per_token=0.0, output_cost_per_token=0.0)
        key: Final = scenario.key(models=[priced, free], max_budget=TINY_BUDGET)
        served(ask(gateway, priced, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refusal(ask(gateway, priced, key))
        provider_calls(wire)
        served(ask(gateway, free, key))
        assert len(provider_calls(wire)) == 1


def test_a_model_with_only_one_zero_price_is_still_budget_checked(gateway: Gateway) -> None:
    """The waiver needs both prices explicitly 0. A model with `input_cost_per_token: 0` but a nonzero output
    price is refused for an exhausted key like any priced model."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        priced: Final = chat_model(scenario, wire)
        half_free: Final = chat_model(scenario, wire, input_cost_per_token=0.0)
        key: Final = scenario.key(models=[priced, half_free], max_budget=TINY_BUDGET)
        served(ask(gateway, priced, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        assert refusal(ask(gateway, half_free, key)).startswith("Budget has been exceeded! Key=")


def test_an_exhausted_key_can_still_list_models(gateway: Gateway) -> None:
    """An exhausted key calls `GET /v1/models`. Model discovery is never budget checked (`MODEL_DISCOVERY_ROUTES`),
    so the answer is 200 with the key's model in `data`, letting clients render a model picker even when out of
    budget."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refusal(ask(gateway, model, key))
        listed: Final = gateway.request("GET", "/v1/models", key=key)
        assert listed.status_code == 200, listed.text
        assert model in [entry["id"] for entry in listed.json()["data"]], listed.text


def test_a_budget_refusal_happens_before_the_provider_and_before_the_body_model_is_routed(gateway: Gateway) -> None:
    """An exhausted key asks for a model that is in its `models` list. The refusal comes back as HTTP 422 without
    the provider seeing any request, and without the `x-litellm-model-id` header a routed response carries,
    because the budget gate runs in auth before routing."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        provider_calls(wire)
        response: Final = ask(gateway, model, key)
        refusal(response)
        assert "x-litellm-model-id" not in response.headers, dict(response.headers)
        assert provider_calls(wire) == ()


@pytest.mark.timeout(240)
def test_an_opted_in_key_over_budget_is_throttled_instead_of_refused(gateway: Gateway, tmp_path: Path) -> None:
    """On a proxy with `litellm_settings.budget_exceeded_throttle_percentage: 0.5`, a key with `rpm_limit` 4,
    tiny `max_budget` and `metadata.throttle_on_budget_exceeded: true` spends past its budget. Its next requests
    are not refused with 422 `budget_exceeded`: within one minute some are served (200) and, once half of the
    4 RPM is used, the rest get 429 from the rate limiter.
    UNVERIFIED: the exact split depends on where the minute window starts; the test pins no 422 and at least one
    200 and one 429 among five quick requests."""
    config: Final = proxy_config(tmp_path, litellm_settings=_THROTTLE)
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], max_budget=TINY_BUDGET, rpm_limit=4, metadata={"throttle_on_budget_exceeded": True}
        )
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        statuses: Final = [ask(owned, model, key).status_code for _ in range(5)]
        assert 422 not in statuses, statuses
        assert 200 in statuses and 429 in statuses, statuses


@pytest.mark.timeout(240)
def test_an_opted_in_key_without_rate_limits_is_still_refused(gateway: Gateway, tmp_path: Path) -> None:
    """Same proxy and opt in, but the key has no `rpm_limit` and no `tpm_limit`. With nothing to scale down,
    throttling would mean unlimited spend, so the key is hard refused with 422 `budget_exceeded`
    (`should_throttle_budget_exceeded`)."""
    config: Final = proxy_config(tmp_path, litellm_settings=_THROTTLE)
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], max_budget=TINY_BUDGET, metadata={"throttle_on_budget_exceeded": True}
        )
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        assert refusal(ask(owned, model, key)).startswith("Budget has been exceeded! Key=")


def test_an_opted_in_key_is_refused_when_no_throttle_percentage_is_configured(gateway: Gateway) -> None:
    """On the suite proxy, which has no `budget_exceeded_throttle_percentage`, a key with `rpm_limit` 4 and
    `metadata.throttle_on_budget_exceeded: true` is refused with 422 once over budget: opting in alone never
    relaxes enforcement."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], max_budget=TINY_BUDGET, rpm_limit=4, metadata={"throttle_on_budget_exceeded": True}
        )
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        assert refusal(ask(gateway, model, key)).startswith("Budget has been exceeded! Key=")


@pytest.mark.timeout(240)
def test_throttling_applies_only_to_the_keys_own_budget_not_to_its_team(gateway: Gateway, tmp_path: Path) -> None:
    """On the throttling proxy, an opted-in key with `rpm_limit` 4 and no budget of its own belongs to a team with
    a tiny `max_budget`. After the team is used up, the key is hard refused with 422 and the team wording
    (`Team=<team_id>`): the throttle only replaces the key's own `max_budget` refusal."""
    config: Final = proxy_config(tmp_path, litellm_settings=_THROTTLE)
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(max_budget=TINY_BUDGET, models=[model])
        key: Final = scenario.key(
            team_id=team, models=[model], rpm_limit=4, metadata={"throttle_on_budget_exceeded": True}
        )
        served(ask(owned, model, key))
        spend_reaches(owned, "team", team, CHAT_COST)
        assert refusal(ask(owned, model, key)).startswith(f"Budget has been exceeded! Team={team} ")
