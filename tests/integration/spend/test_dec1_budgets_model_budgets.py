"""Contract: per-model budgets (`model_max_budget`) beyond the key, team and user cases already pinned, and the
key's `budget_fallbacks` that reroute a call when a model's budget is used up.

Per-model budgets are checked in `litellm/proxy/hooks/model_max_budget_limiter.py`, from spend recorded after
each call, so the refusal comes on the first call after the spend has landed. Each entity has its own wording:

    end user      LiteLLM End User: <id>, exceeded budget for model=<model>
    team member   LiteLLM Team Member: user=<user>, team=<team>, exceeded budget for model=<model>

A key may carry `budget_fallbacks: {<model>: [<fallback>, ...]}`. When `<model>`'s own key budget is used up,
auth rewrites the request to the first fallback that is still within its budget and that the key may call
(`user_api_key_auth.py`, `_enforce_key_model_budget`), so the provider sees the fallback deployment. If no
fallback qualifies, the original refusal stands.

Every served call costs 0.02 (scripted usage, `chat_model` price).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration.spend.dec1_budgets_support import refused
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    ask,
    chat_model,
    hashed,
    new_customer,
    provider_calls,
    scripted_upstream,
    served,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state


def _tiny(model: str) -> dict[str, dict[str, float | str]]:
    return {model: {"max_budget": TINY_BUDGET, "budget_duration": "1d"}}


def _upstream_models(requests: tuple[object, ...]) -> list[object]:
    return [JSON_OBJECT.validate_json(request.body).get("model") for request in requests]  # type: ignore[attr-defined]


def test_end_user_model_budget_refuses_that_model_for_the_customer(gateway: Gateway) -> None:
    """A customer is created with `model_max_budget` for one model. After one call naming the customer in `user`
    spends on that model, the next such call is refused with HTTP 422 and exactly `LiteLLM End User: <customer>,
    exceeded budget for model=<model>`.
    UNVERIFIED: assumes `/customer/new` stores `model_max_budget` on the customer's budget row and auth copies it
    to `end_user_model_max_budget` (`_apply_budget_limits_to_end_user_params`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        customer: Final = new_customer(gateway, scenario, model_max_budget=_tiny(model))
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, user=customer))
        spend_reaches(gateway, "end_user", customer, CHAT_COST)
        expected: Final = f"LiteLLM End User: {customer}, exceeded budget for model={model}"
        assert (
            eventually(
                lambda: ask(gateway, model, key, user=customer),
                lambda response: response.status_code == 422,
                seconds=LANDING_SECONDS,
            )
            is not None
        )
        assert refused(ask(gateway, model, key, user=customer)) == expected


def test_end_user_model_budget_leaves_other_customers_and_other_models_alone(gateway: Gateway) -> None:
    """With the first customer refused on the budgeted model, a second customer without a model budget is still
    served on that model, and the first customer is still served on a second, unbudgeted model."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        other_model: Final = chat_model(scenario, wire)
        capped: Final = new_customer(gateway, scenario, model_max_budget=_tiny(model))
        free: Final = new_customer(gateway, scenario)
        key: Final = scenario.key(models=[model, other_model])
        served(ask(gateway, model, key, user=capped))
        eventually(
            lambda: ask(gateway, model, key, user=capped).status_code,
            lambda status: status == 422,
            seconds=LANDING_SECONDS,
        )
        assert served(ask(gateway, model, key, user=free))["model"] is not None
        assert served(ask(gateway, other_model, key, user=capped))["model"] is not None


def test_team_member_model_budget_refuses_that_member_on_that_model(gateway: Gateway) -> None:
    """A team is created with `team_member_model_max_budget` for one model. A member's key spends on it once, then
    that member is refused on that model with exactly `LiteLLM Team Member: user=<user_id>, team=<team_id>,
    exceeded budget for model=<model>`, while another member of the team is still served on it.
    UNVERIFIED: the member budget is applied per membership row; whether a member added after the team was
    created inherits it is read from `NewTeamRequest.team_member_model_max_budget` only."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model], team_member_model_max_budget=_tiny(model))
        spender: Final = scenario.member(team)
        other: Final = scenario.member(team)
        spender_key: Final = scenario.key(user_id=spender, team_id=team, models=[model])
        other_key: Final = scenario.key(user_id=other, team_id=team, models=[model])
        served(ask(gateway, model, spender_key))
        eventually(
            lambda: ask(gateway, model, spender_key).status_code,
            lambda status: status == 422,
            seconds=LANDING_SECONDS,
        )
        assert refused(ask(gateway, model, spender_key)) == (
            f"LiteLLM Team Member: user={spender}, team={team}, exceeded budget for model={model}"
        )
        served(ask(gateway, model, other_key))


def test_a_key_model_budget_is_per_key_and_does_not_stop_another_key_on_the_same_model(gateway: Gateway) -> None:
    """Two keys call the same model; only the first has `model_max_budget` for it. After the first key is refused
    on the model, the second key, which never had a model budget, is still served on it: model budgets are kept
    per key (`model_budget_spend_cache_key` includes the key hash), not per model."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        capped: Final = scenario.key(models=[model], model_max_budget=_tiny(model))
        free: Final = scenario.key(models=[model])
        served(ask(gateway, model, capped))
        eventually(
            lambda: ask(gateway, model, capped).status_code, lambda status: status == 422, seconds=LANDING_SECONDS
        )
        assert served(ask(gateway, model, free))["model"] is not None


def test_a_streamed_call_counts_toward_the_key_model_budget(gateway: Gateway) -> None:
    """A key with a tiny `model_max_budget` on one model makes one streamed call (read to the end). Its cost is
    recorded against the model budget like a non-streamed call, so the next call to that model is refused with
    `exceeded budget for model=<model>`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], model_max_budget=_tiny(model))
        streamed: Final = ask(gateway, model, key, stream=True)
        assert streamed.status_code == 200 and "data: [DONE]" in streamed.text, streamed.text
        eventually(lambda: ask(gateway, model, key).status_code, lambda status: status == 422, seconds=LANDING_SECONDS)
        assert refused(ask(gateway, model, key)).endswith(f"exceeded budget for model={model}")


def test_budget_fallback_reroutes_to_the_fallback_model_once_the_model_budget_is_used_up(gateway: Gateway) -> None:
    """A key may call two models A and B, has a tiny `model_max_budget` on A and `budget_fallbacks: {A: [B]}`.
    The first call to A is served by A. Once A's model budget is used up, a call asking for A is served (200, not
    422) and the provider receives B's deployment model (`openai/gpt-4o`), not A's (`gpt-4o-mini`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        primary: Final = chat_model(scenario, wire)
        fallback: Final = chat_model(scenario, wire, model="openai/gpt-4o")
        key: Final = scenario.key(
            models=[primary, fallback],
            model_max_budget=_tiny(primary),
            budget_fallbacks={primary: [fallback]},
        )
        served(ask(gateway, primary, key))
        assert _upstream_models(provider_calls(wire)) == ["gpt-4o-mini"]
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)

        def upstream_after_call() -> list[object]:
            served(ask(gateway, primary, key))
            return _upstream_models(provider_calls(wire))

        eventually(upstream_after_call, lambda models: models == ["gpt-4o"], seconds=LANDING_SECONDS)


def test_budget_fallback_that_is_also_used_up_leaves_the_original_refusal(gateway: Gateway) -> None:
    """Both A and its fallback B have tiny model budgets on the key, and both have been spent on. A call asking
    for A is refused with the original model's message, `LiteLLM Virtual Key: <hashed key>, key_alias: None,
    exceeded budget for model=<A>`, and the provider is not called."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        primary: Final = chat_model(scenario, wire)
        fallback: Final = chat_model(scenario, wire, model="openai/gpt-4o")
        key: Final = scenario.key(
            models=[primary, fallback],
            model_max_budget={**_tiny(primary), **_tiny(fallback)},
            budget_fallbacks={primary: [fallback]},
        )
        served(ask(gateway, primary, key))
        served(ask(gateway, fallback, key))
        eventually(
            lambda: ask(gateway, fallback, key).status_code, lambda status: status == 422, seconds=LANDING_SECONDS
        )
        provider_calls(wire)
        assert refused(ask(gateway, primary, key)) == (
            f"LiteLLM Virtual Key: {hashed(key)}, key_alias: None, exceeded budget for model={primary}"
        )
        assert provider_calls(wire) == ()


def test_budget_fallback_to_a_model_the_key_may_not_call_is_not_used(gateway: Gateway) -> None:
    """The key may call only A, and `budget_fallbacks: {A: [B]}` names a model B outside the key's `models`.
    Once A's budget is used up, a call for A is refused (HTTP 422 for A) instead of being silently rerouted to a
    model the key was never granted (`_is_fallback_model_authorized`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        primary: Final = chat_model(scenario, wire)
        forbidden: Final = chat_model(scenario, wire, model="openai/gpt-4o")
        key: Final = scenario.key(
            models=[primary], model_max_budget=_tiny(primary), budget_fallbacks={primary: [forbidden]}
        )
        served(ask(gateway, primary, key))
        eventually(
            lambda: ask(gateway, primary, key).status_code, lambda status: status == 422, seconds=LANDING_SECONDS
        )
        provider_calls(wire)
        assert refused(ask(gateway, primary, key)).endswith(f"exceeded budget for model={primary}")
        assert provider_calls(wire) == ()


def test_budget_fallback_is_not_used_while_the_primary_model_still_has_budget(gateway: Gateway) -> None:
    """A key has `model_max_budget` 1.0 on A and `budget_fallbacks: {A: [B]}`. While A is within budget every call
    for A goes to A's deployment; the fallback only applies once A is used up."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        primary: Final = chat_model(scenario, wire)
        fallback: Final = chat_model(scenario, wire, model="openai/gpt-4o")
        key: Final = scenario.key(
            models=[primary, fallback],
            model_max_budget={primary: {"max_budget": 1.0, "budget_duration": "1d"}},
            budget_fallbacks={primary: [fallback]},
        )
        served(ask(gateway, primary, key))
        served(ask(gateway, primary, key))
        assert _upstream_models(provider_calls(wire)) == ["gpt-4o-mini", "gpt-4o-mini"]


def test_model_budget_aliases_time_period_and_budget_limit_are_accepted(gateway: Gateway) -> None:
    """`model_max_budget` entries may use the older spelling `{budget_limit, time_period}` instead of `{max_budget,
    budget_duration}` (`BudgetConfig.__init__` maps them). A key created with `{A: {budget_limit: 1e-07,
    time_period: "1d"}}` is refused on A after one call, exactly like the new spelling."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], model_max_budget={model: {"budget_limit": TINY_BUDGET, "time_period": "1d"}}
        )
        served(ask(gateway, model, key))
        eventually(lambda: ask(gateway, model, key).status_code, lambda status: status == 422, seconds=LANDING_SECONDS)
        assert refused(ask(gateway, model, key)).endswith(f"exceeded budget for model={model}")
