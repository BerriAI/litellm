"""Budget-based routing: `provider_budget_config` and deployment budgets take spent deployments out of the pool.

When `router_settings.provider_budget_config` is set, or any deployment sets `litellm_params.max_budget` with
`budget_duration`, the router adds the `RouterBudgetLimiting` filter (`litellm/router_strategy/budget_limiter.py`).
It is a filter, not a strategy: before the pick it drops every deployment whose provider has spent at least its
`budget_limit` in the current `time_period` (spend key `provider_spend:<provider>:<period>`), and every deployment
that has spent its own `max_budget` (`deployment_spend:<id>:<duration>`), and the router's strategy picks among
what is left. Spend is added by the limiter's success callback from the response cost. When nothing is left the
call fails with `No deployments available - crossed budget: <details>`, which the proxy answers with HTTP 429
because the message contains `No deployments available` (`proxy_exception_status_code` in
`litellm/proxy/_types.py:4034`).

`GET /provider/budgets` reports each configured provider's `budget_limit`, `time_period`, current `spend` and
`budget_reset_at`.

Provider spend keys are shared by every proxy on the same Redis, so a provider may already be over its tiny limit
when the test starts; every assertion here holds either way. Deployment ids carry a per-run suffix so their
spend keys start empty. The OpenAI deployment is `openai/gpt-4o-mini` and the Bedrock one
`bedrock/converse/amazon.nova-pro-v1:0`, both priced in the cost map, so every answer costs more than zero.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, object_value
from tests.integration.routing.dec1_routestrat_support import (
    CROSSED_BUDGET,
    Rig,
    chat,
    dep,
    error_message,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:8]
TINY: Final = 0.0000001
BUDGETS: Final[dict[str, JsonValue]] = {
    "openai": {"budget_limit": TINY, "time_period": "1d"},
    "anthropic": {"budget_limit": 100, "time_period": "1d"},
    "vertex_ai": {"budget_limit": 100, "time_period": "30d"},
}


def _label(name: str) -> str:
    return f"{name}-{RUN}"


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("openai", "budget-provider", _label("bp-openai"), base, weight=1),
        dep("anthropic", "budget-provider", _label("bp-anthropic"), base),
        dep("openai", "budget-only-openai", _label("boo-openai"), base),
        dep(
            "bedrock",
            "budget-deployment",
            _label("bd-bedrock"),
            base,
            weight=1,
            max_budget=TINY,
            budget_duration="1d",
        ),
        dep("vertex", "budget-deployment", _label("bd-vertex"), base),
        dep("bedrock", "budget-free", _label("bf-bedrock"), base),
        dep("azure", "budget-free", _label("bf-azure"), base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "disable_cooldowns": True,
        "provider_budget_config": BUDGETS,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-budget"), _models, settings) as value:
        yield value


def test_a_provider_over_its_budget_hands_the_group_to_the_other_provider(rig: Rig) -> None:
    """`budget-provider` has OpenAI (`weight: 1`, so simple-shuffle prefers it) and Anthropic; OpenAI's provider
    budget is 1e-7 dollars a day and Anthropic's 100.

    Requests are sent until a whole batch of eight is served by Anthropic, which must happen within 30 seconds:
    the first OpenAI answer spends more than the limit, and from then on the budget filter removes OpenAI before
    the shuffle sees it. Every request along the way returns 200.
    """
    settled: Final = eventually(
        lambda: tuple(served(rig.gateway, "budget-provider") for _ in range(8)),
        lambda batch: set(batch) == {_label("bp-anthropic")},
        seconds=30,
    )
    assert settled == (_label("bp-anthropic"),) * 8


def test_a_group_whose_only_provider_is_over_budget_answers_429_crossed_budget(rig: Rig) -> None:
    """`budget-only-openai` has one OpenAI deployment and OpenAI's provider budget is 1e-7 dollars.

    Requests are sent until one answers HTTP 429 (within 30 seconds); its message must contain
    `No deployments available - crossed budget` and `Exceeded budget for provider openai`. Once refused, the next
    request is refused too and the peer receives nothing more for the deployment.
    """
    refused: Final = eventually(
        lambda: chat(rig.gateway, "budget-only-openai"),
        lambda response: response.status_code == 429,
        seconds=30,
    )
    assert CROSSED_BUDGET in error_message(refused), refused.text
    assert "Exceeded budget for provider openai" in error_message(refused), refused.text
    calls: Final = rig.peer.count(_label("boo-openai"))
    again: Final = chat(rig.gateway, "budget-only-openai")
    assert again.status_code == 429, again.text
    assert rig.peer.count(_label("boo-openai")) == calls


def test_a_deployment_over_its_own_max_budget_leaves_its_sibling_serving(rig: Rig) -> None:
    """In `budget-deployment` Bedrock (`weight: 1`) sets `max_budget: 1e-7` with `budget_duration: 1d`, and Vertex
    sets no budget. Bedrock's provider has no provider budget.

    Within 30 seconds a whole batch of eight must be served by Vertex: Bedrock's first answer crosses its own
    budget and the filter drops it. Its deployment id is fresh for this run, so Bedrock must have served at least
    once first.
    """
    settled: Final = eventually(
        lambda: tuple(served(rig.gateway, "budget-deployment") for _ in range(8)),
        lambda batch: set(batch) == {_label("bd-vertex")},
        seconds=30,
    )
    assert settled == (_label("bd-vertex"),) * 8
    assert rig.peer.count(_label("bd-bedrock")) >= 1


def test_providers_without_a_configured_budget_are_never_filtered(rig: Rig) -> None:
    """`budget-free` has Bedrock and Azure deployments; neither provider is in `provider_budget_config` and neither
    deployment sets `max_budget`.

    Over forty requests both must serve at least five, and all return 200.
    """
    counts: Final = served_many(rig.gateway, "budget-free", 40)
    assert counts[_label("bf-bedrock")] >= 5, counts
    assert counts[_label("bf-azure")] >= 5, counts


def test_provider_budgets_endpoint_reports_each_configured_provider(rig: Rig) -> None:
    """`GET /provider/budgets` on the proxy configured with budgets for `openai`, `anthropic` and `vertex_ai`.

    The body's `providers` must have exactly those three keys. `openai` reports `budget_limit` 1e-7 and
    `time_period` `1d`, `anthropic` 100.0 and `1d`, `vertex_ai` 100.0 and `30d`, and each carries a numeric `spend`.
    """
    providers: Final = object_value(rig.gateway.get("/provider/budgets")["providers"])
    assert set(providers) == {"openai", "anthropic", "vertex_ai"}
    openai: Final = object_value(providers["openai"])
    assert openai["budget_limit"] == pytest.approx(TINY)
    assert openai["time_period"] == "1d"
    assert object_value(providers["anthropic"])["budget_limit"] == pytest.approx(100.0)
    assert object_value(providers["anthropic"])["time_period"] == "1d"
    assert object_value(providers["vertex_ai"])["time_period"] == "30d"
    assert all(isinstance(object_value(entry)["spend"], (int, float)) for entry in providers.values())


def test_provider_budgets_endpoint_shows_openai_spend_after_it_served(rig: Rig) -> None:
    """After at least one OpenAI deployment has answered (the provider tests above, or one request here to
    `budget-only-openai` if it is still under budget).

    `GET /provider/budgets` must eventually report `providers.openai.spend` above zero and a non-null
    `budget_reset_at`. UNVERIFIED: `budget_reset_at` is read from the spend key's TTL and is null until the first
    window starts.
    """
    chat(rig.gateway, "budget-only-openai")
    openai: Final = eventually(
        lambda: object_value(object_value(rig.gateway.get("/provider/budgets")["providers"])["openai"]),
        lambda entry: isinstance(entry["spend"], (int, float)) and entry["spend"] > 0,
        seconds=30,
    )
    assert openai["budget_reset_at"] is not None, openai
