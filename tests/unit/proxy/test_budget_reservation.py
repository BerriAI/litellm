from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.redis.stores.auth_cache import AuthObject
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TagTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.reset_budget_job import _model_access_group_counter_key
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import (
    _get_model_access_group_budget_counters,
    estimate_request_input_cost,
    estimate_request_max_cost,
    get_budget_window_start,
    release_budget_reservation_on_cancel,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore, model_access_group_counter
from litellm.router import Router
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from tests.unit.proxy.rust_auth_stores import GroupBudget, RustAuthStores, install_rust_auth_stores
from tests.unit.proxy.spend_tracking.native_budgets_replica import install_native_owner
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


@pytest.fixture()
def spend_counter_state():
    import litellm.proxy.proxy_server as ps

    original_counter_store = ps.spend_counter_store
    original_key_cache = ps.user_api_key_cache
    original_prisma_client = ps.prisma_client

    counter_cache = FakeSpendCounters(redis=None)
    key_cache = UserApiKeyCache()
    ps.spend_counter_store = counter_cache.store()
    ps.user_api_key_cache = key_cache
    ps.prisma_client = None

    try:
        yield counter_cache, key_cache
    finally:
        ps.spend_counter_store = original_counter_store
        ps.user_api_key_cache = original_key_cache
        ps.prisma_client = original_prisma_client


def _request_body() -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }


def test_should_not_serialize_budget_reservation_on_user_api_key_auth():
    auth = UserAPIKeyAuth(
        token="key-budget-runtime-state",
        budget_reservation={
            "reserved_cost": 0.5,
            "entries": [{"counter_key": "spend:key:key-budget-runtime-state"}],
        },
    )

    assert "budget_reservation" not in auth.model_dump()
    assert "budget_reservation" not in auth.model_dump(exclude_none=True)
    assert "budget_reservation" not in auth.model_dump_json()


@pytest.mark.asyncio
async def test_should_seed_and_update_end_user_and_tag_counters_without_reservation(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key=AuthObject.end_user("customer-1"),
        value=LiteLLM_EndUserTable(
            user_id="customer-1",
            blocked=False,
            spend=4.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0),
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key=AuthObject.tag("paid-tag"),
        value=LiteLLM_TagTable(
            tag_name="paid-tag",
            spend=7.0,
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key=AuthObject.tag("other-tag"),
        value=LiteLLM_TagTable(
            tag_name="other-tag",
            spend=2.0,
        ).model_dump(),
    )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.50,
        end_user_id="customer-1",
        tags=["paid-tag", "paid-tag", "other-tag", ""],
    )

    assert counter_cache.get_local("spend:end_user:customer-1") == pytest.approx(4.50)
    assert counter_cache.get_local("spend:tag:paid-tag") == pytest.approx(7.50)
    assert counter_cache.get_local("spend:tag:other-tag") == pytest.approx(2.50)


@pytest.mark.asyncio
async def test_should_seed_org_counter_from_with_budget_cache(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key=AuthObject.org_with_budget("org-counter-with-budget"),
        value=LiteLLM_OrganizationTable(
            organization_id="org-counter-with-budget",
            organization_alias="shared-org",
            budget_id="org-budget-id",
            spend=2.0,
            models=[],
            created_by="test",
            updated_by="test",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0),
        ).model_dump(),
    )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        org_id="org-counter-with-budget",
        response_cost=0.25,
    )

    assert counter_cache.get_local("spend:org:org-counter-with-budget") == pytest.approx(2.25)


@pytest.mark.asyncio
async def test_should_seed_org_counter_from_plain_org_cache(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key=AuthObject.org("org-counter-plain"),
        value=LiteLLM_OrganizationTable(
            organization_id="org-counter-plain",
            organization_alias="shared-org",
            budget_id="org-budget-id",
            spend=2.0,
            models=[],
            created_by="test",
            updated_by="test",
        ).model_dump(),
    )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        org_id="org-counter-plain",
        response_cost=0.25,
    )

    assert counter_cache.get_local("spend:org:org-counter-plain") == pytest.approx(2.25)


def test_tiered_reservation_is_all_or_nothing_with_output_tier_from_input_length():
    """Dashscope tiered pricing is all-or-nothing: the tier is chosen by the total
    input tokens and every token (input and output) is billed at that tier's rate.

    A long-context request with a large output allowance must reserve the output at
    the input-selected tier, not at the cheapest tier picked from the output volume.
    The earlier graduated calculation under-reserved such requests, letting a caller
    slip past a depleted budget."""
    tiered_pricing = [
        {"range": [0, 32000], "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06},
        {"range": [32000, 128000], "input_cost_per_token": 4e-06, "output_cost_per_token": 8e-06},
    ]
    input_tokens = 100000  # falls entirely in the second tier
    output_tokens = 1000

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={"tiered_pricing": tiered_pricing, "max_output_tokens": 200000},
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body={**_request_body(), "max_tokens": output_tokens},
            route="/chat/completions",
            llm_router=None,
            input_token_counts={"gpt-4o-mini": input_tokens},
        )

    expected = (input_tokens * 4e-06) + (output_tokens * 8e-06)
    assert estimated == pytest.approx(expected)

    # What the old graduated math (with the output tier taken from output volume)
    # would have reserved. The all-or-nothing estimate must be strictly larger.
    graduated_under_reserve = (32000 * 1e-06) + (68000 * 4e-06) + (output_tokens * 2e-06)
    assert estimated > graduated_under_reserve


def test_tiered_reservation_uses_higher_reasoning_output_rate():
    """Some tiered models price reasoning output above standard output. The
    reasoning-token share is unknown before the request runs, so reservation must
    charge every output token at the higher of the two rates to avoid under-reserving
    reasoning-heavy requests."""
    tiered_pricing = [
        {
            "range": [0, 32000],
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 1.2e-06,
            "output_cost_per_reasoning_token": 4e-06,
        }
    ]
    input_tokens = 1000
    output_tokens = 500

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={"tiered_pricing": tiered_pricing, "max_output_tokens": 200000},
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body={**_request_body(), "max_tokens": output_tokens},
            route="/chat/completions",
            llm_router=None,
            input_token_counts={"gpt-4o-mini": input_tokens},
        )

    expected = (input_tokens * 1e-06) + (output_tokens * 4e-06)
    assert estimated == pytest.approx(expected)

    # Reserving output at the plain rate would under-reserve reasoning-heavy calls.
    under_reserve = (input_tokens * 1e-06) + (output_tokens * 1.2e-06)
    assert estimated > under_reserve


def test_flat_reservation_uses_higher_reasoning_output_rate():
    """The same reasoning under-reservation gap exists for flat-rate models that
    declare output_cost_per_reasoning_token above output_cost_per_token."""
    input_tokens = 1000
    output_tokens = 500

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 1.2e-06,
                "output_cost_per_reasoning_token": 4e-06,
                "max_output_tokens": 200000,
            },
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body={**_request_body(), "max_tokens": output_tokens},
            route="/chat/completions",
            llm_router=None,
            input_token_counts={"gpt-4o-mini": input_tokens},
        )

    expected = (input_tokens * 1e-06) + (output_tokens * 4e-06)
    assert estimated == pytest.approx(expected)
    under_reserve = (input_tokens * 1e-06) + (output_tokens * 1.2e-06)
    assert estimated > under_reserve


def test_reservation_uses_most_expensive_deployment_in_group():
    """When a model group mixes deployments the config prices differently, reservation must
    estimate against the most expensive one. Reserving the cheaper sibling would let a caller
    repeatedly hit the alias and exceed the budget once routed to the costlier deployment."""
    cheap = [{"range": [0, 32000], "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}]
    expensive = [{"range": [0, 32000], "input_cost_per_token": 5e-06, "output_cost_per_token": 1e-05}]
    input_tokens = 1000
    output_tokens = 10
    router = Router(
        model_list=[
            {
                "model_name": "mixed-rates",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"},
                "model_info": {"id": deployment_id, "tiered_pricing": tiers, "max_output_tokens": 200_000},
            }
            for deployment_id, tiers in (("cheap-deployment", cheap), ("expensive-deployment", expensive))
        ]
    )

    estimated = estimate_request_max_cost(
        request_body=_long_prompt_request(model="mixed-rates", max_tokens=output_tokens),
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"mixed-rates": input_tokens},
    )

    expected_expensive = (input_tokens * 5e-06) + (output_tokens * 1e-05)
    expected_cheap = (input_tokens * 1e-06) + (output_tokens * 2e-06)
    assert expected_expensive > expected_cheap
    assert estimated == pytest.approx(expected_expensive)


def _long_prompt_request(model: str, max_tokens: int) -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": max_tokens}


def test_reservation_prices_prompt_above_272k_at_the_above_threshold_rate():
    """A prompt past a model's *_above_272k_tokens threshold is billed at the above rate for
    every token, so the reservation must price it there too. Reserving at the base rate
    let a 300K-token request through a budget it then overshot by the full above-rate cost."""
    model_info = litellm.get_model_info("gpt-5.5")
    above_input_rate = model_info.get("input_cost_per_token_above_272k_tokens")
    above_output_rate = model_info.get("output_cost_per_token_above_272k_tokens")
    base_input_rate = model_info.get("input_cost_per_token")
    base_output_rate = model_info.get("output_cost_per_token")
    assert above_input_rate is not None and above_output_rate is not None
    assert base_input_rate is not None and base_output_rate is not None
    input_tokens = 300_000
    output_tokens = 1_000
    request_body = _long_prompt_request(model="gpt-5.5", max_tokens=output_tokens)

    estimated = estimate_request_max_cost(
        request_body=request_body,
        route="/chat/completions",
        llm_router=None,
        input_token_counts={"gpt-5.5": input_tokens},
    )
    estimated_input = estimate_request_input_cost(
        request_body=request_body,
        route="/chat/completions",
        llm_router=None,
        input_token_counts={"gpt-5.5": input_tokens},
    )

    assert estimated is not None
    assert estimated == pytest.approx((input_tokens * above_input_rate) + (output_tokens * above_output_rate))
    assert estimated_input == pytest.approx(input_tokens * above_input_rate)
    base_rate_under_reserve = (input_tokens * base_input_rate) + (output_tokens * base_output_rate)
    assert estimated > base_rate_under_reserve


def test_reservation_prices_every_image_of_a_router_served_image_model():
    """A deployment's own cost entry carries the per-image price, so an image-generation
    request through the router reserves `n x per-image cost` instead of nothing."""
    per_image = 0.04
    router = Router(
        model_list=[
            {
                "model_name": "pixels",
                "litellm_params": {
                    "model": "openai/pixel-painter",
                    "api_key": "sk-fake",
                    "input_cost_per_image": per_image,
                },
                "model_info": {"mode": "image_generation"},
            }
        ]
    )

    estimated = estimate_request_max_cost(
        request_body={"model": "pixels", "prompt": "a cat", "n": 3},
        route="/v1/images/generations",
        llm_router=router,
        input_token_counts={"pixels": 5},
    )

    assert estimated == pytest.approx(3 * per_image)


def test_image_generation_reserves_the_picture_not_a_chat_token_fallback():
    """gpt-image-1 prices its output only per image token, so charging the chat fallback of 16K
    output tokens at that rate would reserve $0.65 against a 1024x1024 low-quality picture that
    bills about a cent, and refuse a key holding enough for the picture. The reservation prices
    the prompt and leaves the picture to the per-image path."""
    from litellm.proxy.spend_tracking.budget_reservation import (
        DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK,
    )

    model_info = litellm.get_model_info("gpt-image-1")
    image_token_rate = model_info.get("output_cost_per_image_token")
    input_rate = model_info.get("input_cost_per_token")
    assert image_token_rate is not None and input_rate is not None
    input_tokens = 8

    estimated = estimate_request_max_cost(
        request_body={"model": "gpt-image-1", "prompt": "a red bicycle", "size": "1024x1024", "quality": "low"},
        route="/v1/images/generations",
        llm_router=None,
        input_token_counts={"gpt-image-1": input_tokens},
    )

    assert estimated == pytest.approx(input_tokens * input_rate)
    assert estimated < DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK * image_token_rate


def _openai_router(model_name: str, api_base: str | None = None) -> Router:
    litellm_params: dict[str, object] = {"model": "openai/gpt-5.5", "api_key": "sk-fake"}
    if api_base is not None:
        litellm_params["api_base"] = api_base
    return Router(model_list=[{"model_name": model_name, "litellm_params": litellm_params}])


def test_reservation_prices_a_priority_tier_request_at_the_priority_rate():
    """A request that asks for the priority service tier is billed at the model's priority rates,
    which run 2.5x the standard ones on gpt-5.5. Reserving at the standard rate lets a priority
    request through a budget its own bill then overshoots."""
    model_info = litellm.get_model_info("gpt-5.5")
    priority_input_rate = model_info.get("input_cost_per_token_priority")
    priority_output_rate = model_info.get("output_cost_per_token_priority")
    assert priority_input_rate is not None and priority_output_rate is not None
    input_tokens = 10_000
    output_tokens = 100
    router = _openai_router("gpt-5.5")
    standard_request = _long_prompt_request(model="gpt-5.5", max_tokens=output_tokens)

    estimated = estimate_request_max_cost(
        request_body={**standard_request, "service_tier": "priority"},
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5": input_tokens},
    )

    assert estimated == pytest.approx((input_tokens * priority_input_rate) + (output_tokens * priority_output_rate))
    standard_estimate = estimate_request_max_cost(
        request_body=standard_request,
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5": input_tokens},
    )
    assert standard_estimate is not None and estimated > standard_estimate


def test_reservation_prices_a_flex_request_at_the_flex_rate():
    """The bill prices a request by the service tier it asked for, and gpt-5.5's flex rates are
    half the standard ones. Reserving at the standard rate holds twice what a flex request can
    bill, so a key with room for the request is refused."""
    model_info = litellm.get_model_info("gpt-5.5")
    flex_input_rate = model_info.get("input_cost_per_token_flex")
    flex_output_rate = model_info.get("output_cost_per_token_flex")
    assert flex_input_rate is not None and flex_output_rate is not None
    input_tokens = 10_000
    output_tokens = 100
    router = _openai_router("gpt-5.5")
    standard_request = _long_prompt_request(model="gpt-5.5", max_tokens=output_tokens)

    estimated = estimate_request_max_cost(
        request_body={**standard_request, "service_tier": "flex"},
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5": input_tokens},
    )

    assert estimated == pytest.approx((input_tokens * flex_input_rate) + (output_tokens * flex_output_rate))
    standard_estimate = estimate_request_max_cost(
        request_body=standard_request,
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5": input_tokens},
    )
    assert standard_estimate is not None and estimated < standard_estimate


def test_reservation_prices_a_regional_deployment_at_the_uplifted_rate():
    """OpenAI charges a flat uplift on every token a regional host processes, so a deployment
    pointed at eu.api.openai.com bills above the global rate. Reserving at the global rate
    under-reserves every request that deployment serves."""
    uplift = litellm.get_model_info("gpt-5.5").get("regional_processing_uplift_multiplier_eu")
    assert uplift is not None and uplift > 1
    input_tokens = 10_000
    output_tokens = 100
    request_body = _long_prompt_request(model="gpt-5.5", max_tokens=output_tokens)

    estimates = [
        estimate_request_max_cost(
            request_body=request_body,
            route="/chat/completions",
            llm_router=_openai_router("gpt-5.5", api_base=api_base),
            input_token_counts={"gpt-5.5": input_tokens},
        )
        for api_base in (None, "https://eu.api.openai.com/v1")
    ]

    global_estimate, regional_estimate = estimates
    assert global_estimate is not None and regional_estimate is not None
    assert regional_estimate == pytest.approx(global_estimate * uplift)


def test_reservation_prices_a_deployment_pinned_service_tier_at_that_tier_rate():
    """The router hands every request that does not name its own service_tier the one pinned in
    the deployment's litellm_params, and the bill follows it: gpt-5.5's priority rates run 2.5x
    the standard ones. Reserving at the standard rate lets those requests through a budget their
    own bill then overshoots. A request that names a tier still wins, so flex on the same
    deployment reserves at the flex rate."""
    model_info = litellm.get_model_info("gpt-5.5")
    priority_input_rate = model_info.get("input_cost_per_token_priority")
    priority_output_rate = model_info.get("output_cost_per_token_priority")
    flex_input_rate = model_info.get("input_cost_per_token_flex")
    flex_output_rate = model_info.get("output_cost_per_token_flex")
    assert priority_input_rate is not None and priority_output_rate is not None
    assert flex_input_rate is not None and flex_output_rate is not None
    input_tokens = 10_000
    output_tokens = 100
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.5-priority",
                "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-fake", "service_tier": "priority"},
            }
        ]
    )
    request_body = _long_prompt_request(model="gpt-5.5-priority", max_tokens=output_tokens)

    estimated = estimate_request_max_cost(
        request_body=request_body,
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5-priority": input_tokens},
    )

    assert estimated == pytest.approx((input_tokens * priority_input_rate) + (output_tokens * priority_output_rate))
    flex_estimate = estimate_request_max_cost(
        request_body={**request_body, "service_tier": "flex"},
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5-priority": input_tokens},
    )
    assert flex_estimate == pytest.approx((input_tokens * flex_input_rate) + (output_tokens * flex_output_rate))


def test_reservation_prices_a_regional_vertex_deployment_at_the_uplifted_rate():
    """Google bills every non-global Vertex endpoint at a flat premium over the global one, 1.1x
    on gemini-3.5-flash, so a deployment pinned to us-east1 costs 10% more than the same model on
    the global endpoint. Reserving at the global rate under-reserves every request it serves."""
    uplift = litellm.get_model_info("vertex_ai/gemini-3.5-flash").get("regional_endpoint_uplift_multiplier")
    assert uplift is not None and uplift > 1
    input_tokens = 10_000
    request_body = _long_prompt_request(model="vertex-flash", max_tokens=100)

    estimates = [
        estimate_request_max_cost(
            request_body=request_body,
            route="/chat/completions",
            llm_router=Router(
                model_list=[
                    {
                        "model_name": "vertex-flash",
                        "litellm_params": {
                            "model": "vertex_ai/gemini-3.5-flash",
                            "vertex_project": "test-project",
                            "vertex_location": vertex_location,
                        },
                    }
                ]
            ),
            input_token_counts={"vertex-flash": input_tokens},
        )
        for vertex_location in ("global", "us-east1")
    ]

    global_estimate, regional_estimate = estimates
    assert global_estimate is not None and regional_estimate is not None
    assert regional_estimate == pytest.approx(global_estimate * uplift)


def test_reservation_prices_a_prompt_at_the_cache_write_rate():
    """OpenAI writes a long prompt into its own cache without the request asking and bills every
    written token at the cache-creation rate: a 305,124-token gpt-5.6-luna request came back with
    `cache_creation_tokens: 305124` and `x-litellm-response-cost-cache-creation: 0.152562` out of a
    0.1525938 bill. Reserving at the input rate leaves that whole gap unreserved."""
    model_info = litellm.get_model_info("gpt-5.6-luna")
    input_rate = model_info.get("input_cost_per_token_above_272k_tokens")
    cache_write_rate = model_info.get("cache_creation_input_token_cost_above_272k_tokens")
    output_rate = model_info.get("output_cost_per_token_above_272k_tokens")
    assert input_rate is not None and output_rate is not None
    assert cache_write_rate is not None and cache_write_rate > input_rate
    input_tokens = 305_124
    output_tokens = 100
    router = Router(
        model_list=[
            {
                "model_name": "luna",
                "litellm_params": {"model": "openai/gpt-5.6-luna", "api_key": "sk-fake"},
            }
        ]
    )

    estimated = estimate_request_max_cost(
        request_body=_long_prompt_request(model="luna", max_tokens=output_tokens),
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"luna": input_tokens},
    )

    assert estimated == pytest.approx((input_tokens * cache_write_rate) + (output_tokens * output_rate))


def test_cancelled_request_keeps_the_cache_write_cost_it_already_incurred():
    """A cancelled in-flight request is reconciled down to its input cost, and the provider has
    already written the prompt into its cache and billed it at the cache-creation rate. Holding
    only the plain input rate releases budget that is already spent, so the next request is
    admitted against headroom the key does not have."""
    model_info = litellm.get_model_info("gpt-5.6-luna")
    input_rate = model_info.get("input_cost_per_token_above_272k_tokens")
    cache_write_rate = model_info.get("cache_creation_input_token_cost_above_272k_tokens")
    assert input_rate is not None
    assert cache_write_rate is not None and cache_write_rate > input_rate
    input_tokens = 305_124

    estimated_input = estimate_request_input_cost(
        request_body=_long_prompt_request(model="gpt-5.6-luna", max_tokens=100),
        route="/chat/completions",
        llm_router=None,
        input_token_counts={"gpt-5.6-luna": input_tokens},
    )

    assert estimated_input == pytest.approx(input_tokens * cache_write_rate)


def test_a_prompt_too_short_to_cache_reserves_the_plain_input_rate():
    """OpenAI and Azure write a prompt into their own cache from 1024 tokens up and nothing below
    that, so a shorter prompt bills the plain input rate. Reserving the 25% dearer write rate for
    it holds budget the request cannot spend, and a cancelled request is reconciled to that
    inflated number as real spend rather than as a hold that gets released."""
    model_info = litellm.get_model_info("gpt-5.6-luna")
    input_rate = model_info.get("input_cost_per_token")
    cache_write_rate = model_info.get("cache_creation_input_token_cost")
    assert input_rate is not None
    assert cache_write_rate is not None and cache_write_rate > input_rate

    below_threshold, at_threshold = [
        estimate_request_input_cost(
            request_body=_long_prompt_request(model="gpt-5.6-luna", max_tokens=100),
            route="/chat/completions",
            llm_router=None,
            input_token_counts={"gpt-5.6-luna": input_tokens},
        )
        for input_tokens in (1_023, 1_024)
    ]

    assert below_threshold == pytest.approx(1_023 * input_rate)
    assert at_threshold == pytest.approx(1_024 * cache_write_rate)


def test_reservation_holds_the_cache_write_rate_only_when_the_request_asks_for_it():
    """Anthropic writes a prompt into its cache only where cache_control marks a block, so a
    request that marks nothing is billed the plain input rate. Holding the 25% dearer write rate
    for every Anthropic request reserves above anything the request can bill, and one that does
    mark a block, or sends a top-level cache_control that turns on automatic caching, still has
    to hold it."""
    model_info = litellm.get_model_info("claude-haiku-4-5")
    input_rate = model_info.get("input_cost_per_token")
    cache_write_rate = model_info.get("cache_creation_input_token_cost")
    assert input_rate is not None
    assert cache_write_rate is not None and cache_write_rate > input_rate
    input_tokens = 72_801
    plain_request = _long_prompt_request(model="claude-haiku-4-5", max_tokens=100)
    cached_request = {
        **plain_request,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    }

    auto_cached_request = {**plain_request, "cache_control": {"type": "ephemeral"}}

    cached_responses_request = {
        "model": "claude-haiku-4-5",
        "max_output_tokens": 100,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    }

    estimates = [
        estimate_request_input_cost(
            request_body=body,
            route=route,
            llm_router=None,
            input_token_counts={"claude-haiku-4-5": input_tokens},
        )
        for body, route in (
            (plain_request, "/chat/completions"),
            (cached_request, "/chat/completions"),
            (cached_responses_request, "/v1/responses"),
            (auto_cached_request, "/v1/messages"),
        )
    ]

    plain, cached, cached_responses, auto_cached = estimates
    assert plain == pytest.approx(input_tokens * input_rate)
    assert cached == pytest.approx(input_tokens * cache_write_rate)
    assert cached_responses == pytest.approx(input_tokens * cache_write_rate)
    assert auto_cached == pytest.approx(input_tokens * cache_write_rate)


def test_reservation_prices_above_272k_rate_through_router_deployment():
    """On the proxy the model name resolves through the router, whose group summary only
    carries flat rates. The reservation must read each deployment's full pricing so the
    above-threshold keys from the cost map reach the estimate."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-fake"},
            }
        ]
    )
    model_info = litellm.get_model_info("openai/gpt-5.5")
    above_input_rate = model_info.get("input_cost_per_token_above_272k_tokens")
    above_output_rate = model_info.get("output_cost_per_token_above_272k_tokens")
    base_input_rate = model_info.get("input_cost_per_token")
    assert above_input_rate is not None and above_output_rate is not None
    assert base_input_rate is not None and above_input_rate > base_input_rate
    input_tokens = 300_000
    output_tokens = 1_000

    estimated = estimate_request_max_cost(
        request_body=_long_prompt_request(model="gpt-5.5", max_tokens=output_tokens),
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5": input_tokens},
    )

    assert estimated == pytest.approx((input_tokens * above_input_rate) + (output_tokens * above_output_rate))


def test_reservation_honors_inclusive_threshold_providers():
    """xAI bills the higher tier once the prompt reaches the threshold while OpenAI bills it
    only past the threshold, so a prompt of exactly the threshold length reserves at the
    above rate for xAI and at the base rate for OpenAI."""
    output_tokens = 10

    def estimate(model: str, input_tokens: int) -> float:
        estimated = estimate_request_max_cost(
            request_body=_long_prompt_request(model=model, max_tokens=output_tokens),
            route="/chat/completions",
            llm_router=None,
            input_token_counts={model: input_tokens},
        )
        assert estimated is not None
        return estimated

    xai_info = litellm.get_model_info("xai/grok-4.6")
    xai_above_input_rate = xai_info.get("input_cost_per_token_above_200k_tokens")
    xai_above_output_rate = xai_info.get("output_cost_per_token_above_200k_tokens")
    assert xai_above_input_rate is not None and xai_above_output_rate is not None
    openai_info = litellm.get_model_info("gpt-5.5")
    openai_base_input_rate = openai_info.get("input_cost_per_token")
    openai_base_output_rate = openai_info.get("output_cost_per_token")
    assert openai_base_input_rate is not None and openai_base_output_rate is not None

    assert estimate("xai/grok-4.6", 200_000) == pytest.approx(
        (200_000 * xai_above_input_rate) + (output_tokens * xai_above_output_rate)
    )
    assert estimate("gpt-5.5", 272_000) == pytest.approx(
        (272_000 * openai_base_input_rate) + (output_tokens * openai_base_output_rate)
    )


def test_reservation_ignores_an_off_peak_discount_whose_window_can_close():
    """A deployment's off-peak window can close between the reservation and the response, and the
    finished request is then billed at the standard rate. Reserving the discounted rate would let
    a request through a budget the standard rate goes on to overshoot."""
    now = datetime.now(timezone.utc)
    open_window = f"{(now - timedelta(hours=1)).strftime('%H:%M')}-{(now + timedelta(hours=1)).strftime('%H:%M')}"
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.5-off-peak",
                "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-fake"},
                "model_info": {"off_peak_pricing": {"hours_utc": open_window, "input_cost_per_token": 1e-08}},
            }
        ]
    )
    above_input_rate = litellm.get_model_info("openai/gpt-5.5").get("input_cost_per_token_above_272k_tokens")
    assert above_input_rate is not None
    input_tokens = 300_000

    estimated_input = estimate_request_input_cost(
        request_body=_long_prompt_request(model="gpt-5.5-off-peak", max_tokens=16),
        route="/chat/completions",
        llm_router=router,
        input_token_counts={"gpt-5.5-off-peak": input_tokens},
    )

    assert estimated_input == pytest.approx(input_tokens * above_input_rate)


@pytest.mark.asyncio
async def test_should_skip_reservation_for_per_pixel_image_model(spend_counter_state, native_budgets_on, monkeypatch):
    """DALL-E 2-style per-pixel pricing depends on the requested ``size``,
    which we don't decode here. Fall through to read-time enforcement
    rather than guess."""
    import litellm.proxy.proxy_server as ps

    counter_cache, key_cache = spend_counter_state
    monkeypatch.setattr(ps, "spend_counter_store", SpendCounterStore(connect(None).spend_counters()))
    install_native_owner(ps.spend_counter_store)
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-image-per-pixel",
        spend=0.0,
        max_budget=1.0,
    )
    await key_cache.async_set_cache(key=AuthObject.virtual_key("key-image-per-pixel"), value=valid_token)

    request_body = {"model": "dall-e-2", "prompt": "a cat", "size": "256x256"}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "mode": "image_generation",
            "input_cost_per_pixel": 2.4414e-07,
            "output_cost_per_pixel": 0.0,
        },
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/v1/images/generations",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is None


@pytest.mark.parametrize("microsecond", (0, 123456, 999999))
def test_should_start_window_without_reset_at_at_duration_boundary(microsecond: int) -> None:
    now: Final = datetime(2026, 11, 15, 5, 48, 2, microsecond, tzinfo=timezone.utc)
    with patch("litellm.proxy.spend_tracking.budget_reservation.datetime") as clock:
        clock.now.return_value = now
        window_start: Final = get_budget_window_start({"budget_duration": "1h"})

    assert window_start == now.replace(microsecond=microsecond // 1000 * 1000) - timedelta(hours=1)


@pytest.mark.asyncio
async def test_release_budget_reservation_on_cancel_swallows_release_errors():
    # If the release itself fails (e.g. Redis unavailable) it must not escape
    # the helper: doing so would replace the in-flight CancelledError /
    # GeneratorExit at the call site and disrupt the disconnect teardown.
    reservation = {
        "reserved_cost": 3.0,
        "entries": [{"counter_key": "spend:key:key-cancel-error"}],
        "finalized": False,
        "input_cost": 0.5,
    }
    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        # must return without raising
        await release_budget_reservation_on_cancel(reservation)


class _ExplodingPrismaClient:
    """Model access group budget rows come from the Rust store; a Prisma read is a bug."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected database access: {name}")


@pytest.fixture
def rust_auth_stores(monkeypatch: pytest.MonkeyPatch) -> RustAuthStores:
    return install_rust_auth_stores(monkeypatch)


def _seed_model_access_group_budgets(stores: RustAuthStores, **max_budget_by_group: float | None) -> None:
    """``LiteLLM_ModelAccessGroupBudgetTable`` rows as the Rust model access group store serves them."""
    stores.model_access_group.groups.update(
        {group: GroupBudget(spend=7.0, max_budget=max_budget) for group, max_budget in max_budget_by_group.items()}
    )


async def _model_access_group_counters(stores: RustAuthStores, matched, **max_budget_by_group):
    _seed_model_access_group_budgets(stores, **max_budget_by_group)
    return await _get_model_access_group_budget_counters(
        valid_token=UserAPIKeyAuth(api_key="hashed", matched_model_access_groups=matched),
        prisma_client=_ExplodingPrismaClient(),
        user_api_key_cache=UserApiKeyCache(),
    )


@pytest.mark.asyncio
async def test_model_access_group_with_a_budget_reserves_against_the_reset_jobs_counter_key(
    rust_auth_stores: RustAuthStores,
):
    counters = await _model_access_group_counters(rust_auth_stores, ["premium"], premium=25.0)

    assert len(counters) == 1
    counter = counters[0]
    assert counter.counter_key == _model_access_group_counter_key(SimpleNamespace(access_group_name="premium"))
    assert counter.source_cache_key == AuthObject.model_access_group("premium")
    assert counter.max_budget == 25.0
    assert counter.fallback_spend == 7.0
    assert counter.entity_type == "Model access group"
    assert counter.entity_id == "premium"


@pytest.mark.asyncio
async def test_model_access_group_without_a_budget_reserves_nothing(rust_auth_stores: RustAuthStores):
    assert await _model_access_group_counters(rust_auth_stores, ["premium"], premium=None) == []


@pytest.mark.asyncio
async def test_model_access_group_with_a_zero_budget_reserves_nothing(rust_auth_stores: RustAuthStores):
    """Zero is how a budget is cleared, not a ceiling that blocks every request."""
    assert await _model_access_group_counters(rust_auth_stores, ["premium"], premium=0.0) == []


@pytest.mark.asyncio
async def test_model_access_group_counters_come_from_the_auth_object(rust_auth_stores: RustAuthStores):
    """Auth already resolved which granted groups serve the model; re-deriving it here would drift."""
    assert await _model_access_group_counters(rust_auth_stores, None, premium=25.0) == []


@pytest.mark.asyncio
async def test_repeated_model_access_group_reserves_once(rust_auth_stores: RustAuthStores):
    counters = await _model_access_group_counters(rust_auth_stores, ["premium", "premium"], premium=25.0)

    assert [counter.entity_id for counter in counters] == ["premium"]


async def _cache_model_access_group_budget(key_cache, group, spend, max_budget=None):
    await key_cache.async_set_cache(
        key=AuthObject.model_access_group(group),
        value=ModelAccessGroupBudget(access_group_name=group, spend=spend, max_budget=max_budget),
        model_type=ModelAccessGroupBudget,
    )


@pytest.mark.asyncio
async def test_model_access_group_counter_accumulates_across_calls_without_a_reservation(spend_counter_state):
    """With reservations disabled nothing writes the counter up front, so the cost callback must.

    Otherwise the read-time budget check enforces against the DB row's spend, which the cache
    holds for the full TTL, and a caller runs past the ceiling for that whole window.
    """
    counter_cache, key_cache = spend_counter_state
    await _cache_model_access_group_budget(key_cache, "premium", spend=1.0, max_budget=25.0)

    from litellm.proxy.proxy_server import increment_spend_counters

    counter_key = model_access_group_counter("premium")

    await increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.25, model_access_groups=["premium"]
    )
    assert counter_cache.get_local(counter_key) == pytest.approx(1.25)

    await increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.75, model_access_groups=["premium", "premium", ""]
    )
    assert counter_cache.get_local(counter_key) == pytest.approx(2.0)
    assert counter_cache.get_local(model_access_group_counter("")) is None
