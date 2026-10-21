from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.hooks.search_free_tier import (
    apply_search_free_tier,
    freed_units,
    search_free_tier_key,
    search_free_tier_ttl,
)
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

NOW: Final = datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc)
RATE: Final = 0.01


class _Counter:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.ttls: dict[str, int | None] = {}

    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> float:
        self.totals[key] = self.totals.get(key, 0) + value
        self.ttls[key] = ttl
        return self.totals[key]


class _BrokenCounter:
    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> float:
        raise ConnectionError("redis is down")


def _model_info(billing_unit: str, count: int, period: str = "month", provider: str = "vertex_ai") -> dict:
    return {
        "litellm_provider": provider,
        "mode": "chat",
        "web_search_billing_unit": billing_unit,
        "search_context_cost_per_query": {"search_context_size_medium": RATE},
        "search_free_requests": {"count": count, "period": period, "shared_group": "test-group"},
    }


def _response(queries: int | None) -> ModelResponse:
    details = PromptTokensDetailsWrapper(web_search_requests=queries) if queries is not None else None
    return ModelResponse(
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=details)
    )


def _call(model_info: dict, response_cost: float, cache_hit: bool | None = None) -> tuple[dict, dict]:
    sl_object = {
        "response_cost": response_cost,
        "cache_hit": cache_hit,
        "custom_llm_provider": model_info["litellm_provider"],
        "cost_breakdown": {"input_cost": 0.1, "output_cost": 0.2, "tool_usage_cost": 0.5, "total_cost": response_cost},
        "model_map_information": {"model_map_key": "gemini-test", "model_map_value": model_info},
    }
    return {"response_cost": response_cost, "standard_logging_object": sl_object}, sl_object


@pytest.mark.parametrize(
    ("allowance", "units", "total_after", "expected"),
    [
        (5000, 5, 5, 5),
        (5000, 5, 5000, 5),
        (5000, 10, 5005, 5),
        (5000, 5, 5010, 0),
        (2, 5, 5, 2),
        (0, 3, 3, 0),
    ],
)
def test_freed_units_covers_only_what_landed_under_the_allowance(allowance, units, total_after, expected):
    assert freed_units(allowance, units, total_after) == expected


def test_counter_key_is_per_provider_group_and_utc_period():
    assert search_free_tier_key("gemini", {"count": 1, "period": "month", "shared_group": "g3"}, NOW) == (
        "litellm:search_free_tier:gemini:g3:2026-10"
    )
    assert search_free_tier_key("vertex_ai", {"count": 1, "period": "day", "shared_group": "g2"}, NOW) == (
        "litellm:search_free_tier:vertex_ai:g2:2026-10-20"
    )


@pytest.mark.parametrize(
    ("period", "boundary"),
    [("month", datetime(2026, 11, 1, tzinfo=timezone.utc)), ("day", datetime(2026, 10, 21, tzinfo=timezone.utc))],
)
def test_counter_ttl_outlives_the_period_boundary_by_under_two_hours(period, boundary):
    expiry = NOW + timedelta(seconds=search_free_tier_ttl({"count": 1, "period": period, "shared_group": "g"}, NOW))
    assert boundary < expiry < boundary + timedelta(hours=2)


@pytest.mark.asyncio
async def test_per_query_allowance_is_shared_across_calls_and_split_on_the_crossing_call():
    counter = _Counter()
    model_info = _model_info("per_query", count=7)
    reduced = []
    for _ in range(3):
        kwargs, sl_object = _call(model_info, response_cost=1.0)
        cost = await apply_search_free_tier(kwargs, sl_object, 1.0, _response(5), counter, NOW)
        reduced.append((cost, kwargs["response_cost"], sl_object["response_cost"], sl_object["cost_breakdown"]))
    (c1, k1, s1, b1), (c2, k2, s2, b2), (c3, k3, s3, b3) = reduced
    assert c1 == k1 == s1 == pytest.approx(1.0 - 5 * RATE)
    assert b1["original_cost"] == 1.0 and b1["discount_amount"] == pytest.approx(5 * RATE)
    assert b1["total_cost"] == pytest.approx(c1) and b1["tool_usage_cost"] == 0.5
    assert c2 == k2 == s2 == pytest.approx(1.0 - 2 * RATE)
    assert b2["discount_amount"] == pytest.approx(2 * RATE)
    assert c3 == k3 == s3 == 1.0 and "discount_amount" not in b3
    assert counter.totals == {"litellm:search_free_tier:vertex_ai:test-group:2026-10": 15}
    assert counter.ttls["litellm:search_free_tier:vertex_ai:test-group:2026-10"] > 0


@pytest.mark.asyncio
async def test_gemini_api_and_vertex_ai_grounding_draw_on_separate_allowances():
    counter = _Counter()
    gemini_api = _model_info("per_query", count=3, provider="gemini")
    vertex_ai = _model_info("per_query", count=3, provider="vertex_ai")
    kwargs, sl_object = _call(gemini_api, response_cost=1.0)
    await apply_search_free_tier(kwargs, sl_object, 1.0, _response(2), counter, NOW)
    kwargs, sl_object = _call(vertex_ai, response_cost=1.0)
    vertex_cost = await apply_search_free_tier(kwargs, sl_object, 1.0, _response(4), counter, NOW)
    kwargs, sl_object = _call(gemini_api, response_cost=1.0)
    gemini_cost = await apply_search_free_tier(kwargs, sl_object, 1.0, _response(1), counter, NOW)
    assert vertex_cost == pytest.approx(1.0 - 3 * RATE), "vertex ai had its whole allowance left"
    assert gemini_cost == pytest.approx(1.0 - RATE), "gemini api had one free query left"


@pytest.mark.asyncio
async def test_per_prompt_allowance_counts_a_grounded_prompt_once_whatever_the_query_count():
    counter = _Counter()
    model_info = _model_info("per_prompt", count=1, period="day")
    kwargs, sl_object = _call(model_info, response_cost=1.0)
    first = await apply_search_free_tier(kwargs, sl_object, 1.0, _response(5), counter, NOW)
    kwargs, sl_object = _call(model_info, response_cost=1.0)
    second = await apply_search_free_tier(kwargs, sl_object, 1.0, _response(5), counter, NOW)
    assert first == pytest.approx(1.0 - RATE)
    assert second == 1.0
    assert counter.totals == {"litellm:search_free_tier:vertex_ai:test-group:2026-10-20": 2}


@pytest.mark.asyncio
async def test_sidecar_shaped_response_mapping_is_counted_like_a_model_response():
    counter = _Counter()
    kwargs, sl_object = _call(_model_info("per_query", count=100), response_cost=1.0)
    response = {
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "prompt_tokens_details": {"web_search_requests": 3},
        }
    }
    assert await apply_search_free_tier(kwargs, sl_object, 1.0, response, counter, NOW) == pytest.approx(1.0 - 3 * RATE)


@pytest.mark.asyncio
@pytest.mark.parametrize("counter", [None, _BrokenCounter()])
async def test_full_price_when_no_counter_is_available(counter):
    kwargs, sl_object = _call(_model_info("per_query", count=100), response_cost=1.0)
    assert await apply_search_free_tier(kwargs, sl_object, 1.0, _response(5), counter, NOW) == 1.0
    assert kwargs["response_cost"] == 1.0 and "discount_amount" not in sl_object["cost_breakdown"]


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [_response(0), _response(None), ModelResponse()])
async def test_requests_without_grounding_never_touch_the_counter(response):
    counter = _Counter()
    kwargs, sl_object = _call(_model_info("per_query", count=100), response_cost=1.0)
    assert await apply_search_free_tier(kwargs, sl_object, 1.0, response, counter, NOW) == 1.0
    assert counter.totals == {}


@pytest.mark.asyncio
async def test_models_without_an_allowance_never_touch_the_counter():
    counter = _Counter()
    model_info = {k: v for k, v in _model_info("per_query", count=100).items() if k != "search_free_requests"}
    kwargs, sl_object = _call(model_info, response_cost=1.0)
    assert await apply_search_free_tier(kwargs, sl_object, 1.0, _response(5), counter, NOW) == 1.0
    assert counter.totals == {}


@pytest.mark.asyncio
async def test_a_second_callback_run_on_the_same_request_counts_it_once():
    counter = _Counter()
    kwargs, sl_object = _call(_model_info("per_query", count=100), response_cost=1.0)
    first = await apply_search_free_tier(kwargs, sl_object, 1.0, _response(5), counter, NOW)
    second = await apply_search_free_tier(kwargs, sl_object, first, _response(5), counter, NOW)
    assert first == second == pytest.approx(1.0 - 5 * RATE)
    assert counter.totals == {"litellm:search_free_tier:vertex_ai:test-group:2026-10": 5}


@pytest.mark.asyncio
async def test_a_cache_hit_with_grounded_usage_is_not_counted_and_stays_at_zero():
    counter = _Counter()
    kwargs, sl_object = _call(_model_info("per_query", count=100), response_cost=0.0, cache_hit=True)
    assert await apply_search_free_tier(kwargs, sl_object, 0.0, _response(2), counter, NOW) == 0.0
    assert counter.totals == {}
    assert "discount_amount" not in sl_object["cost_breakdown"]


@pytest.mark.asyncio
async def test_freed_cost_never_exceeds_the_billed_cost():
    counter = _Counter()
    billed = 3 * RATE
    kwargs, sl_object = _call(_model_info("per_query", count=100), response_cost=billed)
    assert await apply_search_free_tier(kwargs, sl_object, billed, _response(5), counter, NOW) == 0.0
    assert sl_object["cost_breakdown"]["discount_amount"] == pytest.approx(billed)
    assert counter.totals == {"litellm:search_free_tier:vertex_ai:test-group:2026-10": 5}
