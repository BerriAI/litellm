from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest

from litellm.responses.mcp.round_billing import billed_for_every_round, merged_round_usage, summed_cost_breakdown
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import CostBreakdown


def _round(response_id: str, response_cost: object, served_from_cache: bool = False) -> ResponsesAPIResponse:
    response: Final = ResponsesAPIResponse.model_validate(
        MappingProxyType(
            {
                "id": response_id,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-5.6",
                "output": (),
                "usage": MappingProxyType({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
            }
        )
    )
    response.hidden_params["response_cost"] = response_cost
    if served_from_cache:
        response.hidden_params["cache_hit"] = True
    return response


def test_summed_cost_breakdown_adds_each_charge_and_keeps_the_final_round_rates():
    first: Final[CostBreakdown] = {
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
        "additional_costs": {"web_search": 0.01},
        "discount_percent": 0.1,
        "margin_percent": 0.2,
        "service_tier": "default",
    }
    final: Final[CostBreakdown] = {
        "input_cost": 0.004,
        "output_cost": 0.005,
        "total_cost": 0.009,
        "additional_costs": {"web_search": 0.01, "file_search": 0.0025},
        "discount_percent": 0.1,
        "margin_percent": 0.2,
        "service_tier": "default",
    }

    summed: Final = summed_cost_breakdown(
        first=_round("resp_first", 0.003), first_breakdown=first, final=_round("resp_final", 0.009), final_breakdown=final
    )

    assert summed is not None
    assert summed["input_cost"] == pytest.approx(0.005)
    assert summed["output_cost"] == pytest.approx(0.007)
    assert summed["total_cost"] == pytest.approx(0.012)
    assert summed["additional_costs"] == {"web_search": pytest.approx(0.02), "file_search": pytest.approx(0.0025)}
    assert (summed["discount_percent"], summed["margin_percent"]) == (0.1, 0.2)
    assert summed["service_tier"] == "default"


def test_summed_cost_breakdown_is_none_when_the_final_round_was_not_repriced():
    breakdown: Final[CostBreakdown] = {"input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003}
    first: Final = _round("resp_first", 0.003)
    final: Final = _round("resp_final", 0.003)

    assert summed_cost_breakdown(first=first, first_breakdown=breakdown, final=final, final_breakdown=breakdown) is None
    assert summed_cost_breakdown(first=first, first_breakdown=None, final=final, final_breakdown=breakdown) is None
    assert (
        summed_cost_breakdown(
            first=_round("resp_first", 0.003, served_from_cache=True),
            first_breakdown=breakdown,
            final=final,
            final_breakdown=breakdown,
        )
        is None
    )


def test_a_final_round_served_from_the_response_cache_keeps_the_paid_first_round_breakdown():
    breakdown: Final[CostBreakdown] = {"input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003}

    assert (
        summed_cost_breakdown(
            first=_round("resp_first", 0.003),
            first_breakdown=breakdown,
            final=_round("resp_final", 0.0, served_from_cache=True),
            final_breakdown=breakdown,
        )
        == breakdown
    )


def test_merged_round_usage_sums_token_details_and_keeps_provider_flags():
    first: Final = ResponseAPIUsage.model_validate(
        MappingProxyType(
            {
                "input_tokens": 71,
                "input_tokens_details": MappingProxyType({"cached_tokens": 32}),
                "output_tokens": 19,
                "total_tokens": 90,
                "is_byok": True,
            }
        )
    )
    final: Final = ResponseAPIUsage.model_validate(
        MappingProxyType(
            {
                "input_tokens": 149,
                "input_tokens_details": MappingProxyType({"cached_tokens": 64}),
                "output_tokens": 5,
                "total_tokens": 154,
                "is_byok": True,
            }
        )
    )

    merged: Final = merged_round_usage(first, final)

    assert merged is not None
    assert (merged.input_tokens, merged.output_tokens, merged.total_tokens) == (220, 24, 244)
    assert merged.input_tokens_details is not None
    assert merged.input_tokens_details.cached_tokens == 96
    assert merged.model_dump()["is_byok"] is True


def test_billed_for_every_round_leaves_the_cost_to_the_caller_when_a_round_has_no_price():
    billed: Final = billed_for_every_round(
        first=_round("resp_first", None), first_breakdown=None, final=_round("resp_final", 0.004), final_breakdown=None
    )

    assert billed.id == "resp_final"
    assert billed.hidden_params["response_cost"] is None


def test_billed_for_every_round_does_not_rebill_the_final_round_object():
    final: Final = _round("resp_final", 0.004)

    billed: Final = billed_for_every_round(
        first=_round("resp_first", 0.001), first_breakdown=None, final=final, final_breakdown=None
    )

    assert billed.hidden_params["response_cost"] == pytest.approx(0.005)
    assert final.hidden_params["response_cost"] == 0.004


@pytest.mark.parametrize(
    ("final_cached", "request_cost", "request_margin"),
    [
        (False, 0.012, 0.002),
        (True, 0.0032, 0.0012),
    ],
)
def test_the_fixed_margin_is_charged_once_per_request_across_charged_rounds(
    final_cached: bool, request_cost: float, request_margin: float
):
    first_breakdown: Final[CostBreakdown] = {
        "input_cost": 0.0015,
        "output_cost": 0.0005,
        "total_cost": 0.0032,
        "margin_percent": 0.1,
        "margin_fixed_amount": 0.001,
        "margin_total_amount": 0.0012,
    }
    final_breakdown: Final[CostBreakdown] = {
        "input_cost": 0.006,
        "output_cost": 0.002,
        "total_cost": 0.0098,
        "margin_percent": 0.1,
        "margin_fixed_amount": 0.001,
        "margin_total_amount": 0.0018,
    }
    first: Final = _round("resp_first", 0.0032)
    final: Final = _round("resp_final", 0.0 if final_cached else 0.0098, served_from_cache=final_cached)

    billed: Final = billed_for_every_round(
        first=first, first_breakdown=first_breakdown, final=final, final_breakdown=final_breakdown
    )
    breakdown: Final = summed_cost_breakdown(
        first=first, first_breakdown=first_breakdown, final=final, final_breakdown=final_breakdown
    )

    assert billed.hidden_params["response_cost"] == pytest.approx(request_cost)
    assert breakdown is not None
    assert breakdown["total_cost"] == pytest.approx(request_cost)
    assert breakdown["margin_fixed_amount"] == pytest.approx(0.001)
    assert breakdown["margin_total_amount"] == pytest.approx(request_margin)


@pytest.mark.parametrize(
    ("first_cached", "final_cached", "kept_breakdown"),
    [
        (True, False, "final"),
        (False, True, "first"),
        (True, True, None),
    ],
)
def test_a_round_served_from_the_response_cache_drops_out_of_the_breakdown(
    first_cached: bool, final_cached: bool, kept_breakdown: str | None
):
    breakdowns: Final[Mapping[str, CostBreakdown]] = MappingProxyType(
        {
            "first": {"input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003},
            "final": {"input_cost": 0.004, "output_cost": 0.005, "total_cost": 0.009},
        }
    )

    breakdown: Final = summed_cost_breakdown(
        first=_round("resp_first", 0.003, served_from_cache=first_cached),
        first_breakdown=breakdowns["first"],
        final=_round("resp_final", 0.009, served_from_cache=final_cached),
        final_breakdown=breakdowns["final"],
    )

    assert breakdown == (None if kept_breakdown is None else breakdowns[kept_breakdown])
