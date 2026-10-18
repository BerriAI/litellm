from asyncio import CancelledError
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.hidden_params import served_from_cache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import CostBreakdown

_PER_REQUEST_KEYS: Final = frozenset({"discount_percent", "margin_percent", "margin_fixed_amount"})
_TOTALS_WITH_FIXED_MARGIN: Final = ("total_cost", "margin_total_amount")
_COST_BREAKDOWN: Final = TypeAdapter(CostBreakdown)
_STR_KEYED: Final = TypeAdapter(Mapping[str, object])


def _summed(first: object, final: object) -> object:
    if first is None:
        return final
    if final is None:
        return first
    if isinstance(first, bool) or isinstance(final, bool):
        return final
    if isinstance(first, (int, float)) and isinstance(final, (int, float)):
        return first + final
    if isinstance(first, Mapping) and isinstance(final, Mapping):
        return _summed_mapping(_STR_KEYED.validate_python(first), _STR_KEYED.validate_python(final))
    return final


def _summed_entry(key: str, first: Mapping[str, object], final: Mapping[str, object]) -> object:
    if key in _PER_REQUEST_KEYS:
        return final[key] if key in final else first[key]
    return _summed(first.get(key), final.get(key))


def _summed_mapping(first: Mapping[str, object], final: Mapping[str, object]) -> Mapping[str, object]:
    keys: Final = (*first, *(key for key in final if key not in first))
    return MappingProxyType({key: _summed_entry(key, first, final) for key in keys})


def merged_round_usage(first: ResponseAPIUsage | None, final: ResponseAPIUsage | None) -> ResponseAPIUsage | None:
    if first is None or final is None:
        return final if first is None else first
    return ResponseAPIUsage.model_validate(_summed_mapping(first.model_dump(), final.model_dump()))


def _repeated_fixed_margin(
    first: ResponsesAPIResponse,
    first_breakdown: CostBreakdown | None,
    final: ResponsesAPIResponse,
    final_breakdown: CostBreakdown | None,
) -> float:
    if first_breakdown is None or final_breakdown is None or first_breakdown is final_breakdown:
        return 0.0
    if served_from_cache(first) or served_from_cache(final):
        return 0.0
    return min(first_breakdown.get("margin_fixed_amount", 0.0), final_breakdown.get("margin_fixed_amount", 0.0))


def _less_repeated_fixed_margin(summed: Mapping[str, object], repeated: float) -> Mapping[str, object]:
    charged_once: Final = MappingProxyType(
        {key: total - repeated for key in _TOTALS_WITH_FIXED_MARGIN if isinstance(total := summed.get(key), float)}
    )
    return MappingProxyType({**summed, **charged_once})


def summed_cost_breakdown(
    first: ResponsesAPIResponse,
    first_breakdown: CostBreakdown | None,
    final: ResponsesAPIResponse,
    final_breakdown: CostBreakdown | None,
) -> CostBreakdown | None:
    """The breakdown of the charged rounds, or None when a charged round was priced without its own breakdown, since a
    partial one would disagree with the billed cost. The fixed margin is charged once per request, not once per round."""
    if served_from_cache(final):
        return None if served_from_cache(first) else first_breakdown
    if first_breakdown is final_breakdown:
        return None
    if served_from_cache(first):
        return final_breakdown
    if first_breakdown is None or final_breakdown is None:
        return None
    return _COST_BREAKDOWN.validate_python(
        _less_repeated_fixed_margin(
            _summed_mapping(first_breakdown, final_breakdown),
            _repeated_fixed_margin(first, first_breakdown, final, final_breakdown),
        )
    )


def billed_for_every_round(
    first: ResponsesAPIResponse,
    first_breakdown: CostBreakdown | None,
    final: ResponsesAPIResponse,
    final_breakdown: CostBreakdown | None,
) -> ResponsesAPIResponse:
    """The final round's response carrying the usage and cost of both rounds, so the one logged row bills the request.

    Costs are summed per round rather than repriced from the summed usage, because a round's price can depend on its
    own size (tiered rates) or on charges the usage does not show (built-in tools, provider-reported costs). The fixed
    margin both rounds added is charged once, since it is a per-request fee.
    """
    first_cost: Final = first.hidden_params.get("response_cost")
    final_cost: Final = final.hidden_params.get("response_cost")
    total_cost: Final = (
        first_cost + final_cost - _repeated_fixed_margin(first, first_breakdown, final, final_breakdown)
        if isinstance(first_cost, (int, float)) and isinstance(final_cost, (int, float))
        else None
    )
    billed: Final = final.model_copy(update=MappingProxyType({"usage": merged_round_usage(first.usage, final.usage)}))
    billed.hidden_params = {
        **final.hidden_params,
        "response_cost": total_cost,
    }
    return billed


@contextmanager
def first_round_billed_on_failure(logging_obj: object, first: ResponsesAPIResponse) -> Generator[None]:
    try:
        yield
    except (Exception, CancelledError):
        first_cost: Final = first.hidden_params.get("response_cost")
        if isinstance(logging_obj, LiteLLMLoggingObj) and first.usage is not None and not served_from_cache(first):
            logging_obj.record_partial_usage_for_failure(
                ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(first.usage),
                float(first_cost) if isinstance(first_cost, (int, float)) else 0.0,
            )
        raise
