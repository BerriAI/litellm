from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import TypeIs

from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import CostBreakdown

_RATE_KEYS: Final = frozenset({"discount_percent", "margin_percent"})
_COST_BREAKDOWN: Final = TypeAdapter(CostBreakdown)
_HIDDEN_PARAMS: Final = TypeAdapter(Mapping[str, object])


def _is_str_keyed(
    value: object,
) -> TypeIs[Mapping[str, object]]:  # guard-ok: model dumps and CostBreakdown have str keys
    return isinstance(value, Mapping)


def _summed(first: object, final: object) -> object:
    if first is None:
        return final
    if final is None:
        return first
    if isinstance(first, bool) or isinstance(final, bool):
        return final
    if isinstance(first, (int, float)) and isinstance(final, (int, float)):
        return first + final
    if _is_str_keyed(first) and _is_str_keyed(final):
        return _summed_mapping(first, final)
    return final


def _summed_entry(key: str, first: Mapping[str, object], final: Mapping[str, object]) -> object:
    if key in _RATE_KEYS:
        return final[key] if key in final else first[key]
    return _summed(first.get(key), final.get(key))


def _summed_mapping(first: Mapping[str, object], final: Mapping[str, object]) -> Mapping[str, object]:
    keys: Final = (*first, *(key for key in final if key not in first))
    return MappingProxyType({key: _summed_entry(key, first, final) for key in keys})


def merged_round_usage(first: ResponseAPIUsage | None, final: ResponseAPIUsage | None) -> ResponseAPIUsage | None:
    if first is None or final is None:
        return final if first is None else first
    return ResponseAPIUsage.model_validate(_summed_mapping(first.model_dump(), final.model_dump()))


def summed_cost_breakdown(first: CostBreakdown | None, final: CostBreakdown | None) -> CostBreakdown | None:
    """None when either round was priced without a breakdown, since a partial one would disagree with the billed cost."""
    if first is None or final is None or first is final:
        return None
    return _COST_BREAKDOWN.validate_python(_summed_mapping(first, final))


def _hidden_params(response: ResponsesAPIResponse) -> Mapping[str, object]:
    return _HIDDEN_PARAMS.validate_python(getattr(response, "_hidden_params", None))


def billed_for_every_round(first: ResponsesAPIResponse, final: ResponsesAPIResponse) -> ResponsesAPIResponse:
    """The final round's response carrying the usage and cost of both rounds, so the one logged row bills the request.

    Costs are summed per round rather than repriced from the summed usage, because a round's price can depend on its
    own size (tiered rates) or on charges the usage does not show (built-in tools, provider-reported costs).
    """
    final_hidden_params: Final = _hidden_params(final)
    first_cost: Final = _hidden_params(first).get("response_cost")
    final_cost: Final = final_hidden_params.get("response_cost")
    total_cost: Final = (
        first_cost + final_cost
        if isinstance(first_cost, (int, float)) and isinstance(final_cost, (int, float))
        else None
    )
    billed: Final = final.model_copy(update=MappingProxyType({"usage": merged_round_usage(first.usage, final.usage)}))
    billed._hidden_params = {  # pyright: ignore[reportPrivateUsage]  # no public accessor  # mutable-ok: the outer call writes into this dict
        **final_hidden_params,
        "response_cost": total_cost,
    }
    return billed
