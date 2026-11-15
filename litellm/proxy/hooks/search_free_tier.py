from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time
from litellm.litellm_core_utils.llm_cost_calc.finalization import finalize_cost
from litellm.litellm_core_utils.llm_cost_calc.utils import get_web_search_requests_from_usage
from litellm.llms.gemini.cost_calculator import cost_per_web_search_request
from litellm.types.utils import ModelInfo, SearchFreeRequests, StandardLoggingPayload, Usage

_PERIOD_DURATION: Final = MappingProxyType({"day": "1d", "month": "1mo"})
_PERIOD_STAMP: Final = MappingProxyType({"day": "%Y-%m-%d", "month": "%Y-%m"})
_TTL_GRACE_SECONDS: Final = 3600
_COUNTED_KWARG: Final = "search_free_tier_counted"


class SharedCounter(Protocol):
    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> float: ...


class _UsageCarrier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    usage: Usage | None = None


def freed_units(allowance: int, units: int, total_after: int) -> int:
    """Units of this request that landed under the allowance, given the shared total after they were added."""
    return max(0, min(units, allowance - (total_after - units)))


def search_free_tier_key(provider: str, allowance: SearchFreeRequests, now: datetime) -> str:
    period_stamp: Final = now.strftime(_PERIOD_STAMP[allowance["period"]])
    return f"litellm:search_free_tier:{provider}:{allowance['shared_group']}:{period_stamp}"


def search_free_tier_ttl(allowance: SearchFreeRequests, now: datetime) -> int:
    reset: Final = get_next_standardized_reset_time(_PERIOD_DURATION[allowance["period"]], now, "UTC")
    return int((reset - now).total_seconds()) + _TTL_GRACE_SECONDS


def _usage_of(completion_response: object) -> Usage | None:
    try:
        return _UsageCarrier.model_validate(completion_response, from_attributes=True).usage
    except ValidationError:
        return None


def _grounding_queries(usage: Usage) -> int:
    details: Final = usage.prompt_tokens_details
    from_details: Final = getattr(details, "web_search_requests", None) if details is not None else None
    return from_details or get_web_search_requests_from_usage(usage) or 0


def _billable_units(usage: Usage, model_info: ModelInfo) -> int:
    queries: Final = _grounding_queries(usage)
    if queries == 0:
        return 0
    return queries if model_info.get("web_search_billing_unit") == "per_query" else 1


def _record_freed_cost(
    kwargs: MutableMapping[str, object],  # mutable-ok: writes the reduced cost where the spend row reads it
    sl_object: StandardLoggingPayload,
    original_cost: float,
    freed_cost: float,
) -> float:
    finalized: Final = finalize_cost(original_cost, sl_object["cost_breakdown"], freed_cost)
    sl_object["cost_breakdown"] = finalized.logging_breakdown()  # rebind-ok: legacy callback payload
    sl_object["response_cost"] = finalized.amount  # rebind-ok: legacy callback payload
    kwargs["response_cost"] = finalized.amount  # rebind-ok: legacy callback payload
    return finalized.amount


@dataclass(frozen=True, slots=True)
class SearchAllowanceAllocation:
    counter_key: str
    units: int
    free_units: int
    discount: float


async def apply_search_free_tier(
    kwargs: MutableMapping[str, object],  # mutable-ok: writes the reduced cost where the spend row reads it
    sl_object: StandardLoggingPayload,
    response_cost: float,
    completion_response: object,
    counter: SharedCounter | None,
    now: datetime,
) -> float:
    """Zero the grounding queries that landed under the model's shared free allowance, everything else is unchanged."""
    if kwargs.get(_COUNTED_KWARG):
        return response_cost
    allocation: Final = await allocate_search_free_tier(sl_object, response_cost, completion_response, counter, now)
    if allocation is None:
        return response_cost
    kwargs[_COUNTED_KWARG] = True  # rebind-ok: legacy callback deduplication
    return _record_freed_cost(kwargs, sl_object, response_cost, allocation.discount)


@with_service_target("search_free_tier")
async def allocate_search_free_tier(
    sl_object: StandardLoggingPayload,
    response_cost: float,
    completion_response: object,
    counter: SharedCounter | None,
    now: datetime,
) -> SearchAllowanceAllocation | None:
    model_info: Final = sl_object["model_map_information"]["model_map_value"]
    allowance: Final = model_info.get("search_free_requests") if model_info is not None else None
    usage: Final = _usage_of(completion_response)
    if model_info is None or allowance is None or usage is None or sl_object["cache_hit"]:
        return None
    units: Final = _billable_units(usage, model_info)
    if units == 0:
        return None
    if counter is None:
        verbose_proxy_logger.warning(
            "apply_search_free_tier is on but the proxy has no Redis, grounding is billed at full price"
        )
        return None
    provider: Final = sl_object["custom_llm_provider"] or model_info["litellm_provider"]
    key: Final = search_free_tier_key(provider, allowance, now)
    try:
        total_after: Final = await counter.async_increment(key, units, ttl=search_free_tier_ttl(allowance, now))
    except Exception:  # noqa: BLE001  # any counter failure means full price, never a lost spend row
        verbose_proxy_logger.warning(
            "search free tier counter %s is unavailable, grounding is billed at full price", key
        )
        return None
    freed: Final = freed_units(allowance["count"], units, int(total_after))
    freed_cost: Final = min(response_cost, cost_per_web_search_request(usage, model_info) * freed / units)
    return SearchAllowanceAllocation(key, units, freed, freed_cost)
