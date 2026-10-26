from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.utils import get_model_info

PARALLEL_AI_DEFAULT_RESULTS: Final = 10
PARALLEL_AI_ADDITIONAL_RESULT_COST: Final = 0.001
PARALLEL_AI_USAGE_PARAM: Final = "_parallel_ai_usage"
PARALLEL_AI_STANDARD_SEARCH_MODEL: Final = "parallel_ai/search"
PARALLEL_AI_FAST_SEARCH_MODEL: Final = "parallel_ai/search-fast"
PARALLEL_AI_TURBO_SEARCH_MODEL: Final = "parallel_ai/search-turbo"
PARALLEL_AI_PRICING_MODEL_BY_MODE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "fast": PARALLEL_AI_FAST_SEARCH_MODEL,
        "turbo": PARALLEL_AI_TURBO_SEARCH_MODEL,
    }
)
ADVANCED_SETTINGS_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage_count(usage: Sequence[Mapping[str, object]], sku: str) -> int | None:
    counts: Final = tuple(
        count
        for item in usage
        if item.get("name") == sku
        if (count := _non_negative_int(item.get("count"))) is not None
    )
    return sum(counts) if counts else None


def _effective_mode(optional_params: Mapping[str, object]) -> str:
    mode: Final = optional_params.get("mode")
    if isinstance(mode, str):
        return mode

    processor: Final = optional_params.get("processor")
    if processor == "pro":
        return "advanced"
    return "basic"


def _effective_max_results(optional_params: Mapping[str, object]) -> int:
    try:
        advanced_settings: Final = ADVANCED_SETTINGS_ADAPTER.validate_python(optional_params.get("advanced_settings"))
        advanced_max_results: Final = _non_negative_int(advanced_settings.get("max_results"))
        if advanced_max_results is not None:
            return advanced_max_results
    except ValidationError:
        pass

    max_results: Final = _non_negative_int(optional_params.get("max_results"))
    return max_results if max_results is not None else PARALLEL_AI_DEFAULT_RESULTS


def _request_cost(mode: str) -> float:
    pricing_model: Final = PARALLEL_AI_PRICING_MODEL_BY_MODE.get(mode, PARALLEL_AI_STANDARD_SEARCH_MODEL)
    model_info: Final = get_model_info(model=pricing_model, custom_llm_provider="parallel_ai")
    return float(model_info.get("input_cost_per_query") or 0.0)


def _additional_results(
    optional_params: Mapping[str, object],
    usage: Sequence[Mapping[str, object]] | None,
) -> int:
    usage_count: Final = _usage_count(usage, "sku_search_additional_results") if usage is not None else None
    if usage_count is not None:
        return usage_count
    if usage is not None:
        return 0
    return max(_effective_max_results(optional_params) - PARALLEL_AI_DEFAULT_RESULTS, 0)


def parallel_ai_search_cost(
    optional_params: Mapping[str, object],
    usage: Sequence[Mapping[str, object]] | None,
) -> float:
    request_cost: Final = _request_cost(_effective_mode(optional_params))
    request_count_from_usage: Final = _usage_count(usage, "sku_search") if usage is not None else None
    request_count: Final = request_count_from_usage if request_count_from_usage is not None else 1
    additional_results: Final = _additional_results(optional_params, usage)
    return request_count * request_cost + additional_results * PARALLEL_AI_ADDITIONAL_RESULT_COST
