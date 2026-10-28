from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._cost import Usage as TokenUsage
from litellm._cost import as_main_cost
from litellm.litellm_core_utils.llm_cost_calc.utils import price_usage, pricing_row

PARALLEL_AI_DEFAULT_RESULTS: Final = 10
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
    mode: Final = _effective_mode(optional_params)
    pricing_model: Final = PARALLEL_AI_PRICING_MODEL_BY_MODE.get(mode, PARALLEL_AI_STANDARD_SEARCH_MODEL)
    request_count_from_usage: Final = _usage_count(usage, "sku_search") if usage is not None else None
    prompt_cost, completion_cost = as_main_cost(
        price_usage(
            pricing_row(pricing_model, "parallel_ai", None),
            TokenUsage(
                prompt_tokens=0,
                text=0,
                output_text=0,
                search_queries=request_count_from_usage if request_count_from_usage is not None else 1,
                additional_results=_additional_results(optional_params, usage),
            ),
        )
    )
    return prompt_cost + completion_cost
