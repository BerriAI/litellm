from collections.abc import Mapping
from typing import Annotated, Final

from pydantic import Field, StrictFloat, StrictInt, TypeAdapter, ValidationError

from litellm.utils import get_model_info

EXA_AI_COST_PARAM: Final = "_exa_ai_cost_dollars"
EXA_AI_DEFAULT_RESULTS: Final = 10
EXA_AI_ADDITIONAL_RESULT_COST: Final = 0.001
EXA_AI_REPORTED_COST_ADAPTER: Final[TypeAdapter[float | int]] = TypeAdapter(
    Annotated[StrictInt | StrictFloat, Field(ge=0, allow_inf_nan=False)]
)


def exa_ai_reported_cost_dollars(value: object) -> float | None:
    try:
        reported_cost: Final = EXA_AI_REPORTED_COST_ADAPTER.validate_python(value)
        return float(reported_cost)
    except (OverflowError, ValidationError):
        return None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _effective_results(optional_params: Mapping[str, object]) -> int:
    max_results: Final = _non_negative_int(optional_params.get("max_results"))
    if max_results is not None:
        return max_results

    num_results: Final = _non_negative_int(optional_params.get("numResults"))
    return num_results if num_results is not None else EXA_AI_DEFAULT_RESULTS


def exa_ai_search_cost(model: str, optional_params: Mapping[str, object]) -> float:
    reported_cost: Final = exa_ai_reported_cost_dollars(optional_params.get(EXA_AI_COST_PARAM))
    if reported_cost is not None:
        from litellm.search.cost_calculator import model_cost_registered_at_runtime

        if not model_cost_registered_at_runtime(model):
            return reported_cost

    model_info: Final = get_model_info(model=model, custom_llm_provider="exa_ai")
    base_cost: Final = float(model_info.get("input_cost_per_query") or 0.0)
    additional_results: Final = max(_effective_results(optional_params) - EXA_AI_DEFAULT_RESULTS, 0)
    return base_cost + additional_results * EXA_AI_ADDITIONAL_RESULT_COST
