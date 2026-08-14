from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

PARALLEL_AI_DEFAULT_REASONING_EFFORT: Final[str] = "medium"
PARALLEL_AI_EFFORT_TIER_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "parallel-low": "low",
        "parallel-medium": "medium",
        "parallel-high": "high",
    }
)
PARALLEL_AI_REASONING_EFFORTS: Final[frozenset[str]] = frozenset(PARALLEL_AI_EFFORT_TIER_MODELS.values())
REASONING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def is_parallel_ai_response_model(model: str) -> bool:
    model_without_provider: Final[str] = model.removeprefix("parallel_ai/")
    return model_without_provider == "parallel" or model_without_provider in PARALLEL_AI_EFFORT_TIER_MODELS


def _reasoning_effort(optional_params: Mapping[str, object]) -> str:
    try:
        reasoning: Final = REASONING_ADAPTER.validate_python(optional_params.get("reasoning"))
    except ValidationError:
        return PARALLEL_AI_DEFAULT_REASONING_EFFORT
    effort: Final[object] = reasoning.get("effort")
    if isinstance(effort, str) and effort in PARALLEL_AI_REASONING_EFFORTS:
        return effort
    return PARALLEL_AI_DEFAULT_REASONING_EFFORT


def parallel_ai_response_pricing_model(model: str, optional_params: Mapping[str, object]) -> str:
    """Return the cost-map model matching the effective Responses reasoning effort."""
    model_without_provider: Final[str] = model.removeprefix("parallel_ai/")
    effort: Final[str] = PARALLEL_AI_EFFORT_TIER_MODELS.get(model_without_provider) or _reasoning_effort(
        optional_params
    )
    return f"parallel_ai/parallel-{effort}"
