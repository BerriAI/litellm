from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, NamedTuple

from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.llms.base_llm.systemone import (
    HttpJevClassifierClient,
    JevChoiceAnswer,
    JevChoiceQuestion,
    JevClassifierClient,
    JevProbability,
    JevSystemOneRequest,
    JevSystemOneResponse,
    JevUsage,
)
from litellm.router_strategy.complexity_router.config import DEFAULT_JEV_INSTRUCTIONS

__all__: Final = (
    "DEFAULT_JEV_INSTRUCTIONS",
    "HttpJevClassifierClient",
    "JevChoiceAnswer",
    "JevChoiceQuestion",
    "JevClassifierClient",
    "JevProbability",
    "JevSystemOneRequest",
    "JevSystemOneResponse",
    "JevUsage",
    "JevVerdict",
    "build_jev_request",
    "jev_classifier_cost",
)


class JevVerdict(NamedTuple):
    label: str
    probabilities: Mapping[str, float]
    confidence: float
    model: str
    cost: float | None
    provider: str = "typesafe"


class _RegistryPricing(BaseModel):
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0


_REGISTRY_PRICING_ADAPTER: Final = TypeAdapter(_RegistryPricing)


def build_jev_request(
    prompt: str,
    system_prompt: str | None,
    model: str,
    instructions: str,
    criteria: Mapping[str, str],
) -> JevSystemOneRequest:
    state: Final = prompt if system_prompt is None else f"System prompt:\n{system_prompt}\n\nRequest:\n{prompt}"
    question: Final = JevChoiceQuestion(instructions=instructions, criteria=criteria)
    return JevSystemOneRequest(state=state, model=model, questions=MappingProxyType({"tier": question}))


def jev_classifier_cost(
    response: JevSystemOneResponse, configured_model: str, provider: str = "typesafe"
) -> float | None:
    usage: Final = response.usage
    if usage is None:
        return None
    model: Final = response.model or configured_model
    model_key: Final = f"{provider}/{model}"
    if model_key not in litellm.model_cost:  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
        return None
    try:
        pricing: Final = _REGISTRY_PRICING_ADAPTER.validate_python(
            litellm.model_cost[model_key]  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
        )
    except ValidationError:
        return None
    return usage.input_tokens * pricing.input_cost_per_token + usage.output_tokens * pricing.output_cost_per_token
