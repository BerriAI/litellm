"""Configuration for the best-of-n router.

A ``best_of_n/<name>`` deployment fans each request out to every arm in ``models``
in parallel, then hands the successful candidate responses to ``synthesizer``,
whose answer (a synthesis for text requests, a pick for tool-calling requests) is
returned to the client. Arm order is the operator's priority ranking: it decides
which candidate is returned when the synthesizer itself fails.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

from litellm.router_strategy.complexity_router.config import ComplexityTierModel

MIN_BEST_OF_N_ARMS: Final = 2
MAX_BEST_OF_N_ARMS: Final = 8


class BestOfNRouterConfig(BaseModel):
    """Validated shape of ``litellm_params.best_of_n_config``."""

    model_config = ConfigDict(frozen=True)

    models: tuple[ComplexityTierModel, ...]
    synthesizer: ComplexityTierModel

    @field_validator("models", mode="before")
    @classmethod
    def _normalize_models(cls, value: object) -> tuple[ComplexityTierModel, ...]:
        entries: Final = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(
            ComplexityTierModel(model_name=entry)
            if isinstance(entry, str)
            else ComplexityTierModel.model_validate(entry)
            for entry in entries
        )

    @field_validator("synthesizer", mode="before")
    @classmethod
    def _normalize_synthesizer(cls, value: object) -> ComplexityTierModel:
        if isinstance(value, str):
            return ComplexityTierModel(model_name=value)
        if isinstance(value, ComplexityTierModel):
            return value
        return ComplexityTierModel.model_validate(value)

    @field_validator("models", mode="after")
    @classmethod
    def _check_arm_count(cls, value: tuple[ComplexityTierModel, ...]) -> tuple[ComplexityTierModel, ...]:
        if not MIN_BEST_OF_N_ARMS <= len(value) <= MAX_BEST_OF_N_ARMS:
            raise ValueError(
                f"best_of_n_config.models needs between {MIN_BEST_OF_N_ARMS} and {MAX_BEST_OF_N_ARMS} arms, "
                f"got {len(value)}"
            )
        return value
