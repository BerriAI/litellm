from itertools import chain
from typing import Annotated, Final

from pydantic import ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Self

from litellm.types.llms.base import LiteLLMBaseModel


class ModelThrottlingThreshold(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spend: float = Field(
        gt=0, allow_inf_nan=False, description="Total budget-period spend in USD that pauses these models."
    )
    models: tuple[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)], ...] = Field(
        min_length=1, description="Models to pause when this spend threshold is reached."
    )


class ModelThrottling(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thresholds: tuple[ModelThrottlingThreshold, ...] = Field(
        min_length=1, description="Intermediate spend thresholds. Paused models accumulate until the budget resets."
    )

    @model_validator(mode="after")
    def unique_thresholds_and_models(self) -> Self:
        spends: Final = tuple(threshold.spend for threshold in self.thresholds)
        models: Final = tuple(chain.from_iterable(threshold.models for threshold in self.thresholds))
        if len(set(spends)) != len(spends):
            raise ValueError("Combine models at the same spend threshold into one threshold.")
        if len(set(models)) != len(models):
            raise ValueError("Select each model at only one threshold. Earlier thresholds remain active.")
        return self


class ActiveModelThrottle(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    scope: str
    spend: float
    threshold: ModelThrottlingThreshold


def model_throttling_budget_error(config: ModelThrottling | None, max_budget: float | None) -> str | None:
    if config is None:
        return None
    if max_budget is None or max_budget <= 0:
        return "Set a positive final budget before enabling model throttling."
    if any(threshold.spend >= max_budget for threshold in config.thresholds):
        return "Each model throttling threshold must be below the final budget."
    return None
