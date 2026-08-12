from __future__ import annotations

from datetime import date, datetime
from typing import Final, Literal

from pydantic import BaseModel, Field

DEFAULT_DEPRECATION_WARN_DAYS: Final = 30

DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS: Final = 24 * 60 * 60

DEPRECATION_IDLE_POLL_SECONDS: Final = 30

DeprecationStatus = Literal["upcoming", "imminent", "deprecated"]


class ModelDeprecationInfo(BaseModel):
    model_name: str = Field(description="The public name of the model on the proxy (model_group).")
    litellm_model: str | None = Field(
        default=None,
        description="The underlying litellm model string the deprecation date is sourced from.",
    )
    deprecation_date: date = Field(description="The date (UTC) when the model becomes deprecated.")
    days_until_deprecation: int = Field(
        description=("Days remaining until the deprecation date. Negative if the model is already deprecated."),
    )
    status: DeprecationStatus = Field(
        description=(
            "'deprecated' if the date has passed, 'imminent' if it falls within warn_within_days, 'upcoming' otherwise."
        ),
    )
    litellm_provider: str | None = Field(default=None, description="The provider this model belongs to.")


class ModelDeprecationResponse(BaseModel):
    deprecated: list[ModelDeprecationInfo] = Field(
        default_factory=list,
        description="Models whose deprecation date has already passed.",
    )
    imminent: list[ModelDeprecationInfo] = Field(
        default_factory=list,
        description=(
            "Models whose deprecation date is within warn_within_days from "
            "today and require immediate migration planning."
        ),
    )
    upcoming: list[ModelDeprecationInfo] = Field(
        default_factory=list,
        description="Models with a future deprecation date outside the warn window.",
    )
    warn_within_days: int = Field(description="The window (in days) used to bucket 'imminent' models.")
    checked_at: datetime = Field(description="UTC timestamp when the deprecation snapshot was generated.")
