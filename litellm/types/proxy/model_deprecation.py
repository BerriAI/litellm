from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Final, Literal

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

DEFAULT_DEPRECATION_WARN_DAYS: Final = 30

DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS: Final = 24 * 60 * 60

DEPRECATION_IDLE_POLL_SECONDS: Final = 30

DeprecationStatus = Literal["upcoming", "imminent", "deprecated"]


class SuccessorModelSource(str, Enum):
    MODEL_COST_MAP = "model_cost_map"
    DEPLOYMENT_CONFIG = "deployment_config"


class ModelDeprecationInfo(LiteLLMBaseModel):
    model_name: str = Field(description="The public name of the model on the proxy (model_group).")
    team_public_model_name: str | None = Field(
        default=None,
        description="The model name visible to the team, when configured.",
    )
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
    successor_model: str | None = Field(
        default=None,
        description=(
            "The model to migrate to: model_info.successor_model on the deployment when set, "
            "else the successor_model the cost map names for the deprecated model."
        ),
    )

    successor_model_source: SuccessorModelSource | None = Field(
        default=None,
        description="Whether the recommended successor comes from the model cost map or deployment configuration.",
    )
    deprecation_source: str | None = Field(
        default=None, description="Provider deprecation notice linked by the selected model cost-map entry."
    )


class ModelDeprecationResponse(LiteLLMBaseModel):
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


class DeprecationAlertStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class DeprecationAlertUpdate(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1, max_length=1024)
    deprecation_date: date
    status: DeprecationAlertStatus
    last_seen_date: date
    last_seen_requests: int = Field(ge=0)


class DeprecationAlertState(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    deprecation_date: date
    status: DeprecationAlertStatus = DeprecationAlertStatus.OPEN
    last_seen_date: date | None = None
    last_seen_requests: int = 0
    updated_at: datetime | None = None
    updated_by: str | None = None


class DeprecationTagUsage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str
    tag: str
