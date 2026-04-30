"""Type definitions for model deprecation tracking and proactive alerts.

The proxy reads deprecation/sunset metadata from
``litellm.model_cost`` (sourced from ``model_prices_and_context_window.json``)
and surfaces it through the ``/model/deprecations`` endpoint and Slack
alerting. These types describe the response payload and the alert payload.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


DEFAULT_DEPRECATION_WARN_DAYS = int(
    os.getenv("LITELLM_MODEL_DEPRECATION_WARN_DAYS", "30")
)
"""Number of days before the deprecation date to start raising warnings.

Configurable via the ``LITELLM_MODEL_DEPRECATION_WARN_DAYS`` environment
variable. Defaults to 30 days, matching the typical migration window most
LLM providers offer between announcement and removal.
"""

DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS = int(
    os.getenv("LITELLM_MODEL_DEPRECATION_CHECK_INTERVAL", str(24 * 60 * 60))
)
"""How often the periodic background check runs. Defaults to once per day."""


DeprecationStatusLiteral = str
"""One of ``"upcoming"``, ``"imminent"``, ``"deprecated"``.

* ``upcoming`` – deprecation is scheduled but more than the warn window away.
* ``imminent`` – deprecation date is within ``warn_within_days`` from today.
* ``deprecated`` – deprecation date has already passed.
"""


class ModelDeprecationInfo(BaseModel):
    """Per-model deprecation metadata returned by ``/model/deprecations``."""

    model_name: str = Field(
        description="The public name of the model on the proxy (model_group)."
    )
    litellm_model: Optional[str] = Field(
        default=None,
        description="The underlying litellm model string the deprecation date is sourced from.",
    )
    deprecation_date: date = Field(
        description="The date (UTC) when the model becomes deprecated."
    )
    days_until_deprecation: int = Field(
        description=(
            "Days remaining until the deprecation date. Negative if the model "
            "is already deprecated."
        ),
    )
    status: DeprecationStatusLiteral = Field(
        description="One of 'upcoming', 'imminent', or 'deprecated'.",
    )
    litellm_provider: Optional[str] = Field(
        default=None, description="The provider this model belongs to."
    )


class ModelDeprecationResponse(BaseModel):
    """Response payload for ``GET /model/deprecations``."""

    deprecated: List[ModelDeprecationInfo] = Field(
        default_factory=list,
        description="Models whose deprecation date has already passed.",
    )
    imminent: List[ModelDeprecationInfo] = Field(
        default_factory=list,
        description=(
            "Models whose deprecation date is within ``warn_within_days`` from "
            "today and require immediate migration planning."
        ),
    )
    upcoming: List[ModelDeprecationInfo] = Field(
        default_factory=list,
        description="Models with a future deprecation date outside the warn window.",
    )
    warn_within_days: int = Field(
        description="The window (in days) used to bucket 'imminent' models."
    )
    checked_at: datetime = Field(
        description="UTC timestamp when the deprecation snapshot was generated."
    )
