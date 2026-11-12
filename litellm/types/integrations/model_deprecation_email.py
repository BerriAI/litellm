"""Model retirement email milestones and recipient-scoped content."""

from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from typing import Final

from litellm.types.proxy.model_deprecation import ModelDeprecationInfo

MODEL_DEPRECATION_EMAIL_EVENT: Final = "Model Deprecations"


class DeprecationEmailMilestone(IntEnum):
    THIRTY_DAYS = 30
    ONE_DAY = 1


@dataclass(frozen=True, slots=True)
class ModelDeprecationEmail:
    model: ModelDeprecationInfo
    usage_model: str
    recipient_email: str
    milestone: DeprecationEmailMilestone
    key_count: int
    requests: int
    team_names: tuple[str, ...]
    start_date: date
    end_date: date


@dataclass(frozen=True, slots=True)
class RenderedDeprecationEmail:
    subject: str
    html: str
    text: str
