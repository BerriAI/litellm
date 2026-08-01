"""
Types for tag-based rate limiting (`model_info.token_limits` / `request_limits` / `dollar_limits`).
"""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TagLimitUnit = Literal["tokens", "requests", "dollars"]

TAG_LIMIT_FIELD_BY_UNIT: Mapping[TagLimitUnit, str] = {
    "tokens": "token_limits",
    "requests": "request_limits",
    "dollars": "dollar_limits",
}


class TagLimit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tag_id: str
    limit: float = Field(gt=0)
    period_days: float = Field(gt=0)


class TagLimitGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limits: tuple[TagLimit, ...] = ()


class DeploymentTagLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    token_limits: TagLimitGroup | None = None
    request_limits: TagLimitGroup | None = None
    dollar_limits: TagLimitGroup | None = None

    def limits_by_unit(self) -> tuple[tuple[TagLimitUnit, TagLimit], ...]:
        groups: tuple[tuple[TagLimitUnit, TagLimitGroup | None], ...] = (
            ("tokens", self.token_limits),
            ("requests", self.request_limits),
            ("dollars", self.dollar_limits),
        )
        return tuple((unit, limit) for unit, group in groups if group is not None for limit in group.limits)
