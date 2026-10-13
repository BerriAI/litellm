from datetime import date
from typing import Literal, TypeAlias

from pydantic import ConfigDict

from litellm.types.proxy.control_plane.dataplanes import UnreachableDataplane
from litellm.types.proxy.management_endpoints.common_daily_activity import SpendMetrics
from litellm.types.utils import LiteLLMBaseModel

UsageGroupBy: TypeAlias = Literal["dataplane", "model", "model_group", "provider", "api_key", "team", "endpoint"]


class UsageWindowResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    start_date: date
    end_date: date


class DataplaneUsage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    metrics: SpendMetrics


class UsageSummaryResponse(LiteLLMBaseModel):
    """Fleet-wide totals for the window, with the per-dataplane split and whoever did not answer."""

    model_config = ConfigDict(frozen=True)

    window: UsageWindowResponse
    totals: SpendMetrics
    by_dataplane: tuple[DataplaneUsage, ...]
    partial: bool
    unreachable: tuple[UnreachableDataplane, ...]


class DailyUsageRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    group: str
    metrics: SpendMetrics


class DailyUsageResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    window: UsageWindowResponse
    group_by: UsageGroupBy
    rows: tuple[DailyUsageRow, ...]
    partial: bool
    unreachable: tuple[UnreachableDataplane, ...]


class UsageRank(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    group: str
    metrics: SpendMetrics


class TopUsageResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    window: UsageWindowResponse
    group_by: UsageGroupBy
    limit: int
    items: tuple[UsageRank, ...]
    partial: bool
    unreachable: tuple[UnreachableDataplane, ...]
