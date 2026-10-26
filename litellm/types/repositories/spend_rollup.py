from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.clickhouse import RollupBucket, RollupDimension
from litellm.types.repositories.daily_activity import DailyActivityTable

ENTITY_ROLLUP_DIMENSIONS: Final[Mapping[DailyActivityTable, RollupDimension]] = MappingProxyType(
    {
        DailyActivityTable.USER: RollupDimension.USER,
        DailyActivityTable.TEAM: RollupDimension.TEAM_ID,
        DailyActivityTable.TAG: RollupDimension.TAG,
        DailyActivityTable.ORGANIZATION: RollupDimension.ORGANIZATION_ID,
        DailyActivityTable.CUSTOMER: RollupDimension.END_USER,
    }
)


@dataclass(frozen=True, slots=True)
class RollupWindow:
    start: datetime
    end: datetime
    bucket: RollupBucket


@dataclass(frozen=True, slots=True)
class SpendRollupScope:
    """A sub-day usage query answered from ClickHouse `spend_logs` instead of the daily Postgres tables."""

    dimension: RollupDimension
    entity_ids: tuple[str, ...] | None
    api_keys: tuple[str, ...] | None
    model: str | None
    window: RollupWindow
