from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Protocol, TypeAlias


class DailyActivityTable(str, Enum):
    USER = "litellm_dailyuserspend"
    TEAM = "litellm_dailyteamspend"
    TAG = "litellm_dailytagspend"
    ORGANIZATION = "litellm_dailyorganizationspend"
    CUSTOMER = "litellm_dailyenduserspend"
    AGENT = "litellm_dailyagentspend"


_ENTITY_FIELDS: Mapping[DailyActivityTable, frozenset[str]] = MappingProxyType(
    {
        DailyActivityTable.USER: frozenset(("user_id",)),
        DailyActivityTable.TEAM: frozenset(("team_id",)),
        DailyActivityTable.TAG: frozenset(("tag", "team_id")),
        DailyActivityTable.ORGANIZATION: frozenset(("organization_id",)),
        DailyActivityTable.CUSTOMER: frozenset(("end_user_id",)),
        DailyActivityTable.AGENT: frozenset(("agent_id",)),
    }
)


@dataclass(frozen=True, slots=True)
class DailyActivityScope:
    table: DailyActivityTable
    entity_id_field: str
    entity_ids: tuple[str, ...] | None
    exclude_entity_ids: tuple[str, ...]
    api_keys: tuple[str, ...] | None
    start_date: str
    end_date: str
    model: str | None
    timezone_offset_minutes: int | None
    include_current_utc_day: bool = False
    team_ids: tuple[str, ...] | None = None
    exclude_team_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] | None = None
    exclude_tags: tuple[str, ...] = ()
    # drop the automatic "User-Agent: ..." tags (see common_daily_activity._is_user_agent_tag)
    exclude_user_agent_tags: bool = False

    @property
    def dimension_filters(self) -> tuple[tuple[str, tuple[str, ...] | None, tuple[str, ...]], ...]:
        return (
            ("team_id", self.team_ids, self.exclude_team_ids),
            ("tag", self.tags, self.exclude_tags),
        )

    def __post_init__(self) -> None:
        if self.table is not DailyActivityTable.TAG and any(
            included is not None or excluded for _, included, excluded in self.dimension_filters
        ):
            raise ValueError("Team/tag dimension filters require the daily tag spend table")
        if self.exclude_user_agent_tags and self.table is not DailyActivityTable.TAG:
            raise ValueError("exclude_user_agent_tags requires the daily tag spend table")
        if self.entity_id_field not in _ENTITY_FIELDS[self.table]:
            raise ValueError(f"Invalid entity_id_field {self.entity_id_field!r} for {self.table.value}")


@dataclass(frozen=True, slots=True)
class KeySpendRow:
    api_key: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    api_requests: int
    successful_requests: int
    failed_requests: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


@dataclass(frozen=True, slots=True)
class KeyPage:
    rows: tuple[KeySpendRow, ...]
    total_api_keys: int


@dataclass(frozen=True, slots=True)
class KeyMetadataRow:
    api_key: str
    key_alias: str | None
    team_id: str | None
    user_id: str | None
    user_email: str | None
    key_exists: bool
    tags: tuple[str, ...]


class ExportType(str, Enum):
    DAILY = "daily"
    DAILY_WITH_KEYS = "daily_with_keys"
    DAILY_WITH_MODELS = "daily_with_models"
    DAILY_WITH_USERS = "daily_with_users"


@dataclass(frozen=True, slots=True)
class ExportRow:
    date: str
    entity_id: str
    entity_alias: str | None
    api_key: str | None
    key_alias: str | None
    user_id: str | None
    user_email: str | None
    model: str | None
    spend: float
    flat_cost: float
    prompt_tokens: int
    completion_tokens: int
    api_requests: int
    successful_requests: int
    failed_requests: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


@dataclass(frozen=True, slots=True)
class RollupMetricsRow:
    date: str | None
    api_key: str | None
    spend: float | None
    ptu_flat_cost: float | None = field(default=None, kw_only=True)
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    compression_saved_tokens: int | None
    compression_savings_spend: float | None
    prompt_caching_savings_spend: float | None
    gateway_injected_caching_savings_spend: float | None
    autorouter_savings_spend: float | None
    api_requests: int | None
    successful_requests: int | None
    failed_requests: int | None
    total_response_time_ms: int | None
    timed_requests: int | None


@dataclass(frozen=True, slots=True)
class GroupingSetsRow(RollupMetricsRow):
    model: str | None
    model_group: str | None
    custom_llm_provider: str | None
    mcp_namespaced_tool_name: str | None
    endpoint: str | None
    group_level: int
    distinct_api_keys: int | None


@dataclass(frozen=True, slots=True)
class EntityRollupRow(RollupMetricsRow):
    entity_id: str | None
    api_key_rolled: int
    distinct_api_keys: int | None


@dataclass(frozen=True, slots=True)
class AggregatedRows:
    grouping_rows: tuple[GroupingSetsRow, ...]
    entity_rows: tuple[EntityRollupRow, ...] | None
    distinct_api_keys: int


SpendLogsWindow: TypeAlias = tuple[datetime, datetime]


class DailyActivityProxyReads(Protocol):
    async def recover_key_metadata(
        self, resolved: Mapping[str, KeyMetadataRow], api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]: ...


class DailyActivityRow(Protocol):
    id: str
    date: str
    api_key: str
    model: str | None
    model_group: str | None
    custom_llm_provider: str | None
    mcp_namespaced_tool_name: str | None
    endpoint: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    compression_saved_tokens: int
    compression_savings_spend: float
    prompt_caching_savings_spend: float
    gateway_injected_caching_savings_spend: float
    autorouter_savings_spend: float
    spend: float
    api_requests: int
    successful_requests: int
    failed_requests: int
    total_response_time_ms: int
    timed_requests: int


@dataclass(frozen=True, slots=True)
class DailyRowsPage:
    total_count: int
    rows: tuple[DailyActivityRow, ...]
