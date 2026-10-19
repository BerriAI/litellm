import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Final, Literal, NoReturn, Protocol

from fastapi import HTTPException, status
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm import constants
from litellm._logging import verbose_proxy_logger
from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.spend_tracking.key_metadata_recovery import (
    attach_user_details,
    recover_cli_session_key_metadata,
    recover_double_hashed_key_metadata,
    recover_key_metadata_from_spend_logs,
    recover_key_owner_from_daily_spend,
)
from litellm.proxy.spend_tracking.ptu_feature_flag import is_ptu_cost_attribution_enabled
from litellm.proxy.utils import PrismaClient
from litellm.repositories.daily_activity_repository import DailyActivityRepository
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BreakdownMetrics,
    DailySpendData,
    DailySpendMetadata,
    GroupedData,
    KeyMetadata,
    KeyMetricWithMetadata,
    MetricWithMetadata,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
)
from litellm.types.repositories.daily_activity import (
    DailyActivityScope,
    DailyActivityTable,
    EntityRollupRow,
    GroupingSetsRow,
    KeyMetadataRow,
    RollupMetricsRow,
    SpendLogsWindow,
)


@dataclass(frozen=True, slots=True)
class ScopeDenied:
    status_code: Literal[403, 404]
    reason: str


@dataclass(frozen=True, slots=True)
class InvalidDateRange:
    reason: str


def raise_public(error: ScopeDenied | InvalidDateRange) -> NoReturn:
    match error:
        case ScopeDenied():
            raise HTTPException(status_code=error.status_code, detail={"error": error.reason})
        case InvalidDateRange():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": error.reason})
        case _:
            assert_never(error)


@dataclass(frozen=True, slots=True)
class CanonicalDateRange:
    start: date
    end: date


def parse_canonical_date(value: str) -> date | None:
    """The daily spend tables store ``date`` as text and compare it against the raw request
    string, so only the exact ``YYYY-MM-DD`` spelling can match a row. Spellings the parser
    would normalise (``2026-9-24``, ``20260924``, full-width digits) are rejected instead."""
    try:
        parsed: Final = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def parse_canonical_date_range(start_date: str | None, end_date: str | None) -> CanonicalDateRange | InvalidDateRange:
    if start_date is None or end_date is None:
        return InvalidDateRange(reason="Please provide start_date and end_date")
    start: Final = parse_canonical_date(start_date)
    end: Final = parse_canonical_date(end_date)
    if start is None or end is None:
        return InvalidDateRange(reason="start_date and end_date must be valid YYYY-MM-DD dates")
    if end < start:
        return InvalidDateRange(reason="end_date must be on or after start_date")
    return CanonicalDateRange(start=start, end=end)


class DailySpendRecord(Protocol):
    @property
    def date(self) -> str: ...

    @property
    def api_key(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def model_group(self) -> str | None: ...

    @property
    def custom_llm_provider(self) -> str | None: ...

    @property
    def mcp_namespaced_tool_name(self) -> str | None: ...

    @property
    def endpoint(self) -> str | None: ...

    @property
    def prompt_tokens(self) -> int: ...

    @property
    def completion_tokens(self) -> int: ...

    @property
    def spend(self) -> float: ...

    @property
    def cache_read_input_tokens(self) -> int: ...

    @property
    def cache_creation_input_tokens(self) -> int: ...

    @property
    def compression_saved_tokens(self) -> int: ...

    @property
    def compression_savings_spend(self) -> float: ...

    @property
    def prompt_caching_savings_spend(self) -> float: ...

    @property
    def gateway_injected_caching_savings_spend(self) -> float: ...

    @property
    def autorouter_savings_spend(self) -> float: ...

    @property
    def api_requests(self) -> int: ...

    @property
    def successful_requests(self) -> int: ...

    @property
    def failed_requests(self) -> int: ...

    @property
    def total_response_time_ms(self) -> int: ...

    @property
    def timed_requests(self) -> int: ...


class _KeyMetadataDict(TypedDict, total=False):
    key_alias: ReadOnly[str | None]
    team_id: ReadOnly[str | None]
    user_id: ReadOnly[str | None]
    user_email: ReadOnly[str | None]
    key_exists: ReadOnly[bool]


class _AggregatedSpendData(TypedDict):
    results: ReadOnly[list[DailySpendData]]
    totals: ReadOnly[SpendMetrics]


def _key_metadata(api_key_metadata: Mapping[str, KeyMetadataRow], api_key: str) -> KeyMetadata:
    meta: Final = api_key_metadata.get(api_key)
    return KeyMetadata(
        key_alias=meta.key_alias if meta is not None else None,
        team_id=meta.team_id if meta is not None else None,
        user_id=meta.user_id if meta is not None else None,
        user_email=meta.user_email if meta is not None else None,
        key_exists=meta.key_exists if meta is not None else False,
    )


def _reported_flat_cost(record: DailySpendRecord | RollupMetricsRow) -> float:
    """Flat cost a daily row reports, which is zero unless PTU cost attribution is enabled.

    Both read paths funnel through here: the paginated path reads the ``ptu_flat_cost``
    column straight off the row, and the aggregated path reads the SUM() alias. Rows an
    operator accrued during an earlier opt-in stay in the table, so the gate lives on the
    read rather than on the query that produced the rows.

    The row is checked before the flag because this runs once per metric accumulation, and
    a record fans out across roughly a dozen breakdowns. The flag reads through the secret
    manager, uncached, so consulting it for every accumulation put thousands of lookups on
    a shared endpoint that made none before. Only a row actually carrying flat cost, which
    is a sentinel row, reaches it now.
    """
    raw: Final = getattr(record, "ptu_flat_cost", None) or 0.0
    if not raw:
        return 0.0
    if not is_ptu_cost_attribution_enabled():
        return 0.0
    return raw


def update_metrics(existing_metrics: SpendMetrics, record: DailySpendRecord) -> SpendMetrics:
    """Update metrics with new record data.

    Rollup rows can carry None for numeric fields when SUM() spans zero rows
    (e.g. a key with no spend), so coalesce to 0 before accumulating to avoid
    a TypeError. Mirrors the handling in ``_record_to_spend_metrics``.
    """
    prompt_tokens: Final = record.prompt_tokens or 0
    completion_tokens: Final = record.completion_tokens or 0
    existing_metrics.spend += record.spend or 0.0
    existing_metrics.flat_cost += _reported_flat_cost(record)
    existing_metrics.prompt_tokens += prompt_tokens
    existing_metrics.completion_tokens += completion_tokens
    existing_metrics.total_tokens += prompt_tokens + completion_tokens
    existing_metrics.cache_read_input_tokens += record.cache_read_input_tokens or 0
    existing_metrics.cache_creation_input_tokens += record.cache_creation_input_tokens or 0
    existing_metrics.compression_saved_tokens += record.compression_saved_tokens or 0
    existing_metrics.compression_savings_spend += record.compression_savings_spend or 0
    existing_metrics.prompt_caching_savings_spend += record.prompt_caching_savings_spend or 0
    existing_metrics.gateway_injected_caching_savings_spend += record.gateway_injected_caching_savings_spend or 0
    existing_metrics.autorouter_savings_spend += record.autorouter_savings_spend or 0
    existing_metrics.api_requests += record.api_requests or 0
    existing_metrics.successful_requests += record.successful_requests or 0
    existing_metrics.failed_requests += record.failed_requests or 0
    existing_metrics.total_response_time_ms += record.total_response_time_ms or 0
    existing_metrics.timed_requests += record.timed_requests or 0
    return existing_metrics


def _is_user_agent_tag(tag: str | None) -> bool:
    """Determine whether a tag should be treated as a User-Agent tag."""
    if not tag:
        return False
    normalized_tag: Final = tag.strip().lower()
    return normalized_tag.startswith("user-agent:") or normalized_tag.startswith("user agent:")


def compute_tag_metadata_totals(records: Sequence[DailySpendRecord]) -> SpendMetrics:
    """
    Deduplicate spend metrics for tags using request_id, ignoring User-Agent prefixed tags.

    Each unique request_id contributes at most one record (the tag with max spend) to metadata.
    """
    deduped_records: Final[dict[str, DailySpendRecord]] = {}
    for record in records:
        request_id: str | None = getattr(record, "request_id", None)
        if not request_id:
            continue

        tag_value: str | None = getattr(record, "tag", None)
        if _is_user_agent_tag(tag_value):
            continue

        current_best = deduped_records.get(request_id)
        if current_best is None or record.spend > current_best.spend:
            deduped_records[request_id] = record

    metadata_metrics: Final = SpendMetrics()
    for record in deduped_records.values():
        update_metrics(metadata_metrics, record)
    return metadata_metrics


def _entity_metadata(
    entity_metadata_field: Mapping[str, dict[str, object]] | None,
    entity_id: str,
) -> dict[str, object]:
    """The metadata payload for one entity breakdown bucket, empty when the caller passed none."""
    stored: Final = entity_metadata_field.get(entity_id) if entity_metadata_field else None
    return stored if stored is not None else {}


def update_breakdown_metrics(
    breakdown: BreakdownMetrics,
    record: DailySpendRecord,
    model_metadata: Mapping[str, dict[str, object]],
    provider_metadata: Mapping[str, dict[str, object]],
    api_key_metadata: Mapping[str, KeyMetadataRow],
    entity_id_field: str | None = None,
    entity_metadata_field: Mapping[str, dict[str, object]] | None = None,
) -> BreakdownMetrics:
    """Updates breakdown metrics for a single record using the existing update_metrics function.

    PTU sentinel rows (api_key == PTU_SENTINEL_API_KEY) add their flat cost to every
    parent bucket but never appear as an api_key row, and are kept out of the
    per-request provider breakdown."""

    is_ptu_sentinel: Final = record.api_key == PTU_SENTINEL_API_KEY

    # A PTU sentinel row keys on the deployment id so a rename cannot move it, and carries
    # the operator-facing name in model_group. The breakdown key is rendered directly as a
    # label, so display the name; two deployments sharing one name merge here, which is
    # what the write path used to do by collapsing them into a single row.
    model_key: Final = (record.model_group or record.model) if is_ptu_sentinel else record.model

    # Update model breakdown
    if model_key and model_key not in breakdown.models:
        breakdown.models[model_key] = MetricWithMetadata(
            metrics=SpendMetrics(),
            metadata=model_metadata.get(model_key, {}),  # Add any model-specific metadata here
        )
    if model_key:
        breakdown.models[model_key].metrics = update_metrics(breakdown.models[model_key].metrics, record)

        if not is_ptu_sentinel:
            # Update API key breakdown for this model
            if record.api_key not in breakdown.models[model_key].api_key_breakdown:
                breakdown.models[model_key].api_key_breakdown[record.api_key] = KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=_key_metadata(api_key_metadata, record.api_key),
                )
            breakdown.models[model_key].api_key_breakdown[record.api_key].metrics = update_metrics(
                breakdown.models[model_key].api_key_breakdown[record.api_key].metrics,
                record,
            )

    # Update model group breakdown
    model_group_key: Final = record.model_group or record.model
    if model_group_key and model_group_key not in breakdown.model_groups:
        breakdown.model_groups[model_group_key] = MetricWithMetadata(
            metrics=SpendMetrics(),
            metadata=model_metadata.get(model_group_key, {}),
        )
    if model_group_key:
        breakdown.model_groups[model_group_key].metrics = update_metrics(
            breakdown.model_groups[model_group_key].metrics, record
        )

        if not is_ptu_sentinel:
            # Update API key breakdown for this model
            if record.api_key not in breakdown.model_groups[model_group_key].api_key_breakdown:
                breakdown.model_groups[model_group_key].api_key_breakdown[record.api_key] = KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=_key_metadata(api_key_metadata, record.api_key),
                )
            breakdown.model_groups[model_group_key].api_key_breakdown[record.api_key].metrics = update_metrics(
                breakdown.model_groups[model_group_key].api_key_breakdown[record.api_key].metrics,
                record,
            )

    if record.mcp_namespaced_tool_name:
        if record.mcp_namespaced_tool_name not in breakdown.mcp_servers:
            breakdown.mcp_servers[record.mcp_namespaced_tool_name] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata={},
            )
        breakdown.mcp_servers[record.mcp_namespaced_tool_name].metrics = update_metrics(
            breakdown.mcp_servers[record.mcp_namespaced_tool_name].metrics, record
        )

        # Update API key breakdown for this MCP server
        if record.api_key not in breakdown.mcp_servers[record.mcp_namespaced_tool_name].api_key_breakdown:
            breakdown.mcp_servers[record.mcp_namespaced_tool_name].api_key_breakdown[record.api_key] = (
                KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=_key_metadata(api_key_metadata, record.api_key),
                )
            )

        breakdown.mcp_servers[record.mcp_namespaced_tool_name].api_key_breakdown[
            record.api_key
        ].metrics = update_metrics(
            breakdown.mcp_servers[record.mcp_namespaced_tool_name].api_key_breakdown[record.api_key].metrics,
            record,
        )

    if not is_ptu_sentinel:
        # Update provider breakdown
        provider: Final = record.custom_llm_provider or "unknown"
        if provider not in breakdown.providers:
            breakdown.providers[provider] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=provider_metadata.get(provider, {}),  # Add any provider-specific metadata here
            )
        breakdown.providers[provider].metrics = update_metrics(breakdown.providers[provider].metrics, record)

        # Update API key breakdown for this provider
        if record.api_key not in breakdown.providers[provider].api_key_breakdown:
            breakdown.providers[provider].api_key_breakdown[record.api_key] = KeyMetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=_key_metadata(api_key_metadata, record.api_key),
            )
        breakdown.providers[provider].api_key_breakdown[record.api_key].metrics = update_metrics(
            breakdown.providers[provider].api_key_breakdown[record.api_key].metrics,
            record,
        )

    # Update endpoint breakdown
    if record.endpoint:
        if record.endpoint not in breakdown.endpoints:
            breakdown.endpoints[record.endpoint] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata={},
            )
        breakdown.endpoints[record.endpoint].metrics = update_metrics(
            breakdown.endpoints[record.endpoint].metrics, record
        )

        # Update API key breakdown for this endpoint
        if record.api_key not in breakdown.endpoints[record.endpoint].api_key_breakdown:
            breakdown.endpoints[record.endpoint].api_key_breakdown[record.api_key] = KeyMetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=_key_metadata(api_key_metadata, record.api_key),
            )
        breakdown.endpoints[record.endpoint].api_key_breakdown[record.api_key].metrics = update_metrics(
            breakdown.endpoints[record.endpoint].api_key_breakdown[record.api_key].metrics,
            record,
        )

    if not is_ptu_sentinel:
        # Update api key breakdown
        if record.api_key not in breakdown.api_keys:
            breakdown.api_keys[record.api_key] = KeyMetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=_key_metadata(api_key_metadata, record.api_key),
            )
        breakdown.api_keys[record.api_key].metrics = update_metrics(breakdown.api_keys[record.api_key].metrics, record)

    # Update entity-specific metrics if entity_id_field is provided
    if entity_id_field:
        entity_value: Final[str] = getattr(record, entity_id_field, None) or "Unassigned"
        if entity_value not in breakdown.entities:
            breakdown.entities[entity_value] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=_entity_metadata(entity_metadata_field, entity_value),
            )
        breakdown.entities[entity_value].metrics = update_metrics(breakdown.entities[entity_value].metrics, record)

        if not is_ptu_sentinel:
            # Update API key breakdown for this entity
            if record.api_key not in breakdown.entities[entity_value].api_key_breakdown:
                breakdown.entities[entity_value].api_key_breakdown[record.api_key] = KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=_key_metadata(api_key_metadata, record.api_key),
                )
            breakdown.entities[entity_value].api_key_breakdown[record.api_key].metrics = update_metrics(
                breakdown.entities[entity_value].api_key_breakdown[record.api_key].metrics,
                record,
            )

    return breakdown


def spend_logs_window(dates: AbstractSet[str | None]) -> tuple[datetime, datetime] | None:
    parsed: Final = sorted(day for day in (_parse_spend_date(raw) for raw in dates) if day is not None)
    if not parsed:
        return None
    return (parsed[0] - timedelta(days=1), parsed[-1] + timedelta(days=2))


def _parse_spend_date(raw: str | None) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _metadata_with_recovered_owner(
    metadata: Mapping[str, _KeyMetadataDict],
    key: str,
    owner: str,
) -> _KeyMetadataDict:
    current: Final = metadata.get(key)
    if current is None:
        return {"user_id": owner}
    return {**current, "user_id": owner}


@dataclass(frozen=True, slots=True)
class _ProxyDailyActivityReads:
    prisma_client: PrismaClient

    async def recover_key_metadata(
        self, resolved: Mapping[str, KeyMetadataRow], api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]:
        result: Final[dict[str, _KeyMetadataDict]] = {
            key: {
                "key_alias": row.key_alias,
                "team_id": row.team_id,
                "user_id": row.user_id,
                "user_email": row.user_email,
                "key_exists": row.key_exists,
            }
            for key, row in resolved.items()
        }
        from_session_keys: Final = await recover_cli_session_key_metadata(
            self.prisma_client, api_keys - frozenset(result)
        )
        still_missing: Final = api_keys - frozenset(result) - frozenset(from_session_keys)
        from_reverse_hash: Final = (
            await recover_double_hashed_key_metadata(self.prisma_client, still_missing)
            if still_missing
            else MappingProxyType({})
        )
        after_token_recovery: Final = MappingProxyType({**result, **from_session_keys, **from_reverse_hash})
        unresolved: Final = api_keys - frozenset(after_token_recovery)
        from_spend_logs: Final = (
            await recover_key_metadata_from_spend_logs(self.prisma_client, unresolved, window)
            if unresolved and window is not None
            else MappingProxyType({})
        )
        combined: Final = MappingProxyType({**after_token_recovery, **from_spend_logs})
        ownerless: Final = frozenset(
            key
            for key in api_keys
            if not combined.get(key, {}).get("user_id") and not combined.get(key, {}).get("key_exists")
        )
        owners: Final = await recover_key_owner_from_daily_spend(self.prisma_client, ownerless)
        with_owners: Final[Mapping[str, _KeyMetadataDict]] = MappingProxyType(
            {
                **combined,
                **{key: _metadata_with_recovered_owner(combined, key, owner) for key, owner in owners.items()},
            }
        )
        attached: Final = await attach_user_details(self.prisma_client, with_owners)
        return MappingProxyType(
            {
                key: replace(
                    resolved[key],
                    key_alias=value.get("key_alias"),
                    team_id=value.get("team_id"),
                    user_id=value.get("user_id"),
                    user_email=value.get("user_email"),
                    key_exists=value.get("key_exists", False),
                )
                if key in resolved
                else KeyMetadataRow(
                    api_key=key,
                    key_alias=value.get("key_alias"),
                    team_id=value.get("team_id"),
                    user_id=value.get("user_id"),
                    user_email=value.get("user_email"),
                    key_exists=value.get("key_exists", False),
                    tags=(),
                )
                for key, value in attached.items()
            }
        )


def daily_activity_repository(prisma_client: PrismaClient) -> DailyActivityRepository:
    return DailyActivityRepository(prisma_client, proxy_reads=_ProxyDailyActivityReads(prisma_client))


def daily_activity_scope(
    table: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,
    exclude_entity_ids: list[str] | None,
    api_key: str | list[str] | None,
    start_date: str,
    end_date: str,
    model: str | None,
    timezone_offset_minutes: int | None,
    include_current_utc_day: bool = False,
) -> DailyActivityScope:
    table_value: Final = DailyActivityTable(table)
    entity_ids: tuple[str, ...] | None = (
        (entity_id,) if isinstance(entity_id, str) else tuple(entity_id) if entity_id is not None else None
    )
    api_keys: tuple[str, ...] | None = (
        None if api_key in (None, "") else (api_key,) if isinstance(api_key, str) else tuple(api_key)
    )
    return DailyActivityScope(
        table=table_value,
        entity_id_field=entity_id_field,
        entity_ids=entity_ids,
        exclude_entity_ids=tuple(exclude_entity_ids or ()),
        api_keys=api_keys,
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone_offset_minutes,
        include_current_utc_day=include_current_utc_day,
    )


async def get_api_key_metadata(
    prisma_client: PrismaClient, api_keys: AbstractSet[str], spend_logs_window: SpendLogsWindow | None = None
) -> Mapping[str, _KeyMetadataDict]:
    rows: Final = await daily_activity_repository(prisma_client).key_metadata(frozenset(api_keys), spend_logs_window)
    return {
        key: {
            "key_alias": value.key_alias,
            "team_id": value.team_id,
            "user_id": value.user_id,
            "user_email": value.user_email,
            "key_exists": value.key_exists,
        }
        for key, value in rows.items()
    }


def _aggregate_spend_records_sync(
    *,
    records: Sequence[DailySpendRecord],
    api_key_metadata: Mapping[str, KeyMetadataRow],
    entity_id_field: str | None,
    entity_metadata_field: Mapping[str, dict[str, object]] | None,
) -> _AggregatedSpendData:
    model_metadata: Final[dict[str, dict[str, object]]] = {}
    provider_metadata: Final[dict[str, dict[str, object]]] = {}

    results: Final[list[DailySpendData]] = []
    total_metrics = SpendMetrics()
    grouped_data: Final[dict[str, GroupedData]] = {}

    for record in records:
        date_str = record.date
        if date_str not in grouped_data:
            grouped_data[date_str] = {
                "metrics": SpendMetrics(),
                "breakdown": BreakdownMetrics(),
            }

        grouped_data[date_str]["metrics"] = update_metrics(grouped_data[date_str]["metrics"], record)

        grouped_data[date_str]["breakdown"] = update_breakdown_metrics(
            grouped_data[date_str]["breakdown"],
            record,
            model_metadata,
            provider_metadata,
            api_key_metadata,
            entity_id_field=entity_id_field,
            entity_metadata_field=entity_metadata_field,
        )

        total_metrics = update_metrics(total_metrics, record)

    for date_str, data in grouped_data.items():
        results.append(
            DailySpendData(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                metrics=data["metrics"],
                breakdown=data["breakdown"],
            )
        )

    results.sort(key=lambda x: x.date, reverse=True)

    return {"results": results, "totals": total_metrics}


async def _aggregate_spend_records(
    *,
    repository: DailyActivityRepository,
    records: Sequence[DailySpendRecord],
    entity_id_field: str | None,
    entity_metadata_field: Mapping[str, dict[str, object]] | None,
) -> _AggregatedSpendData:
    """Aggregate rows into DailySpendData list and total metrics.

    The per-row loop is offloaded to a worker thread via asyncio.to_thread so
    a large result set doesn't peg the event loop.
    """
    api_keys: Final[set[str]] = {
        record.api_key for record in records if record.api_key and record.api_key != PTU_SENTINEL_API_KEY
    }

    api_key_metadata: Final[Mapping[str, KeyMetadataRow]] = (
        await repository.key_metadata(
            frozenset(api_keys), spend_logs_window(frozenset(record.date for record in records))
        )
        if api_keys
        else MappingProxyType({})
    )

    return await asyncio.to_thread(
        _aggregate_spend_records_sync,
        records=records,
        api_key_metadata=api_key_metadata,
        entity_id_field=entity_id_field,
        entity_metadata_field=entity_metadata_field,
    )


# GROUPING() bitmask values returned by the daily activity repository. Per Postgres semantics, the rightmost argument
# is the least-significant bit. Argument order:
#   date, api_key, model, model_group, custom_llm_provider,
#   mcp_namespaced_tool_name, endpoint
# A bit is 1 when the corresponding column is rolled up (i.e. NOT in the
# current grouping set's key), 0 when the column is part of the key.
_GROUP_GRAND_TOTAL: Final = 127  # 0b1111111 — all rolled up
_GROUP_DATE: Final = 63  # 0b0111111 — only date kept
_API_KEY_ROLLED_UP_BIT: Final = 32  # 0b0100000
_GROUP_DATE_API_KEY: Final = 31  # 0b0011111
_GROUP_DATE_MODEL: Final = 47  # 0b0101111
_GROUP_DATE_MODEL_API_KEY: Final = 15  # 0b0001111
_GROUP_DATE_MODEL_GROUP: Final = 55  # 0b0110111
_GROUP_DATE_MODEL_GROUP_API_KEY: Final = 23  # 0b0010111
_GROUP_DATE_PROVIDER: Final = 59  # 0b0111011
_GROUP_DATE_PROVIDER_API_KEY: Final = 27  # 0b0011011
_GROUP_DATE_MCP: Final = 61  # 0b0111101
_GROUP_DATE_MCP_API_KEY: Final = 29  # 0b0011101
_GROUP_DATE_ENDPOINT: Final = 62  # 0b0111110
_GROUP_DATE_ENDPOINT_API_KEY: Final = 30  # 0b0011110


def _record_to_spend_metrics(record: RollupMetricsRow) -> SpendMetrics:
    """Build a SpendMetrics directly from one already-aggregated rollup row.

    SUM() over zero rows is SQL NULL, so rollup rows (notably the grand-total
    row, which Postgres emits even on an empty match) can carry None values.
    """
    prompt_tokens: Final = record.prompt_tokens or 0
    completion_tokens: Final = record.completion_tokens or 0
    return SpendMetrics(
        spend=record.spend or 0.0,
        flat_cost=_reported_flat_cost(record),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens or 0,
        cache_creation_input_tokens=record.cache_creation_input_tokens or 0,
        compression_saved_tokens=record.compression_saved_tokens or 0,
        compression_savings_spend=record.compression_savings_spend or 0,
        prompt_caching_savings_spend=record.prompt_caching_savings_spend or 0,
        gateway_injected_caching_savings_spend=record.gateway_injected_caching_savings_spend or 0,
        autorouter_savings_spend=record.autorouter_savings_spend or 0,
        api_requests=record.api_requests or 0,
        successful_requests=record.successful_requests or 0,
        failed_requests=record.failed_requests or 0,
        total_response_time_ms=record.total_response_time_ms or 0,
        timed_requests=record.timed_requests or 0,
    )


def _aggregate_grouping_sets_records_sync(
    *,
    records: Sequence[GroupingSetsRow],
    api_key_metadata: Mapping[str, KeyMetadataRow],
) -> _AggregatedSpendData:
    """Build the response from rollup rows produced by the GROUPING SETS query.

    Each row carries a `group_level` bitmask (from Postgres GROUPING()) that
    identifies which rollup level it belongs to. We dispatch the row's
    pre-aggregated metrics straight into the matching bucket — no per-row
    summing in Python and no nested update_metrics calls.
    """
    total_metrics = SpendMetrics()
    grouped_data: Final[dict[str, GroupedData]] = {}

    def ensure_date(date_str: str) -> GroupedData:
        bucket: GroupedData | None = grouped_data.get(date_str)
        if bucket is None:
            bucket = {"metrics": SpendMetrics(), "breakdown": BreakdownMetrics()}
            grouped_data[date_str] = bucket
        return bucket

    def assign_metric_with_metadata(target: dict[str, MetricWithMetadata], key: str, metrics: SpendMetrics) -> None:
        existing: Final = target.get(key)
        if existing is None:
            target[key] = MetricWithMetadata(metrics=metrics, metadata={})
        else:
            existing.metrics = metrics

    def assign_api_key_breakdown(
        target: dict[str, MetricWithMetadata],
        parent_key: str,
        api_key: str,
        metrics: SpendMetrics,
    ) -> None:
        parent = target.get(parent_key)
        if parent is None:
            parent = MetricWithMetadata(metrics=SpendMetrics(), metadata={})
            target[parent_key] = parent
        parent.api_key_breakdown[api_key] = KeyMetricWithMetadata(
            metrics=metrics, metadata=_key_metadata(api_key_metadata, api_key)
        )

    for record in records:
        level = record.group_level
        metrics = _record_to_spend_metrics(record)
        is_ptu_sentinel = record.api_key == PTU_SENTINEL_API_KEY

        if level == _GROUP_GRAND_TOTAL:
            total_metrics = metrics
            continue

        if level == _GROUP_DATE:
            ensure_date(record.date)["metrics"] = metrics
            continue

        breakdown = ensure_date(record.date)["breakdown"]

        if level == _GROUP_DATE_API_KEY:
            if record.api_key and not is_ptu_sentinel:
                breakdown.api_keys[record.api_key] = KeyMetricWithMetadata(
                    metrics=metrics,
                    metadata=_key_metadata(api_key_metadata, record.api_key),
                )
        elif level == _GROUP_DATE_MODEL:
            if record.model:
                assign_metric_with_metadata(breakdown.models, record.model, metrics)
        elif level == _GROUP_DATE_MODEL_API_KEY:
            if record.model and record.api_key and not is_ptu_sentinel:
                assign_api_key_breakdown(breakdown.models, record.model, record.api_key, metrics)
        elif level == _GROUP_DATE_MODEL_GROUP:
            if record.model_group:
                assign_metric_with_metadata(breakdown.model_groups, record.model_group, metrics)
        elif level == _GROUP_DATE_MODEL_GROUP_API_KEY:
            if record.model_group and record.api_key and not is_ptu_sentinel:
                assign_api_key_breakdown(
                    breakdown.model_groups,
                    record.model_group,
                    record.api_key,
                    metrics,
                )
        elif level == _GROUP_DATE_PROVIDER:
            # Only PTU sentinel rows carry ptu_flat_cost and they have no provider, so at
            # this level the sentinel's cost would land under "unknown". Withholding the
            # flat cost matches the per-row path, which skips sentinel rows outright. The
            # bucket itself is still assigned unconditionally: a legacy row predating the
            # api_requests column backfills to all zeroes, and skipping those would drop a
            # provider the base build reported.
            provider_metrics = metrics.model_copy(update={"flat_cost": 0.0})
            provider = record.custom_llm_provider or "unknown"
            assign_metric_with_metadata(breakdown.providers, provider, provider_metrics)
        elif level == _GROUP_DATE_PROVIDER_API_KEY:
            if record.api_key and not is_ptu_sentinel:
                provider = record.custom_llm_provider or "unknown"
                assign_api_key_breakdown(breakdown.providers, provider, record.api_key, metrics)
        elif level == _GROUP_DATE_MCP:
            if record.mcp_namespaced_tool_name:
                assign_metric_with_metadata(breakdown.mcp_servers, record.mcp_namespaced_tool_name, metrics)
        elif level == _GROUP_DATE_MCP_API_KEY:
            if record.mcp_namespaced_tool_name and record.api_key:
                assign_api_key_breakdown(
                    breakdown.mcp_servers,
                    record.mcp_namespaced_tool_name,
                    record.api_key,
                    metrics,
                )
        elif level == _GROUP_DATE_ENDPOINT:
            if record.endpoint:
                assign_metric_with_metadata(breakdown.endpoints, record.endpoint, metrics)
        elif level == _GROUP_DATE_ENDPOINT_API_KEY:
            if record.endpoint and record.api_key:
                assign_api_key_breakdown(breakdown.endpoints, record.endpoint, record.api_key, metrics)

    results: Final = [
        DailySpendData(
            date=datetime.strptime(date_str, "%Y-%m-%d").date(),
            metrics=data["metrics"],
            breakdown=data["breakdown"],
        )
        for date_str, data in grouped_data.items()
    ]
    results.sort(key=lambda x: x.date, reverse=True)

    return {"results": results, "totals": total_metrics}


async def _aggregate_grouping_sets_records(
    *,
    repository: DailyActivityRepository,
    records: Sequence[GroupingSetsRow],
) -> _AggregatedSpendData:
    """Async wrapper: fetch api_key_metadata, then dispatch on a worker thread."""
    api_keys: Final[set[str]] = {r.api_key for r in records if r.api_key and r.api_key != PTU_SENTINEL_API_KEY}

    api_key_metadata: Final[Mapping[str, KeyMetadataRow]] = (
        await repository.key_metadata(frozenset(api_keys), spend_logs_window(frozenset(r.date for r in records)))
        if api_keys
        else MappingProxyType({})
    )

    return await asyncio.to_thread(
        _aggregate_grouping_sets_records_sync,
        records=records,
        api_key_metadata=api_key_metadata,
    )


async def get_daily_activity(
    prisma_client: PrismaClient | None,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,
    entity_metadata_field: Mapping[str, dict[str, object]] | None,
    start_date: str | None,
    end_date: str | None,
    model: str | None,
    api_key: str | list[str] | None,
    page: int,
    page_size: int,
    exclude_entity_ids: list[str] | None = None,
    metadata_metrics_func: Callable[[Sequence[DailySpendRecord]], SpendMetrics] | None = None,
    timezone_offset_minutes: int | None = None,
    include_current_utc_day: bool = False,
    resolve_entity_metadata: Callable[[Sequence[DailySpendRecord]], Awaitable[dict[str, dict[str, object]]]]
    | None = None,
) -> SpendAnalyticsPaginatedResponse:
    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})
    date_range: Final = parse_canonical_date_range(start_date, end_date)
    if isinstance(date_range, InvalidDateRange):
        raise_public(date_range)

    if page < 1 or page_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"page and page_size must be >= 1, got page={page}, page_size={page_size}",
        )

    try:
        scope: Final = daily_activity_scope(
            table_name,
            entity_id_field,
            entity_id,
            exclude_entity_ids,
            api_key,
            date_range.start.isoformat(),
            date_range.end.isoformat(),
            model,
            timezone_offset_minutes,
            include_current_utc_day,
        )
        repository: Final = daily_activity_repository(prisma_client)
        page_data: Final = await repository.daily_rows(scope, page=page, page_size=page_size)
        daily_spend_data: Final = page_data.rows
        resolved_entity_metadata = entity_metadata_field
        if resolve_entity_metadata is not None:
            resolved_entity_metadata = {
                **(entity_metadata_field or {}),
                **(await resolve_entity_metadata(daily_spend_data)),
            }
        aggregated: Final = await _aggregate_spend_records(
            repository=repository,
            records=daily_spend_data,
            entity_id_field=entity_id_field,
            entity_metadata_field=resolved_entity_metadata,
        )
        metadata_metrics = aggregated["totals"]
        if metadata_metrics_func:
            metadata_metrics = metadata_metrics_func(daily_spend_data)
        return SpendAnalyticsPaginatedResponse(
            results=aggregated["results"],
            metadata=DailySpendMetadata(
                total_spend=metadata_metrics.spend,
                total_flat_cost=metadata_metrics.flat_cost,
                total_prompt_tokens=metadata_metrics.prompt_tokens,
                total_completion_tokens=metadata_metrics.completion_tokens,
                total_tokens=metadata_metrics.total_tokens,
                total_api_requests=metadata_metrics.api_requests,
                total_successful_requests=metadata_metrics.successful_requests,
                total_failed_requests=metadata_metrics.failed_requests,
                total_cache_read_input_tokens=metadata_metrics.cache_read_input_tokens,
                total_cache_creation_input_tokens=metadata_metrics.cache_creation_input_tokens,
                total_compression_saved_tokens=metadata_metrics.compression_saved_tokens,
                total_compression_savings_spend=metadata_metrics.compression_savings_spend,
                total_prompt_caching_savings_spend=metadata_metrics.prompt_caching_savings_spend,
                total_gateway_injected_caching_savings_spend=metadata_metrics.gateway_injected_caching_savings_spend,
                total_autorouter_savings_spend=metadata_metrics.autorouter_savings_spend,
                total_response_time_ms=metadata_metrics.total_response_time_ms,
                total_timed_requests=metadata_metrics.timed_requests,
                page=page,
                total_pages=-(-page_data.total_count // page_size),
                has_more=(page * page_size) < page_data.total_count,
            ),
        )
    except Exception as exc:
        verbose_proxy_logger.exception("Error fetching daily activity: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": f"Failed to fetch analytics: {exc}"}
        )


def _fold_entity_rollups_sync(
    *,
    results: Sequence[DailySpendData],
    entity_rows: Sequence[EntityRollupRow],
    api_key_metadata: Mapping[str, KeyMetadataRow],
    entity_metadata_field: Mapping[str, dict[str, object]] | None,  # mutable-ok: shared field shape
) -> None:
    """Write breakdown.entities onto the already-built per-day results."""
    by_date: Final = {day.date.strftime("%Y-%m-%d"): day for day in results}

    for row in entity_rows:
        day = by_date.get(row.date)
        if day is None:
            continue

        entities = day.breakdown.entities
        entity_id = row.entity_id or "Unassigned"
        bucket = entities.get(entity_id)
        if bucket is None:
            bucket = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=_entity_metadata(entity_metadata_field, entity_id),
            )
            entities[entity_id] = bucket

        metrics = _record_to_spend_metrics(row)
        if row.api_key_rolled:
            bucket.metrics = metrics
        elif row.api_key and row.api_key != PTU_SENTINEL_API_KEY:
            bucket.api_key_breakdown[row.api_key] = KeyMetricWithMetadata(
                metrics=metrics, metadata=_key_metadata(api_key_metadata, row.api_key)
            )


async def get_daily_activity_aggregated(
    repository: DailyActivityRepository,
    scope: DailyActivityScope,
    *,
    entity_metadata_field: Mapping[str, dict[str, object]] | None = None,
    include_entity_breakdown: bool = False,
    api_key_limit: int = constants.USAGE_TOP_API_KEYS_DEFAULT,
) -> SpendAnalyticsPaginatedResponse:
    try:
        aggregated_rows: Final = await repository.aggregated(
            scope, include_entity_breakdown=include_entity_breakdown, api_key_limit=api_key_limit
        )
        records: Final = aggregated_rows.grouping_rows
        aggregated: Final = await _aggregate_grouping_sets_records(
            repository=repository,
            records=records,
        )
        entity_total_api_keys: Final[dict[str, int] | None] = (
            {
                row.entity_id or "Unassigned": row.distinct_api_keys
                for row in aggregated_rows.entity_rows or ()
                if row.api_key_rolled and row.distinct_api_keys is not None
            }
            if include_entity_breakdown
            else None
        )
        if aggregated_rows.entity_rows:
            entity_records: Final = aggregated_rows.entity_rows
            entity_api_keys: Final = frozenset(
                row.api_key for row in entity_records if row.api_key and row.api_key != PTU_SENTINEL_API_KEY
            )
            entity_key_metadata: Final[Mapping[str, KeyMetadataRow]] = (
                await repository.key_metadata(
                    entity_api_keys, spend_logs_window(frozenset(row.date for row in entity_records))
                )
                if entity_api_keys
                else MappingProxyType({})
            )
            await asyncio.to_thread(
                _fold_entity_rollups_sync,
                results=aggregated["results"],
                entity_rows=entity_records,
                api_key_metadata=entity_key_metadata,
                entity_metadata_field=entity_metadata_field,
            )
        return SpendAnalyticsPaginatedResponse(
            results=aggregated["results"],
            metadata=DailySpendMetadata(
                total_spend=aggregated["totals"].spend,
                total_flat_cost=aggregated["totals"].flat_cost,
                total_prompt_tokens=aggregated["totals"].prompt_tokens,
                total_completion_tokens=aggregated["totals"].completion_tokens,
                total_tokens=aggregated["totals"].total_tokens,
                total_api_requests=aggregated["totals"].api_requests,
                total_successful_requests=aggregated["totals"].successful_requests,
                total_failed_requests=aggregated["totals"].failed_requests,
                total_cache_read_input_tokens=aggregated["totals"].cache_read_input_tokens,
                total_cache_creation_input_tokens=aggregated["totals"].cache_creation_input_tokens,
                total_compression_saved_tokens=aggregated["totals"].compression_saved_tokens,
                total_compression_savings_spend=aggregated["totals"].compression_savings_spend,
                total_prompt_caching_savings_spend=aggregated["totals"].prompt_caching_savings_spend,
                total_gateway_injected_caching_savings_spend=aggregated[
                    "totals"
                ].gateway_injected_caching_savings_spend,
                total_autorouter_savings_spend=aggregated["totals"].autorouter_savings_spend,
                total_response_time_ms=aggregated["totals"].total_response_time_ms,
                total_timed_requests=aggregated["totals"].timed_requests,
                page=1,
                total_pages=1,
                has_more=False,
                api_key_limit=api_key_limit,
                total_api_keys=aggregated_rows.distinct_api_keys,
                entity_total_api_keys=entity_total_api_keys,
            ),
        )
    except Exception as exc:
        verbose_proxy_logger.exception("Error fetching aggregated daily activity: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": f"Failed to fetch analytics: {exc}"}
        )
