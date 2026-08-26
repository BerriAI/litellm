import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Protocol

from fastapi import HTTPException, status
from typing_extensions import TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.spend_tracking.ptu_feature_flag import is_ptu_cost_attribution_enabled
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import DeletedVerificationTokenRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
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

if TYPE_CHECKING:
    from prisma.models import (
        LiteLLM_DeletedVerificationToken as PrismaDeletedVerificationToken,
    )
    from prisma.models import (
        LiteLLM_VerificationToken as PrismaVerificationToken,
    )

# Mapping from Prisma accessor names to actual PostgreSQL table names.
_PRISMA_TO_PG_TABLE: Final[Mapping[str, str]] = {
    "litellm_dailyuserspend": "LiteLLM_DailyUserSpend",
    "litellm_dailyteamspend": "LiteLLM_DailyTeamSpend",
    "litellm_dailyorganizationspend": "LiteLLM_DailyOrganizationSpend",
    "litellm_dailyenduserspend": "LiteLLM_DailyEndUserSpend",
    "litellm_dailyagentspend": "LiteLLM_DailyAgentSpend",
    "litellm_dailytagspend": "LiteLLM_DailyTagSpend",
}


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
    def autorouter_savings_spend(self) -> float: ...

    @property
    def api_requests(self) -> int: ...

    @property
    def successful_requests(self) -> int: ...

    @property
    def failed_requests(self) -> int: ...


class _KeyMetadataDict(TypedDict, total=False):
    key_alias: str | None
    team_id: str | None


_WhereValue = str | dict[str, object]


class _AggregatedSpendData(TypedDict):
    results: list[DailySpendData]
    totals: SpendMetrics


class _GroupingSetsRow(SimpleNamespace):
    date: str
    api_key: str | None
    model: str | None
    model_group: str | None
    custom_llm_provider: str | None
    mcp_namespaced_tool_name: str | None
    endpoint: str | None
    group_level: int
    spend: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    compression_saved_tokens: int | None
    compression_savings_spend: float | None
    prompt_caching_savings_spend: float | None
    autorouter_savings_spend: float | None
    api_requests: int | None
    successful_requests: int | None
    failed_requests: int | None


class _EntityRollupRow(_GroupingSetsRow):
    entity_id: str | None
    api_key_rolled: int


def _reported_flat_cost(record: DailySpendRecord | _GroupingSetsRow) -> float:
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
    existing_metrics.autorouter_savings_spend += record.autorouter_savings_spend or 0
    existing_metrics.api_requests += record.api_requests or 0
    existing_metrics.successful_requests += record.successful_requests or 0
    existing_metrics.failed_requests += record.failed_requests or 0
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

        tag_value = getattr(record, "tag", None)
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
    return stored if stored is not None else {}  # mutable-ok: payload pydantic validates into its own dict


def update_breakdown_metrics(
    breakdown: BreakdownMetrics,
    record: DailySpendRecord,
    model_metadata: Mapping[str, dict[str, object]],
    provider_metadata: Mapping[str, dict[str, object]],
    api_key_metadata: Mapping[str, _KeyMetadataDict],
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
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                        team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                    ),
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
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                        team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                    ),
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
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                        team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                    ),
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
                metadata=KeyMetadata(
                    key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                    team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                ),
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
                metadata=KeyMetadata(
                    key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                    team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                ),
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
                metadata=KeyMetadata(
                    key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                    team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                ),  # Add any api_key-specific metadata here
            )
        breakdown.api_keys[record.api_key].metrics = update_metrics(breakdown.api_keys[record.api_key].metrics, record)

    # Update entity-specific metrics if entity_id_field is provided
    if entity_id_field:
        entity_value = getattr(record, entity_id_field, None)
        entity_value = entity_value if entity_value else "Unassigned"  # allow for null entity_id_field
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
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get("key_alias", None),
                        team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
                    ),
                )
            breakdown.entities[entity_value].api_key_breakdown[record.api_key].metrics = update_metrics(
                breakdown.entities[entity_value].api_key_breakdown[record.api_key].metrics,
                record,
            )

    return breakdown


async def get_api_key_metadata(
    prisma_client: PrismaClient,
    api_keys: AbstractSet[str],
) -> dict[str, _KeyMetadataDict]:
    """Get api key metadata, falling back to deleted keys table for keys not found in active table.

    This ensures that key_alias and team_id are preserved in historical activity logs
    even after a key is deleted or regenerated.
    """
    key_records: Sequence[PrismaVerificationToken] = await VerificationTokenRepository(prisma_client).table.find_many(
        where={"token": {"in": list(api_keys)}}
    )
    result: Final[dict[str, _KeyMetadataDict]] = {
        k.token: {"key_alias": k.key_alias, "team_id": k.team_id} for k in key_records
    }

    # For any keys not found in the active table, check the deleted keys table
    missing_keys: Final = api_keys - set(result.keys())
    if missing_keys:
        try:
            deleted_key_records: Final[
                Sequence[PrismaDeletedVerificationToken]
            ] = await DeletedVerificationTokenRepository(prisma_client).table.find_many(
                where={"token": {"in": list(missing_keys)}},
                order={"deleted_at": "desc"},
            )
            # Use the most recent deleted record for each token (ordered by deleted_at desc)
            for k in deleted_key_records:
                if k.token not in result:
                    result[k.token] = {
                        "key_alias": k.key_alias,
                        "team_id": k.team_id,
                    }
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to fetch deleted key metadata for %d missing keys: %s",
                len(missing_keys),
                e,
            )

    return result


def _adjust_dates_for_timezone(
    start_date: str,
    end_date: str,
    timezone_offset_minutes: int | None,
    include_current_utc_day: bool = False,
    utc_now: datetime | None = None,
) -> tuple[str, str]:
    """
    Map a caller-local date range onto UTC bucket keys, extending only the live end.

    The aggregation table (e.g. LiteLLM_DailyUserSpend) stores spend in whole-UTC-day
    buckets keyed on date as YYYY-MM-DD. Any conversion of an interior local-day
    boundary using only date arithmetic must round to whole UTC days, allowing up to
    24h of slop at each boundary. A previous implementation expanded the SQL range by
    an extra full UTC day on whichever side the offset pointed, which pulled in 24h of
    unrelated bucket data per boundary and produced approximately 100% over-counting on
    single-day queries (e.g. IST May 29 returning UTC May 28 + UTC May 29 in full).
    Sums of single-day queries then exceeded the equivalent multi-day aggregate, which
    is mathematically impossible. Historical dates therefore stay a pass-through: the
    local date is the UTC bucket key, trading boundary slop for monotonic, additive
    results. Hour-level buckets or pro-rata weighting would fix that properly; both
    require data the current schema does not store.

    The end boundary is different when the range reaches the caller's current day. A
    caller west of UTC asking for a range ending "today" is asking for data up to now,
    but once UTC has rolled past their local midnight, everything they sent since then
    sits in the next UTC bucket, which the pass-through excludes: a PT dashboard goes
    stale every evening from 5pm until local midnight, showing $0 for anything that
    only started accruing that evening. Extending such a range to today's UTC bucket
    cannot over-count, because the only part of that bucket outside the caller's range
    is the future, and the future is empty. ``timezone_offset_minutes`` follows the
    JS ``Date.getTimezoneOffset`` convention: UTC minus local, positive west of UTC.

    The extension is strictly opt-in via ``include_current_utc_day`` so a consumer
    whose axis or reconciliation expects the range to stop at the requested end date
    keeps today's byte-for-byte behaviour; the cost optimization dashboard opts in.
    """
    if not include_current_utc_day or timezone_offset_minutes is None:
        return start_date, end_date
    now: Final = utc_now if utc_now is not None else datetime.now(timezone.utc)
    caller_local_today: Final = (now - timedelta(minutes=timezone_offset_minutes)).date().isoformat()
    if end_date < caller_local_today:
        return start_date, end_date
    return start_date, max(end_date, now.date().isoformat())


def _build_where_conditions(
    *,
    entity_id_field: str,
    entity_id: str | list[str] | None,
    start_date: str,
    end_date: str,
    model: str | None,
    api_key: str | list[str] | None,
    exclude_entity_ids: list[str] | None = None,
    timezone_offset_minutes: int | None = None,
    include_current_utc_day: bool = False,
) -> dict[str, "_WhereValue"]:
    """Build prisma where clause for daily activity queries."""
    # Adjust dates for timezone if provided
    adjusted_start, adjusted_end = _adjust_dates_for_timezone(
        start_date, end_date, timezone_offset_minutes, include_current_utc_day
    )

    where_conditions: Final[dict[str, _WhereValue]] = {
        "date": {
            "gte": adjusted_start,
            "lte": adjusted_end,
        }
    }

    if model:
        where_conditions["model"] = model
    if api_key:
        if isinstance(api_key, list):
            where_conditions["api_key"] = {"in": api_key}
        else:
            where_conditions["api_key"] = api_key

    if entity_id is not None:
        if isinstance(entity_id, list):
            where_conditions[entity_id_field] = {"in": entity_id}
        else:
            where_conditions[entity_id_field] = {"equals": entity_id}

    if exclude_entity_ids:
        current: _WhereValue = where_conditions.get(entity_id_field, {})
        if isinstance(current, str):
            current = {"equals": current}
        current["not"] = {"in": exclude_entity_ids}
        where_conditions[entity_id_field] = current

    return where_conditions


def _build_aggregated_where_clause(
    *,
    entity_id_field: str,
    entity_id: str | list[str] | None,
    adjusted_start: str,
    adjusted_end: str,
    model: str | None,
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None,  # mutable-ok: filter union shared with the paginated path
) -> tuple[str, list[str]]:
    """Build the WHERE clause and $N params shared by the aggregated queries."""
    sql_conditions: Final[list[str]] = []
    sql_params: Final[list[str]] = []
    p = 1  # parameter index (1-based for PostgreSQL $N placeholders)

    # Date range (always present)
    sql_conditions.append(f"date >= ${p}")
    sql_params.append(adjusted_start)
    p += 1

    sql_conditions.append(f"date <= ${p}")
    sql_params.append(adjusted_end)
    p += 1

    # Optional entity filter; an empty list must match nothing, not everything
    if entity_id is not None:
        if isinstance(entity_id, list):
            if entity_id:
                placeholders = ", ".join(f"${p + i}" for i in range(len(entity_id)))
                sql_conditions.append(f'"{entity_id_field}" IN ({placeholders})')
                sql_params.extend(entity_id)
                p += len(entity_id)
            else:
                sql_conditions.append("FALSE")
        else:
            sql_conditions.append(f'"{entity_id_field}" = ${p}')
            sql_params.append(entity_id)
            p += 1

    # Exclude specific entities
    if exclude_entity_ids:
        placeholders = ", ".join(f"${p + i}" for i in range(len(exclude_entity_ids)))
        sql_conditions.append(f'"{entity_id_field}" NOT IN ({placeholders})')
        sql_params.extend(exclude_entity_ids)
        p += len(exclude_entity_ids)

    # Optional model filter
    if model:
        sql_conditions.append(f"model = ${p}")
        sql_params.append(model)
        p += 1

    # Optional api_key filter; an empty list must match nothing, not everything
    if isinstance(api_key, list):
        if api_key:
            placeholders = ", ".join(f"${p + i}" for i in range(len(api_key)))
            sql_conditions.append(f"api_key IN ({placeholders})")
            sql_params.extend(api_key)
            p += len(api_key)
        else:
            sql_conditions.append("FALSE")
    elif api_key:
        sql_conditions.append(f"api_key = ${p}")
        sql_params.append(api_key)
        p += 1

    return " AND ".join(sql_conditions), sql_params


def _ptu_flat_cost_select(table_name: str) -> str:
    """Only LiteLLM_DailyTeamSpend carries ptu_flat_cost; other daily tables emit a
    constant zero so the SpendMetrics.flat_cost response shape stays uniform."""
    if table_name == "litellm_dailyteamspend":
        return "SUM(ptu_flat_cost)::float AS ptu_flat_cost"
    return "0::float AS ptu_flat_cost"


def _build_aggregated_sql_query(
    *,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    start_date: str,
    end_date: str,
    model: str | None,
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None = None,  # mutable-ok: filter union shared with the paginated path
    timezone_offset_minutes: int | None = None,
    include_current_utc_day: bool = False,
) -> tuple[str, list[str]]:  # mutable-ok: SQL text plus its ordered $N params
    """Build a parameterized SQL GROUP BY query for aggregated daily activity.

    Groups by (date, api_key, model, model_group, custom_llm_provider,
    mcp_namespaced_tool_name, endpoint) with SUMs on all metric columns.
    The entity_id column is intentionally omitted from GROUP BY to collapse
    rows across entities — this is where the biggest row reduction comes from.

    Returns:
        Tuple of (sql_query, params_list) ready for prisma_client.db.query_raw().
    """
    pg_table: Final = _PRISMA_TO_PG_TABLE.get(table_name)
    if pg_table is None:
        raise ValueError(f"Unknown table name: {table_name}")

    adjusted_start, adjusted_end = _adjust_dates_for_timezone(
        start_date, end_date, timezone_offset_minutes, include_current_utc_day
    )

    where_clause, sql_params = _build_aggregated_where_clause(
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        adjusted_start=adjusted_start,
        adjusted_end=adjusted_end,
        model=model,
        api_key=api_key,
        exclude_entity_ids=exclude_entity_ids,
    )

    # Postgres computes every rollup level the response needs — per-date
    # totals, per-(date, model), per-(date, model, api_key), per-provider,
    # etc. — in a single pass via GROUPING SETS. The GROUPING() bitmask
    # encodes which level a row belongs to so Python can dispatch rows
    # straight into their buckets without re-summing. The leaf grouping
    # is omitted on purpose: nothing in the response shape needs it once
    # all the rollups are present.
    #
    # TODO: drop the successful_requests/failed_requests aggregates (and the
    # total_successful_requests metadata they feed) once the admin UI reads SGR
    # only from LiteLLM_DailyGatewayRequests. The remaining spend, token and
    # api_requests rollups are still served from here.
    sql_query: Final = f"""
        SELECT
            date,
            api_key,
            model,
            COALESCE(NULLIF(model_group, ''), model) AS model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            GROUPING(date, api_key, model, COALESCE(NULLIF(model_group, ''), model),
                     custom_llm_provider, mcp_namespaced_tool_name,
                     endpoint) AS group_level,
            SUM(spend)::float AS spend,
            {_ptu_flat_cost_select(table_name)},
            SUM(prompt_tokens)::bigint AS prompt_tokens,
            SUM(completion_tokens)::bigint AS completion_tokens,
            SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens,
            SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens,
            SUM(compression_saved_tokens)::bigint AS compression_saved_tokens,
            SUM(compression_savings_spend)::float AS compression_savings_spend,
            SUM(prompt_caching_savings_spend)::float AS prompt_caching_savings_spend,
            SUM(autorouter_savings_spend)::float AS autorouter_savings_spend,
            SUM(api_requests)::bigint AS api_requests,
            SUM(successful_requests)::bigint AS successful_requests,
            SUM(failed_requests)::bigint AS failed_requests
        FROM "{pg_table}"
        WHERE {where_clause}
        GROUP BY GROUPING SETS (
            (date),
            (date, api_key),
            (date, model),
            (date, model, api_key),
            (date, COALESCE(NULLIF(model_group, ''), model)),
            (date, COALESCE(NULLIF(model_group, ''), model), api_key),
            (date, custom_llm_provider),
            (date, custom_llm_provider, api_key),
            (date, mcp_namespaced_tool_name),
            (date, mcp_namespaced_tool_name, api_key),
            (date, endpoint),
            (date, endpoint, api_key),
            ()
        )
    """

    return sql_query, sql_params


def _build_entity_rollup_sql_query(
    *,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    start_date: str,
    end_date: str,
    model: str | None,
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None = None,  # mutable-ok: filter union shared with the paginated path
    timezone_offset_minutes: int | None = None,
    include_current_utc_day: bool = False,
) -> tuple[str, list[str]]:  # mutable-ok: SQL text plus its ordered $N params
    """Per-entity companion to _build_aggregated_sql_query.

    Two rollup levels over the same WHERE clause — (date, entity) and
    (date, entity, api_key) — told apart by GROUPING(api_key): 1 when the
    api_key column is rolled up, 0 when it is part of the key.
    """
    pg_table: Final = _PRISMA_TO_PG_TABLE.get(table_name)
    if pg_table is None:
        raise ValueError(f"Unknown table name: {table_name}")

    adjusted_start, adjusted_end = _adjust_dates_for_timezone(
        start_date, end_date, timezone_offset_minutes, include_current_utc_day
    )

    where_clause, sql_params = _build_aggregated_where_clause(
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        adjusted_start=adjusted_start,
        adjusted_end=adjusted_end,
        model=model,
        api_key=api_key,
        exclude_entity_ids=exclude_entity_ids,
    )

    sql_query: Final = f"""
        SELECT
            "{entity_id_field}" AS entity_id,
            date,
            api_key,
            GROUPING(api_key) AS api_key_rolled,
            SUM(spend)::float AS spend,
            {_ptu_flat_cost_select(table_name)},
            SUM(prompt_tokens)::bigint AS prompt_tokens,
            SUM(completion_tokens)::bigint AS completion_tokens,
            SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens,
            SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens,
            SUM(compression_saved_tokens)::bigint AS compression_saved_tokens,
            SUM(compression_savings_spend)::float AS compression_savings_spend,
            SUM(prompt_caching_savings_spend)::float AS prompt_caching_savings_spend,
            SUM(autorouter_savings_spend)::float AS autorouter_savings_spend,
            SUM(api_requests)::bigint AS api_requests,
            SUM(successful_requests)::bigint AS successful_requests,
            SUM(failed_requests)::bigint AS failed_requests
        FROM "{pg_table}"
        WHERE {where_clause}
        GROUP BY GROUPING SETS (
            (date, "{entity_id_field}"),
            (date, "{entity_id_field}", api_key)
        )
    """

    return sql_query, sql_params


def _aggregate_spend_records_sync(
    *,
    records: Sequence[DailySpendRecord],
    api_key_metadata: Mapping[str, _KeyMetadataDict],
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
    prisma_client: PrismaClient,
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

    api_key_metadata: dict[str, _KeyMetadataDict] = {}
    if api_keys:
        api_key_metadata = await get_api_key_metadata(prisma_client, api_keys)

    return await asyncio.to_thread(
        _aggregate_spend_records_sync,
        records=records,
        api_key_metadata=api_key_metadata,
        entity_id_field=entity_id_field,
        entity_metadata_field=entity_metadata_field,
    )


# GROUPING() bitmask values for each grouping set emitted by
# _build_aggregated_sql_query. Per Postgres semantics, the rightmost argument
# is the least-significant bit. Argument order:
#   date, api_key, model, model_group, custom_llm_provider,
#   mcp_namespaced_tool_name, endpoint
# A bit is 1 when the corresponding column is rolled up (i.e. NOT in the
# current grouping set's key), 0 when the column is part of the key.
_GROUP_GRAND_TOTAL: Final = 127  # 0b1111111 — all rolled up
_GROUP_DATE: Final = 63  # 0b0111111 — only date kept
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


def _record_to_spend_metrics(record: _GroupingSetsRow) -> SpendMetrics:
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
        autorouter_savings_spend=record.autorouter_savings_spend or 0,
        api_requests=record.api_requests or 0,
        successful_requests=record.successful_requests or 0,
        failed_requests=record.failed_requests or 0,
    )


def _key_metadata(api_key_metadata: Mapping[str, _KeyMetadataDict], api_key: str) -> KeyMetadata:
    meta: Final = api_key_metadata.get(api_key, {})
    return KeyMetadata(key_alias=meta.get("key_alias"), team_id=meta.get("team_id"))


def _aggregate_grouping_sets_records_sync(
    *,
    records: Sequence[_GroupingSetsRow],
    api_key_metadata: Mapping[str, _KeyMetadataDict],
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
            provider_metrics = metrics.model_copy(update={"flat_cost": 0.0})  # mutable-ok: pydantic update payload
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
    prisma_client: PrismaClient,
    records: Sequence[_GroupingSetsRow],
) -> _AggregatedSpendData:
    """Async wrapper: fetch api_key_metadata, then dispatch on a worker thread."""
    api_keys: Final[set[str]] = {r.api_key for r in records if r.api_key and r.api_key != PTU_SENTINEL_API_KEY}

    api_key_metadata: dict[str, _KeyMetadataDict] = {}
    if api_keys:
        api_key_metadata = await get_api_key_metadata(prisma_client, api_keys)

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
    """Common function to get daily activity for any entity type.

    ``resolve_entity_metadata`` lets a caller resolve entity metadata from the
    rows actually on the page (e.g. user_id -> user_email) instead of fetching
    the whole entity table upfront, which matters when the entity set is
    unbounded.
    """

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    try:
        where_conditions: Final = _build_where_conditions(
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            exclude_entity_ids=exclude_entity_ids,
            timezone_offset_minutes=timezone_offset_minutes,
            include_current_utc_day=include_current_utc_day,
        )

        # Get total count for pagination
        total_count: Final[int] = await getattr(prisma_client.db, table_name).count(where=where_conditions)

        # Fetch paginated results.
        # ``date`` alone is not a unique sort key -- a busy tenant has many
        # rows per date (one per api_key, model, model_group, provider,
        # endpoint, ...), so offset pagination over ``date desc`` lands on
        # arbitrary boundaries and the same row can be skipped on one page
        # and returned on another. A client that pages through and sums the
        # per-page metrics (the Usage dashboard) then gets a non-deterministic
        # total. Adding ``id`` (the row's UUID primary key, present on both
        # LiteLLM_DailyUserSpend and LiteLLM_DailyTeamSpend) as a tiebreaker
        # gives every page a stable cursor (#30164).
        daily_spend_data: Final[Sequence[DailySpendRecord]] = await getattr(prisma_client.db, table_name).find_many(
            where=where_conditions,
            order=[
                {"date": "desc"},
                {"id": "asc"},
            ],
            skip=(page - 1) * page_size,
            take=page_size,
        )

        resolved_entity_metadata = entity_metadata_field
        if resolve_entity_metadata is not None:
            resolved_entity_metadata = {
                **(entity_metadata_field or {}),
                **(await resolve_entity_metadata(daily_spend_data)),
            }

        aggregated: Final = await _aggregate_spend_records(
            prisma_client=prisma_client,
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
                total_autorouter_savings_spend=metadata_metrics.autorouter_savings_spend,
                page=page,
                total_pages=-(-total_count // page_size),  # Ceiling division
                has_more=(page * page_size) < total_count,
            ),
        )

    except Exception as e:
        verbose_proxy_logger.exception("Error fetching daily activity: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {e}"},
        )


def _fold_entity_rollups_sync(
    *,
    results: Sequence[DailySpendData],
    entity_rows: Sequence[_EntityRollupRow],
    api_key_metadata: Mapping[str, _KeyMetadataDict],
    entity_metadata_field: Mapping[str, dict[str, object]] | None,  # mutable-ok: shared field shape
) -> None:
    """Write breakdown.entities onto the already-built per-day results."""
    by_date: Final = {day.date.strftime("%Y-%m-%d"): day for day in results}  # mutable-ok: local fold index

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
    prisma_client: PrismaClient | None,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,
    entity_metadata_field: Mapping[str, dict[str, object]] | None,
    start_date: str | None,
    end_date: str | None,
    model: str | None,
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None = None,
    timezone_offset_minutes: int | None = None,
    include_entity_breakdown: bool = False,
    include_current_utc_day: bool = False,
) -> SpendAnalyticsPaginatedResponse:
    """Aggregated variant that returns the full result set (no pagination).

    Uses SQL GROUP BY to aggregate rows in the database rather than fetching
    all individual rows into Python. This collapses rows across entities
    (users/teams/orgs), reducing ~150k rows to ~2-3k grouped rows.

    include_entity_breakdown runs a small companion rollup query and folds
    `breakdown.entities` onto the response, as entity-scoped views like Team Usage need.

    Matches the response model of the paginated endpoint so the UI does not need to transform.
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    try:
        sql_query, sql_params = _build_aggregated_sql_query(
            table_name=table_name,
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            exclude_entity_ids=exclude_entity_ids,
            timezone_offset_minutes=timezone_offset_minutes,
            include_current_utc_day=include_current_utc_day,
        )

        entity_query: Final = (
            _build_entity_rollup_sql_query(
                table_name=table_name,
                entity_id_field=entity_id_field,
                entity_id=entity_id,
                start_date=start_date,
                end_date=end_date,
                model=model,
                api_key=api_key,
                exclude_entity_ids=exclude_entity_ids,
                timezone_offset_minutes=timezone_offset_minutes,
                include_current_utc_day=include_current_utc_day,
            )
            if include_entity_breakdown
            else None
        )

        # Execute the GROUPING SETS query (one row per rollup level), alongside
        # the per-entity companion rollup when the caller wants entities.
        raw_rows, raw_entity_rows = (
            await asyncio.gather(
                prisma_client.db.query_raw(sql_query, *sql_params),
                prisma_client.db.query_raw(entity_query[0], *entity_query[1]),
            )
            if entity_query is not None
            else (await prisma_client.db.query_raw(sql_query, *sql_params), None)
        )

        records: Final = [_GroupingSetsRow(**row) for row in (raw_rows or [])]

        # The grouping-sets dispatcher places each row directly in its bucket
        # using the row's GROUPING() bitmask. No Python-side summing needed.
        aggregated: Final = await _aggregate_grouping_sets_records(
            prisma_client=prisma_client,
            records=records,
        )

        if raw_entity_rows:
            entity_records: Final = tuple(_EntityRollupRow(**row) for row in raw_entity_rows)
            entity_api_keys: Final = frozenset(
                r.api_key for r in entity_records if r.api_key and r.api_key != PTU_SENTINEL_API_KEY
            )
            entity_key_metadata: Final = (
                await get_api_key_metadata(prisma_client, entity_api_keys)
                if entity_api_keys
                else {}  # mutable-ok: matches the helper's dict return
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
                total_autorouter_savings_spend=aggregated["totals"].autorouter_savings_spend,
                page=1,
                total_pages=1,
                has_more=False,
            ),
        )

    except Exception as e:
        verbose_proxy_logger.exception("Error fetching aggregated daily activity: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {e}"},
        )
