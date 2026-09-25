"""Shared route-level helpers for the /<entity>/daily/activity surface.

The aggregated, search and export routes on team/tag/organization/customer/agent
daily activity differ only in scoping and labels; the pieces that must behave
identically live here once.
"""

import csv
import dataclasses
import io
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final

from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm.constants import USAGE_TOP_API_KEYS_LIMIT
from litellm.proxy.management_endpoints.common_daily_activity import (
    PRISMA_TO_PG_TABLE,
    adjust_dates_for_timezone,
    build_aggregated_where_clause,
    get_daily_activity_export_rows,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityExportFormat,
    DailyActivityExportMetadata,
    DailyActivityExportResponse,
    DailyActivityExportRow,
    DailyActivityExportType,
)
from litellm.types.proxy.management_endpoints.team_endpoints import TeamDailyActivityExportRow

_MAX_AGGREGATED_RANGE_DAYS: Final = 400


def daily_activity_error(*, status_code: int, message: str) -> HTTPException:
    """Single construction site for the `{"error": ...}` detail shape the
    /team/daily/activity endpoints have always returned."""
    return HTTPException(status_code=status_code, detail={"error": message})  # mutable-ok: FastAPI JSON detail


def aggregated_date_range_error(start_date: str | None, end_date: str | None) -> str | None:
    """The aggregated endpoint has no pagination to bound its work, so malformed
    dates and ranges wider than the UI ever requests are rejected before querying."""
    if start_date is None or end_date is None:
        return "Please provide start_date and end_date"
    try:
        parsed_start: Final = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        parsed_end: Final = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return "start_date and end_date must be valid YYYY-MM-DD dates"
    if parsed_end < parsed_start:
        return "end_date must be on or after start_date"
    if (parsed_end - parsed_start).days > _MAX_AGGREGATED_RANGE_DAYS:
        return f"Date range must be at most {_MAX_AGGREGATED_RANGE_DAYS} days"
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class DailyActivityExportLabels:
    alias_header: str | None
    id_header: str


_EXPORT_CSV_METRIC_HEADERS: Final = (
    "Spend ($)",
    "Requests",
    "Successful Requests",
    "Failed Requests",
    "Total Tokens",
    "Prompt Tokens",
    "Completion Tokens",
    "Cache Read Input Tokens",
    "Cache Creation Input Tokens",
)


def _export_csv_headers(export_type: DailyActivityExportType, labels: DailyActivityExportLabels) -> tuple[str, ...]:
    base: Final = (
        ("Date", labels.alias_header, labels.id_header)
        if labels.alias_header is not None
        else ("Date", labels.id_header)
    )
    if export_type == "daily_with_keys":
        return (*base, "Key Alias", "Key ID", "User ID", "User Email", *_EXPORT_CSV_METRIC_HEADERS)
    if export_type == "daily_with_users":
        return (*base, "User ID", "User Email", "Keys", *_EXPORT_CSV_METRIC_HEADERS)
    if export_type == "daily_with_models":
        return (
            *base,
            "Model",
            "Spend ($)",
            "Requests",
            "Successful",
            "Failed",
            "Total Tokens",
            "Prompt Tokens",
            "Completion Tokens",
            "Cache Read Input Tokens",
            "Cache Creation Input Tokens",
        )
    return (*base, *_EXPORT_CSV_METRIC_HEADERS)


def _csv_safe(value: str) -> str:
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def _export_csv_record(row: DailyActivityExportRow, labels: DailyActivityExportLabels) -> dict[str, object]:
    record: Final[dict[str, object]] = {  # mutable-ok: csv.DictWriter consumes a plain mapping per row
        "Date": row.date,
        labels.id_header: _csv_safe(row.entity_id),
        "Key Alias": _csv_safe(row.key_alias) if row.key_alias else "-",
        "Key ID": row.api_key or "-",
        "User ID": _csv_safe(row.user_id) if row.user_id else "-",
        "User Email": _csv_safe(row.user_email) if row.user_email else "-",
        "Keys": row.keys,
        "Model": _csv_safe(row.model) if row.model else "-",
        "Spend ($)": f"{row.spend:.4f}",
        "Flat Cost ($)": f"{row.flat_cost:.4f}",
        "Total Cost ($)": f"{row.spend + row.flat_cost:.4f}",
        "Requests": row.api_requests,
        "Successful Requests": row.successful_requests,
        "Failed Requests": row.failed_requests,
        "Successful": row.successful_requests,
        "Failed": row.failed_requests,
        "Total Tokens": row.total_tokens,
        "Prompt Tokens": row.prompt_tokens,
        "Completion Tokens": row.completion_tokens,
        "Cache Read Input Tokens": row.cache_read_input_tokens,
        "Cache Creation Input Tokens": row.cache_creation_input_tokens,
    }
    if labels.alias_header is not None:
        record[labels.alias_header] = _csv_safe(row.entity_alias) if row.entity_alias else "-"
    return record


def _export_csv(
    export_type: DailyActivityExportType,
    rows: Sequence[DailyActivityExportRow],
    labels: DailyActivityExportLabels,
) -> str:
    base_headers: Final = _export_csv_headers(export_type, labels)
    spend_index: Final = base_headers.index("Spend ($)") + 1
    headers: Final = (
        (*base_headers[:spend_index], "Flat Cost ($)", "Total Cost ($)", *base_headers[spend_index:])
        if sum(row.flat_cost for row in rows) > 0
        else base_headers
    )
    buffer: Final = io.StringIO()
    writer: Final = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_export_csv_record(row, labels) for row in rows)
    return buffer.getvalue()


_TEAM_EXPORT_LABELS: Final = DailyActivityExportLabels(alias_header="Team", id_header="Team ID")


def _team_export_csv(
    export_type: DailyActivityExportType,
    rows: Sequence[DailyActivityExportRow | TeamDailyActivityExportRow],
) -> str:
    return _export_csv(export_type, tuple(_to_export_row(row) for row in rows), _TEAM_EXPORT_LABELS)


def _to_export_row(row: DailyActivityExportRow | TeamDailyActivityExportRow) -> DailyActivityExportRow:
    if isinstance(row, DailyActivityExportRow):
        return row
    return DailyActivityExportRow(
        entity_id=row.team_id,
        entity_alias=row.team_alias,
        **row.model_dump(exclude={"team_id", "team_alias"}),
    )


def _team_export_row(row: DailyActivityExportRow) -> TeamDailyActivityExportRow:
    return TeamDailyActivityExportRow(
        team_id=row.entity_id,
        team_alias=row.entity_alias,
        **row.model_dump(exclude={"entity_id", "entity_alias"}),
    )


class _KeySearchTokenRow(TypedDict):
    token: str


_KEY_SEARCH_TOKEN_ADAPTER: Final = TypeAdapter(list[_KeySearchTokenRow])


def build_daily_activity_key_search_sql(
    *,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None,  # mutable-ok: filter union shared with the paginated path
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    start_date: str,
    end_date: str,
    timezone_offset_minutes: int | None,
    search: str,
) -> tuple[str, list[str]]:
    """Build the key-search query: top-spend matching tokens that have spend rows inside the
    caller's entity scope, so the LIMIT cannot evict an in-scope match for a foreign one."""
    pg_table: Final = PRISMA_TO_PG_TABLE.get(table_name)
    if pg_table is None:
        raise ValueError(f"Unknown table name: {table_name}")

    adjusted_start, adjusted_end = adjust_dates_for_timezone(start_date, end_date, timezone_offset_minutes)
    where_clause, where_params = build_aggregated_where_clause(
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        adjusted_start=adjusted_start,
        adjusted_end=adjusted_end,
        model=None,
        api_key=api_key,
        exclude_entity_ids=exclude_entity_ids,
    )
    token_param: Final = len(where_params) + 1
    like_param: Final = token_param + 1
    escaped_search: Final = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    sql_query: Final = f"""
        SELECT vt.token
        FROM "LiteLLM_VerificationToken" vt
        WHERE EXISTS (
            SELECT 1 FROM "{pg_table}" s
            WHERE {where_clause} AND s.api_key = vt.token
        )
        AND (vt.token = ${token_param} OR vt.key_alias ILIKE ${like_param} OR vt.user_id ILIKE ${like_param})
        ORDER BY vt.spend DESC NULLS LAST, vt.token
        LIMIT {USAGE_TOP_API_KEYS_LIMIT}
    """
    return sql_query, [*where_params, search, f"%{escaped_search}%"]


async def search_daily_activity_key_tokens(
    *,
    prisma_client: PrismaClient,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None,  # mutable-ok: filter union shared with the paginated path
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    start_date: str | None,
    end_date: str | None,
    timezone_offset_minutes: int | None,
    search: str,
) -> tuple[str, ...]:
    """Token hashes matching `search` restricted to keys with spend rows inside the caller's
    entity and api_key scope, so the spend-ordered LIMIT never trims visible matches."""
    if start_date is None or end_date is None or entity_id == [] or api_key == []:
        return ()
    sql_query, sql_params = build_daily_activity_key_search_sql(
        table_name=table_name,
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        exclude_entity_ids=exclude_entity_ids,
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        timezone_offset_minutes=timezone_offset_minutes,
        search=search,
    )
    raw_rows: Final = await prisma_client.db.query_raw(sql_query, *sql_params)
    return tuple(row["token"] for row in _KEY_SEARCH_TOKEN_ADAPTER.validate_python(raw_rows))


async def build_daily_activity_export_response(
    *,
    prisma_client: PrismaClient,
    table_name: str,
    entity_id_field: str,
    entity_id: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    entity_metadata_field: Mapping[str, dict[str, object]] | None,
    alias_metadata_key: str | None,
    api_key: str | list[str] | None,  # mutable-ok: filter union shared with the paginated path
    exclude_entity_ids: list[str] | None,  # mutable-ok: filter union shared with the paginated path
    start_date: str,
    end_date: str,
    timezone_offset_minutes: int | None,
    export_type: DailyActivityExportType,
    format: DailyActivityExportFormat,
    labels: DailyActivityExportLabels,
    filename_prefix: str,
) -> Response:
    rows: Final = await get_daily_activity_export_rows(
        prisma_client=prisma_client,
        table_name=table_name,
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        entity_metadata_field=entity_metadata_field,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        exclude_entity_ids=exclude_entity_ids,
        timezone_offset_minutes=timezone_offset_minutes,
        export_type=export_type,
        alias_metadata_key=alias_metadata_key,
    )

    now: Final = datetime.now(timezone.utc)
    metadata: Final = DailyActivityExportMetadata(
        export_date=now.isoformat(),
        export_type=export_type,
        start_date=start_date,
        end_date=end_date,
        entity_ids=list(entity_id) if entity_id else None,  # mutable-ok: response model field type
        total_spend=sum(row.spend for row in rows),
        total_flat_cost=sum(row.flat_cost for row in rows),
        total_api_requests=sum(row.api_requests for row in rows),
        total_successful_requests=sum(row.successful_requests for row in rows),
        total_failed_requests=sum(row.failed_requests for row in rows),
        total_tokens=sum(row.total_tokens for row in rows),
    )

    if format == "json":
        return JSONResponse(content=DailyActivityExportResponse(metadata=metadata, data=rows).model_dump(mode="json"))
    return Response(
        content=_export_csv(export_type, rows, labels),
        media_type="text/csv; charset=utf-8",
        headers={  # mutable-ok: starlette Response headers is a dict
            "Content-Disposition": (
                f'attachment; filename="{filename_prefix}_usage_{export_type}_{now.date().isoformat()}.csv"'
            )
        },
    )
