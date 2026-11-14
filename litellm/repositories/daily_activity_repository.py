import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, StrictStr, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm import constants
from litellm._logging import verbose_proxy_logger
from litellm.constants import UNKNOWN_MODEL_SPEND_LOG_MODEL
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker, redact_credentials_in_payload
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.daily_activity_sql import (
    EXPORT_METADATA_DROPPED_KEYS,
    ExportCursor,
    SqlQuery,
    adjust_dates_for_timezone,
    build_aggregated_sql,
    build_cache_leakage_keys_sql,
    build_customer_daily_rows_sql,
    build_customer_team_rollup_sql,
    build_entity_rollup_sql,
    build_export_sql,
    build_guardrail_spend_sql,
    build_key_page_sql,
    build_key_search_sql,
    build_model_top_keys_sql,
    build_user_page_sql,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.repositories.daily_activity import (
    AggregatedRows,
    CustomerTeamRollupRow,
    DailyActivityProxyReads,
    DailyActivityRow,
    DailyActivityScope,
    DailyActivityTable,
    DailyRowsPage,
    EntityRollupRow,
    ExportRow,
    ExportType,
    GroupingSetsRow,
    GuardrailSpendRow,
    KeyMetadataRow,
    KeyPage,
    KeySpendRow,
    SpendLogsWindow,
    UserMetadataRow,
    UserPage,
    UserSpendRow,
)


class _UserTableRow(Protocol):
    user_id: str
    user_email: str | None
    user_alias: str | None


class _VerificationTokenRow(Protocol):
    token: str
    key_alias: str | None
    team_id: str | None
    user_id: str | None
    metadata: Mapping[str, object] | None


class _DeletedVerificationTokenRow(_VerificationTokenRow, Protocol):
    deleted_at: datetime


class _QueryRaw(Protocol):
    async def __call__(self, query: str, *values: object) -> Sequence[Mapping[str, object]] | None: ...


class _DailyActivityDatabase(Protocol):
    query_raw: _QueryRaw
    litellm_usertable: TableActions[_UserTableRow]
    litellm_verificationtoken: TableActions[_VerificationTokenRow]
    litellm_deletedverificationtoken: TableActions[_DeletedVerificationTokenRow]

    @property
    def litellm_dailyuserspend(self) -> TableActions[DailyActivityRow]: ...

    @property
    def litellm_dailyteamspend(self) -> TableActions[DailyActivityRow]: ...

    @property
    def litellm_dailytagspend(self) -> TableActions[DailyActivityRow]: ...

    @property
    def litellm_dailyorganizationspend(self) -> TableActions[DailyActivityRow]: ...

    @property
    def litellm_dailyenduserspend(self) -> TableActions[DailyActivityRow]: ...

    @property
    def litellm_dailycustomermemberspend(self) -> TableActions[DailyActivityRow]: ...

    @property
    def litellm_dailyagentspend(self) -> TableActions[DailyActivityRow]: ...


class DailyActivityDatabase(Protocol):
    @property
    def db(self) -> _DailyActivityDatabase: ...


_GROUPING_ADAPTER: Final = TypeAdapter(tuple[GroupingSetsRow, ...])
_CUSTOMER_TEAM_ADAPTER: Final = TypeAdapter(tuple[CustomerTeamRollupRow, ...])
_ENTITY_ADAPTER: Final = TypeAdapter(tuple[EntityRollupRow, ...])
_GUARDRAIL_SPEND_ADAPTER: Final = TypeAdapter(tuple[GuardrailSpendRow, ...])
_KEY_SPEND_ADAPTER: Final = TypeAdapter(tuple[KeySpendRow, ...])
_KEY_PAGE_TOTAL_ADAPTER: Final[TypeAdapter[int]] = TypeAdapter(int)
_USER_SPEND_ADAPTER: Final = TypeAdapter(tuple[UserSpendRow, ...])
_EXPORT_ADAPTER: Final = TypeAdapter(tuple[ExportRow, ...])
_METADATA_TAGS_ADAPTER: Final = TypeAdapter(list[StrictStr])


class _CustomerDailyRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    date: str
    api_key: str
    end_user_id: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    compression_saved_tokens: int
    compression_savings_spend: float
    prompt_caching_savings_spend: float
    gateway_injected_caching_savings_spend: float
    autorouter_savings_spend: float
    guardrail_spend: float
    spend: float
    api_requests: int
    successful_requests: int
    failed_requests: int
    total_response_time_ms: int
    timed_requests: int
    model: str | None = None
    model_group: str | None = None
    custom_llm_provider: str | None = None
    mcp_namespaced_tool_name: str | None = None
    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class _CustomerDailyRowsResult:
    total_count: int
    rows: tuple[_CustomerDailyRow, ...]


_CUSTOMER_DAILY_ROWS_ADAPTER: Final = TypeAdapter(tuple[_CustomerDailyRowsResult])


def _metadata_tags(value: object) -> tuple[str, ...]:
    stable_value: Final = value
    if not isinstance(value, list):
        return ()
    try:
        return tuple(_METADATA_TAGS_ADAPTER.validate_python(stable_value))
    except ValidationError:
        return ()


def _daily_rows_table(
    prisma_client: DailyActivityDatabase, table: DailyActivityTable
) -> TableActions[DailyActivityRow]:
    if table is DailyActivityTable.USER:
        return prisma_client.db.litellm_dailyuserspend
    if table is DailyActivityTable.TEAM:
        return prisma_client.db.litellm_dailyteamspend
    if table is DailyActivityTable.TAG:
        return prisma_client.db.litellm_dailytagspend
    if table is DailyActivityTable.ORGANIZATION:
        return prisma_client.db.litellm_dailyorganizationspend
    if table is DailyActivityTable.CUSTOMER:
        return prisma_client.db.litellm_dailyenduserspend
    if table is DailyActivityTable.CUSTOMER_MEMBER:
        return prisma_client.db.litellm_dailycustomermemberspend
    if table is DailyActivityTable.AGENT:
        return prisma_client.db.litellm_dailyagentspend
    assert_never(table)
    raise AssertionError("unreachable")


_EXPORT_UNSEPARATED_CREDENTIAL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "apikey",
        "apitoken",
        "accesskey",
        "accesstoken",
        "authkey",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "passwd",
        "privatekey",
        "pwd",
        "refreshtoken",
        "secretkey",
        "sessiontoken",
        "signingkey",
        "webhooksecret",
    }
)
_EXPORT_METADATA_MASKER: Final = SensitiveDataMasker(
    non_sensitive_overrides=frozenset(),
    extra_sensitive_patterns=_EXPORT_UNSEPARATED_CREDENTIAL_NAMES,
    split_camel_case=True,
)


def _redacted_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return redact_credentials_in_payload(metadata, _EXPORT_METADATA_MASKER, dropped_keys=EXPORT_METADATA_DROPPED_KEYS)


def _with_redacted_metadata(row: ExportRow) -> ExportRow:
    if not row.key_metadata and not row.user_metadata:
        return row
    return replace(
        row, key_metadata=_redacted_metadata(row.key_metadata), user_metadata=_redacted_metadata(row.user_metadata)
    )


def _next_export_cursor(batch: tuple[ExportRow, ...], export_type: ExportType) -> ExportCursor:
    last: Final = batch[-1]
    cursor_key: Final = (
        last.api_key
        if export_type is ExportType.DAILY_WITH_KEYS
        else last.model
        if export_type is ExportType.DAILY_WITH_MODELS
        else last.user_id
        if export_type is ExportType.DAILY_WITH_USERS
        else ""
    )
    return ExportCursor(date=last.date, entity_id=last.entity_id, group_key=cursor_key or "")


class DailyActivityRepository:
    def __init__(self, prisma_client: DailyActivityDatabase, *, proxy_reads: DailyActivityProxyReads) -> None:
        self._prisma_client = prisma_client
        self._proxy_reads = proxy_reads

    async def _query(self, query: SqlQuery) -> tuple[Mapping[str, object], ...]:
        from litellm.proxy.db.db_span import db_span
        from litellm.proxy.db.prisma_query_span import sql_relation

        first_line: Final = query.sql.lstrip().splitlines()[0].lstrip("(").strip()
        verbose_proxy_logger.debug("DailyActivityRepository query: %s", first_line)
        async with db_span("daily_activity_query", sql_relation(query.sql)):
            result: Sequence[Mapping[str, object]] | None = await self._prisma_client.db.query_raw(
                query.sql, *query.params
            )
        if result is None:
            return ()
        return tuple(result)

    async def aggregated(
        self, scope: DailyActivityScope, *, include_entity_breakdown: bool, api_key_limit: int
    ) -> AggregatedRows:
        grouping_query: Final = build_aggregated_sql(scope, api_key_limit=api_key_limit)
        entity_query: Final = (
            build_entity_rollup_sql(scope, api_key_limit=api_key_limit) if include_entity_breakdown else None
        )
        grouping_result, entity_result = await asyncio.gather(
            self._query(grouping_query),
            self._query(entity_query) if entity_query is not None else asyncio.sleep(0, result=None),
        )
        grouping_rows: Final = _GROUPING_ADAPTER.validate_python(grouping_result)
        entity_rows: Final = None if entity_result is None else _ENTITY_ADAPTER.validate_python(entity_result)
        distinct_api_keys: Final = next(
            (row.distinct_api_keys for row in grouping_rows if row.distinct_api_keys is not None), 0
        )
        return AggregatedRows(
            grouping_rows=grouping_rows,
            entity_rows=entity_rows,
            distinct_api_keys=distinct_api_keys,
        )

    async def guardrail_spend(self, scope: DailyActivityScope) -> tuple[GuardrailSpendRow, ...]:
        return _GUARDRAIL_SPEND_ADAPTER.validate_python(await self._query(build_guardrail_spend_sql(scope)))

    async def search_keys(self, scope: DailyActivityScope, *, search: str, limit: int) -> tuple[str, ...]:
        if not 1 <= limit <= constants.USAGE_KEY_SEARCH_MAX:
            raise ValueError(f"limit must be between 1 and {constants.USAGE_KEY_SEARCH_MAX}")
        query: Final = build_key_search_sql(scope, search=search, limit=limit)
        rows: Final = _KEY_SPEND_ADAPTER.validate_python(await self._query(query))
        return tuple(row.api_key for row in rows)

    async def key_page(self, scope: DailyActivityScope, *, offset: int, limit: int) -> KeyPage:
        query: Final = build_key_page_sql(scope, offset=offset, limit=limit)
        result: Final = await self._query(query)
        total_api_keys_value: Final = result[0].get("total_api_keys") if result else 0
        total_api_keys: Final = (
            _KEY_PAGE_TOTAL_ADAPTER.validate_python(total_api_keys_value) if total_api_keys_value is not None else 0
        )
        rows: Final = _KEY_SPEND_ADAPTER.validate_python(tuple(row for row in result if row.get("api_key") is not None))
        return KeyPage(rows=rows, total_api_keys=total_api_keys)

    async def user_page(self, scope: DailyActivityScope, *, offset: int, limit: int) -> UserPage:
        query: Final = build_user_page_sql(scope, offset=offset, limit=limit)
        result: Final = await self._query(query)
        total_users_value: Final = result[0].get("total_users") if result else 0
        total_users: Final = (
            _KEY_PAGE_TOTAL_ADAPTER.validate_python(total_users_value) if total_users_value is not None else 0
        )
        rows: Final = _USER_SPEND_ADAPTER.validate_python(
            tuple(
                {**row, "user_id": None} if row.get("user_id") in (None, "") else row
                for row in result
                if row.get("spend") is not None
            )
        )
        return UserPage(rows=rows, total_users=total_users)

    async def user_metadata(self, user_ids: frozenset[str]) -> Mapping[str, UserMetadataRow]:
        if not user_ids:
            return {}
        rows: Final = await find_many_in(self._prisma_client.db.litellm_usertable, "user_id", tuple(user_ids))
        return MappingProxyType(
            {row.user_id: UserMetadataRow(user_email=row.user_email, user_alias=row.user_alias) for row in rows}
        )

    async def model_top_keys(
        self, scope: DailyActivityScope, *, model_group: str, by_model_group: bool, limit: int
    ) -> tuple[KeySpendRow, ...]:
        if not 1 <= limit <= constants.USAGE_MODEL_TOP_KEYS_MAX:
            raise ValueError(f"limit must be between 1 and {constants.USAGE_MODEL_TOP_KEYS_MAX}")
        query: Final = build_model_top_keys_sql(
            scope,
            model_group=model_group,
            by_model_group=by_model_group,
            limit=limit,
        )
        return _KEY_SPEND_ADAPTER.validate_python(await self._query(query))

    async def cache_leakage_keys(self, scope: DailyActivityScope, *, limit: int) -> tuple[KeySpendRow, ...]:
        if not 1 <= limit <= constants.USAGE_CACHE_LEAKAGE_KEYS_MAX:
            raise ValueError(f"limit must be between 1 and {constants.USAGE_CACHE_LEAKAGE_KEYS_MAX}")
        query: Final = build_cache_leakage_keys_sql(scope, limit=limit)
        return _KEY_SPEND_ADAPTER.validate_python(await self._query(query))

    async def export_rows(
        self, scope: DailyActivityScope, *, export_type: ExportType, include_metadata: bool = False
    ) -> AsyncIterator[ExportRow]:
        batch_size: Final = constants.USAGE_EXPORT_BATCH_SIZE
        cursor: ExportCursor | None = None  # rebind-ok: each page advances the export keyset cursor
        while True:
            batch: tuple[ExportRow, ...] = _EXPORT_ADAPTER.validate_python(
                await self._query(
                    build_export_sql(
                        scope,
                        export_type=export_type,
                        after=cursor,
                        batch_size=batch_size,
                        include_metadata=include_metadata,
                    )
                )
            )
            for row in batch:
                yield _with_redacted_metadata(row)
            if len(batch) < batch_size:
                return
            cursor = _next_export_cursor(batch, export_type)

    async def _active_token_rows(self, values: tuple[str, ...]) -> tuple[_VerificationTokenRow, ...]:
        return await find_many_in(self._prisma_client.db.litellm_verificationtoken, "token", values)

    async def _deleted_token_rows(self, values: tuple[str, ...]) -> tuple[_DeletedVerificationTokenRow, ...]:
        try:
            return await find_many_in(self._prisma_client.db.litellm_deletedverificationtoken, "token", values)
        except Exception as exc:
            verbose_proxy_logger.warning("Could not read deleted verification token metadata: %s", exc)
            return ()

    async def key_metadata(
        self, api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]:
        if not api_keys:
            return {}
        values: Final = tuple(api_keys)
        active_rows: Final = await self._active_token_rows(values)
        active: Final = MappingProxyType({row.token: self._metadata_row(row, key_exists=True) for row in active_rows})
        missing: Final = tuple(key for key in values if key not in active)
        deleted_rows: Final = await self._deleted_token_rows(missing)
        deleted_by_token: Final = MappingProxyType(
            {
                token: max(rows, key=lambda row: row.deleted_at)
                for token, rows in groupby(
                    sorted(deleted_rows, key=lambda row: row.token),
                    key=lambda row: row.token,
                )
            }
        )
        deleted: Final = MappingProxyType(
            {
                key: self._metadata_row(deleted_by_token[key], key_exists=False)
                for key in missing
                if key in deleted_by_token
            }
        )
        resolved: Final = MappingProxyType({**deleted, **active})
        return await self._proxy_reads.recover_key_metadata(resolved, api_keys, window)

    @staticmethod
    def _metadata_row(row: _VerificationTokenRow, *, key_exists: bool) -> KeyMetadataRow:
        tags: Final = _metadata_tags(row.metadata.get("tags") if row.metadata is not None else None)
        return KeyMetadataRow(
            api_key=row.token,
            key_alias=row.key_alias,
            team_id=row.team_id,
            user_id=row.user_id,
            user_email=None,
            key_exists=key_exists,
            tags=tags,
        )

    async def daily_rows(self, scope: DailyActivityScope, *, page: int, page_size: int) -> DailyRowsPage:
        if scope.table is DailyActivityTable.CUSTOMER and scope.team_ids is not None:
            result: Final = _CUSTOMER_DAILY_ROWS_ADAPTER.validate_python(
                await self._query(build_customer_daily_rows_sql(scope, page=page, page_size=page_size))
            )[0]
            return DailyRowsPage(total_count=result.total_count, rows=result.rows)
        table: Final = _daily_rows_table(self._prisma_client, scope.table)
        adjusted_start, adjusted_end = adjust_dates_for_timezone(
            scope.start_date,
            scope.end_date,
            scope.timezone_offset_minutes,
            include_current_utc_day=scope.include_current_utc_day,
        )
        exclusion_filter: Final = (
            {
                "OR": [
                    {scope.entity_id_field: None},
                    {scope.entity_id_field: {"not": {"in": list(scope.exclude_entity_ids)}}},
                ]
            }
            if scope.exclude_entity_ids
            else {}
        )
        conditions: Final = {
            "date": {"gte": adjusted_start, "lte": adjusted_end},
            **({scope.entity_id_field: {"in": list(scope.entity_ids)}} if scope.entity_ids is not None else {}),
            **exclusion_filter,
            **(
                {"AND": [{"OR": [{"model": None}, {"model": ""}, {"model": scope.model}]}]}
                if scope.model == UNKNOWN_MODEL_SPEND_LOG_MODEL
                else {"model": scope.model}
                if scope.model
                else {}
            ),
            **({"api_key": {"in": list(scope.api_keys)}} if scope.api_keys is not None else {}),
        }
        count, rows = await asyncio.gather(
            table.count(where=conditions),
            table.find_many(
                where=conditions,
                skip=(page - 1) * page_size,
                take=page_size,
                order=({"date": "desc"}, {"id": "asc"}),
            ),
        )
        return DailyRowsPage(total_count=count, rows=tuple(rows))

    async def customer_team_spend(
        self, scope: DailyActivityScope, *, row_ids: tuple[str, ...] | None = None
    ) -> tuple[CustomerTeamRollupRow, ...]:
        return _CUSTOMER_TEAM_ADAPTER.validate_python(
            await self._query(build_customer_team_rollup_sql(scope, row_ids=row_ids))
        )
