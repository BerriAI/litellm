"""Issue investigations reuse the same aggregate filters as entity Activity."""

from dataclasses import asdict, replace
from datetime import date
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # generated Prisma structural database interface

from pydantic import BaseModel, TypeAdapter

from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.repositories.analytics_budget import AnalyticsDatabase, BoundedAnalyticsDatabase
from litellm.repositories.bounded_daily_activity import BoundedDailyActivityRepository
from litellm.repositories.daily_activity_buckets import api_key_bucket, api_key_from_bucket
from litellm.repositories.daily_activity_repository import DailyActivityQueryDatabase, DailyActivityRepository
from litellm.repositories.daily_activity_sql import PRISMA_TO_PG_TABLE, build_where_clause
from litellm.types.proxy.issues import (
    IssueActivityResponse,
    IssueCohort,
    IssueImpact,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityKeyPageResponse,
    DailyActivityUserPageResponse,
    KeyMetadata,
    KeySpendActivityRow,
    KeySpendMetrics,
    UserActivityRow,
)
from litellm.types.repositories.daily_activity import DailyActivityScope, DailyActivityTable

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _Counts(BaseModel, frozen=True):
    keys: int
    entities: int
    models: int
    requests: int


def issue_activity_scope(cohort: IssueCohort, start: date, end: date) -> DailyActivityScope:
    if start > end or (end - start).days > 365:
        raise ValueError("Choose an ordered date range of at most 366 days")
    return DailyActivityScope(
        table=DailyActivityTable.TEAM if cohort.scope_type == "team" else DailyActivityTable.USER,
        entity_id_field="team_id" if cohort.scope_type == "team" else "user_id",
        entity_ids=(cohort.scope_id,) if cohort.scope_type in ("team", "user") and cohort.scope_id else None,
        exclude_entity_ids=(),
        api_keys=(api_key_from_bucket(cohort.scope_id),) if cohort.scope_type == "key" and cohort.scope_id else None,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        model=cohort.scope_id if cohort.scope_type == "model" else None,
        timezone_offset_minutes=0,
        include_current_utc_day=True,
        provider=cohort.scope_id if cohort.scope_type == "provider" else None,
        model_pairs=tuple((pair.provider, pair.model) for pair in cohort.model_pairs),
    )


class IssueActivityRepository:
    def __init__(self, activity: DailyActivityRepository, database: DailyActivityQueryDatabase) -> None:
        self.activity = activity
        self.database = database

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient") -> "IssueActivityRepository":
        from litellm.proxy.management_endpoints.common_daily_activity import daily_activity_repository

        database: Final = BoundedAnalyticsDatabase(cast(AnalyticsDatabase, prisma.db))  # cast-ok: generated Prisma tx
        return cls(daily_activity_repository(prisma, query_database=database), database)

    async def daily_activity(
        self, cohort: IssueCohort, start: date, end: date, model_limit: int = 20
    ) -> IssueActivityResponse:
        return await BoundedDailyActivityRepository(self.database).aggregate(
            issue_activity_scope(cohort, start, end), model_limit=model_limit if cohort.scope_type == "key" else None
        )

    async def keys(
        self, cohort: IssueCohort, start: date, end: date, limit: int, offset: int
    ) -> DailyActivityKeyPageResponse:
        page: Final = await self.activity.key_page(issue_activity_scope(cohort, start, end), limit=limit, offset=offset)
        metadata: Final = await self.activity.key_metadata(
            frozenset(row.api_key for row in page.rows if row.api_key), None
        )
        return DailyActivityKeyPageResponse(
            api_keys=[
                KeySpendActivityRow(
                    api_key=api_key_bucket(row.api_key),
                    metrics=KeySpendMetrics.model_validate(asdict(row)),
                    metadata=KeyMetadata.model_validate(asdict(metadata[row.api_key]))
                    if row.api_key in metadata
                    else KeyMetadata(),
                )
                for row in page.rows
            ],
            total_api_keys=page.total_api_keys,
            limit=limit,
            offset=offset,
        )

    async def users(
        self, cohort: IssueCohort, start: date, end: date, limit: int, offset: int
    ) -> DailyActivityUserPageResponse:
        if cohort.scope_type == "team":
            raise ValueError("Historical team aggregates do not retain user membership")
        page: Final = await self.activity.user_page(
            issue_activity_scope(cohort, start, end), limit=limit, offset=offset
        )
        metadata: Final = await self.activity.user_metadata(frozenset(row.user_id for row in page.rows if row.user_id))
        return DailyActivityUserPageResponse(
            users=tuple(
                UserActivityRow.model_validate(
                    {
                        **asdict(row),
                        **(
                            asdict(metadata[row.user_id]) if row.user_id is not None and row.user_id in metadata else {}
                        ),
                    }
                )
                for row in page.rows
            ),
            total_users=page.total_users,
            limit=limit,
            offset=offset,
        )

    async def _counts(self, scope: DailyActivityScope) -> _Counts:
        where, params = build_where_clause(scope)
        rows: Final = TypeAdapter(tuple[_Counts, ...]).validate_python(
            await self.database.query_raw(
                f'''SELECT COUNT(DISTINCT NULLIF(api_key,'')) FILTER(WHERE api_key<>${len(params) + 1})::int AS keys,
                COUNT(DISTINCT NULLIF("{scope.entity_id_field}",''))
                    FILTER(WHERE "{scope.entity_id_field}" NOT IN ('default_user_id','litellm-dashboard'))::int AS entities,
                COUNT(DISTINCT NULLIF(model,''))::int AS models,
                COALESCE(SUM(api_requests),0)::bigint AS requests
                FROM "{PRISMA_TO_PG_TABLE[scope.table]}" WHERE {where}''',
                *params,
                PTU_SENTINEL_API_KEY,
            )
        )
        return rows[0]

    async def impact(self, cohort: IssueCohort, start: date, end: date) -> IssueImpact:
        scope: Final = issue_activity_scope(cohort, start, end)
        counts: Final = await self._counts(scope)
        teams: Final = (
            await self._counts(replace(scope, table=DailyActivityTable.TEAM, entity_id_field="team_id"))
            if cohort.scope_type not in ("user", "team")
            else None
        )
        return IssueImpact(
            keys=counts.keys,
            models=counts.models,
            requests=counts.requests,
            users=counts.entities if cohort.scope_type != "team" else None,
            teams=counts.entities if cohort.scope_type == "team" else teams.entities if teams else None,
        )
