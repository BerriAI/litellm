import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    cast,  # noqa: TID251  # generated Prisma client implements query_raw
)

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.common_utils.model_deprecation import deprecation_alert_state_key
from litellm.repositories.usage_alerts import AlertDatabase, UsageAlertsRepository
from litellm.types.proxy.issues import (
    Issue,
    IssueDeliveryBinding,
    IssueDeliveryInput,
    IssueFamily,
    IssueFamilyList,
    IssueFamilySetting,
    IssueFamilyUpdate,
    IssueInput,
    IssueKind,
    IssueList,
    IssuePendingDelivery,
    IssueProjectionSignal,
    IssueScope,
    IssueSignal,
    IssueSignalInput,
    IssueSignalList,
)
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import UsageAlertMetric, UsageAlertStatus

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _IssueRow(BaseModel, frozen=True):
    record: Issue


class _SignalRow(BaseModel, frozen=True):
    record: IssueSignal


class _Count(BaseModel, frozen=True):
    count: int


class _Identifier(BaseModel, frozen=True):
    id: str


_ACK: Final = "CASE WHEN jsonb_typeof(c.param_value)='string' THEN (c.param_value#>>'{}')::jsonb ELSE c.param_value END"
_STATUS: Final = f"""CASE WHEN i.kind <> 'model_deprecation' THEN i.status
    WHEN NOT i.active THEN CASE WHEN ({_ACK})->>'status'='dismissed' THEN 'dismissed' ELSE 'resolved' END
    WHEN COALESCE(({_ACK})->>'status','open')='open'
        OR ({_ACK})->>'last_seen_date' IS NULL
        OR i.record->'deprecation'->>'last_used_date' > ({_ACK})->>'last_seen_date'
        OR (i.record->'deprecation'->>'last_used_date' = ({_ACK})->>'last_seen_date'
            AND (i.record->'deprecation'->>'last_day_requests')::bigint
                > COALESCE((({_ACK})->>'last_seen_requests')::bigint,0)) THEN 'open'
    ELSE ({_ACK})->>'status' END"""
_JOIN: Final = 'FROM "LiteLLM_Issue" i LEFT JOIN "LiteLLM_Config" c ON c.param_name=i.ack_key'
_RECORD: Final = f"""i.record || jsonb_build_object('status',({_STATUS}),'active',i.active,
    'first_seen',i.first_seen AT TIME ZONE 'UTC','last_seen',i.last_seen AT TIME ZONE 'UTC',
    'updated_at',i.updated_at AT TIME ZONE 'UTC') || CASE WHEN i.kind='model_deprecation'
    THEN jsonb_build_object('deprecation',i.record->'deprecation' ||
        jsonb_build_object('state',COALESCE(({_ACK}),i.record->'deprecation'->'state')))
    ELSE '{{}}'::jsonb END"""


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


def _issue_payload(data: IssueInput, status: UsageAlertStatus | None) -> Mapping[str, object]:
    initial: Final = Issue.model_validate(
        {
            **data.model_dump(),
            "status": status or "open",
            "first_seen": data.observed_at,
            "last_seen": data.observed_at,
            "updated_at": data.observed_at,
        }
    )
    return {
        "record": initial.model_dump(mode="json"),
        "ack_key": deprecation_alert_state_key(data.deprecation.usage_model, data.deprecation.model.deprecation_date)
        if data.deprecation
        else None,
    }


class IssuesRepository:
    def __init__(self, database: AlertDatabase) -> None:
        self.database = database

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient") -> "IssuesRepository":
        return cls(cast(AlertDatabase, prisma.writer_db))  # cast-ok: generated Prisma delegate implements query_raw

    async def upsert_issue(
        self, data: IssueInput, reopen: bool = False, status: UsageAlertStatus | None = None
    ) -> Issue:
        rows: Final = await self.upsert_issues((data,), reopen=reopen, status=status)
        return rows[0]

    async def upsert_issues(
        self,
        inputs: tuple[IssueInput, ...],
        reopen: bool = False,
        status: UsageAlertStatus | None = None,
    ) -> tuple[Issue, ...]:
        if not inputs:
            return ()
        if len(inputs) > 500 or len(frozenset(item.id for item in inputs)) != len(inputs):
            raise ValueError("Write at most 500 distinct issues per batch")
        payload: Final = tuple(_issue_payload(item, status) for item in inputs)
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(
                f"""WITH incoming AS (
                SELECT x->'record' AS record,x->>'ack_key' AS ack_key FROM jsonb_array_elements($1::jsonb) x
            ), changed AS (
                INSERT INTO "LiteLLM_Issue" AS old
                    (id,cohort_key,kind,family,scope_type,scope_id,status,ack_key,record,first_seen,last_seen,updated_at)
                SELECT record->>'id',record->>'cohort_key',record->>'kind',record->>'family',
                    record->'cohort'->>'scope_type',record->'cohort'->>'scope_id',record->>'status',ack_key,record,
                    (record->>'observed_at')::timestamptz AT TIME ZONE 'UTC',
                    (record->>'observed_at')::timestamptz AT TIME ZONE 'UTC',
                    (record->>'observed_at')::timestamptz AT TIME ZONE 'UTC' FROM incoming
                ON CONFLICT(id) DO UPDATE SET
                    record=CASE WHEN old.kind='model_deprecation' AND EXCLUDED.last_seen >= old.last_seen
                        THEN EXCLUDED.record ELSE old.record END,
                    last_seen=GREATEST(old.last_seen,EXCLUDED.last_seen),
                    active=CASE WHEN old.kind='model_deprecation' AND EXCLUDED.last_seen >= old.last_seen
                        THEN true ELSE old.active END,
                    status=CASE WHEN $2 AND old.active AND EXCLUDED.last_seen > old.updated_at
                        THEN 'open' ELSE old.status END,
                    updated_at=GREATEST(old.updated_at,EXCLUDED.last_seen)
                WHERE old.kind=EXCLUDED.kind AND old.cohort_key=EXCLUDED.cohort_key
                    AND old.scope_type=EXCLUDED.scope_type AND old.scope_id IS NOT DISTINCT FROM EXCLUDED.scope_id
                RETURNING *
            ) SELECT {_RECORD} AS record FROM changed i LEFT JOIN "LiteLLM_Config" c ON c.param_name=i.ack_key""",
                json.dumps(payload),
                reopen,
            )
        )
        if len(rows) != len(inputs):
            raise ValueError("An issue ID already belongs to another cohort")
        return tuple(row.record for row in rows)

    async def get_issue(self, issue_id: str) -> Issue | None:
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(f"SELECT {_RECORD} AS record {_JOIN} WHERE i.id=$1", issue_id)
        )
        return (await self._labels((rows[0].record,)))[0] if rows else None

    async def list_issues(
        self,
        kind: IssueKind | None = None,
        family: IssueFamily | None = None,
        status: UsageAlertStatus | None = None,
        scope_type: IssueScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> IssueList:
        where: Final = f"""($1::text IS NULL OR i.kind=$1) AND ($2::text IS NULL OR i.family=$2)
            AND ($3::text IS NULL OR ({_STATUS})=$3) AND ($4::text IS NULL OR i.scope_type=$4)
            AND ($5::text IS NULL OR i.scope_id=$5)"""
        args: Final = (kind, family, status, scope_type, scope_id)
        count: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(f"SELECT count(*)::int AS count {_JOIN} WHERE {where}", *args)
        )
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(
                f"SELECT {_RECORD} AS record {_JOIN} WHERE {where} ORDER BY i.last_seen DESC,i.id LIMIT $6 OFFSET $7",
                *args,
                limit,
                offset,
            )
        )
        return IssueList(
            data=await self._labels(tuple(row.record for row in rows)), total=count[0].count, limit=limit, offset=offset
        )

    async def _labels(self, issues: tuple[Issue, ...]) -> tuple[Issue, ...]:
        if not issues:
            return ()
        labels: Final = await UsageAlertsRepository(self.database).resource_labels(
            tuple(issue.cohort for issue in issues)
        )
        return tuple(
            issue.model_copy(update={"scope_label": labels.get((issue.cohort.scope_type, issue.cohort.scope_id))})
            for issue in issues
        )

    async def find_active_issue(self, cohort_key: str) -> Issue | None:
        rows: Final = await self.find_active_issues((cohort_key,))
        return rows[0] if rows else None

    async def find_active_issues(self, cohort_keys: tuple[str, ...]) -> tuple[Issue, ...]:
        if len(cohort_keys) > 500:
            raise ValueError("Read at most 500 issue cohorts per batch")
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(
                f"SELECT DISTINCT ON(i.cohort_key) {_RECORD} AS record {_JOIN} "
                "WHERE i.cohort_key=ANY($1::text[]) AND i.active "
                "ORDER BY i.cohort_key,i.first_seen DESC,i.id",
                list(cohort_keys),
            )
        )
        return tuple(row.record for row in rows)

    async def set_status(
        self,
        issue_id: str,
        status: UsageAlertStatus,
        expected_status: UsageAlertStatus,
        actor: str | None = None,
    ) -> Issue | None:
        changed: Final = TypeAdapter(tuple[_Identifier, ...]).validate_python(
            await self.database.query_raw(
                f"""WITH candidate AS MATERIALIZED (
                    SELECT i.*,({_STATUS}) AS effective_status {_JOIN} WHERE i.id=$1
                ), ack AS (
                    INSERT INTO "LiteLLM_Config" AS prior (param_name,param_value)
                    SELECT ack_key,jsonb_build_object('model',record->'deprecation'->>'usage_model',
                        'deprecation_date',record->'deprecation'->'model'->>'deprecation_date',
                        'status',$2::text,'last_seen_date',record->'deprecation'->>'last_used_date',
                        'last_seen_requests',record->'deprecation'->'last_day_requests',
                        'updated_at',CURRENT_TIMESTAMP,'updated_by',$4::text)
                    FROM candidate WHERE kind='model_deprecation' AND effective_status=$3
                    ON CONFLICT(param_name) DO UPDATE SET param_value=EXCLUDED.param_value
                    WHERE (CASE WHEN COALESCE((CASE WHEN jsonb_typeof(prior.param_value)='string' THEN (prior.param_value#>>'{{}}')::jsonb ELSE prior.param_value END)->>'status','open')='open'
                        OR (CASE WHEN jsonb_typeof(prior.param_value)='string' THEN (prior.param_value#>>'{{}}')::jsonb ELSE prior.param_value END)->>'last_seen_date' IS NULL
                        OR EXCLUDED.param_value->>'last_seen_date' > (CASE WHEN jsonb_typeof(prior.param_value)='string' THEN (prior.param_value#>>'{{}}')::jsonb ELSE prior.param_value END)->>'last_seen_date'
                        OR (EXCLUDED.param_value->>'last_seen_date'=(CASE WHEN jsonb_typeof(prior.param_value)='string' THEN (prior.param_value#>>'{{}}')::jsonb ELSE prior.param_value END)->>'last_seen_date'
                            AND (EXCLUDED.param_value->>'last_seen_requests')::bigint
                                > COALESCE(((CASE WHEN jsonb_typeof(prior.param_value)='string' THEN (prior.param_value#>>'{{}}')::jsonb ELSE prior.param_value END)->>'last_seen_requests')::bigint,0))
                        THEN 'open' ELSE (CASE WHEN jsonb_typeof(prior.param_value)='string' THEN (prior.param_value#>>'{{}}')::jsonb ELSE prior.param_value END)->>'status' END)=$3
                    RETURNING param_name
                ) UPDATE "LiteLLM_Issue" i SET status=$2,updated_at=CURRENT_TIMESTAMP
                FROM candidate c WHERE i.id=c.id AND c.effective_status=$3 AND
                    ((c.kind<>'model_deprecation' AND i.status=$3) OR i.ack_key IN(SELECT param_name FROM ack))
                RETURNING i.id""",
                issue_id,
                status,
                expected_status,
                actor,
            )
        )
        return await self.get_issue(issue_id) if changed else None

    async def add_signals(self, issue_id: str, signals: tuple[IssueSignalInput, ...]) -> None:
        await self.add_signal_batch(
            tuple(IssueSignal.model_validate({**signal.model_dump(), "issue_id": issue_id}) for signal in signals)
        )

    async def add_signal_batch(self, signals: tuple[IssueSignal, ...]) -> None:
        if not signals:
            return
        if len(signals) > 500:
            raise ValueError("Write at most 500 issue signals per batch")
        await self.database.query_raw(
            """INSERT INTO "LiteLLM_IssueSignal" (issue_id,source,source_id,record)
            SELECT DISTINCT ON(x->>'issue_id',x->>'source',x->>'source_id')
                x->>'issue_id',x->>'source',x->>'source_id',x FROM jsonb_array_elements($1::jsonb) x
            WHERE EXISTS(SELECT 1 FROM "LiteLLM_Issue" i WHERE i.id=x->>'issue_id'
                AND (i.scope_type NOT IN ('key','team','user') OR
                    (x->'usage_alert'->>'scope_type'=i.scope_type
                     AND x->'usage_alert'->>'scope_id'=i.scope_id)))
            ON CONFLICT(issue_id,source,source_id) DO NOTHING RETURNING source_id""",
            json.dumps(tuple(signal.model_dump(mode="json") for signal in signals)),
        )

    async def list_signals(self, issue_id: str, limit: int = 50, offset: int = 0) -> IssueSignalList:
        counts: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                'SELECT count(*)::int AS count FROM "LiteLLM_IssueSignal" WHERE issue_id=$1',
                issue_id,
            )
        )
        rows: Final = TypeAdapter(tuple[_SignalRow, ...]).validate_python(
            await self.database.query_raw(
                'SELECT record FROM "LiteLLM_IssueSignal" WHERE issue_id=$1 ORDER BY source,source_id LIMIT $2 OFFSET $3',
                issue_id,
                limit,
                offset,
            )
        )
        return IssueSignalList(
            data=tuple(row.record for row in rows), total=counts[0].count, limit=limit, offset=offset
        )

    async def active_signals(
        self,
        day: date,
        limit: int = 500,
        cursor: str | None = None,
        scope_type: IssueScope | None = None,
        metric: UsageAlertMetric | None = None,
    ) -> tuple[IssueProjectionSignal, ...]:
        return TypeAdapter(tuple[IssueProjectionSignal, ...]).validate_python(
            await self.database.query_raw(
                """SELECT a.record AS alert,r.record AS rule,a.evaluation_day
                FROM "LiteLLM_UsageAlertIncident" a JOIN "LiteLLM_UsageAlertRule" r ON r.id=a.rule_id
                WHERE a.evaluation_day=$1 AND a.breached AND r.enabled AND NOT r.deleted
                    AND NULLIF(a.evaluation_revision,'')::timestamptz=(r.record->>'updated_at')::timestamptz
                    AND a.scope_type=r.scope_type AND a.metric=r.record->>'metric'
                    AND a.record->>'direction'=r.record->>'direction'
                    AND (r.scope_id IS NULL OR a.scope_id=r.scope_id)
                    AND ($2::text IS NULL OR a.id>$2) AND ($4::text IS NULL OR a.scope_type=$4)
                    AND ($5::text IS NULL OR a.metric=$5) ORDER BY a.id LIMIT $3""",
                day.isoformat(),
                cursor,
                limit,
                scope_type,
                metric,
            )
        )

    async def recover_issues(self, day: date) -> int:
        rows: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                """WITH changed AS (
                UPDATE "LiteLLM_Issue" i SET active=false,
                    status=CASE WHEN status='open' THEN 'resolved' ELSE status END,updated_at=CURRENT_TIMESTAMP
                WHERE kind='performance' AND active
                    AND EXISTS(SELECT 1 FROM "LiteLLM_IssueSignal" s WHERE s.issue_id=i.id AND s.source='usage_alert')
                    AND NOT EXISTS(SELECT 1 FROM "LiteLLM_IssueSignal" s
                        LEFT JOIN "LiteLLM_UsageAlertIncident" a ON a.id=s.source_id
                        WHERE s.issue_id=i.id AND s.source='usage_alert'
                            AND (a.id IS NULL OR a.evaluation_day>$1 OR a.breached)) RETURNING i.id
            ) SELECT count(*)::int AS count FROM changed""",
                day.isoformat(),
            )
        )
        return rows[0].count

    async def projection_completed(self, day: date, version: str) -> bool:
        rows: Final = TypeAdapter(tuple[_Identifier, ...]).validate_python(
            await self.database.query_raw(
                'SELECT id FROM "LiteLLM_UsageAlertCheckpoint" WHERE id=$1 AND completed',
                f"issues:{version}:{day.isoformat()}",
            )
        )
        return bool(rows)

    async def complete_projection(self, day: date, version: str) -> None:
        await self.database.query_raw(
            """INSERT INTO "LiteLLM_UsageAlertCheckpoint" (id,completed,expires_at) VALUES($1,true,CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET completed=true,expires_at=CURRENT_TIMESTAMP RETURNING id""",
            f"issues:{version}:{day.isoformat()}",
        )

    async def finish_deprecation_projection(self, observed_at: datetime) -> int:
        rows: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                """WITH changed AS (
                UPDATE "LiteLLM_Issue" SET active=false,
                    status=CASE WHEN status='open' THEN 'resolved' ELSE status END,updated_at=CURRENT_TIMESTAMP
                WHERE kind='model_deprecation' AND active AND last_seen<$1::timestamp RETURNING id
            ) SELECT count(*)::int AS count FROM changed""",
                _utc(observed_at),
            )
        )
        return rows[0].count

    async def family_settings(self) -> "IssueFamilyList":
        from litellm.repositories.issue_families import family_settings

        return await family_settings(self.database)

    async def update_family(
        self, family: "PerformanceIssueFamily", data: "IssueFamilyUpdate"
    ) -> "IssueFamilySetting | None":
        from litellm.repositories.issue_families import update_family

        return await update_family(self.database, family, data)

    async def queue_delivery(
        self,
        data: IssueDeliveryInput,
        rule_id: str,
        rule_revision: str,
        source_id: str,
    ) -> None:
        await self.queue_deliveries(
            (IssueDeliveryBinding(delivery=data, rule_id=rule_id, rule_revision=rule_revision, source_id=source_id),)
        )

    async def queue_deliveries(self, bindings: tuple[IssueDeliveryBinding, ...]) -> None:
        if not bindings:
            return
        if len(bindings) > 500:
            raise ValueError("Queue at most 500 delivery bindings per batch")
        await self.database.query_raw(
            """WITH incoming AS MATERIALIZED (
                SELECT x->'delivery' AS delivery,x->>'rule_id' AS rule_id,
                    x->>'rule_revision' AS revision,x->>'source_id' AS source_id
                FROM jsonb_array_elements($1::jsonb) x
                WHERE EXISTS(SELECT 1 FROM "LiteLLM_Issue" WHERE id=x->'delivery'->>'issue_id')
            ), saved AS (
                INSERT INTO "LiteLLM_IssueDelivery" (id,issue_id,record)
                SELECT DISTINCT ON(delivery->>'id') delivery->>'id',delivery->>'issue_id',delivery
                    FROM incoming ON CONFLICT(id) DO NOTHING RETURNING id
            ) INSERT INTO "LiteLLM_IssueDeliveryRule" (delivery_id,rule_id,source_id,rule_revision)
            SELECT DISTINCT ON(delivery->>'id',rule_id,source_id) delivery->>'id',rule_id,source_id,revision
            FROM incoming WHERE delivery->>'id' IN(SELECT id FROM saved)
                OR EXISTS(SELECT 1 FROM "LiteLLM_IssueDelivery" d WHERE d.id=delivery->>'id'
                    AND d.issue_id=delivery->>'issue_id')
            ON CONFLICT(delivery_id,rule_id,source_id) DO UPDATE SET rule_revision=EXCLUDED.rule_revision
            RETURNING delivery_id""",
            json.dumps(tuple(binding.model_dump(mode="json") for binding in bindings)),
        )

    async def pending_deliveries(self, limit: int = 100) -> tuple[IssuePendingDelivery, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Read between one and 500 deliveries")
        return TypeAdapter(tuple[IssuePendingDelivery, ...]).validate_python(
            await self.database.query_raw(
                f"""SELECT d.record || jsonb_build_object('next_attempt',d.next_attempt AT TIME ZONE 'UTC') AS delivery,
                {_RECORD} AS issue,bound.rule,bound.alert
            FROM "LiteLLM_IssueDelivery" d JOIN "LiteLLM_Issue" i ON i.id=d.issue_id
            LEFT JOIN "LiteLLM_Config" c ON c.param_name=i.ack_key
            JOIN LATERAL (
                SELECT r.record AS rule,a.record AS alert FROM "LiteLLM_IssueDeliveryRule" b
                JOIN "LiteLLM_UsageAlertRule" r ON r.id=b.rule_id
                JOIN "LiteLLM_UsageAlertIncident" a ON a.id=b.source_id AND a.rule_id=r.id
                WHERE b.delivery_id=d.id AND r.enabled AND NOT r.deleted AND a.breached AND a.status='open'
                    AND (r.record->>'updated_at')::timestamptz=b.rule_revision::timestamptz
                    AND NULLIF(a.evaluation_revision,'')::timestamptz=b.rule_revision::timestamptz
                    AND a.scope_type=r.scope_type AND a.metric=r.record->>'metric'
                    AND a.record->>'direction'=r.record->>'direction'
                    AND (r.scope_id IS NULL OR a.scope_id=r.scope_id)
                    AND EXISTS(SELECT 1 FROM "LiteLLM_IssueSignal" s WHERE s.issue_id=i.id
                        AND s.source='usage_alert' AND s.source_id=a.id)
                ORDER BY b.rule_id,b.source_id LIMIT 1
            ) bound ON true
            WHERE NOT d.sent AND d.next_attempt<=CURRENT_TIMESTAMP AND i.active AND ({_STATUS})='open'
            ORDER BY d.next_attempt,d.id LIMIT $1""",
                limit,
            )
        )

    async def mark_delivery_sent(self, delivery_id: str) -> None:
        await self.database.query_raw(
            """UPDATE "LiteLLM_IssueDelivery" SET sent=true,updated_at=CURRENT_TIMESTAMP,
                record=record || jsonb_build_object('last_error',NULL,'attempts',
                    COALESCE((record->>'attempts')::int,0)+1)
            WHERE id=$1 AND NOT sent RETURNING id""",
            delivery_id,
        )

    async def defer_delivery(
        self,
        delivery_id: str,
        error: Literal["delivery_failed", "transport_not_configured"],
    ) -> None:
        await self.database.query_raw(
            """UPDATE "LiteLLM_IssueDelivery" SET next_attempt=CURRENT_TIMESTAMP+interval '1 hour',
                updated_at=CURRENT_TIMESTAMP,record=record || jsonb_build_object('last_error',$2::text,
                    'attempts',COALESCE((record->>'attempts')::int,0)+1)
            WHERE id=$1 AND NOT sent RETURNING id""",
            delivery_id,
            error,
        )

    async def prune(self, retention_days: int = 90, limit: int = 500) -> None:
        if not 1 <= limit <= 500 or retention_days < 1:
            raise ValueError("Retention and cleanup batch must be bounded positive values")
        await self.database.query_raw(
            """WITH expired AS MATERIALIZED (
                SELECT id FROM "LiteLLM_Issue" WHERE status<>'open'
                    AND updated_at<CURRENT_TIMESTAMP-make_interval(days=>$1)
                ORDER BY updated_at,id LIMIT $2
            ), signals AS (
                DELETE FROM "LiteLLM_IssueSignal" WHERE (issue_id,source,source_id) IN (
                    SELECT issue_id,source,source_id FROM "LiteLLM_IssueSignal"
                    WHERE issue_id IN(SELECT id FROM expired) ORDER BY issue_id,source,source_id LIMIT $2)
                RETURNING issue_id
            ), bindings AS (
                DELETE FROM "LiteLLM_IssueDeliveryRule" WHERE (delivery_id,rule_id,source_id) IN (
                    SELECT b.delivery_id,b.rule_id,b.source_id FROM "LiteLLM_IssueDeliveryRule" b
                    JOIN "LiteLLM_IssueDelivery" d ON d.id=b.delivery_id
                    WHERE d.issue_id IN(SELECT id FROM expired) ORDER BY b.delivery_id,b.rule_id,b.source_id LIMIT $2)
                RETURNING delivery_id
            ), deliveries AS (
                DELETE FROM "LiteLLM_IssueDelivery" d WHERE d.id IN (
                    SELECT id FROM "LiteLLM_IssueDelivery" d WHERE issue_id IN(SELECT id FROM expired)
                        AND NOT EXISTS(SELECT 1 FROM "LiteLLM_IssueDeliveryRule" b WHERE b.delivery_id=d.id)
                    ORDER BY id LIMIT $2) RETURNING issue_id
            ), removed AS (
                DELETE FROM "LiteLLM_Issue" i WHERE id IN(SELECT id FROM expired)
                    AND NOT EXISTS(SELECT 1 FROM "LiteLLM_IssueSignal" s WHERE s.issue_id=i.id)
                    AND NOT EXISTS(SELECT 1 FROM "LiteLLM_IssueDelivery" d WHERE d.issue_id=i.id) RETURNING id
            ) SELECT count(*)::int AS count FROM removed""",
            retention_days,
            limit,
        )
