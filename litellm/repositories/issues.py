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

from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.common_utils.model_deprecation import deprecation_alert_state_key
from litellm.repositories.usage_alerts import AlertDatabase, UsageAlertsRepository
from litellm.types.proxy.issues import (
    Alert,
    AlertCohort,
    AlertDeliveryBinding,
    AlertFamily,
    AlertFamilyList,
    AlertFamilySetting,
    AlertFamilyUpdate,
    AlertInput,
    AlertKind,
    AlertList,
    AlertPendingDelivery,
    AlertProjectionSignal,
    AlertScope,
    AlertSignal,
    AlertSignalList,
)
from litellm.types.proxy.usage_alert_defaults import DEFAULT_USAGE_RULE_IDS, PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import UsageAlertMetric, UsageAlertStatus

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _IssueRow(BaseModel, frozen=True):
    record: Alert


class _SignalRow(BaseModel, frozen=True):
    record: AlertSignal


class _Count(BaseModel, frozen=True):
    count: int


class _Identifier(BaseModel, frozen=True):
    id: str


_ACK: Final = "CASE WHEN jsonb_typeof(c.param_value)='string' THEN (c.param_value#>>'{}')::jsonb ELSE c.param_value END"
_ACK_CURRENT: Final = f"""(({_ACK})->>'last_seen_date' > i.record->'deprecation'->>'last_used_date'
    OR (({_ACK})->>'last_seen_date' = i.record->'deprecation'->>'last_used_date'
        AND COALESCE((({_ACK})->>'last_seen_requests')::bigint,0)
            >= (i.record->'deprecation'->>'last_day_requests')::bigint))"""
_STATUS: Final = f"""CASE WHEN i.kind <> 'model_deprecation' THEN i.status
    WHEN NOT i.active THEN CASE WHEN ({_ACK})->>'status'='dismissed' AND ({_ACK_CURRENT})
        THEN 'dismissed' ELSE 'resolved' END
    WHEN COALESCE(({_ACK})->>'status','open')='open'
        OR ({_ACK})->>'last_seen_date' IS NULL
        OR i.record->'deprecation'->>'last_used_date' > ({_ACK})->>'last_seen_date'
        OR (i.record->'deprecation'->>'last_used_date' = ({_ACK})->>'last_seen_date'
            AND (i.record->'deprecation'->>'last_day_requests')::bigint
                > COALESCE((({_ACK})->>'last_seen_requests')::bigint,0)) THEN 'open'
    ELSE ({_ACK})->>'status' END"""
_PRIOR_STATUS: Final = _STATUS.replace(_ACK, _ACK.replace("c.param_value", "prior.param_value"))
_JOIN: Final = 'FROM "LiteLLM_Issue" i LEFT JOIN "LiteLLM_Config" c ON c.param_name=i.ack_key'
_RECORD: Final = f"""i.record || jsonb_build_object('status',({_STATUS}),'active',i.active,
    'first_seen',i.first_seen AT TIME ZONE 'UTC','last_seen',i.last_seen AT TIME ZONE 'UTC',
    'updated_at',i.updated_at AT TIME ZONE 'UTC') || CASE WHEN i.kind='model_deprecation'
    THEN jsonb_build_object('deprecation',i.record->'deprecation' ||
        jsonb_build_object('state',COALESCE(({_ACK}),i.record->'deprecation'->'state')))
    ELSE '{{}}'::jsonb END"""


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


def _issue_payload(data: AlertInput, status: UsageAlertStatus | None) -> Mapping[str, object]:
    initial: Final = Alert.model_validate(
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

    async def upsert_issues(
        self,
        inputs: tuple[AlertInput, ...],
        reopen: bool = False,
        status: UsageAlertStatus | None = None,
    ) -> tuple[Alert, ...]:
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
                    record=CASE WHEN old.kind='model_deprecation' AND (
                        EXCLUDED.last_seen > old.last_seen OR (EXCLUDED.last_seen = old.last_seen AND
                        EXCLUDED.record->'deprecation'->'model'->>'model_name' <
                            old.record->'deprecation'->'model'->>'model_name'))
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
                json.dumps(payload, ensure_ascii=False),
                reopen,
            )
        )
        if len(rows) != len(inputs):
            raise ValueError("An issue ID already belongs to another cohort")
        return tuple(row.record for row in rows)

    async def get_issue(self, issue_id: str) -> Alert | None:
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(f"SELECT {_RECORD} AS record {_JOIN} WHERE i.id=$1", issue_id)
        )
        return (await self._labels((rows[0].record,)))[0] if rows else None

    async def team_activity_access(self, team_id: str, user_id: str) -> LiteLLM_TeamTable | None:
        rows: Final = TypeAdapter(tuple[LiteLLM_TeamTable, ...]).validate_python(
            await self.database.query_raw(
                """SELECT t.team_id,
                    CASE WHEN m.role IS NULL THEN '[]'::jsonb
                        ELSE jsonb_build_array(jsonb_build_object('user_id',$2::text,'role',m.role)) END
                        AS members_with_roles,
                    CASE WHEN '/team/daily/activity'=ANY(t.team_member_permissions)
                        THEN ARRAY['/team/daily/activity'] ELSE ARRAY[]::text[] END AS team_member_permissions
                FROM "LiteLLM_TeamTable" t LEFT JOIN LATERAL (
                    SELECT member->>'role' AS role FROM jsonb_array_elements(t.members_with_roles) member
                    WHERE member->>'user_id'=$2 ORDER BY (member->>'role'='admin') DESC LIMIT 1
                ) m ON true WHERE t.team_id=$1""",
                team_id,
                user_id,
            )
        )
        return rows[0] if rows else None

    async def list_issues(
        self,
        kind: AlertKind | None = None,
        family: AlertFamily | None = None,
        status: UsageAlertStatus | None = None,
        scope_type: AlertScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AlertList:
        where: Final = f"""($1::text IS NULL OR i.kind=$1) AND ($2::text IS NULL OR i.family=$2)
            AND ($3::text IS NULL OR ({_STATUS})=$3) AND ($4::text IS NULL OR i.scope_type=$4)
            AND ($5::text IS NULL OR i.scope_id=$5)"""
        args: Final = (kind, family, status, scope_type, scope_id)
        count: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(f"SELECT count(*)::int AS count {_JOIN} WHERE {where}", *args)
        )
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(
                f"""WITH candidates AS MATERIALIZED (
                    SELECT i.id,i.last_seen,octet_length(i.record::text) AS bytes {_JOIN}
                    WHERE {where} ORDER BY i.last_seen DESC,i.id LIMIT $6 OFFSET $7
                ), sized AS (
                    SELECT *,SUM(bytes) OVER(ORDER BY last_seen DESC,id) AS page_bytes,
                        row_number() OVER(ORDER BY last_seen DESC,id) AS position FROM candidates
                ) SELECT {_RECORD} AS record {_JOIN} JOIN sized s ON s.id=i.id
                WHERE s.page_bytes<=1048576 OR s.position=1 ORDER BY i.last_seen DESC,i.id""",
                *args,
                limit,
                offset,
            )
        )
        return AlertList(
            data=await self._labels(tuple(row.record for row in rows)),
            total=count[0].count,
            limit=len(rows) if rows and offset + len(rows) < count[0].count else limit,
            offset=offset,
        )

    async def _labels(self, issues: tuple[Alert, ...]) -> tuple[Alert, ...]:
        if not issues:
            return ()
        labels: Final = await UsageAlertsRepository(self.database).resource_labels(
            tuple(issue.cohort for issue in issues)
        )
        return tuple(
            issue.model_copy(update={"scope_label": labels.get((issue.cohort.scope_type, issue.cohort.scope_id))})
            for issue in issues
        )

    async def find_active_issues(self, cohort_keys: tuple[str, ...]) -> tuple[Alert, ...]:
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
        authorized_scopes: tuple[AlertCohort, ...] | None = None,
    ) -> Alert | None:
        if authorized_scopes is not None and len(authorized_scopes) > 10:
            raise ValueError("Authorize at most 10 alert source scopes")
        changed: Final = TypeAdapter(tuple[_Identifier, ...]).validate_python(
            await self.database.query_raw(
                f"""WITH candidate AS MATERIALIZED (
                    SELECT i.*,({_STATUS}) AS effective_status {_JOIN} WHERE i.id=$1
                        AND NOT(i.kind='model_deprecation' AND NOT i.active AND $2='open')
                        AND ($5::jsonb IS NULL OR NOT EXISTS (
                            SELECT 1 FROM "LiteLLM_IssueSignal" s WHERE s.issue_id=i.id AND NOT EXISTS (
                                SELECT 1 FROM jsonb_to_recordset($5::jsonb) AS x(scope_type text,scope_id text)
                                WHERE s.source='performance' AND x.scope_type=s.record->'performance'->>'scope_type'
                                    AND x.scope_id IS NOT DISTINCT FROM s.record->'performance'->>'scope_id')))
                ), ack AS (
                    INSERT INTO "LiteLLM_Config" AS prior (param_name,param_value)
                    SELECT ack_key,jsonb_build_object('model',record->'deprecation'->>'usage_model',
                        'deprecation_date',record->'deprecation'->'model'->>'deprecation_date',
                        'status',$2::text,'last_seen_date',record->'deprecation'->>'last_used_date',
                        'last_seen_requests',record->'deprecation'->'last_day_requests',
                        'updated_at',CURRENT_TIMESTAMP,'updated_by',$4::text)
                    FROM candidate WHERE kind='model_deprecation' AND effective_status=$3
                    ON CONFLICT(param_name) DO UPDATE SET param_value=EXCLUDED.param_value
                    WHERE EXISTS(SELECT 1 FROM candidate i WHERE i.ack_key=prior.param_name
                        AND ({_PRIOR_STATUS})=$3)
                    RETURNING param_name
                ) UPDATE "LiteLLM_Issue" i SET status=$2,
                    updated_at=GREATEST(CURRENT_TIMESTAMP,i.updated_at+interval '1 millisecond')
                FROM candidate c WHERE i.id=c.id AND (c.kind='model_deprecation' OR i.updated_at=c.updated_at)
                    AND c.effective_status=$3 AND
                    ((c.kind<>'model_deprecation' AND i.status=$3) OR i.ack_key IN(SELECT param_name FROM ack))
                RETURNING i.id""",
                issue_id,
                status,
                expected_status,
                actor,
                json.dumps(tuple(scope.model_dump(mode="json") for scope in authorized_scopes))
                if authorized_scopes is not None
                else None,
            )
        )
        return await self.get_issue(issue_id) if changed else None

    async def status_scopes(self, issue_id: str) -> tuple[AlertCohort, ...]:
        return TypeAdapter(tuple[AlertCohort, ...]).validate_python(
            await self.database.query_raw(
                """SELECT DISTINCT CASE WHEN source='performance' THEN record->'performance'->>'scope_type'
                    ELSE 'deployment' END AS scope_type,
                    CASE WHEN source='performance' THEN record->'performance'->>'scope_id' END AS scope_id
                FROM "LiteLLM_IssueSignal" WHERE issue_id=$1 ORDER BY scope_type,scope_id LIMIT 11""",
                issue_id,
            )
        )

    async def add_signal_batch(self, signals: tuple[AlertSignal, ...]) -> None:
        if not signals:
            return
        if len(signals) > 500:
            raise ValueError("Write at most 500 issue signals per batch")
        await self.database.query_raw(
            """WITH incoming AS MATERIALIZED (
                SELECT DISTINCT ON(x->>'issue_id',x->>'source',x->>'source_id') x AS record
                FROM jsonb_array_elements($1::jsonb) x
            ), parents AS MATERIALIZED (
                SELECT i.id FROM "LiteLLM_Issue" i WHERE i.id IN(SELECT record->>'issue_id' FROM incoming)
                ORDER BY i.id FOR UPDATE
            ), added AS (
                INSERT INTO "LiteLLM_IssueSignal" (issue_id,source,source_id,record)
                SELECT x.record->>'issue_id',x.record->>'source',x.record->>'source_id',x.record
                FROM incoming x JOIN parents p ON p.id=x.record->>'issue_id'
                ON CONFLICT(issue_id,source,source_id) DO NOTHING RETURNING issue_id,source,record
            ) UPDATE "LiteLLM_Issue" i SET
                updated_at=GREATEST(CURRENT_TIMESTAMP,i.updated_at+interval '1 millisecond'),
                status=CASE WHEN i.kind='performance' AND i.active AND i.status<>'open' AND EXISTS (
                    SELECT 1 FROM added a WHERE a.issue_id=i.id AND a.source='performance'
                        AND (a.record->'performance'->>'scope_type'<>i.scope_type
                            OR a.record->'performance'->>'scope_id' IS DISTINCT FROM i.scope_id)
                        AND NOT EXISTS(SELECT 1 FROM "LiteLLM_IssueSignal" prior
                            WHERE prior.issue_id=i.id AND prior.source='performance'
                                AND prior.record->'performance'->>'scope_type'=a.record->'performance'->>'scope_type'
                                AND prior.record->'performance'->>'scope_id'
                                    IS NOT DISTINCT FROM a.record->'performance'->>'scope_id')
                ) THEN 'open' ELSE i.status END
            WHERE i.id IN(SELECT issue_id FROM added) RETURNING i.id""",
            json.dumps(tuple(signal.model_dump(mode="json") for signal in signals)),
        )

    async def list_signals(
        self,
        issue_id: str,
        limit: int = 50,
        offset: int = 0,
        visible_scope: AlertCohort | None = None,
    ) -> AlertSignalList:
        where: Final = """issue_id=$1 AND ($2::text IS NULL OR (source='performance'
            AND record->'performance'->>'scope_type'=$2
            AND record->'performance'->>'scope_id' IS NOT DISTINCT FROM $3::text))"""
        args: Final = (
            issue_id,
            visible_scope.scope_type if visible_scope else None,
            visible_scope.scope_id if visible_scope else None,
        )
        counts: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                f'SELECT count(*)::int AS count FROM "LiteLLM_IssueSignal" WHERE {where}',
                *args,
            )
        )
        rows: Final = TypeAdapter(tuple[_SignalRow, ...]).validate_python(
            await self.database.query_raw(
                f'SELECT record FROM "LiteLLM_IssueSignal" WHERE {where} ORDER BY source,source_id LIMIT $4 OFFSET $5',
                *args,
                limit,
                offset,
            )
        )
        return AlertSignalList(
            data=tuple(row.record for row in rows), total=counts[0].count, limit=limit, offset=offset
        )

    async def active_signals(
        self,
        day: date,
        limit: int = 500,
        cursor: str | None = None,
        scope_type: AlertScope | None = None,
        metric: UsageAlertMetric | None = None,
        scope_ids: tuple[str, ...] | None = None,
        completed_only: bool = False,
    ) -> tuple[AlertProjectionSignal, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Read between one and 500 source signals")
        if scope_ids == ():
            return ()
        if scope_ids is not None and len(scope_ids) > 500:
            raise ValueError("Filter at most 500 source scopes")
        return TypeAdapter(tuple[AlertProjectionSignal, ...]).validate_python(
            await self.database.query_raw(
                """SELECT a.record AS alert,r.record AS rule,a.evaluation_day
                FROM "LiteLLM_UsageAlertIncident" a JOIN "LiteLLM_UsageAlertRule" r ON r.id=a.rule_id
                WHERE a.evaluation_day=$1 AND a.breached AND r.enabled AND NOT r.deleted
                    AND NULLIF(a.evaluation_revision,'')::timestamptz=(r.record->>'updated_at')::timestamptz
                    AND a.scope_type=r.scope_type AND a.metric=r.record->>'metric'
                    AND a.record->>'direction'=r.record->>'direction'
                    AND (r.scope_id IS NULL OR a.scope_id=r.scope_id)
                    AND ($2::text IS NULL OR a.id>$2) AND ($4::text IS NULL OR a.scope_type=$4)
                    AND ($5::text IS NULL OR a.metric=$5)
                    AND ($6::text[] IS NULL OR a.scope_id=ANY($6::text[]))
                    AND (NOT $7::boolean OR NOT(r.id=ANY($8::text[])) OR NOT EXISTS (
                        SELECT 1 FROM "LiteLLM_UsageAlertRule" dependency
                        WHERE dependency.enabled AND NOT dependency.deleted AND dependency.id=ANY($8::text[])
                        AND ((a.scope_type<>'key' AND dependency.scope_type='key'
                            AND dependency.record->>'metric'=r.record->>'metric'
                            AND dependency.record->>'direction'=r.record->>'direction')
                            OR (r.record->>'metric'='requests' AND r.record->>'direction'='increase'
                                AND dependency.record->>'metric'='spend' AND dependency.record->>'direction'='increase'
                                AND dependency.scope_type IN ('key',r.scope_type)
                                AND NOT COALESCE(a.scope_type='key' AND dependency.scope_type='key'
                                    AND dependency.record->'evaluation_progress'->>'day'=$1::text
                                    AND (dependency.record->'evaluation_progress'->>'revision')::timestamptz=
                                        (dependency.record->>'updated_at')::timestamptz
                                    AND dependency.record->'evaluation_progress'->>'cursor'>=a.scope_id,false)))
                        AND NOT (dependency.record->>'evaluation_status'='insufficient_coverage'
                            AND dependency.record->>'last_evaluated_day'=$1::text)
                        AND NOT EXISTS (SELECT 1 FROM "LiteLLM_UsageAlertCheckpoint" completed WHERE completed.completed
                            AND completed.id=dependency.id||':'||replace(dependency.record->>'updated_at','Z','+00:00')||':'||$1::text)))
                    AND (NOT(r.id=ANY($8::text[])) OR NOT EXISTS (
                        SELECT 1 FROM "LiteLLM_UsageAlertRule" override WHERE override.enabled AND NOT override.deleted
                        AND COALESCE((override.record->>'overrides_defaults')::boolean,false)
                        AND NOT(override.id=ANY($8::text[])) AND override.scope_type=r.scope_type
                        AND override.record->>'metric'=r.record->>'metric'
                        AND override.record->>'direction'=r.record->>'direction'
                        AND (override.scope_id IS NULL OR override.scope_id=a.scope_id)))
                    ORDER BY a.id LIMIT $3""",
                day.isoformat(),
                cursor,
                limit,
                scope_type,
                metric,
                list(scope_ids) if scope_ids is not None else None,
                completed_only,
                tuple(DEFAULT_USAGE_RULE_IDS),
            )
        )

    async def active_resource_issue_ids(self, cursor: str | None = None) -> tuple[str, ...]:
        rows: Final = TypeAdapter(tuple[_Identifier, ...]).validate_python(
            await self.database.query_raw(
                """SELECT id FROM "LiteLLM_Issue" WHERE kind='performance' AND active AND scope_type<>'provider'
                    AND ($1::text IS NULL OR id>$1) ORDER BY id LIMIT 10""",
                cursor,
            )
        )
        return tuple(row.id for row in rows)

    async def recover_issues(self, day: date, issue_ids: tuple[str, ...] | None = None) -> int:
        if issue_ids == ():
            return 0
        if issue_ids is not None and len(issue_ids) > 10:
            raise ValueError("Recover at most 10 resource alerts")
        selection: Final = "i.id=ANY($3::text[])" if issue_ids is not None else "$3::text[] IS NULL"
        rows: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                f"""WITH primary_signals AS (
                SELECT i.id,a.id AS source_id,a.breached,a.evaluation_day,a.evaluation_revision,
                    max(a.evaluation_day) OVER (PARTITION BY i.id) AS latest_observation_day,
                    r.id AS rule_id,r.record AS rule_record,
                    COALESCE(r.enabled AND NOT r.deleted
                        AND r.scope_type=a.scope_type AND r.record->>'metric'=a.metric
                        AND r.record->>'direction'=a.record->>'direction'
                        AND (r.scope_id IS NULL OR r.scope_id=a.scope_id),false) AS eligible,
                    COALESCE(r.id=ANY($2::text[]) AND EXISTS (
                            SELECT 1 FROM "LiteLLM_UsageAlertRule" custom WHERE custom.enabled AND NOT custom.deleted
                            AND COALESCE((custom.record->>'overrides_defaults')::boolean,false)
                            AND NOT(custom.id=ANY($2::text[])) AND custom.scope_type=r.scope_type
                            AND custom.record->>'metric'=r.record->>'metric'
                            AND custom.record->>'direction'=r.record->>'direction'
                            AND (custom.scope_id IS NULL OR custom.scope_id=a.scope_id)),false) AS suppressed
                FROM "LiteLLM_Issue" i JOIN "LiteLLM_IssueSignal" s ON s.issue_id=i.id
                LEFT JOIN "LiteLLM_UsageAlertIncident" a ON a.id=s.source_id
                LEFT JOIN "LiteLLM_UsageAlertRule" r ON r.id=a.rule_id
                WHERE i.kind='performance' AND i.active AND i.scope_type<>'provider' AND s.source='performance'
                    AND ({selection})
                    AND s.record->'performance'->>'scope_type'=i.scope_type
                    AND s.record->'performance'->>'scope_id' IS NOT DISTINCT FROM i.scope_id
                    AND ((i.family='spend' AND s.record->'performance'->>'metric'='spend')
                        OR (i.family='latency' AND s.record->'performance'->>'metric'='latency')
                        OR (i.family='traffic' AND s.record->'performance'->>'metric'='requests')
                        OR (i.family='reliability' AND s.record->'performance'->>'metric' IN ('errors','error_rate')))
            ), observations AS (
                SELECT *,COALESCE(source_id IS NOT NULL AND NOT breached AND evaluation_day<=$1
                    AND evaluation_day>=latest_observation_day
                    AND NULLIF(evaluation_revision,'')::timestamptz=(rule_record->>'updated_at')::timestamptz,false) AS healthy
                FROM primary_signals
            ), healthy AS (
                SELECT id FROM observations GROUP BY id
                HAVING bool_or(eligible AND healthy) AND bool_and(NOT eligible OR suppressed OR healthy)
            ), changed AS (
                UPDATE "LiteLLM_Issue" i SET active=false,
                    status=CASE WHEN status='open' THEN 'resolved' ELSE status END,updated_at=CURRENT_TIMESTAMP
                WHERE id IN(SELECT id FROM healthy) AND active RETURNING id
            ) SELECT count(*)::int AS count FROM changed""",
                day.isoformat(),
                tuple(DEFAULT_USAGE_RULE_IDS),
                issue_ids,
            )
        )
        return rows[0].count

    async def active_provider_issues(self, limit: int = 500, cursor: str | None = None) -> tuple[Alert, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Read between one and 500 provider issues")
        rows: Final = TypeAdapter(tuple[_IssueRow, ...]).validate_python(
            await self.database.query_raw(
                f"""SELECT {_RECORD} AS record {_JOIN}
                WHERE i.kind='performance' AND i.scope_type='provider' AND i.active
                    AND ($2::text IS NULL OR i.id>$2) ORDER BY i.id LIMIT $1""",
                limit,
                cursor,
            )
        )
        return tuple(row.record for row in rows)

    async def recover_issue_ids(self, ids: tuple[str, ...], observed_at: datetime) -> int:
        if len(ids) > 500 or observed_at.tzinfo is None:
            raise ValueError("Recover at most 500 issues with a timezone-aware observation")
        if not ids:
            return 0
        rows: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                """WITH changed AS (
                UPDATE "LiteLLM_Issue" SET active=false,
                    status=CASE WHEN status='open' THEN 'resolved' ELSE status END,updated_at=CURRENT_TIMESTAMP
                WHERE id=ANY($1::text[]) AND kind='performance' AND active AND last_seen<$2::timestamp(3)
                RETURNING id
            ) SELECT count(*)::int AS count FROM changed""",
                list(ids),
                _utc(observed_at),
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
                WHERE kind='model_deprecation' AND active AND last_seen<$1::timestamp(3) RETURNING id
            ) SELECT count(*)::int AS count FROM changed""",
                _utc(observed_at),
            )
        )
        return rows[0].count

    async def family_settings(self) -> "AlertFamilyList":
        from litellm.repositories.issue_families import family_settings

        return await family_settings(self.database)

    async def update_family(
        self, family: "PerformanceIssueFamily", data: "AlertFamilyUpdate"
    ) -> "AlertFamilySetting | None":
        from litellm.repositories.issue_families import update_family

        return await update_family(self.database, family, data)

    async def queue_deliveries(self, bindings: tuple[AlertDeliveryBinding, ...]) -> None:
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

    async def pending_deliveries(self, limit: int = 100) -> tuple[AlertPendingDelivery, ...]:
        return await self._pending_deliveries(limit, None)

    async def pending_delivery(self, delivery_id: str) -> AlertPendingDelivery | None:
        rows: Final = await self._pending_deliveries(1, delivery_id)
        return rows[0] if rows else None

    async def _pending_deliveries(self, limit: int, delivery_id: str | None) -> tuple[AlertPendingDelivery, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Read between one and 500 deliveries")
        return TypeAdapter(tuple[AlertPendingDelivery, ...]).validate_python(
            await self.database.query_raw(
                f"""SELECT d.record || jsonb_build_object('next_attempt',d.next_attempt AT TIME ZONE 'UTC') AS delivery,
                {_RECORD} AS issue,bound.rule,bound.alert
            FROM "LiteLLM_IssueDelivery" d JOIN "LiteLLM_Issue" i ON i.id=d.issue_id
            LEFT JOIN "LiteLLM_Config" c ON c.param_name=i.ack_key
            JOIN LATERAL (
                SELECT r.record AS rule,a.record AS alert FROM "LiteLLM_IssueDeliveryRule" b
                JOIN "LiteLLM_UsageAlertRule" r ON r.id=b.rule_id
                JOIN "LiteLLM_UsageAlertIncident" a ON a.id=b.source_id AND a.rule_id=r.id
                WHERE b.delivery_id=COALESCE(d.record->>'parent_delivery_id',d.id) AND r.enabled AND NOT r.deleted AND a.breached AND a.status='open'
                    AND (r.record->'notification_channels') ? (d.record->>'channel')
                    AND (d.record->>'channel'<>'email' OR
                        (jsonb_array_length(d.record->'email_recipients')>0
                         AND (d.record->'email_recipients') <@ (r.record->'email_recipients')))
                    AND (r.record->>'updated_at')::timestamptz=b.rule_revision::timestamptz
                    AND NULLIF(a.evaluation_revision,'')::timestamptz=b.rule_revision::timestamptz
                    AND a.scope_type=r.scope_type AND a.metric=r.record->>'metric'
                    AND a.record->>'direction'=r.record->>'direction'
                    AND (r.scope_id IS NULL OR a.scope_id=r.scope_id)
                    AND (NOT(r.id=ANY($3::text[])) OR NOT EXISTS (
                        SELECT 1 FROM "LiteLLM_UsageAlertRule" custom WHERE custom.enabled AND NOT custom.deleted
                        AND COALESCE((custom.record->>'overrides_defaults')::boolean,false)
                        AND NOT(custom.id=ANY($3::text[])) AND custom.scope_type=r.scope_type
                        AND custom.record->>'metric'=r.record->>'metric'
                        AND custom.record->>'direction'=r.record->>'direction'
                        AND (custom.scope_id IS NULL OR custom.scope_id=a.scope_id)))
                    AND EXISTS(SELECT 1 FROM "LiteLLM_IssueSignal" s WHERE s.issue_id=i.id
                        AND s.source='performance' AND s.source_id=a.id)
                ORDER BY b.rule_id,b.source_id LIMIT 1
            ) bound ON true
            WHERE NOT d.sent AND d.next_attempt<=CURRENT_TIMESTAMP AND i.active AND ({_STATUS})='open'
                AND ($2::text IS NULL OR d.id=$2)
            ORDER BY d.next_attempt,d.id LIMIT $1""",
                limit,
                delivery_id,
                tuple(DEFAULT_USAGE_RULE_IDS),
            )
        )

    async def reserve_delivery(self, delivery_id: str) -> bool:
        rows: Final = await self.database.query_raw(
            """UPDATE "LiteLLM_IssueDelivery" SET next_attempt=CURRENT_TIMESTAMP+interval '1 hour',
                updated_at=CURRENT_TIMESTAMP WHERE id=$1 AND NOT sent AND next_attempt<=CURRENT_TIMESTAMP RETURNING id""",
            delivery_id,
        )
        return bool(rows)

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
            f"""WITH expired AS MATERIALIZED (
                SELECT i.id {_JOIN} WHERE i.status<>'open' AND ({_STATUS})<>'open'
                    AND i.updated_at<CURRENT_TIMESTAMP-make_interval(days=>$1::int)
                ORDER BY i.updated_at,i.id LIMIT $2
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
