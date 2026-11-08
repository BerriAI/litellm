"""Bounded, writer-consistent storage for daily alert rules and episodes."""

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # PrismaWrapper forwards generated client methods dynamically
)
from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, TypeAdapter

from litellm.repositories.daily_activity_buckets import api_key_from_bucket
from litellm.types.proxy.usage_alert_defaults import DEFAULT_USAGE_RULE_IDS
from litellm.types.proxy.usage_alerts import (
    AlertModelList,
    AlertModelOption,
    AlertRule,
    AlertRuleInput,
    AlertRuleList,
    PerformanceSignal,
    PerformanceSignalList,
    UsageAlertDefaultSummary,
    UsageAlertEvaluationStatus,
    UsageAlertMetric,
    UsageAlertObservation,
    UsageAlertRuleKind,
    UsageAlertScope,
    UsageAlertStatus,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class AlertDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...


class ResourceLabelScope(Protocol):
    @property
    def scope_type(self) -> str: ...

    @property
    def scope_id(self) -> str | None: ...


class _Record(BaseModel, frozen=True):
    record: Mapping[str, object]


class _Count(BaseModel, frozen=True):
    count: int


class _RuleCounts(_Count, frozen=True):
    default_summary: UsageAlertDefaultSummary


class _ResourceLabel(BaseModel, frozen=True):
    scope_type: str
    scope_id: str | None
    label: str | None


class _ModelIdentity(BaseModel, frozen=True):
    model: str


class _Ownership(BaseModel, frozen=True):
    user_id: str | None
    team_id: str | None


_DEFAULT_RULE_IDS: Final = DEFAULT_USAGE_RULE_IDS


def _rule(record: Mapping[str, object]) -> AlertRule:
    return AlertRule.model_validate({**record, "is_default": str(record.get("id")) in _DEFAULT_RULE_IDS})


def activity_path(
    scope: UsageAlertScope, entity_id: str | None, day: date, alert_id: str, baseline_days: int = 7
) -> str:
    routes: Final = {
        "key": ("/api-keys", "key"),
        "team": ("/teams", "team"),
        "user": ("/users", "user"),
        "model": ("/models-and-endpoints", "activity_model"),
        "deployment": ("/usage", ""),
    }
    route, parameter = routes[scope]
    if scope == "deployment":
        return (
            route
            + "?"
            + urlencode(
                {
                    "tab": "keys",
                    "from": (day - timedelta(days=baseline_days)).isoformat(),
                    "to": day.isoformat(),
                    "alert": alert_id,
                }
            )
        )
    query: Final = {
        "activity_from": (day - timedelta(days=baseline_days)).isoformat(),
        "activity_to": day.isoformat(),
        "alert": alert_id,
    }
    if parameter and entity_id and not entity_id.startswith("sk-"):
        query[parameter] = entity_id
        query["detail_tab"] = "activity"
    return route + "?" + urlencode(query)


def _observation_payload(rule: AlertRule, item: UsageAlertObservation, now: datetime) -> Mapping[str, object]:
    alert_id: Final = str(
        uuid5(
            NAMESPACE_URL,
            json.dumps(
                [
                    "litellm:usage-alert",
                    rule.id,
                    rule.scope_type,
                    item.entity_id,
                    rule.metric,
                    rule.direction,
                    item.day.isoformat(),
                    rule.updated_at.isoformat(),
                ]
            ),
        )
    )
    incident: Final = PerformanceSignal(
        id=alert_id,
        rule_id=rule.id,
        rule_name=rule.name,
        scope_type=rule.scope_type,
        scope_id=item.entity_id,
        metric=rule.metric,
        direction=rule.direction,
        status="open",
        day=item.day,
        baseline_start=item.day - timedelta(days=rule.baseline_days),
        baseline_end=item.day - timedelta(days=1),
        baseline_sample_days=item.baseline_sample_days,
        observed_value=item.observed,
        baseline_value=item.baseline,
        threshold=item.threshold,
        first_seen=now,
        last_seen=now,
        activity_path=activity_path(rule.scope_type, item.entity_id, item.day, alert_id, rule.baseline_days),
    )
    return {
        "scope_key": item.entity_id or "",
        "breached": item.breached,
        "day": item.day.isoformat(),
        "record": incident.model_dump(mode="json"),
    }


class UsageAlertsRepository:
    def __init__(self, database: AlertDatabase) -> None:
        self.database: Final = database

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient") -> "UsageAlertsRepository":
        return cls(
            cast(AlertDatabase, prisma.writer_db)  # cast-ok: PrismaWrapper forwards query_raw to its generated client
        )

    async def get_rule(self, rule_id: str) -> AlertRule | None:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                'SELECT record FROM "LiteLLM_UsageAlertRule" WHERE id=$1 AND NOT deleted', rule_id
            )
        )
        return _rule(rows[0].record) if rows else None

    async def save_rule(
        self,
        data: AlertRuleInput,
        created_by: str | None,
        existing: AlertRule | None = None,
        *,
        overrides_defaults: bool = False,
    ) -> AlertRule | None:
        now: Final = datetime.now(timezone.utc)
        rule: Final = AlertRule.model_validate(
            {
                **data.model_dump(),
                "id": existing.id if existing else str(uuid4()),
                "created_by": existing.created_by if existing else created_by,
                "created_at": existing.created_at if existing else now,
                "updated_at": now,
                "overrides_defaults": overrides_defaults,
            }
        )
        rows: Final = await self.database.query_raw(
            """INSERT INTO "LiteLLM_UsageAlertRule" (id,scope_type,scope_id,enabled,record) VALUES ($1,$2,$3,$4,$5::jsonb)
            ON CONFLICT (id) DO UPDATE SET scope_type=EXCLUDED.scope_type,scope_id=EXCLUDED.scope_id,enabled=EXCLUDED.enabled,record=EXCLUDED.record
            WHERE NOT "LiteLLM_UsageAlertRule".deleted
                AND ("LiteLLM_UsageAlertRule".record->>'updated_at')::timestamptz=$6::text::timestamptz
                AND "LiteLLM_UsageAlertRule".scope_type=$7
                AND "LiteLLM_UsageAlertRule".scope_id IS NOT DISTINCT FROM $8::text
            RETURNING record""",
            rule.id,
            rule.scope_type,
            rule.scope_id,
            rule.enabled,
            json.dumps({**rule.model_dump(mode="json"), "overrides_defaults": overrides_defaults}),
            existing.updated_at.isoformat() if existing else None,
            existing.scope_type if existing else None,
            existing.scope_id if existing else None,
        )
        saved: Final = TypeAdapter(tuple[_Record, ...]).validate_python(rows)
        return _rule(saved[0].record) if saved else None

    async def delete_rule(self, rule_id: str, *, existing: AlertRule | None = None) -> bool:
        return bool(
            await self.database.query_raw(
                """UPDATE "LiteLLM_UsageAlertRule" SET deleted=true,enabled=false
            WHERE id=$1 AND NOT deleted AND ($2::text::timestamptz IS NULL OR (
                (record->>'updated_at')::timestamptz=$2::text::timestamptz
                AND scope_type=$3 AND scope_id IS NOT DISTINCT FROM $4::text)) RETURNING id""",
                rule_id,
                existing.updated_at.isoformat() if existing else None,
                existing.scope_type if existing else None,
                existing.scope_id if existing else None,
            )
        )

    async def list_rules(
        self,
        scope_type: UsageAlertScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        kind: UsageAlertRuleKind = "all",
    ) -> AlertRuleList:
        args: Final = (scope_type, scope_id, json.dumps(tuple(sorted(_DEFAULT_RULE_IDS))), kind)
        where: Final = "NOT deleted AND ($1::text IS NULL OR scope_type=$1) AND ($2::text IS NULL OR scope_id=$2)"
        builtin: Final = "id IN (SELECT jsonb_array_elements_text($3::jsonb))"
        count: Final = TypeAdapter(tuple[_RuleCounts, ...]).validate_python(
            await self.database.query_raw(
                f"""WITH scoped AS (
                    SELECT enabled,record,({builtin}) AS builtin FROM "LiteLLM_UsageAlertRule" WHERE {where}
                ) SELECT count(*) FILTER (WHERE $4='all' OR builtin=($4='default'))::int AS count,
                    jsonb_build_object('total',count(*) FILTER (WHERE builtin),
                        'enabled',count(*) FILTER (WHERE builtin AND enabled),
                        'evaluation_status_counts',jsonb_build_object(
                            'pending',count(*) FILTER (WHERE builtin AND enabled AND COALESCE(record->>'evaluation_status','pending')='pending'),
                            'complete',count(*) FILTER (WHERE builtin AND enabled AND record->>'evaluation_status'='complete'),
                            'insufficient_coverage',count(*) FILTER (WHERE builtin AND enabled AND record->>'evaluation_status'='insufficient_coverage'),
                            'insufficient_history',count(*) FILTER (WHERE builtin AND enabled AND record->>'evaluation_status'='insufficient_history')
                        )) AS default_summary FROM scoped""",
                *args,
            )
        )
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                f"SELECT record FROM \"LiteLLM_UsageAlertRule\" WHERE {where} AND ($4='all' OR ({builtin})=($4='default')) ORDER BY id LIMIT $5 OFFSET $6",
                *args,
                limit,
                offset,
            )
        )
        return AlertRuleList(
            data=tuple(_rule(row.record) for row in rows),
            total=count[0].count,
            limit=limit,
            offset=offset,
            default_summary=count[0].default_summary,
        )

    async def get_alert(self, alert_id: str) -> PerformanceSignal | None:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw('SELECT record FROM "LiteLLM_UsageAlertIncident" WHERE id=$1', alert_id)
        )
        return PerformanceSignal.model_validate(rows[0].record) if rows else None

    async def list_alerts(
        self,
        scope_type: UsageAlertScope | None = None,
        scope_id: str | None = None,
        metric: UsageAlertMetric | None = None,
        status: UsageAlertStatus | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PerformanceSignalList:
        args: Final = (
            scope_type,
            scope_id,
            metric,
            status,
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
        )
        where: Final = "($1::text IS NULL OR scope_type=$1) AND ($2::text IS NULL OR scope_id=$2) AND ($3::text IS NULL OR metric=$3) AND ($4::text IS NULL OR status=$4) AND ($5::text IS NULL OR day >= $5) AND ($6::text IS NULL OR day <= $6)"
        count: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                f'SELECT count(*)::int AS count FROM "LiteLLM_UsageAlertIncident" WHERE {where}', *args
            )
        )
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                f'SELECT record FROM "LiteLLM_UsageAlertIncident" WHERE {where} ORDER BY day DESC,id LIMIT $7 OFFSET $8',
                *args,
                limit,
                offset,
            )
        )
        return PerformanceSignalList(
            data=tuple(PerformanceSignal.model_validate(row.record) for row in rows),
            total=count[0].count,
            limit=limit,
            offset=offset,
        )

    async def set_status(self, alert_id: str, status: UsageAlertStatus) -> PerformanceSignal | None:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                """UPDATE "LiteLLM_UsageAlertIncident" SET status=$2, record=jsonb_set(record,'{status}',to_jsonb($2::text)),updated_at=CURRENT_TIMESTAMP WHERE id=$1 RETURNING record""",
                alert_id,
                status,
            )
        )
        return PerformanceSignal.model_validate(rows[0].record) if rows else None

    async def resource_labels(
        self, groups: tuple[ResourceLabelScope, ...]
    ) -> Mapping[tuple[str, str | None], str | None]:
        rows: Final = TypeAdapter(tuple[_ResourceLabel, ...]).validate_python(
            await self.database.query_raw(
                """WITH scopes AS (
                    SELECT DISTINCT scope_type,scope_id,key_id FROM jsonb_to_recordset($1::jsonb)
                    AS x(scope_type text,scope_id text,key_id text)
                ) SELECT s.scope_type,s.scope_id,LEFT(CASE s.scope_type
                    WHEN 'key' THEN COALESCE(NULLIF(k.key_alias,''),NULLIF(k.key_name,''),
                        NULLIF(d.key_alias,''),NULLIF(d.key_name,''))
                    WHEN 'team' THEN NULLIF(t.team_alias,'')
                    WHEN 'user' THEN COALESCE(NULLIF(u.user_alias,''),NULLIF(u.user_email,''))
                    WHEN 'model' THEN s.scope_id WHEN 'provider' THEN s.scope_id
                    WHEN 'deployment' THEN 'Gateway' END,256) AS label
                FROM scopes s LEFT JOIN "LiteLLM_VerificationToken" k ON s.scope_type='key' AND k.token=s.key_id
                    LEFT JOIN LATERAL (
                        SELECT key_alias,key_name FROM "LiteLLM_DeletedVerificationToken"
                        WHERE s.scope_type='key' AND token=s.key_id AND k.token IS NULL
                        ORDER BY deleted_at DESC,id DESC LIMIT 1
                    ) d ON true
                    LEFT JOIN "LiteLLM_TeamTable" t ON s.scope_type='team' AND t.team_id=s.scope_id
                    LEFT JOIN "LiteLLM_UserTable" u ON s.scope_type='user' AND u.user_id=s.scope_id""",
                json.dumps(
                    tuple(
                        {
                            "scope_type": row.scope_type,
                            "scope_id": row.scope_id,
                            "key_id": api_key_from_bucket(row.scope_id)
                            if row.scope_type == "key" and row.scope_id
                            else None,
                        }
                        for row in groups
                    )
                ),
            )
        )
        return MappingProxyType({(row.scope_type, row.scope_id): row.label for row in rows})

    async def list_reported_models(self, search: str | None, cursor: str | None, limit: int) -> AlertModelList:
        today: Final = datetime.now(timezone.utc).date()
        rows: Final = TypeAdapter(tuple[_ModelIdentity, ...]).validate_python(
            await self.database.query_raw(
                """SELECT DISTINCT model FROM "LiteLLM_DailyUserSpend"
                WHERE date >= $1 AND date <= $2 AND model IS NOT NULL AND length(model) BETWEEN 1 AND 1024
                    AND ($3::text IS NULL OR strpos(lower(model),lower($3))>0)
                    AND ($4::text IS NULL OR model>$4)
                ORDER BY model LIMIT $5""",
                (today - timedelta(days=29)).isoformat(),
                today.isoformat(),
                search,
                cursor,
                limit + 1,
            )
        )
        selected: Final = rows[:limit]
        return AlertModelList(
            data=tuple(AlertModelOption(scope_id=row.model, label=row.model) for row in selected),
            limit=limit,
            next_cursor=selected[-1].model if len(rows) > limit else None,
            start_date=today - timedelta(days=29),
            end_date=today,
        )

    async def key_ownership(self, key_id: str) -> tuple[str | None, str | None] | None:
        rows: Final = TypeAdapter(tuple[_Ownership, ...]).validate_python(
            await self.database.query_raw(
                """SELECT user_id,team_id FROM "LiteLLM_VerificationToken" WHERE token=$1
                UNION ALL
                (SELECT user_id,team_id FROM "LiteLLM_DeletedVerificationToken" WHERE token=$1
                    AND NOT EXISTS(SELECT 1 FROM "LiteLLM_VerificationToken" WHERE token=$1)
                    ORDER BY deleted_at DESC,id DESC LIMIT 1)""",
                key_id,
            )
        )
        return (rows[0].user_id, rows[0].team_id) if rows else None

    async def claim_job(self, owner: str, lease_seconds: int = 900) -> bool:
        rows: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                """INSERT INTO "LiteLLM_UsageAlertCheckpoint" (id,owner,expires_at) VALUES ('worker',$1,CURRENT_TIMESTAMP+($2::int * INTERVAL '1 second'))
            ON CONFLICT(id) DO UPDATE SET owner=EXCLUDED.owner,expires_at=EXCLUDED.expires_at WHERE "LiteLLM_UsageAlertCheckpoint".expires_at<CURRENT_TIMESTAMP OR "LiteLLM_UsageAlertCheckpoint".owner=$1 RETURNING 1 AS count""",
                owner,
                lease_seconds,
            )
        )
        return bool(rows)

    async def release_job(self, owner: str) -> None:
        await self.database.query_raw(
            "DELETE FROM \"LiteLLM_UsageAlertCheckpoint\" WHERE id='worker' AND owner=$1 RETURNING id", owner
        )

    @staticmethod
    def _checkpoint(rule: AlertRule, day: date) -> str:
        return f"{rule.id}:{rule.updated_at.isoformat()}:{day.isoformat()}"

    async def ensure_default_rules(self, rules: tuple[AlertRuleInput, ...]) -> None:
        for rule in rules:
            await self._ensure_default_rule(rule)

    async def _ensure_default_rule(self, rule: AlertRuleInput) -> None:
        rule_id: Final = str(
            uuid5(NAMESPACE_URL, f"litellm:usage-alert:{rule.scope_type}:{rule.metric}:{rule.direction}")
        )
        now: Final = datetime.now(timezone.utc)
        record: Final = AlertRule.model_validate(
            {**rule.model_dump(), "id": rule_id, "created_at": now, "updated_at": now, "created_by": None}
        )
        await self.database.query_raw(
            'INSERT INTO "LiteLLM_UsageAlertRule" (id,scope_type,scope_id,enabled,record) VALUES ($1,$2,$3,$4,$5::jsonb) ON CONFLICT(id) DO NOTHING RETURNING id',
            rule_id,
            rule.scope_type,
            rule.scope_id,
            rule.enabled,
            record.model_dump_json(),
        )

    async def record_evaluation_status(self, rule: AlertRule, day: date, status: UsageAlertEvaluationStatus) -> None:
        await self.database.query_raw(
            """UPDATE "LiteLLM_UsageAlertRule" SET record=record || jsonb_build_object('last_evaluated_day',$2::text,'evaluation_status',$3::text) WHERE id=$1 AND (record->>'updated_at')::timestamptz=$4::timestamptz AND NOT deleted RETURNING id""",
            rule.id,
            day.isoformat(),
            status,
            rule.updated_at.isoformat(),
        )

    async def record_observations(
        self, rule: AlertRule, observations: tuple[UsageAlertObservation, ...]
    ) -> tuple[PerformanceSignal, ...]:
        if not observations:
            return ()
        if len(observations) > 500 or len({item.entity_id for item in observations}) != len(observations):
            raise ValueError("An observation batch must contain at most 500 distinct entities")
        now: Final = datetime.now(timezone.utc)
        records: Final = tuple(_observation_payload(rule, item, now) for item in observations)
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                """
            WITH input AS (SELECT * FROM jsonb_to_recordset($2::jsonb) AS x(scope_key text, breached boolean, day text, record jsonb)),
            prior AS (
                SELECT x.*,p.id AS prior_id,p.evaluation_day,p.evaluation_revision AS prior_revision,
                    p.breached AS prior_breached,p.record AS prior_record
                FROM input x LEFT JOIN LATERAL (
                    SELECT * FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 AND scope_key=x.scope_key
                      AND scope_type=x.record->>'scope_type' AND scope_id IS NOT DISTINCT FROM x.record->>'scope_id'
                      AND metric=x.record->>'metric' AND record->>'direction'=x.record->>'direction'
                    ORDER BY evaluation_day DESC,id DESC LIMIT 1
                ) p ON true
                WHERE EXISTS (SELECT 1 FROM "LiteLLM_UsageAlertRule" WHERE id=$1 AND enabled AND NOT deleted AND (record->>'updated_at')::timestamptz=$3::timestamptz)
            ), updated AS (
                UPDATE "LiteLLM_UsageAlertIncident" i SET
                    evaluation_day=p.day, evaluation_revision=$3::text, breached=p.breached,
                    record=i.record || jsonb_build_object(
                        'last_seen',CASE WHEN p.breached THEN p.record->'last_seen' ELSE i.record->'last_seen' END,
                        'status',CASE WHEN i.status='open' AND NOT p.breached THEN 'resolved' ELSE i.status END),
                    status=CASE WHEN i.status='open' AND NOT p.breached THEN 'resolved' ELSE i.status END,
                    updated_at=CURRENT_TIMESTAMP
                FROM prior p WHERE i.id=p.prior_id
                    AND (p.evaluation_day<p.day OR (p.evaluation_day=p.day AND p.prior_revision IS DISTINCT FROM $3))
                    AND (NOT p.breached OR p.prior_breached)
                RETURNING i.record
            ), inserted AS (
                INSERT INTO "LiteLLM_UsageAlertIncident" (id,rule_id,scope_key,scope_type,scope_id,metric,status,day,evaluation_day,evaluation_revision,breached,record)
                SELECT p.record->>'id',$1,p.scope_key,p.record->>'scope_type',p.record->>'scope_id',p.record->>'metric','open',p.day,p.day,$3,true,p.record
                FROM prior p WHERE p.breached AND (p.prior_id IS NULL OR (NOT p.prior_breached AND
                    (p.evaluation_day<p.day OR (p.evaluation_day=p.day AND p.prior_revision IS DISTINCT FROM $3))))
                ON CONFLICT(id) DO NOTHING RETURNING record
            ) SELECT record FROM updated UNION ALL SELECT record FROM inserted
            """,
                rule.id,
                json.dumps(records),
                rule.updated_at.isoformat(),
            )
        )
        return tuple(PerformanceSignal.model_validate(row.record) for row in rows)

    async def prune(self, retention_days: int = 90, limit: int = 500) -> None:
        await self.database.query_raw(
            """DELETE FROM "LiteLLM_UsageAlertIncident" WHERE id IN (SELECT id FROM "LiteLLM_UsageAlertIncident" WHERE status<>'open' AND updated_at<CURRENT_TIMESTAMP-($1::int * INTERVAL '1 day') ORDER BY updated_at LIMIT $2) RETURNING id""",
            retention_days,
            limit,
        )
        await self.database.query_raw(
            """DELETE FROM "LiteLLM_UsageAlertCheckpoint" WHERE id IN (SELECT id FROM "LiteLLM_UsageAlertCheckpoint" WHERE completed AND expires_at<CURRENT_TIMESTAMP-INTERVAL '45 days' ORDER BY expires_at LIMIT $1) RETURNING id""",
            limit,
        )

    async def team_role(self, team_id: str, user_id: str) -> str | None:
        rows: Final = TypeAdapter(tuple[Mapping[str, str], ...]).validate_python(
            await self.database.query_raw(
                """SELECT member->>'role' AS role FROM "LiteLLM_TeamTable", jsonb_array_elements(members_with_roles) member WHERE team_id=$1 AND member->>'user_id'=$2 LIMIT 1""",
                team_id,
                user_id,
            )
        )
        return rows[0]["role"] if rows else None
