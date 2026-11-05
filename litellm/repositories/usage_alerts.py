"""Bounded, writer-consistent storage for daily alert rules and episodes."""

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # PrismaWrapper forwards generated client methods dynamically
)
from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, TypeAdapter

from litellm.types.proxy.usage_alerts import (
    UsageAlert,
    UsageAlertChannel,
    UsageAlertEvaluationStatus,
    UsageAlertList,
    UsageAlertMetric,
    UsageAlertNotification,
    UsageAlertObservation,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertRuleList,
    UsageAlertScope,
    UsageAlertStatus,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class AlertDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...


class _Record(BaseModel, frozen=True):
    record: Mapping[str, object]


class _Count(BaseModel, frozen=True):
    count: int


class _Pending(_Record, frozen=True):
    rule: Mapping[str, object]
    notified_channels: tuple[UsageAlertChannel, ...]


class _Ownership(BaseModel, frozen=True):
    user_id: str | None
    team_id: str | None


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


def _observation_payload(rule: UsageAlertRule, item: UsageAlertObservation, now: datetime) -> Mapping[str, object]:
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
                ]
            ),
        )
    )
    incident: Final = UsageAlert(
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

    async def get_rule(self, rule_id: str) -> UsageAlertRule | None:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                'SELECT record FROM "LiteLLM_UsageAlertRule" WHERE id=$1 AND NOT deleted', rule_id
            )
        )
        return UsageAlertRule.model_validate(rows[0].record) if rows else None

    async def save_rule(
        self, data: UsageAlertRuleInput, created_by: str | None, existing: UsageAlertRule | None = None
    ) -> UsageAlertRule | None:
        now: Final = datetime.now(timezone.utc)
        rule: Final = UsageAlertRule.model_validate(
            {
                **data.model_dump(),
                "id": existing.id if existing else str(uuid4()),
                "created_by": existing.created_by if existing else created_by,
                "created_at": existing.created_at if existing else now,
                "updated_at": now,
            }
        )
        rows: Final = await self.database.query_raw(
            """INSERT INTO "LiteLLM_UsageAlertRule" (id,scope_type,scope_id,enabled,record) VALUES ($1,$2,$3,$4,$5::jsonb)
            ON CONFLICT (id) DO UPDATE SET scope_type=EXCLUDED.scope_type,scope_id=EXCLUDED.scope_id,enabled=EXCLUDED.enabled,record=EXCLUDED.record
            WHERE NOT "LiteLLM_UsageAlertRule".deleted RETURNING record""",
            rule.id,
            rule.scope_type,
            rule.scope_id,
            rule.enabled,
            rule.model_dump_json(),
        )
        saved: Final = TypeAdapter(tuple[_Record, ...]).validate_python(rows)
        return UsageAlertRule.model_validate(saved[0].record) if saved else None

    async def delete_rule(self, rule_id: str) -> None:
        await self.database.query_raw(
            'UPDATE "LiteLLM_UsageAlertRule" SET deleted=true,enabled=false WHERE id=$1 RETURNING id', rule_id
        )

    async def list_rules(
        self, scope_type: UsageAlertScope | None = None, scope_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> UsageAlertRuleList:
        args: Final = (scope_type, scope_id)
        where: Final = "NOT deleted AND ($1::text IS NULL OR scope_type=$1) AND ($2::text IS NULL OR scope_id=$2)"
        count: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                f'SELECT count(*)::int AS count FROM "LiteLLM_UsageAlertRule" WHERE {where}', *args
            )
        )
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                f'SELECT record FROM "LiteLLM_UsageAlertRule" WHERE {where} ORDER BY id LIMIT $3 OFFSET $4',
                *args,
                limit,
                offset,
            )
        )
        return UsageAlertRuleList(
            data=tuple(UsageAlertRule.model_validate(row.record) for row in rows),
            total=count[0].count,
            limit=limit,
            offset=offset,
        )

    async def list_enabled_rules(self, limit: int = 100, offset: int = 0) -> tuple[UsageAlertRule, ...]:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                'SELECT record FROM "LiteLLM_UsageAlertRule" WHERE enabled AND NOT deleted ORDER BY id LIMIT $1 OFFSET $2',
                limit,
                offset,
            )
        )
        return tuple(UsageAlertRule.model_validate(row.record) for row in rows)

    async def get_alert(self, alert_id: str) -> UsageAlert | None:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw('SELECT record FROM "LiteLLM_UsageAlertIncident" WHERE id=$1', alert_id)
        )
        return UsageAlert.model_validate(rows[0].record) if rows else None

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
    ) -> UsageAlertList:
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
        return UsageAlertList(
            data=tuple(UsageAlert.model_validate(row.record) for row in rows),
            total=count[0].count,
            limit=limit,
            offset=offset,
        )

    async def set_status(self, alert_id: str, status: UsageAlertStatus) -> UsageAlert | None:
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                """UPDATE "LiteLLM_UsageAlertIncident" SET status=$2, record=jsonb_set(record,'{status}',to_jsonb($2::text)),updated_at=CURRENT_TIMESTAMP WHERE id=$1 RETURNING record""",
                alert_id,
                status,
            )
        )
        return UsageAlert.model_validate(rows[0].record) if rows else None

    async def key_ownership(self, key_id: str) -> tuple[str | None, str | None] | None:
        rows: Final = TypeAdapter(tuple[_Ownership, ...]).validate_python(
            await self.database.query_raw(
                'SELECT user_id,team_id FROM "LiteLLM_VerificationToken" WHERE token=$1', key_id
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
    def _checkpoint(rule: UsageAlertRule, day: date) -> str:
        return f"{rule.id}:{rule.updated_at.isoformat()}:{day.isoformat()}"

    async def is_day_complete(self, rule: UsageAlertRule, day: date) -> bool:
        rows: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
            await self.database.query_raw(
                'SELECT 1 AS count FROM "LiteLLM_UsageAlertCheckpoint" WHERE id=$1 AND completed',
                self._checkpoint(rule, day),
            )
        )
        return bool(rows)

    async def finish_day(self, rule: UsageAlertRule, day: date) -> None:
        await self.database.query_raw(
            'INSERT INTO "LiteLLM_UsageAlertCheckpoint" (id,completed) VALUES ($1,true) ON CONFLICT(id) DO NOTHING RETURNING id',
            self._checkpoint(rule, day),
        )

    async def ensure_default_rules(self, rules: tuple[UsageAlertRuleInput, ...]) -> None:
        for rule in rules:
            await self._ensure_default_rule(rule)

    async def _ensure_default_rule(self, rule: UsageAlertRuleInput) -> None:
        rule_id: Final = str(
            uuid5(NAMESPACE_URL, f"litellm:usage-alert:{rule.scope_type}:{rule.metric}:{rule.direction}")
        )
        now: Final = datetime.now(timezone.utc)
        record: Final = UsageAlertRule.model_validate(
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

    async def list_breached_entity_ids(self, rule_id: str, limit: int = 500, offset: int = 0) -> tuple[str | None, ...]:
        rows: Final = TypeAdapter(tuple[Mapping[str, str | None], ...]).validate_python(
            await self.database.query_raw(
                """SELECT DISTINCT i.scope_id FROM "LiteLLM_UsageAlertIncident" i JOIN "LiteLLM_UsageAlertRule" r ON r.id=i.rule_id WHERE i.rule_id=$1 AND i.breached AND i.scope_type=r.scope_type AND i.metric=r.record->>'metric' AND i.record->>'direction'=r.record->>'direction' AND (r.scope_id IS NULL OR i.scope_id=r.scope_id) ORDER BY i.scope_id NULLS FIRST LIMIT $2 OFFSET $3""",
                rule_id,
                limit,
                offset,
            )
        )
        return tuple(row["scope_id"] for row in rows)

    async def record_evaluation_status(
        self, rule: UsageAlertRule, day: date, status: UsageAlertEvaluationStatus
    ) -> None:
        await self.database.query_raw(
            """UPDATE "LiteLLM_UsageAlertRule" SET record=record || jsonb_build_object('last_evaluated_day',$2::text,'evaluation_status',$3::text) WHERE id=$1 AND (record->>'updated_at')::timestamptz=$4::timestamptz AND NOT deleted RETURNING id""",
            rule.id,
            day.isoformat(),
            status,
            rule.updated_at.isoformat(),
        )

    async def record_observation(
        self,
        rule: UsageAlertRule,
        entity_id: str | None,
        day: date,
        observed: float,
        baseline: float,
        threshold: float,
        breached: bool,
    ) -> UsageAlert | None:
        observation: Final = UsageAlertObservation(
            entity_id=entity_id, day=day, observed=observed, baseline=baseline, threshold=threshold, breached=breached
        )
        changed: Final = await self.record_observations(rule, (observation,))
        if changed:
            return changed[0]
        rows: Final = TypeAdapter(tuple[_Record, ...]).validate_python(
            await self.database.query_raw(
                """SELECT record FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 AND scope_key=$2 AND scope_type=$3 AND metric=$4 AND record->>'direction'=$5 ORDER BY evaluation_day DESC,id DESC LIMIT 1""",
                rule.id,
                entity_id or "",
                rule.scope_type,
                rule.metric,
                rule.direction,
            )
        )
        return UsageAlert.model_validate(rows[0].record) if rows else None

    async def record_observations(
        self, rule: UsageAlertRule, observations: tuple[UsageAlertObservation, ...]
    ) -> tuple[UsageAlert, ...]:
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
                SELECT x.*,p.id AS prior_id,p.evaluation_day,p.breached AS prior_breached,p.record AS prior_record
                FROM input x LEFT JOIN LATERAL (
                    SELECT * FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 AND scope_key=x.scope_key
                      AND scope_type=x.record->>'scope_type' AND scope_id IS NOT DISTINCT FROM x.record->>'scope_id'
                      AND metric=x.record->>'metric' AND record->>'direction'=x.record->>'direction'
                    ORDER BY evaluation_day DESC,id DESC LIMIT 1
                ) p ON true
                WHERE EXISTS (SELECT 1 FROM "LiteLLM_UsageAlertRule" WHERE id=$1 AND enabled AND NOT deleted AND (record->>'updated_at')::timestamptz=$3::timestamptz)
            ), updated AS (
                UPDATE "LiteLLM_UsageAlertIncident" i SET
                    evaluation_day=p.day, breached=p.breached,
                    day=CASE WHEN p.breached THEN p.day ELSE i.day END,
                    record=(CASE WHEN p.breached THEN p.record || jsonb_build_object(
                        'id',i.id,'first_seen',i.record->'first_seen',
                        'activity_path',replace(p.record->>'activity_path',p.record->>'id',i.id)) ELSE i.record END)
                        || jsonb_build_object('status',CASE WHEN i.status='open' AND NOT p.breached THEN 'resolved' ELSE i.status END),
                    status=CASE WHEN i.status='open' AND NOT p.breached THEN 'resolved' ELSE i.status END,
                    updated_at=CURRENT_TIMESTAMP
                FROM prior p WHERE i.id=p.prior_id AND p.evaluation_day<p.day AND (NOT p.breached OR p.prior_breached)
                RETURNING i.record
            ), inserted AS (
                INSERT INTO "LiteLLM_UsageAlertIncident" (id,rule_id,scope_key,scope_type,scope_id,metric,status,day,evaluation_day,breached,record)
                SELECT p.record->>'id',$1,p.scope_key,p.record->>'scope_type',p.record->>'scope_id',p.record->>'metric','open',p.day,p.day,true,p.record
                FROM prior p WHERE p.breached AND (p.prior_id IS NULL OR (NOT p.prior_breached AND p.evaluation_day<p.day))
                ON CONFLICT(id) DO NOTHING RETURNING record
            ) SELECT record FROM updated UNION ALL SELECT record FROM inserted
            """,
                rule.id,
                json.dumps(records),
                rule.updated_at.isoformat(),
            )
        )
        return tuple(UsageAlert.model_validate(row.record) for row in rows)

    async def pending_notifications(self, limit: int = 100) -> tuple[UsageAlertNotification, ...]:
        rows: Final = TypeAdapter(tuple[_Pending, ...]).validate_python(
            await self.database.query_raw(
                """SELECT i.record,r.record AS rule,i.notified_channels FROM "LiteLLM_UsageAlertIncident" i JOIN "LiteLLM_UsageAlertRule" r ON r.id=i.rule_id
            WHERE i.status='open' AND r.enabled AND NOT r.deleted AND EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(r.record->'notification_channels') channel WHERE NOT (channel=ANY(i.notified_channels)))
            ORDER BY i.updated_at,i.id LIMIT $1""",
                limit,
            )
        )
        return tuple(
            UsageAlertNotification(
                incident=UsageAlert.model_validate(row.record),
                rule=UsageAlertRule.model_validate(row.rule),
                pending_channels=tuple(
                    channel
                    for channel in UsageAlertRule.model_validate(row.rule).notification_channels
                    if channel not in row.notified_channels
                ),
            )
            for row in rows
        )

    async def mark_notified(self, incident_id: str, channel: UsageAlertChannel) -> None:
        await self.database.query_raw(
            'UPDATE "LiteLLM_UsageAlertIncident" SET notified_channels=array_append(notified_channels,$2::text) WHERE id=$1 AND NOT ($2::text=ANY(notified_channels)) RETURNING id',
            incident_id,
            channel,
        )

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
