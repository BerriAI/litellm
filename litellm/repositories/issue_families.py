"""Bounded settings for the built-in performance policies."""

import json
from collections import Counter
from datetime import datetime, timezone
from itertools import chain, product
from typing import Final

from pydantic import BaseModel, TypeAdapter

from litellm.repositories.usage_alerts import AlertDatabase
from litellm.types.proxy.issues import (
    AlertFamilyList,
    AlertFamilyPolicy,
    AlertFamilySetting,
    AlertFamilyUpdate,
)
from litellm.types.proxy.usage_alert_defaults import (
    DEFAULT_USAGE_POLICIES,
    DEFAULT_USAGE_RULE_IDS,
    USAGE_ALERT_SCOPES,
    PerformanceIssueFamily,
    default_usage_rule_id,
)
from litellm.types.proxy.usage_alerts import AlertRule, UsageAlertEvaluationCounts


class _RuleRow(BaseModel, frozen=True):
    record: AlertRule


class _Count(BaseModel, frozen=True):
    count: int


class _ProjectionState(BaseModel, frozen=True):
    pending: bool


def _setting(family: PerformanceIssueFamily, rules: tuple[AlertRule, ...]) -> AlertFamilySetting:
    policies: Final = tuple(policy for policy in DEFAULT_USAGE_POLICIES if policy.family == family)
    groups: Final = tuple(
        (policy, tuple(rule for rule in rules if rule.metric == policy.metric and rule.direction == policy.direction))
        for policy in policies
    )
    values: Final = tuple(
        AlertFamilyPolicy(
            direction=policy.direction,
            enabled=group[0].enabled if group and len({rule.enabled for rule in group}) == 1 else None,
            multiplier=group[0].multiplier if group and len({rule.multiplier for rule in group}) == 1 else None,
            min_value=group[0].min_value if group and len({rule.min_value for rule in group}) == 1 else None,
            min_requests=group[0].min_requests if group and len({rule.min_requests for rule in group}) == 1 else None,
            baseline_days=group[0].baseline_days
            if group and len({rule.baseline_days for rule in group}) == 1
            else None,
            min_history_days=group[0].min_history_days
            if group and len({rule.min_history_days for rule in group}) == 1
            else None,
            mixed=len(
                {
                    (r.enabled, r.multiplier, r.min_value, r.min_requests, r.baseline_days, r.min_history_days)
                    for r in group
                }
            )
            > 1,
            total_rules=len(group),
            enabled_rules=sum(rule.enabled for rule in group),
        )
        for policy, group in groups
    )
    family_rules: Final = tuple(chain.from_iterable(group for _, group in groups))
    return AlertFamilySetting(
        family=family,
        enabled=family_rules[0].enabled if family_rules and len({r.enabled for r in family_rules}) == 1 else None,
        total_rules=len(family_rules),
        enabled_rules=sum(rule.enabled for rule in family_rules),
        policies=values,
        evaluation_status_counts=UsageAlertEvaluationCounts.model_validate(
            Counter(rule.evaluation_status for rule in family_rules if rule.enabled)
        ),
    )


async def family_settings(database: AlertDatabase) -> AlertFamilyList:
    rows: Final = TypeAdapter(tuple[_RuleRow, ...]).validate_python(
        await database.query_raw(
            'SELECT record FROM "LiteLLM_UsageAlertRule" WHERE id=ANY($1::text[]) AND NOT deleted',
            list(DEFAULT_USAGE_RULE_IDS),
        )
    )
    projection: Final = TypeAdapter(tuple[_ProjectionState, ...]).validate_python(
        await database.query_raw(
            """SELECT EXISTS(SELECT 1 FROM "LiteLLM_UsageAlertCheckpoint"
                WHERE id LIKE 'performance-dirty:%' AND NOT completed LIMIT 1) AS pending"""
        )
    )
    return AlertFamilyList(
        data=tuple(
            _setting(family, tuple(row.record for row in rows))
            for family in ("spend", "latency", "reliability", "traffic")
        ),
        projection_pending=projection[0].pending,
    )


async def update_family(
    database: AlertDatabase,
    family: PerformanceIssueFamily,
    data: AlertFamilyUpdate,
) -> AlertFamilySetting | None:
    policies: Final = tuple(policy for policy in DEFAULT_USAGE_POLICIES if policy.family == family)
    if any(update.direction not in {policy.direction for policy in policies} for update in data.policies):
        raise ValueError("This family does not support that direction")
    ids: Final = tuple(
        default_usage_rule_id(scope, policy.metric, policy.direction)
        for policy, scope in product(policies, USAGE_ALERT_SCOPES)
    )
    rows: Final = TypeAdapter(tuple[_RuleRow, ...]).validate_python(
        await database.query_raw(
            'SELECT record FROM "LiteLLM_UsageAlertRule" WHERE id=ANY($1::text[]) AND NOT deleted',
            list(ids),
        )
    )
    timestamp: Final = datetime.now(timezone.utc).isoformat()
    updates: Final = {
        policy.direction: policy.model_dump(exclude_none=True, exclude={"direction"}) for policy in data.policies
    }
    records: Final = tuple(
        AlertRule.model_validate(
            {
                **row.record.model_dump(),
                **({"enabled": data.enabled} if data.enabled is not None else {}),
                **updates.get(row.record.direction, {}),
                "updated_at": timestamp,
                "evaluation_status": "pending",
                "last_evaluated_day": None,
            }
        )
        for row in rows
    )
    if not records:
        return None
    changed: Final = TypeAdapter(tuple[_Count, ...]).validate_python(
        await database.query_raw(
            """WITH incoming AS MATERIALIZED (
            SELECT x->'record' AS record,x->>'revision' AS revision FROM jsonb_array_elements($1::jsonb) x
        ), eligible AS MATERIALIZED (
            SELECT r.id FROM "LiteLLM_UsageAlertRule" r JOIN incoming x ON r.id=x.record->>'id'
            WHERE NOT r.deleted AND (r.record->>'updated_at')::timestamptz=x.revision::timestamptz
            ORDER BY r.id FOR UPDATE OF r
        ), changed AS (
            UPDATE "LiteLLM_UsageAlertRule" r SET record=x.record,enabled=(x.record->>'enabled')::boolean
            FROM incoming x WHERE r.id=x.record->>'id' AND NOT r.deleted
                AND (r.record->>'updated_at')::timestamptz=x.revision::timestamptz
                AND (SELECT count(*) FROM eligible)=(SELECT count(*) FROM incoming) RETURNING r.id
        ) SELECT count(*)::int AS count FROM changed""",
            json.dumps(
                tuple(
                    {"record": record.model_dump(mode="json"), "revision": row.record.updated_at.isoformat()}
                    for record, row in zip(records, rows, strict=True)
                )
            ),
        )
    )
    return _setting(family, records) if changed[0].count == len(records) else None
