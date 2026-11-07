import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Final, cast  # noqa: TID251  # Generated Prisma query_raw has narrower overloads
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway
from prisma import Prisma
from prisma.types import DatasourceOverride
from psycopg import sql
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.proxy.management_endpoints.common_daily_activity import daily_activity_repository
from litellm.repositories.daily_activity_repository import DailyActivityQueryDatabase
from litellm.repositories.daily_activity_sql import build_bounded_daily_activity_sql
from litellm.repositories.issue_activity import IssueActivityRepository
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.usage_alerts import AlertDatabase, UsageAlertsRepository
from litellm.types.proxy.issues import (
    IssueActivityResponse,
    IssueCohort,
    IssueComparison,
    IssueDeliveryBinding,
    IssueDeliveryInput,
    IssueFamilyPolicyUpdate,
    IssueFamilyUpdate,
    IssueInput,
    IssueList,
    IssueModelPair,
    IssueSignal,
    IssueSignalInput,
    IssueSignalList,
)
from litellm.types.proxy.usage_alert_defaults import default_usage_rule_id, default_usage_rules
from litellm.types.proxy.usage_alerts import UsageAlert, UsageAlertObservation, UsageAlertRule, UsageAlertRuleInput


def _issue(cohort: IssueCohort) -> IssueInput:
    identity: Final = str(uuid4())
    day: Final = date.today() - timedelta(days=3)
    return IssueInput(
        id=identity,
        cohort_key=identity,
        kind="performance",
        family="traffic",
        title="Requests increased",
        cohort=cohort,
        observed_on=day,
        observed_at=datetime.now(timezone.utc),
        range_start=day - timedelta(days=7),
        range_end=day,
        grouping_reason="same_resource",
        explanation="Requests increased for this resource.",
        comparisons=(
            IssueComparison(
                metric="requests",
                direction="increase",
                observed_value=420,
                baseline_value=120,
                threshold=360,
                baseline_start=day - timedelta(days=7),
                baseline_end=day - timedelta(days=1),
                baseline_sample_days=7,
            ),
        ),
    )


def _signals(issue_id: str, inputs: tuple[IssueSignalInput, ...]) -> tuple[IssueSignal, ...]:
    return tuple(IssueSignal.model_validate({**signal.model_dump(), "issue_id": issue_id}) for signal in inputs)


def _delivery(data: IssueDeliveryInput, rule: UsageAlertRule, source: UsageAlert) -> IssueDeliveryBinding:
    return IssueDeliveryBinding(
        delivery=data, rule_id=rule.id, rule_revision=rule.updated_at.isoformat(), source_id=source.id
    )


async def _query(db: ManagementDatabase, query: str, *args: object) -> object:
    return await IssuesRepository.from_prisma(db.prisma).database.query_raw(query, *args)


async def _remove_issues(db: ManagementDatabase, ids: tuple[str, ...]) -> None:
    await _query(
        db,
        'DELETE FROM "LiteLLM_IssueDeliveryRule" WHERE delivery_id IN '
        '(SELECT id FROM "LiteLLM_IssueDelivery" WHERE issue_id=ANY($1::text[])) RETURNING delivery_id',
        list(ids),
    )
    await _query(db, 'DELETE FROM "LiteLLM_IssueDelivery" WHERE issue_id=ANY($1::text[]) RETURNING id', list(ids))
    await _query(db, 'DELETE FROM "LiteLLM_IssueSignal" WHERE issue_id=ANY($1::text[]) RETURNING source_id', list(ids))
    await _query(db, 'DELETE FROM "LiteLLM_Issue" WHERE id=ANY($1::text[]) RETURNING id', list(ids))


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_http_scope_pagination_and_source_visibility(gateway: Gateway, db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        viewer: Final = scenario.user(user_role="proxy_admin_viewer")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        info_key: Final = scenario.key(user_id=owner, key_type="default", allowed_routes=["info_routes"])
        outsider_key: Final = scenario.key(user_id=outsider, key_type="management")
        viewer_key: Final = scenario.key(user_id=viewer, key_type="read_only")
        inference_key: Final = scenario.key(user_id=owner, key_type="llm_api", key_alias="Example checkout")
        reporting_id: Final = sha256(inference_key.encode()).hexdigest()
        first: Final = _issue(IssueCohort(scope_type="key", scope_id=reporting_id))
        second: Final = _issue(first.cohort)
        global_issue: Final = _issue(IssueCohort(scope_type="provider", scope_id="example-provider"))
        rule: Final = await alerts.save_rule(
            UsageAlertRuleInput(
                name="Visibility fixture",
                scope_type="key",
                scope_id=reporting_id,
                metric="requests",
                direction="increase",
            ),
            owner,
        )
        assert rule is not None
        try:
            await store.upsert_issues((first, second, global_issue))
            source: Final = (
                await alerts.record_observations(
                    rule,
                    (
                        UsageAlertObservation(
                            entity_id=reporting_id,
                            day=first.observed_on,
                            observed=420,
                            baseline=120,
                            threshold=360,
                            breached=True,
                        ),
                    ),
                )
            )[0]
            broader: Final = source.model_copy(
                update={
                    "id": str(uuid4()),
                    "scope_type": "deployment",
                    "scope_id": None,
                    "observed_value": 99999,
                    "baseline_value": 9999,
                }
            )
            await store.add_signal_batch(
                _signals(
                    first.id,
                    (
                        IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),
                        IssueSignalInput(
                            source="usage_alert",
                            source_id=broader.id,
                            usage_alert=broader,
                            grouping_reason="dominant_key",
                            explanation="Parent contribution evidence.",
                        ),
                    ),
                )
            )
            path: Final = f"/issues/{first.id}"
            own: Final = gateway.request("GET", path, key=owner_key)
            assert own.status_code == 200 and own.json()["scope_label"] == "Example checkout"
            assert own.json()["can_update_status"] is False, "Broader source scopes require their own write permission"
            assert own.json()["can_view_activity"] is True
            assert gateway.request("GET", path, key=viewer_key).json()["can_update_status"] is False
            assert gateway.request("GET", path, key=info_key).json()["can_update_status"] is False
            assert (
                gateway.request(
                    "PATCH", path, {"status": "resolved", "expected_status": "open"}, key=info_key
                ).status_code
                == 403
            )
            assert "99999" not in own.text and "9999" not in own.text
            pages: Final = tuple(
                gateway.request(
                    "GET",
                    "/issues",
                    key=owner_key,
                    params={"scope_type": "key", "scope_id": reporting_id, "limit": "1", "offset": str(offset)},
                )
                for offset in (0, 1, 2)
            )
            assert all(page.status_code == 200 for page in pages)
            for offset, parsed in enumerate(IssueList.model_validate(page.json()) for page in pages):
                assert parsed.total == 2 and len(parsed.data) == (1 if offset < 2 else 0)
            assert gateway.request("GET", "/issues", key=owner_key).status_code == 403
            for suffix in ("", "/signals", "/activity", "/keys", "/users", "/impact"):
                assert gateway.request("GET", f"/issues/{global_issue.id}{suffix}", key=owner_key).status_code == 403
                assert gateway.request("GET", path + suffix, key=outsider_key).status_code == 403
                assert gateway.request("GET", path + suffix, key=inference_key).status_code == 403
            public_signals: Final = gateway.request("GET", path + "/signals", key=owner_key, params={"limit": "1"})
            assert public_signals.status_code == 200
            visible: Final = IssueSignalList.model_validate(public_signals.json())
            assert visible.total == 1 and tuple(row.source_id for row in visible.data) == (source.id,)
            assert "99999" not in public_signals.text
            end: Final = gateway.request("GET", path + "/signals", key=owner_key, params={"limit": "1", "offset": "1"})
            assert end.json()["total"] == 1 and end.json()["data"] == []
            admin: Final = gateway.request("GET", path + "/signals", key=viewer_key)
            assert admin.status_code == 200 and admin.json()["total"] == 2 and "99999" in admin.text
            assert (
                gateway.request(
                    "PATCH", path, {"status": "dismissed", "expected_status": "open"}, key=viewer_key
                ).status_code
                == 403
            )
            assert (
                gateway.request(
                    "PATCH", path, {"status": "dismissed", "expected_status": "open"}, key=owner_key
                ).status_code
                == 403
            )
            assert gateway.request("PATCH", path, {"status": "dismissed", "expected_status": "open"}).status_code == 200
            assert gateway.request("PATCH", path, {"status": "resolved", "expected_status": "open"}).status_code == 409
            assert gateway.request("GET", f"/issues/{second.id}", key=owner_key).json()["status"] == "open"
            assert gateway.request("GET", f"/issues/{second.id}", key=owner_key).json()["can_update_status"] is True
            assert (
                gateway.request(
                    "PATCH", f"/issues/{second.id}", {"status": "dismissed", "expected_status": "open"}, key=owner_key
                ).status_code
                == 200
            )
        finally:
            await _remove_issues(db, (first.id, second.id, global_issue.id))
            await alerts.delete_rule(rule.id)
            await _query(db, 'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id', rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_detail_reports_team_status_permission(gateway: Gateway, db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    with gateway.scenario() as scenario:
        admin: Final = scenario.user(user_role="internal_user")
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(
            members_with_roles=[{"user_id": admin, "role": "admin"}, {"user_id": member, "role": "user"}]
        )
        permitted_team: Final = scenario.team(
            members_with_roles=[{"user_id": member, "role": "user"}],
            team_member_permissions=["/team/daily/activity"],
        )
        admin_key: Final = scenario.key(user_id=admin, key_type="management")
        member_key: Final = scenario.key(user_id=member, key_type="management")
        service_key: Final = scenario.key(
            team_id=team,
            user_id=None,
            key_type="management",
            metadata={"service_account_id": "example-activity-service"},
        )
        issue: Final = _issue(IssueCohort(scope_type="team", scope_id=team))
        permitted_issue: Final = _issue(IssueCohort(scope_type="team", scope_id=permitted_team))
        try:
            await store.upsert_issues((issue, permitted_issue))
            path: Final = f"/issues/{issue.id}"
            assert gateway.request("GET", path, key=admin_key).json()["can_update_status"] is True
            assert gateway.request("GET", path, key=admin_key).json()["can_view_activity"] is True
            assert gateway.request("GET", path, key=member_key).json()["can_update_status"] is False
            assert gateway.request("GET", path, key=member_key).json()["can_view_activity"] is False
            for suffix in ("/activity", "/keys", "/users", "/impact"):
                assert gateway.request("GET", path + suffix, key=member_key).status_code == 403
                assert gateway.request("GET", path + suffix, key=service_key).status_code == 403
            for suffix in ("/activity", "/keys", "/impact"):
                assert gateway.request("GET", path + suffix, key=admin_key).status_code == 200
                assert (
                    gateway.request("GET", f"/issues/{permitted_issue.id}" + suffix, key=member_key).status_code == 200
                )
            assert (
                gateway.request("GET", f"/issues/{permitted_issue.id}", key=member_key).json()["can_view_activity"]
                is True
            )
            assert (
                gateway.request(
                    "PATCH", path, {"status": "resolved", "expected_status": "open"}, key=member_key
                ).status_code
                == 403
            )
            assert (
                gateway.request(
                    "PATCH", path, {"status": "resolved", "expected_status": "open"}, key=admin_key
                ).status_code
                == 200
            )
        finally:
            await _remove_issues(db, (issue.id, permitted_issue.id))


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_activity_bounds_models_without_losing_totals(gateway: Gateway, db: ManagementDatabase) -> None:
    from litellm.proxy.management_endpoints.common_daily_activity import get_daily_activity_aggregated
    from litellm.repositories.issue_activity import issue_activity_scope

    repository: Final = IssueActivityRepository.from_prisma(db.prisma)
    store: Final = IssuesRepository.from_prisma(db.prisma)
    identity: Final = str(uuid4())
    key: Final = sha256(identity.encode()).hexdigest()
    provider: Final = f"example-provider-{identity}"
    day: Final = date.today() - timedelta(days=5)
    model_count: Final = 137
    cohort: Final = IssueCohort(scope_type="key", scope_id=key)
    issue: Final = _issue(cohort).model_copy(update={"range_start": day, "range_end": day + timedelta(days=2)})
    await _query(
        db,
        """INSERT INTO "LiteLLM_DailyUserSpend"
        (id,user_id,date,api_key,model,model_group,custom_llm_provider,mcp_namespaced_tool_name,endpoint,
        api_requests,spend,prompt_tokens,completion_tokens,successful_requests,failed_requests,
        total_response_time_ms,timed_requests,updated_at)
        SELECT $1||'-'||m||'-'||d,$1,($2::date+d::int)::text,$3,
            CASE WHEN m=0 THEN 'Other' ELSE 'example-model-'||lpad(m::text,4,'0') END,
            'group-'||m,$4,'tool-'||m,'endpoint-'||m,
            m+2,CASE WHEN m=0 AND d=0 THEN 10000 ELSE m+1 END,m+1,2,m+1,1,(m+1)*200,m+1,CURRENT_TIMESTAMP
        FROM generate_series(0,$5::int-1) m CROSS JOIN generate_series(0,2) d RETURNING id""",
        identity,
        day.isoformat(),
        key,
        provider,
        model_count,
    )
    try:
        await store.upsert_issues((issue,))
        scope: Final = issue_activity_scope(cohort, day, day + timedelta(days=2))
        original: Final = await get_daily_activity_aggregated(repository.activity, scope, api_key_limit=20)
        response: Final = gateway.request("GET", f"/issues/{issue.id}/activity")
        assert response.status_code == 200
        bounded: Final = IssueActivityResponse.model_validate(response.json())
        assert bounded.metadata.model_dump(exclude={"api_key_limit"}) == original.metadata.model_dump(
            exclude={"api_key_limit"}
        )
        assert tuple((row.date, row.metrics) for row in bounded.results) == tuple(
            (row.date, row.metrics) for row in original.results
        )
        models: Final = bounded.model_breakdown
        assert models is not None and models.total_models == model_count and len(models.models) == models.limit == 20
        assert models.models[0].model == "Other" and models.models[0].metrics.spend == 10002
        assert models.other is not None and models.other.model_count == model_count - models.limit
        components: Final = (*tuple(row.metrics for row in models.models), models.other.metrics)
        assert sum(row.spend for row in components) == bounded.metadata.total_spend
        assert sum(row.api_requests for row in components) == bounded.metadata.total_api_requests
        assert sum(row.total_response_time_ms for row in components) == bounded.metadata.total_response_time_ms
        assert all(not row.breakdown.models and not row.breakdown.endpoints for row in bounded.results)
        query: Final = build_bounded_daily_activity_sql(scope, model_limit=100)
        raw: Final = TypeAdapter(tuple[object, ...]).validate_python(
            await repository.database.query_raw(query.sql, *query.params)
        )
        assert len(raw) == 3 + 1 + 100 + 1
        largest: Final = gateway.request("GET", f"/issues/{issue.id}/activity", params={"model_limit": "100"})
        assert largest.status_code == 200
        large_models: Final = IssueActivityResponse.model_validate(largest.json()).model_breakdown
        assert large_models is not None and len(large_models.models) == 100
        assert large_models.other is not None and large_models.other.model_count == model_count - 100
        assert gateway.request("GET", f"/issues/{issue.id}/activity", params={"model_limit": "101"}).status_code == 422
        assert (
            gateway.request(
                "GET",
                f"/issues/{issue.id}/activity",
                params={"start_date": day.isoformat(), "end_date": (day + timedelta(days=366)).isoformat()},
            ).status_code
            == 422
        )
        non_key: Final = await repository.daily_activity(
            IssueCohort(scope_type="provider", scope_id=provider), day, day + timedelta(days=2)
        )
        assert non_key.model_breakdown is None and non_key.metadata == bounded.metadata
        assert len(non_key.results) == 3
        schemas: Final = TypeAdapter(tuple[Mapping[str, str], ...]).validate_python(
            await _query(db, "SELECT current_schema() AS name")
        )
        assert gateway.database_url is not None
        with psycopg.connect(gateway.database_url) as blocker:
            blocker.execute(
                sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(
                    sql.Identifier(schemas[0]["name"], "LiteLLM_DailyUserSpend")
                )
            )
            busy: Final = gateway.request("GET", f"/issues/{issue.id}/activity")
            assert busy.status_code == 503 and busy.headers["Retry-After"] == "5"
        assert gateway.request("GET", f"/issues/{issue.id}/activity").status_code == 200
    finally:
        await _remove_issues(db, (issue.id,))
        await _query(db, 'DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id=$1 RETURNING id', identity)


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_provider_pairs_filter_activity_pages_counts_and_exports(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    repository: Final = IssueActivityRepository.from_prisma(db.prisma)
    run_id: Final = str(uuid4())
    day: Final = date.today() - timedelta(days=3)
    provider: Final = f"example-provider-{run_id}"
    other: Final = f"other-provider-{run_id}"
    key: Final = sha256(run_id.encode()).hexdigest()
    entries: Final = tuple(
        {
            "id": f"{run_id}-{index}",
            "user_id": f"user-{run_id}",
            "date": day.isoformat(),
            "api_key": key,
            "model": model,
            "custom_llm_provider": name,
            "api_requests": requests,
            "spend": spend,
            "successful_requests": requests,
            "total_response_time_ms": requests * latency,
            "timed_requests": requests,
        }
        for index, (name, model, requests, spend, latency) in enumerate(
            (
                (provider, "example-model-a", 100, 10, 200),
                (other, "example-model-a", 1000, 100, 900),
                (provider, "example-model-b", 50, 5, 400),
            )
        )
    )
    await _query(
        db,
        """INSERT INTO "LiteLLM_DailyUserSpend"
        (id,user_id,date,api_key,model,custom_llm_provider,api_requests,spend,successful_requests,total_response_time_ms,timed_requests,updated_at)
        SELECT id,user_id,date,api_key,model,custom_llm_provider,api_requests,spend,successful_requests,total_response_time_ms,timed_requests,CURRENT_TIMESTAMP
        FROM jsonb_to_recordset($1::jsonb) AS x(id text,user_id text,date text,api_key text,model text,
            custom_llm_provider text,api_requests int,spend float,successful_requests int,total_response_time_ms bigint,timed_requests int)
        RETURNING id""",
        json.dumps(entries),
    )
    cohort: Final = IssueCohort(
        scope_type="provider",
        scope_id=provider,
        model_pairs=(IssueModelPair(provider=provider, model="example-model-a"),),
    )
    store: Final = IssuesRepository.from_prisma(db.prisma)
    issue: Final = _issue(cohort).model_copy(update={"range_start": day, "range_end": day})
    try:
        await store.upsert_issues((issue,))
        activity_http: Final = gateway.request("GET", f"/issues/{issue.id}/activity")
        assert activity_http.status_code == 200 and activity_http.json()["metadata"]["total_api_requests"] == 100
        keys_http: Final = gateway.request("GET", f"/issues/{issue.id}/keys", params={"limit": "1"})
        assert keys_http.status_code == 200 and keys_http.json()["api_keys"][0]["metrics"]["api_requests"] == 100
        users_http: Final = gateway.request("GET", f"/issues/{issue.id}/users")
        assert users_http.status_code == 200 and users_http.json()["total_users"] == 1
        impact_http: Final = gateway.request("GET", f"/issues/{issue.id}/impact")
        assert impact_http.status_code == 200 and impact_http.json()["requests"] == 100
        daily: Final = await repository.daily_activity(cohort, day, day)
        assert daily.metadata.total_api_requests == 100 and daily.metadata.total_spend == 10
        assert daily.metadata.total_response_time_ms == 20000 and daily.metadata.total_timed_requests == 100
        keys: Final = await repository.keys(cohort, day, day, 1, 0)
        assert keys.total_api_keys == 1 and keys.api_keys[0].metrics.api_requests == 100
        assert (
            keys.api_keys[0].metrics.total_response_time_ms == 20000 and keys.api_keys[0].metrics.timed_requests == 100
        )
        users: Final = await repository.users(cohort, day, day, 1, 0)
        assert users.total_users == 1 and users.users[0].api_requests == 100 and users.users[0].timed_requests == 100
        impact: Final = await repository.impact(cohort, day, day)
        assert (impact.keys, impact.users, impact.models, impact.requests) == (1, 1, 1, 100)
        pairs: Final = IssueCohort(
            scope_type="deployment",
            model_pairs=(
                IssueModelPair(provider=provider, model="example-model-a"),
                IssueModelPair(provider=other, model="example-model-b"),
            ),
        )
        assert (await repository.daily_activity(pairs, day, day)).metadata.total_api_requests == 100
        injected: Final = IssueCohort(scope_type="provider", scope_id=provider + "' OR true --")
        assert (await repository.impact(injected, day, day)).requests == 0
        from litellm.repositories.issue_activity import issue_activity_scope
        from litellm.types.repositories.daily_activity import ExportType

        exported: Final = tuple(
            [
                row
                async for row in repository.activity.export_rows(
                    issue_activity_scope(cohort, day, day), export_type=ExportType.DAILY
                )
            ]
        )
        assert len(exported) == 1 and exported[0].api_requests == 100
    finally:
        await _remove_issues(db, (issue.id,))
        await _query(
            db,
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE id=ANY($1::text[]) RETURNING id',
            [entry["id"] for entry in entries],
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_delivery_guards_snapshot_freeze_and_delayed_recovery(db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    original: Final = _issue(IssueCohort(scope_type="user", scope_id=f"fixture-{uuid4()}"))
    definition: Final = UsageAlertRuleInput(
        name="Delivery fixture",
        scope_type="user",
        scope_id=original.cohort.scope_id,
        metric="requests",
        direction="increase",
        notification_channels=("slack", "email"),
        email_recipients=("alerts@example.com",),
    )
    rule: Final = await alerts.save_rule(definition, "qa-admin")
    assert rule is not None
    observation: Final = UsageAlertObservation(
        entity_id=rule.scope_id, day=original.observed_on, observed=420, baseline=120, threshold=360, breached=True
    )
    delivery: Final = IssueDeliveryInput(id=str(uuid4()), issue_id=original.id, channel="slack")
    invalid: Final = IssueDeliveryInput(
        id=str(uuid4()), issue_id=original.id, channel="email", email_recipients=("outside@example.com",)
    )
    next_issue_id: Final = str(uuid4())
    try:
        source: Final = (await alerts.record_observations(rule, (observation,)))[0]
        initial: Final = (await store.upsert_issues((original,)))[0]
        assert tuple(
            item.alert.id
            for item in await store.active_signals(
                original.observed_on, scope_type="user", metric="requests", scope_ids=(str(rule.scope_id),)
            )
        ) == (source.id,)
        assert await store.active_signals(original.observed_on, scope_ids=()) == ()
        assert await store.active_signals(original.observed_on, scope_ids=("other-example-user",)) == ()
        await store.add_signal_batch(
            _signals(initial.id, (IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),))
        )
        await store.queue_deliveries((_delivery(delivery, rule, source),))
        await store.queue_deliveries((_delivery(invalid, rule, source),))
        pending: Final = await store.pending_delivery(delivery.id)
        assert pending is not None and pending.alert.id == source.id and pending.issue.id == initial.id
        assert await store.pending_delivery(invalid.id) is None
        disabled: Final = await alerts.save_rule(definition.model_copy(update={"enabled": False}), "qa-admin", rule)
        assert disabled is not None and await store.pending_delivery(delivery.id) is None
        enabled: Final = await alerts.save_rule(definition, "qa-admin", disabled)
        assert enabled is not None and await store.pending_delivery(delivery.id) is None
        reevaluated: Final = await alerts.record_observations(enabled, (observation,))
        assert len(reevaluated) == 1 and reevaluated[0].id == source.id
        await store.queue_deliveries((_delivery(delivery, enabled, source),))
        assert await store.pending_delivery(delivery.id) is not None
        await alerts.set_status(source.id, "dismissed")
        assert await store.pending_delivery(delivery.id) is None
        await alerts.set_status(source.id, "open")
        assert await store.set_status(initial.id, "dismissed", "open") is not None
        assert await store.pending_delivery(delivery.id) is None
        later: Final = original.model_copy(
            update={
                "observed_on": original.observed_on + timedelta(days=1),
                "observed_at": original.observed_at + timedelta(minutes=1),
                "title": "Later comparison",
                "comparisons": (original.comparisons[0].model_copy(update={"observed_value": 999}),),
            }
        )
        continued: Final = (await store.upsert_issues((later,)))[0]
        assert continued.status == "dismissed" and continued.comparisons == initial.comparisons
        assert continued.title == initial.title and continued.observed_on == initial.observed_on
        assert continued.last_seen > initial.last_seen
        assert await store.set_status(initial.id, "open", "dismissed") is not None
        await store.defer_delivery(delivery.id, "delivery_failed")
        assert await store.pending_delivery(delivery.id) is None
        await _query(
            db,
            'UPDATE "LiteLLM_IssueDelivery" SET next_attempt=CURRENT_TIMESTAMP WHERE id=$1 RETURNING id',
            delivery.id,
        )
        retried: Final = await store.pending_delivery(delivery.id)
        assert (
            retried is not None and retried.delivery.attempts == 1 and retried.delivery.last_error == "delivery_failed"
        )
        await store.mark_delivery_sent(delivery.id)
        await store.queue_deliveries((_delivery(delivery, enabled, source),))
        assert await store.pending_delivery(delivery.id) is None
        healthy: Final = observation.model_copy(
            update={"day": observation.day + timedelta(days=1), "breached": False, "observed": 120}
        )
        await alerts.record_observations(enabled, (healthy,))
        await store.recover_issues(observation.day + timedelta(days=2))
        recovered: Final = await store.get_issue(initial.id)
        assert recovered is not None and not recovered.active and recovered.status == "resolved"
        assert await store.find_active_issues((original.cohort_key,)) == ()
        repeated: Final = (
            await alerts.record_observations(
                enabled, (observation.model_copy(update={"day": observation.day + timedelta(days=2)}),)
            )
        )[0]
        assert repeated.id != source.id
        new_episode: Final = (
            await store.upsert_issues(
                (
                    original.model_copy(
                        update={
                            "id": next_issue_id,
                            "observed_on": repeated.day,
                            "observed_at": later.observed_at + timedelta(minutes=1),
                        }
                    ),
                )
            )
        )[0]
        assert new_episode.active and new_episode.status == "open" and new_episode.id != initial.id
        assert await store.find_active_issues((original.cohort_key,)) == (new_episode,)
    finally:
        await _remove_issues(db, (original.id, next_issue_id))
        await alerts.delete_rule(rule.id)
        await _query(db, 'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id', rule.id)


class _Pid(BaseModel, frozen=True):
    pid: int


class _Blocked(BaseModel, frozen=True):
    blocked: bool


@asynccontextmanager
async def _family_database(db: ManagementDatabase) -> AsyncGenerator[Prisma]:
    url: Final = db.gateway.database_url
    assert url is not None
    schema: Final = f"issues_{uuid4().hex}"
    parsed: Final = urlsplit(url)
    scoped: Final = urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            for table in ("LiteLLM_UsageAlertRule", "LiteLLM_UsageAlertCheckpoint", "LiteLLM_Config"):
                setup.execute(
                    sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING ALL)").format(
                        sql.Identifier(schema, table), sql.Identifier(table)
                    )
                )
            database: Final = Prisma(datasource=DatasourceOverride(url=scoped))
            await database.connect()
            try:
                assert await database.query_raw("SELECT current_schema() AS name") == [{"name": schema}]
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.asyncio(loop_scope="session")
async def test_late_default_override_stops_queued_delivery_and_recovers_on_fresh_evidence(
    db: ManagementDatabase,
) -> None:
    async with _family_database(db) as database:
        borrowed: Final = cast(AlertDatabase, database)  # cast-ok: real generated Prisma query delegate
        for table in (
            "LiteLLM_UsageAlertIncident",
            "LiteLLM_Issue",
            "LiteLLM_IssueSignal",
            "LiteLLM_IssueDelivery",
            "LiteLLM_IssueDeliveryRule",
            "LiteLLM_VerificationToken",
            "LiteLLM_TeamTable",
            "LiteLLM_UserTable",
        ):
            await borrowed.query_raw(f'CREATE TABLE "{table}" (LIKE public."{table}" INCLUDING ALL)')
        alerts: Final = UsageAlertsRepository(borrowed)
        store: Final = IssuesRepository(borrowed)
        await alerts.ensure_default_rules(default_usage_rules(("slack",)))
        rule: Final = await alerts.get_rule(default_usage_rule_id("key", "requests", "increase"))
        assert rule is not None
        issues: Final = tuple(_issue(IssueCohort(scope_type="key", scope_id=key)) for key in ("key-a", "key-b"))
        observations: Final = tuple(
            UsageAlertObservation(
                entity_id=issue.cohort.scope_id,
                day=issue.observed_on,
                observed=420,
                baseline=120,
                threshold=360,
                breached=True,
            )
            for issue in issues
        )
        sources: Final = await alerts.record_observations(rule, observations)
        await store.upsert_issues(issues)
        for issue in issues:
            source: UsageAlert = next(source for source in sources if source.scope_id == issue.cohort.scope_id)
            await store.add_signal_batch(
                _signals(issue.id, (IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),))
            )
            await store.queue_deliveries(
                (_delivery(IssueDeliveryInput(id=issue.id, issue_id=issue.id, channel="slack"), rule, source),)
            )
            assert await store.pending_delivery(issue.id) is not None
        definition: Final = UsageAlertRuleInput(
            name="Project A threshold",
            scope_type="key",
            scope_id="key-a",
            metric="requests",
            direction="increase",
            multiplier=100,
        )
        ordinary: Final = await alerts.save_rule(definition, "example-user")
        assert ordinary is not None and await store.pending_delivery(issues[0].id) is not None
        assert await alerts.save_rule(definition, "qa-admin", ordinary, overrides_defaults=True) is not None
        pending: Final = await store.pending_delivery(issues[0].id)
        unaffected: Final = await store.pending_delivery(issues[1].id)
        await store.recover_issues(issues[0].observed_on + timedelta(days=1), (issues[0].id,))
        awaiting_evidence: Final = await store.get_issue(issues[0].id)
        assert awaiting_evidence is not None and awaiting_evidence.active
        await alerts.record_observations(
            rule,
            (
                observations[0].model_copy(
                    update={"day": observations[0].day + timedelta(days=1), "observed": 120, "breached": False}
                ),
            ),
        )
        await store.recover_issues(issues[0].observed_on + timedelta(days=2), (issues[0].id,))
        recovered: Final = await store.get_issue(issues[0].id)
        other: Final = await store.get_issue(issues[1].id)
        assert recovered is not None and other is not None
        assert (pending is None, unaffected is not None, recovered.active, recovered.status, other.active) == (
            True,
            True,
            False,
            "resolved",
            True,
        ), f"overridden delivery eligible={pending is not None}; recovered issue active={recovered.active}"


@pytest.mark.asyncio(loop_scope="session")
async def test_family_settings_preserve_custom_deleted_rules_and_abort_concurrent_edits(db: ManagementDatabase) -> None:
    async with _family_database(db) as database:
        borrowed: Final = cast(  # cast-ok: real generated Prisma client with narrower overloads
            AlertDatabase, database
        )
        alerts: Final = UsageAlertsRepository(borrowed)
        store: Final = IssuesRepository(borrowed)
        await alerts.ensure_default_rules(default_usage_rules())
        custom: Final = await alerts.save_rule(
            UsageAlertRuleInput(
                name="Custom spend", scope_type="key", scope_id="example-key", metric="spend", direction="increase"
            ),
            "qa-admin",
        )
        assert custom is not None
        removed_id: Final = default_usage_rule_id("model", "spend", "increase")
        await alerts.delete_rule(removed_id)
        result: Final = await store.update_family(
            "spend",
            IssueFamilyUpdate(
                enabled=False, policies=(IssueFamilyPolicyUpdate(direction="increase", multiplier=5, min_value=25),)
            ),
        )
        assert result is not None and result.total_rules == 4 and result.enabled_rules == 0
        assert result.policies[0].multiplier == 5 and result.policies[0].min_value == 25
        assert await alerts.get_rule(removed_id) is None
        assert await alerts.get_rule(custom.id) == custom
        baseline: Final = await store.family_settings()
        with pytest.raises(ValidationError, match="min_history_days"):
            await store.update_family(
                "spend",
                IssueFamilyUpdate(
                    policies=(IssueFamilyPolicyUpdate(direction="increase", baseline_days=1, min_history_days=7),)
                ),
            )
        assert await store.family_settings() == baseline
        ids: Final = tuple(
            sorted(default_usage_rule_id(scope, "spend", "increase") for scope in ("deployment", "key", "team", "user"))
        )
        async with asyncio.TaskGroup() as tasks:
            async with database.tx(timeout=timedelta(seconds=30)) as transaction:
                await transaction.query_raw('SELECT id FROM "LiteLLM_UsageAlertRule" WHERE id=$1 FOR UPDATE', ids[0])
                pid: Final = (
                    TypeAdapter(tuple[_Pid, ...])
                    .validate_python(await transaction.query_raw("SELECT pg_backend_pid() AS pid"))[0]
                    .pid
                )
                update: Final = tasks.create_task(store.update_family("spend", IssueFamilyUpdate(enabled=True)))
                async with asyncio.timeout(10):
                    while (
                        not TypeAdapter(tuple[_Blocked, ...])
                        .validate_python(
                            await database.query_raw(
                                "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE $1::int=ANY(pg_blocking_pids(pid))) AS blocked",
                                pid,
                            )
                        )[0]
                        .blocked
                    ):
                        await asyncio.sleep(0.01)
                await transaction.query_raw(
                    """UPDATE "LiteLLM_UsageAlertRule" SET record=record ||
                    jsonb_build_object('name','Concurrent edit','updated_at',CURRENT_TIMESTAMP)
                    WHERE id=$1 RETURNING id""",
                    ids[0],
                )
        assert update.result() is None
        final: Final = await store.family_settings()
        spend: Final = next(family for family in final.data if family.family == "spend")
        assert spend.total_rules == 4 and spend.enabled_rules == 0
        edited: Final = await alerts.get_rule(ids[0])
        assert edited is not None and edited.name == "Concurrent edit"
        assert await alerts.get_rule(custom.id) == custom


@pytest.mark.asyncio(loop_scope="session")
async def test_family_edit_requires_a_new_evaluation(db: ManagementDatabase) -> None:
    async with _family_database(db) as database:
        borrowed: Final = cast(AlertDatabase, database)  # cast-ok: real generated Prisma query delegate
        alerts: Final = UsageAlertsRepository(borrowed)
        store: Final = IssuesRepository(borrowed)
        await alerts.ensure_default_rules(default_usage_rules())
        rule: Final = await alerts.get_rule(default_usage_rule_id("key", "spend", "increase"))
        assert rule is not None
        await alerts.record_evaluation_status(rule, date.today() - timedelta(days=1), "complete")
        result: Final = await store.update_family(
            "spend", IssueFamilyUpdate(policies=(IssueFamilyPolicyUpdate(direction="increase", multiplier=5),))
        )
        saved: Final = await alerts.get_rule(rule.id)
        assert result is not None and saved is not None
        assert (result.evaluation_status_counts.complete, saved.evaluation_status, saved.last_evaluated_day) == (
            0,
            "pending",
            None,
        ), "Changed family policy must reset stale evaluation status"


@pytest.mark.asyncio(loop_scope="session")
async def test_deleted_key_owner_retains_historical_issue_access(gateway: Gateway, db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        outsider_key: Final = scenario.key(user_id=outsider, key_type="management")
        reporting_id: Final = sha256(uuid4().bytes).hexdigest()
        issue: Final = _issue(IssueCohort(scope_type="key", scope_id=reporting_id))
        await _query(
            db,
            'INSERT INTO "LiteLLM_DeletedVerificationToken" (id,token,user_id,models) VALUES ($1,$1,$2,ARRAY[]::text[])',
            reporting_id,
            owner,
        )
        try:
            await store.upsert_issues((issue,))
            path: Final = f"/issues/{issue.id}"
            detail: Final = gateway.request("GET", path, key=owner_key)
            assert detail.status_code == 200, "Deleted key owner must retain historical issue access"
            assert detail.json()["can_update_status"] is True
            assert gateway.request("GET", path, key=outsider_key).status_code == 403
            assert (
                gateway.request(
                    "PATCH", path, {"status": "dismissed", "expected_status": "open"}, key=owner_key
                ).status_code
                == 200
            )
            await _query(
                db,
                'INSERT INTO "LiteLLM_VerificationToken" (token,user_id,models) VALUES ($1,$2,ARRAY[]::text[])',
                reporting_id,
                outsider,
            )
            assert gateway.request("GET", path, key=owner_key).status_code == 403
            assert gateway.request("GET", path, key=outsider_key).status_code == 200
        finally:
            await _remove_issues(db, (issue.id,))
            await _query(db, 'DELETE FROM "LiteLLM_VerificationToken" WHERE token=$1', reporting_id)
            await _query(db, 'DELETE FROM "LiteLLM_DeletedVerificationToken" WHERE token=$1', reporting_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_status_rechecks_changed_scope_set_and_caps_sources(db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    issue: Final = _issue(IssueCohort(scope_type="user", scope_id=f"fixture-{uuid4()}"))
    rule: Final = await alerts.save_rule(
        UsageAlertRuleInput(
            name="Scope guard",
            scope_type="user",
            scope_id=issue.cohort.scope_id,
            metric="requests",
            direction="increase",
        ),
        "qa-admin",
    )
    assert rule is not None
    try:
        source: Final = (
            await alerts.record_observations(
                rule,
                (
                    UsageAlertObservation(
                        entity_id=rule.scope_id,
                        day=issue.observed_on,
                        observed=420,
                        baseline=120,
                        threshold=360,
                        breached=True,
                    ),
                ),
            )
        )[0]
        await store.upsert_issues((issue,))
        await store.add_signal_batch(
            _signals(issue.id, (IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),))
        )
        authorized: Final = await store.status_scopes(issue.id)
        assert authorized == (issue.cohort,)
        assert await store.set_status(issue.id, "dismissed", "open", authorized_scopes=authorized) is not None
        context: Final = source.model_copy(update={"id": str(uuid4()), "metric": "spend"})
        same_scope: Final = (IssueSignalInput(source="usage_alert", source_id=context.id, usage_alert=context),)
        await store.add_signal_batch(_signals(issue.id, same_scope))
        unchanged: Final = await store.get_issue(issue.id)
        assert unchanged is not None and unchanged.status == "dismissed"
        await store.add_signal_batch(_signals(issue.id, same_scope))
        assert await store.get_issue(issue.id) == unchanged
        broader: Final = source.model_copy(update={"id": str(uuid4()), "scope_type": "deployment", "scope_id": None})
        await store.add_signal_batch(
            _signals(issue.id, (IssueSignalInput(source="usage_alert", source_id=broader.id, usage_alert=broader),))
        )
        assert await store.set_status(issue.id, "dismissed", "open", authorized_scopes=authorized) is None
        current: Final = await store.get_issue(issue.id)
        assert current is not None and current.status == "open"
        all_scopes: Final = await store.status_scopes(issue.id)
        assert await store.set_status(issue.id, "dismissed", "open", authorized_scopes=all_scopes) is not None
        broader_context: Final = broader.model_copy(update={"id": str(uuid4()), "metric": "spend"})
        await store.add_signal_batch(
            _signals(
                issue.id,
                (IssueSignalInput(source="usage_alert", source_id=broader_context.id, usage_alert=broader_context),),
            )
        )
        repeated_scope: Final = await store.get_issue(issue.id)
        assert repeated_scope is not None and repeated_scope.status == "dismissed"
        assert await store.set_status(issue.id, "open", "dismissed") is not None
        racing: Final = source.model_copy(update={"id": str(uuid4()), "scope_type": "team", "scope_id": "racing-team"})
        async with asyncio.TaskGroup() as tasks:
            async with db.prisma.writer_db.tx(timeout=timedelta(seconds=30)) as transaction:
                await transaction.query_raw('SELECT id FROM "LiteLLM_Issue" WHERE id=$1 FOR UPDATE', issue.id)
                pid: Final = (
                    TypeAdapter(tuple[_Pid, ...])
                    .validate_python(await transaction.query_raw("SELECT pg_backend_pid() AS pid"))[0]
                    .pid
                )
                update: Final = tasks.create_task(
                    store.set_status(issue.id, "dismissed", "open", authorized_scopes=all_scopes)
                )
                async with asyncio.timeout(10):
                    while (
                        not TypeAdapter(tuple[_Blocked, ...])
                        .validate_python(
                            await _query(
                                db,
                                "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE $1::int=ANY(pg_blocking_pids(pid))) AS blocked",
                                pid,
                            )
                        )[0]
                        .blocked
                    ):
                        await asyncio.sleep(0.01)
                locked: Final = IssuesRepository(
                    cast(AlertDatabase, transaction)  # cast-ok: generated Prisma transaction query delegate
                )
                await locked.add_signal_batch(
                    _signals(
                        issue.id, (IssueSignalInput(source="usage_alert", source_id=racing.id, usage_alert=racing),)
                    )
                )
        assert update.result() is None
        raced: Final = await store.get_issue(issue.id)
        assert raced is not None and raced.status == "open"
        await store.add_signal_batch(
            _signals(
                issue.id,
                tuple(
                    IssueSignalInput(
                        source="usage_alert",
                        source_id=f"scope-{index}",
                        usage_alert=source.model_copy(
                            update={"id": f"scope-{index}", "scope_id": f"other-user-{index}"}
                        ),
                    )
                    for index in range(20)
                ),
            )
        )
        limited: Final = await store.status_scopes(issue.id)
        assert len(limited) == 11
        with pytest.raises(ValueError, match="at most 10"):
            await store.set_status(issue.id, "open", "dismissed", authorized_scopes=limited)
    finally:
        await _remove_issues(db, (issue.id,))
        await alerts.delete_rule(rule.id)
        await _query(db, 'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1', rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_resource_recovery_ignores_broader_and_traffic_context(db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    fixture: Final = _issue(IssueCohort(scope_type="user", scope_id=f"fixture-{uuid4()}"))
    issue: Final = fixture.model_copy(
        update={"family": "spend", "comparisons": (fixture.comparisons[0].model_copy(update={"metric": "spend"}),)}
    )
    definitions: Final = (
        UsageAlertRuleInput(
            name="Primary cost", scope_type="user", scope_id=issue.cohort.scope_id, metric="spend", direction="increase"
        ),
        UsageAlertRuleInput(
            name="Traffic context",
            scope_type="user",
            scope_id=issue.cohort.scope_id,
            metric="requests",
            direction="increase",
        ),
        UsageAlertRuleInput(name="Broader cost", scope_type="deployment", metric="spend", direction="increase"),
    )
    rules: Final = tuple([await alerts.save_rule(definition, "qa-admin") for definition in definitions])
    assert all(rule is not None for rule in rules)
    try:
        sources: Final = tuple(
            [
                (
                    await alerts.record_observations(
                        rule,
                        (
                            UsageAlertObservation(
                                entity_id=rule.scope_id,
                                day=issue.observed_on,
                                observed=420,
                                baseline=120,
                                threshold=360,
                                breached=True,
                            ),
                        ),
                    )
                )[0]
                for rule in rules
                if rule is not None
            ]
        )
        await store.upsert_issues((issue,))
        await store.add_signal_batch(
            _signals(
                issue.id,
                tuple(
                    IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source)
                    for source in sources
                ),
            )
        )
        primary: Final = rules[0]
        assert primary is not None
        await alerts.record_observations(
            primary,
            (
                UsageAlertObservation(
                    entity_id=primary.scope_id,
                    day=issue.observed_on + timedelta(days=1),
                    observed=120,
                    baseline=120,
                    threshold=360,
                    breached=False,
                ),
            ),
        )
        await store.recover_issues(issue.observed_on + timedelta(days=2))
        recovered: Final = await store.get_issue(issue.id)
        assert recovered is not None and recovered.status == "resolved" and not recovered.active
        supporting: Final = tuple([await alerts.get_alert(source.id) for source in sources[1:]])
        assert all(source is not None and source.status == "open" for source in supporting)
    finally:
        await _remove_issues(db, (issue.id,))
        for rule in rules:
            if rule is not None:
                await alerts.delete_rule(rule.id)
                await _query(db, 'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id', rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_provider_recovery_pages_saved_cohorts_and_rejects_stale_observations(db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    prefix: Final = f"fixture-{uuid4()}"
    initial: Final = _issue(IssueCohort(scope_type="provider", scope_id="example-provider"))
    template: Final = initial.model_copy(update={"observed_at": initial.observed_at.replace(microsecond=123789)})
    inputs: Final = tuple(
        template.model_copy(update={"id": prefix + suffix, "cohort_key": prefix + suffix})
        for suffix in ("-a", "-b", "-c")
    )
    try:
        await store.upsert_issues(inputs)
        first: Final = await store.active_provider_issues(1, prefix)
        second: Final = await store.active_provider_issues(1, first[0].id)
        assert first[0].id == inputs[0].id and second[0].id == inputs[1].id
        assert await store.set_status(inputs[0].id, "dismissed", "open") is not None
        assert await store.recover_issue_ids((inputs[0].id,), template.observed_at - timedelta(seconds=1)) == 0
        assert await store.recover_issue_ids((inputs[0].id, inputs[1].id), template.observed_at) == 0, (
            "A recovery at the last observed timestamp must not close the alert"
        )
        assert (
            await store.recover_issue_ids(
                (inputs[0].id, inputs[1].id), template.observed_at + timedelta(milliseconds=1)
            )
            == 2
        )
        dismissed: Final = await store.get_issue(inputs[0].id)
        resolved: Final = await store.get_issue(inputs[1].id)
        assert dismissed is not None and not dismissed.active and dismissed.status == "dismissed"
        assert resolved is not None and not resolved.active and resolved.status == "resolved"
        await store.recover_issues(template.observed_on + timedelta(days=1))
        remaining: Final = await store.active_provider_issues(1, prefix)
        assert remaining[0].id == inputs[2].id
    finally:
        await _remove_issues(db, tuple(issue.id for issue in inputs))


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_prune_removes_expired_children_in_bounded_passes(
    db: ManagementDatabase,
) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    expired: Final = _issue(IssueCohort(scope_type="user", scope_id=f"fixture-{uuid4()}"))
    open_issue: Final = _issue(expired.cohort)
    recent: Final = _issue(expired.cohort)
    rule: Final = await alerts.save_rule(
        UsageAlertRuleInput(
            name="Prune fixture",
            scope_type="user",
            scope_id=expired.cohort.scope_id,
            metric="requests",
            direction="increase",
        ),
        "qa-admin",
    )
    assert rule is not None
    deliveries: Final = tuple(
        IssueDeliveryInput(id=str(uuid4()), issue_id=expired.id, channel="slack") for _ in range(2)
    )
    try:
        await store.upsert_issues((expired, open_issue, recent))
        source: Final = (
            await alerts.record_observations(
                rule,
                (
                    UsageAlertObservation(
                        entity_id=rule.scope_id,
                        day=expired.observed_on,
                        observed=420,
                        baseline=120,
                        threshold=360,
                        breached=True,
                    ),
                ),
            )
        )[0]
        await store.add_signal_batch(
            _signals(expired.id, (IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),))
        )
        for delivery in deliveries:
            await store.queue_deliveries((_delivery(delivery, rule, source),))
        assert await store.set_status(expired.id, "resolved", "open") is not None
        assert await store.set_status(recent.id, "resolved", "open") is not None
        await _query(
            db,
            "UPDATE \"LiteLLM_Issue\" SET updated_at=TIMESTAMP '2000-01-01' WHERE id=ANY($1::text[]) RETURNING id",
            [expired.id, open_issue.id],
        )
        await store.prune(retention_days=90, limit=1)
        assert await store.get_issue(expired.id) is not None
        for _ in range(7):
            await store.prune(retention_days=90, limit=1)
        assert await store.get_issue(expired.id) is None
        assert await store.get_issue(open_issue.id) is not None
        assert await store.get_issue(recent.id) is not None
        assert (await store.list_signals(expired.id)).total == 0
        assert (
            await _query(
                db,
                'SELECT id FROM "LiteLLM_IssueDelivery" WHERE issue_id=$1',
                expired.id,
            )
            == []
        )
        assert (
            await _query(
                db,
                'SELECT delivery_id FROM "LiteLLM_IssueDeliveryRule" WHERE delivery_id=ANY($1::text[])',
                [delivery.id for delivery in deliveries],
            )
            == []
        )
    finally:
        await _remove_issues(db, (expired.id, open_issue.id, recent.id))
        await alerts.delete_rule(rule.id)
        await _query(
            db,
            'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id',
            rule.id,
        )


class _MeasuredActivityDatabase:
    def __init__(self, database: DailyActivityQueryDatabase) -> None:
        self.database = database
        self.outputs: tuple[tuple[int, int], ...] = ()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]] | None:
        result: Final = await self.database.query_raw(query, *args)
        self.outputs = (*self.outputs, (len(result or ()), len(json.dumps(result, default=str))))
        return result


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_resource_metadata_bounds_labels_and_deleted_history(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    activity: Final = IssueActivityRepository.from_prisma(db.prisma)
    measured: Final = _MeasuredActivityDatabase(activity.database)
    repository: Final = IssueActivityRepository(daily_activity_repository(db.prisma, query_database=measured), measured)
    identity: Final = uuid4().hex
    reporting_key: Final = sha256(identity.encode()).hexdigest()
    day: Final = date.today() - timedelta(days=3)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        issue: Final = _issue(IssueCohort(scope_type="provider", scope_id=identity)).model_copy(
            update={"range_start": day, "range_end": day}
        )
        try:
            await _query(
                db,
                """UPDATE "LiteLLM_UserTable" SET user_alias=repeat('u',1024),
                user_email=repeat('m',400)||'@example.com' WHERE user_id=$1""",
                owner,
            )
            await _query(
                db,
                """INSERT INTO "LiteLLM_DeletedVerificationToken"
                (id,token,key_alias,user_id,models,metadata,deleted_at)
                SELECT $1::text||'-'||n,$2::text,'archived-'||n||'-'||repeat('x',1024),$3::text,
                    ARRAY[]::text[],jsonb_build_object('padding',repeat('x',65536)),
                    CURRENT_TIMESTAMP+n*INTERVAL '1 second' FROM generate_series(1,400) n""",
                identity,
                reporting_key,
                owner,
            )
            await _query(
                db,
                """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,api_requests,successful_requests,updated_at)
                VALUES($1,$2,$3,$4,'example-model',$1,10,10,CURRENT_TIMESTAMP),
                    ($1||'-wide',repeat(chr(128640),8192),$3,repeat(chr(128640),8192),
                     'example-model',$1,20,20,CURRENT_TIMESTAMP)""",
                identity,
                owner,
                day.isoformat(),
                reporting_key,
            )
            await store.upsert_issues((issue,))
            keys: Final = await repository.keys(issue.cohort, day, day, 1, 0)
            assert keys.total_api_keys == 1 and len(keys.api_keys) == 1
            key: Final = keys.api_keys[0]
            assert key.api_key == reporting_key and key.metadata.key_exists is False
            assert key.metadata.key_alias is not None and key.metadata.key_alias.startswith("archived-400-")
            assert len(key.metadata.key_alias) == 256 and key.metadata.user_id == owner
            assert key.metadata.user_email is None
            assert measured.outputs[-1][0] == 1 and measured.outputs[-1][1] < 2048
            users: Final = await repository.users(issue.cohort, day, day, 1, 0)
            assert users.total_users == 1 and len(users.users) == 1
            assert users.users[0].user_id == owner and users.users[0].user_alias == "u" * 256
            assert users.users[0].user_email is None
            assert measured.outputs[-1][0] == 1 and measured.outputs[-1][1] < 2048
            assert all(size < 2048 for _, size in measured.outputs)
            totals: Final = await repository.daily_activity(issue.cohort, day, day)
            assert totals.metadata.total_api_requests == 30
            for endpoint in ("keys", "users"):
                response: Final = gateway.request("GET", f"/issues/{issue.id}/{endpoint}", params={"limit": "1"})
                assert response.status_code == 200 and len(response.content) < 2048
                assert "padding" not in response.text
            await _query(
                db,
                """UPDATE "LiteLLM_DeletedVerificationToken"
                SET user_id=repeat(chr(128640),8192),team_id=repeat(chr(128640),8192)
                WHERE id=$1||'-400'""",
                identity,
            )
            legacy: Final = await repository.keys(issue.cohort, day, day, 1, 0)
            assert legacy.api_keys[0].metadata.user_id is None and legacy.api_keys[0].metadata.team_id is None
            assert all(size < 2048 for _, size in measured.outputs[-2:])
        finally:
            await _remove_issues(db, (issue.id,))
            await _query(db, "DELETE FROM \"LiteLLM_DailyUserSpend\" WHERE id IN ($1,$1||'-wide')", identity)
            await _query(db, 'DELETE FROM "LiteLLM_DeletedVerificationToken" WHERE token=$1', reporting_key)


@pytest.mark.asyncio(loop_scope="session")
async def test_disabled_later_breach_requires_new_healthy_primary_evidence(db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    issue: Final = _issue(IssueCohort(scope_type="user", scope_id=f"fixture-{uuid4()}"))
    definition: Final = UsageAlertRuleInput(
        name="Primary traffic",
        scope_type="user",
        scope_id=issue.cohort.scope_id,
        metric="requests",
        direction="increase",
    )
    first: Final = await alerts.save_rule(definition, "qa-admin")
    second: Final = await alerts.save_rule(definition.model_copy(update={"name": "Another traffic policy"}), "qa-admin")
    assert first is not None and second is not None
    breach: Final = UsageAlertObservation(
        entity_id=issue.cohort.scope_id,
        day=issue.observed_on,
        observed=420,
        baseline=120,
        threshold=360,
        breached=True,
    )
    try:
        sources: Final = tuple([(await alerts.record_observations(rule, (breach,)))[0] for rule in (first, second)])
        await store.upsert_issues((issue,))
        await store.add_signal_batch(
            _signals(
                issue.id,
                tuple(
                    IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source)
                    for source in sources
                ),
            )
        )
        await alerts.record_observations(
            first,
            (
                breach.model_copy(
                    update={
                        "day": issue.observed_on + timedelta(days=1),
                        "observed": 120,
                        "breached": False,
                    }
                ),
            ),
        )
        await alerts.record_observations(
            second,
            (
                breach.model_copy(
                    update={
                        "day": issue.observed_on + timedelta(days=2),
                    }
                ),
            ),
        )
        assert await alerts.save_rule(definition.model_copy(update={"enabled": False}), "qa-admin", second)
        await store.recover_issues(issue.observed_on + timedelta(days=3))
        unresolved: Final = await store.get_issue(issue.id)
        assert unresolved is not None and unresolved.active
        await alerts.record_observations(
            first,
            (
                breach.model_copy(
                    update={
                        "day": issue.observed_on + timedelta(days=3),
                        "observed": 120,
                        "breached": False,
                    }
                ),
            ),
        )
        await store.recover_issues(issue.observed_on + timedelta(days=3))
        recovered: Final = await store.get_issue(issue.id)
        assert recovered is not None and not recovered.active
    finally:
        await _remove_issues(db, (issue.id,))
        for rule in (first, second):
            await alerts.delete_rule(rule.id)
            await _query(db, 'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id', rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_family_settings_expose_projection_backlog_without_delayed_rules(db: ManagementDatabase) -> None:
    async with _family_database(db) as database:
        borrowed: Final = cast(AlertDatabase, database)  # cast-ok: real generated Prisma query delegate
        store: Final = IssuesRepository(borrowed)
        assert not (await store.family_settings()).projection_pending
        await borrowed.query_raw(
            """INSERT INTO "LiteLLM_UsageAlertCheckpoint" (id,completed,owner)
                VALUES ('performance-dirty:2026-11-04',false,'fixture') RETURNING id"""
        )
        pending: Final = await store.family_settings()
        assert pending.projection_pending and all(item.total_rules == 0 for item in pending.data)
        await borrowed.query_raw(
            """UPDATE "LiteLLM_UsageAlertCheckpoint" SET completed=true
                WHERE id='performance-dirty:2026-11-04' RETURNING id"""
        )
        assert not (await store.family_settings()).projection_pending
