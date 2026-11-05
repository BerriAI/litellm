import asyncio
import json
from collections.abc import AsyncGenerator
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

from litellm.repositories.issue_activity import IssueActivityRepository
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.usage_alerts import AlertDatabase, UsageAlertsRepository
from litellm.types.proxy.issues import (
    IssueCohort,
    IssueComparison,
    IssueDeliveryInput,
    IssueFamilyPolicyUpdate,
    IssueFamilyUpdate,
    IssueInput,
    IssueList,
    IssueModelPair,
    IssueSignalInput,
    IssueSignalList,
)
from litellm.types.proxy.usage_alert_defaults import default_usage_rule_id, default_usage_rules
from litellm.types.proxy.usage_alerts import UsageAlertObservation, UsageAlertRuleInput


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
            await store.add_signals(
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
            path: Final = f"/issues/{first.id}"
            own: Final = gateway.request("GET", path, key=owner_key)
            assert own.status_code == 200 and own.json()["scope_label"] == "Example checkout"
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
                == 200
            )
            assert (
                gateway.request(
                    "PATCH", path, {"status": "resolved", "expected_status": "open"}, key=owner_key
                ).status_code
                == 409
            )
            assert gateway.request("GET", f"/issues/{second.id}", key=owner_key).json()["status"] == "open"
        finally:
            await _remove_issues(db, (first.id, second.id, global_issue.id))
            await alerts.delete_rule(rule.id)
            await _query(db, 'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id', rule.id)


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
        await store.upsert_issue(issue)
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
        await _query(
            db,
            'UPDATE "LiteLLM_UsageAlertIncident" SET evaluation_revision=$2 WHERE id=$1 RETURNING id',
            source.id,
            rule.updated_at.isoformat(),
        )
        initial: Final = await store.upsert_issue(original)
        assert tuple(
            item.alert.id
            for item in await store.active_signals(
                original.observed_on, scope_type="user", metric="requests", scope_ids=(str(rule.scope_id),)
            )
        ) == (source.id,)
        assert await store.active_signals(original.observed_on, scope_ids=()) == ()
        assert await store.active_signals(original.observed_on, scope_ids=("other-example-user",)) == ()
        await store.add_signals(
            initial.id, (IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),)
        )
        await store.queue_delivery(delivery, rule.id, rule.updated_at.isoformat(), source.id)
        await store.queue_delivery(invalid, rule.id, rule.updated_at.isoformat(), source.id)
        pending: Final = await store.pending_delivery(delivery.id)
        assert pending is not None and pending.alert.id == source.id and pending.issue.id == initial.id
        assert await store.pending_delivery(invalid.id) is None
        disabled: Final = await alerts.save_rule(definition.model_copy(update={"enabled": False}), "qa-admin", rule)
        assert disabled is not None and await store.pending_delivery(delivery.id) is None
        enabled: Final = await alerts.save_rule(definition, "qa-admin", disabled)
        assert enabled is not None and await store.pending_delivery(delivery.id) is None
        await _query(
            db,
            'UPDATE "LiteLLM_UsageAlertIncident" SET evaluation_revision=$2 WHERE id=$1 RETURNING id',
            source.id,
            enabled.updated_at.isoformat(),
        )
        await store.queue_delivery(delivery, enabled.id, enabled.updated_at.isoformat(), source.id)
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
        continued: Final = await store.upsert_issue(later)
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
        await store.queue_delivery(delivery, enabled.id, enabled.updated_at.isoformat(), source.id)
        assert await store.pending_delivery(delivery.id) is None
        healthy: Final = observation.model_copy(
            update={"day": observation.day + timedelta(days=1), "breached": False, "observed": 120}
        )
        await alerts.record_observations(enabled, (healthy,))
        await store.recover_issues(observation.day + timedelta(days=2))
        recovered: Final = await store.get_issue(initial.id)
        assert recovered is not None and not recovered.active and recovered.status == "resolved"
        assert await store.find_active_issue(original.cohort_key) is None
        repeated: Final = (
            await alerts.record_observations(
                enabled, (observation.model_copy(update={"day": observation.day + timedelta(days=2)}),)
            )
        )[0]
        assert repeated.id != source.id
        new_episode: Final = await store.upsert_issue(
            original.model_copy(
                update={
                    "id": next_issue_id,
                    "observed_on": repeated.day,
                    "observed_at": later.observed_at + timedelta(minutes=1),
                }
            )
        )
        assert new_episode.active and new_episode.status == "open" and new_episode.id != initial.id
        assert (await store.find_active_issue(original.cohort_key)) == new_episode
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
            setup.execute(
                sql.SQL('CREATE TABLE {} (LIKE "LiteLLM_UsageAlertRule" INCLUDING ALL)').format(
                    sql.Identifier(schema, "LiteLLM_UsageAlertRule")
                )
            )
            database: Final = Prisma(datasource=DatasourceOverride(url=scoped))
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


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
        await store.upsert_issue(issue)
        await store.add_signals(
            issue.id,
            tuple(
                IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source) for source in sources
            ),
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
        assert await store.recover_issue_ids((inputs[0].id, inputs[1].id), template.observed_at) == 2
        dismissed: Final = await store.get_issue(inputs[0].id)
        resolved: Final = await store.get_issue(inputs[1].id)
        assert dismissed is not None and not dismissed.active and dismissed.status == "dismissed"
        assert resolved is not None and not resolved.active and resolved.status == "resolved"
        await store.recover_issues(template.observed_on + timedelta(days=1))
        remaining: Final = await store.active_provider_issues(1, prefix)
        assert remaining[0].id == inputs[2].id
    finally:
        await _remove_issues(db, tuple(issue.id for issue in inputs))
