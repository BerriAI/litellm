import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import pytest
from integration._support.actors import ManagementDatabase
from integration._support.prisma_clients import connected_proxy_prisma

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.usage_alerts import UsageAlertDelivery
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.spend_tracking.usage_alert_job import UsageAlertJob
from litellm.proxy.utils import PrismaClient
from litellm.repositories.daily_activity_buckets import api_key_bucket
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.usage_alert_metrics import IssueMetricWindow, UsageAlertMetricsRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import (
    AlertRule,
    AlertRuleInput,
    UsageAlertObservation,
    UsageAlertProjectionWork,
    UsageAlertScope,
)

_DAY: Final = date(2026, 11, 4)
_NOW: Final = datetime(2026, 11, 5, 3, microsecond=987654, tzinfo=timezone.utc)
_KEY_A: Final = sha256(b"synthetic-project-a").hexdigest()
_KEY_B: Final = sha256(b"synthetic-project-b").hexdigest()
_TABLES: Final = (
    "LiteLLM_UsageAlertRule",
    "LiteLLM_UsageAlertIncident",
    "LiteLLM_UsageAlertCheckpoint",
    "LiteLLM_UsageAlertFlush",
    "LiteLLM_Issue",
    "LiteLLM_IssueSignal",
    "LiteLLM_IssueDelivery",
    "LiteLLM_IssueDeliveryRule",
    "LiteLLM_DailyUserSpend",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_Config",
    "LiteLLM_VerificationToken",
    "LiteLLM_DeletedVerificationToken",
    "LiteLLM_UserTable",
    "LiteLLM_TeamTable",
    "LiteLLM_SpendLogDelivery",
)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("scope", ("user", "team", "key", "model"))
async def test_metric_pages_skip_oversized_ids_without_losing_valid_later_entities(
    db: ManagementDatabase, scope: UsageAlertScope
) -> None:
    identities = (
        "a" * 1025,
        "\U0001f600" * 1025,
        "raw:" + "x" * 1016,
        "raw:" + "x" * 1017,
        "oversized-" + "x" * 65536,
        "oversized-" + "\U0001f600" * 8192,
        *(f"valid-{index:02}" for index in range(10)),
        "\U0001f600" * 1024,
    )
    expected = tuple(
        sorted(
            api_key_bucket(value) if scope == "key" else value
            for value in identities
            if len(api_key_bucket(value) if scope == "key" else value) <= 1024
        )
    )
    async with _isolated(db) as prisma:
        for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
            await prisma.writer_db.query_raw(
                f"""INSERT INTO "{table}" (id,{entity},date,api_key,model,spend,api_requests,updated_at)
                SELECT $1||n,identity,$2,identity,identity,2,3,CURRENT_TIMESTAMP
                FROM unnest($3::text[]) WITH ORDINALITY AS items(identity,n) RETURNING id""",
                table,
                _DAY.isoformat(),
                identities,
            )
        metrics = UsageAlertMetricsRepository(prisma.writer_db)
        rows = []
        cursor = None
        for _ in range(3):
            page = await metrics.page(scope, _DAY, 1, cursor=cursor)
            rows.extend(page.rows)
            if page.complete:
                break
            cursor = page.cursor
        else:
            pytest.fail("bounded metric pagination did not finish")
        assert tuple(sorted(row.entity_id for row in rows)) == expected
        assert all((row.spend, row.requests) == (2, 3) for row in rows)
        assert "\U0001f600" * 1024 in expected
        exact = tuple(api_key_bucket(value) if scope == "key" else value for value in identities[:6]) + ("valid-00",)
        chosen = await metrics.page(scope, _DAY, 1, entity_ids=exact)
        assert {row.entity_id for row in chosen.rows} == set(expected).intersection(exact)
        deployment = await metrics.page("deployment", _DAY, 1)
        assert len(deployment.rows) == 1
        assert deployment.rows[0].requests == 3 * len(identities)


@pytest.mark.asyncio(loop_scope="session")
async def test_attribution_bounds_keep_invalid_identity_traffic_in_population_totals(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,spend,api_requests,timed_requests,updated_at)
            SELECT owner||d||k,owner,($1::date-d)::text,
                CASE WHEN k='bad' THEN owner||repeat('x',1025) ELSE owner||'-valid' END,
                'example-model',CASE WHEN k='bad' THEN repeat('p',257) ELSE 'valid-provider' END,
                CASE WHEN d>0 THEN 10 WHEN (owner='invalid-dominant')=(k='bad') THEN 100 ELSE 20 END,
                1,CASE WHEN (owner='invalid-dominant')=(k='bad') THEN 90 ELSE 10 END,CURRENT_TIMESTAMP
            FROM generate_series(0,3) d CROSS JOIN (VALUES('invalid-dominant'),('valid-dominant')) owners(owner)
                CROSS JOIN (VALUES('bad'),('good')) keys(k) RETURNING id""",
            _DAY.isoformat(),
        )
        windows = tuple(
            IssueMetricWindow(owner, owner, _DAY - timedelta(days=3), _DAY)
            for owner in ("invalid-dominant", "valid-dominant")
        )
        metrics = UsageAlertMetricsRepository(prisma.writer_db)
        attributions = await metrics.issue_key_attributions("user", windows)
        assert len(attributions) == 1
        attribution = attributions[0]
        assert (attribution.reference_id, attribution.key_id, attribution.equivalent) == (
            "valid-dominant",
            "valid-dominant-valid",
            False,
        )
        assert attribution.positive_change_share == pytest.approx(0.9)
        assert (attribution.baseline_value, attribution.observed_value) == (20, 120)
        providers = await metrics.issue_source_providers("user", windows)
        assert tuple((row.reference_id, row.provider) for row in providers) == (("valid-dominant", "valid-provider"),)


@pytest.mark.asyncio(loop_scope="session")
async def test_retirement_usage_omits_oversized_models_before_materialization(db: ManagementDatabase) -> None:
    models = ("a" * 65536, "valid-model", "\U0001f600" * 1024)
    async with _isolated(db) as prisma:
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend" (id,user_id,date,api_key,model,api_requests,updated_at)
            SELECT 'retirement-'||n,'owner',$1,'key',model,7,CURRENT_TIMESTAMP
            FROM unnest($2::text[]) WITH ORDINALITY AS items(model,n) RETURNING id""",
            _DAY.isoformat(),
            models,
        )
        rows = await UsageAlertMetricsRepository(prisma.writer_db).deprecation_usage(models, _DAY, _DAY)
        assert frozenset(row.model for row in rows) == frozenset(models[1:])
        assert tuple(row.requests for row in rows) == (7, 7)
        assert tuple(tuple((point.day, point.value) for point in row.daily_requests) for row in rows) == (
            ((_DAY, 7),),
            ((_DAY, 7),),
        )


@asynccontextmanager
async def _isolated(db: ManagementDatabase) -> AsyncIterator[PrismaClient]:
    schema: Final = "issue_test_" + uuid4().hex
    parsed: Final = urlsplit(db.gateway.database_url or os.environ["DATABASE_URL"])
    query: Final = {**dict(parse_qsl(parsed.query)), "schema": schema}
    url: Final = urlunsplit(parsed._replace(query=urlencode(query)))
    await db.prisma.writer_db.execute_raw(f'CREATE SCHEMA "{schema}"')
    try:
        for table in _TABLES:
            await db.prisma.writer_db.execute_raw(f'CREATE TABLE "{schema}"."{table}" (LIKE "{table}" INCLUDING ALL)')
        with pytest.MonkeyPatch.context() as runtime:
            runtime.setenv("DATABASE_URL", url)
            async with connected_proxy_prisma(url) as prisma:
                assert await prisma.writer_db.query_raw("SELECT current_schema() AS name") == [{"name": schema}]
                yield prisma
    finally:
        await db.prisma.writer_db.execute_raw(f'DROP SCHEMA "{schema}" CASCADE')


async def _coverage(prisma: PrismaClient, day: date) -> None:
    await prisma.writer_db.query_raw(
        """INSERT INTO "LiteLLM_UsageAlertFlush" (source,hour,observed_at)
        SELECT s,h,h FROM unnest(ARRAY['user','team']) s
        CROSS JOIN generate_series($1::timestamp,$1::timestamp+INTERVAL '25 hours',INTERVAL '1 hour') h
        ON CONFLICT(source,hour) DO UPDATE SET observed_at=EXCLUDED.observed_at RETURNING source""",
        day.isoformat(),
    )


async def _finish_projection(job: UsageAlertJob, now: datetime, work: UsageAlertProjectionWork) -> None:
    async with asyncio.timeout(60):
        while not await job._performance_projection.run_pending(
            now, work, job._store, asyncio.get_running_loop().time() + 10
        ):
            continue


async def _seed(prisma: PrismaClient, *, reliability: bool = False) -> None:
    for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
        await prisma.writer_db.query_raw(
            f"""INSERT INTO "{table}"
                (id,{entity},date,api_key,model,custom_llm_provider,spend,api_requests,successful_requests,
                 failed_requests,total_response_time_ms,timed_requests,updated_at)
            SELECT $1||d||k.n,CASE WHEN $2='user' THEN 'owner-'||k.n ELSE 'team' END,
                ($3::date-d)::text,k.key,'example-model','example-provider',
                CASE WHEN d=0 AND k.n=1 AND NOT $6::boolean THEN 50 ELSE 1 END,
                CASE WHEN d=0 AND k.n=1 AND NOT $6::boolean THEN 50000 ELSE 1000 END,
                CASE WHEN d=0 AND $6::boolean THEN 900 ELSE 1000 END,
                CASE WHEN d=0 AND $6::boolean THEN 100 ELSE 0 END,
                CASE WHEN d=0 AND k.n=1 AND NOT $6::boolean THEN 5000000 ELSE 100000 END,
                CASE WHEN d=0 AND k.n=1 AND NOT $6::boolean THEN 50000 ELSE 1000 END,CURRENT_TIMESTAMP
            FROM generate_series(0,7) d CROSS JOIN (VALUES(1,$4::text),(2,$5::text)) k(n,key)
            WHERE NOT $6::boolean OR k.n=1 RETURNING id""",
            table,
            "user" if entity == "user_id" else "team",
            _DAY.isoformat(),
            _KEY_A,
            _KEY_B,
            reliability,
        )
    await _coverage(prisma, _DAY)


async def _next_day(prisma: PrismaClient, day: date, first: int, second: int) -> None:
    for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
        await prisma.writer_db.query_raw(
            f"""INSERT INTO "{table}"
                (id,{entity},date,api_key,model,custom_llm_provider,spend,api_requests,successful_requests,
                 failed_requests,total_response_time_ms,timed_requests,updated_at)
            SELECT id||$1,{entity},$1,api_key,model,custom_llm_provider,
                CASE WHEN api_key=$3 THEN $4::double precision ELSE $5::double precision END,
                1000,1000,0,100000,1000,CURRENT_TIMESTAMP FROM "{table}" WHERE date=$2 RETURNING id""",
            day.isoformat(),
            _DAY.isoformat(),
            _KEY_A,
            first,
            second,
        )
    await _coverage(prisma, day)


@pytest.mark.asyncio(loop_scope="session")
async def test_key_spend_issue_consolidates_traffic_and_uses_fresh_days(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        issues: Final = IssuesRepository.from_prisma(prisma)
        await job._run(_NOW)
        first: Final = await issues.list_issues()
        assert first.total == 1
        original: Final = first.data[0]
        assert (original.family, original.cohort.scope_type, original.cohort.scope_id) == ("spend", "key", _KEY_A)
        assert original.comparisons[0].observed_value == 50
        assert original.comparisons[0].baseline_value == 1
        assert original.comparisons[1].observed_value == 50000
        assert original.comparisons[1].threshold is None
        expected_days: Final = tuple(_DAY - timedelta(days=offset) for offset in range(7, -1, -1))
        assert tuple(point.day for point in original.comparisons[0].history) == expected_days
        assert tuple(point.value for point in original.comparisons[0].history) == (1,) * 7 + (50,)
        assert tuple(point.value for point in original.comparisons[1].history) == (1000,) * 7 + (50000,)
        assert (await issues.list_signals(original.id)).total == 10
        await job._run(_NOW + timedelta(hours=1))
        assert (await issues.list_issues()).data == first.data
        assert (await issues.list_signals(original.id)).total == 10

        await _next_day(prisma, _DAY + timedelta(days=1), 50, 100)
        await job._run(_NOW + timedelta(days=1))
        current: Final = await issues.get_issue(original.id)
        assert current is not None and current.comparisons == original.comparisons
        assert current.range_start == original.range_start and current.range_end == original.range_end
        new_key: Final = await issues.list_issues(scope_type="key", scope_id=_KEY_B)
        assert new_key.total == 1 and new_key.data[0].observed_on == _DAY + timedelta(days=1)
        broader: Final = await issues.list_issues(scope_type="deployment")
        assert broader.total == 1
        assert broader.data[0].comparisons[0].observed_value == 150
        assert broader.data[0].comparisons[0].baseline_value == 9

        await _next_day(prisma, _DAY + timedelta(days=2), 1, 100)
        await job._run(_NOW + timedelta(days=2))
        recovered: Final = await issues.get_issue(original.id)
        assert recovered is not None and not recovered.active and recovered.status == "resolved"
        assert (await issues.list_issues(scope_type="key", scope_id=_KEY_B)).data[0].active


@pytest.mark.asyncio(loop_scope="session")
async def test_due_issue_delivery_retries_after_daily_projection_is_complete(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        rules = UsageAlertsRepository.from_prisma(prisma)
        rule = await rules.save_rule(
            AlertRuleInput(
                name="Key spend notification",
                scope_type="key",
                scope_id=_KEY_A,
                metric="spend",
                direction="increase",
                notification_channels=("slack",),
            ),
            "qa-admin",
        )
        assert rule is not None
        job = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        await job._run(_NOW)
        assert await rules.pending_projection_work() is None
        before = await prisma.writer_db.query_raw(
            "SELECT id,(record->>'attempts')::int AS attempts,record->>'last_error' AS error "
            'FROM "LiteLLM_IssueDelivery" WHERE NOT sent'
        )
        assert len(before) == 1
        assert (before[0]["attempts"], before[0]["error"]) == (1, "transport_not_configured")
        await prisma.writer_db.query_raw(
            "UPDATE \"LiteLLM_IssueDelivery\" SET next_attempt=CURRENT_TIMESTAMP-interval '1 second' "
            "WHERE id=$1 RETURNING id",
            before[0]["id"],
        )
        await job._run(_NOW + timedelta(hours=1))
        after = await prisma.writer_db.query_raw(
            "SELECT (record->>'attempts')::int AS attempts FROM \"LiteLLM_IssueDelivery\" WHERE id=$1",
            before[0]["id"],
        )
        assert after == [{"attempts": 2}], "due delivery was not retried after projection completed"
        assert await rules.pending_projection_work() is None


@pytest.mark.asyncio(loop_scope="session")
async def test_key_evaluation_rejects_normalized_sentinels_and_secret_shaped_ids(db: ManagementDatabase) -> None:
    keys = (
        "unattributed",
        "raw:unattributed",
        "sk-synthetic-only",
        "raw:sk-synthetic-only",
        "raw:raw:sk-synthetic-only",
        _KEY_A,
    )
    async with _isolated(db) as prisma:
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend" (id,user_id,date,api_key,model,spend,api_requests,updated_at)
            SELECT 'key-identity-'||n||'-'||d,'owner',($1::date-d)::text,key,'example-model',
                CASE WHEN d=0 THEN 100 ELSE 1 END,1000,CURRENT_TIMESTAMP
            FROM unnest($2::text[]) WITH ORDINALITY AS keys(key,n) CROSS JOIN generate_series(0,7) d RETURNING id""",
            _DAY.isoformat(),
            keys,
        )
        await _coverage(prisma, _DAY)
        job = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        await job._run(_NOW)
        alerts = await UsageAlertsRepository.from_prisma(prisma).list_alerts(scope_type="key")
        assert alerts.total == 1, "normalized invalid keys created incidents"
        assert alerts.data[0].scope_id == _KEY_A


@pytest.mark.asyncio(loop_scope="session")
async def test_retirement_detects_used_model_after_unused_model_in_the_same_alias(db: ManagementDatabase) -> None:
    from litellm import Router
    from litellm.proxy.common_utils.model_deprecation import RouterDeprecationSource
    from litellm.proxy.spend_tracking.deprecation_issues import DeprecationIssueProjector

    models = ("openai/unused-retirement-example", "openai/used-retirement-example")
    async with _isolated(db) as prisma:
        router = Router(
            model_list=[
                {
                    "model_name": "shared-retirement-alias",
                    "litellm_params": {"model": model},
                    "model_info": {"id": model, "deprecation_date": "2026-11-20"},
                }
                for model in models
            ]
        )
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend" (id,user_id,date,api_key,model,api_requests,updated_at)
            VALUES('used-retirement','owner',$1,$2,$3,7,CURRENT_TIMESTAMP) RETURNING id""",
            _DAY.isoformat(),
            _KEY_A,
            models[1],
        )
        await DeprecationIssueProjector.from_prisma(prisma, RouterDeprecationSource(lambda: router)).run(_NOW)
        issues = await IssuesRepository.from_prisma(prisma).list_issues(kind="model_deprecation")
        assert issues.total == 1, "retired model with actual usage was lost"
        assert issues.data[0].cohort.scope_id == models[1]
        assert issues.data[0].impact.requests == 7


@pytest.mark.asyncio(loop_scope="session")
async def test_single_key_reliability_collapses_matching_scopes_without_traffic_growth(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma, reliability=True)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        issues: Final = IssuesRepository.from_prisma(prisma)
        async with asyncio.timeout(60):
            while True:
                await job.run(_NOW)
                if await job._store.pending_projection_work() is None and (await issues.list_issues()).total:
                    break
                await asyncio.sleep(0.1)
        page: Final = await issues.list_issues()
        assert page.total == 1, tuple((item.cohort.scope_type, item.cohort.scope_id, item.title) for item in page.data)
        issue: Final = page.data[0]
        assert (issue.family, issue.cohort.scope_type, issue.cohort.scope_id) == ("reliability", "key", _KEY_A)
        assert issue.comparisons[0].metric == "error_rate"
        assert issue.comparisons[0].observed_value == 10 and issue.comparisons[0].baseline_value == 0
        assert (await issues.list_signals(issue.id)).total == 5


@pytest.mark.asyncio(loop_scope="session")
async def test_same_day_rule_revision_requires_a_fresh_observation(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        alerts: Final = UsageAlertsRepository.from_prisma(prisma)
        issues: Final = IssuesRepository.from_prisma(prisma)
        definition: Final = AlertRuleInput(
            name="Project spend",
            scope_type="key",
            scope_id=_KEY_A,
            metric="spend",
            direction="increase",
            min_requests=0,
        )
        rule: Final = await alerts.save_rule(definition, None)
        assert rule is not None
        observation: Final = UsageAlertObservation(
            entity_id=_KEY_A, day=_DAY, observed=4, baseline=1, threshold=3, breached=True, baseline_sample_days=7
        )
        await alerts.record_observations(rule, (observation,))
        original: Final = (await issues.active_signals(_DAY))[0]
        await alerts.set_status(original.alert.id, "dismissed")
        revised: Final = await alerts.save_rule(definition.model_copy(update={"multiplier": 5}), None, existing=rule)
        assert revised is not None
        await alerts.record_evaluations(((revised, "complete"),), _DAY, completed=True)
        assert await issues.active_signals(_DAY) == ()
        await alerts.record_observations(revised, (observation.model_copy(update={"observed": 6, "threshold": 5}),))
        updated: Final = (await issues.active_signals(_DAY))[0]
        assert updated.alert.id == original.alert.id and updated.alert.status == "dismissed"
        assert updated.alert.observed_value == original.alert.observed_value
        assert updated.rule.updated_at == revised.updated_at


@pytest.mark.asyncio(loop_scope="session")
async def test_late_custom_key_rule_only_overrides_that_keys_default_signal(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        await prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET spend=40 WHERE date=$1 AND api_key=$2 RETURNING id',
            _DAY.isoformat(),
            _KEY_B,
        )
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        # A rule can change after the daily observations, before Alert projection.
        await job._store.ensure_default_rules(job._defaults)
        await job._store.ensure_projection_work(_DAY)
        await job._evaluate(_DAY, asyncio.get_running_loop().time() + 60)
        issues: Final = IssuesRepository.from_prisma(prisma)
        original: Final = await issues.active_signals(_DAY, scope_type="key", metric="spend")
        assert len(original) == 2
        overridden: Final = next(item for item in original if item.alert.scope_id == _KEY_A)
        retained: Final = next(item for item in original if item.alert.scope_id == _KEY_B)
        custom: Final = await UsageAlertsRepository.from_prisma(prisma).save_rule(
            AlertRuleInput(
                name="Project A high spend threshold",
                scope_type="key",
                scope_id=_KEY_A,
                metric="spend",
                direction="increase",
                multiplier=100,
                min_requests=0,
            ),
            None,
        )
        assert custom is not None
        unprivileged: Final = await issues.active_signals(_DAY, scope_type="key", metric="spend")
        assert overridden.alert.id in {item.alert.id for item in unprivileged}
        await prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_UsageAlertRule" SET record=record || \'{"overrides_defaults":true}\'::jsonb WHERE id=$1 RETURNING id',
            custom.id,
        )
        await job._run(_NOW + timedelta(minutes=1))
        attached: Final = await prisma.writer_db.query_raw('SELECT source_id FROM "LiteLLM_IssueSignal"')
        assert overridden.alert.id not in {row["source_id"] for row in attached}
        assert retained.alert.id in {row["source_id"] for row in attached}
        assert await UsageAlertsRepository.from_prisma(prisma).get_alert(overridden.alert.id) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_shared_key_does_not_make_distinct_resource_populations_equivalent(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma, reliability=True)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        scopes: Final[tuple[tuple[UsageAlertScope, str], ...]] = (
            ("user", "owner-1"),
            ("team", "team"),
            ("model", "example-model"),
        )
        for scope, entity in scopes:
            assert (
                await job._metrics.issue_key_attributions(
                    scope,
                    (
                        IssueMetricWindow(
                            reference_id=entity, entity_id=entity, start=_DAY - timedelta(days=7), day=_DAY
                        ),
                    ),
                )
            )[0].equivalent
        for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
            await prisma.writer_db.query_raw(
                f"""INSERT INTO "{table}"
                    (id,{entity},date,api_key,model,custom_llm_provider,spend,api_requests,successful_requests,
                     failed_requests,total_response_time_ms,timed_requests,updated_at)
                SELECT id||'-shared','other-owner',date,api_key,'other-model',custom_llm_provider,spend,
                    api_requests,successful_requests,failed_requests,total_response_time_ms,timed_requests,updated_at
                FROM "{table}" RETURNING id"""
            )
        for scope, entity in scopes:
            assert not (
                await job._metrics.issue_key_attributions(
                    scope,
                    (
                        IssueMetricWindow(
                            reference_id=entity, entity_id=entity, start=_DAY - timedelta(days=7), day=_DAY
                        ),
                    ),
                )
            )[0].equivalent


@pytest.mark.asyncio(loop_scope="session")
async def test_provider_issue_keeps_an_independent_noisy_key_and_recovers_from_fresh_evidence(
    db: ManagementDatabase,
) -> None:
    async with _isolated(db) as prisma:
        noisy: Final = sha256(b"synthetic-noisy-project").hexdigest()
        for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
            await prisma.writer_db.query_raw(
                f"""INSERT INTO "{table}"
                    (id,{entity},date,api_key,model,custom_llm_provider,spend,api_requests,successful_requests,
                     failed_requests,total_response_time_ms,timed_requests,updated_at)
                SELECT $1||d||k.n||m,CASE WHEN $2='user' THEN 'owner-'||k.n ELSE 'team' END,
                    ($3::date-d)::text,k.key,'model-'||m,'example-provider',1,k.requests,k.requests,0,
                    k.requests*(CASE WHEN d=0 THEN k.latency ELSE 100 END),k.requests,CURRENT_TIMESTAMP
                FROM generate_series(0,7) d CROSS JOIN
                    (VALUES(1,$4::text,500,1000),(2,$5::text,500,1000),(3,$6::text,100,10000)) k(n,key,requests,latency)
                    CROSS JOIN generate_series(1,2) m
                WHERE k.n<>3 OR m=1 RETURNING id""",
                table,
                "user" if entity == "user_id" else "team",
                _DAY.isoformat(),
                _KEY_A,
                _KEY_B,
                noisy,
            )
        # Put the two correlated models on opposite sides of the former 100-pair page boundary.
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,spend,api_requests,
                 successful_requests,failed_requests,total_response_time_ms,timed_requests,updated_at)
            SELECT 'boundary-'||n,'boundary-owner',$1,md5(n::text)||md5(n::text),
                'unrelated-'||n,'aaa-unrelated',0,1,1,0,100,1,CURRENT_TIMESTAMP
            FROM generate_series(1,99) n RETURNING id""",
            _DAY.isoformat(),
        )
        await _coverage(prisma, _DAY)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        await job._run(_NOW)
        issues: Final = IssuesRepository.from_prisma(prisma)
        page: Final = await issues.list_issues()
        assert page.total == 2, tuple((item.cohort.scope_type, item.cohort.scope_id, item.title) for item in page.data)
        provider: Final = next(item for item in page.data if item.cohort.scope_type == "provider")
        independent: Final = next(item for item in page.data if item.cohort.scope_type == "key")
        assert independent.cohort.scope_id == noisy
        assert len(provider.cohort.model_pairs) == 2
        assert provider.impact.keys == 3 and provider.impact.models == 2
        assert (await issues.list_signals(provider.id)).total == 8
        assert (await issues.list_signals(independent.id)).total == 2
        for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
            await prisma.writer_db.query_raw(
                f"""INSERT INTO "{table}"
                    (id,{entity},date,api_key,model,custom_llm_provider,spend,api_requests,successful_requests,
                     failed_requests,total_response_time_ms,timed_requests,updated_at)
                SELECT id||'-next',{entity},$1,api_key,model,custom_llm_provider,spend,api_requests,successful_requests,
                    failed_requests,timed_requests*(CASE WHEN api_key=$3 THEN 10000 ELSE 100 END),timed_requests,CURRENT_TIMESTAMP
                FROM "{table}" WHERE date=$2 RETURNING id""",
                (_DAY + timedelta(days=1)).isoformat(),
                _DAY.isoformat(),
                noisy,
            )
        await job._run(_NOW + timedelta(days=1))
        pending: Final = await issues.get_issue(provider.id)
        assert pending is not None and pending.active, "Partial accounting must not resolve a provider alert"
        await _coverage(prisma, _DAY + timedelta(days=1))
        await job._run(_NOW + timedelta(days=1))
        recovered: Final = await issues.get_issue(provider.id)
        assert recovered is not None and not recovered.active
        continuing: Final = await issues.get_issue(independent.id)
        assert continuing is not None and continuing.active


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_drop_coverage_does_not_block_established_spend_issue(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        await prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_UsageAlertFlush" WHERE hour<$1::timestamp RETURNING source',
            (_DAY + timedelta(days=1)).isoformat(),
        )
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        await job._run(_NOW)
        rules: Final = tuple(
            rule
            for rule in (await UsageAlertsRepository.from_prisma(prisma).list_rules(limit=100)).data
            if rule.enabled
        )
        assert any(rule.direction == "decrease" and rule.evaluation_status == "insufficient_coverage" for rule in rules)
        issues: Final = await IssuesRepository.from_prisma(prisma).list_issues()
        assert any(issue.family == "spend" and issue.cohort.scope_id == _KEY_A for issue in issues.data)


@pytest.mark.asyncio(loop_scope="session")
async def test_entity_page_progress_resumes_old_day_without_early_completion(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,spend,api_requests,
                 successful_requests,failed_requests,total_response_time_ms,timed_requests,updated_at)
            SELECT 'resume-'||k||'-'||d,'owner',($1::date-d)::text,md5(k::text)||md5(k::text),
                'model','example-provider',CASE WHEN d=0 THEN 50 ELSE 1 END,
                1000,1000,0,100000,1000,CURRENT_TIMESTAMP
            FROM generate_series(1,31) k CROSS JOIN generate_series(0,7) d RETURNING id""",
            _DAY.isoformat(),
        )
        await _coverage(prisma, _DAY)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        rule: Final = await job._store.save_rule(
            AlertRuleInput(
                name="Resume each key",
                scope_type="key",
                metric="spend",
                direction="increase",
                min_requests=0,
            ),
            None,
        )
        assert rule is not None
        await job._scope_page((rule,), _DAY, True)
        partial: Final = await job._store.get_rule(rule.id)
        assert partial is not None and partial.evaluation_status == "delayed"
        assert partial.evaluation_progress is not None and partial.evaluation_progress.cursor is not None
        assert not await job._store.completed_rule_ids((rule,), _DAY)
        assert "evaluation_progress" not in partial.model_dump()
        assert "evaluation_progress" not in AlertRule.model_json_schema(mode="serialization")["properties"]
        first: Final = await job._store.list_alerts(limit=500)
        assert first.total == 10
        work: Final = await job._store.pending_projection_work()
        assert work is not None
        await _finish_projection(job, _NOW, work)
        await job._store.finish_projection_work(work)
        assert (await IssuesRepository.from_prisma(prisma).list_issues()).total == 10
        await job._scope_page((partial,), _DAY, True)
        next_work: Final = await job._store.pending_projection_work()
        assert next_work is not None and next_work.version != work.version
        await _finish_projection(job, _NOW + timedelta(hours=1), next_work)
        await job._store.finish_projection_work(next_work)
        assert (await IssuesRepository.from_prisma(prisma).list_issues()).total == 20
        await job._run(_NOW + timedelta(days=1))
        assert await job._store.completed_rule_ids((rule,), _DAY)
        final: Final = await job._store.list_alerts(limit=500)
        assert sum(1 for item in final.data if item.rule_id == rule.id) == 31
        count: Final = await prisma.writer_db.query_raw(
            'SELECT count(*)::int AS count FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1',
            rule.id,
        )
        assert count == [{"count": 31}]
        issues: Final = IssuesRepository.from_prisma(prisma)
        first_page: Final = await issues.active_resource_issue_ids()
        second_page: Final = await issues.active_resource_issue_ids(first_page[-1])
        assert len(first_page) == len(second_page) == 10
        assert set(first_page).isdisjoint(second_page)
        with pytest.raises(ValueError, match="at most 10"):
            await issues.recover_issues(_DAY, issue_ids=first_page + second_page[:1])
        resumed: Final = await job._store.get_rule(rule.id)
        assert resumed is not None and resumed.evaluation_progress is None
        edited: Final = await job._store.save_rule(
            AlertRuleInput.model_validate(rule.model_dump(include=set(AlertRuleInput.model_fields))),
            None,
            resumed,
        )
        assert edited is not None and edited.evaluation_progress is None


@pytest.mark.asyncio(loop_scope="session")
async def test_disabled_primary_requires_another_eligible_healthy_source_to_recover(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        definition: Final = AlertRuleInput(
            name="Extra key policy",
            scope_type="key",
            scope_id=_KEY_A,
            metric="spend",
            direction="increase",
            min_requests=0,
        )
        custom: Final = await job._store.save_rule(definition, None)
        assert custom is not None
        await job._run(_NOW)
        issues: Final = IssuesRepository.from_prisma(prisma)
        issue: Final = next(
            item
            for item in (await issues.list_issues()).data
            if item.family == "spend" and item.cohort.scope_id == _KEY_A
        )
        assert await job._store.save_rule(definition.model_copy(update={"enabled": False}), None, custom)
        await issues.recover_issues(_DAY, (issue.id,))
        unchanged: Final = await issues.get_issue(issue.id)
        assert unchanged is not None and unchanged.active
        await _next_day(prisma, _DAY + timedelta(days=1), 1, 1)
        await job._run(_NOW + timedelta(days=1))
        recovered: Final = await issues.get_issue(issue.id)
        assert recovered is not None and not recovered.active
        history: Final = await job._store.list_alerts(scope_type="key", scope_id=_KEY_A)
        assert any(item.rule_id == custom.id and item.status == "open" for item in history.data)


@pytest.mark.asyncio(loop_scope="session")
async def test_lone_disabled_source_does_not_claim_recovery(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        await prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET spend=5 WHERE date=$1 AND api_key=$2 RETURNING id',
            _DAY.isoformat(),
            _KEY_A,
        )
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        definition: Final = AlertRuleInput(
            name="Sensitive key policy",
            scope_type="key",
            scope_id=_KEY_A,
            metric="spend",
            direction="increase",
            min_requests=0,
        )
        custom: Final = await job._store.save_rule(definition, None)
        assert custom is not None
        await job._run(_NOW)
        issues: Final = IssuesRepository.from_prisma(prisma)
        issue: Final = next(
            item
            for item in (await issues.list_issues()).data
            if item.family == "spend" and item.cohort.scope_id == _KEY_A
        )
        assert await job._store.save_rule(definition.model_copy(update={"enabled": False}), None, custom)
        await _next_day(prisma, _DAY + timedelta(days=1), 1, 1)
        await job._run(_NOW + timedelta(days=1))
        preserved: Final = await issues.get_issue(issue.id)
        assert preserved is not None and preserved.active


@pytest.mark.asyncio(loop_scope="session")
async def test_rule_edit_discards_old_cursor_and_rejects_stale_progress(db: ManagementDatabase) -> None:
    from litellm.types.proxy.usage_alerts import UsageAlertProgress

    async with _isolated(db) as prisma:
        store: Final = UsageAlertsRepository.from_prisma(prisma)
        definition: Final = AlertRuleInput(name="Resumable", scope_type="key", metric="spend", direction="increase")
        rule: Final = await store.save_rule(definition, None)
        assert rule is not None
        position: Final = UsageAlertProgress(day=_DAY, revision=rule.updated_at, cursor="a" * 64, evaluated=True)
        await store.save_progress(((rule, position),))
        partial: Final = await store.get_rule(rule.id)
        assert partial is not None and partial.evaluation_progress == position
        edited: Final = await store.save_rule(definition.model_copy(update={"multiplier": 10}), None, partial)
        assert edited is not None and edited.evaluation_progress is None
        await store.save_progress(((rule, position),))
        current: Final = await store.get_rule(rule.id)
        assert current is not None and current.evaluation_progress is None
        assert current.updated_at == edited.updated_at
        await store.record_evaluations(((rule, "complete"),), _DAY, completed=True)
        assert not await store.completed_rule_ids((edited,), _DAY)


@pytest.mark.asyncio(loop_scope="session")
async def test_busy_background_capacity_defers_and_releases_the_worker_lease(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        async with prisma.writer_db.tx() as transaction:
            await transaction.query_raw("SELECT pg_advisory_xact_lock(1876539401,2)::text")
            await job.run(_NOW)
            assert not (await IssuesRepository.from_prisma(prisma).list_issues()).data
            assert await prisma.writer_db.query_raw(
                "SELECT count(*)::int AS count FROM \"LiteLLM_UsageAlertCheckpoint\" WHERE id='worker'"
            ) == [{"count": 0}]
        await job.run(_NOW)
        assert (await IssuesRepository.from_prisma(prisma).list_issues()).total == 1
        assert await prisma.writer_db.query_raw(
            "SELECT count(*)::int AS count FROM \"LiteLLM_UsageAlertCheckpoint\" WHERE id='worker'"
        ) == [{"count": 0}]


@pytest.mark.asyncio(loop_scope="session")
async def test_capacity_failure_before_coverage_does_not_pin_a_drop_rule_to_an_incomplete_day(
    db: ManagementDatabase,
) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        store: Final = UsageAlertsRepository.from_prisma(prisma)
        rule: Final = await store.save_rule(
            AlertRuleInput(
                name="Project traffic drop",
                scope_type="key",
                scope_id=_KEY_A,
                metric="requests",
                direction="decrease",
            ),
            "qa-admin",
        )
        assert rule is not None
        await prisma.writer_db.execute_raw(
            'DELETE FROM "LiteLLM_UsageAlertFlush" WHERE hour<$1::timestamp', (_DAY + timedelta(days=1)).isoformat()
        )
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        async with prisma.writer_db.tx(timeout=timedelta(seconds=10)) as transaction:
            await transaction.execute_raw('LOCK TABLE "LiteLLM_UsageAlertFlush" IN ACCESS EXCLUSIVE MODE')
            assert not await job._attempt_scope_page((rule,), _DAY, False)
        saved: Final = await store.get_rule(rule.id)
        assert saved is not None and saved.evaluation_progress is None
        await _next_day(prisma, _DAY + timedelta(days=1), 1, 1)
        assert await job._attempt_scope_page((saved,), _DAY + timedelta(days=1), False)
        assert await store.completed_rule_ids((saved,), _DAY + timedelta(days=1)) == frozenset((rule.id,))


@pytest.mark.asyncio(loop_scope="session")
async def test_partial_broad_default_waits_for_its_key_family_only(db: ManagementDatabase) -> None:
    from litellm.types.proxy.usage_alert_defaults import default_usage_rules

    async with _isolated(db) as prisma:
        await _seed(prisma)
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,spend,api_requests,
                 successful_requests,failed_requests,total_response_time_ms,timed_requests,updated_at)
            SELECT 'extra-'||k||'-'||d,'owner',($1::date-d)::text,repeat('f',60)||lpad(k::text,4,'0'),
                'model','example-provider',0.01,1000,1000,0,100000,1000,CURRENT_TIMESTAMP
            FROM generate_series(1,30) k CROSS JOIN generate_series(0,7) d RETURNING id""",
            _DAY.isoformat(),
        )
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        await job._store.ensure_default_rules(default_usage_rules())
        broad: Final = await job._store.enabled_rule_page("deployment", True, None)
        await job._scope_page(broad, _DAY, True)
        sources: Final = IssuesRepository.from_prisma(prisma)
        assert await sources.active_signals(_DAY, scope_type="deployment", metric="spend")
        assert not await sources.active_signals(_DAY, scope_type="deployment", metric="spend", completed_only=True)
        key_rules: Final = await job._store.enabled_rule_page("key", True, None)
        spend: Final = tuple(rule for rule in key_rules if rule.metric == "spend")
        await job._scope_page(spend, _DAY, True)
        assert await sources.active_signals(_DAY, scope_type="key", metric="spend", completed_only=True)
        assert not await sources.active_signals(_DAY, scope_type="deployment", metric="spend", completed_only=True)
        work: Final = await job._store.pending_projection_work()
        assert work is not None
        await _finish_projection(job, _NOW, work)
        partial: Final = await sources.list_issues()
        assert len(partial.data) == 1 and partial.data[0].cohort.scope_type == "key"
        while not await job._store.completed_rule_ids((spend[0],), _DAY):
            current: Final = await job._store.get_rule(spend[0].id)
            assert current is not None
            await job._scope_page((current,), _DAY, True)
        assert await sources.active_signals(_DAY, scope_type="deployment", metric="spend", completed_only=True)
        unfinished: Final = tuple(rule for rule in key_rules if rule.metric != "spend")
        assert not await job._store.completed_rule_ids(unfinished, _DAY)
        await job._run(_NOW)
        result: Final = await sources.list_issues()
        assert result.total == 1 and result.data[0].cohort.scope_id == _KEY_A


@pytest.mark.asyncio(loop_scope="session")
async def test_projection_resume_keeps_spend_and_traffic_in_one_episode(db: ManagementDatabase) -> None:
    from litellm.types.proxy.usage_alerts import UsageAlertProjectionCursor

    async with _isolated(db) as prisma:
        await _seed(prisma)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        await job._store.ensure_default_rules(job._defaults)
        await job._evaluate(_DAY, asyncio.get_running_loop().time() + 60)
        work: Final = await job._store.pending_projection_work()
        projector: Final = job._performance_projection
        assert work is not None and projector is not None
        observed_at: Final = _NOW.replace(microsecond=987000)
        position = UsageAlertProjectionCursor(version=work.version, observed_at=observed_at)  # rebind-ok: cursor
        while position.step < 21:
            next_position: Final = await projector._projection_page(position, work, observed_at)
            assert next_position is not None
            position = next_position
        await job._store.save_projection_cursor(position)
        before: Final = await IssuesRepository.from_prisma(prisma).list_issues()
        assert len(before.data) == 1 and before.data[0].family == "spend"
        added: Final = await job._store.save_rule(
            AlertRuleInput(
                name="Later policy", scope_type="key", scope_id=_KEY_B, metric="latency", direction="increase"
            ),
            None,
        )
        assert added is not None
        await job._scope_page((added,), _DAY, False)
        latest: Final = await job._store.pending_projection_work()
        assert latest is not None and latest.version != work.version
        retained: Final = await job._store.projection_cursor(latest)
        assert retained.step == 21 and retained.version == work.version
        restarted: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        assert restarted._performance_projection is not None
        assert not await restarted._performance_projection.run_pending(
            _NOW + timedelta(hours=1), latest, restarted._store, asyncio.get_running_loop().time() + 60
        )
        resumed: Final = await restarted._store.projection_cursor(latest)
        assert resumed.version == latest.version and resumed.step == 0
        assert await restarted._performance_projection.run_pending(
            _NOW + timedelta(hours=2), latest, restarted._store, asyncio.get_running_loop().time() + 60
        )
        after: Final = await IssuesRepository.from_prisma(prisma).list_issues()
        assert tuple(item.id for item in after.data) == tuple(item.id for item in before.data)
        links: Final = await IssuesRepository.from_prisma(prisma).list_signals(after.data[0].id)
        assert any(link.performance is not None and link.performance.metric == "requests" for link in links.data)


@pytest.mark.asyncio(loop_scope="session")
async def test_slack_retry_only_repeats_the_failed_destination(db: ManagementDatabase) -> None:
    paths: Final[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(503 if request.url.path == "/two" and paths.count("/two") == 1 else 200)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    try:
        async with _isolated(db) as prisma:
            await _seed(prisma)
            rules: Final = UsageAlertsRepository.from_prisma(prisma)
            assert (
                await rules.save_rule(
                    AlertRuleInput(
                        name="Project spend",
                        scope_type="key",
                        scope_id=_KEY_A,
                        metric="spend",
                        direction="increase",
                        notification_channels=("slack",),
                    ),
                    "qa-admin",
                )
                is not None
            )
            assert (
                await rules.save_rule(
                    AlertRuleInput(
                        name="Another project rule",
                        scope_type="key",
                        scope_id=_KEY_A,
                        metric="spend",
                        direction="increase",
                        notification_channels=("slack",),
                    ),
                    "qa-admin",
                )
                is not None
            )
            slack: Final = SlackAlerting(
                alerting=["slack"],
                alert_types=[],
                async_http_handler=handler,
                alert_to_webhook_url={
                    "usage_alerts": [
                        "https://hooks.example.com/one",
                        "https://hooks.example.com/two",
                        "https://hooks.example.com/one",
                    ]
                },
            )
            job: Final = UsageAlertJob.from_prisma(prisma, slack)
            await job._run(_NOW)
            assert paths == ["/one", "/two"] or paths == ["/two", "/one"]
            await prisma.writer_db.query_raw(
                'UPDATE "LiteLLM_IssueDelivery" SET next_attempt=CURRENT_TIMESTAMP WHERE NOT sent RETURNING id'
            )
            retry: Final = (await IssuesRepository.from_prisma(prisma).pending_deliveries())[0]
            assert (
                await rules.save_rule(
                    AlertRuleInput.model_validate(
                        retry.rule.model_dump(include=set(AlertRuleInput.model_fields))
                    ).model_copy(update={"enabled": False}),
                    "qa-admin",
                    retry.rule,
                )
                is not None
            )
            await job._performance_projection.notify()
            assert paths.count("/one") == 1 and paths.count("/two") == 2
            assert await IssuesRepository.from_prisma(prisma).pending_deliveries() == ()
            records: Final = await prisma.writer_db.query_raw(
                'SELECT record::text AS value FROM "LiteLLM_IssueDelivery"'
            )
            assert all("hooks.example.com" not in row["value"] for row in records)
    finally:
        await handler.close()


@pytest.mark.asyncio(loop_scope="session")
async def test_cancelled_email_receipt_survives_a_different_delivery(
    db: ManagementDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted: Final = asyncio.Event()
    release: Final = asyncio.Event()
    messages: Final[list[bytes]] = []

    async def smtp(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"220 example.com ready\r\n")
        await writer.drain()
        try:
            while command := await reader.readline():
                if command.upper().startswith(b"DATA"):
                    writer.write(b"354 End with a dot\r\n")
                    await writer.drain()
                    messages.append(await reader.readuntil(b"\r\n.\r\n"))
                    accepted.set()
                    await release.wait()
                    writer.write(b"250 Accepted\r\n")
                elif command.upper().startswith(b"QUIT"):
                    writer.write(b"221 Closing\r\n")
                    break
                else:
                    writer.write(b"250 OK\r\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server: Final = await asyncio.start_server(smtp, "127.0.0.1", 0)
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_PORT", str(server.sockets[0].getsockname()[1]))
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    monkeypatch.setenv("SMTP_TLS", "False")
    monkeypatch.setenv("SMTP_USE_SSL", "False")
    for name in ("SMTP_TIMEOUT", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    try:
        async with _isolated(db) as prisma:
            await _seed(prisma)
            rules: Final = UsageAlertsRepository.from_prisma(prisma)
            assert (
                await rules.save_rule(
                    AlertRuleInput(
                        name="Project spend",
                        scope_type="key",
                        scope_id=_KEY_A,
                        metric="spend",
                        direction="increase",
                        notification_channels=("email",),
                        email_recipients=("qa@example.com",),
                    ),
                    "qa-admin",
                )
                is not None
            )
            job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
            await job._run(_NOW)
            store: Final = IssuesRepository.from_prisma(prisma)
            await prisma.writer_db.query_raw(
                'UPDATE "LiteLLM_IssueDelivery" SET next_attempt=CURRENT_TIMESTAMP WHERE NOT sent RETURNING id'
            )
            pending: Final = (await store.pending_deliveries())[0]
            delivery: Final = UsageAlertDelivery(SlackAlerting(alerting=["email"], alert_types=[]), store)
            sending: Final = asyncio.create_task(delivery.send_issue(pending))
            await asyncio.wait_for(accepted.wait(), 10)
            sending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sending
            release.set()
            retained: Final = UsageAlertDelivery._email_task
            assert retained is not None
            await asyncio.wait_for(asyncio.shield(retained), 10)
            assert await store.pending_delivery(pending.delivery.id) is None, (
                "A cancelled send must persist its successful receipt without another delivery"
            )
            different: Final = pending.model_copy(
                update={"delivery": pending.delivery.model_copy(update={"id": "another-delivery"})}
            )
            assert await delivery.send_issue(different)
            assert await store.pending_delivery(pending.delivery.id) is None
            assert len(messages) == 2
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        if UsageAlertDelivery._email_task is not None:
            await asyncio.wait_for(asyncio.shield(UsageAlertDelivery._email_task), 10)
        UsageAlertDelivery._email_task = None
        UsageAlertDelivery._email_key = None
        UsageAlertDelivery._email_store = None


@pytest.mark.asyncio(loop_scope="session")
async def test_large_custom_rule_scope_does_not_starve_other_scopes_on_resume(db: ManagementDatabase) -> None:
    from litellm.types.proxy.usage_alerts import UsageAlertWorkCursor

    async with _isolated(db) as prisma:
        await _seed(prisma)
        store: Final = UsageAlertsRepository.from_prisma(prisma)
        key_rule: Final = await store.save_rule(
            AlertRuleInput(
                name="Project spend", scope_type="key", scope_id=_KEY_A, metric="spend", direction="increase"
            ),
            "qa-user",
        )
        team_rule: Final = await store.save_rule(
            AlertRuleInput(name="Team spend", scope_type="team", scope_id="team", metric="spend", direction="increase"),
            "qa-admin",
        )
        assert key_rule is not None and team_rule is not None
        await prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_UsageAlertRule" (id,scope_type,scope_id,enabled,record)
            SELECT 'custom-'||lpad(n::text,6,'0'),scope_type,scope_id,enabled,
                record||jsonb_build_object('id','custom-'||lpad(n::text,6,'0'))
            FROM "LiteLLM_UsageAlertRule",generate_series(1,1000) n WHERE id=$1 RETURNING id""",
            key_rule.id,
        )
        await store.save_work_cursor(UsageAlertWorkCursor(phase=3))
        for _ in range(12):
            job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
            cursor, _, _ = await job._evaluation_page(await store.work_cursor(), _DAY)
            await store.save_work_cursor(cursor)
        assert await store.completed_rule_ids((team_rule,), _DAY) == frozenset((team_rule.id,))
        completed: Final = await prisma.writer_db.query_raw(
            """SELECT COUNT(*)::int AS count FROM "LiteLLM_UsageAlertRule"
            WHERE scope_type='key' AND record->>'last_evaluated_day'=$1""",
            _DAY.isoformat(),
        )
        assert completed == [{"count": 20}]
