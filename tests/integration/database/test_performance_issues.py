from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase
from integration._support.prisma_clients import connected_proxy_prisma

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy.spend_tracking.usage_alert_job import UsageAlertJob
from litellm.proxy.utils import PrismaClient
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.usage_alert_metrics import IssueMetricWindow
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import UsageAlertObservation, UsageAlertRule, UsageAlertRuleInput, UsageAlertScope

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
    "LiteLLM_UserTable",
    "LiteLLM_TeamTable",
    "LiteLLM_SpendLogDelivery",
)


@asynccontextmanager
async def _isolated(db: ManagementDatabase) -> AsyncIterator[PrismaClient]:
    schema: Final = "issue_test_" + uuid4().hex
    assert db.gateway.database_url is not None
    parsed: Final = urlsplit(db.gateway.database_url)
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
async def test_single_key_reliability_collapses_matching_scopes_without_traffic_growth(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma, reliability=True)
        await UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))._run(_NOW)
        issues: Final = IssuesRepository.from_prisma(prisma)
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
        definition: Final = UsageAlertRuleInput(
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
        await alerts.finish_day(revised, _DAY)
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
        # A rule can change after the daily observations, before Issue projection.
        await UsageAlertJob(job._store, job._metrics, job._health, job._delivery)._run(_NOW)
        issues: Final = IssuesRepository.from_prisma(prisma)
        original: Final = await issues.active_signals(_DAY, scope_type="key", metric="spend")
        assert len(original) == 2
        overridden: Final = next(item for item in original if item.alert.scope_id == _KEY_A)
        retained: Final = next(item for item in original if item.alert.scope_id == _KEY_B)
        custom: Final = await UsageAlertsRepository.from_prisma(prisma).save_rule(
            UsageAlertRuleInput(
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
        rules: Final = await UsageAlertsRepository.from_prisma(prisma).list_enabled_rules(limit=100)
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
            UsageAlertRuleInput(
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
        assert not await job._store.is_day_complete(rule, _DAY)
        assert "evaluation_progress" not in partial.model_dump()
        assert "evaluation_progress" not in UsageAlertRule.model_json_schema(mode="serialization")["properties"]
        first: Final = await job._store.list_alerts(limit=500)
        assert first.total == 10
        await job._run(_NOW + timedelta(days=1))
        assert await job._store.is_day_complete(rule, _DAY)
        final: Final = await job._store.list_alerts(limit=500)
        assert sum(1 for item in final.data if item.rule_id == rule.id) == 31
        count: Final = await prisma.writer_db.query_raw(
            'SELECT count(*)::int AS count FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1',
            rule.id,
        )
        assert count == [{"count": 31}]
        resumed: Final = await job._store.get_rule(rule.id)
        assert resumed is not None and resumed.evaluation_progress is None
        edited: Final = await job._store.save_rule(
            UsageAlertRuleInput.model_validate(rule.model_dump(include=set(UsageAlertRuleInput.model_fields))),
            None,
            resumed,
        )
        assert edited is not None and edited.evaluation_progress is None


@pytest.mark.asyncio(loop_scope="session")
async def test_disabled_primary_requires_another_eligible_healthy_source_to_recover(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await _seed(prisma)
        job: Final = UsageAlertJob.from_prisma(prisma, SlackAlerting(alerting=[], alert_types=[]))
        definition: Final = UsageAlertRuleInput(
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
        await issues.recover_issues(_DAY)
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
        definition: Final = UsageAlertRuleInput(
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
        definition: Final = UsageAlertRuleInput(
            name="Resumable", scope_type="key", metric="spend", direction="increase"
        )
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
        assert not await store.is_day_complete(edited, _DAY)


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
async def test_partial_broad_default_waits_for_its_key_family_only(db: ManagementDatabase) -> None:
    from litellm.types.proxy.usage_alert_defaults import default_usage_rules

    async with _isolated(db) as prisma:
        await _seed(prisma)
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
        assert await sources.active_signals(_DAY, scope_type="deployment", metric="spend", completed_only=True)
        unfinished: Final = tuple(rule for rule in key_rules if rule.metric != "spend")
        assert not await job._store.completed_rule_ids(unfinished, _DAY)
        await job._run(_NOW)
        result: Final = await sources.list_issues()
        assert result.total == 1 and result.data[0].cohort.scope_id == _KEY_A
