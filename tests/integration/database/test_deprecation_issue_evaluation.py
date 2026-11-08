import asyncio
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway

from litellm import Router
from litellm.proxy.common_utils.model_deprecation import RouterDeprecationSource, deprecation_alert_state_key
from litellm.proxy.spend_tracking.deprecation_issues import (
    DeprecationCandidate,
    DeprecationIssueProjector,
    _ProjectionCheckpoint,
    deprecation_issue,
)
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.usage_alert_metrics import UsageAlertMetricsRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.model_deprecation import DeprecationAlertState, ModelDeprecationInfo, ModelDeprecationUsage


@pytest.mark.asyncio(loop_scope="session")
async def test_deprecation_usage_counts_distinct_resources_without_multiplying_requests(db: ManagementDatabase) -> None:
    prefix = f"deprecation-issue-{uuid4()}"
    model = f"example/{prefix}"
    try:
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,endpoint,api_requests,updated_at)
            SELECT $1||x.id,$1||'owner',x.day,$1||x.key,
                CASE WHEN x.id='other-model' THEN $2||'-other' ELSE $2 END,
                x.endpoint,x.requests,CURRENT_TIMESTAMP
            FROM (VALUES
                ('first','2026-11-04','key-a','/one',10),
                ('second','2026-11-04','key-a','/two',20),
                ('previous','2026-11-03','key-b','/one',7),
                ('zero','2026-11-05','key-c','/one',0),
                ('old','2026-10-01','key-d','/one',999),
                ('other-model','2026-11-04','key-e','/one',999)
            ) x(id,day,key,endpoint,requests) RETURNING id""",
            prefix,
            model,
        )
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyTeamSpend"
                (id,team_id,date,api_key,model,endpoint,api_requests,updated_at)
            SELECT $1||x.id,$1||x.team,x.day,$1||x.key,$2,x.endpoint,x.requests,CURRENT_TIMESTAMP
            FROM (VALUES
                ('first','2026-11-04','key-a','team-a','/one',15),
                ('second','2026-11-04','key-a','team-a','/two',5),
                ('previous','2026-11-03','key-b','team-b','/one',7),
                ('zero','2026-11-05','key-c','team-c','/one',0),
                ('old','2026-10-01','key-d','team-d','/one',999)
            ) x(id,day,key,team,endpoint,requests) RETURNING id""",
            prefix,
            model,
        )
        repository = UsageAlertMetricsRepository(db.prisma.writer_db)
        rows = await repository.deprecation_usage((model, model + "-unused"), date(2026, 10, 7), date(2026, 11, 5))
        assert len(rows) == 1
        assert rows[0].model == model
        assert rows[0].requests == 37
        assert rows[0].last_used_date == date(2026, 11, 4)
        assert rows[0].last_day_requests == 30
        assert (rows[0].keys, rows[0].users, rows[0].teams) == (2, 1, 2)
        assert await repository.deprecation_usage((), date(2026, 10, 7), date(2026, 11, 5)) == ()
    finally:
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE id LIKE $1 RETURNING id', prefix + "%"
        )
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyTeamSpend" WHERE id LIKE $1 RETURNING id', prefix + "%"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_retirement_issue_http_state_survives_replay_and_reopens_only_for_new_usage(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    prefix = f"retirement-lifecycle-{uuid4()}"
    name = f"openai/{prefix}"
    now = datetime(2026, 11, 5, 10, microsecond=123456, tzinfo=timezone.utc)
    retirement = now.date() + timedelta(days=15)
    config = ConfigRepository(db.prisma, use_writer=True)
    previous_checkpoint = await config.get_param("issues:deprecation_projection")
    lease = UsageAlertsRepository.from_prisma(db.prisma)
    assert await lease.claim_job(prefix)

    router = Router(
        model_list=[
            {
                "model_name": prefix,
                "litellm_params": {"model": name},
                "model_info": {"deprecation_date": retirement.isoformat(), "successor_model": "replacement"},
            }
        ]
    )
    source = RouterDeprecationSource(lambda: router)

    async def record(day: date, requests: int) -> None:
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,api_requests,updated_at)
            VALUES($1,$2,$3,$2,$4,$5,CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET api_requests=EXCLUDED.api_requests RETURNING id""",
            prefix + day.isoformat(),
            prefix,
            day.isoformat(),
            name,
            requests,
        )

    try:
        await config.delete_param("issues:deprecation_projection")
        await record(now.date(), 100)
        projector = DeprecationIssueProjector.from_prisma(db.prisma, source)
        await projector.run(now)
        page = gateway.get("/alerts", {"scope_type": "model", "scope_id": name})
        assert page["total"] == 1
        original = page["data"][0]
        path = f"/alerts/{original['id']}"
        assert original["status"] == "open"
        assert original["impact"]["requests"] == 100
        assert gateway.get(path + "/signals")["total"] == 1
        assert gateway.get(path + "/activity")["metadata"]["total_api_requests"] == 100
        state = {"model": name, "deprecation_date": retirement.isoformat()}
        resolved = gateway.request(
            "PUT",
            "/alerts/model-deprecations/state",
            {**state, "status": "resolved", "last_seen_date": now.date().isoformat(), "last_seen_requests": 100},
        )
        assert resolved.status_code == 200
        assert gateway.get(path)["status"] == "resolved"
        await config.delete_param("issues:deprecation_projection")
        await projector.run(now)
        assert gateway.get(path)["status"] == "resolved"
        stale = gateway.request("PATCH", path, {"status": "dismissed", "expected_status": "open"})
        assert stale.status_code == 409
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_Config" SET param_value=to_jsonb($2::text) WHERE param_name=$1 RETURNING param_name',
            deprecation_alert_state_key(name, retirement),
            resolved.text,
        )
        assert gateway.get(path)["status"] == "resolved"
        dismissed = gateway.request("PATCH", path, {"status": "dismissed", "expected_status": "resolved"})
        assert dismissed.status_code == 200
        assert gateway.get("/alerts/model-deprecations/state", state)["status"] == "dismissed"
        await record(now.date(), 110)
        await projector.run(now + timedelta(minutes=20))
        assert gateway.get(path)["status"] == "dismissed"
        await projector.run(now + timedelta(hours=1))
        reopened = gateway.get(path)
        assert reopened["status"] == "open"
        assert reopened["impact"]["requests"] == 110
        assert reopened["first_seen"] == original["first_seen"]
        assert gateway.get(path + "/signals")["total"] == 1
        expired_at = now + timedelta(days=35)
        await projector.run(expired_at)
        expired = gateway.get(path)
        assert expired["status"] == "resolved" and expired["active"] is False
        assert gateway.request("PATCH", path, {"status": "open", "expected_status": "resolved"}).status_code == 409
        await record(expired_at.date(), 5)
        await projector.run(expired_at + timedelta(hours=1))
        returned = gateway.get(path)
        assert returned["status"] == "open" and returned["active"] is True
        assert returned["impact"]["requests"] == 5
        assert returned["first_seen"] == original["first_seen"]
    finally:
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_IssueSignal" WHERE issue_id IN '
            '(SELECT id FROM "LiteLLM_Issue" WHERE scope_id=$1) RETURNING issue_id',
            name,
        )
        await db.prisma.writer_db.query_raw('DELETE FROM "LiteLLM_Issue" WHERE scope_id=$1 RETURNING id', name)
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE id LIKE $1 RETURNING id', prefix + "%"
        )
        await config.delete_param(deprecation_alert_state_key(name, retirement))
        if previous_checkpoint is None:
            await config.delete_param("issues:deprecation_projection")
        else:
            await config.set_param("issues:deprecation_projection", previous_checkpoint.param_value)
        await lease.release_job(prefix)


@pytest.mark.asyncio(loop_scope="session")
async def test_retirement_scan_resumes_bounded_pages_without_premature_recovery(db: ManagementDatabase) -> None:
    prefix = f"retirement-pages-{uuid4()}"
    now = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
    retirement = date(2026, 11, 20)
    config = ConfigRepository(db.prisma, use_writer=True)
    checkpoint_name = "issues:deprecation_projection"
    previous = await config.get_param(checkpoint_name)
    lease = UsageAlertsRepository.from_prisma(db.prisma)
    assert await lease.claim_job(prefix)
    repository = IssuesRepository(db.prisma.writer_db)
    old_model = f"openai/{prefix}-old"
    old = deprecation_issue(
        DeprecationCandidate(
            old_model,
            ModelDeprecationInfo(
                model_name=prefix + "-old",
                litellm_model=old_model,
                deprecation_date=retirement,
                days_until_deprecation=15,
                status="imminent",
            ),
        ),
        ModelDeprecationUsage(
            model=old_model, requests=1, last_used_date=now.date(), last_day_requests=1, keys=1, users=1, teams=0
        ),
        DeprecationAlertState(model=old_model, deprecation_date=retirement),
        now - timedelta(hours=1),
    )
    router = Router(
        model_list=[
            {
                "model_name": f"{prefix}-{index}",
                "litellm_params": {"model": f"openai/{prefix}-{index}"},
                "model_info": {"id": f"{prefix}-{index}", "deprecation_date": retirement.isoformat()},
            }
            for index in range(600)
        ]
    )
    try:
        await config.delete_param(checkpoint_name)
        await repository.upsert_issues((old,))
        earlier_alias = old.model_copy(
            update={
                "title": "alphabetically first configured alias",
                "deprecation": old.deprecation.model_copy(
                    update={
                        "model": old.deprecation.model.model_copy(update={"model_name": "a-" + prefix}),
                    }
                ),
            }
        )
        await repository.upsert_issues((earlier_alias,))
        await repository.upsert_issues((old,))
        assert (await repository.get_issue(old.id)).title == earlier_alias.title
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,api_requests,updated_at)
            SELECT $1||n,$1,$2,$1,'openai/'||$1||'-'||n,1,CURRENT_TIMESTAMP
            FROM generate_series(0,599) n RETURNING id""",
            prefix,
            now.date().isoformat(),
        )
        source = RouterDeprecationSource(lambda: router)
        projector = DeprecationIssueProjector.from_prisma(db.prisma, source)

        async def interrupt_after_page(
            projector: DeprecationIssueProjector, observed_at: datetime, revision: str, after: int = 0
        ) -> _ProjectionCheckpoint:
            task = asyncio.create_task(projector.run(observed_at))
            try:

                async def checkpoint() -> _ProjectionCheckpoint:
                    while not task.done():
                        value = await config.get_param(checkpoint_name)
                        if value is not None:
                            state = _ProjectionCheckpoint.model_validate(value.param_value)
                            if state.source_revision == revision and after < state.cursor.deployment < 600:
                                return state
                        await asyncio.sleep(0)
                    pytest.fail("projector finished before observing a partial scan")

                return await asyncio.wait_for(checkpoint(), timeout=10)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        revision = await source.prepare_revision(now.date())
        assert revision is not None
        partial = await interrupt_after_page(projector, now, revision)
        assert partial.complete is False
        assert (await repository.get_issue(old.id)).active is True
        replica = Router(model_list=list(router.registry().all()))
        next_source = RouterDeprecationSource(lambda: replica)
        assert await next_source.prepare_revision(now.date()) == revision
        replacement = DeprecationIssueProjector.from_prisma(db.prisma, next_source)
        resumed = await interrupt_after_page(
            replacement, now + timedelta(seconds=10), revision, partial.cursor.deployment
        )
        assert resumed.observed_at == partial.observed_at
        assert resumed.cursor.deployment > partial.cursor.deployment
        assert (await repository.get_issue(old.id)).active is True
        for iteration in range(10):
            current = replacement if iteration % 2 == 0 else projector
            await current.run(now + timedelta(seconds=20 + iteration))
            complete = await config.get_param(checkpoint_name)
            if _ProjectionCheckpoint.model_validate(complete.param_value).complete:
                break
        else:
            pytest.fail("retirement scan did not resume to completion")
        assert (await repository.get_issue(old.id)).active is False
        rows = await db.prisma.writer_db.query_raw(
            'SELECT count(*)::int AS count FROM "LiteLLM_Issue" WHERE scope_id LIKE $1 AND active',
            f"openai/{prefix}-%",
        )
        assert rows[0]["count"] == 600
    finally:
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_IssueSignal" WHERE issue_id IN (SELECT id FROM "LiteLLM_Issue" '
            "WHERE scope_id LIKE $1) RETURNING issue_id",
            f"openai/{prefix}-%",
        )
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_Issue" WHERE scope_id LIKE $1 RETURNING id',
            f"openai/{prefix}-%",
        )
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE id LIKE $1 RETURNING id',
            prefix + "%",
        )
        if previous is None:
            await config.delete_param(checkpoint_name)
        else:
            await config.set_param(checkpoint_name, previous.param_value)
        await lease.release_job(prefix)
