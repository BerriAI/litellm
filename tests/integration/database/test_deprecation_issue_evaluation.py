from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway

from litellm.proxy.common_utils.model_deprecation import deprecation_alert_state_key
from litellm.proxy.spend_tracking.deprecation_issues import DeprecationIssueProjector
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.usage_alert_metrics import UsageAlertMetricsRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.model_deprecation import ModelDeprecationInfo, ModelDeprecationResponse


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
    name = f"example/{prefix}"
    now = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
    retirement = now.date() + timedelta(days=15)
    config = ConfigRepository(db.prisma, use_writer=True)
    previous_checkpoint = await config.get_param("issues:deprecation_projection")
    lease = UsageAlertsRepository.from_prisma(db.prisma)
    assert await lease.claim_job(prefix)

    def schedule(today: date) -> ModelDeprecationResponse:
        info = ModelDeprecationInfo(
            model_name=prefix,
            litellm_model=name,
            deprecation_date=retirement,
            days_until_deprecation=(retirement - today).days,
            status="deprecated" if retirement < today else "imminent",
            successor_model="example/replacement",
        )
        return ModelDeprecationResponse(
            deprecated=(info,) if retirement < today else (),
            imminent=(info,) if retirement >= today else (),
            upcoming=(),
            warn_within_days=30,
            checked_at=now,
        )

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
        projector = DeprecationIssueProjector.from_prisma(db.prisma, schedule)
        await projector.run(now)
        page = gateway.get("/issues", {"scope_type": "model", "scope_id": name})
        assert page["total"] == 1
        original = page["data"][0]
        path = f"/issues/{original['id']}"
        assert original["status"] == "open"
        assert original["impact"]["requests"] == 100
        assert gateway.get(path + "/signals")["total"] == 1
        assert gateway.get(path + "/activity")["metadata"]["total_api_requests"] == 100
        state = {"model": name, "deprecation_date": retirement.isoformat()}
        resolved = gateway.request(
            "PUT",
            "/alerting/model-deprecations/state",
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
        assert gateway.get("/alerting/model-deprecations/state", state)["status"] == "dismissed"
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
