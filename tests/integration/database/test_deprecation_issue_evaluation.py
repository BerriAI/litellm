from datetime import date
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase

from litellm.repositories.usage_alert_metrics import UsageAlertMetricsRepository


@pytest.mark.asyncio(loop_scope="session")
async def test_deprecation_usage_counts_distinct_resources_without_multiplying_requests(db: ManagementDatabase):
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
