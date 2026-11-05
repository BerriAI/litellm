from datetime import date
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase

from litellm.repositories.provider_latency import ProviderLatencyRepository
from litellm.types.proxy.issues import IssueModelPair
from litellm.types.proxy.usage_alerts import UsageAlertRuleInput


@pytest.mark.asyncio(loop_scope="session")
async def test_provider_latency_requires_established_within_model_regressions(db: ManagementDatabase) -> None:
    prefix = f"provider-latency-{uuid4()}"
    day = date(2031, 1, 12)
    pairs = tuple(IssueModelPair(provider=prefix, model=f"{prefix}-{suffix}") for suffix in ("a", "b"))
    keys = (prefix + "-key-a", prefix + "-key-b")
    rule = UsageAlertRuleInput(name="Latency", scope_type="key", metric="latency", direction="increase", min_value=500)
    repository = ProviderLatencyRepository(db.prisma.writer_db)
    try:
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,updated_at)
            SELECT $1||d||m.suffix||k.suffix,$1||'-owner',($2::date-d)::text,$1||'-key-'||k.suffix,
                   $1||'-'||m.suffix,$1,100,100,100*m.latency*(CASE WHEN d=0 THEN 5 ELSE 1 END),CURRENT_TIMESTAMP
            FROM generate_series(0,7) d CROSS JOIN (VALUES('a',200),('b',1000)) m(suffix,latency)
                CROSS JOIN (VALUES('a'),('b')) k(suffix) RETURNING id""",
            prefix,
            day.isoformat(),
        )
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyTeamSpend"
                (id,team_id,date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,updated_at)
            SELECT id||'team',$1||CASE WHEN api_key=$1||'-key-a' THEN 'team1' ELSE 'team2' END,
                   date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,CURRENT_TIMESTAMP
            FROM "LiteLLM_DailyUserSpend" WHERE custom_llm_provider=$1 RETURNING id""",
            prefix,
        )
        first = await repository.candidate_pairs(day, IssueModelPair(provider=prefix, model="0"), limit=1)
        assert first == pairs[:1]
        assert await repository.candidate_pairs(day, first[0], limit=1) == pairs[1:]
        evidence = await repository.latency_cohort(pairs, day, rule)
        assert evidence is not None
        assert evidence.cohort.model_pairs == pairs
        assert evidence.comparison.observed_value == 3000
        assert evidence.comparison.baseline_value == 600
        assert evidence.comparison.baseline_sample_days == 7
        assert (evidence.impact.keys, evidence.impact.users, evidence.impact.teams) == (2, 1, 2)
        assert (evidence.impact.measured_keys, evidence.impact.measured_models) == (2, 2)
        assert evidence.impact.requests == 400
        assert not await repository.cohort_recovered(pairs, day, rule)
        assert await repository.source_members(pairs, day, rule, "key", keys) == frozenset(keys)
        models = tuple(pair.model for pair in pairs)
        assert await repository.source_members(pairs, day, rule, "model", models) == frozenset(models)
        assert await repository.source_members(pairs, day, rule, "user", (prefix + "-owner",)) == {prefix + "-owner"}
        teams = (prefix + "team1", prefix + "team2")
        assert await repository.source_members(pairs, day, rule, "team", teams) == frozenset(teams)
        assert await repository.source_members(pairs, day, rule, "deployment", ("deployment",)) == {"deployment"}
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,updated_at)
            SELECT id||'-new-owner',$1||'-new-owner',date,api_key,model,custom_llm_provider,api_requests,
                   timed_requests,total_response_time_ms,CURRENT_TIMESTAMP
            FROM "LiteLLM_DailyUserSpend" WHERE custom_llm_provider=$1 AND date=$2 RETURNING id""",
            prefix,
            day.isoformat(),
        )
        assert await repository.source_members(
            pairs, day, rule, "user", (prefix + "-owner", prefix + "-new-owner")
        ) == {prefix + "-owner"}
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id=$1 RETURNING id', prefix + "-new-owner"
        )
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET total_response_time_ms=total_response_time_ms/5 '
            "WHERE api_key=$1 AND date=$2 RETURNING id",
            keys[1],
            day.isoformat(),
        )
        assert await repository.latency_cohort(pairs, day, rule) is None
        assert await repository.cohort_recovered(pairs, day, rule)
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET total_response_time_ms=total_response_time_ms*5 '
            "WHERE api_key=$1 AND date=$2 RETURNING id",
            keys[1],
            day.isoformat(),
        )
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,updated_at)
            SELECT $1||'new'||m,$1||'-new-owner',$2,$1||'-new-key',$1||'-'||m,$1,50,50,2500000,CURRENT_TIMESTAMP
            FROM (VALUES('a'),('b')) models(m) RETURNING id""",
            prefix,
            day.isoformat(),
        )
        with_new_key = await repository.latency_cohort(pairs, day, rule)
        assert with_new_key is not None
        assert with_new_key.comparison == evidence.comparison
        assert (with_new_key.total_timed_requests, with_new_key.comparable_timed_requests) == (500, 400)
        assert (with_new_key.impact.keys, with_new_key.impact.users, with_new_key.impact.requests) == (3, 2, 500)
        assert with_new_key.impact.measured_keys == 2
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET timed_requests=51 WHERE api_key=$1 RETURNING id', prefix + "-new-key"
        )
        assert await repository.latency_cohort(pairs, day, rule) is None
        assert not await repository.cohort_recovered(pairs, day, rule)
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1 RETURNING id', prefix + "-new-key"
        )
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyUserSpend"
                (id,user_id,date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,updated_at)
            SELECT id||'-other',user_id,date,api_key,model,$1||'-other',1000,1000,100000,CURRENT_TIMESTAMP
            FROM "LiteLLM_DailyUserSpend" WHERE custom_llm_provider=$1 AND date=$2 RETURNING id""",
            prefix,
            day.isoformat(),
        )
        assert await repository.latency_cohort(pairs, day, rule) == evidence
        assert await repository.source_members(pairs, day, rule, "key", keys) == frozenset()
        assert await repository.source_members(pairs, day, rule, "model", models) == frozenset()
        assert await repository.source_members(pairs, day, rule, "user", (prefix + "-owner",)) == frozenset()
        assert await repository.source_members(pairs, day, rule, "deployment", ("deployment",)) == frozenset()
        assert await repository.source_members(pairs, day, rule, "team", teams) == frozenset(teams)
        await db.prisma.writer_db.query_raw(
            """INSERT INTO "LiteLLM_DailyTeamSpend"
                (id,team_id,date,api_key,model,custom_llm_provider,api_requests,timed_requests,total_response_time_ms,updated_at)
            SELECT id||'-other',team_id,date,api_key,model,$1||'-other',1000,1000,100000,CURRENT_TIMESTAMP
            FROM "LiteLLM_DailyTeamSpend" WHERE custom_llm_provider=$1 AND date=$2 RETURNING id""",
            prefix,
            day.isoformat(),
        )
        assert await repository.source_members(pairs, day, rule, "team", teams) == frozenset()
        await db.prisma.writer_db.query_raw(
            """UPDATE "LiteLLM_DailyUserSpend"
               SET api_requests=CASE WHEN model=$1||'-a' AND date<$2 THEN 10000
                                     WHEN model=$1||'-b' AND date=$2 THEN 5000 ELSE 100 END,
                   timed_requests=CASE WHEN model=$1||'-a' AND date<$2 THEN 10000
                                       WHEN model=$1||'-b' AND date=$2 THEN 5000 ELSE 100 END,
                   total_response_time_ms=(CASE WHEN model=$1||'-a' THEN 200 ELSE 1000 END)::bigint *
                       (CASE WHEN model=$1||'-a' AND date<$2 THEN 10000
                             WHEN model=$1||'-b' AND date=$2 THEN 5000 ELSE 100 END)
               WHERE custom_llm_provider=$1 RETURNING id""",
            prefix,
            day.isoformat(),
        )
        assert await repository.latency_cohort(pairs, day, rule) is None
        assert await repository.cohort_recovered(pairs, day, rule)
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET total_response_time_ms=total_response_time_ms*5 '
            "WHERE custom_llm_provider=$1 AND date=$2 RETURNING id",
            prefix,
            day.isoformat(),
        )
        assert await repository.latency_cohort(pairs, day, rule) is not None
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET total_response_time_ms=0 WHERE custom_llm_provider=$1 AND date<$2 RETURNING id',
            prefix,
            day.isoformat(),
        )
        assert await repository.latency_cohort(pairs, day, rule) is None
        assert not await repository.cohort_recovered(pairs, day, rule)
        await db.prisma.writer_db.query_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET total_response_time_ms=timed_requests*'
            "(CASE WHEN model=$1||'-a' THEN 200 ELSE 1000 END) "
            "WHERE custom_llm_provider=$1 AND date<$2 RETURNING id",
            prefix,
            day.isoformat(),
        )
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE custom_llm_provider=$1 AND date<($2::date-2)::text RETURNING id',
            prefix,
            day.isoformat(),
        )
        assert await repository.latency_cohort(pairs, day, rule) is None
        assert not await repository.cohort_recovered(pairs, day, rule)
    finally:
        for table in ('"LiteLLM_DailyUserSpend"', '"LiteLLM_DailyTeamSpend"'):
            await db.prisma.writer_db.query_raw(f"DELETE FROM {table} WHERE id LIKE $1 RETURNING id", prefix + "%")
