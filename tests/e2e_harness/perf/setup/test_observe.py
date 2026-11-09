from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.small_bodies_5rps import CASE
from memory_readings import RssCapture
from setup.datastore import DbCounters, StatementCalls
from setup.observe import Contributor, Observation, breakdown, evaluate, junit_properties
from setup.performance_test_case import Limit
from setup.wire import WireRecord

NO_RSS: Final = RssCapture(readings=(), failures=())


def served(seq: int, total_s: float) -> WireRecord:
    return WireRecord(
        seq=seq,
        marker=f"perfreq-{seq:06d}-n",
        caller="key-a1",
        endpoint="chat_completions",
        stream=False,
        blocked=False,
        started_at=0.0,
        status=200,
        header_s=total_s,
        first_byte_s=total_s,
        first_token_s=None,
        total_s=total_s,
        text="",
        ok=True,
        mock_delay_s=1.0,
        call_id=None,
        error=None,
    )


def _key_lookup(calls: int) -> StatementCalls:
    return StatementCalls(
        queryid=1,
        query='SELECT * FROM "LiteLLM_VerificationToken" WHERE token = $1',
        calls=calls,
        total_exec_ms=calls * 2.0,
    )


def test_junit_properties_carry_each_observed_metric_and_each_limit_including_unobserved_ones() -> None:
    case: Final = replace(
        CASE,
        limits=(
            Limit("gateway_added_seconds", percentile=50, max=0.625),
            Limit("redis_calls_per_request", max=35),
        ),
    )
    observation: Final = Observation(
        records=(served(0, 1.25), served(1, 1.5), served(2, 1.75)),
        mock_requests=(),
        guardrail_calls=(),
        rss_before=NO_RSS,
        rss_after=NO_RSS,
        wall_seconds=3.0,
        db_before=DbCounters(commits=100, statements=(_key_lookup(calls=100),)),
        db_after=DbCounters(commits=130, statements=(_key_lookup(calls=130),)),
    )
    rows, _ = evaluate(case, observation)

    properties: Final = junit_properties(rows)

    assert len({name for name, _ in properties}) == len(properties)
    assert {name: value for name, value in properties if not name.endswith("_breakdown")} == {
        "perf_unexpected_status_count": 0.0,
        "perf_gateway_added_seconds_p50": 0.5,
        "perf_gateway_added_seconds_p50_limit": 0.625,
        "perf_gateway_added_seconds_p99": 0.75,
        "perf_postgres_calls_per_request": 10.0,
        "perf_postgres_background_calls_per_request": 0.0,
        "perf_postgres_max_statement_per_request": 10.0,
        "perf_postgres_max_background_statement_per_request": 0.0,
        "perf_postgres_active_sessions": 0.02,
        "perf_postgres_slowest_statement_seconds": 0.002,
        "perf_redis_calls_per_request_limit": 35.0,
    }


def _statement(queryid: int, query: str, calls: int) -> StatementCalls:
    return StatementCalls(queryid=queryid, query=query, calls=calls)


def _observation_with(statements: tuple[StatementCalls, ...]) -> Observation:
    before: Final = tuple(replace(item, calls=0) for item in statements)
    return Observation(
        records=tuple(served(seq, 1.0) for seq in range(1000)),
        mock_requests=(),
        guardrail_calls=(),
        rss_before=NO_RSS,
        rss_after=NO_RSS,
        wall_seconds=600.0,
        db_before=DbCounters(commits=0, statements=before),
        db_after=DbCounters(commits=0, statements=statements),
    )


MAX_STATEMENT_CASE: Final = replace(CASE, limits=(Limit("postgres_max_statement_per_request", max=0.5),))
BACKGROUND_CAP_CASE: Final = replace(
    CASE,
    limits=(Limit("postgres_max_background_statement_per_request", max=0.9),),
    background_cap_exempt_tables=frozenset({"LiteLLM_DailyGuardrailMetrics"}),
)
GUARDRAIL_UPSERT: Final = 'INSERT INTO "LiteLLM_DailyGuardrailMetrics" ("guardrail_id","date") VALUES ($1,$2)'
MODEL_USAGE_UPSERT: Final = 'INSERT INTO "LiteLLM_DailyModelUsage" ("date","model") VALUES ($1,$2)'
TEAM_REFRESH: Final = 'SELECT * FROM "LiteLLM_TeamMembership" WHERE user_id = $1'
USER_LOOKUP: Final = 'SELECT * FROM "LiteLLM_UserTable" WHERE user_id = $1'


def test_cache_refreshes_whose_sum_exceeds_the_cap_pass_the_per_statement_limit() -> None:
    statements: Final = (
        _statement(1, TEAM_REFRESH, calls=357),
        _statement(2, USER_LOOKUP, calls=245),
        _statement(3, 'INSERT INTO "LiteLLM_DailyModelUsage" VALUES ($1)', calls=1000),
    )

    rows, exceeded = evaluate(MAX_STATEMENT_CASE, _observation_with(statements))

    assert exceeded == []
    assert dict(junit_properties(rows))["perf_postgres_max_statement_per_request"] == 0.357
    assert dict(junit_properties(rows))["perf_postgres_calls_per_request"] == 0.602


def test_a_lookup_on_every_request_fails_the_per_statement_limit_and_names_the_statement() -> None:
    statements: Final = (_statement(1, TEAM_REFRESH, calls=357), _statement(2, USER_LOOKUP, calls=1000))

    _, exceeded = evaluate(MAX_STATEMENT_CASE, _observation_with(statements))

    assert exceeded == [f"postgres_max_statement_per_request: observed 1.0000 > limit 0.5000 ({USER_LOOKUP})"], exceeded


def test_per_statement_limit_without_db_counters_reports_nothing_observed() -> None:
    observation: Final = replace(_observation_with(()), db_before=None, db_after=None)

    _, exceeded = evaluate(MAX_STATEMENT_CASE, observation)

    assert exceeded == ["postgres_max_statement_per_request: limit set but nothing observed"]


def test_per_flush_rollups_pass_the_background_cap_and_the_exempt_table_is_skipped() -> None:
    statements: Final = (
        _statement(1, GUARDRAIL_UPSERT, calls=1150),
        _statement(2, MODEL_USAGE_UPSERT, calls=810),
        _statement(3, USER_LOOKUP, calls=1000),
    )

    rows, exceeded = evaluate(BACKGROUND_CAP_CASE, _observation_with(statements))

    assert exceeded == []
    assert dict(junit_properties(rows))["perf_postgres_max_background_statement_per_request"] == 0.81


def test_a_rollup_written_on_every_request_fails_the_background_cap_and_names_the_statement() -> None:
    statements: Final = (_statement(1, GUARDRAIL_UPSERT, calls=1050), _statement(2, MODEL_USAGE_UPSERT, calls=1000))

    _, exceeded = evaluate(BACKGROUND_CAP_CASE, _observation_with(statements))

    assert exceeded == [
        f"postgres_max_background_statement_per_request: observed 1.0000 > limit 0.9000 ({MODEL_USAGE_UPSERT})"
    ], exceeded


def test_background_cap_without_exemptions_counts_the_guardrail_upsert() -> None:
    case: Final = replace(BACKGROUND_CAP_CASE, background_cap_exempt_tables=frozenset())
    statements: Final = (_statement(1, GUARDRAIL_UPSERT, calls=1150), _statement(2, MODEL_USAGE_UPSERT, calls=810))

    _, exceeded = evaluate(case, _observation_with(statements))

    assert exceeded == [
        f"postgres_max_background_statement_per_request: observed 1.1500 > limit 0.9000 ({GUARDRAIL_UPSERT})"
    ], exceeded


def _guardrail_rollup(calls: int) -> StatementCalls:
    return StatementCalls(
        queryid=2,
        query='INSERT INTO "LiteLLM_DailyGuardrailMetrics" ("guardrail_id","date") VALUES ($1,$2) ON CONFLICT DO UPDATE',
        calls=calls,
        total_exec_ms=calls * 1.0,
    )


def _model_rollup(calls: int) -> StatementCalls:
    return StatementCalls(
        queryid=3,
        query='INSERT INTO "LiteLLM_DailyModelUsage" ("model","date") VALUES ($1,$2) ON CONFLICT DO UPDATE',
        calls=calls,
        total_exec_ms=calls * 1.0,
    )


def test_breakdown_names_the_slices_behind_each_metric_and_skips_plain_gauges() -> None:
    failed: Final = served(3, 1.0).model_copy(update={"status": 500, "ok": False})
    observation: Final = Observation(
        records=(served(0, 1.25), served(1, 1.5), served(2, 1.75), failed),
        mock_requests=(),
        guardrail_calls=(),
        rss_before=NO_RSS,
        rss_after=NO_RSS,
        wall_seconds=4.0,
        db_before=DbCounters(
            commits=100, statements=(_key_lookup(calls=100), _guardrail_rollup(calls=100), _model_rollup(calls=100))
        ),
        db_after=DbCounters(
            commits=130, statements=(_key_lookup(calls=130), _guardrail_rollup(calls=124), _model_rollup(calls=112))
        ),
        gateway_cpu_seconds=2.0,
    )

    busiest: Final = breakdown("postgres_max_background_statement_per_request", None, observation)
    assert [(item.value, item.share) for item in busiest] == [(6.0, 24 / 36), (3.0, 12 / 36)]
    assert "LiteLLM_DailyGuardrailMetrics" in busiest[0].label
    assert breakdown("gateway_added_seconds", 50, observation) == (Contributor("chat_completions", 0.5, 1.0),)
    assert breakdown("unexpected_status_count", None, observation) == (
        Contributor("chat_completions status 500", 1.0, 1.0),
    )
    assert breakdown("cpu_seconds_per_request", None, observation) == ()

    rows, _ = evaluate(CASE, observation)
    properties: Final = dict(junit_properties(rows))
    assert "perf_cpu_seconds_per_request_breakdown" not in properties
    assert properties["perf_unexpected_status_count_breakdown"] == (
        '[{"label": "chat_completions status 500", "value": 1.0, "share": 1.0}]'
    )
