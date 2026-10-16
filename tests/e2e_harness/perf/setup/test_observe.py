from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.small_bodies_5rps import CASE
from memory_readings import RssCapture
from setup.datastore import DbCounters
from setup.observe import Observation, evaluate, junit_properties
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
        db_before=DbCounters(commits=100),
        db_after=DbCounters(commits=130),
    )
    rows, _ = evaluate(case, observation)

    properties: Final = junit_properties(rows)

    assert len({name for name, _ in properties}) == len(properties)
    assert dict(properties) == {
        "perf_unexpected_status_count": 0.0,
        "perf_gateway_added_seconds_p50": 0.5,
        "perf_gateway_added_seconds_p50_limit": 0.625,
        "perf_gateway_added_seconds_p99": 0.75,
        "perf_postgres_calls_per_request": 10.0,
        "perf_redis_calls_per_request_limit": 35.0,
    }
