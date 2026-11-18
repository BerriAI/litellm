"""Warmed response-cache hits with 400 KB to 1.2 MB answers, 800 KB on average, for github issue #46276.
The input budget adds no file-reading turns, leaving the fixed system prompt and marked question within the
default four-message cache limit. Warmup and measurement replay the same cacheable population across versions.
The mock waits 4 s before answering misses. Locust accepts gzip, but whether answers are compressed depends
on the gateway version. The output cap and client-total p50 ceiling cover the large cached responses."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import (
    Limit,
    MockLatency,
    PerformanceTestCase,
    RequestShape,
    Team,
    Traffic,
    User,
    VirtualKey,
)

LARGE_ANSWER: Final = RequestShape("large_answer", input_tokens=(0, 0), output_tokens=400_000)

CASE: Final = PerformanceTestCase(
    name="large_bodies_300_users",
    teams=(Team("team-a", models=("gpt-6-luna",), max_budget=1_000_000.0),),
    users=(User("alice", team="team-a"), User("grace")),
    virtual_keys=(
        VirtualKey("key-a1", team="team-a"),
        VirtualKey("key-a2", team="team-a", user="alice"),
        VirtualKey("key-plain"),
    ),
    model_groups=("gpt-6-luna",),
    mock_latency=MockLatency(seconds=4.0, jitter_seconds=0.5),
    guardrails=(),
    traffic=Traffic(
        seed=29,
        rps=60.0,
        total_requests=1_200,
        endpoints={"chat_completions": 1.0},
        shapes={LARGE_ANSWER: 1.0},
        stream_share=0.0,
        concurrency=400,
    ),
    limits=(
        Limit("unexpected_status_count", max=3),
        Limit("client_total_seconds", percentile=50, max=0.3),
        Limit("cpu_seconds_per_request", max=0.5),
    ),
)
