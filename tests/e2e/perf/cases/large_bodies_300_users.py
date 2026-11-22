"""Hundreds of callers in flight at once, each answer 400 KB to 1.2 MB of JSON (800 KB on average) and gzipped
on the way out, the load of github issue #46276 (750 concurrent users, where v1.105.0 took 4 to 7 times as long
as v1.101.0) with the bodies that make every buffered copy and every compression pass show up. The mock waits 4 s
before it answers, so at 60 rps about 250 requests sit in the gateway at once and a middleware that holds or
copies a body costs every one of them. Locust's client asks for gzip on every request, so the gateway's gzip
middleware compresses each answer."""

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

LARGE_ANSWER: Final = RequestShape("large_answer", input_tokens=(2_500, 4_000), output_tokens=400_000)

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
        Limit("client_total_seconds", percentile=50, max=1.5),
        Limit("litellm_response_seconds", percentile=50, max=5.0),
        Limit("cpu_seconds_per_request", max=0.09),
    ),
)
