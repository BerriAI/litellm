"""Small bodies at a low rate on the small gateway: the request sizes of github issue #46276 (short
chat requests, half streamed) from the same callers as the mixed case (two teams, JWT users in and
out of a team, virtual keys, the pre and post guardrail) at 5 rps on one 4-worker pod, far under
what the pod sustains, so nothing queues and gateway_added_seconds is the gateway's own per-request
overhead. A slower middleware chain or an extra Redis round trip per request shows up directly
instead of hiding behind queue wait."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import (
    Guardrail,
    Limit,
    MockLatency,
    PerformanceTestCase,
    RequestShape,
    Team,
    Traffic,
    User,
    VirtualKey,
)

TINY: Final = RequestShape("tiny", input_tokens=(2_500, 4_000))

CASE: Final = PerformanceTestCase(
    name="small_bodies_5rps",
    teams=(
        Team("team-a", models=("gpt-6-luna", "claude-sonnet-5"), max_budget=5000.0, rpm_limit=6000),
        Team("team-b", models=("claude-sonnet-5-bedrock", "claude-sonnet-5"), max_budget=5000.0),
    ),
    users=(
        User("alice", team="team-a"),
        User("bob", team="team-a"),
        User("carol", team="team-a"),
        User("dave", team="team-b"),
        User("erin", team="team-b"),
        User("frank", team="team-b"),
        User("grace"),
        User("heidi"),
        User("ivan"),
        User("judy", max_budget=1000.0),
    ),
    virtual_keys=(
        VirtualKey("key-a1", team="team-a", models=("gpt-6-luna",), rpm_limit=1200),
        VirtualKey("key-a2", team="team-a", user="alice"),
        VirtualKey("key-b1", team="team-b"),
        VirtualKey("key-b2", team="team-b", user="dave"),
        VirtualKey("key-plain"),
    ),
    model_groups=("gpt-6-luna", "claude-sonnet-5", "claude-sonnet-5-bedrock"),
    mock_latency=MockLatency(seconds=1.0, jitter_seconds=0.25),
    guardrails=(Guardrail("perf-pre", mode="pre_call"), Guardrail("perf-post", mode="post_call")),
    traffic=Traffic(
        seed=11,
        rps=5.0,
        total_requests=3000,
        endpoints={"chat_completions": 0.6, "responses": 0.2, "messages": 0.2},
        shapes={TINY: 1.0},
        stream_share=0.5,
        intervene_share=0.0,
        block_share=0.0,
        concurrency=20,
    ),
    limits=(
        Limit("unexpected_status_count", max=3),
        Limit("gateway_added_seconds", percentile=50, max=0.165),
        Limit("litellm_overhead_seconds", percentile=50, max=0.006),
        Limit("litellm_overhead_seconds", percentile=99, max=0.35),
        Limit("litellm_response_seconds", percentile=50, max=0.91),
        Limit("litellm_response_seconds", percentile=99, max=1.6),
        Limit("prometheus_redis_latency_mean_seconds", max=0.0026),
        Limit("postgres_active_sessions", max=0.0052),
        Limit("postgres_slowest_statement_seconds", max=0.011),
        Limit("postgres_peak_connections", max=23),
        Limit("time_to_first_token_seconds", percentile=99, max=2.05),
        Limit("memory_growth_megabytes", max=100),
        Limit("postgres_calls_per_request", max=1.31),
        Limit("postgres_max_statement_per_request", max=0.27),
        Limit("postgres_max_background_statement_per_request", max=0.9),
        Limit("redis_calls_per_request", max=13.8),
        Limit("cpu_seconds_per_request", max=0.15),
    ),
)
