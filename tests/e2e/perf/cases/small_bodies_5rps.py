"""Small bodies at a low rate on the small gateway: the shape of github issue #46276 (short chat
requests, master key and virtual keys, no JWT, no guardrails) at 5 rps on one 4-worker pod, far
under what the pod sustains, so nothing queues and gateway_added_seconds is the gateway's own
per-request overhead. A slower middleware chain or an extra Redis round trip per request shows up
directly instead of hiding behind queue wait."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import (
    Limit,
    MockLatency,
    PerformanceTestCase,
    RequestShape,
    Team,
    Traffic,
    VirtualKey,
)

TINY: Final = RequestShape("tiny", input_tokens=(2_500, 4_000))

CASE: Final = PerformanceTestCase(
    name="small_bodies_5rps",
    teams=(Team("team-a", models=("gpt-6-luna", "claude-sonnet-5", "claude-sonnet-5-bedrock")),),
    users=(),
    virtual_keys=(
        VirtualKey("key-a1", team="team-a"),
        VirtualKey("key-a2", team="team-a"),
        VirtualKey("key-plain"),
    ),
    model_groups=("gpt-6-luna", "claude-sonnet-5", "claude-sonnet-5-bedrock"),
    mock_latency=MockLatency(seconds=1.0, jitter_seconds=0.25),
    guardrails=(),
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
        Limit("gateway_added_seconds", percentile=50, max=0.38),
        Limit("time_to_first_token_seconds", percentile=99, max=3.0),
        Limit("memory_growth_megabytes", max=100),
        Limit("postgres_calls_per_request", max=13),
        Limit("redis_calls_per_request", max=35),
        Limit("cpu_seconds_per_request", max=0.19),
    ),
)
