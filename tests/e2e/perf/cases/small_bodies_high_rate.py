"""Small bodies at a high rate on the small gateway: the shape of github issue #46276 (short chat
requests, master key and virtual keys, no JWT, no guardrails, a few hundred rps on one 4-worker
pod), where the gateway's own per-request overhead is the whole number and a slower middleware
chain or an extra Redis round trip per request shows up directly."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import (
    Limit,
    PerformanceTestCase,
    RequestShape,
    Team,
    Traffic,
    VirtualKey,
)

TINY: Final = RequestShape("tiny", input_tokens=(2_500, 4_000))

CASE: Final = PerformanceTestCase(
    name="small_bodies_high_rate",
    teams=(Team("team-a", models=("gpt-6-luna", "claude-sonnet-5", "claude-sonnet-5-bedrock"), max_budget=5000.0),),
    users=(),
    virtual_keys=(
        VirtualKey("key-a1", team="team-a"),
        VirtualKey("key-a2", team="team-a"),
        VirtualKey("key-plain"),
    ),
    model_groups=("gpt-6-luna", "claude-sonnet-5", "claude-sonnet-5-bedrock"),
    guardrails=(),
    traffic=Traffic(
        seed=11,
        rps=200.0,
        total_requests=60000,
        endpoints={"chat_completions": 0.6, "responses": 0.2, "messages": 0.2},
        shapes={TINY: 1.0},
        stream_share=0.5,
        intervene_share=0.0,
        block_share=0.0,
        concurrency=400,
    ),
    limits=(
        Limit("unexpected_status_count", max=0),
        Limit("gateway_added_seconds", percentile=50, max=0.25),
        Limit("gateway_added_seconds", percentile=99, max=1.0),
        Limit("time_to_first_token_seconds", percentile=50, max=1.0),
        Limit("time_to_first_token_seconds", percentile=99, max=2.0),
        Limit("memory_growth_megabytes", max=300),
    ),
)
