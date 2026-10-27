"""Mixed-size requests at 2 rps on the small gateway: three real model groups over the mock providers,
two teams plus users in no team, JWT and key callers, a pre and
a post guardrail, half the requests streaming, short chats, long contexts and tool calls. Budgets
are real money at cost-map prices (a 335k-token `claude-sonnet-5` request is about $0.70 and the
run spends roughly $600), so they are sized so no caller runs dry mid-run."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import (
    LONG,
    SHORT,
    Guardrail,
    Limit,
    MockLatency,
    PerformanceTestCase,
    Team,
    Traffic,
    User,
    VirtualKey,
)

CASE: Final = PerformanceTestCase(
    name="mixed_size_requests_2rps",
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
        seed=7,
        rps=2.0,
        total_requests=1200,
        endpoints={"chat_completions": 0.5, "responses": 0.25, "messages": 0.25},
        shapes={SHORT: 0.75, LONG: 0.25},
        stream_share=0.5,
        intervene_share=0.0,
        block_share=0.0,
        concurrency=100,
    ),
    limits=(
        Limit("unexpected_status_count", max=3),
        Limit("gateway_added_seconds", percentile=50, max=0.29),
        Limit("gateway_added_seconds", percentile=99, max=2.7),
        Limit("litellm_overhead_seconds", percentile=50, max=0.052),
        Limit("litellm_overhead_seconds", percentile=99, max=1.8),
        Limit("litellm_response_seconds", percentile=50, max=0.93),
        Limit("litellm_response_seconds", percentile=99, max=2.7),
        Limit("prometheus_redis_latency_mean_seconds", max=0.002),
        Limit("postgres_active_sessions", max=0.0094),
        Limit("postgres_slowest_statement_seconds", max=0.042),
        Limit("postgres_peak_connections", max=23),
        Limit("time_to_first_token_seconds", percentile=50, max=1.56),
        Limit("time_to_first_token_seconds", percentile=99, max=4.5),
        Limit("memory_growth_megabytes", max=600),
        Limit("cpu_seconds_per_request", max=0.57),
        Limit("postgres_calls_per_request", max=2.64),
        Limit("postgres_max_statement_per_request", max=0.43),
        Limit("postgres_max_background_statement_per_request", max=0.86),
        Limit("redis_calls_per_request", max=15.4),
    ),
    background_cap_exempt_tables=frozenset({"LiteLLM_DailyGuardrailMetrics"}),
)
