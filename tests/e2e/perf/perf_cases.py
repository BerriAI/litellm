"""The perf cases the suite runs, one `PerformanceTestCase` each. Add a case here to add a perf test."""

from __future__ import annotations

from typing import Final

from perf_case import (
    ANTHROPIC,
    BEDROCK,
    OPENAI,
    Guardrail,
    Limit,
    MockDeployment,
    ModelGroup,
    PerformanceTestCase,
    Team,
    Traffic,
    User,
    VirtualKey,
)

MIXED_5RPS_SMALL_GATEWAY: Final = PerformanceTestCase(
    name="mixed_5rps_small_gateway",
    teams=(
        Team("team-a", models=("openai-group", "anthropic-group"), max_budget=50.0, rpm_limit=600),
        Team("team-b", models=("bedrock-group", "anthropic-group"), max_budget=20.0),
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
        User("judy", max_budget=5.0),
    ),
    virtual_keys=(
        VirtualKey("key-a1", team="team-a", models=("openai-group",), rpm_limit=120),
        VirtualKey("key-a2", team="team-a", user="alice"),
        VirtualKey("key-b1", team="team-b"),
        VirtualKey("key-b2", team="team-b", user="dave"),
        VirtualKey("key-plain"),
    ),
    model_groups=(
        ModelGroup(
            "openai-group",
            deployments=(
                MockDeployment(OPENAI),
                MockDeployment(OPENAI, latency_s=0.9),
                MockDeployment(OPENAI, latency_s=0.5, flake_percent=30, flake_status=529),
            ),
        ),
        ModelGroup(
            "anthropic-group",
            deployments=(
                MockDeployment(ANTHROPIC),
                MockDeployment(ANTHROPIC, latency_s=0.8),
            ),
        ),
        ModelGroup(
            "bedrock-group",
            deployments=(
                MockDeployment(BEDROCK),
                MockDeployment(BEDROCK, latency_s=1.0, bedrock_route="invoke"),
            ),
        ),
    ),
    guardrails=(
        Guardrail("perf-pre", mode="pre_call", latency_s=0.05),
        Guardrail("perf-post", mode="post_call", latency_s=0.05),
    ),
    traffic=Traffic(
        seed=7,
        rps=5.0,
        total_requests=1500,
        endpoints={"chat_completions": 0.5, "responses": 0.25, "messages": 0.25},
        stream_share=0.5,
        intervene_share=0.1,
        block_share=0.02,
        concurrency=20,
    ),
    limits=(
        Limit("added_latency_s", percentile=50, max=0.25),
        Limit("added_latency_s", percentile=99, max=1.0),
        Limit("header_added_s", percentile=95, max=0.5),
        Limit("ttft_added_s", percentile=95, max=0.5),
        Limit("client_5xx", max=0),
        Limit("incorrect_responses", max=0),
        Limit("guardrail_missed", max=0),
        Limit("rss_growth_mb", max=150),
        Limit("wall_seconds", max=360),
    ),
)

CASES: Final[tuple[PerformanceTestCase, ...]] = (MIXED_5RPS_SMALL_GATEWAY,)
