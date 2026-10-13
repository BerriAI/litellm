"""Mixed traffic at 50 rps on the small gateway: three real model groups over the mock providers
(one `gpt-6-luna` deployment flaky), two teams plus users in no team, JWT and key callers, a pre and
a post guardrail, half the requests streaming, short chats, long contexts and tool calls. Budgets
are real money at cost-map prices (a 335k-token `claude-sonnet-5` request is about $0.70 and the
run spends roughly $3,000), so they are sized so no caller runs dry mid-run."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import (
    LONG_CONTEXT,
    SHORT_CHAT,
    TOOL_CALL,
    Guardrail,
    Limit,
    PerformanceTestCase,
    Team,
    Traffic,
    User,
    VirtualKey,
)

CASE: Final = PerformanceTestCase(
    name="mixed_50rps_small_gateway",
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
    guardrails=(Guardrail("perf-pre", mode="pre_call"), Guardrail("perf-post", mode="post_call")),
    traffic=Traffic(
        seed=7,
        rps=50.0,
        total_requests=15000,
        endpoints={"chat_completions": 0.5, "responses": 0.25, "messages": 0.25},
        shapes={SHORT_CHAT: 0.6, LONG_CONTEXT: 0.2, TOOL_CALL: 0.2},
        stream_share=0.5,
        intervene_share=0.1,
        block_share=0.02,
        concurrency=600,
    ),
    limits=(
        Limit("added_latency_s", percentile=50, max=0.25),
        Limit("added_latency_s", percentile=99, max=1.0),
        Limit("header_added_s", percentile=95, max=0.5),
        Limit("ttft_added_s", percentile=95, max=0.5),
        Limit("non_200", max=0),
        Limit("rss_growth_mb", max=150),
        Limit("wall_seconds", max=360),
    ),
)
