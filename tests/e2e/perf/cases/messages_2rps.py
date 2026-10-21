"""`/v1/messages` at 2 rps: `claude-sonnet-5` is served by the Anthropic mock's Messages route,
`claude-sonnet-5-bedrock-converse` (`bedrock/converse/anthropic.claude-sonnet-5`) is bridged to the
Bedrock mock's Converse route."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import Limit

from .direct_vs_bridge import direct_vs_bridge

CASE: Final = direct_vs_bridge(
    "messages_2rps",
    "messages",
    direct="claude-sonnet-5",
    bridge="claude-sonnet-5-bedrock-converse",
    limits=(
        Limit("unexpected_status_count", max=3),
        Limit("gateway_added_seconds", percentile=50, max=0.296),
        Limit("gateway_added_seconds", percentile=99, max=3.36),
        Limit("litellm_overhead_seconds", percentile=50, max=0.214),
        Limit("litellm_overhead_seconds", percentile=99, max=3.15),
        Limit("litellm_response_seconds", percentile=50, max=1.18),
        Limit("litellm_response_seconds", percentile=99, max=3.31),
        Limit("prometheus_redis_latency_mean_seconds", max=0.00216),
        Limit("postgres_active_sessions", max=0.00816),
        Limit("postgres_slowest_statement_seconds", max=0.042),
        Limit("postgres_peak_connections", max=24),
        Limit("time_to_first_token_seconds", percentile=50, max=1.56),
        Limit("time_to_first_token_seconds", percentile=99, max=5.07),
        Limit("memory_growth_megabytes", max=600),
        Limit("cpu_seconds_per_request", max=0.74),
        Limit("postgres_calls_per_request", max=2.59),
        Limit("postgres_max_statement_per_request", max=0.427),
        Limit("postgres_max_background_statement_per_request", max=0.856),
        Limit("redis_calls_per_request", max=18.1),
    ),
)
