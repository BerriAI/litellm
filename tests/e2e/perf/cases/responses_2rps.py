"""`/v1/responses` at 2 rps: `gpt-6-luna` is served by the OpenAI mock's Responses route,
`claude-sonnet-5` is bridged to the Anthropic mock's Messages route."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import Limit

from .direct_vs_bridge import direct_vs_bridge

CASE: Final = direct_vs_bridge(
    "responses_2rps",
    "responses",
    direct="gpt-6-luna",
    bridge="claude-sonnet-5",
    limits=(
        Limit("unexpected_status_count", max=3),
        Limit("gateway_added_seconds", percentile=50, max=0.24),
        Limit("gateway_added_seconds", percentile=99, max=2.05),
        Limit("litellm_overhead_seconds", percentile=50, max=0.153),
        Limit("litellm_overhead_seconds", percentile=99, max=1.8),
        Limit("litellm_response_seconds", percentile=50, max=1.11),
        Limit("litellm_response_seconds", percentile=99, max=3.27),
        Limit("prometheus_redis_latency_mean_seconds", max=0.00228),
        Limit("postgres_active_sessions", max=0.011),
        Limit("postgres_slowest_statement_seconds", max=0.042),
        Limit("postgres_peak_connections", max=26),
        Limit("time_to_first_token_seconds", percentile=50, max=1.43),
        Limit("time_to_first_token_seconds", percentile=99, max=2.69),
        Limit("memory_growth_megabytes", max=600),
        Limit("cpu_seconds_per_request", max=0.382),
        Limit("postgres_calls_per_request", max=2.59),
        Limit("postgres_max_statement_per_request", max=0.426),
        Limit("postgres_max_background_statement_per_request", max=1.51),
        Limit("redis_calls_per_request", max=15.7),
    ),
)
