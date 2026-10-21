"""`/v1/chat/completions` at 2 rps: `gpt-6-luna` is served by the OpenAI mock's chat completions
route, `gpt-6-luna-responses` (`openai/responses/gpt-6-luna`) is bridged to its Responses route."""

from __future__ import annotations

from typing import Final

from setup.performance_test_case import Limit

from .direct_vs_bridge import direct_vs_bridge

CASE: Final = direct_vs_bridge(
    "chat_completions_2rps",
    "chat_completions",
    direct="gpt-6-luna",
    bridge="gpt-6-luna-responses",
    limits=(
        Limit("unexpected_status_count", max=3),
        Limit("gateway_added_seconds", percentile=50, max=0.326),
        Limit("gateway_added_seconds", percentile=99, max=2.23),
        Limit("litellm_overhead_seconds", percentile=50, max=0.0372),
        Limit("litellm_overhead_seconds", percentile=99, max=1.53),
        Limit("litellm_response_seconds", percentile=50, max=1.05),
        Limit("litellm_response_seconds", percentile=99, max=3.06),
        Limit("prometheus_redis_latency_mean_seconds", max=0.00372),
        Limit("postgres_active_sessions", max=0.00636),
        Limit("postgres_slowest_statement_seconds", max=0.042),
        Limit("postgres_peak_connections", max=24),
        Limit("time_to_first_token_seconds", percentile=50, max=1.62),
        Limit("time_to_first_token_seconds", percentile=99, max=3.66),
        Limit("memory_growth_megabytes", max=600),
        Limit("cpu_seconds_per_request", max=0.577),
        Limit("postgres_calls_per_request", max=2.6),
        Limit("postgres_max_statement_per_request", max=0.427),
        Limit("postgres_max_background_statement_per_request", max=1.08),
        Limit("redis_calls_per_request", max=15.0),
    ),
)
