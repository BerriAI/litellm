"""Live perf: the gateway serves 1k requests per second over mocked deployments of every major provider
with zero failures, flat latency ceilings and no worker growing its RSS past a flat ceiling.

Deselected unless E2E_PERF_TESTS is set.
"""

from __future__ import annotations

import os
from typing import Final

import pytest
from perf_rig import LoadShape, PerfBudget, PerfRig

pytestmark: Final = [pytest.mark.e2e, pytest.mark.perf]

PROVIDER_MODELS: Final = (
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-5",
    "azure/gpt-5.1",
    "bedrock/converse/us.anthropic.claude-sonnet-5",
    "vertex_ai/gemini-3-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
)
ENDPOINTS: Final = ("/chat/completions", "/v1/messages")
SHAPE: Final = LoadShape(
    users=int(os.environ.get("E2E_PERF_USERS", "200")),
    spawn_rate=float(os.environ.get("E2E_PERF_SPAWN_RATE", "100")),
    warmup_seconds=float(os.environ.get("E2E_PERF_WARMUP_SECONDS", "15")),
    measured_seconds=float(os.environ.get("E2E_PERF_MEASURED_SECONDS", "60")),
    processes=int(os.environ.get("E2E_PERF_LOCUST_PROCESSES", "4")),
)
BUDGET: Final = PerfBudget(
    min_requests_per_second=float(os.environ.get("E2E_PERF_TARGET_RPS", "1000")),
    p50_ceiling_seconds=1.0,
    p90_ceiling_seconds=2.0,
    p99_ceiling_seconds=3.0,
    rss_growth_ceiling_mb=150.0,
)


class TestThousandRps:
    def test_gateway_holds_1k_rps_over_mocked_providers(self, perf_rig: PerfRig) -> None:
        group: Final = perf_rig.mocked_group(PROVIDER_MODELS)
        run: Final = perf_rig.measure(group, ENDPOINTS, SHAPE)
        perf_rig.check("1k rps", run, BUDGET)
