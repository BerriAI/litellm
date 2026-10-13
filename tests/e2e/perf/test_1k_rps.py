"""Live perf: the gateway serves 1k requests per second over mocked deployments of every major provider
that answer with provider-like delays, with zero failures, latency ceilings set from those delays plus a
gateway overhead allowance, and no worker growing its RSS past a flat ceiling.

Deselected unless E2E_PERF_TESTS is set.
"""

from __future__ import annotations

import math
import os
from statistics import mean, median
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
MOCK_DELAYS_SECONDS: Final = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
ENDPOINTS: Final = ("/chat/completions", "/v1/messages")
TARGET_RPS: Final = float(os.environ.get("E2E_PERF_TARGET_RPS", "1000"))
GATEWAY_OVERHEAD_ALLOWANCE_SECONDS: Final = 0.5
SHAPE: Final = LoadShape(
    users=int(
        os.environ.get(
            "E2E_PERF_USERS",
            math.ceil(TARGET_RPS * (mean(MOCK_DELAYS_SECONDS) + GATEWAY_OVERHEAD_ALLOWANCE_SECONDS) * 1.25),
        )
    ),
    spawn_rate=float(os.environ.get("E2E_PERF_SPAWN_RATE", "100")),
    warmup_seconds=float(os.environ.get("E2E_PERF_WARMUP_SECONDS", "15")),
    measured_seconds=float(os.environ.get("E2E_PERF_MEASURED_SECONDS", "60")),
    processes=int(os.environ.get("E2E_PERF_LOCUST_PROCESSES", "4")),
)
BUDGET: Final = PerfBudget(
    min_requests_per_second=TARGET_RPS,
    p50_ceiling_seconds=median(MOCK_DELAYS_SECONDS) + GATEWAY_OVERHEAD_ALLOWANCE_SECONDS,
    p90_ceiling_seconds=max(MOCK_DELAYS_SECONDS) + GATEWAY_OVERHEAD_ALLOWANCE_SECONDS,
    p99_ceiling_seconds=max(MOCK_DELAYS_SECONDS) + 2 * GATEWAY_OVERHEAD_ALLOWANCE_SECONDS,
    rss_growth_ceiling_mb=150.0,
)


class TestThousandRps:
    def test_gateway_holds_1k_rps_over_mocked_providers(self, perf_rig: PerfRig) -> None:
        group: Final = perf_rig.mocked_group(PROVIDER_MODELS, MOCK_DELAYS_SECONDS)
        run: Final = perf_rig.measure(group, ENDPOINTS, SHAPE)
        problems: Final = perf_rig.problems("1k rps", run, BUDGET)
        assert not problems, "; ".join((*problems, run.report()))
