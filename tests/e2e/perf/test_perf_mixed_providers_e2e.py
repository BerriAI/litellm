"""Live perf: one model group spread over mocked deployments of every major provider stays
fast and lean under Locust load.

Six deployments share one model group, one per provider family (OpenAI, Anthropic, Azure,
Bedrock converse, Vertex AI, Gemini), each registered through /model/new with a
`mock_response`, so the proxy runs every request's auth, routing, logging and spend path
while the provider call itself returns instantly and costs nothing. Traffic is split round
robin over /chat/completions, /v1/messages and /v1/responses, one endpoint per simulated
user, so each route's own request path is under load, not just chat completions.

A warm-up window runs first and is discarded, so cold caches and worker spin-up never land
in the numbers. The measured window then has to hold zero failed requests, flat latency
ceilings, and a flat ceiling on how much any one worker's RSS grew across it, read over
the network from /debug/memory/summary so the proxy can run anywhere (a pod, a sidecar,
a laptop). The measured numbers are printed so the build log carries them.

Deselected unless E2E_PERF_TESTS is set.
"""

from __future__ import annotations

import os
from typing import Final

import pytest
from e2e_config import PROXY_BASE_URL, unique_marker
from lifecycle import ResourceManager
from load_client import LoadClient
from locust_load import LoadResult, run_gateway_load
from memory_readings import RssReading, WorkerKey, read_rss_everywhere
from models import KeyGenerateBody, LiteLLMParamsBody
from phase_budget import AbsoluteBudget, Budget, violations
from proxy_client import ProxyClient

pytestmark: Final = [pytest.mark.e2e, pytest.mark.perf]

STEP: Final = "perf-load-mixed-providers"
PROVIDER_MODELS: Final = (
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-5",
    "azure/gpt-5.1",
    "bedrock/converse/us.anthropic.claude-sonnet-5",
    "vertex_ai/gemini-3-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
)
ENDPOINTS: Final = ("/chat/completions", "/v1/messages", "/v1/responses")
KEY_POOL_SIZE: Final = 8
LOCUST_USERS: Final = int(os.environ.get("E2E_PERF_USERS", "50"))
LOCUST_SPAWN_RATE: Final = float(os.environ.get("E2E_PERF_SPAWN_RATE", "50"))
WARMUP_SECONDS: Final = float(os.environ.get("E2E_PERF_WARMUP_SECONDS", "15"))
MEASURED_SECONDS: Final = float(os.environ.get("E2E_PERF_MEASURED_SECONDS", "60"))
RSS_SAMPLE_READS: Final = 32

# Flat ceilings, calibrated loosely on purpose for the first runs: the compare step across
# builds is what catches drift, these only catch a run that is broken outright.
P50_CEILING_SECONDS: Final = 1.0
P90_CEILING_SECONDS: Final = 2.0
P99_CEILING_SECONDS: Final = 3.0
RSS_GROWTH_CEILING_MB: Final = 150.0


def _mock_params(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, api_key="sk-perf-not-used", mock_response="perf ok")


def _sample_workers(proxy: ProxyClient) -> dict[WorkerKey, RssReading]:
    """One RSS reading per worker. A multi-worker proxy answers /debug/memory/summary from
    whichever worker took the connection, so one read sees one worker; reading several
    times lands on each of them, keyed by replica, host and pid."""
    captures: Final = tuple(read_rss_everywhere(proxy) for _ in range(RSS_SAMPLE_READS))
    return {
        reading.worker: reading for capture in captures for reading in capture.readings
    }  # comprehension-ok: flattening one level of readings


def _worst_worker_growth_mb(before: dict[WorkerKey, RssReading], after: dict[WorkerKey, RssReading]) -> float:
    """The largest RSS increase of any worker seen in both samples, so one leaking worker
    cannot hide behind the average and a worker seen only once is not counted."""
    growths: Final = tuple(
        after[worker].ram_usage_mb - before[worker].ram_usage_mb for worker in after if worker in before
    )
    assert growths, (
        f"no worker answered /debug/memory/summary both before and after the load; "
        f"before: {sorted(before)}, after: {sorted(after)}"
    )
    return max(growths)


def _budgets(load: LoadResult, rss_growth_mb: float) -> tuple[Budget, ...]:
    return (
        AbsoluteBudget(
            name="p50 latency", measured=load.p50_seconds, ceiling=P50_CEILING_SECONDS, unit="s", decimals=3
        ),
        AbsoluteBudget(
            name="p90 latency", measured=load.p90_seconds, ceiling=P90_CEILING_SECONDS, unit="s", decimals=3
        ),
        AbsoluteBudget(
            name="p99 latency", measured=load.p99_seconds, ceiling=P99_CEILING_SECONDS, unit="s", decimals=3
        ),
        AbsoluteBudget(
            name="worst worker RSS growth", measured=rss_growth_mb, ceiling=RSS_GROWTH_CEILING_MB, unit=" MB"
        ),
    )


class TestMixedProviderLoad:
    def test_mixed_provider_group_holds_its_budgets_under_load(
        self, client: LoadClient, resources: ResourceManager
    ) -> None:
        group: Final = f"perf-mixed-{unique_marker()}"
        for model in PROVIDER_MODELS:
            model_id = client.proxy.create_model(group, _mock_params(model))
            resources.defer(lambda model_id=model_id: client.proxy.delete_model(model_id))
        keys: Final = tuple(
            client.proxy.generate_key(KeyGenerateBody(models=[group], key_alias=f"{group}-{index}"))
            for index in range(KEY_POOL_SIZE)
        )
        for key in keys:
            resources.defer(lambda key=key: client.proxy.delete_key(key))

        def drive(seconds: float) -> LoadResult:
            return run_gateway_load(
                base_url=PROXY_BASE_URL,
                api_keys=keys,
                model=group,
                endpoints=ENDPOINTS,
                users=LOCUST_USERS,
                spawn_rate=LOCUST_SPAWN_RATE,
                duration_seconds=seconds,
            )

        warmup: Final = drive(WARMUP_SECONDS)
        assert warmup.requests > 0, f"warm-up sent nothing: {warmup.diagnosis()}"
        before: Final = _sample_workers(client.proxy)
        load: Final = drive(MEASURED_SECONDS)
        after: Final = _sample_workers(client.proxy)

        rss_growth_mb: Final = _worst_worker_growth_mb(before, after)
        failed: Final = violations(_budgets(load, rss_growth_mb))
        report: Final = (
            f"{load.requests} requests, {load.failures} failures, {load.requests_per_second:.0f} rps, "
            f"{load.latency_summary()}, worst worker RSS growth {rss_growth_mb:.1f} MB; "
            f"by endpoint: {load.endpoint_summary()}"
        )
        print(f"\nmixed provider load: {report}")  # noqa: T201  # the numbers this test exists to report, read off the CI log
        assert load.failures == 0, f"{load.failures} requests failed: {load.diagnosis()}; {report}"
        assert not load.generator_warnings, f"locust saturated itself, run invalid: {load.generator_warnings}"
        assert not failed, "; ".join((*failed, report))
