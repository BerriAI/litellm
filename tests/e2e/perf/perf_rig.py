"""The setup every perf test runs on, so a test is a few lines naming what it loads and what it must hold.

`PerfRig.mocked_group` registers one model group over deployments that answer with `mock_response`, so
the proxy runs auth, routing, logging and spend for every request while the provider call costs nothing,
plus a pool of keys scoped to it, all torn down by `ResourceManager`. `PerfRig.measure` drives Locust
through a warm-up phase that is discarded and a measured phase that is kept, reading every worker's RSS
over the network from /debug/memory/summary before and after so the proxy can run anywhere (new pods that replaced
the old ones mid-load, an upgrade under traffic, are measured against the old pods' median). `PerfRig.problems`
reports what the run broke in one `PerfBudget` and prints the numbers so the build log carries them
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Final

from e2e_config import PROXY_BASE_URL, unique_marker
from lifecycle import ResourceManager
from locust_load import LoadResult, run_gateway_load
from memory_readings import RssReading, WorkerKey, read_rss_everywhere
from models import KeyGenerateBody, LiteLLMParamsBody
from phase_budget import AbsoluteBudget, Budget, violations
from proxy_client import ProxyClient

RSS_SAMPLE_READS: Final = 32


@dataclass(frozen=True, slots=True)
class MockedGroup:
    name: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadShape:
    users: int
    spawn_rate: float
    warmup_seconds: float
    measured_seconds: float
    processes: int = 1


@dataclass(frozen=True, slots=True)
class PerfRun:
    load: LoadResult
    rss_growth_mb: float

    def report(self) -> str:
        return (
            f"{self.load.requests} requests, {self.load.failures} failures, "
            f"{self.load.requests_per_second:.0f} rps, {self.load.latency_summary()}, "
            f"worst worker RSS growth {self.rss_growth_mb:.1f} MB; by endpoint: {self.load.endpoint_summary()}"
        )


@dataclass(frozen=True, slots=True)
class PerfBudget:
    """Flat limits a measured phase must hold on its own, with no earlier run to compare against."""

    min_requests_per_second: float
    p50_ceiling_seconds: float
    p90_ceiling_seconds: float
    p99_ceiling_seconds: float
    rss_growth_ceiling_mb: float

    def violations(self, run: PerfRun) -> tuple[str, ...]:
        load: Final = run.load
        ceilings: Final[tuple[Budget, ...]] = (
            AbsoluteBudget(
                name="p50 latency", measured=load.p50_seconds, ceiling=self.p50_ceiling_seconds, unit="s", decimals=3
            ),
            AbsoluteBudget(
                name="p90 latency", measured=load.p90_seconds, ceiling=self.p90_ceiling_seconds, unit="s", decimals=3
            ),
            AbsoluteBudget(
                name="p99 latency", measured=load.p99_seconds, ceiling=self.p99_ceiling_seconds, unit="s", decimals=3
            ),
            AbsoluteBudget(
                name="worst worker RSS growth",
                measured=run.rss_growth_mb,
                ceiling=self.rss_growth_ceiling_mb,
                unit=" MB",
            ),
        )
        throughput: Final = (
            ()
            if load.requests_per_second >= self.min_requests_per_second
            else (
                f"throughput {load.requests_per_second:.0f} rps is under the {self.min_requests_per_second:.0f} rps floor",
            )
        )
        return (*throughput, *violations(ceilings))


def _mock_params(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, api_key="sk-perf-not-used", mock_response="perf ok")


def _sample_workers(proxy: ProxyClient) -> dict[WorkerKey, RssReading]:
    """One reading per worker. Each read of /debug/memory/summary lands on whichever worker took the
    connection, so repeated reads are what covers all of them, keyed by replica, host and pid."""
    captures: Final = tuple(read_rss_everywhere(proxy) for _ in range(RSS_SAMPLE_READS))
    return {
        reading.worker: reading for capture in captures for reading in capture.readings
    }  # comprehension-ok: flattening one level of readings


def _worst_worker_growth_mb(before: dict[WorkerKey, RssReading], after: dict[WorkerKey, RssReading]) -> float:
    """Growth per worker when the same workers answered before and after. When none did, the pods were
    replaced during the load (an upgrade rolling under traffic), so every new worker is measured against
    the median old worker instead: the new version under load may not sit more than the ceiling above the
    old one at warm-up."""
    assert before and after, (
        f"no worker answered /debug/memory/summary; before: {sorted(before)}, after: {sorted(after)}"
    )
    paired: Final = tuple(
        after[worker].ram_usage_mb - before[worker].ram_usage_mb for worker in after if worker in before
    )
    if paired:
        return max(paired)
    baseline: Final = median(reading.ram_usage_mb for reading in before.values())
    return max(reading.ram_usage_mb for reading in after.values()) - baseline


@dataclass(frozen=True, slots=True)
class PerfRig:
    proxy: ProxyClient
    resources: ResourceManager

    def mocked_group(self, models: tuple[str, ...], *, key_pool_size: int = 8) -> MockedGroup:
        name: Final = f"perf-{unique_marker()}"
        for model in models:
            model_id = self.proxy.create_model(name, _mock_params(model))
            self.resources.defer(lambda model_id=model_id: self.proxy.delete_model(model_id))
        keys: Final = tuple(
            self.proxy.generate_key(KeyGenerateBody(models=[name], key_alias=f"{name}-{index}"))
            for index in range(key_pool_size)
        )
        for key in keys:
            self.resources.defer(lambda key=key: self.proxy.delete_key(key))
        return MockedGroup(name=name, keys=keys)

    def measure(self, group: MockedGroup, endpoints: tuple[str, ...], shape: LoadShape) -> PerfRun:
        def drive(seconds: float) -> LoadResult:
            return run_gateway_load(
                base_url=PROXY_BASE_URL,
                api_keys=group.keys,
                model=group.name,
                endpoints=endpoints,
                users=shape.users,
                spawn_rate=shape.spawn_rate,
                duration_seconds=seconds,
                processes=shape.processes,
            )

        warmup: Final = drive(shape.warmup_seconds)
        assert warmup.requests > 0, f"warm-up sent nothing: {warmup.diagnosis()}"
        before: Final = _sample_workers(self.proxy)
        load: Final = drive(shape.measured_seconds)
        after: Final = _sample_workers(self.proxy)
        return PerfRun(load=load, rss_growth_mb=_worst_worker_growth_mb(before, after))

    def problems(self, label: str, run: PerfRun, budget: PerfBudget) -> tuple[str, ...]:
        """Everything wrong with the run, empty when it held the budget, printed so the build log carries the numbers."""
        report: Final = run.report()
        print(f"\n{label}: {report}")  # noqa: T201  # the numbers the perf suite exists to report, read off the CI log
        failures: Final = (f"{run.load.failures} requests failed: {run.load.diagnosis()}",) if run.load.failures else ()
        saturated: Final = (
            (f"locust saturated itself, run invalid: {run.load.generator_warnings}",)
            if run.load.generator_warnings
            else ()
        )
        return (*failures, *saturated, *budget.violations(run))
