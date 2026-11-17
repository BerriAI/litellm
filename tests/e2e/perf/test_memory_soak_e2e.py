"""Live e2e: a soak of mixed traffic, served, failing, retried into a fallback and streamed, with SpendLogs writes
on, must hold every worker's RSS flat, both as the traffic runs and once a full garbage collection has had its
chance to free what the requests left behind. It reads the proxy's own RSS, so it runs in the perf lane, where
nothing else is hitting the stack.

The regressions it guards each reached a release. Every request's Redis batch scope stayed alive for good (about
40 KiB per request from v1.110.0-dev.1 until #50751), a pod grew about 100 MB a minute after v1.105.1 (#48678),
SpendLogs writes collapsed while memory grew under sustained concurrency (#46435), a failed call kept its whole
request body alive in an exception and frame cycle until a full collection (#52278), and every streamed call built
its own FallbackStreamWrapper class and a reference cycle with it (#51936). The first kind is still there after a
full collection, the cycle kind piles up between full collections, so both readings are taken.

Each batch sends the four paths in equal shares: a 200 from the configured mock deployment, a streamed 200 from
it, a refusal from a model group whose deployments refuse at the socket carrying a large request body, and a call
to that refusing group that retries and then falls back to the mock deployment for a 200. Two batches warm the
proxy's caches and allocator arenas, the rest are measured. After every batch each worker's RSS is read until no
new worker answers for a settle window, and the least-squares slope of those readings across the measured
batches, divided by the requests a worker served per batch, is the RSS slope. Around the measured window every
worker's gc thresholds are dropped to one so the next allocations trigger full collections, its RSS is read
again and its thresholds put back, and the growth between those two readings divided by the requests a worker
served is the memory each request kept. A sample of the served calls must have its SpendLogs rows written by the
end, the check that the writes ran the whole time."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Literal

import pytest
from e2e_config import (
    MEMORY_RSS_SAMPLE_INTERVAL_SECONDS,
    MEMORY_SOAK_BATCH_REQUESTS,
    MEMORY_SOAK_CONCURRENCY,
    MEMORY_SOAK_MEASURED_BATCHES,
    MEMORY_SOAK_RETAINED_KIB_PER_REQUEST,
    MEMORY_SOAK_SLOPE_KIB_PER_REQUEST,
    unique_marker,
)
from e2e_http import NoBody
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from lifecycle import ResourceManager
from memory_readings import RssReading, WorkerKey, settled_rss_per_worker
from models import ChatMessage, LiteLLMParamsBody, ReliabilityChatBody, RouterSettingsOverride, SpendLogsParams
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.perf]

SERVED_MODEL: Final = "gpt-6-luna"
REFUSING_MODEL: Final = "openai/gpt-5.5"
DEPLOYMENTS_PER_REFUSING_GROUP: Final = 2
WARMUP_BATCHES: Final = 2
MAX_TOKENS: Final = 2_048
LARGE_BODY_TURNS: Final = 64
LARGE_BODY_TURN_CHARS: Final = 16_384
FORCED_GC_THRESHOLDS: Final = (1, 1, 1)
DEFAULT_GC_THRESHOLDS: Final = (700, 10, 10)
GC_CONFIGURE_CALLS_PER_REPLICA: Final = 40
FORCED_GC_SECONDS: Final = 5.0
SPEND_SAMPLE: Final = 40
SPEND_WAIT_SECONDS: Final = 90.0
SPEND_LOGGED_SHARE_FLOOR: Final = 0.9

SoakPath = Literal["served", "streamed", "failed", "fallback"]
PATHS: Final[tuple[SoakPath, ...]] = ("served", "streamed", "failed", "fallback")


class GcThresholds(BaseModel, frozen=True):
    generation_0: int
    generation_1: int
    generation_2: int


@dataclass(frozen=True, slots=True)
class SoakCall:
    path: SoakPath
    status_code: int
    call_id: str | None
    body_head: str


@dataclass(frozen=True, slots=True)
class WorkerSlope:
    worker: WorkerKey
    where: str
    kib_per_request: float


@dataclass(frozen=True, slots=True)
class WorkerRetention:
    before: RssReading
    after: RssReading
    kib_per_request: float


def _large_transcript(marker: str) -> tuple[ChatMessage, ...]:
    return tuple(
        ChatMessage(
            role="user" if turn % 2 == 0 else "assistant",
            content=f"{marker} turn {turn} " + "x" * LARGE_BODY_TURN_CHARS,
        )
        for turn in range(LARGE_BODY_TURNS)
    )


def _register_refusing_group(proxy: ProxyClient, resources: ResourceManager, name: str) -> None:
    params: Final = LiteLLMParamsBody(
        model=REFUSING_MODEL, api_key="sk-unreachable", api_base="http://127.0.0.1:9/v1", cooldown_time=0
    )
    for model_id in tuple(proxy.create_model(name, params) for _ in range(DEPLOYMENTS_PER_REFUSING_GROUP)):
        resources.defer(lambda model_id=model_id: proxy.delete_model(model_id))


def _body(path: SoakPath, refusing: str, transcript: tuple[ChatMessage, ...]) -> ReliabilityChatBody:
    prompt: Final = ChatMessage(role="user", content=f"memory soak {path} {unique_marker()}")
    match path:
        case "served":
            return ReliabilityChatBody(model=SERVED_MODEL, messages=[prompt], max_tokens=MAX_TOKENS)
        case "streamed":
            return ReliabilityChatBody(model=SERVED_MODEL, messages=[prompt], max_tokens=MAX_TOKENS, stream=True)
        case "failed":
            return ReliabilityChatBody(
                model=refusing,
                messages=[*transcript, prompt],
                max_tokens=MAX_TOKENS,
                router_settings_override=RouterSettingsOverride(num_retries=0),
            )
        case "fallback":
            return ReliabilityChatBody(
                model=refusing,
                messages=[prompt],
                max_tokens=MAX_TOKENS,
                router_settings_override=RouterSettingsOverride(num_retries=1, fallbacks=[{refusing: [SERVED_MODEL]}]),
            )


def _call(proxy: ProxyClient, key: str, path: SoakPath, refusing: str, transcript: tuple[ChatMessage, ...]) -> SoakCall:
    resp: Final = proxy.transport.send(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=_body(path, refusing, transcript),
        stream=path == "streamed",
    )
    return SoakCall(path, resp.status_code, resp.call_id, resp.body[:300])


def _batch(proxy: ProxyClient, key: str, refusing: str, transcript: tuple[ChatMessage, ...]) -> tuple[SoakCall, ...]:
    paths: Final[tuple[SoakPath, ...]] = tuple(PATHS[index % len(PATHS)] for index in range(MEMORY_SOAK_BATCH_REQUESTS))
    with ThreadPoolExecutor(max_workers=MEMORY_SOAK_CONCURRENCY) as pool:
        futures: Final = tuple(pool.submit(_call, proxy, key, path, refusing, transcript) for path in paths)
        return tuple(future.result() for future in futures)


def _assert_every_path_answered_as_expected(calls: Sequence[SoakCall]) -> None:
    wrong: Final = tuple(call for call in calls if (call.status_code == 200) == (call.path == "failed"))
    assert not wrong, (
        f"{len(wrong)} of {len(calls)} soak calls did not answer the way their path should (a 200 for served, "
        f"streamed and fallback calls, an error for failed ones), so the soak did not run the paths it claims: "
        f"{wrong[0]}"
    )


def _set_gc_everywhere(proxy: ProxyClient, thresholds: tuple[int, int, int]) -> None:
    params: Final = GcThresholds(generation_0=thresholds[0], generation_1=thresholds[1], generation_2=thresholds[2])
    answers: Final = tuple(
        transport.send(
            "/debug/memory/gc/configure",
            headers=proxy.management_headers(transport=transport),
            json=NoBody(),
            params=params,
        ).status_code
        for transport in proxy.replicas.values()
        for _ in range(GC_CONFIGURE_CALLS_PER_REPLICA)  # comprehension-ok: each worker behind a replica needs a call
    )
    refused: Final = tuple(status for status in answers if status != 200)
    assert not refused, f"/debug/memory/gc/configure answered {refused[0]} {len(refused)} times, so gc was not forced"


def _rss_after_forced_full_gc(proxy: ProxyClient) -> Mapping[WorkerKey, RssReading]:
    _set_gc_everywhere(proxy, FORCED_GC_THRESHOLDS)
    try:
        time.sleep(FORCED_GC_SECONDS)
        return settled_rss_per_worker(proxy)
    finally:
        _set_gc_everywhere(proxy, DEFAULT_GC_THRESHOLDS)


def _requests_per_worker(total: int, workers: int) -> float:
    return total / max(1, workers)


def _slope(values: Sequence[float]) -> float:
    xs: Final = tuple(float(index) for index in range(len(values)))
    mean_x: Final = sum(xs) / len(xs)
    mean_y: Final = sum(values) / len(values)
    spread: Final = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=True)) / spread


def _worker_slopes(checkpoints: Sequence[Mapping[WorkerKey, RssReading]]) -> tuple[WorkerSlope, ...]:
    workers: Final[frozenset[WorkerKey]] = frozenset(checkpoints[0]).intersection(*checkpoints[1:])
    per_worker_batch: Final = _requests_per_worker(MEMORY_SOAK_BATCH_REQUESTS, len(workers))
    return tuple(
        WorkerSlope(
            worker,
            checkpoints[-1][worker].where,
            _slope(tuple(checkpoint[worker].ram_usage_mb for checkpoint in checkpoints)) * 1024 / per_worker_batch,
        )
        for worker in workers
    )


def _worker_retention(
    before: Mapping[WorkerKey, RssReading], after: Mapping[WorkerKey, RssReading], requests: int
) -> tuple[WorkerRetention, ...]:
    assert before.keys() == after.keys(), (
        f"the workers answering /debug/memory/summary changed across the soak, so not every worker can be compared "
        f"with itself: gone {sorted(before.keys() - after.keys())} (a restart or OOM kill), new "
        f"{sorted(after.keys() - before.keys())}"
    )
    per_worker: Final = _requests_per_worker(requests, len(before))
    return tuple(
        WorkerRetention(
            before[worker],
            after[worker],
            (after[worker].ram_usage_mb - before[worker].ram_usage_mb) * 1024 / per_worker,
        )
        for worker in before
    )


def _spend_logged_share(proxy: ProxyClient, call_ids: Sequence[str], deadline: float) -> float:
    logged: Final = sum(1 for call_id in call_ids if proxy.spend_logs(SpendLogsParams(request_id=call_id)))
    share: Final = logged / max(1, len(call_ids))
    if share >= SPEND_LOGGED_SHARE_FLOOR or time.monotonic() > deadline:
        return share
    time.sleep(MEMORY_RSS_SAMPLE_INTERVAL_SECONDS * 5)
    return _spend_logged_share(proxy, call_ids, deadline)


def _soak(
    proxy: ProxyClient,
    key: str,
    refusing: str,
    transcript: tuple[ChatMessage, ...],
    batches: int,
    record_property: Callable[[str, object], None],
    first_batch: int,
) -> tuple[tuple[SoakCall, ...], tuple[Mapping[WorkerKey, RssReading], ...]]:
    def one(batch: int) -> tuple[tuple[SoakCall, ...], Mapping[WorkerKey, RssReading]]:
        calls: Final = _batch(proxy, key, refusing, transcript)
        _assert_every_path_answered_as_expected(calls)
        settled: Final = settled_rss_per_worker(proxy)
        for reading in settled.values():
            record_property(
                "rss_soak_batch",
                json.dumps({"batch": batch, "worker": reading.where, "rss_mb": reading.ram_usage_mb}),
            )
        return calls, settled

    results: Final = tuple(one(batch) for batch in range(first_batch, first_batch + batches))
    return tuple(call for calls, _ in results for call in calls), tuple(settled for _, settled in results)


class TestMemorySoak:
    @pytest.mark.filterwarnings("ignore:record_property is incompatible with junit_family")
    @pytest.mark.covers("reliability.perf.memory.under_slo")
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.OPENAI,),
            models=(SERVED_MODEL, REFUSING_MODEL),
            mode=Mode.STREAM,
        )
    )
    def test_mixed_traffic_soak_keeps_rss_flat_and_frees_requests(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        scoped_key: str,
        record_property: Callable[[str, object], None],
    ) -> None:
        marker: Final = unique_marker()
        refusing: Final = f"memory-soak-refusing-{marker}"
        _register_refusing_group(proxy, resources, refusing)
        transcript: Final = _large_transcript(marker)

        _soak(proxy, scoped_key, refusing, transcript, WARMUP_BATCHES, record_property, 1)
        before: Final = _rss_after_forced_full_gc(proxy)
        calls, checkpoints = _soak(
            proxy, scoped_key, refusing, transcript, MEMORY_SOAK_MEASURED_BATCHES, record_property, WARMUP_BATCHES + 1
        )
        after: Final = _rss_after_forced_full_gc(proxy)

        slopes: Final = _worker_slopes(checkpoints)
        steepest: Final = max(slopes, key=lambda slope: slope.kib_per_request)
        retained: Final = max(
            _worker_retention(before, after, len(calls)), key=lambda retention: retention.kib_per_request
        )
        served: Final = tuple(call.call_id for call in calls if call.status_code == 200 and call.call_id)
        spend_share: Final = _spend_logged_share(
            proxy, served[:: max(1, len(served) // SPEND_SAMPLE)][:SPEND_SAMPLE], time.monotonic() + SPEND_WAIT_SECONDS
        )
        record_property("rss_soak_slope_kib_per_request", round(steepest.kib_per_request, 3))
        record_property("rss_soak_retained_kib_per_request", round(retained.kib_per_request, 3))
        record_property("rss_soak_spend_logged_share", round(spend_share, 3))

        assert steepest.kib_per_request <= MEMORY_SOAK_SLOPE_KIB_PER_REQUEST, (
            f"{steepest.where} grew {steepest.kib_per_request:.1f} KiB of RSS per request it served across "
            f"{MEMORY_SOAK_MEASURED_BATCHES} measured batches of served, streamed, failed and fallback calls, past "
            f"the {MEMORY_SOAK_SLOPE_KIB_PER_REQUEST} KiB ceiling; the Redis batch scope leak fixed in #50751 grew "
            f"about 40 KiB per request"
        )
        assert retained.kib_per_request <= MEMORY_SOAK_RETAINED_KIB_PER_REQUEST, (
            f"{retained.after.where} kept {retained.kib_per_request:.1f} KiB per request after a forced full gc "
            f"({retained.before.ram_usage_mb:.0f} MB before the measured batches, {retained.after.ram_usage_mb:.0f} "
            f"MB after), past the {MEMORY_SOAK_RETAINED_KIB_PER_REQUEST} KiB ceiling, so requests leave memory "
            f"behind that no collection frees"
        )
        assert spend_share >= SPEND_LOGGED_SHARE_FLOOR, (
            f"only {spend_share:.0%} of a sample of served soak calls had a SpendLogs row within "
            f"{SPEND_WAIT_SECONDS:.0f}s, so spend writes stalled under the soak the way #46435 reported"
        )
