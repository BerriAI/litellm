"""Live e2e: a few hundred requests that fail before any provider answers must not
grow the proxy's resident memory past a loose ceiling once the proxy is warm.

The regression this guards shipped in v1.100.0: every retry breadcrumb copied the
whole request and the copies nested into one router-global list, so a proxy under
retry-heavy failing traffic grew until it was OOM-killed. The traffic here has that
shape: a model group whose deployments refuse at the socket (an unreachable base
URL) with cooldown_time 0 so the router keeps retrying them, per-request retries,
and a fallback group that refuses the same way, each request carrying a long chat
transcript so every whole-request copy costs hundreds of containers instead of a
handful. Under the stack's cooldown policy a
deployment that fails a handful of times in a row is benched (a bad-credential 401
included), the router answers "No deployments available" without retrying, and the
retry loop that leaks stops running; cooldown_time 0 keeps it running.

Four identical batches run back to back. The first two are the warmup that grows
the proxy's caches and allocator arenas, the last two are the measured window, and
each worker's RSS may grow at most a loose ceiling over that window. RSS cannot be
held to a tight per-batch budget: on the x86 release gate (rc/1.107.0 and v1.105.1
on isolated stacks, 28 batches each) the allocator kept setting new high-water marks
for the first several batches, about 60 MB on the third batch of a fresh stack and
more after the rest of the suite, with the gc-tracked object count flat to 0.1%, and
only settled to a couple of MB per batch after twenty. So the ceiling catches a gross
leak and nothing slower. RSS is read through /debug/memory/summary on every
configured replica; a burst of failing calls leaves a transient bulge of garbage that
gc reclaims within seconds, so each checkpoint samples until no new worker has
answered for a settle window and keeps the lowest reading per worker. A checkpoint
runs before the warmup and after every batch, and each batch's settled growth per
worker rides out as a repeated `rss_batch_growth` JUnit property as soon as that
batch settles, so a run that fails keeps the curve up to the batch it failed in. The
ceiling is judged per worker (by replica address, hostname and pid, since pods in
their own pid namespaces report the same pids) so each worker is compared with
itself, and the checkpoints around the measured window must see the same workers: a
single load-balanced address reaches the workers behind it one answer at a time, and
a worker that answered only one checkpoint would otherwise drop out of the
comparison, which is where a leaking worker could hide.

RSS alone is a coarse gauge: on the release stack (spend logs storing prompts,
json logs, prometheus and otel callbacks) the same v1.100.0 breadcrumbs grew RSS
by only about 15 MB per 300 failing requests, while every failing request's stored
request snapshot carried a copy of the request per failed attempt, over 100 KB on
the first call and a couple of MB once the copies nested, against tens of KB with
the fix. So the first check sends one failing request before the phases, reads its
spend log back through /spend/logs, and holds the stored request body to a fixed
size budget: the deterministic catch for a breadcrumb that copies the whole
request. It runs before the phases because the leaking writer drops its own rows
under the phases' traffic (a queue budget hit, a recursion limit on the nested
copies), which would turn the size check into a missing-row check.

A second, cheaper check holds the idle footprint: every worker's RSS as the harness
read it at collection time, before this pytest process sent any traffic (see
conftest.pytest_collection_finish), must sit under a fixed budget. On the
release gate that is a fresh stack right after its readiness gate, one worker per
gateway replica, so the reading is what a DB-backed boot costs on its own. A
v1.100.x worker with a database idled at 886 MB RSS where v1.101.0rc1 idled at
544 MB on the same database (1.3 GB against 560 MB at the pod level, under a
2 GiB limit): the generated Prisma client at prisma-client-py's default recursive
type depth, 91k TypedDict classes that v1.101.0 cut to 19k with
recursive_type_depth = -1. The budget starts at the rc1 reading plus headroom and
E2E_MEMORY_IDLE_RSS_BUDGET_MB overrides it; a later session on the same stack (the
changed-files workflow's repeat passes, a developer's local loop) measures a proxy
already warmed by traffic, which that headroom also has to cover.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import (
    MEMORY_CONCURRENCY,
    MEMORY_IDLE_RSS_BUDGET_MB,
    MEMORY_REQUESTS_PER_PHASE,
    MEMORY_RETRIES_PER_REQUEST,
    MEMORY_RSS_CEILING_MB,
    MEMORY_RSS_SAMPLE_INTERVAL_SECONDS,
    MEMORY_RSS_SETTLE_SAMPLES,
    MEMORY_STORED_REQUEST_BUDGET_KB,
    MEMORY_TRANSCRIPT_TURNS,
    PROXY_REPLICA_URLS,
    unique_marker,
)
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from lifecycle import ResourceManager
from memory_readings import RssCapture, RssReading, WorkerKey, read_rss_everywhere
from models import ChatMessage, RouterSettingsOverride, SpendLogRow
from proxy_client import ProxyClient
from reliability_support import REAL_MODEL, chat_override, create_never_benched_refusing_deployment

pytestmark = [pytest.mark.e2e, pytest.mark.quiet_stack]

DEPLOYMENTS_PER_GROUP: Final = 2
RSS_SAMPLE_CAP: Final = 4 * MEMORY_RSS_SETTLE_SAMPLES
WARMUP_BATCHES: Final = 2
MEASURED_BATCHES: Final = 2
FIRST_MEASURED_BATCH: Final = WARMUP_BATCHES + 1
LAST_BATCH: Final = WARMUP_BATCHES + MEASURED_BATCHES


@dataclass(frozen=True, slots=True)
class FailedCall:
    status_code: int
    seconds: float
    body_head: str
    call_id: str | None


@dataclass(frozen=True, slots=True)
class WorkerGrowth:
    warm: RssReading
    after: RssReading

    @property
    def growth_mb(self) -> float:
        return self.after.ram_usage_mb - self.warm.ram_usage_mb


def _register_refusing_group(proxy: ProxyClient, resources: ResourceManager, name: str) -> None:
    for model_id in tuple(create_never_benched_refusing_deployment(proxy, name) for _ in range(DEPLOYMENTS_PER_GROUP)):
        resources.defer(lambda model_id=model_id: proxy.delete_model(model_id))


def _transcript(turns: int) -> tuple[ChatMessage, ...]:
    return tuple(
        ChatMessage(role=role, content=f"turn {turn} {role}")
        for turn in range(turns)
        for role in ("user", "assistant")
    )


TRANSCRIPT: Final = _transcript(MEMORY_TRANSCRIPT_TURNS)


def _fail_once(proxy: ProxyClient, key: str, model: str, override: RouterSettingsOverride) -> FailedCall:
    started: Final = time.perf_counter()
    resp: Final = chat_override(
        proxy, key, model, f"memory regression {unique_marker()}", override=override, history=TRANSCRIPT
    )
    return FailedCall(resp.status_code, time.perf_counter() - started, resp.body[:300], resp.call_id)


def _fail_many(proxy: ProxyClient, key: str, model: str, override: RouterSettingsOverride) -> tuple[FailedCall, ...]:
    with ThreadPoolExecutor(max_workers=MEMORY_CONCURRENCY) as pool:
        futures: Final = tuple(
            pool.submit(_fail_once, proxy, key, model, override) for _ in range(MEMORY_REQUESTS_PER_PHASE)
        )
        return tuple(future.result() for future in futures)


def _read_rss_everywhere_after_pause(proxy: ProxyClient) -> tuple[RssReading, ...]:
    time.sleep(MEMORY_RSS_SAMPLE_INTERVAL_SECONDS)
    capture: Final = read_rss_everywhere(proxy)
    assert not capture.failures, (
        f"{len(capture.failures)} replica(s) gave no RSS reading mid-checkpoint, so their workers cannot be "
        f"compared with themselves: {'; '.join(capture.failures)}"
    )
    return capture.readings


def _readings_until_no_new_worker(
    proxy: ProxyClient, readings: tuple[RssReading, ...], samples: int, samples_since_new_worker: int
) -> tuple[RssReading, ...]:
    if samples >= RSS_SAMPLE_CAP or samples_since_new_worker >= MEMORY_RSS_SETTLE_SAMPLES:
        return readings
    sample: Final = _read_rss_everywhere_after_pause(proxy)
    known: Final = frozenset(reading.worker for reading in readings)
    new_worker_answered: Final = any(reading.worker not in known for reading in sample)
    return _readings_until_no_new_worker(
        proxy, readings + sample, samples + 1, 0 if new_worker_answered else samples_since_new_worker + 1
    )


def _settled_rss_per_worker(proxy: ProxyClient) -> Mapping[WorkerKey, RssReading]:
    readings: Final = _readings_until_no_new_worker(proxy, (), 0, 0)
    assert readings, "no /debug/memory/summary read carried ram_usage_mb, so the proxy cannot report its RSS"
    return MappingProxyType(
        {
            worker: min((reading for reading in readings if reading.worker == worker), key=lambda r: r.ram_usage_mb)
            for worker in {reading.worker for reading in readings}
        }
    )


def _record_batch_growth(
    record_property: Callable[[str, object], None],
    batch: int,
    before: Mapping[WorkerKey, RssReading],
    after: Mapping[WorkerKey, RssReading],
) -> None:
    for growth in (WorkerGrowth(before[worker], after[worker]) for worker in before.keys() & after.keys()):
        record_property(
            "rss_batch_growth",
            json.dumps(
                {
                    "batch": batch,
                    "worker": growth.warm.where,
                    "before_mb": growth.warm.ram_usage_mb,
                    "after_mb": growth.after.ram_usage_mb,
                    "growth_mb": round(growth.growth_mb, 2),
                }
            ),
        )


def _settled_rss_after_batches(
    proxy: ProxyClient,
    key: str,
    model: str,
    override: RouterSettingsOverride,
    fallback: str,
    record_property: Callable[[str, object], None],
    before: Mapping[WorkerKey, RssReading],
    batch: int,
    last_batch: int,
) -> Mapping[WorkerKey, RssReading]:
    calls: Final = _fail_many(proxy, key, model, override)
    _assert_every_call_failed_through_fallback(calls, fallback)
    after: Final = _settled_rss_per_worker(proxy)
    _record_batch_growth(record_property, batch, before, after)
    if batch == last_batch:
        return after
    return _settled_rss_after_batches(
        proxy, key, model, override, fallback, record_property, after, batch + 1, last_batch
    )


def _heaviest_worker_growth(
    warm: Mapping[WorkerKey, RssReading], after: Mapping[WorkerKey, RssReading]
) -> WorkerGrowth:
    assert warm.keys() == after.keys(), (
        f"the workers answering /debug/memory/summary changed around measured batches {FIRST_MEASURED_BATCH}-"
        f"{LAST_BATCH}, so not every "
        f"worker can be compared with itself: gone after it {sorted(warm.keys() - after.keys())} (a worker "
        f"that died or was restarted under failing traffic, which is what an OOM kill looks like), first seen "
        f"after it {sorted(after.keys() - warm.keys())} (the warm window never reached them, so they have no "
        f"baseline; raise E2E_MEMORY_RSS_SETTLE_SAMPLES if the stack has more workers than the window covers)"
    )
    return max((WorkerGrowth(warm[worker], after[worker]) for worker in warm), key=lambda growth: growth.growth_mb)


def _assert_every_call_failed_through_fallback(calls: Sequence[FailedCall], fallback: str) -> None:
    served: Final = tuple(call for call in calls if call.status_code == 200)
    assert not served, (
        f"{len(served)} of {len(calls)} calls came back 200, so they reached a provider and never "
        f"exercised the retry loop: {served[0].body_head}"
    )
    without_fallback: Final = tuple(call for call in calls if fallback not in call.body_head)
    assert not without_fallback, (
        f"{len(without_fallback)} of {len(calls)} failures never named the fallback group {fallback}, "
        f"so the request did not run through retries into the fallback: {without_fallback[0].body_head}"
    )


def _stored_request_kb(proxy: ProxyClient, call: FailedCall) -> float:
    assert call.call_id, (
        f"the failing call carried no x-litellm-call-id header, so its spend log cannot be read back: {call.body_head}"
    )
    rows: Final[Sequence[SpendLogRow]] = proxy.poll_logs_for_request_id(call.call_id)
    assert rows, (
        f"no spend log row appeared for failing call {call.call_id} within the poll window: either the stack "
        "writes no spend logs or its writer dropped the row, which the v1.100.0 one did once the stored "
        "request outgrew the writer's queue budget"
    )
    snapshot: Final = rows[0].proxy_server_request
    assert snapshot, (
        f"spend log {call.call_id} stored no request body, so the stack is not running with "
        "general_settings.store_prompts_in_spend_logs and the stored-request check would pass vacuously"
    )
    return len(json.dumps(snapshot).encode()) / 1024


class TestReliabilityMemory:
    @pytest.mark.covers("reliability.perf.idle_memory.under_slo")
    @meta(
        Subject(
            domain=Domain.DEPLOY_OPS,
        )
    )
    def test_workers_idle_under_rss_budget_before_traffic(self, idle_rss: RssCapture) -> None:
        assert not idle_rss.failures, (
            f"{len(idle_rss.failures)} replica(s) gave no RSS reading when the session started, so their idle "
            f"footprint went unmeasured: {'; '.join(idle_rss.failures)}"
        )
        unmeasured: Final = frozenset(PROXY_REPLICA_URLS) - frozenset(reading.replica for reading in idle_rss.readings)
        assert not unmeasured, (
            f"{len(unmeasured)} of {len(PROXY_REPLICA_URLS)} replica(s) gave neither an RSS reading nor a failure "
            f"reason when the session started, so their idle footprint went unmeasured: {', '.join(sorted(unmeasured))}"
        )
        heaviest: Final = idle_rss.heaviest
        assert heaviest is not None, "no replica was configured to read, so nothing was measured"
        assert heaviest.ram_usage_mb <= MEMORY_IDLE_RSS_BUDGET_MB, (
            f"{heaviest.where} sat at {heaviest.ram_usage_mb:.0f} MB RSS when the session started, before it sent "
            f"any traffic, past the {MEMORY_IDLE_RSS_BUDGET_MB:.0f} MB idle budget; a DB-backed v1.100.x worker idled "
            f"at 886 MB where v1.101.0rc1 idled at 544 MB, and at that size the release stack's 2 GiB pod limit "
            f"leaves the worker little room for traffic"
        )

    @pytest.mark.filterwarnings("ignore:record_property is incompatible with junit_family")
    @pytest.mark.covers("reliability.perf.memory.under_slo")
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.OPENAI,),
            models=(REAL_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_failing_requests_do_not_grow_rss_or_stored_request(
        self,
        client: ComplexityRouterClient,
        resources: ResourceManager,
        scoped_key: str,
        record_property: Callable[[str, object], None],
    ) -> None:
        marker: Final = unique_marker()
        primary: Final = f"reliability-memory-{marker}"
        fallback: Final = f"reliability-memory-fb-{marker}"
        _register_refusing_group(client.proxy, resources, primary)
        _register_refusing_group(client.proxy, resources, fallback)
        override: Final = RouterSettingsOverride(
            num_retries=MEMORY_RETRIES_PER_REQUEST, fallbacks=[{primary: [fallback]}]
        )

        probe: Final = _fail_once(client.proxy, scoped_key, primary, override)
        _assert_every_call_failed_through_fallback((probe,), fallback)
        stored_kb: Final = _stored_request_kb(client.proxy, probe)
        assert stored_kb <= MEMORY_STORED_REQUEST_BUDGET_KB, (
            f"the spend log of one failing request stored a {stored_kb:.0f} KB request body, past the "
            f"{MEMORY_STORED_REQUEST_BUDGET_KB:.0f} KB budget for a {len(TRANSCRIPT)}-message transcript with "
            f"{MEMORY_RETRIES_PER_REQUEST} retries and a fallback; the retry breadcrumbs are copying the whole "
            f"request into the stored snapshot the way the v1.100.0 ones did"
        )

        warm: Final = _settled_rss_after_batches(
            client.proxy,
            scoped_key,
            primary,
            override,
            fallback,
            record_property,
            _settled_rss_per_worker(client.proxy),
            1,
            WARMUP_BATCHES,
        )
        after: Final = _settled_rss_after_batches(
            client.proxy,
            scoped_key,
            primary,
            override,
            fallback,
            record_property,
            warm,
            FIRST_MEASURED_BATCH,
            LAST_BATCH,
        )

        heaviest: Final = _heaviest_worker_growth(warm, after)
        assert heaviest.growth_mb <= MEMORY_RSS_CEILING_MB, (
            f"proxy RSS grew {heaviest.growth_mb:.1f} MB over measured batches {FIRST_MEASURED_BATCH}-{LAST_BATCH} "
            f"({MEASURED_BATCHES * MEMORY_REQUESTS_PER_PHASE} failing requests) after {WARMUP_BATCHES} identical "
            f"warmup batches, past the {MEMORY_RSS_CEILING_MB:.0f} MB ceiling: worker pid {heaviest.warm.worker_pid} on "
            f"{heaviest.warm.hostname or 'an unnamed host'} behind {heaviest.warm.replica} settled at "
            f"{heaviest.warm.ram_usage_mb:.1f} MB warm and {heaviest.after.ram_usage_mb:.1f} MB after; failing "
            f"requests are leaking memory the way the v1.100.0 retry breadcrumbs did"
        )
