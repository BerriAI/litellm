"""Live e2e: a few hundred requests that fail before any provider answers must not
grow the proxy's resident memory past a fixed budget once the proxy is warm.

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

Two identical phases run back to back. The first is the warmup that grows the
proxy's caches and allocator arenas to their steady state, the second is the one
the budget applies to, so a healthy proxy shows the second phase adding roughly
nothing while a leaking one adds a fixed amount per request. RSS is read through
/debug/memory/summary on every configured replica; a burst of failing calls leaves
a transient bulge of garbage that gc reclaims within seconds, so each checkpoint
samples until no new worker has answered for a settle window and keeps the lowest
reading per worker. The growth is judged per worker (by replica address, hostname
and pid, since pods in their own pid namespaces report the same pids) so each
worker is compared with itself, and the two checkpoints must see the same workers:
a single load-balanced address reaches the workers behind it one answer at a time,
and a worker that answered only one checkpoint would otherwise drop out of the
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
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import (
    MEMORY_CONCURRENCY,
    MEMORY_REQUESTS_PER_PHASE,
    MEMORY_RETRIES_PER_REQUEST,
    MEMORY_RSS_BUDGET_MB,
    MEMORY_RSS_SAMPLE_INTERVAL_SECONDS,
    MEMORY_RSS_SETTLE_SAMPLES,
    MEMORY_STORED_REQUEST_BUDGET_KB,
    MEMORY_TRANSCRIPT_TURNS,
    unique_marker,
)
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatMessage, RouterSettingsOverride, SpendLogRow
from proxy_client import ProxyClient
from reliability_support import chat_override, create_never_benched_refusing_deployment

pytestmark = pytest.mark.e2e

DEPLOYMENTS_PER_GROUP: Final = 2
RSS_SAMPLE_CAP: Final = 4 * MEMORY_RSS_SETTLE_SAMPLES


@dataclass(frozen=True, slots=True)
class FailedCall:
    status_code: int
    seconds: float
    body_head: str
    call_id: str | None


WorkerKey = tuple[str, str | None, int]


@dataclass(frozen=True, slots=True)
class RssReading:
    replica: str
    hostname: str | None
    worker_pid: int
    ram_usage_mb: float

    @property
    def worker(self) -> WorkerKey:
        return (self.replica, self.hostname, self.worker_pid)


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
    return tuple(
        RssReading(replica, body.hostname, body.worker_pid, body.memory.ram_usage_mb)
        for replica, result in proxy.memory_summary_everywhere().items()
        for body in (unwrap(result),)
        if body.memory.ram_usage_mb is not None
    )


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


def _heaviest_worker_growth(
    warm: Mapping[WorkerKey, RssReading], after: Mapping[WorkerKey, RssReading]
) -> WorkerGrowth:
    assert warm.keys() == after.keys(), (
        f"the workers answering /debug/memory/summary changed between the checkpoints, so not every worker can "
        f"be compared with itself: gone after the measured batch {sorted(warm.keys() - after.keys())} (a worker "
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
    @pytest.mark.covers("reliability.perf.memory.under_slo")
    def test_failing_requests_do_not_grow_rss_or_stored_request(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
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

        warmup: Final = _fail_many(client.proxy, scoped_key, primary, override)
        _assert_every_call_failed_through_fallback(warmup, fallback)
        warm: Final = _settled_rss_per_worker(client.proxy)

        measured: Final = _fail_many(client.proxy, scoped_key, primary, override)
        _assert_every_call_failed_through_fallback(measured, fallback)
        after: Final = _settled_rss_per_worker(client.proxy)

        heaviest: Final = _heaviest_worker_growth(warm, after)
        assert heaviest.growth_mb <= MEMORY_RSS_BUDGET_MB, (
            f"proxy RSS grew {heaviest.growth_mb:.1f} MB over a second batch of {MEMORY_REQUESTS_PER_PHASE} failing "
            f"requests ({MEMORY_RETRIES_PER_REQUEST} retries each plus a fallback) after an identical warmup batch, "
            f"past the {MEMORY_RSS_BUDGET_MB:.0f} MB budget: worker pid {heaviest.warm.worker_pid} on "
            f"{heaviest.warm.hostname or 'an unnamed host'} behind {heaviest.warm.replica} settled at "
            f"{heaviest.warm.ram_usage_mb:.1f} MB warm and "
            f"{heaviest.after.ram_usage_mb:.1f} MB after; failing requests are leaking memory the way the "
            f"v1.100.0 retry breadcrumbs did"
        )
