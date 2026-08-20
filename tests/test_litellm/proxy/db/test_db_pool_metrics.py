import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.db.db_pool_metrics import DBPoolMetricsSampler, parse_pool_sample


class _Value:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.labels = {}
        self.description = ""


class _Hist:
    def __init__(self, total_sum, count):
        self.sum = total_sum
        self.count = count
        self.buckets = []


class _Metrics:
    """Stands in for prisma's Metrics model, matching the shape the engine returns."""

    def __init__(self, *, busy=0.0, idle=0.0, open_=0.0, wait=0.0, wait_ms=0.0, waits=0, query_ms=0.0, queries=0):
        self.counters = []
        self.gauges = [
            _Value("prisma_pool_connections_busy", busy),
            _Value("prisma_pool_connections_idle", idle),
            _Value("prisma_pool_connections_open", open_),
            _Value("prisma_client_queries_wait", wait),
        ]
        self.histograms = [
            _Value("prisma_client_queries_wait_histogram_ms", _Hist(wait_ms, waits)),
            _Value("prisma_datasource_queries_duration_histogram_ms", _Hist(query_ms, queries)),
        ]


class _Client:
    def __init__(self, *metrics, fail=False):
        self._metrics = list(metrics)
        self._fail = fail
        self.calls = 0

    async def get_pool_sample(self):
        self.calls += 1
        if self._fail:
            raise RuntimeError("engine unreachable")
        return parse_pool_sample(self._metrics[min(self.calls - 1, len(self._metrics) - 1)])


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_max_connections_is_busy_plus_idle_not_parsed_from_the_url():
    """The engine reports idle as remaining capacity, so busy + idle is the
    configured connection_limit. Verified against a live engine at
    connection_limit=5 under saturation: busy=5, idle=0."""
    sample = parse_pool_sample(_Metrics(busy=5.0, idle=0.0, open_=5.0))
    assert sample.max_connections == 5.0

    sample = parse_pool_sample(_Metrics(busy=2.0, idle=3.0, open_=5.0))
    assert sample.max_connections == 5.0


def test_parse_separates_pool_wait_from_query_execution():
    sample = parse_pool_sample(_Metrics(wait=3.0, wait_ms=1500.0, waits=6, query_ms=800.0, queries=4))
    assert sample.pending_acquirers == 3.0
    assert sample.acquire_wait_seconds_total == 1.5
    assert sample.acquire_count_total == 6
    assert sample.query_duration_seconds_total == 0.8
    assert sample.query_count_total == 4


def test_parse_tolerates_a_metric_the_engine_did_not_report():
    empty = _Metrics()
    empty.gauges = []
    empty.histograms = []
    sample = parse_pool_sample(empty)
    assert sample.max_connections == 0.0
    assert sample.acquire_count_total == 0


@pytest.mark.asyncio
async def test_sampler_throttles_until_the_interval_elapses():
    clock = _Clock()
    client = _Client(_Metrics(busy=1.0, idle=9.0))
    sampler = DBPoolMetricsSampler(min_interval_seconds=10.0, monotonic=clock)

    assert await sampler.maybe_sample(lambda: client) is not None
    assert client.calls == 1

    clock.now = 9.9
    assert sampler.is_due() is False
    assert await sampler.maybe_sample(lambda: client) is None
    assert client.calls == 1, "engine must not be re-read before the interval elapses"

    clock.now = 10.0
    assert sampler.is_due() is True
    assert await sampler.maybe_sample(lambda: client) is not None
    assert client.calls == 2


@pytest.mark.asyncio
async def test_cumulative_totals_are_published_as_deltas():
    clock = _Clock()
    client = _Client(
        _Metrics(wait_ms=1000.0, waits=10, query_ms=500.0, queries=5),
        _Metrics(wait_ms=2500.0, waits=25, query_ms=900.0, queries=9),
    )
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=clock)

    first = await sampler.maybe_sample(lambda: client)
    assert first is not None
    assert first.acquire_count_delta == 0, "first sample has no predecessor to diff against"
    assert first.acquire_wait_seconds_delta == 0.0

    clock.now = 5.0
    second = await sampler.maybe_sample(lambda: client)
    assert second is not None
    assert second.acquire_count_delta == 15
    assert second.acquire_wait_seconds_delta == pytest.approx(1.5)
    assert second.query_count_delta == 4
    assert second.query_duration_seconds_delta == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_engine_restart_does_not_publish_a_negative_delta():
    clock = _Clock()
    client = _Client(
        _Metrics(wait_ms=9000.0, waits=90, query_ms=4000.0, queries=40),
        _Metrics(wait_ms=10.0, waits=1, query_ms=5.0, queries=1),
    )
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=clock)

    await sampler.maybe_sample(lambda: client)
    clock.now = 5.0
    after_restart = await sampler.maybe_sample(lambda: client)

    assert after_restart is not None
    assert after_restart.acquire_count_delta == 0
    assert after_restart.acquire_wait_seconds_delta == 0.0
    assert after_restart.query_count_delta == 0
    assert after_restart.sample.acquire_count_total == 1, "the fresh absolute reading is still reported"


@pytest.mark.asyncio
async def test_an_unreachable_engine_never_raises_into_the_db_call():
    clock = _Clock()
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=clock)
    assert await sampler.maybe_sample(lambda: _Client(_Metrics(), fail=True)) is None


@pytest.mark.asyncio
async def test_a_failed_sample_still_consumes_the_interval():
    """Otherwise a wedged engine is re-probed on every single DB call."""
    clock = _Clock()
    client = _Client(_Metrics(), fail=True)
    sampler = DBPoolMetricsSampler(min_interval_seconds=10.0, monotonic=clock)

    await sampler.maybe_sample(lambda: client)
    clock.now = 1.0
    await sampler.maybe_sample(lambda: client)
    assert client.calls == 1


@pytest.mark.asyncio
async def test_a_client_that_delegates_via_getattr_is_still_sampled():
    """PrismaWrapper resolves its surface through an instance-level __getattr__,
    so a class-level structural check rejects it while the call itself works.
    An earlier revision guarded with isinstance() against a runtime_checkable
    Protocol and silently published zeroes on a live proxy; this pins the shape
    that regression had."""

    class _Delegating:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    inner = _Client(_Metrics(busy=4.0, idle=1.0))
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=_Clock())

    update = await sampler.maybe_sample(lambda: _Delegating(inner))

    assert update is not None
    assert update.sample.max_connections == 5.0
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_the_interval_is_consumed_even_when_there_is_no_client_to_sample():
    """Otherwise a proxy with prometheus disabled, or one sampled before the DB
    client exists, never records an attempt and so re-enters this path on every
    single database call instead of once per interval."""
    clock = _Clock()
    sampler = DBPoolMetricsSampler(min_interval_seconds=10.0, monotonic=clock)

    assert await sampler.maybe_sample(lambda: None) is None
    assert sampler.is_due() is False, "a no-client attempt must still consume the interval"

    clock.now = 10.0
    assert sampler.is_due() is True


@pytest.mark.asyncio
async def test_a_resolver_that_raises_is_contained():
    def _boom():
        raise RuntimeError("proxy_server not importable")

    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=_Clock())
    assert await sampler.maybe_sample(_boom) is None


@pytest.mark.asyncio
async def test_pool_timeouts_do_not_latch_the_pending_gauge():
    """prisma decrements its waiter gauge on acquisition but not on timeout, so
    every P2024 latches it one higher for the life of the process. Sequence
    reproduced against a live engine at connection_limit=2: after two bursts
    producing seven timeouts, a fully drained pool still reported pending=7.
    Free capacity proves nobody is queued, so an idle reading is the latch."""
    clock = _Clock()
    client = _Client(
        _Metrics(busy=0.0, idle=2.0, wait=0.0),   # at rest
        _Metrics(busy=2.0, idle=0.0, wait=4.0),   # saturated, 4 real waiters
        _Metrics(busy=0.0, idle=2.0, wait=4.0),   # drained, 4 latched timeouts
        _Metrics(busy=2.0, idle=0.0, wait=7.0),   # saturated again, 3 real waiters
        _Metrics(busy=0.0, idle=2.0, wait=7.0),   # drained again
    )
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=clock)

    observed = []
    for tick in range(5):
        clock.now = tick * 2.0
        update = await sampler.maybe_sample(lambda: client)
        assert update is not None
        observed.append(update.pending_acquirers)

    assert observed == [0.0, 4.0, 0.0, 3.0, 0.0], (
        f"expected the latch to be subtracted, got {observed}"
    )


@pytest.mark.asyncio
async def test_the_raw_engine_reading_is_still_carried_on_the_sample():
    """The correction applies to what is published, not to what was observed."""
    clock = _Clock()
    client = _Client(_Metrics(busy=0.0, idle=2.0, wait=4.0))
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=clock)

    update = await sampler.maybe_sample(lambda: client)

    assert update is not None
    assert update.pending_acquirers == 0.0
    assert update.sample.pending_acquirers == 4.0


@pytest.mark.asyncio
async def test_an_engine_restart_clears_the_pending_latch():
    """The restart guard zeroes the counter deltas; the waiter baseline has to go
    with them. A fresh engine's gauge carries no latch, so keeping the old
    baseline would subtract waiters that are really queued and can report zero
    during the saturation that caused the restart."""
    clock = _Clock()
    client = _Client(
        # saturated, 6 waiters of which 4 are latched timeouts
        _Metrics(busy=2.0, idle=0.0, wait=6.0, waits=40, queries=40),
        # drained: baseline arms at 4
        _Metrics(busy=0.0, idle=2.0, wait=4.0, waits=40, queries=40),
        # engine restarted (totals reset) and is saturated again with 3 real waiters
        _Metrics(busy=2.0, idle=0.0, wait=3.0, waits=1, queries=1),
    )
    sampler = DBPoolMetricsSampler(min_interval_seconds=1.0, monotonic=clock)

    observed = []
    for tick in range(3):
        clock.now = tick * 2.0
        update = await sampler.maybe_sample(lambda: client)
        assert update is not None
        observed.append(update.pending_acquirers)

    assert observed[1] == 0.0, "an idle pool reports no waiters and arms the baseline"
    assert observed[2] == 3.0, (
        f"after a restart the fresh gauge must be reported in full, got {observed[2]}"
    )
