import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.total_deadline import total_deadline
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.model_concurrency_metrics import ModelConcurrencyQueueObserver
from litellm.proxy.hooks.model_concurrency_queue import (
    MODEL_CONCURRENCY_DESCRIPTOR_KEY,
    ModelConcurrencyLimit,
    _general_settings_model_concurrency,
    parse_model_concurrency_limits,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    _request_stash,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.types.utils import ModelResponse
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits

_MODEL_GROUP = "gpt-4o"
_AUTH = UserAPIKeyAuth(api_key="sk-model-concurrency")
_HOST_PROMETHEUS_MULTIPROC_DIR: Final = os.environ.get("PROMETHEUS_MULTIPROC_DIR")


@pytest.fixture(autouse=True)
def _isolated_request_stash() -> Iterator[None]:
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


def _install_gauge_scripts(
    fakes: FakeRateLimits,
    *,
    on_acquire: Callable[[], None] | None = None,
) -> dict[str, set[str]]:
    gauges: dict[str, set[str]] = {}

    async def acquire(keys: list[str], args: list[int | str]) -> list[int]:
        limit, _ttl, slot_id = args
        if on_acquire is not None:
            on_acquire()
        slots = gauges.setdefault(keys[0], set())
        if len(slots) + 1 > int(limit):
            return [1, 1, len(slots), int(limit)]
        slots.add(str(slot_id))
        return [0, len(slots)]

    async def release(keys: list[str], args: list[int | str]) -> list[int]:
        slots = gauges.setdefault(keys[0], set())
        slots.discard(str(args[0]))
        return [len(slots)]

    fakes.parallel_acquire_script = acquire
    fakes.parallel_release_script = release
    return gauges


class _RecordingQueueObserver(ModelConcurrencyQueueObserver):
    def __init__(self) -> None:
        self.depths: dict[str, int] = {}
        self.events: list[tuple[str, str]] = []
        self.queued_event: asyncio.Event = asyncio.Event()

    def request_queued(self, model_group: str) -> None:
        self.depths[model_group] = self.depths.get(model_group, 0) + 1
        self.events.append(("queued", model_group))
        self.queued_event.set()

    def request_dequeued(self, model_group: str) -> None:
        self.depths[model_group] = self.depths.get(model_group, 0) - 1
        self.events.append(("dequeued", model_group))

    def depth(self, model_group: str = _MODEL_GROUP) -> int:
        return self.depths.get(model_group, 0)


def _handler(
    fakes: FakeRateLimits,
    limits: Mapping[str, ModelConcurrencyLimit],
    *,
    model_group_resolver: Callable[[str], str | None] | None = None,
    queue_observer: ModelConcurrencyQueueObserver | None = None,
) -> PROXY_MaxParallelRequestsHandler_v3:
    return PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=fakes.store(),
        model_concurrency_limit_resolver=lambda: limits,
        model_group_resolver=model_group_resolver or (lambda name: name),
        model_concurrency_queue_observer=queue_observer or _RecordingQueueObserver(),
    )


def _in_flight(handler: PROXY_MaxParallelRequestsHandler_v3, model_group: str = _MODEL_GROUP) -> int:
    gauge = RateLimitScope(MODEL_CONCURRENCY_DESCRIPTOR_KEY, model_group).gauge
    return handler._gauge_in_flight_from_cache_value(handler.rate_limits.local((gauge,))[0])


async def _fresh_request(operation: Callable[[], Awaitable[object]]) -> object:
    """Run ``operation`` in its own task, the way each proxy request gets its own request stash."""

    async def run() -> object:
        _request_stash.set(None)
        return await operation()

    return await asyncio.create_task(run())


@pytest.mark.asyncio
async def test_queue_times_out_with_429_when_every_slot_stays_taken() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=0)})
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
    assert _in_flight(handler) == 1

    with pytest.raises(ProxyRateLimitError) as exc:
        await _fresh_request(lambda: handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion"))

    assert exc.value.status_code == 429
    assert exc.value.descriptor_key == MODEL_CONCURRENCY_DESCRIPTOR_KEY
    assert "concurrency limit of 1" in str(exc.value.detail)
    assert _in_flight(handler) == 1

    await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)
    assert _in_flight(handler) == 0


@pytest.mark.asyncio
async def test_waiter_is_admitted_when_a_slot_frees() -> None:
    attempted = asyncio.Event()
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes, on_acquire=attempted.set)
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=5)})
    cache = DualCache()
    waiter_acquired = asyncio.Event()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
    assert _in_flight(handler) == 1

    async def waiter() -> None:
        _request_stash.set(None)
        await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
        waiter_acquired.set()
        await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)

    task = asyncio.create_task(waiter())
    await asyncio.wait_for(attempted.wait(), timeout=5)
    assert not task.done()
    assert not waiter_acquired.is_set()

    await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)
    await asyncio.wait_for(task, timeout=5)

    assert waiter_acquired.is_set()
    assert _in_flight(handler) == 0


@pytest.mark.asyncio
async def test_waiter_is_counted_in_the_queue_until_it_gets_a_slot() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=5)},
        queue_observer=observer,
    )
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
    assert observer.depth() == 0

    async def waiter() -> None:
        _request_stash.set(None)
        await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
        await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)

    task = asyncio.create_task(waiter())
    await asyncio.wait_for(observer.queued_event.wait(), timeout=5)
    assert observer.depth() == 1
    assert not task.done()

    await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)
    await asyncio.wait_for(task, timeout=5)

    assert observer.depth() == 0
    assert observer.events == [("queued", _MODEL_GROUP), ("dequeued", _MODEL_GROUP)]


@pytest.mark.asyncio
async def test_waiter_that_times_out_leaves_the_queue() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=0)},
        queue_observer=observer,
    )
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    with pytest.raises(ProxyRateLimitError):
        await _fresh_request(lambda: handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion"))

    assert observer.depth() == 0
    assert observer.events == [("queued", _MODEL_GROUP), ("dequeued", _MODEL_GROUP)]


@pytest.mark.asyncio
async def test_cancelled_waiter_leaves_the_queue() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=5)},
        queue_observer=observer,
    )
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    async def waiter() -> None:
        _request_stash.set(None)
        await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    task = asyncio.create_task(waiter())
    await asyncio.wait_for(observer.queued_event.wait(), timeout=5)
    assert observer.depth() == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert observer.depth() == 0


@pytest.mark.asyncio
async def test_request_that_gets_a_slot_immediately_is_not_queued() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=5)},
        queue_observer=observer,
    )

    await handler.async_pre_call_hook(_AUTH, DualCache(), {"model": _MODEL_GROUP}, "acompletion")

    assert observer.events == []
    assert observer.depth() == 0


@pytest.mark.asyncio
async def test_post_call_success_releases_the_slot() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=0)})
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
    assert _in_flight(handler) == 1

    await handler.async_post_call_success_hook(
        data={"model": _MODEL_GROUP}, user_api_key_dict=_AUTH, response=ModelResponse()
    )
    assert _in_flight(handler) == 0

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
    assert _in_flight(handler) == 1


@pytest.mark.asyncio
async def test_model_group_alias_shares_the_group_slot() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=0)},
        model_group_resolver=lambda name: {"my-alias": _MODEL_GROUP}.get(name),
        queue_observer=observer,
    )
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
    assert _in_flight(handler) == 1

    with pytest.raises(ProxyRateLimitError):
        await _fresh_request(lambda: handler.async_pre_call_hook(_AUTH, cache, {"model": "my-alias"}, "acompletion"))

    assert observer.events == [("queued", _MODEL_GROUP), ("dequeued", _MODEL_GROUP)]


@pytest.mark.asyncio
async def test_unconfigured_model_group_is_not_queued() -> None:
    calls: list[str] = []
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes, on_acquire=lambda: calls.append("acquire"))
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=0)})

    await handler.async_pre_call_hook(_AUTH, DualCache(), {"model": "other-model"}, "acompletion")

    assert calls == []
    assert _in_flight(handler, "other-model") == 0


@pytest.mark.asyncio
async def test_a_waiter_that_times_out_frees_its_queue_admission() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=0, max_queue_size=1)},
    )
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    for _ in range(3):
        with pytest.raises(ProxyRateLimitError) as exc:
            await _fresh_request(
                lambda: handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
            )
        assert "No slot freed up" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_full_queue_rejects_a_new_waiter_without_making_it_wait() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=30, max_queue_size=1)},
        queue_observer=observer,
    )
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    async def waiter() -> None:
        _request_stash.set(None)
        await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")
        await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)

    first_waiter = asyncio.create_task(waiter())
    await asyncio.wait_for(observer.queued_event.wait(), timeout=5)
    assert observer.depth() == 1

    started = time.monotonic()
    with pytest.raises(ProxyRateLimitError) as exc:
        await _fresh_request(lambda: handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion"))

    assert time.monotonic() - started < 1
    assert "already waiting" in str(exc.value.detail)
    assert observer.depth() == 1

    await handler.async_release_max_parallel_requests_on_disconnect(_AUTH)
    await asyncio.wait_for(first_waiter, timeout=5)


@pytest.mark.asyncio
async def test_the_key_parallel_limit_is_enforced_before_a_request_joins_the_queue() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    observer = _RecordingQueueObserver()
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=30)},
        queue_observer=observer,
    )
    cache = DualCache()
    key = UserAPIKeyAuth(api_key="sk-model-concurrency-order", max_parallel_requests=1)

    await handler.async_pre_call_hook(key, cache, {"model": _MODEL_GROUP}, "acompletion")

    with pytest.raises(ProxyRateLimitError) as exc:
        await _fresh_request(lambda: handler.async_pre_call_hook(key, cache, {"model": _MODEL_GROUP}, "acompletion"))

    assert "max_parallel_requests" in str(exc.value.detail)
    assert observer.events == []


@pytest.mark.asyncio
async def test_the_queue_wait_is_capped_by_the_request_total_timeout() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=1, queue_timeout_seconds=60)})
    cache = DualCache()

    await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    async def waiter() -> None:
        _request_stash.set(None)
        async with total_deadline(0.2, from_body=True):
            await handler.async_pre_call_hook(_AUTH, cache, {"model": _MODEL_GROUP}, "acompletion")

    with pytest.raises(ProxyRateLimitError) as exc:
        await asyncio.wait_for(_fresh_request(waiter), timeout=5)

    assert "total_timeout" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_comma_separated_model_with_a_limited_group_is_rejected() -> None:
    calls: list[str] = []
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes, on_acquire=lambda: calls.append("acquire"))
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=5)})

    with pytest.raises(ProxyRateLimitError) as exc:
        await _fresh_request(
            lambda: handler.async_pre_call_hook(
                _AUTH, DualCache(), {"model": f"{_MODEL_GROUP},{_MODEL_GROUP}"}, "acompletion"
            )
        )

    assert exc.value.status_code == 429
    assert "comma-separated" in str(exc.value.detail)
    assert calls == []
    assert _in_flight(handler) == 0


@pytest.mark.asyncio
async def test_a_comma_separated_alias_of_a_limited_group_is_rejected() -> None:
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes)
    handler = _handler(
        fakes,
        {_MODEL_GROUP: ModelConcurrencyLimit(limit=5)},
        model_group_resolver=lambda name: {"my-alias": _MODEL_GROUP}.get(name),
    )

    with pytest.raises(ProxyRateLimitError) as exc:
        await _fresh_request(
            lambda: handler.async_pre_call_hook(_AUTH, DualCache(), {"model": "my-alias,other"}, "acompletion")
        )

    assert _MODEL_GROUP in str(exc.value.detail)


@pytest.mark.asyncio
async def test_a_comma_separated_model_without_limited_groups_is_not_touched() -> None:
    calls: list[str] = []
    fakes = FakeRateLimits()
    _install_gauge_scripts(fakes, on_acquire=lambda: calls.append("acquire"))
    handler = _handler(fakes, {_MODEL_GROUP: ModelConcurrencyLimit(limit=5)})

    await handler.async_pre_call_hook(_AUTH, DualCache(), {"model": "other-a,other-b"}, "acompletion")

    assert calls == []


def test_parse_model_concurrency_limits_skips_invalid_entries() -> None:
    parsed = parse_model_concurrency_limits(
        {
            "gpt-4o": {"limit": 5},
            "claude": {"limit": 2, "queue_timeout_seconds": 0},
            "zero-limit": {"limit": 0},
            "not-a-mapping": 3,
            "bad-timeout": {"limit": 1, "queue_timeout_seconds": -1},
        }
    )

    assert parsed == {
        "gpt-4o": ModelConcurrencyLimit(limit=5, queue_timeout_seconds=60.0),
        "claude": ModelConcurrencyLimit(limit=2, queue_timeout_seconds=0.0),
    }
    assert parse_model_concurrency_limits(None) == {}
    assert parse_model_concurrency_limits("gpt-4o") == {}

    class _Opaque:
        pass

    assert parse_model_concurrency_limits(MappingProxyType({"gpt-4o": _Opaque()})) == {}


def test_general_settings_reads_the_model_concurrency_key() -> None:
    assert _general_settings_model_concurrency({"model_concurrency": {"gpt-4o": {"limit": 3}}}) == {
        "gpt-4o": ModelConcurrencyLimit(limit=3)
    }
    assert _general_settings_model_concurrency({}) == {}
    assert _general_settings_model_concurrency({"model_concurrency": "nope"}) == {}


def test_prometheus_observer_tracks_queue_depth_per_model_group(monkeypatch: pytest.MonkeyPatch) -> None:
    if _HOST_PROMETHEUS_MULTIPROC_DIR is not None:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", _HOST_PROMETHEUS_MULTIPROC_DIR)

    from prometheus_client import REGISTRY

    from litellm.proxy.hooks.model_concurrency_metrics import (
        MODEL_CONCURRENCY_QUEUE_SIZE_METRIC,
        PrometheusModelConcurrencyQueueObserver,
    )

    observer = PrometheusModelConcurrencyQueueObserver()
    labels = {"model_group": "gpt-4o-metrics-test"}

    observer.request_queued(labels["model_group"])
    observer.request_queued(labels["model_group"])
    try:
        assert REGISTRY.get_sample_value(MODEL_CONCURRENCY_QUEUE_SIZE_METRIC, labels) == 2
    finally:
        observer.request_dequeued(labels["model_group"])
        observer.request_dequeued(labels["model_group"])

    assert REGISTRY.get_sample_value(MODEL_CONCURRENCY_QUEUE_SIZE_METRIC, labels) == 0
