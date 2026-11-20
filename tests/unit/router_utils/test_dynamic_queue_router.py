import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, TypeVar

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.total_deadline import total_deadline
from litellm.redis.stores.connection import RedisStores
from litellm.redis.stores.router_usage import RouterUsageStore, UsageCounter
from litellm.redis.stores.router_usage.handle import RouterUsageHandle
from litellm.router import router_arguments
from litellm.router_config import build_router_config
from litellm.router_python_host import Endpoint, PythonHost
from litellm.router_rust.invoke.slots import max_parallel_requests_limit
from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
from litellm.router_utils.dynamic_queue import (
    AdmitReply,
    CoordinationUnavailable,
    Grant,
    LocalQueueBackend,
    MatchingFailure,
    OtherFailure,
    Outcome,
    PassThrough,
    PollReply,
    Probe,
    QueueFull,
    QueueSettings,
    QueueTimeout,
    Succeeded,
    Waiting,
    WaitRequest,
)
from litellm.router_utils.dynamic_queue_router import (
    MAX_WAITING_LIMIT,
    QUEUE_FULL_CODE,
    Admission,
    Backends,
    DynamicQueueRouter,
    DynamicQueueSettings,
    DynamicQueueTimeoutError,
    WithoutQueue,
    bound_request_queue,
    on_redis,
    parse_dynamic_queue,
    rejection_error,
)
from litellm.types.router import RouterRateLimitError
from tests.unit.router_utils.test_dynamic_queue import ManualTimer, _finished, _reap
from tests.unit.router_utils.test_dynamic_queue_redis import UnreachableRedis

ENABLED: Final = DynamicQueueSettings(enabled=True, max_waiting=8)
_SIZED: Final = {"enabled": True, "max_waiting": 8}
_T = TypeVar("_T")


class _ProviderAnswered(Exception):
    def __init__(self, status_code: int, retry_after: int | None = None) -> None:
        super().__init__("upstream failure")
        self.message = "upstream failure"
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        headers: Final = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = httpx.Response(
            status_code=status_code, request=self.request, text="upstream failure", headers=headers
        )


def _provider_error(status_code: int, retry_after: int | None = None) -> Exception:
    try:
        answered: Final = _ProviderAnswered(status_code, retry_after)
        exception_type(model="m", original_exception=answered, custom_llm_provider="openai")
    except Exception as mapped:  # noqa: BLE001  # the mapped error is the value under test
        return mapped
    raise AssertionError("exception_type always raises for a provider error")


def _in_memory(timer: ManualTimer, successes: int | None = None) -> Backends:
    """The queue coordinated in this process on the test's clock, standing in for its Redis."""

    def backend(settings: QueueSettings) -> LocalQueueBackend:
        tuned: Final = settings if successes is None else dataclasses.replace(settings, required_successes=successes)
        return LocalQueueBackend(tuned, clock=timer.now)

    return backend


def _router(
    timer: ManualTimer, settings: DynamicQueueSettings = ENABLED, successes: int | None = None
) -> DynamicQueueRouter:
    return DynamicQueueRouter(settings, _in_memory(timer, successes), timer=timer)


async def _no_cooldown() -> float:
    return 0.0


def _cooling_until(timer: ManualTimer, ends_at: float) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        return max(ends_at - timer.now(), 0.0)

    return left


async def _waited(timer: ManualTimer, call: Coroutine[object, object, _T], step: float) -> _T | BaseException:
    """Drives a call, moving time ``step`` at a time whenever it is parked on a queue recheck, and returns what it
    returned or the error it raised."""
    task: Final = asyncio.create_task(call)
    try:
        for _ in range(10_000):
            await timer.parked(task)
            if task.done():
                return task.exception() or task.result()
            timer.advance(step)
        raise AssertionError("deadlock safeguard: the admission never settled")
    finally:
        await _reap(task)


async def _congest(router: DynamicQueueRouter, capacity: str, retry_after_ms: int) -> None:
    request_queue: Final = router.request(priority=0, request_id=f"congest-{capacity}")
    admission: Final = await request_queue.admit(capacity, "m", _no_cooldown)
    assert isinstance(admission, Admission)
    assert await admission.finish(MatchingFailure(retry_after_ms=retry_after_ms)) is True


def test_enabling_the_queue_needs_a_default_priority() -> None:
    with pytest.raises(ValueError, match="default_priority"):
        parse_dynamic_queue({"enabled": True, "max_waiting": 8}, None)

    assert parse_dynamic_queue({"enabled": True, "max_waiting": 8}, -3) == ENABLED
    assert parse_dynamic_queue({"enabled": False}, None) is None
    assert parse_dynamic_queue(None, 0) is None


def test_an_unknown_dynamic_queue_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        parse_dynamic_queue({"enabled": True, "max_waiting": 8, "queue_timeot": 5}, 0)


@pytest.mark.parametrize(
    "block",
    [
        {"trigger_status_codes": [429]},
        {"queue_timeout": float("inf")},
        {"queue_timeout": float("nan")},
        {"max_waiting": 0},
        {"max_waiting": 65},
        {"max_waiting": "8"},
        {"max_waiting": True},
        {"max_waiting": 8.0},
        {"coordination": "local"},
        {"recovery_successes": 3},
        {"initial_backoff": 0.5},
    ],
    ids=[
        "triggers_are_not_a_setting",
        "endless_queue",
        "nan_queue",
        "no_room",
        "past_the_supported_limit",
        "text_size",
        "bool_size",
        "float_size",
        "coordination_is_not_a_setting",
        "recovery_is_not_a_setting",
        "backoff_is_not_a_setting",
    ],
)
def test_an_invalid_or_internal_dynamic_queue_setting_is_refused(block: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_dynamic_queue({"enabled": True, "max_waiting": 8, **block}, 0)


def test_an_enabled_queue_without_its_size_is_refused_and_a_disabled_one_needs_none() -> None:
    with pytest.raises(ValidationError, match="requires max_waiting"):
        parse_dynamic_queue({"enabled": True}, 0)

    assert parse_dynamic_queue({"enabled": False}, 0) is None
    assert parse_dynamic_queue({}, 0) is None


@pytest.mark.parametrize("size", [1, MAX_WAITING_LIMIT])
def test_a_queue_size_from_one_up_to_the_supported_limit_is_the_queues_capacity(size: int) -> None:
    settings: Final = parse_dynamic_queue({"enabled": True, "max_waiting": size}, 0)

    assert settings is not None and settings.queue_settings().max_waiting == size


def test_a_router_enabling_the_queue_without_its_size_fails_to_build() -> None:
    with pytest.raises(ValueError, match="requires max_waiting"):
        litellm.Router(model_list=_model_list(), default_priority=0, dynamic_queue={"enabled": True})


def test_the_queue_limits_an_operator_sets_drive_the_queue_with_internal_recovery_defaults() -> None:
    settings: Final = parse_dynamic_queue({"enabled": True, "queue_timeout": 12, "max_waiting": 40}, 0)

    assert settings is not None and settings.queue_timeout == 12
    assert settings.queue_settings() == dataclasses.replace(QueueSettings(), max_waiting=40)


@pytest.mark.parametrize(("status_code", "congests"), [(429, True), (529, True), (503, False), (500, False)])
def test_only_a_provider_answering_429_or_529_congests(status_code: int, congests: bool) -> None:
    outcome: Final = _router(ManualTimer()).outcome_of(_provider_error(status_code), None)

    assert isinstance(outcome, MatchingFailure) is congests
    assert isinstance(outcome, OtherFailure) is not congests


def test_a_rate_limit_litellm_raised_itself_never_congests() -> None:
    synthetic: Final = litellm.RateLimitError(message="deployment at capacity", llm_provider="", model="m")

    assert isinstance(_router(ManualTimer()).outcome_of(synthetic, 30.0), OtherFailure)


@pytest.mark.parametrize(("retry_after_s", "waits_ms"), [(None, 0), (12.0, 12_000)])
def test_a_congesting_failure_holds_the_next_probe_for_its_providers_retry_after(
    retry_after_s: float | None, waits_ms: int
) -> None:
    assert _router(ManualTimer()).outcome_of(_provider_error(429), retry_after_s) == MatchingFailure(
        retry_after_ms=waits_ms
    )


@pytest.mark.parametrize(
    ("cooldown_ends_at", "probe_runs_at"),
    [(0.0, 1.0), (5.0, 5.0)],
    ids=["deployment_without_cooldown", "deployment_still_cooling_down"],
)
async def test_a_wait_stays_queued_until_its_deployments_own_cooldown_ends(
    cooldown_ends_at: float, probe_runs_at: float
) -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    await _congest(router, "cap", retry_after_ms=0)
    cooling: Final = _cooling_until(timer, cooldown_ends_at)

    probe: Final = await _waited(timer, router.request(0, "r").admit("cap", "m", cooling), 0.25)
    assert isinstance(probe, Admission)
    await probe.finish(None)

    assert timer.at == probe_runs_at


async def test_a_wait_whose_cooldown_outlasts_the_queue_time_times_out_without_holding_the_probe() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=3.0))
    await _congest(router, "cap", retry_after_ms=0)
    cooling: Final = _cooling_until(timer, 60.0)

    refused: Final = await _waited(timer, router.request(0, "r").admit("cap", "m", cooling), 0.25)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3.0


async def test_attempts_a_healthy_capacity_admits_cost_none_of_the_requests_queue_time() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=3.0))
    request_queue: Final = router.request(priority=0, request_id="r")
    for _ in range(3):
        healthy = await request_queue.admit("cap", "m", _no_cooldown)
        assert isinstance(healthy, Admission)
        timer.advance(1.0)
        _ = await healthy.finish(None)
    await _congest(router, "cap", retry_after_ms=60_000)

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 6.0


async def test_a_wait_through_a_recovery_drain_is_spent_from_the_requests_queue_time() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=3.0), successes=1)
    await _congest(router, "cap", retry_after_ms=1_000)
    timer.advance(1.0)
    probe: Final = await router.request(0, "probe").admit("cap", "m", _no_cooldown)
    assert isinstance(probe, Admission)
    held: Final = asyncio.create_task(router.request(0, "held").admit("cap", "m", _no_cooldown))
    try:
        await timer.parked(held)
        await probe.finish(Succeeded())
        request_queue: Final = router.request(priority=0, request_id="r")
        drained: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)
        assert isinstance(drained, Admission)
        await drained.finish(None)
        released: Final = await _finished(held)
        assert isinstance(released, Admission)
        _ = await released.finish(None)
    finally:
        await _reap(held)
    await _congest(router, "cap", retry_after_ms=60_000)

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 4.0


async def test_queue_time_is_spent_once_across_every_wait_of_one_request() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=3.0))
    await _congest(router, "cap", retry_after_ms=1_000)
    request_queue: Final = router.request(priority=0, request_id="r")

    probe: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)
    assert isinstance(probe, Admission)
    assert timer.at == 1.0
    await probe.finish(MatchingFailure(retry_after_ms=60_000))
    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3.0


async def test_a_wait_never_outlasts_the_requests_total_timeout() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=7_200))
    await _congest(router, "cap", retry_after_ms=10_000_000)

    async with total_deadline(3_600, from_body=True, clock=timer.now):
        waiting: Final = router.request(priority=0, request_id="r").admit("cap", "m", _no_cooldown)
        refused: Final = await _waited(timer, waiting, 100.0)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3_600


async def test_a_wait_after_the_requests_own_context_ended_still_ends_at_its_total_timeout() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=7_200))
    await _congest(router, "cap", retry_after_ms=10_000_000)
    async with total_deadline(3_600, from_body=True, clock=timer.now):
        request_queue: Final = router.request(priority=0, request_id="r")

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 100.0)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3_600


async def test_a_request_whose_queue_redis_is_unreachable_goes_on_without_the_queue() -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(ENABLED, on_redis(lambda: UnreachableRedis()), timer=timer)
    request_queue: Final = router.request(priority=0, request_id="r")

    admitted: Final = await request_queue.admit("deployment:d1", "m", _no_cooldown)

    assert isinstance(admitted, WithoutQueue)
    assert admitted.still_cooling is False


async def test_a_request_with_no_queue_backend_goes_on_without_the_queue_and_a_later_one_queues_again() -> None:
    timer: Final = ManualTimer()
    backends: Final[list[LocalQueueBackend]] = []  # mutable-ok: the Redis the proxy attaches after the first request
    router: Final = DynamicQueueRouter(ENABLED, lambda settings: backends[0] if backends else None, timer=timer)
    first: Final = router.request(priority=0, request_id="first")

    without_queue: Final = await first.admit("deployment:d1", "m", _no_cooldown)
    backends.append(LocalQueueBackend(router.queue_settings, clock=timer.now))
    still_without: Final = await first.admit("deployment:d1", "m", _no_cooldown)
    queued: Final = await router.request(priority=0, request_id="later").admit("deployment:d1", "m", _no_cooldown)

    assert isinstance(without_queue, WithoutQueue)
    assert without_queue.still_cooling is False
    assert isinstance(still_without, WithoutQueue)
    assert still_without.still_cooling is False
    assert isinstance(queued, Admission)
    assert await queued.finish(Succeeded()) is True


class _ReportsUnreachable(LocalQueueBackend):
    """Admits in this process, then loses every report, like a Redis that drops after the turn was granted."""

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return CoordinationUnavailable(reason="redis_unavailable")


async def test_a_report_the_queue_cannot_take_takes_the_rest_of_the_request_off_the_queue() -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(
        ENABLED, lambda settings: _ReportsUnreachable(settings, clock=timer.now), timer=timer
    )
    request_queue: Final = router.request(priority=0, request_id="r")
    admission: Final = await request_queue.admit("deployment:d1", "m", _no_cooldown)
    assert isinstance(admission, Admission)

    learned: Final = await admission.finish(MatchingFailure())
    retried: Final = await request_queue.admit("deployment:d1", "m", _no_cooldown)

    assert learned is False
    assert request_queue.coordinated is False
    assert isinstance(retried, WithoutQueue)
    assert retried.still_cooling is False


def test_a_full_queue_answers_503_queue_full_and_running_out_of_time_is_the_usual_timeout() -> None:
    full: Final = rejection_error(QueueFull(), "m")
    late: Final = rejection_error(QueueTimeout(), "m")

    assert isinstance(full, litellm.ServiceUnavailableError)
    assert (full.status_code, getattr(full, "code", None)) == (503, QUEUE_FULL_CODE)
    assert isinstance(late, litellm.Timeout)
    assert late.status_code == 408


def _model_list() -> list[dict[str, object]]:
    return [{"model_name": "group", "litellm_params": {"model": "openai/gpt-x", "mock_response": "served"}}]


def test_a_router_enabling_the_queue_without_a_default_priority_fails_to_build() -> None:
    with pytest.raises(ValueError, match="default_priority"):
        litellm.Router(model_list=_model_list(), dynamic_queue={"enabled": True, "max_waiting": 8})


async def test_a_router_whose_queue_has_no_redis_serves_a_request_as_it_would_without_the_queue() -> None:
    router: Final = litellm.Router(
        model_list=_model_list(), default_priority=0, num_retries=2, dynamic_queue={"enabled": True, "max_waiting": 8}
    )

    response: Final = await router.acompletion(model="group", messages=[{"role": "user", "content": "hi"}])

    assert "served" in str(response)


async def test_a_router_with_a_queue_serves_a_healthy_request_straight_through() -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _on_the_clock(host, model_list=_model_list(), default_priority=0, dynamic_queue=_SIZED)

    response: Final = await router.acompletion(model="group", messages=[{"role": "user", "content": "hi"}])

    assert "served" in str(response)


def _deployments() -> list[dict[str, object]]:
    served: Final = {"model": "openai/gpt-x", "mock_response": "served"}
    streams: Final = {**served, "stream": True}
    named: Final = (("d1", served), ("d2", served), ("sets-stream", streams))
    return [{"model_name": "group", "litellm_params": params, "model_info": {"id": name}} for name, params in named]


async def _chat(router: litellm.Router, model: str) -> object:
    return await router.acompletion(model=model, messages=MESSAGES)


async def _chat_stream(router: litellm.Router, model: str) -> object:
    stream: Final = await router.acompletion(model=model, messages=MESSAGES, stream=True)
    return [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]  # the returned stream


async def _text(router: litellm.Router, model: str) -> object:
    return await router.atext_completion(model=model, prompt="hi")


@pytest.mark.parametrize(
    ("call", "model", "queued"),
    [
        (_chat, "d1", True),
        (_chat, "d2", False),
        (_chat_stream, "d1", False),
        (_chat, "sets-stream", True),
        (_text, "d1", False),
    ],
    ids=[
        "chat_on_the_congested_deployment",
        "chat_on_another_deployment",
        "chat_stream",
        "non_stream_request_on_a_deployment_setting_stream",
        "text_completion",
    ],
)
async def test_only_a_non_streaming_chat_picked_onto_a_congested_deployment_waits_in_its_queue(
    call: "Callable[[litellm.Router, str], Coroutine[object, object, object]]", model: str, queued: bool
) -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _on_the_clock(
        host,
        model_list=_deployments(),
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 0.5},
    )
    await _congest_directly(host, "deployment:d1")
    await _congest_directly(host, "deployment:sets-stream")

    outcome: Final = await _waited(timer, call(router, model), 0.25)

    assert isinstance(outcome, DynamicQueueTimeoutError) if queued else not isinstance(outcome, BaseException)
    assert timer.at == (0.5 if queued else 0.0)


async def test_a_router_without_the_queue_never_waits_in_the_queue_of_a_request_it_runs_inside() -> None:
    timer: Final = ManualTimer()
    outer: Final = _router(timer)
    await _congest(outer, "deployment:d1", retry_after_ms=60_000)
    inner: Final = litellm.Router(model_list=_deployments(), num_retries=0)

    with bound_request_queue(outer.request(priority=0, request_id="outer")):
        outcome: Final = await _waited(timer, _chat(inner, "d1"), 0.25)

    assert "served" in str(outcome)
    assert timer.at == 0.0


def _ordered_pair() -> list[dict[str, object]]:
    return [
        {"model_name": "group", "litellm_params": {"model": "openai/gpt-x", "order": order}, "model_info": {"id": name}}
        for name, order in (("first", 1), ("second", 2))
    ]


async def _congest_directly(host: "_QueueHost", capacity: str) -> None:
    await host.backend.report(capacity, "in-flight", PassThrough(0), MatchingFailure(retry_after_ms=60_000))


def _on_the_clock(host: "_QueueHost", **arguments: object) -> litellm.Router:
    """A router built on ``host``, which puts its queue on the test's clock in place of its Redis."""
    return litellm.Router.from_config(build_router_config(router_arguments(**arguments)), host)  # pyright: ignore[reportArgumentType]  # each test's router arguments


MESSAGES: Final = [{"role": "user", "content": "hi"}]


async def test_a_probe_a_full_parallel_slot_refuses_gives_its_turn_back_without_ending_the_recovery() -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _on_the_clock(
        host,
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "openai/gpt-x", "max_parallel_requests": 1},
                "model_info": {"id": "only"},
            }
        ],
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 30},
    )
    core: Final = router._core  # pyright: ignore[reportPrivateUsage]  # the router's own Rust core
    limit: Final = max_parallel_requests_limit(core._port, core._records_of(["only"])[0], {})  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]  # the slot the router keeps for the deployment
    assert isinstance(limit, MaxParallelRequestsLimit)
    observer: Final = WaitRequest("deployment:only", "observer", 0)

    async def call(answer: object) -> object:
        try:
            return await router.acompletion(model="group", messages=MESSAGES, mock_response=answer)
        except litellm.RateLimitError as refused:
            return refused

    congested: Final = await _waited(timer, call(_provider_error(429)), 0.25)
    timer.advance(1.0)
    first: Final = await _waited(timer, call("first success"), 0.25)
    with limit:
        refused: Final = await _waited(timer, call("never reached"), 0.25)
    still_probing: Final = await host.backend.admit(observer, 60_000)
    await host.backend.abandon(observer.capacity, observer.waiter_id)
    second: Final = await _waited(timer, call("second success"), 0.25)
    recovered: Final = await host.backend.admit(observer, 60_000)

    assert isinstance(congested, litellm.RateLimitError) and isinstance(refused, litellm.RateLimitError)
    assert "first success" in str(first) and "second success" in str(second)
    assert isinstance(still_probing, Probe)
    assert isinstance(recovered, PassThrough)
    assert timer.at == 1.0


class _RecordsBackoffs(PythonHost):
    def __init__(self) -> None:
        self.backoffs: list[float] = []  # mutable-ok: each retry backoff the router asked for

    def sleep(self, seconds: float) -> Awaitable[None]:
        self.backoffs.append(seconds)
        return _no_wait()


async def _no_wait() -> None:
    return None


class _QueueHost(_RecordsBackoffs):
    """Also builds the router's queue on the test's clock, on ``backends`` or else on ``backend``, one in-process
    queue the test can congest directly, in place of its Redis."""

    def __init__(self, timer: ManualTimer, backends: Backends | None = None) -> None:
        super().__init__()
        self._timer = timer
        self.backend = LocalQueueBackend(QueueSettings(max_waiting=8), clock=timer.now)
        self._backends: Backends = backends or (lambda settings: self.backend)

    def dynamic_queue(self, settings: DynamicQueueSettings, redis: Callable[[], object]) -> DynamicQueueRouter:
        return DynamicQueueRouter(settings, self._backends, timer=self._timer)


class _HoldsAbandon:
    """Coordinates in process, but holds every abandon until the test releases it."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        self.inner = inner
        self.abandoning = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def settings(self) -> QueueSettings:
        return self.inner.settings

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        return await self.inner.poll(capacity, waiter_id, ready_in_ms)

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        return await self.inner.renew(capacity, waiter_id, token)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return await self.inner.report(capacity, waiter_id, grant, outcome)

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        self.abandoning.set()
        await self.release.wait()
        return await self.inner.abandon(capacity, waiter_id)


def _least_busy_on(host: _QueueHost) -> litellm.Router:
    return _on_the_clock(
        host,
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "deployment-key"},
                "model_info": {"id": "only"},
            }
        ],
        routing_strategy="least-busy",
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 30},
    )


async def test_a_probe_cancelled_again_while_giving_its_turn_back_still_releases_its_in_flight_count(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    called: Final = asyncio.Event()

    async def never_answers(_: httpx.Request) -> httpx.Response:
        called.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=never_answers)
    timer: Final = ManualTimer()
    holding: Final = _HoldsAbandon(LocalQueueBackend(QueueSettings(max_waiting=8), clock=timer.now))
    router: Final = _least_busy_on(_QueueHost(timer, lambda settings: holding))
    await holding.inner.report("deployment:only", "in-flight", PassThrough(0), MatchingFailure(retry_after_ms=0))
    timer.advance(1.0)
    probing: Final = asyncio.create_task(router.acompletion(model="group", messages=MESSAGES))
    try:
        await asyncio.wait_for(called.wait(), 10)
        in_flight: Final = router.cache.get_cache(key="group_request_count:only")
        _ = probing.cancel()
        await asyncio.wait_for(holding.abandoning.wait(), 10)
        _ = probing.cancel()
        holding.release.set()
        outcome: Final = (await asyncio.wait_for(asyncio.gather(probing, return_exceptions=True), 10))[0]
    finally:
        holding.release.set()
        await _reap(probing)

    assert in_flight == 1
    assert isinstance(outcome, asyncio.CancelledError)
    assert not router.cache.get_cache(key="group_request_count:only")


class _RefundWaits(RouterUsageStore):
    """Counts as the router's own usage store, but holds every refund until the test opens ``gate``."""

    def __init__(self, handle: RouterUsageHandle, refunding: asyncio.Event, gate: asyncio.Event) -> None:
        super().__init__(handle)
        self._refunding = refunding
        self._gate = gate

    async def increment(self, counter: UsageCounter, value: float, ttl: int) -> float | None:
        if value < 0:
            self._refunding.set()
            await self._gate.wait()
        return await super().increment(counter, value, ttl)


class _RateLimitedProviderHost(_QueueHost):
    """Answers every chat call with a provider 429 raised before litellm runs any callback of its own."""

    def __init__(self, timer: ManualTimer) -> None:
        super().__init__(timer)
        self.refunding = asyncio.Event()
        self.gate = asyncio.Event()

    def router_usage(self, redis: RedisStores) -> RouterUsageStore:
        return _RefundWaits(redis.router_usage(), self.refunding, self.gate)

    def invoke(self, op: str) -> Endpoint:
        if op != "acompletion":
            return super().invoke(op)

        async def rate_limited(*args: object, **kwargs: object) -> object:  # kwargs-ok: the chat call's arguments
            raise _provider_error(429)

        return rate_limited


@pytest.mark.parametrize(("probing", "probe_in_ms"), [(False, 1_000), (True, 2_000)], ids=["healthy", "probe"])
async def test_a_request_cancelled_while_releasing_a_failed_attempt_still_reports_the_failure_and_settles_its_turn(
    probing: bool, probe_in_ms: int
) -> None:
    timer: Final = ManualTimer()
    host: Final = _RateLimitedProviderHost(timer)
    router: Final = _on_the_clock(
        host,
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "deployment-key", "itpm": 1_000_000},
                "model_info": {"id": "only"},
            }
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 30},
    )
    if probing:
        await _congest_directly(host, "deployment:only")
        timer.advance(60.0)
    failing: Final = asyncio.create_task(router.acompletion(model="group", messages=MESSAGES))
    try:
        await asyncio.wait_for(host.refunding.wait(), 10)
        _ = failing.cancel()
        host.gate.set()
        outcome: Final = (await asyncio.wait_for(asyncio.gather(failing, return_exceptions=True), 10))[0]
    finally:
        host.gate.set()
        await _reap(failing)
    observed: Final = await host.backend.admit(WaitRequest("deployment:only", "observer", 0), 60_000)

    assert isinstance(outcome, asyncio.CancelledError)
    assert observed == Waiting(probe_in_ms=probe_in_ms)
    assert timer.pending() == 0


def _retrying_router(host: _RecordsBackoffs, *, queued: bool = True, queue_timeout: float = 0.5) -> litellm.Router:
    arguments: Final = router_arguments(
        model_list=[{"model_name": "group", "litellm_params": {"model": "openai/gpt-x"}, "model_info": {"id": "only"}}],
        default_priority=0,
        num_retries=1,
        dynamic_queue={"enabled": queued, "max_waiting": 8, "queue_timeout": queue_timeout},
    )
    return litellm.Router.from_config(build_router_config(arguments), host)


async def test_a_retry_after_a_congesting_failure_keeps_its_ordinary_backoff_and_then_waits_its_turn() -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _retrying_router(host)

    waited: Final = await _waited(
        timer, router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429)), 0.25
    )

    assert isinstance(waited, DynamicQueueTimeoutError)
    assert timer.at == 0.5
    assert host.backoffs


async def test_a_retry_after_a_failure_the_queue_does_not_handle_keeps_its_retry_backoff() -> None:
    queued: Final = _QueueHost(ManualTimer())
    unqueued: Final = _RecordsBackoffs()
    for host, enabled in ((queued, True), (unqueued, False)):
        with pytest.raises(litellm.ServiceUnavailableError):
            _ = await _retrying_router(host, queued=enabled).acompletion(
                model="group", messages=MESSAGES, mock_response=_provider_error(503)
            )

    assert queued.backoffs
    assert len(queued.backoffs) == len(unqueued.backoffs)


@pytest.mark.parametrize(
    "backends",
    [
        lambda timer: on_redis(lambda: UnreachableRedis()),
        lambda timer: lambda settings: _ReportsUnreachable(settings, clock=timer.now),
    ],
    ids=["redis_unreachable_from_the_start", "redis_lost_after_the_turn_was_granted"],
)
async def test_a_congesting_failure_without_the_queues_redis_keeps_the_providers_error_and_its_retry_backoff(
    backends: Callable[[ManualTimer], Backends],
) -> None:
    timer: Final = ManualTimer()
    queued: Final = _QueueHost(timer, backends(timer))
    unqueued: Final = _RecordsBackoffs()
    for host, router in ((queued, _retrying_router(queued)), (unqueued, _retrying_router(unqueued, queued=False))):
        with pytest.raises(litellm.RateLimitError) as raised:
            _ = await router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429))
        assert raised.value.status_code == 429

    assert queued.backoffs
    assert len(queued.backoffs) == len(unqueued.backoffs)


@pytest.mark.parametrize(
    ("override", "served"),
    [({"api_key": "caller-own-key"}, True), ({"api_key": None}, False)],
    ids=["caller_key_on_the_fallback", "empty_api_key_field_on_the_fallback"],
)
async def test_a_fallback_target_sending_its_own_credentials_skips_the_queue_and_an_empty_field_waits_in_it(
    override: Mapping[str, object], served: bool
) -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _on_the_clock(
        host,
        model_list=[
            {"model_name": name, "litellm_params": {"model": f"openai/{name}"}, "model_info": {"id": f"{name}-1"}}
            for name in ("primary", "backup")
        ],
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 0.5},
    )
    await _congest_directly(host, "deployment:backup-1")

    fallback: Final = {"model": "backup", "mock_response": "served", **override}
    outcome: Final = await _waited(
        timer,
        router.acompletion(
            model="primary", messages=MESSAGES, mock_response=_provider_error(429), fallbacks=[fallback]
        ),
        0.25,
    )

    if served:
        assert "served" in str(outcome)
    else:
        assert isinstance(outcome, DynamicQueueTimeoutError)


async def test_a_group_left_only_cooling_with_the_queues_redis_unreachable_fails_as_it_would_without_the_queue() -> (
    None
):
    timer: Final = ManualTimer()
    router: Final = _on_the_clock(
        _QueueHost(timer, on_redis(lambda: UnreachableRedis())),
        model_list=_ordered_pair(),
        default_priority=0,
        num_retries=0,
        cooldown_time=60,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 0.5},
    )
    with pytest.raises(litellm.RateLimitError):
        _ = await router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429))

    with pytest.raises(RouterRateLimitError):
        _ = await router.acompletion(model="group", messages=MESSAGES, mock_response="never reached")


def _never_answers(started: asyncio.Event) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return left


async def test_a_cooldown_lookup_that_never_answers_lets_the_attempt_go_on_without_the_queue_after_its_timeout() -> (
    None
):
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    await _congest(router, "cap", retry_after_ms=0)
    timer.advance(1.0)
    looking: Final = asyncio.Event()
    request_queue: Final = router.request(0, "r")
    probe: Final = asyncio.create_task(request_queue.admit("cap", "m", _never_answers(looking)))
    try:
        await asyncio.wait_for(looking.wait(), 10)
        unqueued: Final = await _settled_on(timer, probe, 0.25)
    finally:
        await _reap(probe)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)
    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert isinstance(unqueued, WithoutQueue)
    assert unqueued.still_cooling is False
    assert request_queue.coordinated is False


@pytest.mark.parametrize(
    ("cooling", "still_cooling"), [(60.0, True), (0.0, False)], ids=["still_cooling", "cooldown_over"]
)
async def test_a_request_without_the_queue_learns_whether_its_deployment_is_still_cooling(
    cooling: float, still_cooling: bool
) -> None:
    timer: Final = ManualTimer()
    request_queue: Final = DynamicQueueRouter(ENABLED, on_redis(lambda: None), timer=timer).request(0, "r")

    unqueued: Final = await request_queue.admit("deployment:d1", "m", _cooling_until(timer, cooling))

    assert isinstance(unqueued, WithoutQueue)
    assert unqueued.still_cooling is still_cooling


async def test_a_stalled_cooldown_lookup_without_the_queue_is_given_up_after_the_operation_limit() -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(ENABLED, on_redis(lambda: None), timer=timer)
    looking: Final = asyncio.Event()
    unqueued: Final = asyncio.create_task(router.request(0, "r").admit("deployment:d1", "m", _never_answers(looking)))
    try:
        await asyncio.wait_for(looking.wait(), 10)
        await timer.parked(count=1)
        timer.advance(router.queue_settings.operation_timeout_ms / 1000)
        result: Final = await _finished(unqueued)
    finally:
        await _reap(unqueued)

    assert isinstance(result, WithoutQueue)
    assert result.still_cooling is False
    assert timer.pending() == 0


def _answers_once_then(
    timer: ManualTimer, ends_at: float, fails: bool
) -> "Callable[[], Coroutine[object, object, float]]":
    """Answers the cooldown left until ``ends_at`` once, then never answers again, or fails when ``fails``."""
    answered: Final = asyncio.Event()

    async def left() -> float:
        if answered.is_set() and fails:
            raise ConnectionError("the cooldown store went away")
        if answered.is_set():
            await asyncio.Event().wait()
        answered.set()
        return max(ends_at - timer.now(), 0.0)

    return left


async def _settled_on(timer: ManualTimer, task: "asyncio.Task[_T]", step: float) -> _T | BaseException:
    """Moves time ``step`` at a time whenever anything is parked on the timer, until ``task`` settles."""
    for _ in range(10_000):
        sleeping: Final = asyncio.create_task(timer.parked(count=1))
        _ = await asyncio.wait({task, sleeping}, return_when=asyncio.FIRST_COMPLETED)
        await _reap(sleeping)
        if task.done():
            return task.exception() or task.result()
        timer.advance(step)
    raise AssertionError("deadlock safeguard: the task never settled")


@pytest.mark.parametrize(
    ("fails", "cooling_until", "still_cooling"),
    [(False, 60.0, True), (False, 0.5, False), (True, 60.0, True)],
    ids=["stalled_reread_active", "stalled_reread_ended", "failed_reread_active"],
)
async def test_a_cooldown_seen_before_the_queue_was_lost_holds_until_it_ends_when_the_reread_does_not_answer(
    fails: bool, cooling_until: float, still_cooling: bool
) -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=30.0))
    await _congest(router, "cap", retry_after_ms=0)
    request_queue: Final = router.request(0, "r")
    lookup: Final = _answers_once_then(timer, cooling_until, fails)
    admitting: Final = asyncio.create_task(request_queue.admit("cap", "m", lookup))
    try:
        unqueued: Final = await _settled_on(timer, admitting, 0.25)
    finally:
        await _reap(admitting)

    assert isinstance(unqueued, WithoutQueue)
    assert unqueued.still_cooling is still_cooling
    assert request_queue.coordinated is False
    assert timer.at < 60.0


async def test_cancelling_a_request_during_its_cooldown_lookup_without_the_queue_cancels_it() -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(ENABLED, on_redis(lambda: None), timer=timer)
    looking: Final = asyncio.Event()
    unqueued: Final = asyncio.create_task(router.request(0, "r").admit("deployment:d1", "m", _never_answers(looking)))
    try:
        await asyncio.wait_for(looking.wait(), 10)
        await timer.parked(count=1)
        _ = unqueued.cancel()
        outcome: Final = await asyncio.wait_for(asyncio.gather(unqueued, return_exceptions=True), 10)
    finally:
        await _reap(unqueued)

    assert isinstance(outcome[0], asyncio.CancelledError)
    assert timer.pending() == 0


def _slow_cooldown_lookup(timer: ManualTimer, seconds: float) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        await timer.sleep(seconds)
        return 0.0

    return left


async def test_a_cooldown_lookup_slower_than_the_queue_time_left_times_out_and_dispatches_nothing() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, max_waiting=8, queue_timeout=0.5))
    await _congest(router, "cap", retry_after_ms=0)
    timer.advance(1.0)
    probe: Final = asyncio.create_task(router.request(0, "r").admit("cap", "m", _slow_cooldown_lookup(timer, 0.75)))
    try:
        await timer.parked(count=2)
        timer.advance(0.5)
        refused: Final = await asyncio.wait_for(asyncio.gather(probe, return_exceptions=True), 10)
    finally:
        await _reap(probe)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)
    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert isinstance(refused[0], DynamicQueueTimeoutError)
    assert timer.at == 1.5


@pytest.mark.parametrize("credentials", [{}, {"api_key": "caller-own-key"}], ids=["none_sent", "caller_api_key"])
async def test_every_chat_request_keeps_the_priority_scheduler_with_the_queue_enabled(
    credentials: Mapping[str, object],
) -> None:
    router: Final = litellm.Router(model_list=_model_list(), default_priority=0, dynamic_queue=_SIZED)

    response: Final = await router.acompletion(model="group", messages=MESSAGES, priority=1, **credentials)

    headers: Final = response._hidden_params.get("additional_headers") or {}  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the response's hidden params
    assert headers.get("x-litellm-request-prioritization-used") is True  # pyright: ignore[reportUnknownMemberType]  # see above


@pytest.mark.parametrize("forwarded", [True, False], ids=["forwarded_authorization", "deployment_key"])
async def test_a_congested_deployment_reached_on_a_forwarded_authorization_goes_out_without_waiting(
    forwarded: bool,
) -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _on_the_clock(
        host,
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "mock_response": "served"},
                "model_info": {"id": "isolated"},
            }
        ],
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 0.5},
    )
    await _congest_directly(host, "deployment:isolated")
    header: Final = {"custom_llm_provider": "anthropic", "extra_headers": {"Authorization": "Bearer caller-own"}}
    caller: Final = {"provider_specific_header": header} if forwarded else {}

    outcome: Final = await _waited(timer, router.acompletion(model="group", messages=MESSAGES, **caller), 0.25)

    assert "served" in str(outcome) if forwarded else isinstance(outcome, DynamicQueueTimeoutError)


async def test_an_empty_api_key_field_still_waits_in_its_configured_deployments_queue() -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _on_the_clock(
        host,
        model_list=[
            {"model_name": "group", "litellm_params": {"model": "openai/gpt-x"}, "model_info": {"id": "vertex"}}
        ],
        default_priority=0,
        num_retries=0,
        dynamic_queue={"enabled": True, "max_waiting": 8, "queue_timeout": 0.5},
    )
    await _congest_directly(host, "deployment:vertex")

    waited: Final = await _waited(
        timer, router.acompletion(model="group", messages=MESSAGES, api_key=None, mock_response="never reached"), 0.25
    )

    assert isinstance(waited, DynamicQueueTimeoutError)


RETRY_AFTER_S: Final = 20


async def test_a_retry_whose_backoff_outlasts_the_total_timeout_ends_at_once_with_the_providers_error() -> None:
    timer: Final = ManualTimer()
    host: Final = _QueueHost(timer)
    router: Final = _retrying_router(host, queue_timeout=30)

    with pytest.raises(litellm.RateLimitError) as raised:
        async with total_deadline(10.0, from_body=True, clock=timer.now):
            _ = await router.acompletion(
                model="group", messages=MESSAGES, mock_response=_provider_error(429, RETRY_AFTER_S)
            )

    assert raised.value.status_code == 429
    assert host.backoffs == []
    assert timer.at == 0.0
