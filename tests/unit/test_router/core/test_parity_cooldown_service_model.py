"""
Parity of the Rust core with ``router_python_core.py`` on three things the logging pipeline and the
fuzzer read: the cooldown event Prometheus counts (``router_cooldown_event_callback``, scheduled by
``set_cooldown_deployments``), the ``ServiceTypes.ROUTER`` service events, and a ``model`` that is no
string, which the native calls cannot take.

``tests/unit/test_router/core/conftest.py`` skips a test that reaches ``RustCoreUnsupported``. These
members are served, so ``served`` turns that refusal into a failure.

The scripted host runs a failure callback where the test puts it. ``litellm`` runs it on the loop's
thread for an async call, inline for a sync one, and on a thread with no loop for a stream; in every
case ``CustomLogger.log_event`` swallows what the callback raises. ``logged_like_litellm`` and
``streams_log_off_the_loop`` give the host those two behaviours.
"""

import asyncio
import contextlib
import functools
import inspect
import re
import time
from collections.abc import Awaitable, Callable, Coroutine, Generator, Mapping
from dataclasses import dataclass, field
from typing import Final, TypeVar

import pydantic
import pytest

import litellm
from litellm._service_logger import ServiceLogging
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_core import RouterCore
from litellm.router_utils import cooldown_callbacks, cooldown_handlers
from litellm.types.services import ServiceTypes
from tests.unit.test_router.core.retries_health_helpers import (
    MESSAGES,
    api_connection,
    attempt,
    attempt_sync,
    cooled_down,
    fresh,
    not_found,
    rate_limited,
    unprocessable,
)
from tests.unit.test_router.core.scripted_host import Call, ScriptedHost, deployment, rate_limit, server_error
from tests.unit.test_router.core.streaming_helpers import HostLogging, chat_stream, collect_texts, collect_texts_sync

MakeCore = Callable[..., RouterCore]
_Test = TypeVar("_Test", bound=Callable[..., object])

ACQUIRE: Final = "async_get_available_deployment"
STRATEGY: Final = "<routing_strategy>.async_get_available_deployments"


def _refused(error: BaseException) -> None:
    from litellm.router_rust.core import RustCoreUnsupported

    if isinstance(error, RustCoreUnsupported) or isinstance(error.__cause__, RustCoreUnsupported):
        pytest.fail(f"the Rust core stopped serving this: {error}")


def served(test: _Test) -> _Test:
    """Fail, rather than skip, when the test reaches ``RustCoreUnsupported``."""

    @functools.wraps(test)
    def run(*args: object, **kwargs: object) -> object:  # kwargs-ok: the test's fixtures
        try:
            return test(*args, **kwargs)
        except Exception as error:
            _refused(error)
            raise

    @functools.wraps(test)
    async def arun(*args: object, **kwargs: object) -> object:  # kwargs-ok: the test's fixtures
        try:
            return await test(*args, **kwargs)  # pyright: ignore[reportGeneralTypeIssues]  # an async test
        except Exception as error:
            _refused(error)
            raise

    return arun if inspect.iscoroutinefunction(test) else run  # pyright: ignore[reportReturnType]  # the test, wrapped


async def settle() -> None:
    """Let the tasks a call scheduled run."""
    for _ in range(20):
        await asyncio.sleep(0)


# --- cooldown events ------------------------------------------------------------------------------


@dataclass
class CooldownEvents:
    """``router_cooldown_event_callback`` calls, as ``(deployment id, status, cooldown time)``."""

    fired: list[tuple[str, object, object]] = field(default_factory=list)  # mutable-ok: appended as they fire


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> CooldownEvents:
    recorded: Final = CooldownEvents()

    async def record(ctx: object, deployment_id: str, exception_status: object, cooldown_time: object) -> None:
        recorded.fired.append((deployment_id, exception_status, cooldown_time))

    monkeypatch.setattr(cooldown_handlers, "router_cooldown_event_callback", record)
    return recorded


@pytest.fixture
def logged_like_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CustomLogger.log_event``: what a sync failure callback raises stops at the logging."""
    log_sync: Final = ScriptedHost.log_sync

    def swallowed(host: ScriptedHost, call: Call, error: BaseException | None, response: object) -> None:
        with contextlib.suppress(RuntimeError):
            log_sync(host, call, error, response)

    monkeypatch.setattr(ScriptedHost, "log_sync", swallowed)


@pytest.fixture
def streams_log_off_the_loop(monkeypatch: pytest.MonkeyPatch, logged_like_litellm: None) -> None:
    """``dispatch_failure_handlers``: a stream's sync callbacks run in ``executor.submit``, off the loop."""

    def dispatch(logging: HostLogging, exception: Exception, _traceback: str) -> Coroutine[None, None, None]:
        async def run() -> None:
            await asyncio.to_thread(logging.failure_handler, exception)

        return run()

    monkeypatch.setattr(HostLogging, "dispatch_attempt_failure_handlers", dispatch)


def fired_exactly(events: CooldownEvents) -> list[str]:  # mutable-ok: a fresh list
    """The events as written: ``5 == 5.0``, and the cooldown's type is part of the event."""
    return [repr(fired) for fired in events.fired]


def lone(**arguments: object) -> Mapping[str, object]:  # kwargs-ok: the router's constructor arguments
    return {"model_list": [deployment("g", "a")], "num_retries": 0, **arguments}


def ordered(**arguments: object) -> Mapping[str, object]:  # kwargs-ok: the router's constructor arguments
    return {
        "model_list": [deployment("g", "a", rpm=100, order=1), deployment("g", "b", order=2)],
        "num_retries": 0,
        "allowed_fails": 0,
        **arguments,
    }


@served
async def test_a_failed_attempt_fires_one_event_with_the_status_and_the_cooldown(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents
) -> None:
    core: Final = make_core(**lone(cooldown_time=7))
    host.answer("openai/a", fresh(not_found))

    assert isinstance(await attempt(core), litellm.NotFoundError)
    await settle()

    assert fired_exactly(events) == [repr(("a", 404, 7))]


@served
async def test_a_second_call_refused_by_the_cooldown_fires_nothing_more(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents
) -> None:
    core: Final = make_core(**lone())
    host.answer("openai/a", fresh(not_found))

    _ = await attempt(core)
    assert isinstance(await attempt(core), litellm.RouterRateLimitError)
    await settle()

    assert [fired[0] for fired in events.fired] == ["a"]


@served
async def test_each_retry_that_cools_down_fires_its_own_event(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=1, retry_after=0, allowed_fails=0
    )
    host.answer("*", fresh(server_error))

    _ = await attempt(core)
    await settle()

    assert sorted(fired[0] for fired in events.fired) == ["a", "b"]


@pytest.mark.parametrize("builder", [unprocessable, api_connection])
@served
async def test_a_failure_that_sets_no_cooldown_fires_nothing(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents, builder: Callable[[], Exception]
) -> None:
    core: Final = make_core(**lone())
    host.answer("openai/a", fresh(builder))

    _ = await attempt(core)
    await settle()

    assert cooled_down(core, ["a"]) == {}
    assert events.fired == []


@served
async def test_disabled_cooldowns_fire_nothing(make_core: MakeCore, host: ScriptedHost, events: CooldownEvents) -> None:
    core: Final = make_core(**lone(disable_cooldowns=True))
    host.answer("openai/a", fresh(not_found))

    _ = await attempt(core)
    await settle()

    assert cooled_down(core, ["a"]) == {}
    assert events.fired == []


@served
async def test_a_sync_call_on_a_running_loop_fires_on_that_loop(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents
) -> None:
    core: Final = make_core(**lone())
    host.answer("openai/a", fresh(not_found))

    assert isinstance(attempt_sync(core), litellm.NotFoundError)
    await settle()

    assert fired_exactly(events) == [repr(("a", 404, DEFAULT_COOLDOWN_TIME_SECONDS))]


@served
def test_a_sync_call_with_no_loop_cools_down_and_fires_on_its_own_loop(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents, logged_like_litellm: None
) -> None:
    """``function_with_fallbacks`` runs the sync walk on a private loop, which the event's task lands on."""
    core: Final = make_core(**lone())
    host.answer("openai/a", fresh(not_found))

    assert isinstance(attempt_sync(core), litellm.NotFoundError)

    assert set(cooled_down(core, ["a"])) == {"a"}
    assert fired_exactly(events) == [repr(("a", 404, DEFAULT_COOLDOWN_TIME_SECONDS))]


SYNC_ENDPOINTS: Final[Mapping[str, Callable[[RouterCore], object]]] = {
    "completion": lambda core: core.completion(model="g", messages=MESSAGES),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownLambdaType]  # endpoint method
    "embedding": lambda core: core.embedding(model="g", input=["hi"]),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownLambdaType]  # endpoint method
    "image_generation": lambda core: core.image_generation(model="g", prompt="hi"),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownLambdaType]  # endpoint method
}


@served
@pytest.mark.parametrize("endpoint", sorted(SYNC_ENDPOINTS))
def test_every_sync_endpoint_fires_the_default_cooldown_as_the_int_it_is(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents, logged_like_litellm: None, endpoint: str
) -> None:
    host.answer("openai/a", fresh(not_found))

    with pytest.raises(litellm.NotFoundError, match="no such model"):
        _ = SYNC_ENDPOINTS[endpoint](make_core(**lone()))

    assert fired_exactly(events) == [repr(("a", 404, DEFAULT_COOLDOWN_TIME_SECONDS))]


@served
@pytest.mark.parametrize(
    ("arguments", "request_fields", "expected"),
    [
        pytest.param({"cooldown_time": 7}, {}, 7, id="router-int"),
        pytest.param({"cooldown_time": 7.0}, {}, 7.0, id="router-whole-float"),
        pytest.param({"cooldown_time": 7.5}, {}, 7.5, id="router-float"),
        pytest.param({"cooldown_time": 0}, {}, DEFAULT_COOLDOWN_TIME_SECONDS, id="router-zero-is-unset"),
        pytest.param({"model_list": [deployment("g", "a", cooldown_time=7)]}, {}, 7, id="deployment-int"),
        pytest.param({"model_list": [deployment("g", "a", cooldown_time=7.0)]}, {}, 7.0, id="deployment-float"),
        pytest.param(
            {"cooldown_time": 7, "model_list": [deployment("g", "a", cooldown_time=7.0)]},
            {},
            7.0,
            id="deployment-float-over-router-int",
        ),
        pytest.param(
            {"cooldown_time": 7.0, "model_list": [deployment("g", "a", cooldown_time=7)]},
            {},
            7,
            id="deployment-int-over-router-float",
        ),
        pytest.param({}, {"cooldown_time": 7}, 7, id="request-int"),
        pytest.param({}, {"cooldown_time": 7.0}, 7.0, id="request-float"),
        pytest.param(
            {"model_list": [deployment("g", "a", cooldown_time=7.0)]},
            {"cooldown_time": 7},
            7,
            id="request-int-over-deployment-float",
        ),
        pytest.param(
            {"model_list": [deployment("g", "a", model_info={"cooldown_time": 7})]},
            {"cooldown_time": 7.0},
            7,
            id="model-info-int-over-request-float",
        ),
    ],
)
def test_the_event_carries_the_number_python_chose_as_it_was_written(
    make_core: MakeCore,
    host: ScriptedHost,
    events: CooldownEvents,
    logged_like_litellm: None,
    arguments: Mapping[str, object],
    request_fields: Mapping[str, object],
    expected: float,
) -> None:
    host.answer("openai/a", fresh(not_found))

    assert isinstance(attempt_sync(make_core(**lone(**arguments)), **request_fields), litellm.NotFoundError)

    assert fired_exactly(events) == [repr(("a", 404, expected))]


@served
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param({}, 9, id="over-the-default"),
        pytest.param({"cooldown_time": 9.0}, 9, id="an-int-over-an-equal-router-float"),
    ],
)
def test_a_retry_after_header_fires_as_the_int_python_reads(
    make_core: MakeCore,
    host: ScriptedHost,
    events: CooldownEvents,
    logged_like_litellm: None,
    arguments: Mapping[str, object],
    expected: int,
) -> None:
    for name in ("a", "b"):
        host.answer(f"openai/{name}", fresh(lambda: rate_limited("9")))
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, allowed_fails=0, **arguments
    )

    assert isinstance(attempt_sync(core), litellm.RateLimitError)

    assert {repr(fired[1:]) for fired in events.fired} == {repr((429, expected))}


@served
async def test_an_async_stream_failure_cools_down_and_fires_nothing(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents, streams_log_off_the_loop: None
) -> None:
    host.answer("openai/a", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True))
    core: Final = make_core(**ordered())

    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        await collect_texts(stream)
    await asyncio.sleep(0.05)
    await settle()

    assert set(cooled_down(core, ["a", "b"])) == {"a"}
    assert events.fired == []


@served
async def test_a_sync_stream_failure_on_a_running_loop_fires_nothing(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents, logged_like_litellm: None
) -> None:
    host.answer("openai/a", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True, sync=True))
    core: Final = make_core(**ordered())

    stream: Final = core.completion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        collect_texts_sync(stream)
    await asyncio.sleep(0.05)
    await settle()

    assert events.fired == []


class _RefusingCheck(CustomLogger):
    async def async_pre_call_check(self, deployment: dict, parent_otel_span: object) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride, reportMissingTypeArgument, reportUnknownParameterType]  # CustomLogger's own signature
        raise rate_limit(message="refused")


@served
async def test_a_refusing_pre_call_check_fires_the_cooldown_it_sets(
    make_core: MakeCore, host: ScriptedHost, events: CooldownEvents
) -> None:
    host.lists["callbacks"].append(_RefusingCheck())
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, allowed_fails=0, cooldown_time=7
    )

    assert isinstance(await attempt(core), litellm.RateLimitError)
    await settle()

    assert len(events.fired) == 1
    assert repr(events.fired[0][1:]) == repr((429, 7))
    assert events.fired[0][0] in cooled_down(core, ["a", "b"])


@served
@pytest.mark.parametrize(
    ("params", "cooldown_time", "expected"),
    [
        pytest.param({}, 7.0, 7.0, id="router-float"),
        pytest.param({}, None, DEFAULT_COOLDOWN_TIME_SECONDS, id="default"),
        pytest.param({"cooldown_time": 9}, 7, 7, id="the-deployment-is-not-read"),
    ],
)
async def test_a_refusing_pre_call_check_fires_the_routers_own_cooldown(
    make_core: MakeCore,
    host: ScriptedHost,
    events: CooldownEvents,
    params: Mapping[str, object],
    cooldown_time: float | None,
    expected: float,
) -> None:
    host.lists["callbacks"].append(_RefusingCheck())
    core: Final = make_core(
        model_list=[deployment("g", "a", **params), deployment("g", "b", **params)],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=cooldown_time,
    )

    assert isinstance(await attempt(core), litellm.RateLimitError)
    await settle()

    assert [repr(fired[1:]) for fired in events.fired] == [repr((429, expected))]


@served
async def test_a_recorded_failure_fires_with_the_status_it_was_given(
    make_core: MakeCore, events: CooldownEvents
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0)

    assert core.health().record_failure("a", server_error(), 500) is True
    await settle()

    assert fired_exactly(events) == [repr(("a", 500, DEFAULT_COOLDOWN_TIME_SECONDS))]


@served
@pytest.mark.parametrize(
    ("params", "cooldown_time", "error", "expected"),
    [
        pytest.param({}, 7, server_error, 7, id="router-int"),
        pytest.param({}, 7.0, server_error, 7.0, id="router-float"),
        pytest.param({"cooldown_time": 9}, 7.0, server_error, 7.0, id="the-deployment-is-not-read"),
        pytest.param({}, 7.0, lambda: rate_limited("9"), 7.0, id="the-header-is-not-read"),
    ],
)
async def test_a_recorded_failure_fires_the_routers_own_cooldown(
    make_core: MakeCore,
    events: CooldownEvents,
    params: Mapping[str, object],
    cooldown_time: float,
    error: Callable[[], Exception],
    expected: float,
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", **params), deployment("g", "b", **params)],
        allowed_fails=0,
        cooldown_time=cooldown_time,
    )

    assert core.health().record_failure("a", error(), 500) is True
    await settle()

    assert fired_exactly(events) == [repr(("a", 500, expected))]


@served
@pytest.mark.parametrize(
    ("model_info", "litellm_params", "cooldown_time", "error", "expected"),
    [
        pytest.param({}, {}, 7, server_error, 7, id="router-int"),
        pytest.param({}, {}, 7.0, server_error, 7.0, id="router-float"),
        pytest.param({}, {"cooldown_time": 7}, 7.0, server_error, 7, id="params-int-over-router-float"),
        pytest.param({}, {"cooldown_time": 7.0}, 7, server_error, 7.0, id="params-float-over-router-int"),
        pytest.param({"cooldown_time": 7}, {"cooldown_time": 7.0}, None, server_error, 7, id="info-over-params"),
        pytest.param({"cooldown_time": -1}, {"cooldown_time": 7}, 7.0, server_error, 7.0, id="a-negative-is-unset"),
        pytest.param({}, {}, 9.0, lambda: rate_limited("9"), 9, id="header-int-over-router-float"),
        pytest.param({}, {"cooldown_time": 9.0}, None, lambda: rate_limited("9"), 9.0, id="params-over-header"),
    ],
)
async def test_a_logged_failure_fires_the_callbacks_precedence(
    make_core: MakeCore,
    events: CooldownEvents,
    model_info: Mapping[str, object],
    litellm_params: Mapping[str, object],
    cooldown_time: float | None,
    error: Callable[[], Exception],
    expected: float,
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, cooldown_time=cooldown_time
    )
    logged: Final = {
        "exception": error(),
        "litellm_params": {
            "metadata": {"model_group": "g"},
            "model_info": {"id": "a", **model_info},
            **litellm_params,
        },
    }

    assert core.deployment_callback_on_failure(logged, None, 0.0, 0.0) is True  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the failure callback
    await settle()

    assert [repr(fired[2]) for fired in events.fired] == [repr(expected)]


@served
def test_a_recorded_failure_with_no_loop_raises_and_fires_nothing(make_core: MakeCore, events: CooldownEvents) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0)

    with pytest.raises(RuntimeError, match="no running event loop"):
        core.health().record_failure("a", server_error(), 500)

    assert events.fired == []


class _Prometheus:
    def __init__(self) -> None:
        self.cooled: Final[list[Mapping[str, object]]] = []  # mutable-ok: appended as the event runs
        self.outages: Final[list[Mapping[str, object]]] = []  # mutable-ok: same

    def set_deployment_complete_outage(self, **labels: object) -> None:  # kwargs-ok: the metric's labels
        self.outages.append(labels)

    def increment_deployment_cooled_down(self, **labels: object) -> None:  # kwargs-ok: the metric's labels
        self.cooled.append(labels)


@served
async def test_the_event_reaches_prometheus_with_the_deployments_labels(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    prometheus: Final = _Prometheus()
    monkeypatch.setattr(cooldown_callbacks, "get_prometheus_logger_from_callbacks", lambda: prometheus)
    core: Final = make_core(**lone())
    host.answer("openai/a", fresh(not_found))

    _ = await attempt(core)
    await settle()

    assert [(labels["model_id"], labels["exception_status"]) for labels in prometheus.cooled] == [("a", "404")]
    assert [labels["model_id"] for labels in prometheus.outages] == ["a"]


# --- service events -------------------------------------------------------------------------------


class _ServiceEvents:
    """Stands in for ``service_logger_obj``."""

    def __init__(self) -> None:
        self.seen: Final[list[Mapping[str, object]]] = []  # mutable-ok: appended as they run

    async def async_service_success_hook(self, **event: object) -> None:  # kwargs-ok: the hook's arguments
        self.seen.append(event)

    def call_types(self) -> list[object]:  # mutable-ok: a fresh list
        return [event["call_type"] for event in self.seen]


def service_events(core: RouterCore) -> _ServiceEvents:
    recorded: Final = _ServiceEvents()
    core.service_logger_obj = recorded  # pyright: ignore[reportAttributeAccessIssue]  # the attribute both cores keep
    return recorded


@served
def test_the_core_serves_its_service_logger(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())

    assert isinstance(core.service_logger_obj, ServiceLogging)  # pyright: ignore[reportAttributeAccessIssue]  # see service_events


@served
async def test_a_served_acompletion_logs_the_pick_and_the_call(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())
    recorded: Final = service_events(core)
    span: Final = object()

    _ = await attempt(core, metadata={"litellm_parent_otel_span": span})
    await settle()

    assert sorted(map(str, recorded.call_types())) == sorted([ACQUIRE, "acompletion"])
    for event in recorded.seen:
        assert event["service"] is ServiceTypes.ROUTER
        assert event["parent_otel_span"] is span
        duration: object = event["duration"]
        start_time: object = event["start_time"]
        end_time: object = event["end_time"]
        assert isinstance(duration, float) and isinstance(start_time, float) and isinstance(end_time, float)
        assert duration == pytest.approx(end_time - start_time)
        assert abs(end_time - time.time()) < 60


@served
async def test_a_retry_logs_a_pick_per_attempt_and_one_call(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=1, retry_after=0)
    host.answer("*", server_error(), None)
    recorded: Final = service_events(core)

    _ = await attempt(core)
    await settle()

    assert recorded.call_types().count(ACQUIRE) == 2
    assert recorded.call_types().count("acompletion") == 1


@served
async def test_a_failed_acompletion_logs_the_pick_alone(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(**lone())
    host.answer("openai/a", fresh(not_found))
    recorded: Final = service_events(core)

    _ = await attempt(core)
    await settle()

    assert recorded.call_types() == [ACQUIRE]


@served
async def test_a_refused_pick_logs_nothing(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())
    recorded: Final = service_events(core)

    assert isinstance(await attempt(core, model="nope"), litellm.BadRequestError)
    await settle()

    assert recorded.call_types() == []


@served
async def test_a_strategy_pick_logs_its_own_event(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, routing_strategy="least-busy"
    )
    recorded: Final = service_events(core)

    _ = await attempt(core)
    await settle()

    assert sorted(map(str, recorded.call_types())) == sorted([STRATEGY, ACQUIRE, "acompletion"])


@served
async def test_another_endpoint_logs_only_the_strategy_pick(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, routing_strategy="least-busy"
    )
    recorded: Final = service_events(core)

    _ = await core.aembedding(model="g", input=["hi"])  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method
    await settle()

    assert recorded.call_types() == [STRATEGY]


@served
async def test_a_shuffled_endpoint_logs_nothing(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())
    recorded: Final = service_events(core)

    _ = await core.aembedding(model="g", input=["hi"])  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method
    await settle()

    assert recorded.call_types() == []


@served
async def test_a_sync_call_logs_nothing(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())
    recorded: Final = service_events(core)

    _ = attempt_sync(core)
    await settle()

    assert recorded.call_types() == []


@served
async def test_a_pass_through_strategy_pick_runs_on_the_perf_counter(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "a", use_in_pass_through=True),
            deployment("g", "b", use_in_pass_through=True),
        ],
        num_retries=0,
        routing_strategy="least-busy",
    )
    recorded: Final = service_events(core)

    _ = await core.async_get_available_deployment_for_pass_through(model="g", request_kwargs={})  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the legacy member
    await settle()

    assert recorded.call_types() == [STRATEGY]
    start_time: object = recorded.seen[0]["start_time"]
    assert isinstance(start_time, float) and abs(start_time - time.perf_counter()) < 60


# --- a model that is no string --------------------------------------------------------------------

Endpoint = Callable[[RouterCore, object], Awaitable[object]]


def _call(name: str, **request: object) -> Endpoint:  # kwargs-ok: the endpoint's request
    def call(core: RouterCore, model: object) -> Awaitable[object]:
        answer: Final[object] = getattr(core, name)(model=model, **request)  # pyright: ignore[reportAny]  # an endpoint

        async def awaited() -> object:
            return await answer if inspect.isawaitable(answer) else answer

        return awaited()

    return call


ENDPOINTS: Final = {
    "acompletion": _call("acompletion", messages=MESSAGES),
    "atext_completion": _call("atext_completion", prompt="hi"),
    "aembedding": _call("aembedding", input=["hi"]),
    "aimage_generation": _call("aimage_generation", prompt="hi"),
    "aresponses": _call("aresponses", input="hi"),
    "aanthropic_messages": _call("aanthropic_messages", messages=MESSAGES, max_tokens=5),
    "completion": _call("completion", messages=MESSAGES),
    "embedding": _call("embedding", input=["hi"]),
    "image_generation": _call("image_generation", prompt="hi"),
}


def search_error(model: object) -> str:
    """What ``"/" in model`` raises (``declared_provider``)."""
    with pytest.raises(TypeError) as raised:
        _ = "/" in model  # pyright: ignore[reportOperatorIssue]  # raising is the point
    return str(raised.value)


def hash_error(lookup: Callable[[], object]) -> str:
    with pytest.raises(TypeError) as raised:
        _ = lookup()
    return str(raised.value)


SEARCH_FAILS: Final = (b"bytes", 1, 1.5, True)
ROUTES_NOWHERE: Final = (0, 0.0, False, b"", (), ("a",))


@pytest.mark.parametrize("value", SEARCH_FAILS, ids=repr)
@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
@served
async def test_a_model_no_search_can_read_raises_that_search(make_core: MakeCore, endpoint: str, value: object) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(TypeError, match=f"^{re.escape(search_error(value))}$"):
        await ENDPOINTS[endpoint](core, value)


@pytest.mark.parametrize(
    "mock",
    [
        {"mock_testing_fallbacks": True},
        {"mock_testing_rate_limit_error": True},
        {"mock_response": "litellm.RateLimitError"},
    ],
    ids=["fallbacks", "rate_limit", "response"],
)
@served
async def test_bytes_raise_before_any_mock(make_core: MakeCore, mock: Mapping[str, object]) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(TypeError, match=r"^a bytes-like object is required, not 'str'$"):
        await core.aimage_generation(model=b"bytes", prompt="hi", **mock)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method


@pytest.mark.parametrize("fallbacks", [None, [{"*": ["other"]}]], ids=["plain", "default_fallback"])
@served
async def test_bytes_raise_whatever_the_fallbacks(make_core: MakeCore, fallbacks: object) -> None:
    core: Final = make_core(**lone(fallbacks=fallbacks))

    with pytest.raises(TypeError, match=r"^a bytes-like object is required, not 'str'$"):
        await core.acompletion(model=b"bytes", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method


@pytest.mark.parametrize("value", [["g"], {"a": 1}], ids=["list", "dict"])
@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
@served
async def test_an_unhashable_model_raises_as_the_alias_lookup_raises(
    make_core: MakeCore, endpoint: str, value: object
) -> None:
    core: Final = make_core(**lone())
    expected: Final = hash_error(lambda: {}.get(value))  # pyright: ignore[reportUnknownLambdaType, reportArgumentType, reportUnknownMemberType, reportCallIssue]  # raising is the point

    with pytest.raises(TypeError, match=f"^{re.escape(expected)}$"):
        await ENDPOINTS[endpoint](core, value)


@served
def test_a_sync_pick_reads_the_names_first(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())
    expected: Final = hash_error(lambda: ["g"] in set[object]())  # pyright: ignore[reportUnhashable]  # raising is the point

    with pytest.raises(TypeError, match=f"^{re.escape(expected)}$"):
        core.text_completion(model=["g"], prompt="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method


@pytest.mark.parametrize("value", ROUTES_NOWHERE, ids=repr)
@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
@served
async def test_a_model_the_lookups_accept_is_a_name_no_deployment_serves(
    make_core: MakeCore, endpoint: str, value: object
) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(litellm.BadRequestError, match=re.escape(f"You passed in model={value}. There are no healthy")):
        await ENDPOINTS[endpoint](core, value)


@pytest.mark.parametrize("value", ROUTES_NOWHERE, ids=repr)
@served
async def test_a_mock_failure_names_the_model_as_given(make_core: MakeCore, value: object) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(litellm.InternalServerError, match=re.escape(f"This is a mock exception for model={value},")):
        await core.aimage_generation(model=value, prompt="hi", mock_testing_fallbacks=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method
    with pytest.raises(litellm.RateLimitError, match=re.escape(f"This is a mock exception for model={value},")):
        await core.acompletion(model=value, messages=MESSAGES, mock_testing_rate_limit_error=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method


@pytest.mark.parametrize("value", ROUTES_NOWHERE, ids=repr)
@served
async def test_a_default_fallback_serves_such_a_model(make_core: MakeCore, host: ScriptedHost, value: object) -> None:
    core: Final = make_core(**lone(fallbacks=[{"*": ["g"]}]))

    _ = await core.acompletion(model=value, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.parametrize("value", [0, b"", ()], ids=repr)
@served
async def test_a_catch_all_wildcard_serves_such_a_model(make_core: MakeCore, host: ScriptedHost, value: object) -> None:
    core: Final = make_core(model_list=[deployment("*", "w")], num_retries=0)

    _ = await core.acompletion(model=value, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method

    assert host.deployments_called("acompletion") == ["w"]


@pytest.mark.parametrize("kind", ["pick", "healthy", "pass_through"])
@served
async def test_a_selection_raises_as_a_call_does(make_core: MakeCore, kind: str) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(TypeError, match=r"^a bytes-like object is required, not 'str'$"):
        await core.select(b"bytes", kind=kind)  # pyright: ignore[reportArgumentType]  # raising is the point
    with pytest.raises(litellm.BadRequestError, match=r"You passed in model=0\. There are no healthy"):
        await core.select(0, kind=kind)  # pyright: ignore[reportArgumentType]  # see above


@served
def test_the_sync_pick_and_text_completion_raise_as_a_call_does(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(TypeError, match=r"^a bytes-like object is required, not 'str'$"):
        core.get_available_deployment(model=b"bytes")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the legacy member
    with pytest.raises(TypeError, match=r"^a bytes-like object is required, not 'str'$"):
        core.text_completion(model=b"bytes", prompt="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method
    with pytest.raises(litellm.BadRequestError, match=r"You passed in model=0\. There are no healthy"):
        core.text_completion(model=0, prompt="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method


@served
async def test_moderation_raises_or_goes_straight_to_the_host(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(TypeError, match=r"^a bytes-like object is required, not 'str'$"):
        await core.amoderation(model=b"bytes", input="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method
    _ = await core.amoderation(model=("a",), input="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method

    assert len(host.calls_to("amoderation")) == 1


@served
async def test_a_prioritized_call_fails_building_its_queue_item(make_core: MakeCore) -> None:
    core: Final = make_core(**lone())

    with pytest.raises(pydantic.ValidationError, match="model_name"):
        await core.acompletion(model=0, messages=MESSAGES, priority=1)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # endpoint method
