import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import pytest

import litellm
from litellm import DualCache, Router
from litellm._internal_context import current_service_target
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RateLimitDescriptor,
    RateLimitResponse,
    claim_request_stash_for_data,
    should_rate_limit,
)
from litellm.redis.stores.rate_limits import RateLimitScope, RateLimitStore
from litellm.rust_bridge.host import redis_events
from litellm.types.utils import (
    ModelLimitBorrowingConfig,
    ModelResponse,
    PriorityReservationSettings,
    Usage,
)
from tests.unit.proxy.hooks.native_capacity_handler import counter_key, fake_redis, listening, startup_pings


class Limiter(Protocol):
    @property
    def rate_limits(self) -> RateLimitStore: ...

    @property
    def window_size(self) -> int: ...

    async def should_rate_limit(
        self, descriptors: list[RateLimitDescriptor], *, read_only: bool, skip_parallel_check: bool
    ) -> RateLimitResponse: ...


@runtime_checkable
class CapacityHandler(Protocol):
    @property
    def v3_limiter(self) -> Limiter: ...

    def update_variables(self, llm_router: Router) -> None: ...

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None: ...

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None: ...

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None: ...

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool: ...


PreCall = Callable[[CapacityHandler, UserAPIKeyAuth, dict[str, object]], Awaitable[object]]
Build = Callable[[DualCache], CapacityHandler]
Scenario = Callable[[pytest.MonkeyPatch, Build, PreCall], Awaitable[object]]

_MODEL: Final = "shared-model"


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": _MODEL,
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": 100},
            }
        ]
    )


def _borrowing(monkeypatch: pytest.MonkeyPatch, fail_closed: bool, saturation_threshold: float = 0.1) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(
        litellm,
        "priority_reservation_settings",
        PriorityReservationSettings(
            saturation_threshold=saturation_threshold,
            model_limit_borrowing_config=ModelLimitBorrowingConfig(borrow_idle_capacity=("team",)),
        ),
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings", {"fail_closed_rate_limit_enforcement": fail_closed}
    )


def _team_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-borrow",
        team_id="team-a",
        metadata={"priority": "interactive"},
        team_metadata={"model_tpm_limit": {_MODEL: 100}},
    )


def _request() -> dict[str, object]:
    return {"model": _MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}


def _outcome(error: BaseException | None) -> object:
    if error is None:
        return None
    cause: Final = error.__cause__
    return (type(error).__name__, getattr(error, "status_code", None), None if cause is None else type(cause).__name__)


def _unreachable_port() -> int:
    with listening() as (_, port):
        return port


async def _attempt(call: Awaitable[object]) -> object:
    try:
        await call
    except Exception as error:
        return _outcome(error)
    return None


async def batch_trips(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    _borrowing(monkeypatch, fail_closed=False)
    seen: Final[list[object]] = []  # mutable-ok: each request's round trips and the scripts it sent
    with fake_redis(emulate=True) as redis:
        handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=redis.port)))
        handler.update_variables(llm_router=_router())
        await startup_pings()
        for _ in range(3):
            trips: int = redis.trips
            sent: int = len(redis.commands)
            with request_redis_batch_scope():
                outcome = await _attempt(pre_call(handler, _team_key(), _request()))
            scripts = [
                tuple(part.decode() for part in command[3 : 3 + int(command[2])])
                for command in redis.commands[sent:]
                if command[0].upper() in (b"EVAL", b"EVALSHA")
            ]
            seen.append((outcome, redis.trips - trips, scripts))
    return seen[1:]


async def redis_down_fails_open(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    _borrowing(monkeypatch, fail_closed=False)
    handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=_unreachable_port())))
    handler.update_variables(llm_router=_router())
    return [await _attempt(pre_call(handler, _team_key(), _request())) for _ in range(2)]


async def redis_down_fails_closed(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    _borrowing(monkeypatch, fail_closed=True)
    handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=_unreachable_port())))
    handler.update_variables(llm_router=_router())
    admitted: Final = await _attempt(pre_call(handler, _team_key(), _request()))
    handler.v3_limiter.rate_limits.set_local(
        RateLimitScope("model_saturation_check", _MODEL).counter("tokens"), 150, 60
    )
    framed: Final = await _attempt(
        handler.enforce_token_limits_for_frame(user_api_key_dict=_team_key(), requested_model=_MODEL)
    )
    return [admitted, framed]


def _borrower(team_id: str = "team-a") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-borrow", team_id=team_id, team_metadata={"model_tpm_limit": {_MODEL: 100}})


async def _pre_call_at(pre_call: PreCall, handler: CapacityHandler, auth: UserAPIKeyAuth, saturation: float) -> object:
    _saturate(handler, saturation)
    return await _attempt(pre_call(handler, auth, {"model": _MODEL, "max_tokens": 0}))


def _saturate(handler: CapacityHandler, saturation: float) -> None:
    handler.v3_limiter.rate_limits.set_local(
        RateLimitScope("model_saturation_check", _MODEL).counter("tokens"), round(saturation * 1000), 60
    )


async def _settle(handler: CapacityHandler, tokens: int, call_id: str | None = None) -> None:
    await handler.async_log_success_event(
        kwargs={
            "litellm_params": {"metadata": {"model_group": _MODEL}},
            "standard_logging_object": {
                "metadata": {"user_api_key_team_id": "team-a", "user_api_key_auth_metadata": {}}
            },
            **({"litellm_call_id": call_id} if call_id is not None else {}),
        },
        response_obj=ModelResponse(usage=Usage(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


async def _team_tokens_remaining(handler: CapacityHandler, team_id: str = "team-a") -> object:
    share: Final = RateLimitDescriptor(
        key="model_per_team_saturated",
        value=f"{team_id}:{_MODEL}",
        rate_limit={"tokens_per_unit": 100, "requests_per_unit": None, "window_size": handler.v3_limiter.window_size},
    )
    response: Final = await should_rate_limit(handler.v3_limiter, [share], read_only=True, skip_parallel_check=True)
    return next(status["limit_remaining"] for status in response["statuses"] if status["rate_limit_type"] == "tokens")


async def _isolated(run: Coroutine[object, object, object]) -> object:
    return await asyncio.get_running_loop().create_task(run, context=contextvars.Context())


def _borrowing_in_memory(monkeypatch: pytest.MonkeyPatch, build: Build) -> CapacityHandler:
    _borrowing(monkeypatch, fail_closed=False, saturation_threshold=0.5)
    router: Final = _router()
    monkeypatch.setattr(litellm, "callbacks", ["dynamic_rate_limiter_v3"])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    handler: Final = build(DualCache())
    handler.update_variables(llm_router=router)
    return handler


async def nested_call_not_charged(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    async def run() -> object:
        handler: Final = _borrowing_in_memory(monkeypatch, build)
        claim_request_stash_for_data({"litellm_call_id": "outer-call"})
        admitted: Final = await _pre_call_at(pre_call, handler, _borrower(), 0.5)
        await _settle(handler, 60, call_id="llm-judge-call")
        after_nested: Final = await _team_tokens_remaining(handler)
        await _settle(handler, 60, call_id="outer-call")
        return [admitted, after_nested, await _team_tokens_remaining(handler)]

    return await _isolated(run())


async def idle_usage_forgiven(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    async def run() -> object:
        handler: Final = _borrowing_in_memory(monkeypatch, build)
        idle: Final = await _pre_call_at(pre_call, handler, _borrower(), 0.0)
        await _settle(handler, 150)
        after_idle: Final = await _team_tokens_remaining(handler)
        saturated: Final = await _pre_call_at(pre_call, handler, _borrower(), 0.5)
        await _settle(handler, 30)
        after_saturated: Final = await _team_tokens_remaining(handler)
        other: Final = await _pre_call_at(pre_call, handler, _borrower("team-b"), 0.5)
        return [idle, after_idle, saturated, after_saturated, other, await _team_tokens_remaining(handler, "team-b")]

    return await _isolated(run())


async def fallback_idle_forgiven(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    async def run() -> object:
        handler: Final = _borrowing_in_memory(monkeypatch, build)
        _saturate(handler, 0.0)
        allowed: Final = await handler.async_fallback_target_allowed(_borrower(), _MODEL, {})
        await _settle(handler, 150)
        saturated: Final = await _pre_call_at(pre_call, handler, _borrower(), 0.5)
        return [allowed, saturated, await _team_tokens_remaining(handler)]

    return await _isolated(run())


async def redis_call_targets(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    async def run() -> object:
        targets: Final[list[tuple[str, str | None]]] = []  # mutable-ok: each Redis call and its service target

        def record(
            event_loop: asyncio.AbstractEventLoop | None,
            call_type: str,
            start_time: float,
            end_time: float,
            failure: redis_events.Failure | None,
        ) -> None:
            targets.append((call_type, current_service_target()))

        monkeypatch.setattr(redis_events, "record_redis_call", record)
        _borrowing(monkeypatch, fail_closed=False, saturation_threshold=0.0)
        router: Final = _router()
        monkeypatch.setattr(litellm, "callbacks", ["dynamic_rate_limiter_v3"])
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
        with fake_redis(emulate=True) as redis:
            handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=redis.port)))
            handler.update_variables(llm_router=router)
            admitted: Final = await _attempt(pre_call(handler, _borrower(), {"model": _MODEL, "max_tokens": 0}))
            await handler.async_log_failure_event(
                kwargs={
                    "litellm_params": {"metadata": {"model_group": _MODEL}},
                    "standard_logging_object": {
                        "metadata": {"user_api_key_team_id": "team-a", "user_api_key_auth_metadata": {}}
                    },
                },
                response_obj=ModelResponse(usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10)),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            settled = -1
            while settled != len(targets):
                settled = len(targets)
                await asyncio.sleep(0.2)
            return [admitted, sorted(targets, key=str)]

    return await _isolated(run())


async def saturation_read_each_check(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    async def run() -> object:
        monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
        monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.5})
        monkeypatch.setattr(
            litellm, "priority_reservation_settings", PriorityReservationSettings(saturation_threshold=0.5)
        )
        router: Final = Router(
            model_list=[
                {
                    "model_name": "shared",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "rpm": 4},
                }
            ]
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
        high: Final = UserAPIKeyAuth(api_key="sk-high", metadata={"priority": "high"})
        saturation: Final = counter_key(RateLimitScope("model_saturation_check", "shared").counter("requests"))
        with fake_redis(emulate=True) as redis, redis.answering({saturation: "3"}):
            handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=redis.port)))
            handler.update_variables(llm_router=router)
            outcomes: Final = [
                await _attempt(pre_call(handler, high, {"model": "shared", "max_tokens": 0})) for _ in range(3)
            ]
            return [outcomes, redis.read_keys()]

    return await _isolated(run())


SCENARIOS: Final[Mapping[str, Scenario]] = {
    "test_a_request_batch_takes_the_round_trips_python_takes": batch_trips,
    "test_a_redis_outage_fails_open_like_python": redis_down_fails_open,
    "test_a_redis_outage_under_fail_closed_is_handled_like_python": redis_down_fails_closed,
    "test_a_nested_calls_tokens_are_not_charged_to_the_requests_team_share": nested_call_not_charged,
    "test_a_borrowers_usage_while_the_model_was_idle_is_forgiven_once_it_saturates": idle_usage_forgiven,
    "test_a_fallback_target_still_borrows_while_idle_and_its_tokens_are_forgiven_once_it_saturates": fallback_idle_forgiven,
    "test_every_redis_call_of_an_attempt_runs_under_the_rate_limits_service_target_like_python": redis_call_targets,
    "test_saturation_is_read_from_redis_on_every_check_like_main": saturation_read_each_check,
}
