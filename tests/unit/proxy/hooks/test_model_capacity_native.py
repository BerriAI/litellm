import asyncio
import contextvars
import logging
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Protocol

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError

import litellm
from litellm import DualCache, Router
from litellm._internal_context import current_service_target
from litellm._logging import verbose_proxy_logger
from litellm.constants import RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.host import request_scope, reset_host_for_tests
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    clear_subscribers_cache_for_tests,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)
from litellm.integrations.custom_logger_v2.rust import ABORT_ERRORS
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    arm_saturation_fallback_walk,
    mark_saturation_fallbacks_exhausted,
)
from litellm.proxy.hooks import parallel_request_limiter_v3
from litellm.proxy.hooks.model_capacity_native import (
    UNLICENSED_PRIORITY_MESSAGE,
    NativeDynamicRateLimiter,
    build_native_dynamic_rate_limiter,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitUnverifiableError,
    RequestRateLimiterStash,
    get_or_create_request_stash,
)
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import HookRun, InternalUsageCache, ProxyLogging, hook_fields, raise_v2_abort
from litellm.proxy.v2_proxy_fields import (
    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
    DERIVED_POST_CALL_SUCCESS_FIELDS,
    DERIVED_RESPONSE_HEADERS_FIELDS,
    PRE_CALL_SOURCES,
)
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET, RateLimitCounter, RateLimitScope
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome
from litellm.types.utils import ModelResponse, PriorityReservationSettings, Usage
from tests.unit.proxy.hooks.model_capacity_golden import golden

pytestmark = pytest.mark.asyncio


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 11, 13, 12, 0, 0)

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


@pytest.fixture(autouse=True)
def golden_case(request: pytest.FixtureRequest) -> Iterator[None]:
    golden.begin(request.node.nodeid)
    yield
    golden.finish()


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Clock:
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    clock: Final = Clock()
    monkeypatch.setattr(time, "time", lambda: clock.now().timestamp())
    monkeypatch.setenv("LITELLM_LICENSE", "test-license")
    monkeypatch.setattr(litellm, "priority_reservation", None)
    monkeypatch.setattr(litellm, "priority_reservation_settings", PriorityReservationSettings())
    return clock


@dataclass
class Outcome:
    raised: tuple[object, ...] | None = None
    model: object = None
    metadata: object = None
    max_tokens: object = None
    statuses: object = None
    matched_groups: object = None


def _raised(error: BaseException) -> tuple[object, ...]:
    if isinstance(error, HTTPException):
        return (
            type(error).__name__,
            error.status_code,
            error.detail,
            error.headers,
            getattr(error, "rate_limit_type", None),
            getattr(error, "model", None),
            getattr(error, "llm_provider", None),
            getattr(error, "descriptor_key", None),
        )
    return (type(error).__name__, str(error))


def _outcome(data: Mapping[str, object], live_auth: UserAPIKeyAuth) -> Outcome:
    metadata: Final = data.get("metadata")
    stash: Final = get_or_create_request_stash()
    response: Final = stash.rate_limit_response
    return Outcome(
        model=data.get("model"),
        metadata=None
        if not isinstance(metadata, Mapping)
        else {k: v for k, v in metadata.items() if k != "user_api_key_auth"},
        max_tokens=(data.get("max_tokens"), data.get("max_output_tokens")),
        statuses=None if response is None else response["statuses"],
        matched_groups=live_auth.matched_model_access_groups,
    )


class Side(Protocol):
    limiter: PROXY_MaxParallelRequestsHandler_v3

    def pre_call(self, data: dict[str, object], call_type: str) -> Awaitable[Outcome]: ...

    def post_call(self, data: dict[str, object], response: ModelResponse) -> Awaitable[object]: ...

    def response_headers(self, data: dict[str, object]) -> Awaitable[object]: ...

    def post_call_failure(
        self,
        data: dict[str, object],  # mutable-ok: the legacy failure hook takes the proxy's request dict
    ) -> Awaitable[None]: ...

    def ended(self, kwargs: dict[str, object], response: object, failed: bool) -> Awaitable[None]: ...

    def fallback(
        self, auth: UserAPIKeyAuth, model: str, data: dict[str, object], commit: bool
    ) -> Awaitable[object]: ...

    def frame(self, auth: UserAPIKeyAuth, model: str | None) -> Awaitable[object]: ...


def _auth(data: Mapping[str, object]) -> UserAPIKeyAuth:
    metadata: Final = data["metadata"]
    assert isinstance(metadata, Mapping)
    auth: Final = metadata["user_api_key_auth"]
    assert isinstance(auth, UserAPIKeyAuth)
    return auth


def _call(data: Mapping[str, object], call_type: str) -> CallFacts:
    return CallFacts(
        ids=CallIds(call_id=str(data.get("litellm_call_id") or "")),
        kind=CallKind(call_type=call_type),
        target=CallTarget(model=str(data.get("model"))),
        timing=CallTiming(start=1.0),
    )


async def _no_legacy(_logger: object, _data: object) -> None:
    return None


class NativeSide:
    def __init__(self, router: Router, clock: Clock) -> None:
        self.logger: Final[NativeDynamicRateLimiter] = build_native_dynamic_rate_limiter(
            DualCache(), time_provider=clock.now
        )
        self.logger.update_variables(router)
        self.limiter = self.logger.v3_limiter
        self.callbacks: Final[list[object]] = [self.logger]

    async def pre_call(self, data: dict[str, object], call_type: str) -> Outcome:
        live_auth: Final = _auth(data)
        auth_token: Final = REQUEST_AUTH.set(live_auth)
        try:
            result = await run_subscribers(
                hook_subscribers(Hook.pre_call, self.callbacks),
                with_sources(
                    KwargsFields(data, PROXY_REQUEST_KEYS),
                    PRE_CALL_SOURCES,
                    HookArgs(auth=live_auth, request=data, call_type=call_type),
                ),
                _no_legacy,
                call=_call(data, call_type),
            )
            after = kwargs_after(raise_v2_abort(result))
        except Exception as error:
            return Outcome(raised=_raised(error))
        finally:
            REQUEST_AUTH.reset(auth_token)
        return _outcome(after if isinstance(after, dict) else data, live_auth)

    async def post_call(self, data: dict[str, object], response: ModelResponse) -> object:
        live_auth: Final = _auth(data)
        auth_token: Final = REQUEST_AUTH.set(live_auth)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_success, self.callbacks),
                with_sources(
                    hook_fields(data, HookRun(response=response), Field.response),
                    DERIVED_POST_CALL_SUCCESS_FIELDS,
                    HookArgs(auth=live_auth, request=data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)
        return dict(response._hidden_params.get("additional_headers") or {})

    async def response_headers(self, data: dict[str, object]) -> object:
        merged: Final[dict[str, str]] = {}
        await run_subscribers(
            hook_subscribers(Hook.response_headers, self.callbacks, admit=Admit.every_entry),
            with_sources(
                hook_fields(data, HookRun(response=merged), Field.response_headers),
                DERIVED_RESPONSE_HEADERS_FIELDS,
                HookArgs(auth=_auth(data), request=data),
            ),
            _no_legacy,
        )
        return merged or None

    async def post_call_failure(
        self,
        data: dict[str, object],  # mutable-ok: the native runs on the proxy's request dict
    ) -> None:
        live_auth: Final = _auth(data)
        auth_token: Final = REQUEST_AUTH.set(live_auth)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_failure, self.callbacks, admit=Admit.every_entry),
                with_sources(
                    hook_fields(data, HookRun(response=RuntimeError("a later hook refused")), Field.error),
                    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
                    HookArgs(auth=live_auth, request=data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)

    async def ended(self, kwargs: dict[str, object], response: object, failed: bool) -> None:
        await run_subscribers(
            hook_subscribers(Hook.attempt_ended, self.callbacks, admit=Admit.every_entry),
            with_sources(
                KwargsFields(kwargs, LOGGING_KEYS),
                DERIVED_ATTEMPT_FIELDS,
                HookArgs(
                    request=kwargs,
                    result=response,
                    outcome=AttemptOutcome.failure if failed else AttemptOutcome.success,
                ),
            ),
            _no_legacy,
            call=_call(kwargs, "acompletion"),
        )

    async def fallback(self, auth: UserAPIKeyAuth, model: str, data: dict[str, object], commit: bool) -> object:
        try:
            return await self.logger.async_fallback_target_allowed(auth, model, data, commit=commit)
        except Exception as error:
            return _raised(error)

    async def frame(self, auth: UserAPIKeyAuth, model: str | None) -> object:
        try:
            await self.logger.enforce_token_limits_for_frame(user_api_key_dict=auth, requested_model=model)
        except Exception as error:
            return _raised(error)
        return None


@dataclass
class Request:
    auth: UserAPIKeyAuth
    model: str
    call_id: str | None
    messages: Sequence[Mapping[str, object]] = ({"role": "user", "content": "hello there"},)
    call_type: str = "acompletion"
    extra: Mapping[str, object] = field(default_factory=dict)

    def data(self) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": list(self.messages),
            **({} if self.call_id is None else {"litellm_call_id": self.call_id}),
            "metadata": {"user_api_key_auth": self.auth.model_copy(deep=True)},
            **self.extra,
        }


def _kwargs(
    request: Request, model_group: object, tokens: tuple[int, int], failed: bool
) -> dict[str, object]:  # mutable-ok: the legacy logging kwargs are a dict
    auth: Final = request.auth
    return {
        "litellm_call_id": request.call_id,
        "call_type": request.call_type,
        "model": request.model,
        "litellm_params": {"metadata": {"model_group": model_group}},
        "standard_logging_object": {
            "status": "failure" if failed else "success",
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "total_tokens": sum(tokens),
            "metadata": {
                "user_api_key_auth_metadata": auth.metadata,
                "user_api_key_team_id": auth.team_id,
                "user_api_key_team_alias": auth.team_alias,
            },
        },
        **({"exception": RuntimeError("provider down")} if failed else {}),
    }


def _turn_kwargs(connection: Request, turn: "Turn", index: int) -> dict[str, object]:
    kwargs: Final = _kwargs(connection, turn.model, turn.tokens, turn.failed)
    litellm_params: Final = kwargs["litellm_params"]
    assert isinstance(litellm_params, dict)
    litellm_params["litellm_metadata"] = {
        RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY: turn.connection or connection.call_id
    }
    return {**kwargs, "litellm_call_id": f"{connection.call_id}-turn-{index}", "call_type": turn.call_type}


def _response(tokens: tuple[int, int]) -> ModelResponse:
    return ModelResponse(usage=Usage(prompt_tokens=tokens[0], completion_tokens=tokens[1], total_tokens=sum(tokens)))


@dataclass
class Turn:
    model: str | None
    tokens: tuple[int, int] = (10, 20)
    call_type: str = "aresponses_websocket"
    failed: bool = False
    connection: str | None = None


@dataclass
class Step:
    request: Request
    tokens: tuple[int, int] | None = (10, 20)
    served_group: object = None
    failed: bool = False
    partial: tuple[int, int] | None = None
    retried: bool = False
    streamed: tuple[int, int] | None = None
    fallbacks: Sequence[tuple[str, bool]] = ()
    turns: Sequence[Turn] = ()
    advance: float = 0.0
    reply: tuple[int, int] | None = None
    logged_metadata: object = None
    walk: Sequence[str] = ()
    refused: bool = False


async def _pre_call_walk(
    side: Side,
    step: Step,
    data: dict[str, object],  # mutable-ok: the hooks take the proxy's request dict
) -> list[Outcome]:  # mutable-ok: the walk's outcomes, built in order
    outcomes: Final[list[Outcome]] = []  # mutable-ok: one outcome per walk state, in order
    for state in step.walk or ("off",):
        if state == "armed":
            arm_saturation_fallback_walk()
        if state == "exhausted":
            mark_saturation_fallbacks_exhausted()
        outcomes.append(await side.pre_call(data, step.request.call_type))
    return outcomes


async def _run_step(side: Side, step: Step) -> list[object]:
    async def lifecycle() -> list[object]:
        data: Final = step.request.data()
        transcript: Final[list[object]] = []
        get_or_create_request_stash()
        outcomes: Final = await _pre_call_walk(side, step, data)
        transcript.extend(outcomes)
        outcome: Final = outcomes[-1]
        if outcome.raised is not None:
            return transcript
        transcript.extend(
            [await side.fallback(step.request.auth, model, data, commit) for model, commit in step.fallbacks]
        )
        if step.refused:
            await side.post_call_failure(data)
            await side.post_call_failure(data)
            return transcript
        for index, turn in enumerate(step.turns):
            refused = await side.frame(step.request.auth, turn.model)
            transcript.append(refused)
            if refused is None and turn.model is not None:
                await side.ended(
                    _turn_kwargs(step.request, turn, index),
                    None if turn.failed else _response(turn.tokens),
                    turn.failed,
                )
        if step.tokens is None:
            return transcript
        response: Final = _response(step.streamed or step.reply or step.tokens)
        if not step.failed:
            transcript.append(await side.post_call(data, response))
        transcript.append(await side.response_headers(data))
        served: Final = (
            step.served_group
            if step.served_group is not None
            else (outcome.model if isinstance(outcome.model, str) else None)
        )
        if step.retried:
            await side.ended(_kwargs(step.request, served, (0, 0), True), None, True)
        kwargs: Final = {
            **_kwargs(step.request, served, step.tokens, step.failed),
            **({"exception": RuntimeError("an earlier attempt failed")} if step.retried else {}),
            **({} if step.streamed is None else {"async_complete_streaming_response": response}),
        }
        if step.logged_metadata is not None:
            logged = kwargs["standard_logging_object"]
            assert isinstance(logged, dict)
            kwargs["standard_logging_object"] = {**logged, "metadata": step.logged_metadata}
        if step.partial is not None:
            kwargs["combined_usage_object"] = Usage(
                prompt_tokens=step.partial[0], completion_tokens=step.partial[1], total_tokens=sum(step.partial)
            )
        await side.ended(kwargs, None if step.failed else response, step.failed)
        return transcript

    return await asyncio.create_task(lifecycle(), context=contextvars.Context())


async def _counters(side: Side, scopes: Sequence[tuple[str, str]]) -> list[object]:
    return [
        (key, value, kind, await side.limiter.rate_limits.get(RateLimitScope(key, value).counter(kind)))
        for key, value in scopes
        for kind in ("requests", "tokens")
    ]


def _router(*groups: tuple[str, int | None, int | None]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "https://example.invalid",
                    **({} if tpm is None else {"tpm": tpm}),
                    **({} if rpm is None else {"rpm": rpm}),
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
            }
            for name, tpm, rpm in groups
        ]
    )


class _MetricsTap(PrometheusLogger):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def increment_dynamic_limiter_throttled(
        self, model_group: str, team: str, team_alias: str | None, priority: str, rate_limit_type: str, reason: str
    ) -> None:
        self.calls.append(("throttled", model_group, team, team_alias, priority, rate_limit_type, reason))

    def set_dynamic_limiter_share(
        self,
        model_group: str,
        team: str,
        team_alias: str | None,
        priority: str,
        share_tokens: int,
        borrowed_tokens: int,
    ) -> None:
        self.calls.append(("share", model_group, team, team_alias, priority, share_tokens, borrowed_tokens))


class _UnlicensedTap(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.count = 0
        self.lines: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == UNLICENSED_PRIORITY_MESSAGE:
            self.count += 1
        self.lines.append((record.levelname, record.getMessage()))

    def take(self) -> int:
        taken: Final = self.count
        self.count = 0
        return taken

    def take_lines(self) -> tuple[tuple[str, str], ...]:
        taken: Final = tuple(self.lines)
        self.lines.clear()
        return taken


async def _differential(
    monkeypatch: pytest.MonkeyPatch,
    clock: Clock,
    router: Router,
    steps: Sequence[Step],
    scopes: Sequence[tuple[str, str]],
    prepare: Callable[[Side], None] | None = None,
    metrics: bool = False,
) -> list[object]:
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    tap: Final = _UnlicensedTap()
    verbose_proxy_logger.addHandler(tap)
    monkeypatch.setattr(verbose_proxy_logger, "level", logging.INFO)
    side: Final = NativeSide(router, clock)
    if prepare is not None:
        prepare(side)
    gauges: Final = _MetricsTap()
    monkeypatch.setattr(litellm, "callbacks", [side.logger, gauges] if metrics else [side.logger])
    try:
        transcript: Final = [*await _transcript(side, clock, steps, scopes, tap), tuple(gauges.calls)]
    finally:
        verbose_proxy_logger.removeHandler(tap)
    golden.check(transcript)
    return transcript


async def _transcript(
    side: Side, clock: Clock, steps: Sequence[Step], scopes: Sequence[tuple[str, str]], tap: _UnlicensedTap
) -> list[object]:
    transcript: Final[list[object]] = []
    for step in steps:
        clock.advance(step.advance)
        transcript.append([*await _run_step(side, step), ("unlicensed", tap.take()), ("logged", tap.take_lines())])
    transcript.append(await _counters(side, scopes))
    return transcript


def _refusal_prefixes(transcript: Sequence[object]) -> list[str]:
    return [
        str(step[0].raised[2]["error"]).split(" for ")[0].split(".")[0]
        for step in transcript
        if isinstance(step, list) and isinstance(step[0], Outcome) and step[0].raised is not None
    ]


def _key(name: str, **fields: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key=f"sk-{name}", **fields)  # pyright: ignore[reportArgumentType]


async def test_the_group_is_held_to_its_deployments_requests(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    router: Final = _router(("capped", None, 3))
    key: Final = _key("a")
    steps: Final = [Step(Request(key, "capped", f"c{i}")) for i in range(5)]
    native: Final = await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "capped")])
    refusals: Final = [step[0].raised for step in native[:5] if step[0].raised is not None]
    assert len(refusals) == 2


async def test_capacity_refusals_keep_their_own_error_beside_the_parallel_limiters(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    def limiters_refusal(_wire: object) -> Exception:
        return RuntimeError("the parallel limiter's refusal")

    monkeypatch.setitem(ABORT_ERRORS, "proxy_rate_limit", limiters_refusal)
    router: Final = _router(("capped", None, 1))
    steps: Final = [Step(Request(_key("a"), "capped", f"k{i}")) for i in range(2)]
    native: Final = await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "capped")])
    assert native[1][0].raised[0] == "ProxyRateLimitError", native[1]
    assert ABORT_ERRORS["proxy_rate_limit"] is limiters_refusal


async def test_busy_groups_hold_priorities_to_their_share_and_tokens_settle(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.9, "low": 0.1})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.3)
    router: Final = _router(("shared", 1000, 20))
    high: Final = _key("high", metadata={"priority": "high"})
    low: Final = _key("low", metadata={"priority": "low"})
    plain: Final = _key("plain")
    steps: Final = [
        *(Step(Request(high, "shared", f"h{i}"), tokens=(60, 40)) for i in range(4)),
        *(Step(Request(low, "shared", f"l{i}"), tokens=(10, 10)) for i in range(3)),
        *(Step(Request(plain, "shared", f"p{i}"), tokens=(10, 10)) for i in range(2)),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "shared"),
            ("priority_model", "shared:high"),
            ("priority_model", "shared:low"),
            ("priority_model", "shared:default_pool"),
        ],
    )
    assert _refusal_prefixes(native) == ["Priority-based rate limit exceeded"], native


async def test_over_allocated_and_absolute_reservations_normalize_the_same_way(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    monkeypatch.setattr(
        litellm,
        "priority_reservation",
        {"a": 0.6, "b": {"type": "rpm", "value": 8}, "c": {"type": "tpm", "value": 300}},
    )
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    router: Final = _router(("norm", 1000, 10))
    steps: Final = [
        Step(Request(_key(name, metadata={"priority": name}), "norm", f"{name}{i}"), tokens=(30, 30))
        for i in range(3)
        for name in ("a", "b", "c")
    ]
    await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "norm"), *(("priority_model", f"norm:{name}") for name in "abc")],
    )


async def test_team_priority_wins_and_headers_follow_it(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"gold": 0.5})
    router: Final = _router(("hdr", 1000, 100))
    keys: Final = [
        _key("t", team_id="t1", team_metadata={"priority": "gold"}, metadata={"priority": "other"}),
        _key("u", metadata={"priority": "ünïcødé"}),
        _key("v", metadata={"priority": "日本"}),
    ]
    steps: Final = [Step(Request(key, "hdr", f"h{i}")) for i, key in enumerate(keys)]
    await _differential(monkeypatch, isolated, router, steps, [("priority_model", "hdr:gold")])


async def test_class_split_holds_each_active_team_to_its_slice(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.75, "batch": 0.25})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.2)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "priority_class_share_splits_equally_among_active_teams", True
    )
    router: Final = _router(("split", 1000, None))
    teams: Final = [
        _key("a", team_id="team-a", team_metadata={"priority": "interactive"}),
        _key("b", team_id="team-b", team_metadata={"priority": "interactive"}),
        _key("c", team_id="team-c", team_metadata={"priority": "batch"}),
        _key("pool", metadata={"priority": "batch"}),
    ]
    steps: Final = [
        *(Step(Request(team, "split", f"w-{index}"), tokens=(50, 40)) for index, team in enumerate(teams)),
        *(Step(Request(teams[0], "split", f"a-{index}"), tokens=(100, 100)) for index in range(4)),
        *(
            Step(Request(team, "split", f"s{round_}-{index}"), tokens=(50, 40))
            for round_ in range(3)
            for index, team in enumerate(teams)
        ),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "split"),
            ("priority_class_share_saturated", "5:split:team-a"),
            ("priority_class_share_saturated", "5:split:team-b"),
            ("priority_class_share_saturated", "5:split:team-c"),
            ("priority_model", "split:batch"),
        ],
        metrics=True,
    )
    assert {call[0] for call in native[-1]} == {"throttled", "share"}, native[-1]
    assert "Priority class share reached" in _refusal_prefixes(native), native


def _borrowing(monkeypatch: pytest.MonkeyPatch, threshold: float) -> None:
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", threshold)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "team_model_limit_borrows_idle_capacity_from_other_teams", True
    )


async def test_borrowing_teams_hold_tokens_settle_and_refund(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.3)
    router: Final = _router(("borrow", 1000, None))
    team: Final = _key("team", team_id="team-a", team_metadata={"model_tpm_limit": {"borrow": 120}})
    capped: Final = _key(
        "capped",
        team_id="team-b",
        team_metadata={"model_tpm_limit": {"borrow": 60}, "team_model_limit_max_borrow_multiple": 1.5},
    )
    long_prompt: Final = ({"role": "user", "content": "word " * 40},)
    steps: Final = [
        Step(Request(team, "borrow", "b0", messages=long_prompt), tokens=(50, 30)),
        Step(Request(capped, "borrow", "b1"), tokens=(40, 30)),
        Step(Request(capped, "borrow", "b2", extra={"max_tokens": 500}), tokens=(40, 30)),
        Step(Request(team, "borrow", "b3"), tokens=(30, 0), failed=True, partial=(5, 7)),
        Step(Request(team, "borrow", "b4"), tokens=(30, 0), failed=True),
        Step(Request(team, "borrow", "b5", messages=long_prompt), tokens=(200, 200)),
        *(Step(Request(team, "borrow", f"r{i}"), tokens=(100, 100)) for i in range(4)),
        Step(Request(capped, "borrow", "b6"), tokens=(10, 10)),
        Step(Request(team, "borrow", "w0"), tokens=(10, 10), advance=61),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "borrow"),
            ("model_per_team_saturated", "team-a:borrow"),
            ("model_per_team_saturated", "team-b:borrow"),
            ("model_per_team_max_borrow_multiple", "team-b:borrow"),
        ],
        metrics=True,
    )
    assert {call[0] for call in native[-1]} == {"throttled", "share"}, native[-1]
    assert set(_refusal_prefixes(native)) == {
        "Team model_tpm_limit borrow cap reached",
        "Team model_tpm_limit reached",
    }, native


async def test_a_retried_attempt_and_a_streamed_reply_settle_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("retry", 1000, None))
    team: Final = _key("team", team_id="team-t", team_metadata={"model_tpm_limit": {"retry": 400}})
    steps: Final = [
        Step(Request(team, "retry", "t0"), tokens=(40, 20), retried=True),
        Step(Request(team, "retry", "t1"), tokens=(5, 5), streamed=(70, 50)),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "retry"), ("model_per_team_saturated", "team-t:retry")],
    )
    assert native[-2][3][3] == 60 + 120, native[-2]


async def test_an_unavailable_team_registry_and_a_failing_post_call_log_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.75, "batch": 0.25})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "priority_class_share_splits_equally_among_active_teams", True
    )
    router: Final = _router(("down", 1000, None))
    team: Final = _key("a", team_id="team-a", team_metadata={"priority": "interactive"})

    async def registry_down(*_args: object, **_kwargs: object) -> object:
        raise RedisError("registry down")

    async def post_call_fails(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("post-call down")

    def prepare(side: Side) -> None:
        registry = side.logger.delegate.active_teams
        monkeypatch.setattr(registry, "touch", registry_down)
        monkeypatch.setattr(side.limiter, "async_post_call_success_hook", post_call_fails)

    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        [Step(Request(team, "down", "d0"))],
        [("model_saturation_check", "down")],
        prepare,
    )
    assert [line[1].split(":")[0] for line in native[0][-1][1]] == [
        "Active team registry unavailable for down, holding team team-a to the class pool",
        "Error in dynamic rate limiter v3 post-call hook",
    ], native[0]


async def test_responses_calls_cap_their_output_the_same_way(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("resp", 1000, None))
    team: Final = _key("team", team_id="team-r", team_metadata={"model_tpm_limit": {"resp": 80}})
    steps: Final = [
        Step(Request(team, "resp", "r0", call_type="aresponses", extra={"input": "short prompt"}), tokens=(5, 5)),
        Step(Request(team, "resp", "r1", call_type="aresponses", extra={"input": "short", "max_output_tokens": 9})),
    ]
    await _differential(monkeypatch, isolated, router, steps, [("model_per_team_saturated", "team-r:resp")])


async def test_fallback_targets_are_held_to_the_teams_share_and_settle_there(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    _borrowing(monkeypatch, 0.2)
    router: Final = _router(("primary", 1000, None), ("backup", 500, None))
    team: Final = _key("team", team_id="team-f", team_metadata={"model_tpm_limit": {"primary": 100, "backup": 60}})
    steps: Final = [
        Step(Request(team, "primary", "f0"), fallbacks=(("backup", False), ("backup", True)), served_group="backup"),
        Step(
            Request(team, "primary", "f1"),
            fallbacks=(("backup", False), ("backup", True)),
            served_group="backup",
            failed=True,
            partial=(3, 4),
        ),
        Step(Request(team, "primary", "f2"), tokens=(200, 200), served_group="backup", fallbacks=(("backup", True),)),
        Step(Request(team, "primary", "f3"), fallbacks=(("backup", False), ("backup", True), ("backup", True))),
        Step(
            Request(team, "primary", "f4", messages=({"role": "user", "content": "word " * 80},)),
            fallbacks=(("backup", False), ("backup", True)),
        ),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "primary"),
            ("model_saturation_check", "backup"),
            ("model_per_team_saturated", "team-f:primary"),
            ("model_per_team_saturated", "team-f:backup"),
        ],
    )
    assert native[4][1:3] == [False, False], native[4]


async def test_requests_outside_the_limiter_pass_untouched(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    router: Final = _router(("known", 100, 2))
    key: Final = _key("k")
    steps: Final = [
        Step(Request(key, "unknown-model", "u0")),
        Step(Request(key, "known", "k0"), served_group=None),
        Step(Request(key, "known", "k1"), tokens=(0, 0)),
    ]
    await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "known")])


async def test_bad_priority_configuration_fails_open_the_same_way(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"a": "not a number"})
    router: Final = _router(("open", 100, 1))
    steps: Final = [Step(Request(_key("a", metadata={"priority": "a"}), "open", f"o{i}")) for i in range(3)]
    await _differential(monkeypatch, isolated, router, steps, [("model_saturation_check", "open")])


async def test_non_string_priorities_render_and_match_like_python(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {7: 0.5, 1: 0.2, "x": 0.3})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    router: Final = _router(("nsp", 1000, 6))
    priorities: Final = (7, 7.0, True, 3, [1], {"a": 1}, "日本", "x", None)
    steps: Final = [
        Step(Request(_key(f"k{index}", metadata={"priority": priority}), "nsp", f"n{round_}-{index}"), tokens=(5, 5))
        for round_ in range(2)
        for index, priority in enumerate(priorities)
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "nsp"),
            *(("priority_model", f"nsp:{name}") for name in ("7", "7.0", "True", "1", "x", "default_pool")),
        ],
    )
    assert _refusal_prefixes(native), native


async def test_team_shares_sum_float_weights_like_python(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"a": 0.1, 2: 0.2, "c": 0.3, 4.5: 0.4})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "priority_class_share_splits_equally_among_active_teams", True
    )
    router: Final = _router(("sum", 997, None))
    teams: Final = [
        _key("a", team_id="team-a", team_metadata={"priority": "a"}),
        _key("b", team_id="team-b", team_metadata={"priority": 2}),
        _key("c", team_id="team-c", team_metadata={"priority": "c"}),
        _key("d", team_id="team-d", team_metadata={"priority": 4.5}),
        _key("e", team_id="team-e", team_metadata={"priority": 2.0}),
    ]
    steps: Final = [
        Step(Request(team, "sum", f"s{round_}-{index}"), tokens=(70, 60))
        for round_ in range(3)
        for index, team in enumerate(teams)
    ]
    await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_saturation_check", "sum"),
            *(("priority_class_share_saturated", f"{len(teams)}:sum:team-{name}") for name in "abcde"),
        ],
    )


async def test_requests_without_a_call_id_keep_their_stash(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.5)
    router: Final = _router(("luna", 1000, 4))
    team: Final = _key("team", team_id="flex", team_metadata={"model_tpm_limit": {"luna": 100}})
    steps: Final = [
        Step(Request(_key("plain"), "luna", None)),
        Step(Request(_key("plain"), "luna", None)),
        Step(Request(team, "luna", None)),
        Step(Request(team, "luna", None), failed=True, partial=(2, 3)),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "luna"), ("model_per_team_saturated", "flex:luna")],
    )
    assert native[-2][3][3] == 30 + 5, native[-2]


@pytest.mark.parametrize(
    ("priorities", "unhashable"),
    [
        ({"team_metadata": {"priority": [1]}, "metadata": {"priority": "x"}}, "list"),
        ({"team_metadata": {"priority": {"a": 1}}}, "dict"),
        ({"team_metadata": {"priority": None}, "metadata": {"priority": [2]}}, "list"),
        ({"team_metadata": {"priority": 7}, "metadata": {"priority": [2]}}, None),
    ],
)
async def test_priorities_the_contract_cannot_hold_fail_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, priorities: Mapping[str, object], unhashable: str | None
):
    monkeypatch.setattr(litellm, "priority_reservation", {7: 0.5, "x": 0.5})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    router: Final = _router(("nsp", 1000, 6))
    key: Final = _key("held", team_id="team-h", **priorities)
    steps: Final = [Step(Request(key, "nsp", f"h{index}"), tokens=(5, 5)) for index in range(2)]
    native: Final = await _differential(
        monkeypatch, isolated, router, steps, [("model_saturation_check", "nsp"), ("priority_model", "nsp:7")]
    )
    logged: Final = " ".join(line[1] for line in native[0][-1][1])
    assert (f"unhashable type: '{unhashable}'" in logged) is (unhashable is not None), native[0]


@pytest.mark.parametrize("team_id", [["team-l"], 5, True])
async def test_team_ids_python_formats_share_the_class_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, team_id: object
):
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "priority_class_share_splits_equally_among_active_teams", True
    )
    router: Final = _router(("ids", 400, None))
    key: Final = _key("odd", team_metadata={"priority": "interactive"})
    key.team_id = team_id  # pyright: ignore[reportAttributeAccessIssue]  # a team id another writer left in any shape
    steps: Final = [Step(Request(key, "ids", f"i{index}"), tokens=(60, 40)) for index in range(4)]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "ids"), ("priority_class_share_saturated", f"3:ids:{team_id}")],
    )
    assert [status["descriptor_value"] for status in native[0][0].statuses] == ["ids", f"3:ids:{team_id}"], native[0]


async def test_a_reply_whose_usage_differs_from_the_logged_counts_settles_the_reply_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("reply", 1000, None))
    team: Final = _key("team", team_id="team-r", team_metadata={"model_tpm_limit": {"reply": 400}})
    steps: Final = [Step(Request(team, "reply", "r0"), tokens=(5, 5), reply=(70, 50))]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "reply"), ("model_per_team_saturated", "team-r:reply")],
    )
    assert native[-2][3][3] == 120, native[-2]


@pytest.mark.parametrize(
    ("logged_metadata", "settles"),
    [
        ({"user_api_key_team_id": 5, "user_api_key_auth_metadata": {"priority": "gold"}}, False),
        ({"user_api_key_team_id": "team-v", "user_api_key_team_alias": True}, False),
        ({"user_api_key_team_id": ["team-v"]}, False),
        ({"user_api_key_team_id": "team-v", "user_api_key_team_alias": ["alias"]}, False),
        ({"user_api_key_team_id": "team-v", "user_api_key_auth_metadata": "not a mapping"}, False),
        ("not a mapping", False),
        ({"user_api_key_team_id": "team-v", "user_api_key_auth_metadata": {"priority": 7}}, True),
    ],
)
async def test_logged_metadata_python_cannot_validate_settles_nothing_and_logs_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, logged_metadata: object, settles: bool
):
    monkeypatch.setattr(litellm, "priority_reservation", {"gold": 0.5, 7: 0.5})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    router: Final = _router(("valid", 1000, None))
    team: Final = _key("team", team_id="team-v", metadata={"priority": "gold"})
    steps: Final = [
        Step(Request(team, "valid", "v0"), tokens=(30, 10)),
        Step(Request(team, "valid", "v1"), tokens=(30, 10), logged_metadata=logged_metadata),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "valid"), ("priority_model", "valid:gold"), ("priority_model", "valid:7")],
    )
    logged: Final = [line[1].split(":")[0] for line in native[1][-1][1]]
    assert logged == ([] if settles else ["Error in dynamic rate limiter success event"]), native[1]
    assert native[-2][1][3] == (80 if settles else 40), native[-2]


async def test_unlicensed_priorities_log_the_premium_line_the_same_number_of_times(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    monkeypatch.delenv("LITELLM_LICENSE")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.7, "low": 0.3})
    router: Final = _router(("lic", 1000, 10))
    steps: Final = [
        Step(Request(_key(name, metadata={"priority": name}), "lic", f"{name}{i}"))
        for i in range(2)
        for name in ("high", "low", "none")
    ]
    native: Final = await _differential(monkeypatch, isolated, router, steps, [("priority_model", "lic:high")])
    assert sum(step[-2][1] for step in native[:-2]) > 0, native


def _socket_connection(name: str, key: UserAPIKeyAuth, turns: Sequence[Turn]) -> Step:
    return Step(Request(key, "sock", f"conn-{name}", call_type="aresponses_websocket"), tokens=None, turns=tuple(turns))


async def test_socket_frames_hold_turns_to_the_team_share_and_turns_settle_on_the_connection(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("sock", 1000, None))
    team: Final = _key("team", team_id="team-s", team_metadata={"model_tpm_limit": {"sock": 150}})
    turns: Final = (
        Turn("sock", (40, 20)),
        Turn("sock", (30, 10), call_type="aresponses"),
        Turn("sock", (10, 0), failed=True),
        Turn("sock", (50, 30)),
        Turn("sock"),
        Turn(None),
        Turn("sock", call_type="aresponses", connection="someone-else"),
    )
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        [_socket_connection("a", team, turns), _socket_connection("b", team, (Turn("sock"),))],
        [("model_saturation_check", "sock"), ("model_per_team_saturated", "team-s:sock")],
    )
    assert any(isinstance(entry, tuple) and entry[0] == "ProxyRateLimitError" for entry in native[0]), native[0]


async def test_socket_frames_hold_turns_to_the_class_share(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.0)
    monkeypatch.setattr(
        litellm.priority_reservation_settings, "priority_class_share_splits_equally_among_active_teams", True
    )
    router: Final = _router(("sock", 400, None))
    first: Final = _key("a", team_id="team-a", team_metadata={"priority": "interactive"})
    second: Final = _key("b", team_id="team-b", team_metadata={"priority": "interactive"})
    unhashable: Final = _key("l", team_id="team-l", metadata={"priority": [1]})
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        [
            _socket_connection("b", second, (Turn("sock", (5, 5)),)),
            _socket_connection("a", first, [Turn("sock", (60, 40)) for _ in range(4)]),
            _socket_connection("l", unhashable, (Turn("sock"),)),
        ],
        [
            ("model_saturation_check", "sock"),
            ("priority_class_share_saturated", "2:sock:team-a"),
            ("priority_class_share_saturated", "2:sock:team-b"),
        ],
    )
    assert native[2][1] == ("TypeError", "unhashable type: 'list'"), native[2]
    assert any(
        isinstance(entry, tuple) and str(entry[2]).startswith("{'error': 'Priority class share reached")
        for entry in native[1]
    ), native[1]


async def test_an_unverifiable_frame_check_is_a_service_unavailable(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("sock", 1000, None))
    team: Final = _key("team", team_id="team-u", team_metadata={"model_tpm_limit": {"sock": 150}})

    async def unverifiable(*_args: object, **_kwargs: object) -> object:
        raise RateLimitUnverifiableError()

    def prepare(side: Side) -> None:
        monkeypatch.setattr(side.limiter, "should_rate_limit", unverifiable)

    native: Final = await _differential(
        monkeypatch, isolated, router, [_socket_connection("u", team, (Turn("sock"),))], [], prepare
    )
    assert native[0][1][0] == "ServiceUnavailableError", native[0]


async def _settled() -> None:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    for _ in range(500):
        await asyncio.sleep(0.01)
        if GLOBAL_LOGGING_WORKER.pending_count == 0:
            return


async def _logged_lifecycle(side: Side, router: Router, request: Request, failed: bool) -> list[object]:
    async def lifecycle() -> list[object]:
        data: Final = request.data()
        get_or_create_request_stash()
        outcome: Final = await side.pre_call(data, request.call_type)
        if outcome.raised is not None:
            return [outcome]
        auth: Final = request.auth
        try:
            await router.acompletion(
                model=request.model,
                messages=list(request.messages),
                litellm_call_id=request.call_id,
                metadata={
                    "user_api_key_auth": auth,
                    "user_api_key_team_id": auth.team_id,
                    "user_api_key_team_alias": auth.team_alias,
                },
                mock_response=Exception("provider down") if failed else "settled through logging",
                num_retries=0,
            )
        except Exception as error:
            await _settled()
            return [outcome, type(error).__name__]
        await _settled()
        return [outcome, None]

    return await asyncio.create_task(lifecycle(), context=contextvars.Context())


async def test_attempts_settle_through_the_real_logging_path(monkeypatch: pytest.MonkeyPatch, isolated: Clock):
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("logged", 1000, None))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    team: Final = _key(
        "team", team_id="team-l", team_alias="logged-team", team_metadata={"model_tpm_limit": {"logged": 400}}
    )
    plan: Final = [("l0", False), ("l1", True), ("l2", False), ("l3", True)]
    scopes: Final = [("model_saturation_check", "logged"), ("model_per_team_saturated", "team-l:logged")]
    side: Final = NativeSide(router, isolated)
    for name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    litellm.callbacks.append(side.logger)
    native: Final[list[object]] = []  # mutable-ok: the native's transcript, in order
    for call_id, failed in plan:
        native.append(await _logged_lifecycle(side, router, Request(team, "logged", call_id), failed))
        native.append(await _counters(side, scopes))
    golden.check(native)
    assert [step[1][3] for step in native[1::2]] == [30, 30, 60, 60], native
    assert [step[1] for step in native[0::2]] == [None, "InternalServerError", None, "InternalServerError"], native


class _PayloadCapture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[object] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.payloads.append(kwargs.get("standard_logging_object"))


def _payload_headers(payload: object) -> object:
    hidden: Final = payload.get("hidden_params") if isinstance(payload, Mapping) else None
    return hidden.get("additional_headers") if isinstance(hidden, Mapping) else None


@dataclass(frozen=True)
class ProxyRun:
    stashes_created: int
    model: object
    response_headers: object
    header_hook: object
    payload_headers: object
    matched_groups: object


async def _through_the_proxy(
    proxy_logging: ProxyLogging, router: Router, capture: _PayloadCapture, request: Request, metadata_auth: bool
) -> ProxyRun:
    created: Final[list[RequestRateLimiterStash]] = []

    class _Counted(RequestRateLimiterStash):
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    async def lifecycle() -> ProxyRun:
        data: Final = request.data()
        live_auth: Final = _auth(data)
        if not metadata_auth:
            data["metadata"] = {}
        with request_scope(), pytest.MonkeyPatch.context() as patch:
            patch.setattr(parallel_request_limiter_v3, "RequestRateLimiterStash", _Counted)
            admitted: Final = await proxy_logging.pre_call_hook(
                user_api_key_dict=live_auth, data=data, call_type="acompletion"
            )
            assert admitted is not None
            response: Final = await router.acompletion(
                model=str(admitted["model"]),
                messages=list(request.messages),
                litellm_call_id=request.call_id,
                metadata={"user_api_key_auth": live_auth},
                mock_response="through the proxy",
            )
            await proxy_logging.post_call_success_hook(data=admitted, response=response, user_api_key_dict=live_auth)
            header_hook: Final = await proxy_logging.post_call_response_headers_hook(
                data=admitted, user_api_key_dict=live_auth, response=response
            )
            await _settled()
        return ProxyRun(
            stashes_created=len(created),
            model=admitted["model"],
            response_headers=dict(response._hidden_params.get("additional_headers") or {}),
            header_hook=header_hook,
            payload_headers=_payload_headers(capture.payloads[-1]) if capture.payloads else None,
            matched_groups=live_auth.matched_model_access_groups,
        )

    return await asyncio.create_task(lifecycle(), context=contextvars.Context())


@pytest.mark.parametrize("metadata_auth", [True, False])
@pytest.mark.parametrize("order", ["alone", "before_parallel_limiter", "after_parallel_limiter"])
async def test_the_real_proxy_hooks_share_one_request_stash_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, order: str, metadata_auth: bool
) -> None:
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    requests: Final = [
        Request(_key("plain"), "luna", "p0"),
        Request(_key("plain"), "luna", "p1"),
        Request(_key("team", team_id="flex"), "luna", "f0"),
    ]
    router: Final = _router(("luna", None, 4))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    limiter: Final = NativeSide(router, isolated).logger
    parallel: Final = PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(dual_cache=DualCache()))
    capture: Final = _PayloadCapture()
    registered: Final = {
        "alone": [limiter],
        "before_parallel_limiter": [limiter, parallel],
        "after_parallel_limiter": [parallel, limiter],
    }[order]
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [*registered, capture])
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    native: Final = [
        await _through_the_proxy(proxy_logging, router, capture, request, metadata_auth) for request in requests
    ]
    golden.check(native)
    assert [run.stashes_created for run in native] == [1, 1, 1], native
    assert [run.model for run in native] == ["luna", "luna", "luna"], native
    assert all(
        isinstance(run.payload_headers, Mapping)
        and "x-ratelimit-model_saturation_check-remaining-requests" in run.payload_headers
        and "x-ratelimit-model_saturation_check-remaining-requests" in run.response_headers
        for run in native
    ), native


async def _unscoped_through_the_proxy(
    proxy_logging: ProxyLogging, router: Router, request: Request, failed: bool
) -> list[object]:
    async def lifecycle() -> list[object]:
        data: Final = request.data()
        live_auth: Final = _auth(data)
        try:
            admitted: Final = await proxy_logging.pre_call_hook(
                user_api_key_dict=live_auth,
                data=data,
                call_type=request.call_type,  # pyright: ignore[reportArgumentType]  # the endpoints pass their own call type string
            )
        except Exception as error:
            return [_raised(error)]
        assert admitted is not None
        try:
            response: Final = await router.acompletion(
                model=str(admitted["model"]),
                messages=list(request.messages),
                litellm_call_id=request.call_id,
                metadata={
                    "user_api_key_auth": live_auth,
                    "user_api_key_team_id": live_auth.team_id,
                    "user_api_key_team_alias": live_auth.team_alias,
                },
                mock_response=Exception("provider down") if failed else "outside a request scope",
                num_retries=0,
            )
        except Exception as error:
            await _settled()
            return [admitted["model"], type(error).__name__]
        header_hook: Final = await proxy_logging.post_call_response_headers_hook(
            data=admitted, user_api_key_dict=live_auth, response=response
        )
        await _settled()
        return [admitted["model"], header_hook]

    return await asyncio.create_task(lifecycle(), context=contextvars.Context())


@pytest.mark.parametrize("call_type", ["aimage_generation", "arerank", "amoderation", "acompletion"])
async def test_requests_without_a_request_scope_keep_their_stash_through_the_real_proxy_hooks(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, call_type: str
) -> None:
    _borrowing(monkeypatch, 0.5)
    flexible: Final = _key("flex", team_id="flex", team_metadata={"model_tpm_limit": {"luna": 400}})
    solo: Final = _key("solo", team_id="solo-team", team_metadata={"model_tpm_limit": {"solo": 400}})
    plan: Final = [
        (Request(_key("plain"), "luna", "p0", call_type=call_type), False),
        (Request(_key("plain"), "luna", "p1", call_type=call_type), False),
        (Request(flexible, "luna", "f0", call_type=call_type), False),
        (Request(flexible, "luna", "f1", call_type=call_type), True),
        *((Request(solo, "solo", f"s{index}", call_type=call_type), index == 3) for index in range(4)),
    ]
    scopes: Final = [
        ("model_saturation_check", "luna"),
        ("model_per_team_saturated", "flex:luna"),
        ("model_per_team_saturated", "solo-team:solo"),
    ]
    router: Final = _router(("luna", 1000, 4), ("solo", 1000, 4))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    side: Final = NativeSide(router, isolated)
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [side.logger])
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    native: Final[list[object]] = []  # mutable-ok: the native's transcript, in order
    for request, failed in plan:
        native.append(await _unscoped_through_the_proxy(proxy_logging, router, request, failed))
        native.append(await _counters(side, scopes))
    golden.check(native)
    assert native[4] == ["luna", {}], native[4]
    assert native[6][1] == "InternalServerError", native[6]
    assert [entry[5][3] for entry in native[9::2]] == [None, None, 30, 30], native


async def test_sync_sdk_calls_settle_nothing_like_python(monkeypatch: pytest.MonkeyPatch, isolated: Clock) -> None:
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("sync", 1000, None))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    team: Final = _key("team", team_id="team-y", team_metadata={"model_tpm_limit": {"sync": 400}})
    scopes: Final = [("model_saturation_check", "sync"), ("model_per_team_saturated", "team-y:sync")]
    side: Final = NativeSide(router, isolated)
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [side.logger])
    request: Final = Request(team, "sync", "y0")

    async def lifecycle() -> object:
        outcome = await side.pre_call(request.data(), request.call_type)
        await asyncio.to_thread(
            router.completion,
            model=request.model,
            messages=list(request.messages),
            litellm_call_id=request.call_id,
            metadata={"user_api_key_auth": team, "user_api_key_team_id": team.team_id},
            mock_response="a sync sdk call",
        )
        await _settled()
        return outcome

    outcome: Final = await asyncio.create_task(lifecycle(), context=contextvars.Context())
    native: Final = [outcome, await _counters(side, scopes)]
    golden.check(native)
    assert native[1][3][3] is not None, native


class _AuthSeen(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[object] = []

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> None:
        self.seen.append(REQUEST_AUTH.get())


async def test_the_request_auth_stays_unbound_without_a_native_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    witness: Final = _AuthSeen()
    monkeypatch.setattr(litellm, "callbacks", [witness])
    auth: Final = _key("plain")
    await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
        user_api_key_dict=auth,
        data=Request(auth, "luna", "u0").data(),
        call_type="acompletion",  # pyright: ignore[reportArgumentType]  # the proxy's call type string
    )
    assert witness.seen == [None]


_UNSET: Final = object()


def _opted_in(name: str, call_id: str, fallback_on_saturation: object, walk: Sequence[str]) -> Step:
    override: Final = {} if fallback_on_saturation is _UNSET else {"fallback_on_saturation": fallback_on_saturation}
    return Step(
        Request(
            _key(name, metadata={"priority": "gold"}),
            "sat",
            call_id,
            extra={"router_settings_override": override},
        ),
        walk=walk,
    )


@pytest.mark.parametrize("fallback_on_saturation", [True, False, "yes", 1, None, _UNSET])
@pytest.mark.parametrize("walk", [("armed",), ("off",), ("armed", "exhausted")])
async def test_a_saturated_model_hands_an_opted_in_request_to_the_fallback_walk_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, fallback_on_saturation: object, walk: Sequence[str]
) -> None:
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    router: Final = _router(("sat", None, 4))
    steps: Final = [
        _opted_in("idle", "s0", fallback_on_saturation, walk),
        Step(Request(_key("plain"), "sat", "p1")),
        _opted_in("busy", "s2", fallback_on_saturation, walk),
    ]
    native: Final = await _differential(
        monkeypatch, isolated, router, steps, [("model_saturation_check", "sat")], metrics=True
    )
    handed_off: Final = fallback_on_saturation is True and walk[0] == "armed"
    assert [outcome.raised is None for outcome in native[0][: len(walk)]] == [True] * len(walk), native[0]
    refused: Final = native[2][0].raised
    assert (None if refused is None else (refused[2], refused[3], refused[-1])) == (
        (
            {
                "error": f"Model sat is at {25 * (len(walk) + 1)}% of its capacity, at or above the saturation "
                "threshold of 50%, trying fallbacks first. Priority: gold"
            },
            {"retry-after": "60", "x-litellm-priority": "gold"},
            "model_saturation_fallback",
        )
        if handed_off
        else None
    ), native[2]
    warned: Final = isinstance(fallback_on_saturation, str | int) and not isinstance(fallback_on_saturation, bool)
    assert [line for line in native[2][-1][1] if "router_settings" in line[1]] == (
        [
            (
                "WARNING",
                f"router_settings.fallback_on_saturation={fallback_on_saturation!r} is not a boolean, "
                "treating it as disabled",
            )
        ]
        * (1 if handed_off or len(walk) == 1 else 2)
        if warned
        else []
    ), native[2]
    assert not [line for line in native[0][-1][1] if "router_settings" in line[1]], native[0]
    assert native[-1] == (), native[-1]


async def test_a_handed_off_request_takes_no_hold_and_its_replay_is_admitted_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    _borrowing(monkeypatch, 0.5)
    router: Final = _router(("sat", 1000, 4))
    team: Final = _key("team", team_id="team-w", team_metadata={"model_tpm_limit": {"sat": 100}})
    opted_in: Final = {"router_settings_override": {"fallback_on_saturation": True}}
    steps: Final = [
        Step(Request(_key("plain"), "sat", "p0")),
        Step(Request(_key("plain"), "sat", "p1")),
        Step(Request(team, "sat", "w0", extra=opted_in), walk=("armed",)),
        Step(Request(team, "sat", "w1", extra=opted_in), walk=("armed", "exhausted"), tokens=(20, 10)),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [("model_saturation_check", "sat"), ("model_per_team_saturated", "team-w:sat")],
    )
    assert native[2][0].raised is not None and native[2][0].raised[-1] == "model_saturation_fallback", native[2]
    assert native[3][0].raised is not None and native[3][1].raised is None, native[3]
    assert native[-2][0][3] == 3, native[-2]
    assert native[-2][3][3] == 30, native[-2]


def _capped_team(name: str, team_id: str, limits: Mapping[str, int]) -> UserAPIKeyAuth:
    return _key(
        name,
        team_id=team_id,
        team_metadata={"model_tpm_limit": dict(limits), "team_model_limit_max_borrow_multiple": 2},
    )


async def test_a_refused_admission_gives_back_its_team_holds_once_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("held", 1000, None))
    team: Final = _key("team", team_id="team-h", team_metadata={"model_tpm_limit": {"held": 400}})
    capped: Final = _capped_team("capped", "team-c", {"held": 400})
    long_prompt: Final = ({"role": "user", "content": "word " * 40},)
    steps: Final = [
        Step(Request(team, "held", "h0", messages=long_prompt), refused=True),
        Step(Request(capped, "held", "c0", messages=long_prompt), refused=True),
        Step(Request(capped, "held", "c1", messages=long_prompt), tokens=(30, 10)),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_per_team_saturated", "team-h:held"),
            ("model_per_team_saturated", "team-c:held"),
            ("model_per_team_max_borrow_multiple", "team-c:held"),
        ],
    )
    assert [entry[3] for entry in native[-2][1::2]] == [0, 40, 40], native[-2]


async def test_a_failed_attempt_settles_outside_the_rate_limits_service_target_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("traced", 1000, None))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    team: Final = _key("team", team_id="team-t", team_metadata={"model_tpm_limit": {"traced": 400}})
    steps: Final = [
        Step(Request(team, "traced", "t0"), tokens=(20, 10)),
        Step(Request(team, "traced", "t1"), tokens=(30, 0), failed=True, partial=(5, 5)),
        Step(Request(team, "traced", "t2"), refused=True),
    ]
    side: Final = NativeSide(router, isolated)
    targets: Final[list[str | None]] = []  # mutable-ok: the service target of each increment, in order
    increment: Final = side.limiter.async_increment_reservation_aware_tokens

    async def traced(
        *args: object,
        **kwargs: object,  # kwargs-ok: the wrapped limiter method's own keywords pass through
    ) -> None:
        targets.append(current_service_target())
        await increment(*args, **kwargs)

    monkeypatch.setattr(side.limiter, "async_increment_reservation_aware_tokens", traced)
    monkeypatch.setattr(litellm, "callbacks", [side.logger])
    for step in steps:
        await _run_step(side, step)
    golden.check(targets)
    assert targets == [RATE_LIMITS_TARGET, None, RATE_LIMITS_TARGET], targets


async def test_a_capped_team_holds_its_cap_on_fallbacks_and_frames_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    _borrowing(monkeypatch, 0.9)
    router: Final = _router(("primary", 1000, None), ("backup", 1000, None), ("sock", 1000, None))
    capped: Final = _capped_team("capped", "team-k", {"primary": 60, "backup": 40, "sock": 50})
    steps: Final = [
        Step(Request(capped, "primary", "k0"), fallbacks=(("backup", False), ("backup", True)), served_group="backup"),
        Step(
            Request(capped, "primary", "k1"),
            fallbacks=(("backup", True),),
            served_group="backup",
            failed=True,
            partial=(10, 10),
        ),
        Step(Request(capped, "primary", "k2"), tokens=(40, 40)),
        Step(Request(capped, "primary", "k3", messages=({"role": "user", "content": "word " * 60},))),
        _socket_connection("k", capped, [Turn("sock", (30, 30)) for _ in range(3)]),
    ]
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_per_team_saturated", "team-k:primary"),
            ("model_per_team_max_borrow_multiple", "team-k:primary"),
            ("model_per_team_max_borrow_multiple", "team-k:backup"),
            ("model_per_team_max_borrow_multiple", "team-k:sock"),
        ],
        metrics=True,
    )
    assert "Team model_tpm_limit borrow cap reached" in _refusal_prefixes(native), native
    assert {call[0] for call in native[-1]} >= {"share"}, native[-1]


async def test_a_failing_redis_saturation_read_falls_back_to_this_workers_copy_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.5})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    router: Final = _router(("down", None, 4))

    async def redis_down(
        *_args: object,
        **_kwargs: object,  # kwargs-ok: stands in for the store's read, whatever keywords it takes
    ) -> object:
        raise RedisError("redis down")

    def prepare(side: Side) -> None:
        monkeypatch.setattr(side.limiter.rate_limits, "get_from_redis", redis_down)

    high: Final = _key("high", metadata={"priority": "high"})
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        [Step(Request(high, "down", f"d{index}")) for index in range(4)],
        [("model_saturation_check", "down"), ("priority_model", "down:high")],
        prepare,
    )
    assert {
        line for step in native[:4] for line in step[-1][1]
    } == {  # comprehension-ok: every log line of the first four steps
        ("WARNING", "Saturation read failed in Redis, using this worker's copy: redis down")
    }, native
    assert set(_refusal_prefixes(native)) == {"Priority-based rate limit exceeded"}, native


@pytest.mark.parametrize("model_group", [5, True, "", ["listed"], ("tupled",)])
async def test_logged_model_groups_of_any_type_settle_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, model_group: object
) -> None:
    _borrowing(monkeypatch, 0.0)
    router: Final = _router(("odd", 1000, None))
    team: Final = _key("team", team_id="team-o", team_metadata={"model_tpm_limit": {"odd": 400}})
    steps: Final = [Step(Request(team, "odd", "o0"), tokens=(20, 10), served_group=model_group)]
    await _differential(
        monkeypatch,
        isolated,
        router,
        steps,
        [
            ("model_per_team_saturated", "team-o:odd"),
            ("model_saturation_check", str(model_group)),
            ("priority_model", f"{model_group}:default_pool"),
        ],
    )


async def test_saturation_is_read_from_redis_on_every_check_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.5})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    router: Final = _router(("shared", None, 4))
    seen: Final[list[tuple[str, str, object]]] = []  # mutable-ok: the reads the native records, in order

    def prepare(side: Side) -> None:
        store: Final = side.limiter.rate_limits
        from_redis: Final = store.get_from_redis
        from_memory: Final = store.get

        async def other_workers(counter: RateLimitCounter, backfill_ttl: float | None = None) -> object:
            seen.append(("redis", counter.scope.descriptor_key, backfill_ttl))
            if counter.scope.descriptor_key == "model_saturation_check":
                return 3
            return await from_redis(counter, backfill_ttl=backfill_ttl)

        async def memory_first(entry: object, backfill_ttl: float | None = None) -> object:
            seen.append(("memory", type(entry).__name__, backfill_ttl))
            return await from_memory(entry, backfill_ttl=backfill_ttl)  # pyright: ignore[reportArgumentType]  # the store reads any rate limit entry

        monkeypatch.setattr(store, "get_from_redis", other_workers)
        monkeypatch.setattr(store, "get", memory_first)

    high: Final = _key("high", metadata={"priority": "high"})
    native: Final = await _differential(
        monkeypatch,
        isolated,
        router,
        [Step(Request(high, "shared", f"h{index}")) for index in range(3)],
        [("model_saturation_check", "shared"), ("priority_model", "shared:high")],
        prepare,
    )
    golden.check(seen)
    assert [read for read in seen if read[0] == "redis"] == [("redis", "model_saturation_check", 60)] * 3, seen
    assert _refusal_prefixes(native) == ["Priority-based rate limit exceeded"], native


@dataclass(frozen=True)
class _ThroughTheProxy:
    proxy_logging: ProxyLogging
    live_auth: UserAPIKeyAuth
    request: Callable[[], Mapping[str, object]]
    run: list[object]  # mutable-ok: the side's transcript, in order
    refuse: Mapping[str, Exception]
    opted_in: bool

    async def __call__(
        self,
        **_kwargs: object,  # kwargs-ok: the patched pre-call logic takes the proxy's keywords, which this stand-in ignores
    ) -> tuple[dict[str, object], None]:  # mutable-ok: the proxy's request dict
        data: Final = {**self.request(), "router_settings_override": {"fallback_on_saturation": self.opted_in}}
        try:
            admitted: Final = await self.proxy_logging.pre_call_hook(self.live_auth, data, "acompletion")
        except Exception as error:
            self.run.append(_raised(error))
            raise
        self.run.append(admitted["model"])
        refusal: Final = self.refuse.get(str(admitted["model"]))
        if refusal is not None:
            raise refusal
        return admitted, None


@dataclass(frozen=True)
class _Walk:
    key: UserAPIKeyAuth = field(default_factory=lambda: _key("opted", metadata={"priority": "gold"}))
    groups: tuple[tuple[str, int | None, int | None], ...] = (("busy", None, 4), ("spare", None, 4))
    refuse: Mapping[str, Exception] = field(default_factory=lambda: MappingProxyType({}))
    scopes: tuple[tuple[str, str], ...] = (("model_saturation_check", "busy"), ("model_saturation_check", "spare"))
    opted_in: bool = True


async def _saturation_walk(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock, shape: _Walk | None = None
) -> list[object]:  # mutable-ok: the native's transcript, in order
    from starlette.requests import Request as StarletteRequest

    from litellm.proxy import proxy_server
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.common_utils.proxy_rate_limit_error import rate_limit_fallback_from_request_data

    walk: Final = shape or _Walk()
    router: Final = _router(*walk.groups)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    side: Final = NativeSide(router, isolated)
    monkeypatch.setattr(litellm, "callbacks", [side.logger])
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())  # pyright: ignore[reportArgumentType]  # the proxy builds it from a DualCache too
    run: Final[list[object]] = []  # mutable-ok: the native's transcript, in order
    for index in range(2):
        plain = Request(_key("plain"), "busy", f"p{index}").data()
        run.append((await proxy_logging.pre_call_hook(_auth(plain), plain, "acompletion"))["model"])
    request: Final = Request(walk.key, "busy", "w0", messages=({"role": "user", "content": "word " * 40},)).data()
    live_auth: Final = _auth(request)
    processor: Final = ProxyBaseLLMRequestProcessing(data=request)
    monkeypatch.setattr(
        processor,
        "common_processing_pre_call_logic",
        _ThroughTheProxy(proxy_logging, live_auth, lambda: processor.data, run, walk.refuse, walk.opted_in),
    )

    async def lifecycle() -> tuple[dict[str, object], object]:  # mutable-ok: the proxy's request dict
        return await processor._pre_call_with_fallbacks(  # pyright: ignore[reportPrivateUsage]  # the walk under test
            request=StarletteRequest(
                {"type": "http", "method": "POST", "path": "/chat/completions", "headers": [], "query_string": b""}
            ),
            general_settings={},
            proxy_logging_obj=proxy_logging,
            user_api_key_dict=live_auth,
            version=None,
            proxy_config=_SavedRouterSettings(),  # pyright: ignore[reportArgumentType]  # stands in for the saved router_settings read
            user_model=None,
            user_temperature=None,
            user_request_timeout=None,
            user_max_tokens=None,
            user_api_base=None,
            model="busy",
            route_type="acompletion",
            llm_router=router,
        )

    served: Final = (await asyncio.create_task(lifecycle(), context=contextvars.Context()))[0]
    stamp: Final = rate_limit_fallback_from_request_data(served)
    run.append(
        (
            served["model"],
            None if stamp is None else (stamp.original_model_group, stamp.descriptor_key, stamp.attempts),
        )
    )
    run.append(await _counters(side, walk.scopes))
    return run


async def test_the_proxy_fallback_walk_takes_the_natives_saturation_handoff_like_pythons(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "enforce_fallback_model_access", False)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.5)
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    native: Final = await _saturation_walk(monkeypatch, isolated)
    golden.check(native)
    handoff: Final = native[2]
    assert isinstance(handoff, tuple)
    assert (*handoff[:4], handoff[-1]) == (
        "ProxyRateLimitError",
        429,
        {
            "error": "Model busy is at 50% of its capacity, at or above the saturation threshold of 50%, "
            "trying fallbacks first. Priority: gold"
        },
        {"retry-after": "60", "x-litellm-priority": "gold"},
        "model_saturation_fallback",
    ), native
    assert native[3:5] == ["spare", ("spare", ("busy", "model_saturation_fallback", 1))], native


class _SavedRouterSettings:
    async def get_hierarchical_router_settings(
        self,
        **_kwargs: object,  # kwargs-ok: the proxy config read's keywords, which this stand-in ignores
    ) -> Mapping[str, object]:
        return {"fallbacks": [{"busy": ["spare"]}], "fallback_on_saturation": True}


async def test_the_logger_registry_and_the_borrowing_check_find_the_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,  # pyright: ignore[reportPrivateUsage]  # the registry's own builder
        get_custom_logger_compatible_class,
    )
    from litellm.proxy.hooks import model_capacity_native
    from litellm.proxy.hooks.parallel_request_limiter_v3 import dynamic_rate_limiter_covers

    router: Final = _router(("held", 1000, None))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(model_capacity_native, "_REGISTERED", type(model_capacity_native._REGISTERED)())  # pyright: ignore[reportPrivateUsage]  # a fresh registration for this test
    monkeypatch.setattr("litellm.litellm_core_utils.litellm_logging._in_memory_loggers", [])
    built: Final = _init_custom_logger_compatible_class(
        "dynamic_rate_limiter_v3", internal_usage_cache=DualCache(), llm_router=router
    )
    monkeypatch.setattr(litellm, "callbacks", [built])
    found: Final = (
        get_custom_logger_compatible_class("dynamic_rate_limiter_v3") is built,
        dynamic_rate_limiter_covers("held"),
        isinstance(built, NativeDynamicRateLimiter),
    )
    assert found == (True, True, True), found


async def test_the_router_fallback_check_holds_a_fallback_through_the_native_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    from litellm.proxy.auth.fallback_rate_limit import RouterFallbackRateLimitCheck
    from litellm.router_api import RoutingContext

    _borrowing(monkeypatch, 0.0)
    capped: Final = _capped_team("capped", "team-f", {"spare": 40})
    router: Final = _router(("busy", 1000, None), ("spare", 1000, None))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    side: Final = NativeSide(router, isolated)
    monkeypatch.setattr(litellm, "callbacks", [side.logger])
    proxy_limiter: Final = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))
    check: Final = RouterFallbackRateLimitCheck(limiter=lambda: proxy_limiter)
    native: Final[list[object]] = []  # mutable-ok: the native's transcript, in order
    for index in range(4):
        request = Request(capped, "busy", f"f{index}", messages=({"role": "user", "content": "word " * 20},))
        native.append(await check(model="spare", request_kwargs=request.data(), ctx=RoutingContext(router)))
    native.append(
        await _counters(
            side,
            [
                ("model_saturation_check", "spare"),
                ("model_per_team_saturated", "team-f:spare"),
                ("model_per_team_max_borrow_multiple", "team-f:spare"),
            ],
        )
    )
    golden.check(native)
    assert False in native[:4] and True in native[:4], native


_TEAM_SCOPES: Final = (
    ("model_per_team_saturated", "team-w:busy"),
    ("model_per_team_max_borrow_multiple", "team-w:busy"),
    ("model_per_team_saturated", "team-w:spare"),
    ("model_per_team_max_borrow_multiple", "team-w:spare"),
)


def _team_tokens(transcript: Sequence[object]) -> tuple[object, ...]:
    counters: Final = transcript[-1]
    assert isinstance(counters, list)
    return tuple(entry[3] for entry in counters if isinstance(entry, tuple) and entry[2] == "tokens")


async def test_a_model_the_walk_leaves_gives_back_its_capped_team_hold_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "enforce_fallback_model_access", False)
    _borrowing(monkeypatch, 0.5)
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    walk: Final = _Walk(
        key=_capped_team("capped", "team-w", {"busy": 400, "spare": 400}),
        groups=(("busy", 1000, 4), ("spare", 1000, 4)),
        refuse=MappingProxyType({"busy": ProxyRateLimitError(detail="a later hook's limit", headers={})}),
        scopes=_TEAM_SCOPES,
        opted_in=False,
    )
    native: Final = await _saturation_walk(monkeypatch, isolated, walk)
    golden.check(native)
    assert native[2:5] == ["busy", "spare", ("spare", ("busy", "unknown", 1))], native
    assert _team_tokens(native) == (0.0, 0.0, None, 150), native


async def test_a_fallback_refused_after_admission_gives_back_its_hold_and_the_replay_holds_once_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "enforce_fallback_model_access", False)
    _borrowing(monkeypatch, 0.5)
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    walk: Final = _Walk(
        key=_capped_team("capped", "team-w", {"busy": 400, "spare": 400}),
        groups=(("busy", 1000, 4), ("spare", 1000, 4)),
        refuse=MappingProxyType({"spare": HTTPException(status_code=400, detail="a later hook refused the fallback")}),
        scopes=_TEAM_SCOPES,
    )
    native: Final = await _saturation_walk(monkeypatch, isolated, walk)
    golden.check(native)
    handoff: Final = native[2]
    assert isinstance(handoff, tuple)
    assert handoff[-1] == "model_saturation_fallback", native
    assert native[3:6] == ["spare", "busy", ("busy", None)], native
    assert _team_tokens(native) == (150, 150, None, 0.0), native


async def test_a_released_hold_is_not_refunded_again_by_the_failure_line_like_python(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    from litellm.proxy.common_request_processing import (
        _release_pre_call_hold,  # pyright: ignore[reportPrivateUsage]  # the release under test
    )

    _borrowing(monkeypatch, 0.0)
    for name in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    team: Final = _capped_team("capped", "team-w", {"busy": 400})
    scopes: Final = _TEAM_SCOPES[:2]
    router: Final = _router(("busy", 1000, 4))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    side: Final = NativeSide(router, isolated)
    monkeypatch.setattr(litellm, "callbacks", [side.logger])
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())  # pyright: ignore[reportArgumentType]  # the proxy builds it from a DualCache too
    native: Final = await asyncio.create_task(
        _release_twice(side, proxy_logging, team, scopes, _release_pre_call_hold), context=contextvars.Context()
    )
    golden.check(native)
    assert [[entry[3] for entry in counters if entry[2] == "tokens"] for counters in native] == [  # pyright: ignore[reportIndexIssue, reportUnknownVariableType, reportAttributeAccessIssue]  # counter rows
        [150, 150],
        [0, 0],
        [0, 0],
        [0, 0],
    ], native


async def _release_twice(
    side: Side,
    proxy_logging: ProxyLogging,
    team: UserAPIKeyAuth,
    scopes: Sequence[tuple[str, str]],
    release: Callable[..., Awaitable[None]],
) -> list[object]:  # mutable-ok: the side's counters after each step, in order
    get_or_create_request_stash()
    data: Final = Request(team, "busy", "x0", messages=({"role": "user", "content": "word " * 40},)).data()
    live_auth: Final = _auth(data)
    refusal: Final = ProxyRateLimitError(detail="a later hook's limit", headers={})
    admitted: Final = await proxy_logging.pre_call_hook(live_auth, data, "acompletion")
    seen: Final[list[object]] = [await _counters(side, scopes)]  # mutable-ok: counters after each step, in order
    await release(proxy_logging, admitted, refusal, live_auth)
    seen.append(await _counters(side, scopes))
    await release(proxy_logging, admitted, refusal, live_auth)
    seen.append(await _counters(side, scopes))
    await proxy_logging.post_call_failure_hook(
        request_data=dict(admitted), original_exception=refusal, user_api_key_dict=live_auth
    )
    seen.append(await _counters(side, scopes))
    return seen


async def test_the_hold_release_reaches_the_dynamic_limiter_once(
    monkeypatch: pytest.MonkeyPatch, isolated: Clock
) -> None:
    from litellm.proxy.common_request_processing import (
        _release_pre_call_hold,  # pyright: ignore[reportPrivateUsage]  # the release under test
    )

    router: Final = _router(("busy", 1000, 4))
    limiter: Final = NativeSide(router, isolated).logger
    calls: Final[list[str]] = []  # mutable-ok: the limiters the release reached, in order

    async def released_by(
        *_args: object,
        **_kwargs: object,  # kwargs-ok: the hook's own keywords
    ) -> None:
        calls.append("native")

    monkeypatch.setattr(limiter, "async_post_call_failure_hook", released_by)
    for name in ("success_callback", "failure_callback", "_async_success_callback"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [limiter, limiter])
    monkeypatch.setattr(litellm, "_async_failure_callback", [limiter, limiter])
    data: Final = Request(_key("plain"), "busy", "x0").data()
    await _release_pre_call_hold(
        ProxyLogging(user_api_key_cache=DualCache()),  # pyright: ignore[reportArgumentType]  # the proxy builds it from a DualCache too
        data,
        ProxyRateLimitError(detail="a later hook's limit", headers={}),
        _auth(data),
    )
    assert calls == ["native"], calls
