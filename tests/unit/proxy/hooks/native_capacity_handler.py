import asyncio
import socket
import threading
import time
import weakref
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from litellm import DualCache, Router
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_capacity_native import (
    NativeDynamicRateLimiter,
    build_native_dynamic_rate_limiter,
    shared_rate_limit_store,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    BorrowingTeamModelLimits,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitedModel,
    get_request_stash,
    model_limit_borrowers,
    rate_limited_model,
)
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import HookRun, InternalUsageCache, hook_fields, raise_v2_abort
from litellm.proxy.v2_proxy_fields import (
    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
    DERIVED_POST_CALL_SUCCESS_FIELDS,
    DERIVED_RESPONSE_HEADERS_FIELDS,
    PRE_CALL_SOURCES,
)
from litellm.redis.stores.rate_limits import RateLimitCounter, RateLimitScope, RateLimitStore
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome
from litellm.types.router import ModelGroupInfo


@dataclass(frozen=True, slots=True)
class TeamShare:
    share: RateLimitDescriptor
    ceiling: RateLimitDescriptor | None


def _team_descriptor(
    limiter: PROXY_MaxParallelRequestsHandler_v3, key: str, value: str, limits: BorrowingTeamModelLimits
) -> RateLimitDescriptor:
    return RateLimitDescriptor(
        key=key,
        value=value,
        rate_limit={
            "tokens_per_unit": limits.tpm,
            "requests_per_unit": limits.rpm,
            "window_size": limiter.window_size,
        },
    )


def borrowing_shares(
    handler: "NativeDynamicRateLimitHandler", model: str, auth: UserAPIKeyAuth
) -> dict[str, TeamShare]:
    limiter: Final = handler.v3_limiter
    rate_limited: Final = rate_limited_model(limiter, model) or RateLimitedModel(requested=model, group=model)
    shares: Final[dict[str, TeamShare]] = {}  # mutable-ok: built once per call
    for borrower in model_limit_borrowers(auth, rate_limited):
        value = f"{borrower.entity_id}:{rate_limited.group}"
        ceiling = borrower.limits.borrow_ceiling()
        shares[borrower.kind] = TeamShare(
            share=_team_descriptor(limiter, f"model_per_{borrower.kind}_saturated", value, borrower.limits),
            ceiling=None
            if ceiling is None
            else _team_descriptor(limiter, f"model_per_{borrower.kind}_max_borrow_multiple", value, ceiling),
        )
    return shares


def borrowing_team_share(
    handler: "NativeDynamicRateLimitHandler", model: str, auth: UserAPIKeyAuth
) -> TeamShare | None:
    return borrowing_shares(handler, model, auth).get("team")


def model_tracking_descriptor(
    handler: "NativeDynamicRateLimitHandler", model: str, model_group_info: ModelGroupInfo
) -> RateLimitDescriptor:
    return RateLimitDescriptor(
        key="model_saturation_check",
        value=model,
        rate_limit={
            "requests_per_unit": model_group_info.rpm or None,
            "tokens_per_unit": model_group_info.tpm or None,
            "window_size": handler.v3_limiter.window_size,
        },
    )


async def model_saturation(
    handler: "NativeDynamicRateLimitHandler", model: str, model_group_info: ModelGroupInfo
) -> float:
    rate_limits: Final = handler.rate_limits
    scope: Final = RateLimitScope("model_saturation_check", model)
    requests: Final = await rate_limits.get_from_redis(scope.counter("requests"))
    tokens: Final = await rate_limits.get_from_redis(scope.counter("tokens"))
    return max(
        0.0,
        0.0 if not model_group_info.rpm or requests is None else int(str(requests)) / model_group_info.rpm,
        0.0 if not model_group_info.tpm or tokens is None else float(str(tokens)) / model_group_info.tpm,
    )


def seed_counter(handler: "NativeDynamicRateLimitHandler", counter: RateLimitCounter, value: int) -> None:
    handler.saturation_redis()
    rate_limits: Final = handler.rate_limits
    if rate_limits.local((counter.window,))[0] is None:
        rate_limits.set_local(counter.window, str(int(handler.now())), handler.window_size)
    rate_limits.set_local(counter, value, handler.window_size)


def saturation_reads(handler: "NativeDynamicRateLimitHandler", model: str, saturation: float) -> dict[str, str | None]:
    group: Final = handler.llm_router.registry().group(model)
    assert group is not None
    scope: Final = RateLimitScope("model_saturation_check", model)
    return {
        counter_key(scope.counter("tokens")): str(round(saturation * group.tpm)) if group.tpm else None,
        counter_key(scope.counter("requests")): None
        if group.tpm or not group.rpm
        else str(round(saturation * group.rpm)),
    }


async def pre_call_at_saturation(
    handler: "NativeDynamicRateLimitHandler",
    user_api_key_dict: UserAPIKeyAuth,
    model: str,
    saturation: float,
    data: dict[str, object] | None = None,  # mutable-ok: the pre-call hook rewrites the request dict in place
) -> None:
    redis: Final = handler.saturation_redis()
    request: Final[dict[str, object]] = data if data is not None else {"max_tokens": 0}
    requested: Final = request.get("model")
    request["model"] = model
    try:
        with redis.answering(saturation_reads(handler, model, saturation)):
            await handler.async_pre_call_hook(user_api_key_dict, DualCache(), request, "acompletion")
    finally:
        if requested is None:
            del request["model"]
        else:
            request["model"] = requested


def _published_priority_descriptors(window_size: int) -> list[RateLimitDescriptor]:
    stash: Final = get_request_stash()
    response: Final = None if stash is None else stash.rate_limit_response
    statuses: Final = () if response is None else response["statuses"]
    values: Final = tuple(
        dict.fromkeys(
            status.get("descriptor_value") for status in statuses if status["descriptor_key"] == "priority_model"
        )
    )
    return [
        RateLimitDescriptor(
            key="priority_model",
            value=str(value),
            rate_limit={
                "requests_per_unit": next(
                    (
                        status["current_limit"]
                        for status in statuses
                        if status.get("descriptor_value") == value and status["rate_limit_type"] == "requests"
                    ),
                    None,
                ),
                "tokens_per_unit": next(
                    (
                        status["current_limit"]
                        for status in statuses
                        if status.get("descriptor_value") == value and status["rate_limit_type"] == "tokens"
                    ),
                    None,
                ),
                "window_size": window_size,
            },
        )
        for value in values
    ]


async def priority_based_descriptors(
    handler: "NativeDynamicRateLimitHandler", model: str, user_api_key_dict: UserAPIKeyAuth, priority: str | None
) -> list[RateLimitDescriptor]:
    assert (user_api_key_dict.metadata or {}).get("priority") == priority
    await handler.async_pre_call_hook(user_api_key_dict, DualCache(), {"model": model, "max_tokens": 0}, "acompletion")
    return _published_priority_descriptors(handler.window_size)


async def _no_legacy(_logger: object, _data: object) -> None:
    return None


def _call(data: Mapping[str, object], call_type: str) -> CallFacts:
    return CallFacts(
        ids=CallIds(call_id=str(data.get("litellm_call_id") or "")),
        kind=CallKind(call_type=call_type),
        target=CallTarget(model=str(data.get("model"))),
        timing=CallTiming(start=1.0),
    )


async def startup_pings() -> None:
    await asyncio.gather(
        *(task for task in asyncio.all_tasks() if task.get_coro().__qualname__ == RedisCache.ping.__qualname__)
    )


class NativeDynamicRateLimitHandler:
    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        build: Callable[..., NativeDynamicRateLimiter] = build_native_dynamic_rate_limiter,
    ) -> None:
        self._internal_usage_cache: Final = internal_usage_cache
        self._time_provider: Final = time_provider
        self._build: Final = build
        self.logger: NativeDynamicRateLimiter = build(internal_usage_cache, time_provider=time_provider)
        self.callbacks: list[object] = [self.logger]
        self._limiter: PROXY_MaxParallelRequestsHandler_v3 = self._reader()
        self._redis: FakeRedis | None = None
        self._touched = False

    def saturation_redis(self) -> "FakeRedis":
        if self._redis is not None:
            return self._redis
        assert not self._touched, "the saturation reads need the fake Redis before the handler counts anything"
        servers: Final = ExitStack()
        weakref.finalize(self, servers.close)
        redis: Final = servers.enter_context(fake_redis())
        self._rebuild(shared_rate_limit_store(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=redis.port))))
        self._redis = redis  # rebind-ok: the handler counts on this fake Redis from now on
        return redis

    def _reader(self) -> PROXY_MaxParallelRequestsHandler_v3:
        return PROXY_MaxParallelRequestsHandler_v3(
            InternalUsageCache(dual_cache=self._internal_usage_cache),
            time_provider=self._time_provider,
            rate_limit_store=self.logger.rate_limits,
        )

    def _rebuild(
        self,
        rate_limits: RateLimitStore,
        media_model_resolver: Callable[[str, str | None], tuple[str, ...]] | None = None,
    ) -> None:
        router: Final = self.logger.delegate.llm_router
        active_teams: Final = self.logger.delegate.active_teams
        self.logger = self._build(  # rebind-ok: the engine is built on the store the test hands in
            self._internal_usage_cache,
            time_provider=self._time_provider,
            rate_limits=rate_limits,
            media_model_resolver=media_model_resolver,
        )
        self.callbacks = [self.logger]  # rebind-ok: the rebuilt logger takes the list position
        self._limiter = self._reader()  # rebind-ok: the test's reader follows the rebuilt store
        vars(self.logger.delegate)["active_teams"] = active_teams
        if router is not None:
            self.logger.update_variables(router)

    @property
    def rate_limits(self) -> RateLimitStore:
        self._touched = True  # rebind-ok: a test reading or seeding the store keeps it
        return self.logger.rate_limits

    @property
    def window_size(self) -> int:
        return self.logger.window_size

    def now(self) -> float:
        return (self._time_provider or datetime.now)().timestamp()

    @property
    def v3_limiter(self) -> PROXY_MaxParallelRequestsHandler_v3:
        return self._limiter

    @property
    def internal_usage_cache(self) -> InternalUsageCache:
        return self._limiter.internal_usage_cache

    @property
    def active_teams(self) -> ActiveTeamRegistry:
        return self.logger.delegate.active_teams

    @property
    def llm_router(self) -> Router:
        router: Final = self.logger.delegate.llm_router
        assert router is not None
        return router

    @v3_limiter.setter
    def v3_limiter(self, limiter: PROXY_MaxParallelRequestsHandler_v3) -> None:
        self._touched = True  # rebind-ok: the test's own store stays
        same_cache: Final = limiter.internal_usage_cache is self._limiter.internal_usage_cache
        if self._redis is not None and not same_cache:
            self._redis = None  # rebind-ok: a limiter on another cache takes its own Redis
        self._rebuild(
            limiter.rate_limits if self._redis is None else self.logger.rate_limits,
            limiter.media_model_resolver,
        )
        self._limiter = limiter  # rebind-ok: the test reads through the limiter it handed in

    @active_teams.setter
    def active_teams(self, registry: ActiveTeamRegistry) -> None:
        vars(self.logger.delegate)["active_teams"] = registry

    def update_variables(self, llm_router: Router) -> None:
        self.logger.update_variables(llm_router)

    def _borrowing_shares(self, model: str, user_api_key_dict: UserAPIKeyAuth) -> dict[str, TeamShare]:
        return borrowing_shares(self, model, user_api_key_dict)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: the legacy pre-call hook rewrites the proxy's request dict
        call_type: str,
    ) -> None:
        self._touched = True  # rebind-ok: the handler has counted on its store
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            result: Final = await run_subscribers(
                hook_subscribers(Hook.pre_call, self.callbacks),
                with_sources(
                    KwargsFields(data, PROXY_REQUEST_KEYS),
                    PRE_CALL_SOURCES,
                    HookArgs(auth=user_api_key_dict, request=data, call_type=call_type),
                ),
                _no_legacy,
                call=_call(data, call_type),
            )
            after: Final = kwargs_after(raise_v2_abort(result))
        finally:
            REQUEST_AUTH.reset(auth_token)
        if isinstance(after, dict) and after is not data:
            data.clear()
            data.update(after)

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self._touched = True  # rebind-ok: the handler has counted on its store
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_success, self.callbacks),
                with_sources(
                    hook_fields(data, HookRun(response=response), Field.response),
                    DERIVED_POST_CALL_SUCCESS_FIELDS,
                    HookArgs(auth=user_api_key_dict, request=data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)
        return response

    async def async_post_call_response_headers_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, str] | None:  # mutable-ok: the legacy hook returns the headers it adds
        merged: Final[dict[str, str]] = {}
        await run_subscribers(
            hook_subscribers(Hook.response_headers, self.callbacks, admit=Admit.every_entry),
            with_sources(
                hook_fields(data, HookRun(response=merged), Field.response_headers),
                DERIVED_RESPONSE_HEADERS_FIELDS,
                HookArgs(auth=user_api_key_dict, request=data),
            ),
            _no_legacy,
        )
        return merged or None

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: the native runs on the proxy's request dict
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self._touched = True  # rebind-ok: the handler has counted on its store
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_failure, self.callbacks, admit=Admit.every_entry),
                with_sources(
                    hook_fields(request_data, HookRun(response=original_exception), Field.error),
                    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
                    HookArgs(auth=user_api_key_dict, request=request_data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)

    async def _attempt_ended(self, kwargs: dict[str, object], response_obj: object, failed: bool) -> None:
        self._touched = True  # rebind-ok: the handler has counted on its store
        call_type: Final = kwargs.get("call_type")
        await run_subscribers(
            hook_subscribers(Hook.attempt_ended, self.callbacks, admit=Admit.every_entry),
            with_sources(
                KwargsFields(kwargs, LOGGING_KEYS),
                DERIVED_ATTEMPT_FIELDS,
                HookArgs(
                    request=kwargs,
                    result=response_obj,
                    outcome=AttemptOutcome.failure if failed else AttemptOutcome.success,
                ),
            ),
            _no_legacy,
            call=_call(kwargs, call_type if isinstance(call_type, str) else "acompletion"),
        )

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._attempt_ended(kwargs, response_obj, failed=False)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._attempt_ended(kwargs, response_obj, failed=True)

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool:
        self._touched = True  # rebind-ok: the handler has counted on its store
        return await self.logger.async_fallback_target_allowed(user_api_key_dict, model, request_data, commit=commit)

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        self._touched = True  # rebind-ok: the handler has counted on its store
        await self.logger.enforce_token_limits_for_frame(
            user_api_key_dict=user_api_key_dict, requested_model=requested_model
        )


def _resp_command(buffer: bytes) -> tuple[tuple[bytes, ...], bytes] | None:
    lines: Final = buffer.split(b"\r\n", 1)
    if len(lines) < 2 or not lines[0].startswith(b"*"):
        return None
    rest = lines[1]
    parts: Final[list[bytes]] = []  # mutable-ok: the command's arguments, read in order
    for _ in range(int(lines[0][1:])):
        header = rest.split(b"\r\n", 1)
        if len(header) < 2:
            return None
        size = int(header[0][1:])
        if len(header[1]) < size + 2:
            return None
        parts.append(header[1][:size])
        rest = header[1][size + 2 :]
    return tuple(parts), rest


def _bulk(text: str | None) -> bytes:
    if text is None:
        return b"$-1\r\n"
    return b"$" + str(len(text.encode())).encode() + b"\r\n" + text.encode() + b"\r\n"


def _integer(number: int) -> bytes:
    return b":" + str(number).encode() + b"\r\n"


def _array(items: Sequence[bytes]) -> bytes:
    return b"*" + str(len(items)).encode() + b"\r\n" + b"".join(items)


def _float_text(number: float) -> str:
    return f"{number:.17g}"


def counter_key(counter: RateLimitCounter) -> str:
    return f"{{{counter.scope.descriptor_key}:{counter.scope.descriptor_value}}}:{counter.kind}"


@dataclass
class FakeRedis:
    port: int
    error: str
    emulate: bool = False
    read_error: str | None = None
    reads: dict[str, str | None] = field(default_factory=dict)  # mutable-ok: what the saturation read answers now
    values: dict[str, str] = field(default_factory=dict)  # mutable-ok: the keyspace the emulated scripts keep
    expiring: set[str] = field(default_factory=set)  # mutable-ok: keys an EXPIRE gave a TTL
    refused_reads: set[str] = field(default_factory=set)  # mutable-ok: counters whose reads this Redis refuses
    commands: list[tuple[bytes, ...]] = field(default_factory=list)  # mutable-ok: every command, in arrival order
    trips: int = 0

    @contextmanager
    def answering(self, reads: Mapping[str, str | None]) -> Generator[None, None, None]:
        self.reads.update(reads)
        try:
            yield
        finally:
            self.reads.clear()

    def increments(self) -> list[tuple[str, float]]:
        return [
            (command[1].decode(), float(command[2]))
            for command in self.commands
            if command[0].upper() in (b"INCRBYFLOAT", b"INCRBY")
        ]

    def atomic_descriptor_keys(self) -> list[list[str]]:
        return [
            [key.decode().split(":")[0].removeprefix("{") for key in keys[::2]]
            for keys in (
                command[3 : 3 + int(command[2])]
                for command in self.commands
                if command[0].upper() in (b"EVALSHA", b"EVAL")
                and int(command[2]) > 0
                and len(command) == 3 + 3 * int(command[2])
            )
            if all(key.endswith(b":window") for key in keys[::2])
        ]

    def read_keys(self) -> list[str]:
        return [
            command[4].decode()
            for command in self.commands
            if command[0].upper() in (b"EVALSHA", b"EVAL") and len(command) == 3 + int(command[2])
        ]

    def _refused(self, error: str | None = None) -> bytes:
        return b"-ERR " + (error or self.error).encode() + b"\r\n"

    def _number(self, key: str) -> float:
        return float(self.values.get(key, "0"))

    def _incrby(self, key: str, amount: float) -> float:
        value: Final = self._number(key) + amount
        self.values[key] = _float_text(value)
        return value

    def _expire(self, key: str) -> None:
        self.expiring.add(key)

    def _reset(self, window_key: str) -> None:
        prefix: Final = window_key.removesuffix(":window")
        self.values.pop(prefix + ":requests", None)
        self.values.pop(prefix + ":tokens", None)

    def _read(self, keys: Sequence[str]) -> bytes:
        if self.read_error is not None:
            return self._refused(self.read_error)
        if self.refused_reads.intersection(keys):
            return self._refused()
        unanswered: Final = [
            key for key in keys if key not in self.reads and not key.endswith(":window") and not self.emulate
        ]
        if unanswered:
            return self._refused()
        return _array([_bulk(self.reads[key] if key in self.reads else self.values.get(key)) for key in keys])

    def _check_and_increment(self, keys: Sequence[str], argv: Sequence[float]) -> bytes:
        now: Final = int(time.time())
        states: list[tuple[bool, float, str | None]] = []  # mutable-ok: each descriptor's state, read before any write
        for index in range(len(keys) // 2):
            limit, increment, _ttl, window_size = argv[index * 4 : index * 4 + 4]
            start = self.values.get(keys[index * 2])
            expired = start is None or now - int(float(start)) >= window_size
            current = 0.0 if expired else self._number(keys[index * 2 + 1])
            if (current + increment > limit) if increment > 0 else (current >= limit):
                return _array([_integer(1), _integer(index + 1), _integer(int(current)), _integer(int(limit))])
            states.append((expired, current, start))
        replies: list[bytes] = [_integer(0)]  # mutable-ok: the success reply, built in descriptor order
        reset: set[str] = set()  # mutable-ok: windows this call already reset
        for index, (expired, _current, start) in enumerate(states):
            window_key, counter_key_ = keys[index * 2], keys[index * 2 + 1]
            increment = argv[index * 4 + 1]
            if expired:
                if window_key not in reset:
                    self._reset(window_key)
                    reset.add(window_key)
                self.values[window_key] = str(now)
                self.values[counter_key_] = _float_text(increment)
                replies.extend([_integer(int(increment)), _integer(now)])
            else:
                replies.extend([_integer(int(self._incrby(counter_key_, increment))), _integer(int(float(start or 0)))])
        return _array(replies)

    def _sliding_window(self, keys: Sequence[str], now: int, window_size: float) -> bytes:
        replies: list[bytes] = []  # mutable-ok: window start then count per counter
        for index in range(0, len(keys), 2):
            window_key, counter_key_ = keys[index], keys[index + 1]
            start = self.values.get(window_key)
            if start is None or now - int(float(start)) >= window_size:
                self._reset(window_key)
                self.values[window_key] = str(now)
                self.values[counter_key_] = "1"
                replies.extend([_bulk(str(now)), _integer(1)])
            else:
                replies.extend([_bulk(start), _integer(int(self._incrby(counter_key_, 1)))])
        return _array(replies)

    def _guarded(self, keys: Sequence[str], argv: Sequence[bytes]) -> bytes:
        replies: list[bytes] = []  # mutable-ok: applied flag then count per counter
        for index in range(0, len(keys), 2):
            expected, increment = argv[index // 2 * 3].decode(), float(argv[index // 2 * 3 + 1])
            if self.values.get(keys[index]) == expected:
                replies.extend([_integer(1), _integer(int(self._incrby(keys[index + 1], increment)))])
            else:
                replies.extend([_integer(0), _integer(int(self._number(keys[index + 1])))])
        return _array(replies)

    def _token_increment(self, keys: Sequence[str], argv: Sequence[bytes]) -> bytes:
        return _array([_bulk(_float_text(self._incrby(key, float(argv[index * 2])))) for index, key in enumerate(keys)])

    def _script(self, command: tuple[bytes, ...]) -> bytes:
        count: Final = int(command[2])
        keys: Final = [key.decode() for key in command[3 : 3 + count]]
        argv: Final = command[3 + count :]
        if not argv:
            return self._read(keys)
        if not self.emulate:
            return self._refused()
        paired: Final = count > 0 and all(key.endswith(":window") for key in keys[::2])
        if paired and len(argv) == 2 * count:
            return self._check_and_increment(keys, [float(value) for value in argv])
        if paired and len(argv) == 2:
            return self._sliding_window(keys, int(float(argv[0])), float(argv[1]))
        if paired and len(argv) * 2 == 3 * count:
            return self._guarded(keys, argv)
        if not paired and len(argv) == 2 * count:
            return self._token_increment(keys, argv)
        return self._refused()

    def _command(self, command: tuple[bytes, ...]) -> bytes:
        name: Final = command[0].upper()
        arguments: Final = [argument.decode() for argument in command[1:]]
        match name:
            case b"GET":
                return _bulk(self.values.get(arguments[0]))
            case b"MGET":
                return _array([_bulk(self.values.get(key)) for key in arguments])
            case b"SET":
                self.values[arguments[0]] = arguments[1]
                return b"+OK\r\n"
            case b"DEL":
                return _integer(sum(self.values.pop(key, None) is not None for key in arguments))
            case b"INCRBY" | b"INCR":
                return _integer(int(self._incrby(arguments[0], float(arguments[1]) if len(arguments) > 1 else 1)))
            case b"INCRBYFLOAT":
                return _bulk(_float_text(self._incrby(arguments[0], float(arguments[1]))))
            case b"TTL":
                return _integer(-2 if arguments[0] not in self.values else 60 if arguments[0] in self.expiring else -1)
            case b"EXPIRE":
                self._expire(arguments[0])
                return _integer(1)
            case _:
                return self._refused()

    def _reply(self, command: tuple[bytes, ...]) -> bytes:
        name: Final = command[0].upper()
        if name == b"PING":
            return b"+PONG\r\n"
        if name == b"CLIENT":
            return b"+OK\r\n"
        if name in (b"EVALSHA", b"EVAL"):
            return self._script(command)
        if not self.emulate:
            return self._refused()
        return self._command(command)

    def answer(self, connection: socket.socket) -> None:
        buffer = b""
        queued: list[tuple[bytes, ...]] | None = None  # mutable-ok: the commands a MULTI holds until EXEC
        with connection:
            while chunk := connection.recv(65536):
                if not buffer:
                    self.trips += 1
                buffer += chunk
                replies: list[bytes] = []  # mutable-ok: the replies to every whole command received so far
                while (parsed := _resp_command(buffer)) is not None:
                    command, buffer = parsed
                    self.commands.append(command)
                    name = command[0].upper()
                    if name == b"MULTI":
                        queued = []
                        replies.append(b"+OK\r\n")
                    elif name == b"EXEC" and queued is not None:
                        replies.append(_array([self._reply(held) for held in queued]))
                        queued = None
                    elif queued is not None:
                        queued.append(command)
                        replies.append(b"+QUEUED\r\n")
                    else:
                        replies.append(self._reply(command))
                connection.sendall(b"".join(replies))


@contextmanager
def listening() -> Generator[tuple[socket.socket, int], None, None]:
    listener: Final = socket.create_server(("127.0.0.1", 0))
    try:
        yield listener, listener.getsockname()[1]
    finally:
        listener.close()


@contextmanager
def fake_redis(
    error: str = "the fake Redis answers only saturation reads", *, emulate: bool = False, read_error: str | None = None
) -> Generator[FakeRedis, None, None]:
    servers: Final = ExitStack()
    listener, port = servers.enter_context(listening())
    stopped: Final = threading.Event()
    redis: Final = FakeRedis(port=port, error=error, emulate=emulate, read_error=read_error)

    def serve() -> None:
        while not stopped.is_set():
            try:
                connection = listener.accept()[0]
            except OSError:
                return
            threading.Thread(target=redis.answer, args=(connection,), daemon=True).start()

    server: Final = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        yield redis
    finally:
        stopped.set()
        servers.close()
        server.join(timeout=5)


@contextmanager
def redis_answering_counts(counts: Mapping[str, str | None]) -> Generator[int, None, None]:
    with fake_redis() as redis, redis.answering(counts):
        yield redis.port
