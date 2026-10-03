from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from string import ascii_letters, digits
from threading import Lock
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, TypeAdapter
from redis.crc import key_slot

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.internal_call_metadata import EvaluationBillingOwner
from litellm.proxy._types import Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    model_budget_spend_cache_key,
    model_budget_start_time_cache_key,
    resolve_model_budget,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_input_cost,
    estimate_request_max_cost,
    reconcile_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # legacy reservation entries are untyped
    reserve_budget_for_request,
)
from litellm.router_utils.common_utils import resolve_model_group_alias
from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes

_NUMBER: Final = TypeAdapter(float)
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_REQUEST: Final = TypeAdapter(dict[str, object])
_HOLDS: Final = TypeAdapter(Mapping[str, float])
_LEASE_SECONDS: Final = 60
_LOCAL_LOCK: Final = Lock()
_LEASES: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: asyncio weakly references pending tasks
_HOLD_SCRIPT: Final = """
local actual = tonumber(redis.call('GET', KEYS[2]) or '0')
if not actual then return redis.error_reply('Invalid model budget spend') end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local total = 0
for _, member in ipairs(redis.call('ZRANGEBYSCORE', KEYS[1], '(' .. now, '+inf')) do
    if member ~= ARGV[2] then total = total + tonumber(string.match(member, ':([^:]+)$')) end
end
if ARGV[1] == 'renew' then
    local expiry = redis.call('ZSCORE', KEYS[1], ARGV[2])
    if not expiry or tonumber(expiry) <= now then return redis.error_reply('Evaluation reservation expired') end
end
if ARGV[1] == 'reserve' or ARGV[1] == 'renew' then
    total = total + tonumber(string.match(ARGV[2], ':([^:]+)$'))
end
if ARGV[1] == 'reserve' and ARGV[6] ~= '' then
    local estimate = tonumber(string.match(ARGV[2], ':([^:]+)$'))
    if actual + total > tonumber(ARGV[6]) or actual + total - estimate >= tonumber(ARGV[6]) then
        return tostring(actual + total)
    end
end
if ARGV[1] == 'settle' and tonumber(ARGV[4]) > 0 then
    actual = tonumber(redis.call('INCRBYFLOAT', KEYS[2], ARGV[4]))
    redis.call('EXPIRE', KEYS[2], ARGV[5])
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if ARGV[1] == 'reserve' or ARGV[1] == 'renew' then
    redis.call('ZADD', KEYS[1], now + tonumber(ARGV[3]), ARGV[2])
    redis.call('EXPIRE', KEYS[1], ARGV[3])
elseif ARGV[1] == 'settle' then redis.call('ZREM', KEYS[1], ARGV[2]) end
return tostring(actual + total)
"""


@runtime_checkable
class _NumericCache(Protocol):
    def get_cache(self, key: str) -> object: ...
    def set_cache(self, key: str, value: object, *, ttl: int) -> object: ...
    def delete_cache(self, key: str) -> object: ...
    def increment_cache(self, key: str, value: float, *, ttl: int, refresh_ttl: bool = False) -> float: ...


@runtime_checkable
class _Script(Protocol):
    async def __call__(self, *, keys: Sequence[str], args: Sequence[str | int | float]) -> object: ...


_LOCAL: Final = TypeAdapter(_NumericCache, config=ConfigDict(arbitrary_types_allowed=True))
_SCRIPT: Final = TypeAdapter(_Script, config=ConfigDict(arbitrary_types_allowed=True))


@lru_cache(maxsize=4096)
def _model_hold_key(effective_key: str) -> str:
    prefix: Final = f"{effective_key}:evaluation_holds:"
    tagged: Final = f"{prefix}{{{effective_key}}}"
    slot: Final = key_slot(effective_key.encode())
    if key_slot(tagged.encode()) == slot:
        return tagged
    candidates: Final = (prefix + "".join(chars) for chars in product(ascii_letters + digits + "-_", repeat=3))
    return next(candidate for candidate in candidates if key_slot(candidate.encode()) == slot)


@with_service_target("model_budgets")
async def model_budget_spend(
    cache: DualCache,
    spend_key: str,
    *,
    operation: Literal["read", "reserve", "renew", "settle"] = "read",
    member: str = "",
    adjustment: float = 0.0,
    ttl: int = 1,
    limit: float | None = None,
) -> float:
    if cache.redis_cache is not None:
        key: Final = cache.redis_cache.check_and_fix_namespace(spend_key)
        script: Final = _SCRIPT.validate_python(cache.redis_cache.async_register_script(_HOLD_SCRIPT))
        return _NUMBER.validate_python(
            await script(
                keys=(_model_hold_key(key), key),
                args=(operation, member, _LEASE_SECONDS, adjustment, ttl, "" if limit is None else limit),
            )
        )
    local: Final = _LOCAL.validate_python(cache.in_memory_cache)
    hold_key: Final = _model_hold_key(spend_key)
    with _LOCAL_LOCK:
        actual: Final = _NUMBER.validate_python(local.get_cache(spend_key) or 0.0)
        now: Final = cache.in_memory_cache._clock()  # pyright: ignore[reportPrivateUsage]  # lease expiry uses the cache's injected clock
        active: Final = {
            token: expiry
            for token, expiry in _HOLDS.validate_python(local.get_cache(hold_key) or {}).items()
            if expiry > now
        }
        if operation == "renew" and member not in active:
            raise RuntimeError("Evaluation reservation expired")
        updated: Final = {
            **{token: expiry for token, expiry in active.items() if token != member},
            **({member: now + _LEASE_SECONDS} if operation in ("reserve", "renew") else {}),
        }
        held: Final = sum(float(token.rsplit(":", 1)[1]) for token in updated)
        proposed: Final = actual + held
        if (
            operation == "reserve"
            and limit is not None
            and (proposed > limit or proposed - float(member.rsplit(":", 1)[1]) >= limit)
        ):
            return proposed
        settled: Final = (
            local.increment_cache(spend_key, adjustment, ttl=ttl, refresh_ttl=True)
            if operation == "settle" and adjustment
            else actual
        )
        local.delete_cache(hold_key)
        if updated:
            local.set_cache(hold_key, updated, ttl=_LEASE_SECONDS)
        return settled + held


async def _complete(operation: Awaitable[EvaluationAttempt | None]) -> EvaluationAttempt | None:
    task: Final = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        drained: Final = asyncio.gather(task, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        if not task.cancelled() and task.exception() is None and (attempt := task.result()) is not None:
            await attempt.settle(0.0)
        raise


@dataclass(frozen=True, slots=True)
class _ModelBudget:
    spend_key: str
    start_key: str
    duration: int
    limit: float


@with_service_target("model_budgets")
async def _model_window(cache: DualCache, model: _ModelBudget) -> int:
    if cache.redis_cache is not None:
        script: Final = _SCRIPT.validate_python(
            cache.redis_cache.async_register_script(
                "local clock = redis.call('TIME'); local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000; "
                "local start = tonumber(redis.call('GET', KEYS[1]) or now); "
                "if now - start >= tonumber(ARGV[1]) then start = now end; "
                "local ttl = math.max(1, math.ceil(tonumber(ARGV[1]) - (now - start))); "
                "redis.call('SET', KEYS[1], start, 'EX', ttl); return ttl"
            )
        )
        return math.ceil(_NUMBER.validate_python(await script(keys=(model.start_key,), args=(model.duration,))))
    local: Final = _LOCAL.validate_python(cache.in_memory_cache)
    now: Final = time.time()
    cached: Final = local.get_cache(model.start_key)
    previous: Final = _NUMBER.validate_python(now if cached is None else cached)
    start: Final = now if now - previous >= model.duration else previous
    ttl: Final = max(1, math.ceil(model.duration - (now - start)))
    local.delete_cache(model.start_key)
    local.set_cache(model.start_key, start, ttl=ttl)
    return ttl


@dataclass(slots=True)
class EvaluationAttempt:
    cache: DualCache
    model: _ModelBudget | None
    total: dict[str, object] | None = None  # mutable-ok: existing spend writer owns this reservation format
    member: str = ""
    input_cost: float = 0.0
    known_cost: float = 0.0
    model_applied_cost: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def settle(self, cost: float, cancelled: bool = False) -> None:
        await _complete(self._settle(max(cost, self.input_cost if cancelled else 0.0)))

    async def _settle(self, cost: float) -> None:
        async with self.lock:
            self.known_cost = max(cost, self.known_cost)
            try:
                await reconcile_budget_reservation(
                    {**self.total, "finalized": False, "externally_settled": False} if self.total is not None else None,
                    self.known_cost,
                    finalize=False,
                )
                if self.total is not None:
                    self.total["externally_settled"] = True
            finally:
                if self.model is not None:
                    delta: Final = max(self.known_cost - (self.model_applied_cost or 0.0), 0.0)
                    ttl: Final = await _model_window(self.cache, self.model) if delta else 1
                    await model_budget_spend(
                        self.cache,
                        self.model.spend_key,
                        operation="settle",
                        member=self.member,
                        adjustment=delta,
                        ttl=ttl,
                    )
                    self.model_applied_cost = self.known_cost

    async def renew(self, request_task: asyncio.Task[object] | None) -> None:
        deadline: Final = time.monotonic() + 2 * litellm.request_timeout
        while self.model_applied_cost is None and self.model is not None:
            await asyncio.sleep(_LEASE_SECONDS / 2)
            async with self.lock:
                if self.model_applied_cost is not None:
                    return
                try:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Evaluation reservation exceeded request timeout")
                    await model_budget_spend(self.cache, self.model.spend_key, operation="renew", member=self.member)
                except Exception:  # noqa: BLE001  # an active call must stop when its budget hold cannot be renewed
                    verbose_proxy_logger.exception("Unable to renew evaluation budget reservation")
                    if request_task is not None and not request_task.done():
                        request_task.cancel()
                    return


async def reserve_evaluation_budget(
    owner: EvaluationBillingOwner,
    request: Mapping[str, object],
    call_type: str,
) -> EvaluationAttempt | None:
    return await _complete(_reserve(owner, request, call_type, asyncio.current_task()))


async def _reserve(
    owner: EvaluationBillingOwner,
    request: Mapping[str, object],
    call_type: str,
    request_task: asyncio.Task[object] | None,
) -> EvaluationAttempt | None:
    from litellm.proxy import proxy_server as proxy

    llm_router: Final = proxy.llm_router
    metadata: Final = _MAPPING.validate_python(request.get("litellm_metadata") or request.get("metadata") or {})
    logical_model: Final = str(metadata.get("model_group") or request["model"])
    resolved: Final = resolve_model_budget(logical_model, owner.user_model_max_budget or {})
    model: Final = (
        _ModelBudget(
            model_budget_spend_cache_key(
                Litellm_EntityType.USER, owner.user_id, resolved.budget_model, resolved.budget_config.budget_duration
            ),
            model_budget_start_time_cache_key(
                Litellm_EntityType.USER, owner.user_id, resolved.budget_model, resolved.budget_config.budget_duration
            ),
            duration_in_seconds(str(resolved.budget_config.budget_duration)),
            resolved.budget_config.max_budget,
        )
        if resolved is not None
        and resolved.budget_config.max_budget is not None
        and math.isfinite(resolved.budget_config.max_budget)
        and resolved.budget_config.max_budget >= 0
        else None
    )
    total_budget: Final = owner.max_budget is not None and math.isfinite(owner.max_budget)
    if not total_budget and model is None:
        return None
    if total_budget:
        current: Final = await proxy.get_current_spend(
            counter_key=f"spend:user:{owner.user_id}", fallback_spend=owner.spend, max_budget=owner.max_budget
        )
        if current >= _NUMBER.validate_python(owner.max_budget):
            raise litellm.BudgetExceededError(
                current_cost=current,
                max_budget=_NUMBER.validate_python(owner.max_budget),
                entity_type=Litellm_EntityType.USER.value,
                entity_id=owner.user_id,
            )
    route: Final = next(route for route, types in API_ROUTE_TO_CALL_TYPES.items() if CallTypes(call_type) in types)
    model_info: Final = _MAPPING.validate_python(request.get("model_info") or metadata.get("model_info") or {})
    deployment_id: Final = model_info.get("id")
    deployment: Final = (
        llm_router.get_deployment(deployment_id) if llm_router is not None and isinstance(deployment_id, str) else None
    )
    pricing_model: Final = (
        deployment.model_name
        if deployment is not None
        else (
            resolve_model_group_alias(llm_router.model_group_alias, logical_model) or logical_model
            if llm_router is not None
            else logical_model
        )
    )
    body: Final = _REQUEST.validate_python(
        {**request, "model": pricing_model, "metadata": {}, "litellm_metadata": {}, "tags": []}
    )
    attempt: Final = EvaluationAttempt(cache=proxy.model_max_budget_limiter.dual_cache, model=model)
    try:
        attempt.total = await reserve_budget_for_request(
            request_body=body,
            route=route,
            llm_router=llm_router,
            valid_token=UserAPIKeyAuth(user_id=owner.user_id),
            team_object=None,
            user_object=LiteLLM_UserTable(user_id=owner.user_id, max_budget=owner.max_budget, spend=owner.spend),
            prisma_client=proxy.prisma_client,
            user_api_key_cache=proxy.user_api_key_cache,
            proxy_logging_obj=proxy.proxy_logging_obj,
            fail_closed_budget_enforcement=True,
            request_task=request_task,
        )
        estimate: Final = (
            _NUMBER.validate_python(attempt.total["reserved_cost"])
            if attempt.total is not None
            else estimate_request_max_cost(body, route, llm_router)
        )
        if estimate is None or not math.isfinite(estimate) or estimate < 0:
            raise ValueError("Evaluation budget cannot be checked for an unpriced model")
        attempt.member = f"{uuid.uuid4()}:{estimate}"
        attempt.input_cost = (
            _NUMBER.validate_python(attempt.total["input_cost"])
            if attempt.total is not None
            else estimate_request_input_cost(body, route, llm_router) or 0.0
        )
        if model is not None:
            spend: Final = await model_budget_spend(
                attempt.cache, model.spend_key, operation="reserve", member=attempt.member, limit=model.limit
            )
            if spend > model.limit or spend - estimate >= model.limit:
                raise litellm.BudgetExceededError(
                    current_cost=spend - estimate,
                    max_budget=model.limit,
                    entity_type=Litellm_EntityType.USER.value,
                    entity_id=owner.user_id,
                )
            lease: Final = asyncio.create_task(attempt.renew(request_task))
            _LEASES.add(lease)
            lease.add_done_callback(_LEASES.discard)
    except Exception:
        await attempt.settle(0.0)
        raise
    return attempt
