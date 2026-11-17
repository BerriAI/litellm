from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Final, Generic, TypeAlias, TypeVar
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import httpx
from pydantic import ConfigDict, StrictStr, TypeAdapter

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.total_deadline import current_total_deadline

FALLBACK_PLAN_KWARG: Final = "_vector_store_fallback_plan"
SERVED_BY_HEADER: Final = "x-litellm-vector-store-id"
_FALLBACK_IDS: Final = TypeAdapter(list[StrictStr], config=ConfigDict(strict=True, hide_input_in_errors=True))
_CACHED: Final = TypeAdapter(Mapping[str, object])
_Value = TypeVar("_Value")  # rebind-ok: a TypeVar cannot be Final


@dataclass(frozen=True, slots=True)
class VectorStoreTarget:
    store_id: str
    provider: str
    params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class VectorStoreFallbackPlan:
    targets: tuple[VectorStoreTarget, ...]
    physical_index: str | None = None
    cooldown_time: float = DEFAULT_COOLDOWN_TIME_SECONDS


_FALLBACK_CONTEXT: Final[ContextVar[VectorStoreFallbackPlan | None]] = ContextVar(
    "vector_store_fallback_plan", default=None
)


@contextmanager
def vector_store_fallback_context(plan: VectorStoreFallbackPlan | None) -> Generator[None]:
    token: Final = _FALLBACK_CONTEXT.set(plan)
    try:
        yield
    finally:
        _FALLBACK_CONTEXT.reset(token)


def current_fallback_plan(vector_store_id: str) -> VectorStoreFallbackPlan | None:
    plan: Final = _FALLBACK_CONTEXT.get()
    if plan is None or not plan.targets:
        return None
    return (
        plan
        if vector_store_id in (plan.targets[0].store_id, native_store_id(plan.targets[0]), plan.physical_index)
        else None
    )


def native_store_id(target: VectorStoreTarget) -> str:
    resource: Final = target.params.get("vector_store_id", target.store_id)
    if not isinstance(resource, str) or not resource:
        raise ValueError("A vector store physical resource ID must be a nonempty string")
    return resource


def fallback_ids(params: Mapping[str, object]) -> tuple[str, ...]:
    if "fallbacks" not in params:
        return ()
    ids: Final = tuple(_FALLBACK_IDS.validate_python(params["fallbacks"]))
    if any(not store_id.strip() for store_id in ids) or len(frozenset(ids)) != len(ids):
        raise ValueError("Vector store fallbacks must contain distinct nonempty registered store IDs")
    return ids


def _target(store: Mapping[str, object]) -> VectorStoreTarget:
    params: Final = as_str_mapping(store.get("litellm_params")) or MappingProxyType({})
    store_id: Final = store.get("vector_store_id", params.get("vector_store_id"))
    provider: Final = store.get("custom_llm_provider", params.get("custom_llm_provider"))
    if not isinstance(store_id, str) or not store_id or not isinstance(provider, str) or not provider:
        raise ValueError("Vector store fallback targets require a registered ID and provider")
    return VectorStoreTarget(
        store_id=store_id,
        provider=provider,
        params=MappingProxyType(
            {
                **{key: value for key, value in params.items() if key not in ("fallbacks", FALLBACK_PLAN_KWARG)},
                "vector_store_id": params.get("vector_store_id", store_id),
                "custom_llm_provider": provider,
                **(
                    {"litellm_credential_name": store["litellm_credential_name"]}
                    if store.get("litellm_credential_name") is not None
                    else {}
                ),
            }
        ),
    )


def targets_for_store(
    primary: Mapping[str, object], registry_stores: Iterable[Mapping[str, object]]
) -> tuple[VectorStoreTarget, ...]:
    first: Final = _target(primary)
    references: Final = fallback_ids(as_str_mapping(primary.get("litellm_params")) or {})
    stores: Final = tuple(registry_stores)
    if first.store_id in references:
        raise ValueError("A vector store cannot fall back to itself")
    return (first, *(_referenced_target(store_id, stores) for store_id in references))


def _referenced_target(store_id: str, stores: tuple[Mapping[str, object], ...]) -> VectorStoreTarget:
    matches: Final = tuple(store for store in stores if _target_id(store) == store_id)
    if len(matches) != 1:
        raise ValueError(f"Vector store fallback '{store_id}' must identify exactly one registered store")
    if fallback_ids(as_str_mapping(matches[0].get("litellm_params")) or {}):
        raise ValueError("Nested vector store fallback chains are not supported")
    return _target(matches[0])


def _target_id(store: Mapping[str, object]) -> object:
    return store.get("vector_store_id", (as_str_mapping(store.get("litellm_params")) or {}).get("vector_store_id"))


@dataclass(frozen=True, slots=True)
class Served(Generic[_Value]):
    target: VectorStoreTarget
    value: _Value
    failed_over: int = 0


@dataclass(frozen=True, slots=True)
class Failed(Generic[_Value]):
    target: VectorStoreTarget
    error: Exception | None
    retryable: bool
    value: _Value | None = None


Attempt: TypeAlias = Served[_Value] | Failed[_Value]
vector_store_fallback_cooldowns: Final = DualCache()
_TRANSPORT_ERRORS: Final = (
    litellm.Timeout,
    litellm.APIConnectionError,
    httpx.TransportError,
    aiohttp.ClientConnectionError,
    TimeoutError,
)


def classify_status(status: int) -> bool:
    return status in (408, 429) or 500 <= status <= 599


def _raised_from(error: BaseException | None, _: int) -> BaseException | None:
    return None if error is None else error.__cause__ or error.__context__


def classify_error(target: VectorStoreTarget, error: Exception) -> Failed[_Value]:
    chain: Final = accumulate(range(15), _raised_from, initial=error)
    transport: Final = any(isinstance(link, _TRANSPORT_ERRORS) for link in chain)
    status: Final[object] = getattr(error, "status_code", None)
    return Failed(target, error, classify_status(status) if isinstance(status, int) else transport)


def _cooldown_key(target: VectorStoreTarget) -> str:
    return f"vector_store_fallback_cooldown:{target.store_id}"


@with_service_target("vector_store_fallback_cooldowns")
async def _cooldown_ends(cache: DualCache, targets: tuple[VectorStoreTarget, ...]) -> tuple[Mapping[str, object], ...]:
    keys: Final = [_cooldown_key(target) for target in targets]
    local: Final[object] = await cache.in_memory_cache.async_batch_get_cache(keys=keys)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # legacy cache boundary
    local_ends: Final = MappingProxyType(dict(zip(keys, as_object_list(local) or (), strict=False)))  # pyright: ignore[reportUnknownArgumentType]  # legacy cache boundary
    redis: Final = cache.redis_cache
    if redis is None:
        return (local_ends,)
    try:
        return (local_ends, _CACHED.validate_python(await redis.async_batch_get_cache(key_list=keys)))  # pyright: ignore[reportUnknownMemberType]  # legacy cache boundary
    except Exception:  # noqa: BLE001  # cooldown Redis failures must retain local health rather than stop provider calls
        verbose_logger.warning("Vector store cooldown Redis read failed; using local health")
        return (local_ends,)


def _end(value: object) -> float:
    return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) else 0


@with_service_target("vector_store_fallback_cooldowns")
async def _record_failure(cache: DualCache, target: VectorStoreTarget, ttl: float, now: float) -> None:
    if ttl <= 0:
        return
    await cache.in_memory_cache.async_set_cache(_cooldown_key(target), now + ttl, ttl=math.ceil(ttl))  # pyright: ignore[reportUnknownMemberType]  # legacy cache boundary
    if cache.redis_cache is not None:
        try:
            await cache.redis_cache.async_set_cache(_cooldown_key(target), now + ttl, ttl=math.ceil(ttl))  # pyright: ignore[reportUnknownMemberType]  # legacy cache boundary
        except Exception:  # noqa: BLE001  # keep the successful local write if Redis is unavailable
            verbose_logger.warning("Vector store cooldown Redis write failed; using local health")


async def execute_with_vector_store_fallbacks(
    plan: VectorStoreFallbackPlan,
    attempt: Callable[[VectorStoreTarget], Awaitable[Attempt[_Value]]],
    *,
    cache: DualCache = vector_store_fallback_cooldowns,
    clock: Callable[[], float] = time.time,
) -> Attempt[_Value]:
    if not plan.targets:
        raise ValueError("A vector store fallback plan needs a primary")
    sources: Final = await _cooldown_ends(cache, plan.targets)
    now: Final = clock()
    eligible: Final = tuple(
        target for target in plan.targets if all(_end(source.get(_cooldown_key(target))) <= now for source in sources)
    )
    if not eligible:
        wait: Final = max(
            1,
            math.ceil(
                min(max(_end(source.get(_cooldown_key(target))) for source in sources) for target in plan.targets) - now
            ),
        )
        return Failed(
            plan.targets[0],
            litellm.ServiceUnavailableError(
                message="All configured vector stores are in cooldown",
                llm_provider=plan.targets[0].provider,
                model=plan.targets[0].store_id,
                response=httpx.Response(
                    503,
                    headers={"Retry-After": str(wait)},
                    request=httpx.Request("POST", "https://litellm.invalid/vector_stores/search"),
                ),
            ),
            False,
        )
    return await _next(plan, eligible, attempt, cache, clock, 0)


async def _next(
    plan: VectorStoreFallbackPlan,
    remaining: tuple[VectorStoreTarget, ...],
    attempt: Callable[[VectorStoreTarget], Awaitable[Attempt[_Value]]],
    cache: DualCache,
    clock: Callable[[], float],
    sent_fallbacks: int,
) -> Attempt[_Value]:
    deadline: Final = current_total_deadline()
    if deadline is not None and not deadline.allows_wait(0):
        return Failed(
            remaining[0],
            litellm.Timeout(
                message="Vector store request deadline exceeded", model=None, llm_provider=remaining[0].provider
            ),
            False,
        )
    sent: Final = sent_fallbacks + int(remaining[0].store_id != plan.targets[0].store_id)
    outcome: Final = await attempt(remaining[0])
    if isinstance(outcome, Served):
        return Served(outcome.target, outcome.value, sent)
    if outcome.retryable:
        await _record_failure(cache, outcome.target, plan.cooldown_time, clock())
    if not outcome.retryable or len(remaining) == 1:
        return outcome
    return await _next(plan, remaining[1:], attempt, cache, clock, sent)


def served_api_base(target: VectorStoreTarget) -> str:
    api_base: Final = target.params.get("api_base")
    if not isinstance(api_base, str):
        return ""
    parts: Final = urlsplit(api_base)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def served_headers(served: Served[_Value]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            SERVED_BY_HEADER: served.target.store_id,
            "x-litellm-attempted-fallbacks": str(served.failed_over),
            "x-litellm-model-api-base": served_api_base(served.target),
        }
    )
