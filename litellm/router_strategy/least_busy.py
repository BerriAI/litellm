import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import log_redis_failure
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type

IN_FLIGHT_COUNT_TTL_SECONDS: Final = 60 * 60


class _ModelInfo(TypedDict, total=False):
    id: ReadOnly[str | int | None]


class _Metadata(TypedDict, total=False):
    model_group: ReadOnly[str | None]


class _LitellmParams(TypedDict, total=False):
    metadata: ReadOnly[_Metadata | None]
    model_info: ReadOnly[_ModelInfo | None]


class _CallKwargs(TypedDict, total=False):
    litellm_call_id: ReadOnly[str | None]
    litellm_params: ReadOnly[_LitellmParams | None]


class _DeploymentModelInfo(TypedDict):
    id: ReadOnly[str | int]


class _Deployment(TypedDict):
    model_info: ReadOnly[_DeploymentModelInfo]


_CALL_KWARGS: Final = TypeAdapter(_CallKwargs)
_DEPLOYMENTS: Final = TypeAdapter(list[_Deployment])
_MEMORY_COUNTS: Final = TypeAdapter(tuple[float | None, ...] | None)


@dataclass(frozen=True, slots=True)
class _CallRef:
    model_group: str
    deployment_id: str
    call_id: str | None


def _request_count_key(model_group: str, deployment_id: str) -> str:
    return f"{model_group}_request_count:{deployment_id}"


def _in_flight_key(model_group: str, deployment_id: str) -> str:
    return f"{model_group}_in_flight:{deployment_id}"


def _call_ref(kwargs: Mapping[str, object]) -> _CallRef | None:
    if is_batch_retrieve_call_type(kwargs.get("call_type")):
        return None
    try:
        call: Final = _CALL_KWARGS.validate_python(kwargs)
    except ValidationError:
        return None
    litellm_params: Final = call.get("litellm_params")
    metadata: Final = litellm_params.get("metadata") if litellm_params else None
    model_info: Final = litellm_params.get("model_info") if litellm_params else None
    model_group: Final = metadata.get("model_group") if metadata else None
    deployment_id: Final = model_info.get("id") if model_info else None
    if model_group is None or deployment_id is None:
        return None
    return _CallRef(model_group, str(deployment_id), call.get("litellm_call_id"))


def _deployment_ids(healthy_deployments: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(
        str(deployment["model_info"]["id"]) for deployment in _DEPLOYMENTS.validate_python(healthy_deployments)
    )


def _as_counts(values: Sequence[float | None]) -> tuple[int, ...]:
    return tuple(0 if value is None else int(value) for value in values)


def _local_counts(raw: object, keys: tuple[str, ...]) -> tuple[int, ...]:
    values: Final = _MEMORY_COUNTS.validate_python(raw)
    if values is None or len(values) != len(keys):
        return (0,) * len(keys)
    return _as_counts(values)


def _least_busy(
    healthy_deployments: Sequence[Mapping[str, object]], counts: tuple[int, ...]
) -> Mapping[str, object] | None:
    if not healthy_deployments:
        return None
    return healthy_deployments[min(range(len(healthy_deployments)), key=lambda index: counts[index])]


def _warn_unreadable(model_group: str, error: Exception) -> None:
    log_redis_failure(
        verbose_router_logger,
        logging.WARNING,
        f"least-busy routing could not read the shared in-flight counts for {model_group}, "
        "falling back to this worker's own counts",
        error,
    )


def _warn_unwritable(key: str, error: Exception) -> None:
    log_redis_failure(
        verbose_router_logger,
        logging.WARNING,
        f"least-busy routing could not update the in-flight count under {key}",
        error,
    )


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(self, router_cache: DualCache, clock: Callable[[], float] = time.time):
        self.router_cache = router_cache
        self.router_cache_id = str(id(router_cache))
        self._clock = clock
        self._worker_id = uuid.uuid4().hex

    def _attempt_member(self, shared_key: str, call_id: str, delta: int) -> str:
        """Name this call's sorted-set member, unique per concurrent attempt sharing a ``litellm_call_id``.

        Callers can supply their own call id, so two concurrent calls may share one. Numbering the
        attempts per worker keeps one member per call, and finishing either call removes only one.
        """
        attempts_key: Final = f"{shared_key}:{call_id}:attempts"
        memory: Final = self.router_cache.in_memory_cache
        after: Final = int(memory.increment_cache(attempts_key, delta, ttl=IN_FLIGHT_COUNT_TTL_SECONDS))  # pyright: ignore[reportUnknownMemberType]  # InMemoryCache.increment_cache leaves key untyped
        if after <= 0:
            memory.delete_cache(attempts_key)
        return f"{call_id}:{self._worker_id}:{after if delta > 0 else after + 1}"

    @with_service_target("router_usage")
    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        self._increment(kwargs, 1)

    @with_service_target("router_usage")
    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._increment(kwargs, -1)
        if self.test_flag:
            self.logged_success += 1

    @with_service_target("router_usage")
    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._increment(kwargs, -1)
        if self.test_flag:
            self.logged_failure += 1

    @with_service_target("router_usage")
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._async_increment(kwargs, -1)
        if self.test_flag:
            self.logged_success += 1

    @with_service_target("router_usage")
    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._async_increment(kwargs, -1)
        if self.test_flag:
            self.logged_failure += 1

    @with_service_target("router_usage")
    def get_available_deployments(
        self, model_group: str, healthy_deployments: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object] | None:
        deployment_ids: Final = _deployment_ids(healthy_deployments)
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is not None:
            try:
                shared: Final = redis_cache.batch_count_in_flight(
                    [_in_flight_key(model_group, deployment_id) for deployment_id in deployment_ids],
                    self._clock() - IN_FLIGHT_COUNT_TTL_SECONDS,
                )
            except Exception as e:
                _warn_unreadable(model_group, e)
            else:
                return _least_busy(healthy_deployments, shared)
        keys: Final = tuple(_request_count_key(model_group, deployment_id) for deployment_id in deployment_ids)
        local: Final = _local_counts(self.router_cache.batch_get_cache(list(keys), local_only=True), keys)
        return _least_busy(healthy_deployments, local)

    @with_service_target("router_usage")
    async def async_get_available_deployments(
        self, model_group: str, healthy_deployments: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object] | None:
        deployment_ids: Final = _deployment_ids(healthy_deployments)
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is not None:
            try:
                shared: Final = await redis_cache.async_batch_count_in_flight(
                    [_in_flight_key(model_group, deployment_id) for deployment_id in deployment_ids],
                    self._clock() - IN_FLIGHT_COUNT_TTL_SECONDS,
                )
            except Exception as e:
                _warn_unreadable(model_group, e)
            else:
                return _least_busy(healthy_deployments, shared)
        keys: Final = tuple(_request_count_key(model_group, deployment_id) for deployment_id in deployment_ids)
        local: Final = _local_counts(await self.router_cache.async_batch_get_cache(list(keys), local_only=True), keys)
        return _least_busy(healthy_deployments, local)

    def _increment(self, kwargs: Mapping[str, object], delta: int) -> None:
        ref: Final = _call_ref(kwargs)
        if ref is None:
            return
        key: Final = _request_count_key(ref.model_group, ref.deployment_id)
        redis_cache: Final = self.router_cache.redis_cache
        try:
            local: Final = self.router_cache.increment_cache(
                key, delta, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS
            )
            if local < 0:
                self.router_cache.set_cache(key, 0, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS)
            if redis_cache is None or ref.call_id is None:
                return
            shared_key: Final = _in_flight_key(ref.model_group, ref.deployment_id)
            member: Final = self._attempt_member(shared_key, ref.call_id, delta)
            if delta > 0:
                redis_cache.track_in_flight(shared_key, member, self._clock(), IN_FLIGHT_COUNT_TTL_SECONDS)
            else:
                redis_cache.untrack_in_flight(shared_key, member)
        except Exception as e:
            _warn_unwritable(key, e)

    async def _async_increment(self, kwargs: Mapping[str, object], delta: int) -> None:
        ref: Final = _call_ref(kwargs)
        if ref is None:
            return
        key: Final = _request_count_key(ref.model_group, ref.deployment_id)
        redis_cache: Final = self.router_cache.redis_cache
        try:
            local: Final = await self.router_cache.async_increment_cache(
                key, delta, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS
            )
            if local is not None and local < 0:
                await self.router_cache.async_set_cache(key, 0, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS)
            if redis_cache is None or ref.call_id is None:
                return
            shared_key: Final = _in_flight_key(ref.model_group, ref.deployment_id)
            member: Final = self._attempt_member(shared_key, ref.call_id, delta)
            if delta > 0:
                await redis_cache.async_track_in_flight(shared_key, member, self._clock(), IN_FLIGHT_COUNT_TTL_SECONDS)
            else:
                await redis_cache.async_untrack_in_flight(shared_key, member)
        except Exception as e:
            _warn_unwritable(key, e)
