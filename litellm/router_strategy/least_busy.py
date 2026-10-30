import contextlib
import logging
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextvars import ContextVar
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import log_redis_failure
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type

IN_FLIGHT_COUNT_TTL_SECONDS: Final = 60 * 60
SELECTION_RESERVATION_TTL_SECONDS: Final = 30.0
SELECTION_RESERVATION_MAX_KEYS: Final = 10_000
_RESERVATIONS_LOCK: Final = threading.Lock()
_PICK_RESERVES: Final[ContextVar[bool]] = ContextVar("least_busy_pick_reserves")


@contextlib.contextmanager
def picks_reserve(reserve: bool) -> Generator[None, None, None]:
    token: Final = _PICK_RESERVES.set(reserve)
    try:
        yield
    finally:
        _PICK_RESERVES.reset(token)


def pick_reserves(default: bool) -> bool:
    return _PICK_RESERVES.get(default)


class _ModelInfo(TypedDict, total=False):
    id: ReadOnly[str | int | None]


class _Metadata(TypedDict, total=False):
    model_group: ReadOnly[str | None]
    model_info: ReadOnly[_ModelInfo | None]


class _LitellmParams(TypedDict, total=False):
    metadata: ReadOnly[_Metadata | None]
    litellm_metadata: ReadOnly[_Metadata | None]
    model_info: ReadOnly[_ModelInfo | None]


class _CallKwargs(TypedDict, total=False):
    litellm_params: ReadOnly[_LitellmParams | None]


class _DeploymentModelInfo(TypedDict):
    id: ReadOnly[str | int]


class _Deployment(TypedDict):
    model_info: ReadOnly[_DeploymentModelInfo]


_CALL_KWARGS: Final = TypeAdapter(_CallKwargs)
_DEPLOYMENTS: Final = TypeAdapter(list[_Deployment])
_MEMORY_COUNTS: Final[TypeAdapter[tuple[float | None, ...] | None]] = TypeAdapter(tuple[float | None, ...] | None)
_RESERVATION_TIMES: Final[TypeAdapter[tuple[float, ...] | None]] = TypeAdapter(tuple[float, ...] | None)


def _request_count_key(model_group: str, deployment_id: str) -> str:
    return f"{model_group}_request_count:{deployment_id}"


def _model_group(metadata: _Metadata | None) -> str | None:
    return metadata.get("model_group") if metadata else None


def _deployment_id(model_info: _ModelInfo | None) -> str | int | None:
    return model_info.get("id") if model_info else None


def _routed_deployment_id(metadata: _Metadata | None) -> str | int | None:
    return _deployment_id(metadata.get("model_info")) if metadata else None


def _deployment_ref(kwargs: Mapping[str, object]) -> tuple[str, str] | None:
    if is_batch_retrieve_call_type(kwargs.get("call_type")):
        return None
    try:
        call: Final = _CALL_KWARGS.validate_python(kwargs)
    except ValidationError:
        return None
    litellm_params: Final = call.get("litellm_params") or _LitellmParams()
    routing_metadata: Final = (litellm_params.get("litellm_metadata"), litellm_params.get("metadata"))
    model_group: Final = next((group for group in map(_model_group, routing_metadata) if group is not None), None)
    deployment_id: Final = next(
        (
            deployment_id
            for deployment_id in (
                _deployment_id(litellm_params.get("model_info")),
                *map(_routed_deployment_id, routing_metadata),
            )
            if deployment_id is not None
        ),
        None,
    )
    if model_group is None or deployment_id is None:
        return None
    return model_group, str(deployment_id)


def _reservation_key(deployment_id: str) -> str:
    return f"least_busy_reservations:{deployment_id}"


def _started_attempts_key(call_id: str, ref: tuple[str, str]) -> str:
    return f"least_busy_started:{call_id}:{_request_count_key(*ref)}"


def _deployment_ids(healthy_deployments: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(
        str(deployment["model_info"]["id"]) for deployment in _DEPLOYMENTS.validate_python(healthy_deployments)
    )


def _request_count_keys(model_group: str, deployment_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_request_count_key(model_group, deployment_id) for deployment_id in deployment_ids)


def _as_counts(values: Sequence[float | None]) -> tuple[int, ...]:
    return tuple(0 if value is None else int(value) for value in values)


def _local_counts(raw: object, keys: tuple[str, ...]) -> tuple[int, ...]:
    values: Final = _MEMORY_COUNTS.validate_python(raw)
    if values is None or len(values) != len(keys):
        return (0,) * len(keys)
    return _as_counts(values)


def _unexpired(times: Sequence[float], now: float) -> tuple[float, ...]:
    return tuple(t for t in times if now - t < SELECTION_RESERVATION_TTL_SECONDS)


def _least_busy_index(counts: tuple[int, ...]) -> int:
    return min(range(len(counts)), key=lambda index: counts[index])


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


class _ReservationStore(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: tuple[float, ...], *, ttl: float) -> None: ...

    def delete_cache(self, key: str) -> None: ...


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(
        self,
        router_cache: DualCache,
        clock: Callable[[], float] = time.monotonic,
        reservation_store: _ReservationStore | None = None,
    ):
        self.router_cache = router_cache
        self.router_cache_id = str(id(router_cache))
        self._clock = clock
        self._reservations: Final[_ReservationStore] = (
            reservation_store
            if reservation_store is not None
            else InMemoryCache(max_size_in_memory=SELECTION_RESERVATION_MAX_KEYS, clock=clock)
        )

    def _times(self, key: str) -> tuple[float, ...]:
        return _RESERVATION_TIMES.validate_python(self._reservations.get_cache(key)) or ()

    def _store_times(self, key: str, times: tuple[float, ...], ttl: float) -> None:
        # InMemoryCache keeps a live key's first TTL, so drop the key to restart it
        self._reservations.delete_cache(key)
        if times:
            self._reservations.set_cache(key, times, ttl=ttl)

    def _live_reservations(self, deployment_id: str, now: float) -> tuple[float, ...]:
        return _unexpired(self._times(_reservation_key(deployment_id)), now)

    def _store_reservations(self, deployment_id: str, times: tuple[float, ...]) -> None:
        self._store_times(_reservation_key(deployment_id), times, SELECTION_RESERVATION_TTL_SECONDS)

    def _drop_oldest_reservation(self, deployment_id: str, now: float) -> None:
        self._store_reservations(deployment_id, self._live_reservations(deployment_id, now)[1:])

    def _select(
        self,
        healthy_deployments: Sequence[Mapping[str, object]],
        deployment_ids: tuple[str, ...],
        counts: tuple[int, ...],
        reserve: bool,
    ) -> Mapping[str, object] | None:
        if not healthy_deployments:
            return None
        with _RESERVATIONS_LOCK:
            now: Final = self._clock()
            live: Final = tuple(self._live_reservations(deployment_id, now) for deployment_id in deployment_ids)
            index: Final = _least_busy_index(tuple(count + len(r) for count, r in zip(counts, live, strict=True)))
            if reserve:
                self._store_reservations(deployment_ids[index], (*live[index], now))
        return healthy_deployments[index]

    def _start_attempt(self, ref: tuple[str, str], call_id: object) -> None:
        with _RESERVATIONS_LOCK:
            now: Final = self._clock()
            self._drop_oldest_reservation(ref[1], now)
            if isinstance(call_id, str):
                key: Final = _started_attempts_key(call_id, ref)
                self._store_times(key, (*self._times(key), now), IN_FLIGHT_COUNT_TTL_SECONDS)

    def _end_attempt(self, ref: tuple[str, str], call_id: object) -> bool:
        if not isinstance(call_id, str):
            return True
        key: Final = _started_attempts_key(call_id, ref)
        with _RESERVATIONS_LOCK:
            started: Final = self._times(key)
            if started:
                self._store_times(key, started[1:], IN_FLIGHT_COUNT_TTL_SECONDS)
            else:
                self._drop_oldest_reservation(ref[1], self._clock())
        return bool(started)

    @with_service_target("router_usage")
    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        ref: Final = _deployment_ref(kwargs)
        if ref is None:
            return
        self._start_attempt(ref, kwargs.get("litellm_call_id"))
        self._increment(_request_count_key(*ref), 1)

    @with_service_target("router_usage")
    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._settle(kwargs)
        if self.test_flag:
            self.logged_success += 1

    @with_service_target("router_usage")
    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._settle(kwargs)
        if self.test_flag:
            self.logged_failure += 1

    @with_service_target("router_usage")
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._async_settle(kwargs)
        if self.test_flag:
            self.logged_success += 1

    @with_service_target("router_usage")
    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._async_settle(kwargs)
        if self.test_flag:
            self.logged_failure += 1

    @with_service_target("router_usage")
    def get_available_deployments(
        self, model_group: str, healthy_deployments: Sequence[Mapping[str, object]], reserve: bool = False
    ) -> Mapping[str, object] | None:
        deployment_ids: Final = _deployment_ids(healthy_deployments)
        keys: Final = _request_count_keys(model_group, deployment_ids)
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is not None:
            try:
                shared: Final = _as_counts(redis_cache.batch_get_counts(list(keys)))
            except Exception as e:
                _warn_unreadable(model_group, e)
            else:
                return self._select(
                    healthy_deployments, deployment_ids, self._at_least_this_workers(keys, shared), reserve
                )
        return self._select(healthy_deployments, deployment_ids, self._this_workers_counts(keys), reserve)

    @with_service_target("router_usage")
    async def async_get_available_deployments(
        self, model_group: str, healthy_deployments: Sequence[Mapping[str, object]], reserve: bool = False
    ) -> Mapping[str, object] | None:
        deployment_ids: Final = _deployment_ids(healthy_deployments)
        keys: Final = _request_count_keys(model_group, deployment_ids)
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is not None:
            try:
                shared: Final = _as_counts(await redis_cache.async_batch_get_counts(list(keys)))
            except Exception as e:
                _warn_unreadable(model_group, e)
            else:
                return self._select(
                    healthy_deployments, deployment_ids, self._at_least_this_workers(keys, shared), reserve
                )
        return self._select(healthy_deployments, deployment_ids, self._this_workers_counts(keys), reserve)

    def _settle(self, kwargs: Mapping[str, object]) -> None:
        ref: Final = _deployment_ref(kwargs)
        if ref is None or not self._end_attempt(ref, kwargs.get("litellm_call_id")):
            return
        self._increment(_request_count_key(*ref), -1)

    async def _async_settle(self, kwargs: Mapping[str, object]) -> None:
        ref: Final = _deployment_ref(kwargs)
        if ref is None or not self._end_attempt(ref, kwargs.get("litellm_call_id")):
            return
        await self._async_increment(_request_count_key(*ref), -1)

    def _this_workers_counts(self, keys: tuple[str, ...]) -> tuple[int, ...]:
        return _local_counts(self.router_cache.batch_get_cache(list(keys), local_only=True), keys)

    def _at_least_this_workers(self, keys: tuple[str, ...], shared: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(max(counts) for counts in zip(shared, self._this_workers_counts(keys), strict=True))

    def _increment(self, key: str, delta: int) -> None:
        redis_cache: Final = self.router_cache.redis_cache
        try:
            local: Final = self.router_cache.increment_cache(
                key, delta, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS
            )
            if local < 0:
                self.router_cache.set_cache(key, 0, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS)
            if redis_cache is None:
                return
            redis_cache.increment_with_floor(key, delta, IN_FLIGHT_COUNT_TTL_SECONDS)
        except Exception as e:
            _warn_unwritable(key, e)

    async def _async_increment(self, key: str, delta: int) -> None:
        redis_cache: Final = self.router_cache.redis_cache
        try:
            local: Final = await self.router_cache.async_increment_cache(
                key, delta, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS
            )
            if local is not None and local < 0:
                await self.router_cache.async_set_cache(key, 0, local_only=True, ttl=IN_FLIGHT_COUNT_TTL_SECONDS)
            if redis_cache is None:
                return
            await redis_cache.async_increment_with_floor(key, delta, IN_FLIGHT_COUNT_TTL_SECONDS)
        except Exception as e:
            _warn_unwritable(key, e)
