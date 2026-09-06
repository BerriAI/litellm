from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger

IN_FLIGHT_COUNT_TTL_SECONDS: Final = 60 * 60


class _ModelInfo(TypedDict, total=False):
    id: ReadOnly[str | int | None]


class _Metadata(TypedDict, total=False):
    model_group: ReadOnly[str | None]


class _LitellmParams(TypedDict, total=False):
    metadata: ReadOnly[_Metadata | None]
    model_info: ReadOnly[_ModelInfo | None]


class _CallKwargs(TypedDict, total=False):
    litellm_params: ReadOnly[_LitellmParams | None]


class _DeploymentModelInfo(TypedDict):
    id: ReadOnly[str | int]


class _Deployment(TypedDict):
    model_info: ReadOnly[_DeploymentModelInfo]


_CALL_KWARGS: Final = TypeAdapter(_CallKwargs)
_DEPLOYMENTS: Final = TypeAdapter(list[_Deployment])
_MEMORY_COUNTS: Final = TypeAdapter(tuple[float | None, ...] | None)


def _request_count_key(model_group: str, deployment_id: str) -> str:
    return f"{model_group}_request_count:{deployment_id}"


def _deployment_ref(kwargs: Mapping[str, object]) -> tuple[str, str] | None:
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
    return model_group, str(deployment_id)


def _request_count_keys(model_group: str, healthy_deployments: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(
        _request_count_key(model_group, str(deployment["model_info"]["id"]))
        for deployment in _DEPLOYMENTS.validate_python(healthy_deployments)
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
    verbose_router_logger.warning(
        "least-busy routing could not read the shared in-flight counts for %s, "
        "falling back to this worker's own counts: %s",
        model_group,
        error,
    )


def _warn_unwritable(key: str, error: Exception) -> None:
    verbose_router_logger.warning("least-busy routing could not update the in-flight count under %s: %s", key, error)


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(self, router_cache: DualCache):
        self.router_cache = router_cache
        self.router_cache_id = str(id(router_cache))

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        self._increment(kwargs, 1)

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._increment(kwargs, -1)
        if self.test_flag:
            self.logged_success += 1

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._increment(kwargs, -1)
        if self.test_flag:
            self.logged_failure += 1

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._async_increment(kwargs, -1)
        if self.test_flag:
            self.logged_success += 1

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._async_increment(kwargs, -1)
        if self.test_flag:
            self.logged_failure += 1

    def get_available_deployments(
        self, model_group: str, healthy_deployments: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object] | None:
        keys: Final = _request_count_keys(model_group, healthy_deployments)
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is not None:
            try:
                shared: Final = _as_counts(redis_cache.batch_get_counts(list(keys)))
            except Exception as e:
                _warn_unreadable(model_group, e)
            else:
                return _least_busy(healthy_deployments, shared)
        local: Final = _local_counts(self.router_cache.batch_get_cache(list(keys), local_only=True), keys)
        return _least_busy(healthy_deployments, local)

    async def async_get_available_deployments(
        self, model_group: str, healthy_deployments: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object] | None:
        keys: Final = _request_count_keys(model_group, healthy_deployments)
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is not None:
            try:
                shared: Final = _as_counts(await redis_cache.async_batch_get_counts(list(keys)))
            except Exception as e:
                _warn_unreadable(model_group, e)
            else:
                return _least_busy(healthy_deployments, shared)
        local: Final = _local_counts(await self.router_cache.async_batch_get_cache(list(keys), local_only=True), keys)
        return _least_busy(healthy_deployments, local)

    def _increment(self, kwargs: Mapping[str, object], delta: int) -> None:
        ref: Final = _deployment_ref(kwargs)
        if ref is None:
            return
        key: Final = _request_count_key(*ref)
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

    async def _async_increment(self, kwargs: Mapping[str, object], delta: int) -> None:
        ref: Final = _deployment_ref(kwargs)
        if ref is None:
            return
        key: Final = _request_count_key(*ref)
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
