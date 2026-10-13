import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from math import ceil
from types import MappingProxyType
from typing import Final

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm._internal_context import with_service_target
from litellm.caching.caching import DualCache
from litellm.constants import LITELLM_METADATA_FIELD
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.token_counter import (
    token_counter,  # pyright: ignore[reportUnknownVariableType]  # legacy custom-tokenizer argument is untyped
)
from litellm.router_utils.batch_utils import is_batch_retrieve_call_type
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import DeploymentTypedDict
from litellm.types.utils import LiteLLMPydanticObjectBase


class _FrozenPayload(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True, from_attributes=True, protected_namespaces=())


class RoutingArgs(_FrozenPayload):
    price_exponent: float = 2.0
    top_n: int | None = Field(default=None, gt=0)
    health_window_seconds: float = Field(default=30.0, gt=0)
    demotion_failure_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    demotion_min_failures: int = Field(default=3, gt=0)
    default_output_tokens: int = 256


class _ModelInfo(_FrozenPayload):
    id: str | int | None = None


class _RequestMetadata(_FrozenPayload):
    model_group: str | None = None


class _LiteLLMParams(_FrozenPayload):
    model: str | None = None
    model_info: _ModelInfo | None = None
    metadata: _RequestMetadata | None = None
    litellm_metadata: _RequestMetadata | None = None


class _Deployment(_FrozenPayload):
    model_info: _ModelInfo | None = None
    litellm_params: _LiteLLMParams | None = None


class _CallbackPayload(_FrozenPayload):
    exception: object | None = None
    call_type: str | None = None
    litellm_params: _LiteLLMParams | None = None


class _StatusCode(_FrozenPayload):
    status_code: int | None = None


class _Usage(_FrozenPayload):
    completion_tokens: int | None = None


class _ResponseUsage(_FrozenPayload):
    usage: _Usage | None = None


class _CacheAccess:
    def __init__(self, cache: DualCache) -> None:
        self._cache: Final = cache

    def increment_cache(self, key: str, *, ttl: int) -> None:
        self._cache.increment_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped DualCache API
            key=key, value=1, ttl=ttl
        )

    async def async_increment_cache(self, key: str, *, ttl: int) -> None:
        await self._cache.async_increment_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped DualCache API
            key=key, value=1, ttl=ttl
        )

    def batch_get_cache(self, keys: Sequence[str]) -> tuple[object | None, ...] | None:
        return _CACHE_VALUES_ADAPTER.validate_python(
            self._cache.batch_get_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped DualCache API
                keys=list(keys)
            )
        )

    async def async_batch_get_cache(self, keys: Sequence[str]) -> tuple[object | None, ...] | None:
        return _CACHE_VALUES_ADAPTER.validate_python(
            await self._cache.async_batch_get_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped DualCache API
                keys=list(keys)
            )
        )


_CACHE_VALUES_ADAPTER: Final[TypeAdapter[tuple[object | None, ...] | None]] = TypeAdapter(
    tuple[object | None, ...] | None
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    deployment: DeploymentTypedDict
    deployment_id: str | None
    cost: float | None


def _callback_deployment_id(payload: _CallbackPayload) -> str | None:
    litellm_params: Final = payload.litellm_params
    model_info: Final = None if litellm_params is None else litellm_params.model_info
    deployment_id: Final = None if model_info is None else model_info.id
    return None if deployment_id is None else str(deployment_id)


def _counter_key(deployment_id: str, bucket: int, counter: str) -> str:
    return f"price_weighted:{deployment_id}:{bucket}:{counter}"


def _candidate_counter_keys(candidate: _Candidate, buckets: tuple[int, int]) -> tuple[str, ...]:
    if candidate.deployment_id is None:
        return ()
    attempts_keys: Final = tuple(_counter_key(candidate.deployment_id, bucket, "attempts") for bucket in buckets)
    failures_keys: Final = tuple(_counter_key(candidate.deployment_id, bucket, "failures") for bucket in buckets)
    return attempts_keys + failures_keys


def _counter_keys(candidates: tuple[_Candidate, ...], buckets: tuple[int, int]) -> tuple[str, ...]:
    return tuple(chain.from_iterable(_candidate_counter_keys(candidate, buckets) for candidate in candidates))


def _counter_value(value: object | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _deployment(value: object) -> _Deployment | None:
    try:
        return _Deployment.model_validate(value)
    except ValidationError:
        return None


def _input_token_count(
    messages: Sequence[AllMessageValues] | None,
    input: str | Sequence[object] | None,
) -> int:
    text: Final = (
        input if input is None or isinstance(input, str) else "".join(item for item in input if isinstance(item, str))
    )
    try:
        return token_counter(messages=messages, text=text)
    except Exception:  # noqa: BLE001  # unsupported request content must not block routing
        return 0


def _completion_tokens(response_obj: object) -> int | None:
    try:
        response: Final = _ResponseUsage.model_validate(response_obj)
    except ValidationError:
        return None
    return None if response.usage is None else response.usage.completion_tokens


class PriceWeightedLoggingHandler(CustomLogger):
    price_lookup: Callable[[str, str], tuple[float | None, float | None]]

    def __init__(
        self,
        router_cache: DualCache,
        price_lookup: Callable[[str, str], tuple[float | None, float | None]],
        routing_args: Mapping[str, object] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.router_cache: Final[_CacheAccess] = _CacheAccess(router_cache)
        self.price_lookup = price_lookup
        self.routing_args = RoutingArgs.model_validate(routing_args if routing_args is not None else {})
        self.rng = rng if rng is not None else random.Random()
        self._output_tokens_ema: Mapping[str, float] = MappingProxyType({})
        self._output_tokens_ema_lock = threading.Lock()

    def _retryable_failure_deployment_id(self, kwargs: Mapping[str, object]) -> str | None:
        payload: Final = _CallbackPayload.model_validate(kwargs)
        exception: Final = payload.exception
        if not isinstance(exception, Exception):
            return None
        if not isinstance(exception, litellm.Timeout):
            try:
                status_code: Final = _StatusCode.model_validate(exception).status_code
            except ValidationError:
                return None
            if status_code is None or (status_code != 429 and status_code < 500):
                return None
        return _callback_deployment_id(payload)

    def _counter_bucket(self) -> int:
        return int(time.time() // self.routing_args.health_window_seconds)

    def _counter_buckets(self, now: float) -> tuple[int, int]:
        current_bucket: Final = int(now // self.routing_args.health_window_seconds)
        return current_bucket, current_bucket - 1

    def _counter_ttl(self) -> int:
        return max(1, ceil(2 * self.routing_args.health_window_seconds))

    def _increment_counters(self, deployment_id: str, *, failure: bool) -> None:
        bucket: Final = self._counter_bucket()
        ttl: Final = self._counter_ttl()
        self.router_cache.increment_cache(
            key=_counter_key(deployment_id, bucket, "attempts"),
            ttl=ttl,
        )
        if failure:
            self.router_cache.increment_cache(
                key=_counter_key(deployment_id, bucket, "failures"),
                ttl=ttl,
            )

    async def _async_increment_counters(self, deployment_id: str, *, failure: bool) -> None:
        bucket: Final = self._counter_bucket()
        ttl: Final = self._counter_ttl()
        await self.router_cache.async_increment_cache(
            key=_counter_key(deployment_id, bucket, "attempts"),
            ttl=ttl,
        )
        if failure:
            await self.router_cache.async_increment_cache(
                key=_counter_key(deployment_id, bucket, "failures"),
                ttl=ttl,
            )

    @with_service_target("router_usage")
    def log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        deployment_id: Final = self._retryable_failure_deployment_id(kwargs)
        if deployment_id is None:
            return
        self._increment_counters(deployment_id, failure=True)

    @with_service_target("router_usage")
    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        deployment_id: Final = self._retryable_failure_deployment_id(kwargs)
        if deployment_id is None:
            return
        await self._async_increment_counters(deployment_id, failure=True)

    def _record_success_event(
        self,
        payload: _CallbackPayload,
        response_obj: object,
        *,
        has_litellm_metadata: bool,
    ) -> None:
        if payload.call_type is not None and is_batch_retrieve_call_type(payload.call_type):
            return
        litellm_params: Final = payload.litellm_params
        if litellm_params is None:
            return
        metadata: Final = litellm_params.litellm_metadata if has_litellm_metadata else litellm_params.metadata
        model_group: Final = None if metadata is None else metadata.model_group
        completion_tokens: Final = _completion_tokens(response_obj)
        if not isinstance(model_group, str) or completion_tokens is None:
            return
        with self._output_tokens_ema_lock:
            previous: Final = self._output_tokens_ema.get(model_group)
            next_ema: Final = float(completion_tokens) if previous is None else 0.2 * completion_tokens + 0.8 * previous
            self._output_tokens_ema = MappingProxyType({**self._output_tokens_ema, model_group: next_ema})

    @with_service_target("router_usage")
    def log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        payload: Final = _CallbackPayload.model_validate(kwargs)
        deployment_id: Final = _callback_deployment_id(payload)
        if deployment_id is not None:
            self._increment_counters(deployment_id, failure=False)
        self._record_success_event(
            payload,
            response_obj,
            has_litellm_metadata=LITELLM_METADATA_FIELD in kwargs,
        )

    @with_service_target("router_usage")
    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        payload: Final = _CallbackPayload.model_validate(kwargs)
        deployment_id: Final = _callback_deployment_id(payload)
        if deployment_id is not None:
            await self._async_increment_counters(deployment_id, failure=False)
        self._record_success_event(
            payload,
            response_obj,
            has_litellm_metadata=LITELLM_METADATA_FIELD in kwargs,
        )

    def _candidates(
        self,
        healthy_deployments: Sequence[DeploymentTypedDict],
        input_tokens: int,
        output_tokens: float,
    ) -> tuple[_Candidate, ...]:
        return tuple(self._candidate(deployment, input_tokens, output_tokens) for deployment in healthy_deployments)

    def _candidate(
        self,
        deployment: DeploymentTypedDict,
        input_tokens: int,
        output_tokens: float,
    ) -> _Candidate:
        details: Final = _deployment(deployment)
        model_info: Final = None if details is None else details.model_info
        litellm_params: Final = None if details is None else details.litellm_params
        raw_deployment_id: Final = None if model_info is None else model_info.id
        deployment_id: Final = None if raw_deployment_id is None else str(raw_deployment_id)
        model_name: Final = None if litellm_params is None else litellm_params.model
        if deployment_id is None or model_name is None:
            return _Candidate(deployment=deployment, deployment_id=deployment_id, cost=None)
        input_price, output_price = self.price_lookup(deployment_id, model_name)
        if input_price is None or output_price is None:
            return _Candidate(deployment=deployment, deployment_id=deployment_id, cost=None)
        cost: Final = input_tokens * input_price + output_tokens * output_price
        return _Candidate(deployment=deployment, deployment_id=deployment_id, cost=cost)

    def _request_candidates(
        self,
        model_group: str,
        healthy_deployments: Sequence[DeploymentTypedDict],
        messages: Sequence[AllMessageValues] | None,
        input: str | Sequence[object] | None,
    ) -> tuple[_Candidate, ...]:
        with self._output_tokens_ema_lock:
            output_tokens: Final = self._output_tokens_ema.get(
                model_group, float(self.routing_args.default_output_tokens)
            )
        return self._candidates(
            healthy_deployments=healthy_deployments,
            input_tokens=_input_token_count(messages=messages, input=input),
            output_tokens=output_tokens,
        )

    def _stable_candidates(
        self,
        candidates: tuple[_Candidate, ...],
        counter_keys: tuple[str, ...],
        cache_values: Sequence[object | None] | None,
        buckets: tuple[int, int],
        previous_bucket_weight: float,
    ) -> tuple[_Candidate, ...]:
        if cache_values is None:
            return candidates
        values_by_key: Final = MappingProxyType(dict(zip(counter_keys, cache_values)))
        stable: Final = tuple(
            candidate
            for candidate in candidates
            if not self._is_candidate_demoted(candidate, values_by_key, buckets, previous_bucket_weight)
        )
        return stable if stable else candidates

    def _is_candidate_demoted(
        self,
        candidate: _Candidate,
        values_by_key: Mapping[str, object | None],
        buckets: tuple[int, int],
        previous_bucket_weight: float,
    ) -> bool:
        if candidate.deployment_id is None:
            return False
        current_bucket, previous_bucket = buckets
        attempts: Final = _counter_value(
            values_by_key.get(_counter_key(candidate.deployment_id, current_bucket, "attempts"))
        ) + previous_bucket_weight * _counter_value(
            values_by_key.get(_counter_key(candidate.deployment_id, previous_bucket, "attempts"))
        )
        failures: Final = _counter_value(
            values_by_key.get(_counter_key(candidate.deployment_id, current_bucket, "failures"))
        ) + previous_bucket_weight * _counter_value(
            values_by_key.get(_counter_key(candidate.deployment_id, previous_bucket, "failures"))
        )
        return (
            failures >= self.routing_args.demotion_min_failures
            and attempts > 0
            and failures / attempts >= self.routing_args.demotion_failure_ratio
        )

    def _pick_candidate(self, candidates: tuple[_Candidate, ...]) -> DeploymentTypedDict:
        known_costs: Final = tuple(candidate.cost for candidate in candidates if candidate.cost is not None)
        if not known_costs:
            return self.rng.choice(candidates).deployment
        highest_known_cost: Final = max(known_costs)
        costs: Final = tuple(
            (candidate, candidate.cost if candidate.cost is not None else highest_known_cost)
            for candidate in candidates
        )
        zero_cost_candidates: Final = tuple(candidate for candidate, cost in costs if cost == 0)
        if zero_cost_candidates:
            return self.rng.choice(zero_cost_candidates).deployment
        ranked: Final = tuple(sorted(costs, key=lambda candidate_cost: candidate_cost[1]))
        limited: Final = ranked if self.routing_args.top_n is None else ranked[: self.routing_args.top_n]
        population: Final = tuple(candidate for candidate, _ in limited)
        weights: Final = tuple(cost**-self.routing_args.price_exponent for _, cost in limited)
        return self.rng.choices(population=population, weights=weights, k=1)[0].deployment

    def _select_deployment(
        self,
        candidates: tuple[_Candidate, ...],
        counter_keys: tuple[str, ...],
        cache_values: Sequence[object | None] | None,
        buckets: tuple[int, int],
        previous_bucket_weight: float,
    ) -> DeploymentTypedDict | None:
        if not candidates:
            return None
        stable_candidates: Final = self._stable_candidates(
            candidates=candidates,
            counter_keys=counter_keys,
            cache_values=cache_values,
            buckets=buckets,
            previous_bucket_weight=previous_bucket_weight,
        )
        return self._pick_candidate(candidates=stable_candidates)

    @with_service_target("router_usage")
    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: Sequence[DeploymentTypedDict],
        messages: Sequence[AllMessageValues] | None = None,
        input: str | Sequence[object] | None = None,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> DeploymentTypedDict | None:
        if not healthy_deployments:
            return None
        candidates: Final = self._request_candidates(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            messages=messages,
            input=input,
        )
        now: Final = time.time()
        window: Final = self.routing_args.health_window_seconds
        buckets: Final = self._counter_buckets(now)
        previous_bucket_weight: Final = 1.0 - (now % window) / window
        counter_keys: Final = _counter_keys(candidates, buckets)
        cache_values: Final = self.router_cache.batch_get_cache(counter_keys) if counter_keys else ()
        return self._select_deployment(
            candidates=candidates,
            counter_keys=counter_keys,
            cache_values=cache_values,
            buckets=buckets,
            previous_bucket_weight=previous_bucket_weight,
        )

    @with_service_target("router_usage")
    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: Sequence[DeploymentTypedDict],
        messages: Sequence[AllMessageValues] | None = None,
        input: str | Sequence[object] | None = None,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> DeploymentTypedDict | None:
        if not healthy_deployments:
            return None
        candidates: Final = self._request_candidates(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            messages=messages,
            input=input,
        )
        now: Final = time.time()
        window: Final = self.routing_args.health_window_seconds
        buckets: Final = self._counter_buckets(now)
        previous_bucket_weight: Final = 1.0 - (now % window) / window
        counter_keys: Final = _counter_keys(candidates, buckets)
        cache_values: Final = await self.router_cache.async_batch_get_cache(counter_keys) if counter_keys else ()
        return self._select_deployment(
            candidates=candidates,
            counter_keys=counter_keys,
            cache_values=cache_values,
            buckets=buckets,
            previous_bucket_weight=previous_bucket_weight,
        )
