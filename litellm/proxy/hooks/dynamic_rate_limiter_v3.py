"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.active_request import active_request_disconnected
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.fairness_queue import (
    FairnessQueueRejectReason,
    FairQueue,
    QueueAdmitted,
    QueueRejected,
    QueueReleased,
    QueueTicket,
)
from litellm.proxy.hooks.fairness_stats import FairnessMetric, FairnessStats
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    RateLimitResponse,
    RateLimitStatus,
    RequestRateLimiterStash,
    _PROXY_MaxParallelRequestsHandler_v3,
    call_id_from_callback_kwargs,
    claim_request_stash_for_data,
    get_or_create_request_stash,
    get_request_stash_for_call,
)
from litellm.proxy.hooks.rate_limiter_utils import (
    convert_priority_to_percent,
    resolve_llm_provider_for_rate_limit,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.proxy.fairness import (
    DEFAULT_POOL_NAME,
    MAX_QUEUE_WAIT_HEADER,
    FairnessRejectReason,
    FairnessSettings,
    ModelFairnessStatus,
    WorkloadClassStatus,
)
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import CallTypesLiteral

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.types.utils import PriorityReservationSettings

_HEADER_FLOAT_ADAPTER: Final = TypeAdapter(float)
_COUNTER_ADAPTER: Final = TypeAdapter(float)
_RAW_VALUES_ADAPTER: Final = TypeAdapter(tuple[object, ...])
_REJECT_METRIC: Final[Mapping[FairnessQueueRejectReason, FairnessMetric]] = MappingProxyType(
    {
        "queue_full": "rejected_queue_full",
        "queue_deadline_exceeded": "rejected_deadline",
        "client_disconnected": "disconnected",
    }
)


_TRACKING_LIMIT_MULTIPLIER: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    model: str
    class_name: str
    enforced: tuple[RateLimitDescriptor, ...]
    tracking_only: tuple[RateLimitDescriptor, ...]
    estimated_tokens: int

    @property
    def token_scopes(self) -> frozenset[tuple[str, str]]:
        return _token_scopes(*self.enforced, *self.tracking_only)

    @property
    def descriptors(self) -> tuple[RateLimitDescriptor, ...]:
        return (*self.enforced, *(_with_tracking_limits(descriptor) for descriptor in self.tracking_only))


def _scaled_limit(limit: int | None) -> int | None:
    return None if limit is None else max(limit, 1) * _TRACKING_LIMIT_MULTIPLIER


def _with_tracking_limits(descriptor: RateLimitDescriptor) -> RateLimitDescriptor:
    rate_limit: Final = descriptor["rate_limit"]
    if rate_limit is None:
        return descriptor
    return RateLimitDescriptor(
        key=descriptor["key"],
        value=descriptor["value"],
        rate_limit=RateLimitDescriptorRateLimitObject(
            requests_per_unit=_scaled_limit(rate_limit.get("requests_per_unit")),
            tokens_per_unit=_scaled_limit(rate_limit.get("tokens_per_unit")),
            window_size=rate_limit.get("window_size"),
        ),
    )


def _configured_limit(descriptor: RateLimitDescriptor, rate_limit_type: str) -> int | None:
    rate_limit: Final = descriptor["rate_limit"]
    if rate_limit is None:
        return None
    return rate_limit.get("requests_per_unit") if rate_limit_type == "requests" else rate_limit.get("tokens_per_unit")


def _with_configured_limit(status: RateLimitStatus, tracking_only: Sequence[RateLimitDescriptor]) -> RateLimitStatus:
    descriptor: Final = next(
        (
            candidate
            for candidate in tracking_only
            if (candidate["key"], candidate["value"]) == (status["descriptor_key"], status.get("descriptor_value"))
        ),
        None,
    )
    configured: Final = _configured_limit(descriptor, status["rate_limit_type"]) if descriptor is not None else None
    if configured is None:
        return status
    used: Final = status["current_limit"] - status["limit_remaining"]
    return RateLimitStatus(
        code=status["code"],
        current_limit=configured,
        limit_remaining=max(0, configured - used),
        rate_limit_type=status["rate_limit_type"],
        descriptor_key=status["descriptor_key"],
        descriptor_value=status.get("descriptor_value", ""),
    )


def _tracks_tokens(descriptor: RateLimitDescriptor) -> bool:
    rate_limit: Final = descriptor.get("rate_limit")
    return rate_limit is not None and rate_limit.get("tokens_per_unit") is not None


def _token_scopes(*descriptors: RateLimitDescriptor) -> frozenset[tuple[str, str]]:
    return frozenset(
        (descriptor["key"], descriptor["value"]) for descriptor in descriptors if _tracks_tokens(descriptor)
    )


def _counter_value(raw: object) -> float:
    if raw is None:
        return 0.0
    try:
        return _COUNTER_ADAPTER.validate_python(raw)
    except ValidationError:
        return 0.0


def _requested_max_queue_wait(data: Mapping[str, object]) -> float | None:
    proxy_request: Final = data.get("proxy_server_request")
    if not isinstance(proxy_request, Mapping):
        return None
    headers: Final = proxy_request.get("headers")
    if not isinstance(headers, Mapping):
        return None
    raw: Final = headers.get(MAX_QUEUE_WAIT_HEADER)
    if raw is None:
        return None
    try:
        return _HEADER_FLOAT_ADAPTER.validate_python(raw)
    except ValidationError:
        return None


def _get_priority_settings() -> "PriorityReservationSettings":
    """
    Get the priority reservation settings, guaranteed to be non-None.

    The settings are lazy-loaded in litellm.__init__ and always return an instance.
    This helper provides proper type narrowing for mypy.
    """
    settings: Final = litellm.priority_reservation_settings
    if settings is None:
        # This should never happen due to lazy loading, but satisfy mypy
        from litellm.types.utils import PriorityReservationSettings

        return PriorityReservationSettings()
    return settings


def _is_latin1_encodable(value: object) -> bool:
    return all(ord(char) < 256 for char in str(value))


class _PROXY_DynamicRateLimitHandlerV3(CustomLogger):
    """
    Saturation-aware priority-based rate limiter using v3 infrastructure.

    Key features:
    1. Model capacity ALWAYS enforced at 100% (prevents over-allocation)
    2. Priority usage tracked from first request (accurate accounting)
    3. Priority limits only enforced when saturated >= threshold
    4. Three-phase checking prevents partial counter increments
    5. Reuses v3 limiter's Redis-based tracking (multi-instance safe)

    How it works:
    - Phase 1: Read-only check of ALL limits (no increments)
    - Phase 2: Decide enforcement based on saturation
    - Phase 3: Increment counters only if request allowed
    - When under-saturated: priorities can borrow unused capacity (generous)
    - When saturated: strict priority-based limits enforced (fair)
    - Uses v3 limiter's atomic Lua scripts for race-free increments
    """

    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        fair_queue: FairQueue | None = None,
        fairness_stats: FairnessStats | None = None,
        is_client_disconnected: Callable[[], Awaitable[bool]] = active_request_disconnected,
    ):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self.v3_limiter = _PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache, time_provider=time_provider)
        self.fair_queue = fair_queue if fair_queue is not None else FairQueue()
        self.fairness_stats = (
            fairness_stats if fairness_stats is not None else FairnessStats(cache=self.internal_usage_cache)
        )
        self.is_client_disconnected = is_client_disconnected
        self.llm_router: Router | None = None
        self.enforcing = True

    def update_variables(self, llm_router: Router):
        self.llm_router = llm_router

    @staticmethod
    def _fairness_settings() -> FairnessSettings | None:
        return litellm.fairness_settings

    @staticmethod
    def _class_name_for(priority: str | None) -> str:
        has_explicit_priority: Final = (
            priority is not None
            and litellm.priority_reservation is not None
            and priority in litellm.priority_reservation
        )
        return priority if has_explicit_priority and priority is not None else DEFAULT_POOL_NAME

    def _get_saturation_check_cache_ttl(self) -> int:
        """Get the configurable TTL for local cache when reading saturation values."""
        return _get_priority_settings().saturation_check_cache_ttl

    async def _get_saturation_value_from_cache(
        self,
        counter_key: str,
    ) -> str | None:
        """
        Get saturation value with configurable local cache TTL.

        Uses DualCache with configurable TTL for local cache storage.
        TTL is configurable via litellm.priority_reservation_settings.saturation_check_cache_ttl

        Args:
            counter_key: The cache key for the saturation counter

        Returns:
            Counter value as string, or None if not found
        """
        local_cache_ttl: Final = self._get_saturation_check_cache_ttl()

        return await self.internal_usage_cache.async_get_cache(
            key=counter_key,
            litellm_parent_otel_span=None,
            local_only=False,
            ttl=local_cache_ttl,
        )

    def _get_priority_weight(self, priority: str | None, model_info: ModelGroupInfo | None = None) -> float:
        """Get the weight for a given priority from litellm.priority_reservation"""
        weight: float = _get_priority_settings().default_priority
        if litellm.priority_reservation is None or priority not in litellm.priority_reservation:
            verbose_proxy_logger.debug("Priority Reservation not set for the given priority.")
        elif priority is not None and litellm.priority_reservation is not None:
            if os.getenv("LITELLM_LICENSE", None) is None:
                verbose_proxy_logger.error(
                    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\nGet a license: https://docs.litellm.ai/docs/proxy/enterprise."
                )
            else:
                value: Final = litellm.priority_reservation[priority]
                weight = convert_priority_to_percent(value, model_info)
        return weight

    def _get_priority_from_user_api_key_dict(self, user_api_key_dict: UserAPIKeyAuth) -> str | None:
        """
        Get priority from user_api_key_dict.

        Checks team metadata first (takes precedence), then falls back to key metadata.

        Args:
            user_api_key_dict: User authentication info

        Returns:
            Priority string if found, None otherwise
        """
        priority: str | None = None

        # Check team metadata first (takes precedence)
        if user_api_key_dict.team_metadata is not None:
            priority = user_api_key_dict.team_metadata.get("priority", None)

        # Fall back to key metadata
        if priority is None:
            priority = user_api_key_dict.metadata.get("priority", None)

        return priority

    def _normalize_priority_weights(self, model_info: ModelGroupInfo) -> dict[str, float]:
        """
        Normalize priority weights if they sum to > 1.0

        Handles over-allocation: {key_a: 0.60, key_b: 0.80} -> {key_a: 0.43, key_b: 0.57}
        Converts absolute rpm/tpm values to percentages based on model capacity.
        """
        if litellm.priority_reservation is None:
            return {}

        # Convert all values to percentages first
        weights: Final[dict[str, float]] = {}
        for k, v in litellm.priority_reservation.items():
            weights[k] = convert_priority_to_percent(v, model_info)

        total_weight: Final = sum(weights.values())

        if total_weight > 1.0:
            normalized: Final = {k: v / total_weight for k, v in weights.items()}
            verbose_proxy_logger.debug("Normalized over-allocated priorities: %s -> %s", weights, normalized)
            return normalized

        return weights

    def _get_priority_allocation(
        self,
        model: str,
        priority: str | None,
        normalized_weights: dict[str, float],
        model_info: ModelGroupInfo | None = None,
    ) -> tuple[float, str]:
        """
        Get priority weight and pool key for a given priority.

        For explicit priorities: returns specific allocation and unique pool key
        For default priority: returns default allocation and shared pool key

        Args:
            model: Model name
            priority: Priority level (None for default)
            normalized_weights: Pre-computed normalized weights
            model_info: Model configuration (optional, for fallback conversion)

        Returns:
            tuple: (priority_weight, priority_key)
        """
        # Check if this key has an explicit priority in litellm.priority_reservation
        has_explicit_priority: Final = (
            priority is not None
            and litellm.priority_reservation is not None
            and priority in litellm.priority_reservation
        )

        if has_explicit_priority and priority is not None:
            # Explicit priority: get its specific allocation
            priority_weight = normalized_weights.get(priority, self._get_priority_weight(priority, model_info))
            # Use unique key per priority level
            priority_key = f"{model}:{priority}"
        else:
            # No explicit priority: share the default_priority pool with ALL other default keys
            priority_weight = _get_priority_settings().default_priority
            # Use shared key for all default-priority requests
            priority_key = f"{model}:default_pool"

        return priority_weight, priority_key

    async def _check_model_saturation(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
    ) -> float:
        """
        Check current saturation by directly querying v3 limiter's cache keys.

        Reuses v3 limiter's Redis-based tracking (works across multiple instances).
        Reads counters WITHOUT incrementing them.

        Returns:
            float: Saturation ratio (0.0 = empty, 1.0 = at capacity, >1.0 = over)
        """
        try:
            max_saturation = 0.0

            # Query RPM saturation - always read from Redis for multi-node consistency
            if model_group_info.rpm is not None and model_group_info.rpm > 0:
                # Use v3 limiter's key format: {key:value}:rate_limit_type
                counter_key = self.v3_limiter.create_rate_limit_keys(
                    key="model_saturation_check",
                    value=model,
                    rate_limit_type="requests",
                )

                # Query Redis directly for current counter value (skip local cache for consistency)
                counter_value = await self._get_saturation_value_from_cache(counter_key=counter_key)

                if counter_value is not None:
                    current_requests: Final = int(counter_value)
                    rpm_saturation: Final = current_requests / model_group_info.rpm
                    max_saturation = max(max_saturation, rpm_saturation)

                    verbose_proxy_logger.debug(
                        f"Model {model} RPM: {current_requests}/{model_group_info.rpm} ({rpm_saturation:.1%})"
                    )

            # Query TPM saturation
            if model_group_info.tpm is not None and model_group_info.tpm > 0:
                counter_key = self.v3_limiter.create_rate_limit_keys(
                    key="model_saturation_check",
                    value=model,
                    rate_limit_type="tokens",
                )

                counter_value = await self._get_saturation_value_from_cache(counter_key=counter_key)

                if counter_value is not None:
                    current_tokens: Final = float(counter_value)
                    tpm_saturation: Final = current_tokens / model_group_info.tpm
                    max_saturation = max(max_saturation, tpm_saturation)

                    verbose_proxy_logger.debug(
                        f"Model {model} TPM: {current_tokens}/{model_group_info.tpm} ({tpm_saturation:.1%})"
                    )

            verbose_proxy_logger.debug(f"Model {model} overall saturation: {max_saturation:.1%}")

            return max_saturation

        except Exception as e:
            verbose_proxy_logger.error("Error checking saturation for %s: %s", model, e)
            # Fail open: assume not saturated on error
            return 0.0

    def _create_priority_based_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
    ) -> list[RateLimitDescriptor]:
        """
        Create rate limit descriptors with normalized priority weights.

        Uses normalized weights to handle over-allocation scenarios.

        For explicit priorities: each priority gets its own pool (e.g., prod gets 75%)
        For default priority: ALL keys without explicit priority share ONE pool (e.g., all share 25%)
        """
        descriptors: Final[list[RateLimitDescriptor]] = []

        if litellm.priority_reservation is None:
            return descriptors

        if self.llm_router is None:
            return descriptors
        model_group_info: Final[ModelGroupInfo | None] = self.llm_router.get_model_group_info(model_group=model)
        if model_group_info is None:
            return descriptors

        # Get normalized priority weight and pool key
        normalized_weights: Final = self._normalize_priority_weights(model_group_info)
        priority_weight, priority_key = self._get_priority_allocation(
            model=model,
            priority=priority,
            normalized_weights=normalized_weights,
            model_info=model_group_info,
        )

        rate_limit_config: Final[RateLimitDescriptorRateLimitObject] = {}

        # Apply priority weight to model limits
        if model_group_info.tpm is not None:
            reserved_tpm: Final = int(model_group_info.tpm * priority_weight)
            rate_limit_config["tokens_per_unit"] = reserved_tpm

        if model_group_info.rpm is not None:
            reserved_rpm: Final = int(model_group_info.rpm * priority_weight)
            rate_limit_config["requests_per_unit"] = reserved_rpm

        if rate_limit_config:
            rate_limit_config["window_size"] = self.v3_limiter.window_size

            descriptors.append(
                RateLimitDescriptor(
                    key="priority_model",
                    value=priority_key,
                    rate_limit=rate_limit_config,
                )
            )

        return descriptors

    def _create_model_tracking_descriptor(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        high_limit_multiplier: int = 1,
    ) -> RateLimitDescriptor:
        """
        Create a descriptor for tracking model-wide usage.

        Args:
            model: Model name
            model_group_info: Model configuration with RPM/TPM limits
            high_limit_multiplier: Multiplier for limits (use >1 for tracking-only)

        Returns:
            Rate limit descriptor for model-wide tracking
        """
        return RateLimitDescriptor(
            key="model_saturation_check",
            value=model,
            rate_limit={
                "requests_per_unit": (model_group_info.rpm * high_limit_multiplier if model_group_info.rpm else None),
                "tokens_per_unit": (model_group_info.tpm * high_limit_multiplier if model_group_info.tpm else None),
                "window_size": self.v3_limiter.window_size,
            },
        )

    def _build_admission_plan(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        saturation: float,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> _AdmissionPlan:
        should_enforce_priority: Final = saturation >= _get_priority_settings().saturation_threshold
        model_wide_descriptor: Final = self._create_model_tracking_descriptor(
            model=model,
            model_group_info=model_group_info,
            high_limit_multiplier=1,
        )
        priority_descriptors: Final = tuple(
            self._create_priority_based_descriptors(model=model, user_api_key_dict=user_api_key_dict, priority=priority)
        )
        fairness: Final = self._fairness_settings()
        reserve_tokens: Final = fairness is not None and fairness.enabled and model_group_info.tpm is not None
        estimated_tokens: Final = (
            max(
                self.v3_limiter.estimate_tokens_for_request(
                    data, model=model, min_configured_tpm_limit=model_group_info.tpm, call_type=call_type
                ),
                1,
            )
            if reserve_tokens
            else 0
        )
        return _AdmissionPlan(
            model=model,
            class_name=self._class_name_for(priority),
            enforced=(model_wide_descriptor, *priority_descriptors)
            if should_enforce_priority
            else (model_wide_descriptor,),
            tracking_only=() if should_enforce_priority else priority_descriptors,
            estimated_tokens=estimated_tokens,
        )

    async def _try_admit(
        self,
        plan: _AdmissionPlan,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> RateLimitResponse:
        increment: Final[Mapping[Literal["requests", "tokens"], int]] = MappingProxyType(
            {"requests": 1, "tokens": plan.estimated_tokens}
        )
        descriptors: Final = plan.descriptors
        atomic_response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=descriptors,
            increments=tuple(increment for _ in descriptors),
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        verbose_proxy_logger.debug("Atomic check+increment response: %s", atomic_response)
        if not plan.tracking_only:
            return atomic_response
        return RateLimitResponse(
            overall_code=atomic_response["overall_code"],
            statuses=[_with_configured_limit(status, plan.tracking_only) for status in atomic_response["statuses"]],
            reservation_windows=atomic_response.get("reservation_windows", frozenset()),
        )

    def _record_reservation(self, plan: _AdmissionPlan, response: RateLimitResponse) -> None:
        stash: Final = get_or_create_request_stash()
        stash.rate_limit_response = response
        if plan.estimated_tokens <= 0:
            return
        stash.dynamic_reserved_tokens = plan.estimated_tokens
        stash.dynamic_token_scopes = plan.token_scopes
        stash.dynamic_reservation_windows = response.get("reservation_windows", frozenset())
        stash.dynamic_reservation_settled = False

    async def _settle_reservation(
        self,
        stash: RequestRateLimiterStash,
        actual_tokens: int,
        parent_otel_span: "Span | None",
    ) -> bool:
        if stash.dynamic_reserved_tokens <= 0 or stash.dynamic_reservation_settled:
            return False
        stash.dynamic_reservation_settled = True
        try:
            await self.v3_limiter.async_increment_reservation_aware_tokens(
                pipeline_operations=self.v3_limiter.build_project_reservation_ops(
                    targets=sorted(stash.dynamic_token_scopes),
                    reserved_scopes=stash.dynamic_token_scopes,
                    actual_tokens=actual_tokens,
                    reserved_tokens=stash.dynamic_reserved_tokens,
                    reservation_window_identities=stash.dynamic_reservation_windows,
                ),
                parent_otel_span=parent_otel_span,
            )
        except Exception:
            stash.dynamic_reservation_settled = False
            raise
        return True

    def _queue_weights(self, model_group_info: ModelGroupInfo, fairness: FairnessSettings) -> Mapping[str, float]:
        normalized: Final = self._normalize_priority_weights(model_group_info)
        return MappingProxyType(
            {
                name: max(weight, 0.01)
                for name, weight in (*normalized.items(), (DEFAULT_POOL_NAME, fairness.default_reserved_share))
            }
        )

    def _raise_over_limit(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        priority: str | None,
        saturation: float,
        status: RateLimitStatus,
        waited_seconds: float,
    ) -> None:
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
        descriptor_key: Final = status["descriptor_key"]
        headers: Final = MappingProxyType(
            {
                "retry-after": str(self.v3_limiter.window_size),
                "rate_limit_type": str(status["rate_limit_type"]),
                "x-litellm-priority": priority or DEFAULT_POOL_NAME,
                "x-litellm-saturation": f"{saturation:.2%}",
                "x-litellm-fairness-reason": "capacity_exhausted",
                "x-litellm-queue-wait-seconds": f"{waited_seconds:.3f}",
            }
        )
        limits: Final = (
            f"Rate limit type: {status['rate_limit_type']}, "
            f"Model TPM: {model_group_info.tpm if model_group_info.tpm is not None else 'not configured'}, "
            f"Model RPM: {model_group_info.rpm if model_group_info.rpm is not None else 'not configured'}, "
            f"Remaining: {status['limit_remaining']}"
        )
        if descriptor_key not in ("model_saturation_check", "priority_model"):
            verbose_proxy_logger.error(
                "Dynamic rate limiter: OVER_LIMIT for unknown descriptor_key %s, refusing request", descriptor_key
            )
        error: Final = (
            f"Model capacity reached for {model}. Priority: {priority}, {limits}"
            if descriptor_key == "model_saturation_check"
            else f"Priority-based rate limit exceeded. Model: {model}, Priority: {priority}, {limits}, "
            f"Model saturation: {saturation:.1%}"
            if descriptor_key == "priority_model"
            else "Rate limit exceeded"
        )
        raise ProxyRateLimitError(
            detail={  # mutable-ok: one-shot HTTPException detail payload, never mutated after construction
                "error": error,
                "fairness_reason": "capacity_exhausted",
                "queue_wait_seconds": waited_seconds,
            },
            headers=headers,
            rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
            model=resolved_model,
            llm_provider=llm_provider,
        )

    def _raise_queue_rejected(
        self,
        model: str,
        priority: str | None,
        reason: FairnessRejectReason,
        waited_seconds: float,
        max_wait_seconds: float,
        status: RateLimitStatus,
    ) -> None:
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
        raise ProxyRateLimitError(
            detail={  # mutable-ok: one-shot HTTPException detail payload, never mutated after construction
                "error": f"Request for {model} could not be admitted within {max_wait_seconds:.1f}s ({reason}). "
                f"Priority: {priority or DEFAULT_POOL_NAME}, waited {waited_seconds:.1f}s",
                "fairness_reason": reason,
                "queue_wait_seconds": waited_seconds,
            },
            headers=MappingProxyType(
                {
                    "retry-after": str(self.v3_limiter.window_size),
                    "rate_limit_type": str(status["rate_limit_type"]),
                    "x-litellm-priority": priority or DEFAULT_POOL_NAME,
                    "x-litellm-fairness-reason": reason,
                    "x-litellm-queue-wait-seconds": f"{waited_seconds:.3f}",
                }
            ),
            rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
            model=resolved_model,
            llm_provider=llm_provider,
        )

    @staticmethod
    def _first_over_limit(response: RateLimitResponse) -> RateLimitStatus:
        offending: Final = next((s for s in response["statuses"] if s["code"] == "OVER_LIMIT"), None)
        if offending is None:
            return RateLimitStatus(
                code="OVER_LIMIT",
                current_limit=0,
                limit_remaining=0,
                rate_limit_type="requests",
                descriptor_key="unknown",
            )
        return offending

    async def _check_rate_limits(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        saturation: float,
        data: dict[str, object] | None = None,
        call_type: CallTypesLiteral = "completion",
    ) -> None:
        request: Final[Mapping[str, object]] = data if data is not None else MappingProxyType({})
        plan: Final = self._build_admission_plan(
            model, model_group_info, user_api_key_dict, priority, saturation, request, call_type
        )
        first_response: Final = await self._try_admit(plan, user_api_key_dict)
        if first_response["overall_code"] != "OVER_LIMIT":
            self._record_reservation(plan, first_response)
            return

        over_limit: Final = self._first_over_limit(first_response)
        fairness: Final = self._fairness_settings()
        requested_wait: Final = _requested_max_queue_wait(request)
        class_wait: Final = fairness.max_queue_wait_for(plan.class_name) if fairness is not None else 0.0
        max_wait: Final = (
            min(requested_wait, class_wait) if requested_wait is not None and requested_wait >= 0.0 else class_wait
        )
        if fairness is None or not fairness.enabled or max_wait <= 0.0:
            await self.fairness_stats.record(model, plan.class_name, "rejected_capacity")
            self._raise_over_limit(model, model_group_info, priority, saturation, over_limit, 0.0)
            return

        async def try_admit() -> bool:
            live_saturation: Final = await self._check_model_saturation(model, model_group_info)
            live_plan: Final = self._build_admission_plan(
                model, model_group_info, user_api_key_dict, priority, live_saturation, request, call_type
            )
            response: Final = await self._try_admit(live_plan, user_api_key_dict)
            if response["overall_code"] == "OVER_LIMIT":
                return False
            self._record_reservation(live_plan, response)
            return True

        call_id: Final = request.get("litellm_call_id")
        ticket: Final = QueueTicket(
            model=model,
            class_name=plan.class_name,
            request_id=call_id if isinstance(call_id, str) else f"{id(request)}",
        )
        await self.fairness_stats.record(model, plan.class_name, "queued")
        outcome: Final = await self.fair_queue.wait_for_admission(
            ticket=ticket,
            weights=self._queue_weights(model_group_info, fairness),
            max_wait_seconds=max_wait,
            max_depth=fairness.max_queue_depth_per_class,
            poll_interval_seconds=fairness.queue_poll_interval_seconds,
            try_admit=try_admit,
            is_cancelled=self.is_client_disconnected,
            is_released=lambda: not self.enforcing,
        )
        match outcome:
            case QueueAdmitted(waited_seconds=waited):
                get_or_create_request_stash().fairness_queue_wait_seconds = waited
                await self.fairness_stats.record(model, plan.class_name, "admitted_after_wait")
                await self.fairness_stats.record(model, plan.class_name, "wait_seconds_sum", waited)
            case QueueReleased(waited_seconds=waited):
                released_stash: Final = get_or_create_request_stash()
                released_stash.fairness_queue_wait_seconds = waited
                released_stash.dynamic_admission_bypassed = True
                await self.fairness_stats.record(model, plan.class_name, "admitted_after_wait")
                await self.fairness_stats.record(model, plan.class_name, "wait_seconds_sum", waited)
            case QueueRejected(reason=reason, waited_seconds=waited):
                await self.fairness_stats.record(model, plan.class_name, _REJECT_METRIC[reason])
                await self.fairness_stats.record(model, plan.class_name, "wait_seconds_sum", waited)
                self._raise_queue_rejected(model, priority, reason, waited, max_wait, over_limit)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        """
        Saturation-aware pre-call hook for priority-based rate limiting.

        Flow:
        1. Check current saturation level
        2. THREE-PHASE rate limit check:
           - PHASE 1: Read-only check of ALL limits (no increments)
           - PHASE 2: Decide which limits to enforce based on saturation
           - PHASE 3: Increment ALL counters atomically if request allowed

        This three-phase approach ensures:
        - Model capacity is NEVER exceeded (always enforced at 100%)
        - Priority usage tracked from first request (accurate metrics)
        - Counters only increment when request will be allowed (prevents phantom usage)
        - When under-saturated: priorities can borrow unused capacity (generous)
        - When saturated: fair allocation based on normalized priority weights (strict)

        Example with 100 RPM model, 60% priority allocation, 80% threshold:
        - Saturation < 80%: Priority can use up to 100 RPM (model limit enforced only)
        - Saturation >= 80%: Priority limited to 60 RPM (both limits enforced)

        Prevents bugs where:
        - Model counter increments but priority check fails → model over-capacity
        - Priority counter increments but not enforced → inaccurate metrics

        Args:
            user_api_key_dict: User authentication and metadata
            cache: Dual cache instance
            data: Request data containing model name
            call_type: Type of API call being made

        Returns:
            None if request is allowed, otherwise raises HTTPException
        """
        if "model" not in data or self.llm_router is None:
            return None

        stash: Final = claim_request_stash_for_data(data)
        if not self.enforcing:
            stash.dynamic_admission_bypassed = True
            return None
        model: Final = data["model"]
        priority: Final = self._get_priority_from_user_api_key_dict(user_api_key_dict=user_api_key_dict)

        # Get model configuration
        model_group_info: Final[ModelGroupInfo | None] = self.llm_router.get_model_group_info(model_group=model)
        if model_group_info is None:
            verbose_proxy_logger.debug("No model group info for %s, allowing request", model)
            return None

        try:
            # STEP 1: Check current saturation level
            saturation: Final = await self._check_model_saturation(model, model_group_info)

            saturation_threshold: Final = _get_priority_settings().saturation_threshold

            verbose_proxy_logger.debug(
                f"[Dynamic Rate Limiter] Model={model}, Saturation={saturation:.1%}, "
                f"Threshold={saturation_threshold:.1%}, Priority={priority}"
            )

            # STEP 2: Check rate limits in THREE phases
            # Phase 1: Read-only check of ALL limits (no increments)
            # Phase 2: Decide which limits to enforce (based on saturation)
            # Phase 3: Increment ALL counters only if request will be allowed
            # This prevents partial increments and ensures accurate tracking
            await self._check_rate_limits(
                model=model,
                model_group_info=model_group_info,
                user_api_key_dict=user_api_key_dict,
                priority=priority,
                saturation=saturation,
                data=data,
                call_type=call_type,
            )

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Error in dynamic rate limiter: %s, allowing request", e)
            # Fail open on unexpected errors
            return None

        return None

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response):
        """
        Post-call hook to add rate limit headers to response.
        Leverages v3 limiter's post-call hook functionality.
        """
        try:
            # Call v3 limiter's post-call hook to add standard rate limit headers
            await self.v3_limiter.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )

            if response_has_hidden_params(response):
                priority: Final = self._get_priority_from_user_api_key_dict(user_api_key_dict=user_api_key_dict)
                additional_headers: Final = ensure_response_additional_headers(response)
                priority_header: Final = priority or "default"
                if _is_latin1_encodable(priority_header):
                    additional_headers["x-litellm-priority"] = priority_header
                else:
                    verbose_proxy_logger.debug(
                        "Skipping x-litellm-priority header: priority %r is not Latin-1 encodable", priority
                    )
                additional_headers["x-litellm-rate-limiter-version"] = "v3"

            return response

        except Exception as e:
            verbose_proxy_logger.exception("Error in dynamic rate limiter v3 post-call hook: %s", e)
            return response

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Update token usage for priority-based rate limiting after successful API calls.

        Increments token counters for:
        - model_saturation_check: Model-wide token tracking
        - priority_model: Priority-specific token tracking
        """
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        try:
            verbose_proxy_logger.debug("INSIDE dynamic rate limiter ASYNC SUCCESS LOGGING")

            litellm_parent_otel_span: Final = _get_parent_otel_span_from_kwargs(kwargs)

            # Get metadata from standard_logging_object
            standard_logging_object: Final = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata: Final = standard_logging_object.get("metadata") or {}

            # Get model and priority
            model_group: Final = get_model_group_from_litellm_kwargs(kwargs)
            if not model_group:
                return

            # Get priority from user_api_key_auth_metadata in standard_logging_metadata
            # This is where user_api_key_dict.metadata is stored during pre-call
            user_api_key_auth_metadata: Final = standard_logging_metadata.get("user_api_key_auth_metadata") or {}
            key_priority: Final[str | None] = user_api_key_auth_metadata.get("priority")

            total_tokens: Final = self.v3_limiter.success_usage_tokens(
                kwargs, response_obj, self.v3_limiter.get_rate_limit_type()
            )

            stash: Final = get_request_stash_for_call(call_id_from_callback_kwargs(kwargs))
            if stash is not None and await self._settle_reservation(stash, total_tokens, litellm_parent_otel_span):
                return

            bypassed: Final = stash.dynamic_admission_bypassed if stash is not None else not self.enforcing
            if total_tokens == 0 or bypassed:
                return

            # Create pipeline operations for token increments
            pipeline_operations: Final[list[RedisPipelineIncrementOperation]] = []

            # Model-wide token tracking (model_saturation_check)
            model_token_key: Final = self.v3_limiter.create_rate_limit_keys(
                key="model_saturation_check",
                value=model_group,
                rate_limit_type="tokens",
            )
            pipeline_operations.append(
                RedisPipelineIncrementOperation(
                    key=model_token_key,
                    increment_value=total_tokens,
                    ttl=self.v3_limiter.window_size,
                )
            )

            # Priority-specific token tracking (priority_model)
            # Determine priority key (same logic as _get_priority_allocation)
            has_explicit_priority: Final = (
                key_priority is not None
                and litellm.priority_reservation is not None
                and key_priority in litellm.priority_reservation
            )

            if has_explicit_priority and key_priority is not None:
                priority_key = f"{model_group}:{key_priority}"
            else:
                priority_key = f"{model_group}:default_pool"

            priority_token_key: Final = self.v3_limiter.create_rate_limit_keys(
                key="priority_model",
                value=priority_key,
                rate_limit_type="tokens",
            )
            pipeline_operations.append(
                RedisPipelineIncrementOperation(
                    key=priority_token_key,
                    increment_value=total_tokens,
                    ttl=self.v3_limiter.window_size,
                )
            )

            # Execute token increments with TTL preservation
            if pipeline_operations:
                await self.v3_limiter.async_increment_tokens_with_ttl_preservation(
                    pipeline_operations=pipeline_operations,
                    parent_otel_span=litellm_parent_otel_span,
                )

                # Only log 'priority' if it's known safe; otherwise, redact.
                SAFE_PRIORITIES: Final = {"low", "medium", "high", "default"}
                logged_priority: Final = key_priority if key_priority in SAFE_PRIORITIES else "REDACTED"
                verbose_proxy_logger.debug(
                    "[Dynamic Rate Limiter] Incremented tokens by %s for model=%s, priority=%s",
                    total_tokens,
                    model_group,
                    logged_priority,
                )

        except Exception as e:
            verbose_proxy_logger.exception("Error in dynamic rate limiter success event: %s", e)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> None:
        stash: Final = get_request_stash_for_call(call_id_from_callback_kwargs(kwargs))
        if stash is None:
            return
        recovered_tokens: Final = self.v3_limiter.recovered_partial_usage_tokens(kwargs)[0]
        try:
            await self._settle_reservation(stash, recovered_tokens, None)
        except Exception as e:
            verbose_proxy_logger.exception("Error refunding dynamic rate limiter reservation: %s", e)

    async def fairness_status(self, models: Sequence[str]) -> tuple[ModelFairnessStatus, ...]:
        router: Final = self.llm_router
        if router is None:
            return ()
        known: Final = tuple(
            (model, info) for model in models if (info := router.get_model_group_info(model_group=model)) is not None
        )
        return tuple(await asyncio.gather(*(self._model_fairness_status(model, info) for model, info in known)))

    async def _model_fairness_status(self, model: str, info: ModelGroupInfo) -> ModelFairnessStatus:
        fairness: Final = self._fairness_settings()
        reserved_names: Final = tuple(litellm.priority_reservation or ())
        class_names: Final = (*reserved_names, DEFAULT_POOL_NAME)
        weights: Final = self._normalize_priority_weights(info)
        default_share: Final = _get_priority_settings().default_priority
        depths: Final = self.fair_queue.depths(model)
        stats: Final = await self.fairness_stats.read(model, class_names)
        counter_pools: Final = (
            ("model_saturation_check", model),
            *(("priority_model", f"{model}:{name}") for name in reserved_names),
            ("priority_model", f"{model}:default_pool"),
        )
        counter_keys: Final = tuple(
            self.v3_limiter.create_rate_limit_keys(key=key, value=value, rate_limit_type=rate_limit_type)
            for (key, value), rate_limit_type in product(counter_pools, ("requests", "tokens"))
        )
        raw_counters: Final = _RAW_VALUES_ADAPTER.validate_python(
            await self.internal_usage_cache.async_batch_get_cache(keys=counter_keys) or ()
        )
        counters: Final = tuple(_counter_value(raw) for raw in raw_counters) + (0.0,) * (
            len(counter_keys) - len(raw_counters)
        )
        rpm_saturation: Final = counters[0] / info.rpm if info.rpm else 0.0
        tpm_saturation: Final = counters[1] / info.tpm if info.tpm else 0.0
        saturation: Final = max(rpm_saturation, tpm_saturation)

        def class_status(index: int, name: str) -> WorkloadClassStatus:
            share: Final = weights.get(name, default_share)
            class_stats: Final = stats[name]
            return WorkloadClassStatus(
                name=name,
                reserved_share=share,
                reserved_rpm=int(info.rpm * share) if info.rpm is not None else None,
                reserved_tpm=int(info.tpm * share) if info.tpm is not None else None,
                current_requests=int(counters[2 + 2 * index]),
                current_tokens=int(counters[3 + 2 * index]),
                queue_depth=depths.get(name, 0),
                max_queue_wait_seconds=fairness.max_queue_wait_for(name) if fairness is not None else 0.0,
                queued_total=class_stats.queued,
                admitted_after_wait_total=class_stats.admitted_after_wait,
                rejected_capacity_total=class_stats.rejected_capacity,
                rejected_queue_full_total=class_stats.rejected_queue_full,
                rejected_deadline_total=class_stats.rejected_deadline,
                disconnected_total=class_stats.disconnected,
                avg_queue_wait_seconds=class_stats.avg_queue_wait_seconds,
            )

        return ModelFairnessStatus(
            model_group=model,
            rpm=info.rpm,
            tpm=info.tpm,
            saturation=saturation,
            enforcing_reservations=bool(reserved_names) and saturation >= _get_priority_settings().saturation_threshold,
            current_requests=int(counters[0]),
            current_tokens=int(counters[1]),
            classes=tuple(class_status(index, name) for index, name in enumerate(class_names)),
        )
