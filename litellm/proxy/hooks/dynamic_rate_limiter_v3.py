"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

import litellm
from litellm import Router
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.cost_calculator import get_usage_object
from litellm.exceptions import ServiceUnavailableError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (  # noqa: F401  # legacy module exports
    RATE_LIMIT_UNVERIFIABLE_MESSAGE,
    BorrowingTeamModelHold,
    BorrowingTeamModelLimits,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    RateLimitedModel,
    RateLimitResponse,
    RateLimitStatus,
    RateLimitUnverifiableError,
    RequestRateLimiterStash,
    ReservationOperation,
    ReservationWindowIdentity,
    WindowGuardedIncrement,
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _socket_turn_tpm_windows,  # pyright: ignore[reportPrivateUsage]  # same per-turn TPM window the regular limiter checks
    borrowing_team_model_limits,
    callback_kwargs_mapping,
    claim_request_stash_for_data,
    get_or_create_request_stash,
    request_stash_for_callback,
)
from litellm.proxy.hooks.rate_limiter_utils import (
    convert_priority_to_percent,
    resolve_llm_provider_for_rate_limit,
    socket_session_usage,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET, CounterIncrement, RateLimitCounter, RateLimitScope
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes, Usage

if TYPE_CHECKING:
    from litellm.types.utils import PriorityReservationSettings


class _SettlementMetadata(BaseModel):
    """The two fields of a request's standard logging metadata that token settlement reads."""

    model_config = ConfigDict(frozen=True)

    user_api_key_auth_metadata: Mapping[str, object] | None = None
    user_api_key_team_id: str | None = None

    @property
    def priority(self) -> str | None:
        value: Final = (self.user_api_key_auth_metadata or {}).get("priority")
        return value if isinstance(value, str) else None


class _SettlementLoggingObject(BaseModel):
    """The slice of a request's standard logging object that token settlement reads."""

    model_config = ConfigDict(frozen=True)

    metadata: _SettlementMetadata | None = None


def _usage_of(kwargs: Mapping[str, object], response_obj: object) -> Usage | None:
    """Usage of a modeled response, a Responses WebSocket turn's combined usage, or its raw event list."""
    usage: Final = getattr(response_obj, "usage", None)
    if isinstance(usage, Usage):
        return usage
    combined_usage: Final = kwargs.get("combined_usage_object")
    if isinstance(combined_usage, Usage):
        return combined_usage
    socket_usage: Final = socket_session_usage(kwargs, response_obj)
    return socket_usage if socket_usage is not None else get_usage_object(response_obj)


def _settled_tokens(usage: Usage | None, rate_limit_type: str) -> int:
    if usage is None:
        return 0
    match rate_limit_type:
        case "output":
            return usage.completion_tokens
        case "input":
            return usage.prompt_tokens
        case _:
            return usage.total_tokens


TEAM_MODEL_LIMIT_AT_SATURATION: Final = "model_per_team_saturated"
TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE: Final = "model_per_team_max_borrow_multiple"


@dataclass(frozen=True, slots=True)
class _BorrowingTeamShare:
    """The team's share and, when the team caps how far it borrows, its ceiling, both on the share's counter."""

    limits: BorrowingTeamModelLimits
    share: RateLimitDescriptor
    ceiling: RateLimitDescriptor | None

    def enforced(self, saturated: bool) -> RateLimitDescriptor | None:
        return self.share if saturated else self.ceiling


def _team_limit_name(rate_limit_type: str) -> str:
    return "model_rpm_limit" if rate_limit_type == "requests" else "model_tpm_limit"


def _team_model_token_increment(
    descriptor: RateLimitDescriptor | None, amount: int, ttl: int
) -> tuple[CounterIncrement, ...]:
    """Move the team counter the pre-call hook tracked by ``amount``, and only when that team limit has a TPM share."""
    if descriptor is None or amount == 0:
        return ()
    rate_limit: Final = descriptor["rate_limit"]
    if rate_limit is None or rate_limit.get("tokens_per_unit") is None:
        return ()
    return (
        CounterIncrement(
            counter=RateLimitScope(descriptor["key"], descriptor["value"]).counter("tokens"),
            amount=amount,
            ttl=ttl,
        ),
    )


def _window_guarded(
    operations: tuple[CounterIncrement, ...], window: ReservationWindowIdentity | None
) -> tuple[ReservationOperation, ...]:
    """Apply a reservation's settlement only while the window it was held in is still open, never to a newer window."""
    if window is None:
        return operations
    return tuple(
        WindowGuardedIncrement(increment=op, expected_window_start=window[1], reservation_backend=window[2])
        if op.counter == window[0]
        else op
        for op in operations
    )


def _distinct_fallback_holds(stash: RequestRateLimiterStash) -> tuple[BorrowingTeamModelHold, ...]:
    """Each fallback hold once, though the stash keys it by both the requested name and the resolved group."""
    return tuple(
        {hold.descriptor["value"]: hold for hold in stash.borrowing_team_model_fallback_limits.values()}.values()
    )


def _hold_refunds(holds: tuple[BorrowingTeamModelHold, ...], ttl: int) -> tuple[ReservationOperation, ...]:
    return tuple(
        chain.from_iterable(
            _window_guarded(_team_model_token_increment(hold.descriptor, -hold.reserved_tokens, ttl), hold.window)
            for hold in holds
        )
    )


def _team_model_settlement(
    stash: RequestRateLimiterStash | None, model_group: str | None, total_tokens: int, ttl: int
) -> tuple[ReservationOperation, ...]:
    """Settle the counter that served the call to actual usage and give back every other hold the call took.

    The admitting team counter serves unless a fallback group did, in which case that fallback's hold serves and
    the admitting reservation is returned. Holds on fallback targets that did not serve are returned too.
    """
    if stash is None:
        return ()
    served: Final = stash.borrowing_team_model_fallback_limits.get(model_group) if model_group is not None else None
    others: Final = tuple(hold for hold in _distinct_fallback_holds(stash) if hold is not served)
    admitted: Final = stash.borrowing_team_model_limit
    reserved_tokens: Final = stash.borrowing_team_model_reserved_tokens
    window: Final = stash.borrowing_team_model_reserved_window
    if served is None:
        return (
            *_window_guarded(_team_model_token_increment(admitted, total_tokens - reserved_tokens, ttl), window),
            *_hold_refunds(others, ttl),
        )
    return (
        *_window_guarded(_team_model_token_increment(admitted, -reserved_tokens, ttl), window),
        *_window_guarded(
            _team_model_token_increment(served.descriptor, total_tokens - served.reserved_tokens, ttl), served.window
        ),
        *_hold_refunds(others, ttl),
    )


def _reserved_window(
    response: RateLimitResponse, descriptor: RateLimitDescriptor | None
) -> ReservationWindowIdentity | None:
    if descriptor is None:
        return None
    counter: Final = RateLimitScope(descriptor["key"], descriptor["value"]).counter("tokens")
    return next((w for w in response.get("reservation_windows", frozenset()) if w[0] == counter), None)


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
    ):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache, time_provider=time_provider)

    def update_variables(self, llm_router: Router) -> None:
        self.llm_router = llm_router

    def _get_saturation_check_cache_ttl(self) -> int:
        """Get the configurable TTL for local cache when reading saturation values."""
        return _get_priority_settings().saturation_check_cache_ttl

    async def _get_saturation_value_from_cache(
        self,
        counter: RateLimitCounter,
    ) -> int | float | str | None:
        """
        Get saturation value with configurable local cache TTL.

        Uses DualCache with configurable TTL for local cache storage.
        TTL is configurable via litellm.priority_reservation_settings.saturation_check_cache_ttl

        Args:
            counter: The saturation counter

        Returns:
            Counter value, or None if not found
        """
        value: Final = await self.v3_limiter.rate_limits.get(
            counter, backfill_ttl=self._get_saturation_check_cache_ttl()
        )
        return None if isinstance(value, dict) else value

    def _get_priority_weight(self, priority: str | None, model_info: ModelGroupInfo | None = None) -> float:
        """Get the weight for a given priority from litellm.priority_reservation"""
        weight: float = _get_priority_settings().default_priority
        if litellm.priority_reservation is None or priority not in litellm.priority_reservation:
            verbose_proxy_logger.debug("Priority Reservation not set for the given priority.")
        elif priority is not None and litellm.priority_reservation is not None:
            if os.getenv("LITELLM_LICENSE", None) is None:
                verbose_proxy_logger.error(
                    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\nGet a license: https://docs.litellm.ai/docs/enterprise."
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
                # Query Redis directly for current counter value (skip local cache for consistency)
                counter_value = await self._get_saturation_value_from_cache(
                    RateLimitScope("model_saturation_check", model).counter("requests")
                )

                if counter_value is not None:
                    current_requests: Final = int(counter_value)
                    rpm_saturation: Final = current_requests / model_group_info.rpm
                    max_saturation = max(max_saturation, rpm_saturation)

                    verbose_proxy_logger.debug(
                        f"Model {model} RPM: {current_requests}/{model_group_info.rpm} ({rpm_saturation:.1%})"
                    )

            # Query TPM saturation
            if model_group_info.tpm is not None and model_group_info.tpm > 0:
                counter_value = await self._get_saturation_value_from_cache(
                    RateLimitScope("model_saturation_check", model).counter("tokens")
                )

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

        # Get model group info
        model_group_info: Final[ModelGroupInfo | None] = self.llm_router.registry().group(model)
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

    def _borrowing_team_share(self, model: str, user_api_key_dict: UserAPIKeyAuth) -> _BorrowingTeamShare | None:
        """One counter per team and resolved model group, so an alias and its target share the team's share."""
        rate_limited: Final = self.v3_limiter.rate_limited_model(model) or RateLimitedModel(
            requested=model, group=model
        )
        limits: Final = borrowing_team_model_limits(user_api_key_dict, rate_limited)
        if limits.tpm is None and limits.rpm is None:
            return None
        ceiling: Final = limits.borrow_ceiling()
        value: Final = f"{user_api_key_dict.team_id}:{rate_limited.group}"
        return _BorrowingTeamShare(
            limits=limits,
            share=self._team_share_descriptor(value, limits),
            ceiling=None if ceiling is None else self._team_share_descriptor(value, ceiling),
        )

    def _team_share_descriptor(self, value: str, limits: BorrowingTeamModelLimits) -> RateLimitDescriptor:
        return RateLimitDescriptor(
            key=TEAM_MODEL_LIMIT_AT_SATURATION,
            value=value,
            rate_limit={
                "tokens_per_unit": limits.tpm,
                "requests_per_unit": limits.rpm,
                "window_size": self.v3_limiter.window_size,
            },
        )

    def _team_share_error(
        self,
        model: str,
        status: RateLimitStatus,
        saturation: float,
        priority: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        team_share: _BorrowingTeamShare,
    ) -> ProxyRateLimitError:
        if saturation >= _get_priority_settings().saturation_threshold:
            return self._team_limit_error(model, status, saturation, priority, user_api_key_dict)
        return self._team_borrow_ceiling_error(
            model, status, saturation, priority, user_api_key_dict, team_share.limits
        )

    def _team_borrow_ceiling_error(
        self,
        model: str,
        status: RateLimitStatus,
        saturation: float,
        priority: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        limits: BorrowingTeamModelLimits,
    ) -> ProxyRateLimitError:
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
        limit_name: Final = _team_limit_name(status["rate_limit_type"])
        return ProxyRateLimitError(
            detail={
                "error": f"Team {limit_name} borrow cap reached for {model}: the team may use at most "
                f"{limits.max_borrow_multiple}x its {limit_name} while borrowing idle capacity. "
                f"Team: {user_api_key_dict.team_id}, "
                f"Rate limit type: {status['rate_limit_type']}, "
                f"Team {limit_name}: {limits.share(status['rate_limit_type'])}, "
                f"Borrow cap: {status['current_limit']}, "
                f"Remaining: {status['limit_remaining']}, "
                f"Model saturation: {saturation:.1%}"
            },
            headers={
                "retry-after": str(self.v3_limiter.window_size),
                "rate_limit_type": str(status["rate_limit_type"]),
                "rate_limited_by": TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE,
                "x-litellm-priority": priority or "default",
                "x-litellm-saturation": f"{saturation:.2%}",
            },
            rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
            model=resolved_model,
            llm_provider=llm_provider,
        )

    def _team_limit_error(
        self,
        model: str,
        status: RateLimitStatus,
        saturation: float,
        priority: str | None,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> ProxyRateLimitError:
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
        return ProxyRateLimitError(
            detail={
                "error": f"Team {_team_limit_name(status['rate_limit_type'])} reached for {model} "
                "and the model is saturated, so there is no idle capacity to borrow. "
                f"Team: {user_api_key_dict.team_id}, "
                f"Rate limit type: {status['rate_limit_type']}, "
                f"Team {_team_limit_name(status['rate_limit_type'])}: {status['current_limit']}, "
                f"Remaining: {status['limit_remaining']}, "
                f"Model saturation: {saturation:.1%}"
            },
            headers={
                "retry-after": str(self.v3_limiter.window_size),
                "rate_limit_type": str(status["rate_limit_type"]),
                "rate_limited_by": TEAM_MODEL_LIMIT_AT_SATURATION,
                "x-litellm-priority": priority or "default",
                "x-litellm-saturation": f"{saturation:.2%}",
            },
            rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
            model=resolved_model,
            llm_provider=llm_provider,
        )

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        """Hold a Responses WebSocket turn to the team's share once the model saturates, like the regular limiter does."""
        if requested_model is None:
            return
        team_share: Final = self._borrowing_team_share(model=requested_model, user_api_key_dict=user_api_key_dict)
        model_group_info: Final = self.llm_router.registry().group(requested_model)
        if team_share is None or model_group_info is None:
            return
        saturation: Final = await self._check_model_saturation(requested_model, model_group_info)
        enforced: Final = team_share.enforced(saturation >= _get_priority_settings().saturation_threshold)
        if enforced is None:
            return
        tpm_windows: Final = _socket_turn_tpm_windows((enforced,))
        try:
            response: Final = await self.v3_limiter.should_rate_limit(
                tpm_windows,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                read_only=True,
                skip_parallel_check=True,
            )
        except RateLimitUnverifiableError as unverifiable:
            raise ServiceUnavailableError(
                message=RATE_LIMIT_UNVERIFIABLE_MESSAGE, llm_provider="litellm_proxy", model=requested_model
            ) from unverifiable
        over_limit: Final = next((s for s in response["statuses"] if s["code"] == "OVER_LIMIT"), None)
        if over_limit is None:
            return
        raise self._team_share_error(
            requested_model,
            over_limit,
            saturation,
            self._get_priority_from_user_api_key_dict(user_api_key_dict),
            user_api_key_dict,
            team_share,
        )

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

    async def _check_rate_limits(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        saturation: float,
        data: Mapping[str, object] = MappingProxyType({}),
        call_type: str | None = None,
    ) -> None:
        """
        Check rate limits using THREE-PHASE approach to prevent partial increments.

        Phase 1: Read-only check of ALL limits (no increments)
        Phase 2: Decide which limits to enforce based on saturation
        Phase 3: Increment ALL counters atomically (model + priority)

        This prevents the bug where:
        - Model counter increments in stage 1
        - Priority check fails in stage 2
        - Request blocked but model counter already incremented

        Key behaviors:
        - All checks performed first (read-only)
        - Only increment counters if request will be allowed
        - Model capacity: Always enforced at 100%
        - Priority limits: Only enforced when saturated >= threshold
        - Both counters tracked from first request (accurate accounting)

        Args:
            model: Model name
            model_group_info: Model configuration
            user_api_key_dict: User authentication info
            priority: User's priority level
            saturation: Current saturation level
            data: Request body, used to size the token reservation a saturated team limit holds
            call_type: The endpoint's call type, used to read the body's prompt fields

        Raises:
            HTTPException: If any limit is exceeded
        """

        saturation_threshold: Final = _get_priority_settings().saturation_threshold
        should_enforce_priority: Final = saturation >= saturation_threshold

        # Build ALL descriptors upfront
        descriptors_to_check: Final[list[RateLimitDescriptor]] = []

        # Model-wide descriptor (always enforce)
        model_wide_descriptor: Final = self._create_model_tracking_descriptor(
            model=model,
            model_group_info=model_group_info,
            high_limit_multiplier=1,
        )
        descriptors_to_check.append(model_wide_descriptor)

        # Priority descriptors (always track, conditionally enforce)
        priority_descriptors: Final = self._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user_api_key_dict,
            priority=priority,
        )
        if priority_descriptors:
            descriptors_to_check.extend(priority_descriptors)

        # Atomic check-and-increment for the ENFORCED descriptor set:
        # - model_saturation_check is always enforced
        # - priority_model is enforced only when saturation crosses threshold
        #
        # Backed by a Redis Lua script (multi-process atomic) with an
        # asyncio.Lock + in-memory fallback for single-process deployments.
        # All-or-nothing: if any enforced descriptor would exceed its limit,
        # no counter is modified and the response carries "OVER_LIMIT".
        team_share: Final = self._borrowing_team_share(model=model, user_api_key_dict=user_api_key_dict)
        get_or_create_request_stash().borrowing_team_model_limit = None if team_share is None else team_share.share
        if team_share is not None:
            descriptors_to_check.append(team_share.share)
        enforced_team_share: Final = None if team_share is None else team_share.enforced(should_enforce_priority)

        enforced_descriptors: Final[list[RateLimitDescriptor]] = [model_wide_descriptor]
        if priority_descriptors and should_enforce_priority:
            enforced_descriptors.extend(priority_descriptors)
        if enforced_team_share is not None:
            enforced_descriptors.append(enforced_team_share)

        per_request_increment: Final[dict[Literal["requests", "tokens"], int]] = {
            "requests": 1,
            "tokens": 0,
        }
        team_reservation: Final = (
            await self._team_token_reservation(enforced_team_share, data, model, call_type, user_api_key_dict)
            if enforced_team_share is not None
            else 0
        )
        atomic_response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=enforced_descriptors,
            increments=[
                {"requests": 1, "tokens": team_reservation}
                if descriptor is enforced_team_share
                else per_request_increment
                for descriptor in enforced_descriptors
            ],
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        verbose_proxy_logger.debug("Atomic check+increment response: %s", atomic_response)

        if atomic_response["overall_code"] == "OVER_LIMIT":
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
            for status in atomic_response["statuses"]:
                if status["code"] != "OVER_LIMIT":
                    continue
                descriptor_key = status["descriptor_key"]
                if descriptor_key == "model_saturation_check":
                    raise ProxyRateLimitError(
                        detail={
                            "error": f"Model capacity reached for {model}. "
                            f"Priority: {priority}, "
                            f"Rate limit type: {status['rate_limit_type']}, "
                            f"Model TPM: {model_group_info.tpm if model_group_info.tpm is not None else 'not configured'}, "
                            f"Model RPM: {model_group_info.rpm if model_group_info.rpm is not None else 'not configured'}, "
                            f"Remaining: {status['limit_remaining']}"
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                        },
                        rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                        model=resolved_model,
                        llm_provider=llm_provider,
                    )
                if descriptor_key == "priority_model":
                    verbose_proxy_logger.debug(
                        f"Enforcing priority limits for {model}, saturation: {saturation:.1%}, priority: {priority}"
                    )
                    raise ProxyRateLimitError(
                        detail={
                            "error": f"Priority-based rate limit exceeded. "
                            f"Model: {model}, "
                            f"Priority: {priority}, "
                            f"Rate limit type: {status['rate_limit_type']}, "
                            f"Model TPM: {model_group_info.tpm if model_group_info.tpm is not None else 'not configured'}, "
                            f"Model RPM: {model_group_info.rpm if model_group_info.rpm is not None else 'not configured'}, "
                            f"Remaining: {status['limit_remaining']}, "
                            f"Model saturation: {saturation:.1%}"
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                            "x-litellm-saturation": f"{saturation:.2%}",
                        },
                        rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                        model=resolved_model,
                        llm_provider=llm_provider,
                    )

                if descriptor_key == TEAM_MODEL_LIMIT_AT_SATURATION and team_share is not None:
                    raise self._team_share_error(model, status, saturation, priority, user_api_key_dict, team_share)

            # Fail-closed guard: overall_code says OVER_LIMIT but no status
            # matched a descriptor key we know how to translate into a 429.
            # Refuse the request rather than silently fall through and let an
            # over-limit request proceed to the model. Without this, a future
            # caller wiring an unfamiliar descriptor into enforced_descriptors
            # would silently bypass the rate limit.
            offending: Final = next(
                (s for s in atomic_response["statuses"] if s["code"] == "OVER_LIMIT"),
                None,
            )
            verbose_proxy_logger.error(
                "Dynamic rate limiter: OVER_LIMIT response with unknown descriptor_key(s) — refusing request. response=%s",
                atomic_response,
            )
            raise ProxyRateLimitError(
                detail={
                    "error": "Rate limit exceeded",
                    "descriptor_key": (offending["descriptor_key"] if offending else "unknown"),
                    "rate_limit_type": (str(offending["rate_limit_type"]) if offending else "unknown"),
                },
                rate_limit_type=map_v3_rate_limit_type(offending["rate_limit_type"] if offending else None),
                headers={
                    "retry-after": str(self.v3_limiter.window_size),
                    "x-litellm-priority": priority or "default",
                },
                model=resolved_model,
                llm_provider=llm_provider,
            )

        # If priority is NOT enforced (saturation below threshold) but
        # priority_descriptors exist, increment them for tracking only — no
        # check, no rollback. This matches the prior tracking semantics.
        #
        # Using the non-atomic should_rate_limit (instead of
        # atomic_check_and_increment_by_n) is intentional here: we don't want
        # to enforce the limit, we only want to bump the counter so the
        # priority allocation has accurate usage when it later becomes
        # enforced. The increment-then-check semantics of should_rate_limit
        # are fine because we ignore the OVER_LIMIT response.
        stash: Final = get_or_create_request_stash()
        stash.borrowing_team_model_reserved_tokens = team_reservation
        stash.borrowing_team_model_reserved_window = (
            _reserved_window(atomic_response, enforced_team_share) if team_reservation else None
        )
        if team_share is not None and enforced_team_share is None:
            await self.v3_limiter.atomic_check_and_increment_by_n(
                descriptors=[team_share.share],
                increments=[per_request_increment],
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )
        if priority_descriptors and not should_enforce_priority:
            priority_tracking_response: Final = await self.v3_limiter.should_rate_limit(
                descriptors=priority_descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                read_only=False,
            )
            get_or_create_request_stash().rate_limit_response = RateLimitResponse(
                overall_code=atomic_response["overall_code"],
                statuses=atomic_response["statuses"] + priority_tracking_response["statuses"],
            )
        else:
            get_or_create_request_stash().rate_limit_response = atomic_response

    async def _team_token_reservation(
        self,
        descriptor: RateLimitDescriptor,
        data: Mapping[str, object],
        model: str,
        call_type: str | None,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> int:
        """Tokens to hold against the team's share before the call, as the regular limiter does for a hard limit."""
        tpm_share: Final = (descriptor["rate_limit"] or {}).get("tokens_per_unit")
        if tpm_share is None or not data:
            return 0
        try:
            estimate: Final = await offload_token_count(self.v3_limiter.estimate_request_tokens)(
                data=data,
                model=model,
                min_configured_tpm_limit=tpm_share,
                call_type=call_type,
                team_id=user_api_key_dict.team_id,
            )
        except Exception as e:  # noqa: BLE001  # an unreadable prompt still holds the minimum so admission runs
            verbose_proxy_logger.warning(
                "Dynamic rate limiter: token estimate failed for %s, holding one token: %s", model, e
            )
            return 1
        return max(estimate, 1)

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool:
        """Hold a router fallback target to the team's share on it once it saturates, and to its borrow ceiling before.

        With ``commit`` the attempt is counted toward the target's saturation and the team's counter, holding the
        prompt's estimated tokens when the share is enforced, otherwise only checked.
        """
        team_share: Final = self._borrowing_team_share(model=model, user_api_key_dict=user_api_key_dict)
        model_group_info: Final = self.llm_router.registry().group(model)
        if team_share is None or model_group_info is None:
            return True
        saturation: Final = await self._check_model_saturation(model, model_group_info)
        enforced: Final = team_share.enforced(saturation >= _get_priority_settings().saturation_threshold)
        reservation: Final = (
            await self._team_token_reservation(
                enforced, {**request_data, "model": model}, model, None, user_api_key_dict
            )
            if enforced is not None
            else 0
        )
        if not commit:
            return enforced is None or await self._share_has_room(enforced, reservation, user_api_key_dict)
        model_wide_descriptor: Final = self._create_model_tracking_descriptor(
            model=model, model_group_info=model_group_info, high_limit_multiplier=1
        )
        response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=[model_wide_descriptor, *(() if enforced is None else (enforced,))],
            increments=[
                {"requests": 1, "tokens": 0},
                *(() if enforced is None else ({"requests": 1, "tokens": reservation},)),
            ],
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        if response["overall_code"] == "OVER_LIMIT":
            return False
        if enforced is None:
            await self.v3_limiter.atomic_check_and_increment_by_n(
                descriptors=[team_share.share],
                increments=[{"requests": 1, "tokens": reservation}],
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )
        self._hold_fallback_share(model, team_share.share, reservation, _reserved_window(response, enforced))
        return True

    async def _share_has_room(
        self, descriptor: RateLimitDescriptor, tokens: int, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        checked: Final = await self.v3_limiter.should_rate_limit(
            [descriptor], parent_otel_span=user_api_key_dict.parent_otel_span, read_only=True
        )
        return all(
            status["code"] != "OVER_LIMIT"
            and (status["rate_limit_type"] != "tokens" or (status["limit_remaining"] or 0) >= tokens)
            for status in checked["statuses"]
        )

    def _hold_fallback_share(
        self, model: str, share: RateLimitDescriptor, reserved_tokens: int, window: ReservationWindowIdentity | None
    ) -> None:
        """Remember the hold under the requested name and the resolved group, so settlement finds it either way."""
        stash: Final = get_or_create_request_stash()
        earlier: Final = stash.borrowing_team_model_fallback_limits.get(model)
        carried: Final = earlier.reserved_tokens if earlier is not None and earlier.window == window else 0
        hold: Final = BorrowingTeamModelHold(descriptor=share, reserved_tokens=reserved_tokens + carried, window=window)
        stash.borrowing_team_model_fallback_limits = MappingProxyType(
            {**stash.borrowing_team_model_fallback_limits, model: hold, share["value"].rpartition(":")[2]: hold}
        )

    @with_service_target(RATE_LIMITS_TARGET)
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
        if "model" not in data:
            return None

        claim_request_stash_for_data(data)
        model: Final = data["model"]
        priority: Final = self._get_priority_from_user_api_key_dict(user_api_key_dict=user_api_key_dict)

        # Get model configuration
        model_group_info: Final[ModelGroupInfo | None] = self.llm_router.registry().group(model)
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

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ) -> LLMResponseTypes:
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

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        """
        Update token usage for priority-based rate limiting after successful API calls.

        Increments token counters for:
        - model_saturation_check: Model-wide token tracking
        - priority_model: Priority-specific token tracking
        """
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        try:
            verbose_proxy_logger.debug("INSIDE dynamic rate limiter ASYNC SUCCESS LOGGING")

            litellm_parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs)
            callback_kwargs: Final = callback_kwargs_mapping(kwargs)
            settlement_metadata: Final = (
                _SettlementLoggingObject.model_validate(callback_kwargs.get("standard_logging_object") or {}).metadata
                or _SettlementMetadata()
            )

            model_group: Final = get_model_group_from_litellm_kwargs(dict(kwargs))
            total_tokens: Final = _settled_tokens(
                _usage_of(callback_kwargs, response_obj), self.v3_limiter.get_rate_limit_type()
            )
            stash: Final = request_stash_for_callback(callback_kwargs)
            team_model_tokens: Final = _team_model_settlement(
                stash, model_group, total_tokens, self.v3_limiter.window_size
            )
            if stash is not None:
                stash.borrowing_team_model_reserved_tokens = 0
                stash.borrowing_team_model_reserved_window = None
                stash.borrowing_team_model_fallback_limits = MappingProxyType({})
            if not model_group:
                if team_model_tokens:
                    await self.v3_limiter.async_increment_reservation_aware_tokens(
                        pipeline_operations=team_model_tokens, parent_otel_span=litellm_parent_otel_span
                    )
                return

            # Get priority from user_api_key_auth_metadata in standard_logging_metadata
            # This is where user_api_key_dict.metadata is stored during pre-call
            key_priority: Final = settlement_metadata.priority

            if total_tokens == 0 and not team_model_tokens:
                return

            # Model-wide token tracking (model_saturation_check)
            model_tokens: Final = CounterIncrement(
                counter=RateLimitScope("model_saturation_check", model_group).counter("tokens"),
                amount=total_tokens,
                ttl=self.v3_limiter.window_size,
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

            usage_operations: Final = (
                ()
                if total_tokens == 0
                else (
                    model_tokens,
                    CounterIncrement(
                        counter=RateLimitScope("priority_model", priority_key).counter("tokens"),
                        amount=total_tokens,
                        ttl=self.v3_limiter.window_size,
                    ),
                )
            )
            pipeline_operations: Final[tuple[ReservationOperation, ...]] = (*usage_operations, *team_model_tokens)

            # Execute token increments with TTL preservation
            if pipeline_operations:
                await self.v3_limiter.async_increment_reservation_aware_tokens(
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

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_log_failure_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        """Give back the tokens held for an attempt that produced no usage, on the admitting team counter and on
        the fallback target that failed, leaving holds on other fallback targets to the attempts still running."""
        from litellm.proxy.common_utils.callback_utils import get_model_group_from_litellm_kwargs

        callback_kwargs: Final = callback_kwargs_mapping(kwargs)
        stash: Final = request_stash_for_callback(callback_kwargs)
        if stash is None:
            return
        failed_group: Final = get_model_group_from_litellm_kwargs(dict(callback_kwargs))
        failed_hold: Final = (
            stash.borrowing_team_model_fallback_limits.get(failed_group) if failed_group is not None else None
        )
        refund: Final = (
            *_window_guarded(
                _team_model_token_increment(
                    stash.borrowing_team_model_limit,
                    -stash.borrowing_team_model_reserved_tokens,
                    self.v3_limiter.window_size,
                ),
                stash.borrowing_team_model_reserved_window,
            ),
            *_hold_refunds(() if failed_hold is None else (failed_hold,), self.v3_limiter.window_size),
        )
        stash.borrowing_team_model_reserved_tokens = 0
        stash.borrowing_team_model_reserved_window = None
        if failed_hold is not None:
            stash.borrowing_team_model_fallback_limits = MappingProxyType(
                {k: h for k, h in stash.borrowing_team_model_fallback_limits.items() if h is not failed_hold}
            )
        if refund:
            await self.v3_limiter.async_increment_reservation_aware_tokens(pipeline_operations=refund)


PROXY_DynamicRateLimitHandlerV3: Final = _PROXY_DynamicRateLimitHandlerV3
