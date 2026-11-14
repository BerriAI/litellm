"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm import Router
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
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
    SATURATION_FALLBACK_DESCRIPTOR_KEY,
    ProxyRateLimitError,
    fallback_on_saturation_requested,
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
    get_request_stash,
    request_stash_for_callback,
)
from litellm.proxy.hooks.priority_class_share import (
    DEFAULT_CLASS,
    ActiveTeamRegistry,
    TeamShare,
    counter_value,
    pool_member,
    team_member,
    team_share,
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
from litellm.utils import is_health_check_probe_log

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.types.utils import PriorityReservationSettings


class _SettlementMetadata(BaseModel):
    """The two fields of a request's standard logging metadata that token settlement reads."""

    model_config = ConfigDict(frozen=True)

    user_api_key_auth_metadata: Mapping[str, object] | None = None
    user_api_key_team_id: str | None = None
    user_api_key_team_alias: str | None = None

    @property
    def priority(self) -> str | None:
        value: Final = (self.user_api_key_auth_metadata or {}).get("priority")
        return value if isinstance(value, str) else None


class _SettlementLoggingObject(BaseModel):
    """The slice of a request's standard logging object that token settlement reads."""

    model_config = ConfigDict(frozen=True)

    metadata: _SettlementMetadata | None = None


def _prometheus_loggers() -> "Sequence[PrometheusLogger]":
    from litellm.integrations.prometheus import PrometheusLogger

    return tuple(
        logger
        for logger in litellm.logging_callback_manager.get_custom_loggers_for_type(PrometheusLogger)
        if isinstance(logger, PrometheusLogger)
    )


def _tightest_team_share(descriptors: Iterable[RateLimitDescriptor | None]) -> tuple[RateLimitDescriptor, int] | None:
    """The team share descriptor with the smallest TPM share, the one that binds first at saturation."""
    shares: Final = tuple(
        (descriptor, tokens)
        for descriptor in descriptors
        if descriptor is not None and (tokens := (descriptor["rate_limit"] or {}).get("tokens_per_unit")) is not None
    )
    return min(shares, key=lambda share: share[1], default=None)


def _tightest_of(*shares: tuple[RateLimitDescriptor, int] | None) -> tuple[RateLimitDescriptor, int] | None:
    return min((share for share in shares if share is not None), key=lambda share: share[1], default=None)


def _share_on_model(share: RateLimitDescriptor | None, team: str, model_group: str) -> RateLimitDescriptor | None:
    return share if share is not None and share["value"] == f"{team}:{model_group}" else None


def _capped_idle_share(
    share: RateLimitDescriptor | None, cap: RateLimitDescriptor | None
) -> tuple[RateLimitDescriptor, int] | None:
    """A capped team's usage while the model is idle sits on its cap counter, read against its share."""
    tightest: Final = _tightest_team_share((share,))
    if tightest is None or cap is None:
        return None
    return cap, tightest[1]


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
PRIORITY_CLASS_SHARE_AT_SATURATION: Final = "priority_class_share_saturated"
TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE: Final = "model_per_team_max_borrow_multiple"


@dataclass(frozen=True, slots=True)
class _BorrowingTeamShare:
    """The team's share and, when the team caps how far it borrows, its ceiling, both on the share's counter."""

    limits: BorrowingTeamModelLimits
    share: RateLimitDescriptor
    ceiling: RateLimitDescriptor | None

    def enforced(self, saturated: bool) -> RateLimitDescriptor | None:
        return self.share if saturated else None


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


def _team_model_settlement(
    stash: RequestRateLimiterStash | None, model_group: str | None, total_tokens: int, ttl: int
) -> tuple[ReservationOperation, ...]:
    """Settle the counter that served the call to actual usage and return the admitting reservation it did not use.

    The admitting team counter serves unless a fallback group did, in which case that fallback's hold serves and
    the admitting reservation is returned. A hold on a fallback target that did not serve stays for that attempt's
    own failure callback, which may still be reporting the usage it recorded.
    """
    if stash is None:
        return ()
    served: Final = stash.borrowing_team_model_fallback_limits.get(model_group) if model_group is not None else None
    admitted: Final = stash.borrowing_team_model_limit
    cap: Final = stash.borrowing_team_model_cap
    reserved_tokens: Final = stash.borrowing_team_model_reserved_tokens
    cap_reserved_tokens: Final = stash.borrowing_team_model_cap_reserved_tokens
    window: Final = stash.borrowing_team_model_reserved_window
    cap_window: Final = stash.borrowing_team_model_cap_reserved_window
    if served is None:
        return (
            *_window_guarded(_team_model_token_increment(admitted, total_tokens - reserved_tokens, ttl), window),
            *_window_guarded(_team_model_token_increment(cap, total_tokens - cap_reserved_tokens, ttl), cap_window),
        )
    return (
        *_window_guarded(_team_model_token_increment(admitted, -reserved_tokens, ttl), window),
        *_window_guarded(_team_model_token_increment(cap, -cap_reserved_tokens, ttl), cap_window),
        *_window_guarded(
            _team_model_token_increment(served.descriptor, total_tokens - served.reserved_tokens, ttl), served.window
        ),
        *_window_guarded(
            _team_model_token_increment(served.cap, total_tokens - served.cap_reserved_tokens, ttl), served.cap_window
        ),
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
        self.active_teams = ActiveTeamRegistry(internal_usage_cache)

    def update_variables(self, llm_router: Router) -> None:
        self.llm_router = llm_router

    def _get_saturation_check_cache_ttl(self) -> int:
        """Get the configurable TTL for local cache when reading saturation values."""
        return _get_priority_settings().saturation_check_cache_ttl

    async def _get_saturation_value_from_cache(
        self,
        counter: RateLimitCounter,
    ) -> int | float | str | None:
        """The model's counter as Redis holds it, so every worker agrees on saturation. A worker's own in-memory copy
        carries only the usage it served itself, and steady traffic keeps that copy alive past any TTL. The read is
        written back to memory for ``saturation_check_cache_ttl`` seconds; without Redis, or when the read fails, memory is
        all there is."""
        rate_limits: Final = self.v3_limiter.rate_limits
        try:
            value: Final = await rate_limits.get_from_redis(
                counter, backfill_ttl=self._get_saturation_check_cache_ttl()
            )
        except Exception as e:  # noqa: BLE001  # any Redis failure degrades to the local mirror, as the limiter's counter reads do
            verbose_proxy_logger.warning("Saturation read failed in Redis, using this worker's copy: %s", e)
            local: Final = rate_limits.local((counter,))[0]
            return None if isinstance(local, dict) else local
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

    def _priority_class(self, priority: str | None) -> str:
        """The class a team is counted under: its explicit priority, or one shared default class."""
        if (
            priority is not None
            and litellm.priority_reservation is not None
            and priority in litellm.priority_reservation
        ):
            return priority
        return DEFAULT_CLASS

    async def _team_class_share(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        normalized_weights: Mapping[str, float],
    ) -> TeamShare | None:
        """The caller's slice of its class share, or None when the split is off or the key has no team. A key with no
        team keeps the class pool but still marks its class active, so the pool's capacity is not lent away."""
        if (
            litellm.priority_reservation is None
            or not _get_priority_settings().priority_class_share_splits_equally_among_active_teams
        ):
            return None
        priority_class: Final = self._priority_class(priority)
        member: Final = (
            pool_member(priority_class) if user_api_key_dict.team_id is None else team_member(user_api_key_dict.team_id)
        )
        from redis.exceptions import RedisError

        try:
            members: Final = await self.active_teams.touch(model, member, priority_class, self.v3_limiter.window_size)
        except (RedisError, RedisCircuitBreakerOpenError, ValidationError) as e:
            verbose_proxy_logger.warning(
                "Active team registry unavailable for %s, holding team %s to the class pool: %s",
                model,
                user_api_key_dict.team_id,
                e,
            )
            return None
        if user_api_key_dict.team_id is None:
            return None
        return team_share(priority_class, members, normalized_weights, _get_priority_settings().default_priority)

    def _create_priority_based_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        share: TeamShare | None = None,
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
        weight: Final = priority_weight if share is None else share.fraction
        floor: Final = 0 if share is None else 1

        # Apply priority weight to model limits
        if model_group_info.tpm is not None:
            reserved_tpm: Final = max(floor, int(model_group_info.tpm * weight))
            rate_limit_config["tokens_per_unit"] = reserved_tpm

        if model_group_info.rpm is not None:
            reserved_rpm: Final = max(floor, int(model_group_info.rpm * weight))
            rate_limit_config["requests_per_unit"] = reserved_rpm

        if rate_limit_config:
            rate_limit_config["window_size"] = self.v3_limiter.window_size

            descriptors.append(
                RateLimitDescriptor(
                    key="priority_model" if share is None else PRIORITY_CLASS_SHARE_AT_SATURATION,
                    value=priority_key if share is None else counter_value(model, str(user_api_key_dict.team_id)),
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
            ceiling=None
            if ceiling is None
            else self._team_share_descriptor(value, ceiling, TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE),
        )

    def _team_share_descriptor(
        self, value: str, limits: BorrowingTeamModelLimits, key: str = TEAM_MODEL_LIMIT_AT_SATURATION
    ) -> RateLimitDescriptor:
        return RateLimitDescriptor(
            key=key,
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
        if status["descriptor_key"] != TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE:
            return self._team_limit_error(model, status, saturation, priority, user_api_key_dict)
        return self._team_borrow_ceiling_error(
            model, status, saturation, priority, user_api_key_dict, team_share.limits
        )

    def _throttled(
        self, refusal: ProxyRateLimitError, model: str, user_api_key_dict: UserAPIKeyAuth
    ) -> ProxyRateLimitError:
        """Count a refusal that reaches the caller under the reason its ``rate_limited_by`` header carries."""
        headers: Final = refusal.headers or {}
        reason: Final = headers.get("rate_limited_by")
        if user_api_key_dict.team_id is None or reason is None:
            return refusal
        for logger in _prometheus_loggers():
            try:
                logger.increment_dynamic_limiter_throttled(
                    model_group=model,
                    team=user_api_key_dict.team_id,
                    team_alias=user_api_key_dict.team_alias,
                    priority=headers.get("x-litellm-priority") or "default",
                    rate_limit_type=headers.get("rate_limit_type") or "tokens",
                    reason=reason,
                )
            except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change the 429
                verbose_proxy_logger.warning("Dynamic rate limiter throttle metric failed: %s", metrics_error)
        return refusal

    async def _record_team_share(
        self,
        stash: RequestRateLimiterStash | None,
        model_group: str | None,
        settlement: _SettlementMetadata,
        hold: BorrowingTeamModelHold | None,
    ) -> None:
        """Publish the team's share on the model that served the request, the fallback target's own hold when a
        router fallback served it, and the tokens settled above that share so far this window."""
        try:
            loggers: Final = _prometheus_loggers()
            if not loggers or stash is None or model_group is None or settlement.user_api_key_team_id is None:
                return
            share: Final = (
                _tightest_of(_tightest_team_share((hold.descriptor,)), _capped_idle_share(hold.share, hold.cap))
                if hold is not None
                else _tightest_of(
                    _tightest_team_share((stash.borrowing_team_model_limit, stash.priority_class_share)),
                    _capped_idle_share(stash.borrowing_team_model_share, stash.borrowing_team_model_cap),
                )
            ) or _tightest_team_share(
                (
                    hold.share
                    if hold is not None
                    else _share_on_model(
                        stash.borrowing_team_model_share,
                        settlement.user_api_key_team_id,
                        (
                            self.v3_limiter.rate_limited_model(model_group)
                            or RateLimitedModel(requested=model_group, group=model_group)
                        ).group,
                    ),
                )
            )
            if share is None:
                return
            descriptor, share_tokens = share
            used: Final = await self._get_saturation_value_from_cache(
                RateLimitScope(descriptor["key"], descriptor["value"]).counter("tokens")
            )
            used_tokens: Final = 0 if used is None else int(float(used))
            for logger in loggers:
                logger.set_dynamic_limiter_share(
                    model_group=model_group,
                    team=settlement.user_api_key_team_id,
                    team_alias=settlement.user_api_key_team_alias,
                    priority=stash.dynamic_limiter_priority or "default",
                    share_tokens=share_tokens,
                    borrowed_tokens=max(0, used_tokens - share_tokens),
                )
        except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change settlement
            verbose_proxy_logger.warning("Dynamic rate limiter share metrics failed: %s", metrics_error)

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

    def _class_share_error(
        self,
        model: str,
        status: RateLimitStatus,
        saturation: float,
        priority: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        share: TeamShare,
    ) -> ProxyRateLimitError:
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
        error: Final = ProxyRateLimitError(
            detail={
                "error": f"Priority class share reached for {model} and the model is saturated. "
                f"Team: {user_api_key_dict.team_id}, "
                f"Priority: {priority or 'default'}, "
                f"Active teams in class: {share.active_teams_in_class}, "
                f"Active classes: {share.active_classes}, "
                f"Rate limit type: {status['rate_limit_type']}, "
                f"Team share: {status['current_limit']}, "
                f"Remaining: {status['limit_remaining']}, "
                f"Model saturation: {saturation:.1%}"
            },
            headers={
                "retry-after": str(self.v3_limiter.window_size),
                "rate_limit_type": str(status["rate_limit_type"]),
                "rate_limited_by": PRIORITY_CLASS_SHARE_AT_SATURATION,
                "x-litellm-priority": priority or "default",
                "x-litellm-saturation": f"{saturation:.2%}",
            },
            rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
            model=resolved_model,
            llm_provider=llm_provider,
        )
        return error

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        """Hold a Responses WebSocket turn to the team's shares once the model saturates, like the regular limiter does."""
        if requested_model is None:
            return
        model_group_info: Final = self.llm_router.registry().group(requested_model)
        if model_group_info is None:
            return
        priority: Final = self._get_priority_from_user_api_key_dict(user_api_key_dict)
        share: Final = await self._team_class_share(
            requested_model, user_api_key_dict, priority, self._normalize_priority_weights(model_group_info)
        )
        share_descriptors: Final = (
            ()
            if share is None
            else tuple(
                self._create_priority_based_descriptors(
                    model=requested_model, user_api_key_dict=user_api_key_dict, priority=priority, share=share
                )
            )
        )
        frame_share: Final = next(
            (d for d in share_descriptors if d["key"] == PRIORITY_CLASS_SHARE_AT_SATURATION), None
        )
        connection_stash: Final = get_request_stash()
        if connection_stash is not None and frame_share is not None:
            connection_stash.priority_class_share = frame_share
        borrowing: Final = self._borrowing_team_share(model=requested_model, user_api_key_dict=user_api_key_dict)
        if not share_descriptors and borrowing is None:
            return
        saturation: Final = await self._check_model_saturation(requested_model, model_group_info)
        saturated: Final = saturation >= _get_priority_settings().saturation_threshold
        enforced_team: Final = None if borrowing is None else borrowing.enforced(saturated)
        team_cap: Final = None if borrowing is None else borrowing.ceiling
        if connection_stash is not None and borrowing is not None:
            connection_stash.borrowing_team_model_limit = (
                connection_stash.borrowing_team_model_limit
                if enforced_team is None and connection_stash.borrowing_team_model_reserved_tokens
                else enforced_team
            )
            connection_stash.borrowing_team_model_share = borrowing.share
            connection_stash.borrowing_team_model_cap = team_cap
        descriptors: Final = (
            (share_descriptors if saturated else ())
            + (() if enforced_team is None else (enforced_team,))
            + (() if team_cap is None else (team_cap,))
        )
        if not descriptors:
            return
        tpm_windows: Final = _socket_turn_tpm_windows(descriptors)
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
        over_class_share: Final = over_limit["descriptor_key"] == PRIORITY_CLASS_SHARE_AT_SATURATION
        if share is not None and (over_class_share or borrowing is None):
            raise self._throttled(
                self._class_share_error(requested_model, over_limit, saturation, priority, user_api_key_dict, share),
                requested_model,
                user_api_key_dict,
            )
        if borrowing is not None:
            raise self._throttled(
                self._team_share_error(requested_model, over_limit, saturation, priority, user_api_key_dict, borrowing),
                requested_model,
                user_api_key_dict,
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
        share: Final = await self._team_class_share(
            model, user_api_key_dict, priority, self._normalize_priority_weights(model_group_info)
        )
        priority_descriptors: Final = self._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user_api_key_dict,
            priority=priority,
            share=share,
        )
        get_or_create_request_stash().priority_class_share = next(
            (d for d in priority_descriptors if d["key"] == PRIORITY_CLASS_SHARE_AT_SATURATION), None
        )
        get_or_create_request_stash().dynamic_limiter_priority = priority
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
        if team_share is not None:
            descriptors_to_check.append(team_share.share)
        enforced_team_share: Final = None if team_share is None else team_share.enforced(should_enforce_priority)
        team_cap: Final = None if team_share is None else team_share.ceiling
        held_team_share: Final = enforced_team_share if enforced_team_share is not None else team_cap
        get_or_create_request_stash().borrowing_team_model_limit = enforced_team_share
        get_or_create_request_stash().borrowing_team_model_share = None if team_share is None else team_share.share
        get_or_create_request_stash().borrowing_team_model_cap = team_cap

        enforced_descriptors: Final[list[RateLimitDescriptor]] = [model_wide_descriptor]
        if priority_descriptors and should_enforce_priority:
            enforced_descriptors.extend(priority_descriptors)
        if enforced_team_share is not None:
            enforced_descriptors.append(enforced_team_share)
        if team_cap is not None:
            enforced_descriptors.append(team_cap)

        per_request_increment: Final[dict[Literal["requests", "tokens"], int]] = {
            "requests": 1,
            "tokens": 0,
        }
        team_reservation: Final = (
            await self._team_token_reservation(team_share.share, data, model, call_type, user_api_key_dict)
            if team_share is not None and held_team_share is not None
            else 0
        )
        atomic_response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=enforced_descriptors,
            increments=[
                {"requests": 1, "tokens": team_reservation}
                if descriptor is enforced_team_share or descriptor is team_cap
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

                if (
                    descriptor_key in (TEAM_MODEL_LIMIT_AT_SATURATION, TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE)
                    and team_share is not None
                ):
                    raise self._team_share_error(model, status, saturation, priority, user_api_key_dict, team_share)
                if descriptor_key == PRIORITY_CLASS_SHARE_AT_SATURATION and share is not None:
                    raise self._class_share_error(model, status, saturation, priority, user_api_key_dict, share)

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
        stash.borrowing_team_model_reserved_tokens = team_reservation if enforced_team_share is not None else 0
        stash.borrowing_team_model_cap_reserved_tokens = team_reservation if team_cap is not None else 0
        stash.borrowing_team_model_reserved_window = (
            _reserved_window(atomic_response, enforced_team_share) if team_reservation else None
        )
        stash.borrowing_team_model_cap_reserved_window = (
            _reserved_window(atomic_response, team_cap) if team_reservation else None
        )
        self._cap_admitted_output(None if team_share is None else team_share.share, team_reservation, data, call_type)
        if priority_descriptors and share is not None and not should_enforce_priority:
            get_or_create_request_stash().rate_limit_response = atomic_response
        elif priority_descriptors and not should_enforce_priority:
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

    def _cap_admitted_output(
        self,
        enforced_team_share: RateLimitDescriptor | None,
        team_reservation: int,
        data: Mapping[str, object],
        call_type: str | None,
    ) -> None:
        """Write the output cap the hold was estimated with onto the admitted request, unless only input tokens count."""
        tpm_share: Final = (
            None if enforced_team_share is None else (enforced_team_share["rate_limit"] or {}).get("tokens_per_unit")
        )
        if team_reservation and tpm_share is not None and self.v3_limiter.get_rate_limit_type() != "input":
            self.v3_limiter.bound_output_for_limit(data, tpm_share, call_type)

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
        as_dispatched: Final = dict(data, model=model)
        if self.v3_limiter.get_rate_limit_type() != "input":
            self.v3_limiter.bound_output_for_limit(as_dispatched, tpm_share, call_type)
        try:
            estimate: Final = await offload_token_count(self.v3_limiter.estimate_request_tokens)(
                data=as_dispatched,
                model=model,
                min_configured_tpm_limit=tpm_share,
                call_type=call_type,
                team_id=user_api_key_dict.team_id,
                rate_limit_type=self.v3_limiter.get_rate_limit_type(),
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

        With ``commit`` the attempt is counted toward the target's saturation, and toward the team's counter only
        while the share is enforced, holding the prompt's estimated tokens there. An idle target is only checked.
        """
        team_share: Final = self._borrowing_team_share(model=model, user_api_key_dict=user_api_key_dict)
        model_group_info: Final = self.llm_router.registry().group(model)
        if team_share is None or model_group_info is None:
            return True
        saturation: Final = await self._check_model_saturation(model, model_group_info)
        enforced: Final = team_share.enforced(saturation >= _get_priority_settings().saturation_threshold)
        cap: Final = team_share.ceiling
        held: Final = enforced if enforced is not None else cap
        reservation: Final = (
            await self._team_token_reservation(
                team_share.share,
                {**request_data, "model": model},
                model,
                get_or_create_request_stash().call_type,
                user_api_key_dict,
            )
            if held is not None
            else 0
        )
        team_descriptors: Final = tuple(d for d in (enforced, cap) if d is not None)
        if not commit:
            for descriptor in team_descriptors:
                if not await self._share_has_room(descriptor, reservation, user_api_key_dict):
                    return False
            return True
        model_wide_descriptor: Final = self._create_model_tracking_descriptor(
            model=model, model_group_info=model_group_info, high_limit_multiplier=1
        )
        response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=[model_wide_descriptor, *team_descriptors],
            increments=[
                {"requests": 1, "tokens": 0},
                *({"requests": 1, "tokens": reservation} for _ in team_descriptors),
            ],
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        if response["overall_code"] == "OVER_LIMIT":
            return False
        self._hold_fallback_share(
            model,
            team_share.share,
            enforced,
            reservation if enforced is not None else 0,
            _reserved_window(response, enforced),
            cap,
            reservation if cap is not None else 0,
            _reserved_window(response, cap),
        )
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
        self,
        model: str,
        share: RateLimitDescriptor,
        charged: RateLimitDescriptor | None,
        reserved_tokens: int,
        window: ReservationWindowIdentity | None,
        cap: RateLimitDescriptor | None = None,
        cap_reserved_tokens: int = 0,
        cap_window: ReservationWindowIdentity | None = None,
    ) -> None:
        """Remember the hold under the requested name and the resolved group, so settlement finds it either way.

        ``charged`` is the share the target's usage settles to, or None for an idle target whose usage is forgiven.
        ``cap`` is the team's borrow ceiling on the target, charged whether or not the target is saturated. Each
        counter carries an earlier hold on the same target only while that hold sits in the counter's current window.
        """
        stash: Final = get_or_create_request_stash()
        earlier: Final = stash.borrowing_team_model_fallback_limits.get(model)
        carried: Final = (
            earlier.reserved_tokens if earlier is not None and window is not None and earlier.window == window else 0
        )
        carried_cap: Final = (
            earlier.cap_reserved_tokens
            if earlier is not None and cap_window is not None and earlier.cap_window == cap_window
            else 0
        )
        hold: Final = BorrowingTeamModelHold(
            descriptor=charged,
            reserved_tokens=reserved_tokens + carried,
            window=window,
            cap=cap,
            cap_reserved_tokens=cap_reserved_tokens + carried_cap,
            cap_window=cap_window,
            share=share,
        )
        stash.borrowing_team_model_fallback_limits = MappingProxyType(
            {**stash.borrowing_team_model_fallback_limits, model: hold, share["value"].rpartition(":")[2]: hold}
        )

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
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
        if not isinstance(model, str):
            return None

        # Get model configuration
        requested_group_info: Final[ModelGroupInfo | None] = self.llm_router.registry().group(model)
        if requested_group_info is None:
            verbose_proxy_logger.debug("No model group info for %s, allowing request", model)
            return None

        try:
            await self._admit(model, requested_group_info, user_api_key_dict, data, call_type)
        except ProxyRateLimitError as refusal:
            raise self._throttled(refusal, model, user_api_key_dict) from None
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Error in dynamic rate limiter: %s, allowing request", e)
            # Fail open on unexpected errors
            return None
        return None

    async def _admit(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        data: Mapping[str, object],
        call_type: str | None,
    ) -> None:
        """Read saturation, then the three-phase rate limit check, raising the 429 when ``model`` is out of capacity.
        A key or team with ``router_settings.fallback_on_saturation`` is sent to its fallbacks first once ``model`` is
        at or above ``saturation_threshold``, and comes back here only when none of them admitted the request."""
        priority: Final = self._get_priority_from_user_api_key_dict(user_api_key_dict=user_api_key_dict)
        saturation: Final = await self._check_model_saturation(model, model_group_info)
        saturation_threshold: Final = _get_priority_settings().saturation_threshold
        verbose_proxy_logger.debug(
            f"[Dynamic Rate Limiter] Model={model}, Saturation={saturation:.1%}, "
            f"Threshold={saturation_threshold:.1%}, Priority={priority}"
        )
        if saturation >= saturation_threshold and fallback_on_saturation_requested(data):
            raise self._saturation_fallback_error(model, saturation, priority)
        await self._check_rate_limits(
            model=model,
            model_group_info=model_group_info,
            user_api_key_dict=user_api_key_dict,
            priority=priority,
            saturation=saturation,
            data=data,
            call_type=call_type,
        )

    def _saturation_fallback_error(self, model: str, saturation: float, priority: str | None) -> ProxyRateLimitError:
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
        return ProxyRateLimitError(
            detail={
                "error": f"Model {model} is at {saturation:.0%} of its capacity, at or above the saturation "
                f"threshold of {_get_priority_settings().saturation_threshold:.0%}, trying fallbacks first. "
                f"Priority: {priority}"
            },
            headers={"retry-after": str(self.v3_limiter.window_size), "x-litellm-priority": priority or "default"},
            model=resolved_model,
            llm_provider=llm_provider,
            descriptor_key=SATURATION_FALLBACK_DESCRIPTOR_KEY,
        )

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_post_call_success_hook(
        self,
        data: dict[str, object],  # mutable-ok: the CustomLogger contract hands the v3 limiter a mutable dict
        user_api_key_dict: UserAPIKeyAuth,
        response,
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
            if is_health_check_probe_log(callback_kwargs.get("standard_logging_object")):
                return
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
            served: Final = (
                None
                if stash is None or model_group is None
                else stash.borrowing_team_model_fallback_limits.get(model_group)
            )
            if stash is not None:
                stash.borrowing_team_model_reserved_tokens = 0
                stash.borrowing_team_model_cap_reserved_tokens = 0
                stash.borrowing_team_model_reserved_window = None
                stash.borrowing_team_model_cap_reserved_window = None
                stash.borrowing_team_model_fallback_limits = MappingProxyType(
                    {k: h for k, h in stash.borrowing_team_model_fallback_limits.items() if h is not served}
                )
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

            class_share: Final = None if stash is None else stash.priority_class_share
            settled_team_id: Final = settlement_metadata.user_api_key_team_id
            priority_scope: Final = (
                RateLimitScope("priority_model", priority_key)
                if class_share is None
                else RateLimitScope(
                    class_share["key"],
                    class_share["value"] if settled_team_id is None else counter_value(model_group, settled_team_id),
                )
            )
            usage_operations: Final = (
                ()
                if total_tokens == 0
                else (
                    model_tokens,
                    CounterIncrement(
                        counter=priority_scope.counter("tokens"),
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
                await self._record_team_share(stash, model_group, settlement_metadata, served)

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
    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: the base hook declares this parameter as a dict
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """Give back the team-share tokens held at admission when the proxy refuses the request before the call runs,
        as a later hook or the pre-call fallback walk does. ``async_log_failure_event`` is a completion callback and
        never fires for those refusals, so without this the hold stays charged until the window resets. Idempotent:
        the first release zeroes the stash's reservation, so a settlement that still follows charges only usage."""
        stash: Final = get_request_stash()
        if stash is None:
            return
        refund: Final = _team_model_settlement(stash, None, 0, self.v3_limiter.window_size)
        stash.borrowing_team_model_reserved_tokens = 0
        stash.borrowing_team_model_cap_reserved_tokens = 0
        stash.borrowing_team_model_reserved_window = None
        stash.borrowing_team_model_cap_reserved_window = None
        if refund:
            await self.v3_limiter.async_increment_reservation_aware_tokens(
                pipeline_operations=refund, parent_otel_span=user_api_key_dict.parent_otel_span
            )

    async def async_log_failure_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        """Settle the tokens held for a failed attempt to the usage it recorded, or give them all back when it recorded
        none, on the admitting team counter or the fallback target that failed, leaving holds on other fallback
        targets to the attempts still running."""
        from litellm.proxy.common_utils.callback_utils import get_model_group_from_litellm_kwargs

        callback_kwargs: Final = callback_kwargs_mapping(kwargs)
        stash: Final = request_stash_for_callback(callback_kwargs)
        if stash is None:
            return
        failed_group: Final = get_model_group_from_litellm_kwargs(dict(callback_kwargs))
        failed_hold: Final = (
            stash.borrowing_team_model_fallback_limits.get(failed_group) if failed_group is not None else None
        )
        used: Final = _settled_tokens(
            _usage_of(callback_kwargs, None if isinstance(response_obj, BaseException) else response_obj),
            self.v3_limiter.get_rate_limit_type(),
        )
        own_usage: Final = used if failed_hold is None else 0
        own_delta: Final = own_usage - stash.borrowing_team_model_reserved_tokens
        own_cap_delta: Final = own_usage - stash.borrowing_team_model_cap_reserved_tokens
        refund: Final = (
            *_window_guarded(
                _team_model_token_increment(stash.borrowing_team_model_limit, own_delta, self.v3_limiter.window_size),
                stash.borrowing_team_model_reserved_window,
            ),
            *_window_guarded(
                _team_model_token_increment(stash.borrowing_team_model_cap, own_cap_delta, self.v3_limiter.window_size),
                stash.borrowing_team_model_cap_reserved_window,
            ),
            *(
                ()
                if failed_hold is None
                else (
                    *_window_guarded(
                        _team_model_token_increment(
                            failed_hold.descriptor, used - failed_hold.reserved_tokens, self.v3_limiter.window_size
                        ),
                        failed_hold.window,
                    ),
                    *_window_guarded(
                        _team_model_token_increment(
                            failed_hold.cap, used - failed_hold.cap_reserved_tokens, self.v3_limiter.window_size
                        ),
                        failed_hold.cap_window,
                    ),
                )
            ),
        )
        stash.borrowing_team_model_reserved_tokens = 0
        stash.borrowing_team_model_cap_reserved_tokens = 0
        stash.borrowing_team_model_reserved_window = None
        stash.borrowing_team_model_cap_reserved_window = None
        if failed_hold is not None:
            stash.borrowing_team_model_fallback_limits = MappingProxyType(
                {k: h for k, h in stash.borrowing_team_model_fallback_limits.items() if h is not failed_hold}
            )
        if refund:
            await self.v3_limiter.async_increment_reservation_aware_tokens(pipeline_operations=refund)
        await self._record_team_share(
            stash,
            failed_group,
            _SettlementLoggingObject.model_validate(callback_kwargs.get("standard_logging_object") or {}).metadata
            or _SettlementMetadata(),
            failed_hold,
        )


PROXY_DynamicRateLimitHandlerV3: Final = _PROXY_DynamicRateLimitHandlerV3
