"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting
"""

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import TypeAdapter

import litellm
from litellm import ModelResponse, Router
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (  # noqa: F401  # legacy module exports
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    RateLimitResponse,
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    claim_request_stash_for_data,
    get_or_create_request_stash,
)
from litellm.proxy.hooks.rate_limiter_utils import (
    convert_priority_to_percent,
    resolve_llm_provider_for_rate_limit,
)
from litellm.proxy.hooks.team_shares import (
    DEFAULT_CLASS,
    TeamShare,
    TeamShareRegistry,
    ThrottleReason,
    compute_team_share,
    team_burst_multiplier,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET, CounterIncrement, RateLimitCounter, RateLimitScope
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.types.utils import PriorityReservationSettings


def _prometheus_loggers() -> "Sequence[PrometheusLogger]":
    from litellm.integrations.prometheus import PrometheusLogger

    return [
        logger
        for logger in litellm.logging_callback_manager.get_custom_loggers_for_type(PrometheusLogger)
        if isinstance(logger, PrometheusLogger)
    ]


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


TEAM_SHARE_DESCRIPTOR_KEY: Final = "team_share"
_OBJECT_MAP: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_OPTIONAL_TEXT: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)


@dataclass(frozen=True, slots=True)
class _TeamSharePlan:
    """What the limiter holds one team to on this request: its share when the model is saturated, its burst cap
    when it is not and one is configured, otherwise nothing (the descriptor is only tracked)."""

    share: TeamShare
    descriptor: RateLimitDescriptor
    reason: ThrottleReason | None

    @property
    def enforced(self) -> bool:
        return self.reason is not None

    def record(self, user_api_key_dict: UserAPIKeyAuth, model: str, model_tpm: int | None, used_tokens: int) -> None:
        share_tokens: Final = self.share.tokens(model_tpm)
        if share_tokens is None:
            return
        for logger in _prometheus_loggers():
            try:
                logger.set_dynamic_limiter_share(
                    model_group=model,
                    team=self.share.team_id,
                    team_alias=user_api_key_dict.team_alias,
                    priority=self.share.priority,
                    share_tokens=share_tokens,
                    used_tokens=used_tokens,
                )
            except Exception as error:  # noqa: BLE001 - metrics never change an admission decision
                verbose_proxy_logger.warning("Dynamic rate limiter: share metrics failed for %s: %s", model, error)

    def record_throttled(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, rate_limit_type: str, reason: ThrottleReason
    ) -> None:
        for logger in _prometheus_loggers():
            try:
                logger.increment_dynamic_limiter_throttled(
                    model_group=model,
                    team=self.share.team_id,
                    team_alias=user_api_key_dict.team_alias,
                    priority=self.share.priority,
                    rate_limit_type=rate_limit_type,
                    reason=reason,
                )
            except Exception as error:  # noqa: BLE001 - metrics never change an admission decision
                verbose_proxy_logger.warning("Dynamic rate limiter: throttle metrics failed for %s: %s", model, error)


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
        settings: Final = _get_priority_settings()
        if settings.guaranteed_team_shares and settings.saturation_check_cache_ttl * 4 > self.v3_limiter.window_size:
            verbose_proxy_logger.warning(
                "priority_reservation_settings.saturation_check_cache_ttl=%ss is more than a quarter of the %ss "
                "rate limit window, so with guaranteed_team_shares workers can keep admitting borrowers for up to "
                "that long after a model saturates. Lower it to %ss or less.",
                settings.saturation_check_cache_ttl,
                self.v3_limiter.window_size,
                self.v3_limiter.window_size // 4,
            )

    def _team_share_registry(self) -> TeamShareRegistry:
        return TeamShareRegistry(
            dual_cache=self.internal_usage_cache.dual_cache,
            window_size=self.v3_limiter.window_size,
            local_ttl=_get_priority_settings().saturation_check_cache_ttl,
        )

    def _team_share_class(self, priority: str | None) -> str:
        if (
            priority is not None
            and litellm.priority_reservation is not None
            and priority in litellm.priority_reservation
        ):
            return priority
        return DEFAULT_CLASS

    async def _team_share_plan(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        saturated: bool,
    ) -> _TeamSharePlan | None:
        """``None`` unless guaranteed team shares are on, the request belongs to a team and the model has a limit."""
        settings: Final = _get_priority_settings()
        team_id: Final = user_api_key_dict.team_id
        if not settings.guaranteed_team_shares or not team_id or litellm.priority_reservation is None:
            return None
        if model_group_info.tpm is None and model_group_info.rpm is None:
            return None
        members: Final = await self._team_share_registry().members(
            model=model, team_id=team_id, priority=self._team_share_class(priority)
        )
        share: Final = compute_team_share(
            team_id=team_id,
            members=members,
            class_weights=self._normalize_priority_weights(model_group_info),
            default_weight=settings.default_priority,
        )
        multiplier: Final = team_burst_multiplier(user_api_key_dict, settings.default_max_borrow_multiplier)
        scale: Final = 1.0 if saturated or multiplier is None else multiplier
        reason: Final[ThrottleReason | None] = (
            "over_share_saturated" if saturated else "burst_cap" if multiplier else None
        )
        descriptor: Final = RateLimitDescriptor(
            key=TEAM_SHARE_DESCRIPTOR_KEY,
            value=f"{model}:{team_id}",
            rate_limit={
                "tokens_per_unit": share.tokens(model_group_info.tpm, scale),
                "requests_per_unit": share.requests(model_group_info.rpm, scale),
                "window_size": self.v3_limiter.window_size,
            },
        )
        return _TeamSharePlan(share=share, descriptor=descriptor, reason=reason)

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

    async def _record_team_share(
        self, plan: _TeamSharePlan, user_api_key_dict: UserAPIKeyAuth, model: str, model_tpm: int | None
    ) -> None:
        """Publish the team's share and borrowing. Metrics only, so a failure here never changes admission."""
        try:
            used_value: Final = await self._get_saturation_value_from_cache(
                RateLimitScope(TEAM_SHARE_DESCRIPTOR_KEY, plan.descriptor["value"]).counter("tokens")
            )
        except Exception as error:  # noqa: BLE001 - metrics never change an admission decision
            verbose_proxy_logger.warning("Dynamic rate limiter: could not read team usage for %s: %s", model, error)
            return
        plan.record(
            user_api_key_dict, model, model_tpm, used_tokens=0 if used_value is None else int(float(used_value))
        )

    async def _check_rate_limits(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        priority: str | None,
        saturation: float,
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

        Raises:
            HTTPException: If any limit is exceeded
        """

        saturation_threshold: Final = _get_priority_settings().saturation_threshold
        should_enforce_priority: Final = saturation >= saturation_threshold

        # Model-wide descriptor (always enforce)
        model_wide_descriptor: Final = self._create_model_tracking_descriptor(
            model=model,
            model_group_info=model_group_info,
            high_limit_multiplier=1,
        )

        team_share_plan: Final = await self._team_share_plan(
            model=model,
            model_group_info=model_group_info,
            user_api_key_dict=user_api_key_dict,
            priority=priority,
            saturated=should_enforce_priority,
        )

        # Priority descriptors (always track, conditionally enforce)
        priority_descriptors: Final = (
            [team_share_plan.descriptor]
            if team_share_plan is not None
            else self._create_priority_based_descriptors(
                model=model,
                user_api_key_dict=user_api_key_dict,
                priority=priority,
            )
        )
        enforce_pool: Final = team_share_plan.enforced if team_share_plan is not None else should_enforce_priority

        # Atomic check-and-increment for the ENFORCED descriptor set:
        # - model_saturation_check is always enforced
        # - priority_model is enforced only when saturation crosses threshold
        #
        # Backed by a Redis Lua script (multi-process atomic) with an
        # asyncio.Lock + in-memory fallback for single-process deployments.
        # All-or-nothing: if any enforced descriptor would exceed its limit,
        # no counter is modified and the response carries "OVER_LIMIT".
        enforced_descriptors: Final[list[RateLimitDescriptor]] = [model_wide_descriptor]
        if priority_descriptors and enforce_pool:
            enforced_descriptors.extend(priority_descriptors)

        per_request_increment: Final[dict[Literal["requests", "tokens"], int]] = {
            "requests": 1,
            "tokens": 0,
        }
        atomic_response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=enforced_descriptors,
            increments=[per_request_increment for _ in enforced_descriptors],
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        verbose_proxy_logger.debug("Atomic check+increment response: %s", atomic_response)

        if team_share_plan is not None:
            await self._record_team_share(team_share_plan, user_api_key_dict, model, model_group_info.tpm)

        if atomic_response["overall_code"] == "OVER_LIMIT":
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model)
            share: Final = team_share_plan.share if team_share_plan is not None else None
            for status in atomic_response["statuses"]:
                if status["code"] != "OVER_LIMIT":
                    continue
                descriptor_key = status["descriptor_key"]
                if team_share_plan is not None:
                    team_share_plan.record_throttled(
                        user_api_key_dict,
                        model,
                        str(status["rate_limit_type"]),
                        "model_ceiling"
                        if descriptor_key == "model_saturation_check"
                        else team_share_plan.reason or "burst_cap",
                    )
                if descriptor_key == "model_saturation_check":
                    raise ProxyRateLimitError(
                        detail={
                            "error": f"Model capacity reached for {model}. "
                            f"Priority: {priority}, "
                            f"Rate limit type: {status['rate_limit_type']}, "
                            f"Model TPM: {model_group_info.tpm if model_group_info.tpm is not None else 'not configured'}, "
                            f"Model RPM: {model_group_info.rpm if model_group_info.rpm is not None else 'not configured'}, "
                            f"Remaining: {status['limit_remaining']}",
                            "reason": "model_ceiling",
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                            "x-litellm-rate-limit-reason": "model_ceiling",
                        },
                        rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                        model=resolved_model,
                        llm_provider=llm_provider,
                    )
                if descriptor_key == TEAM_SHARE_DESCRIPTOR_KEY and team_share_plan is not None and share is not None:
                    raise ProxyRateLimitError(
                        detail={
                            "error": f"Team share exceeded for {model}. "
                            f"Team: {user_api_key_dict.team_alias or share.team_id}, "
                            f"Priority: {share.priority}, "
                            f"Reason: {team_share_plan.reason}, "
                            f"Share: {share.fraction:.1%} of the model "
                            f"({share.active_teams_in_class} active {share.priority} team(s), "
                            f"{share.active_classes} active class(es)), "
                            f"Rate limit type: {status['rate_limit_type']}, "
                            f"Limit: {status['current_limit']}, "
                            f"Model TPM: {model_group_info.tpm if model_group_info.tpm is not None else 'not configured'}, "
                            f"Model RPM: {model_group_info.rpm if model_group_info.rpm is not None else 'not configured'}, "
                            f"Remaining: {status['limit_remaining']}, "
                            f"Model saturation: {saturation:.1%}",
                            "reason": team_share_plan.reason,
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                            "x-litellm-saturation": f"{saturation:.2%}",
                            "x-litellm-rate-limit-reason": team_share_plan.reason,
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
                            f"Model saturation: {saturation:.1%}",
                            "reason": "over_share_saturated",
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                            "x-litellm-saturation": f"{saturation:.2%}",
                            "x-litellm-rate-limit-reason": "over_share_saturated",
                        },
                        rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                        model=resolved_model,
                        llm_provider=llm_provider,
                    )

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
        if priority_descriptors and not enforce_pool:
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
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
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
        from litellm.types.utils import Usage

        try:
            verbose_proxy_logger.debug("INSIDE dynamic rate limiter ASYNC SUCCESS LOGGING")

            litellm_parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs)

            # Get metadata from standard_logging_object
            standard_logging_object: Final = _OBJECT_MAP.validate_python(kwargs.get("standard_logging_object") or {})
            standard_logging_metadata: Final = _OBJECT_MAP.validate_python(
                standard_logging_object.get("metadata") or {}
            )

            # Get model and priority
            model_group: Final = get_model_group_from_litellm_kwargs(kwargs)
            if not model_group:
                return

            # Get priority from user_api_key_auth_metadata in standard_logging_metadata
            # This is where user_api_key_dict.metadata is stored during pre-call
            user_api_key_auth_metadata: Final = _OBJECT_MAP.validate_python(
                standard_logging_metadata.get("user_api_key_auth_metadata") or {}
            )
            raw_priority: Final = user_api_key_auth_metadata.get("priority")
            key_priority: Final = raw_priority if isinstance(raw_priority, str) else None

            # Get total tokens from response
            total_tokens = 0
            rate_limit_type: Final = self.v3_limiter.get_rate_limit_type()

            if isinstance(response_obj, ModelResponse):
                _usage: Final = getattr(response_obj, "usage", None)
                if _usage and isinstance(_usage, Usage):
                    if rate_limit_type == "output":
                        total_tokens = _usage.completion_tokens
                    elif rate_limit_type == "input":
                        total_tokens = _usage.prompt_tokens
                    elif rate_limit_type == "total":
                        total_tokens = _usage.total_tokens

            if total_tokens == 0:
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

            team_id: Final = _OPTIONAL_TEXT.validate_python(standard_logging_metadata.get("user_api_key_team_id"))
            team_share_tokens: Final = (
                (
                    CounterIncrement(
                        counter=RateLimitScope(TEAM_SHARE_DESCRIPTOR_KEY, f"{model_group}:{team_id}").counter("tokens"),
                        amount=total_tokens,
                        ttl=self.v3_limiter.window_size,
                    ),
                )
                if team_id and _get_priority_settings().guaranteed_team_shares
                else ()
            )
            pipeline_operations: Final = (
                model_tokens,
                CounterIncrement(
                    counter=RateLimitScope("priority_model", priority_key).counter("tokens"),
                    amount=total_tokens,
                    ttl=self.v3_limiter.window_size,
                ),
                *team_share_tokens,
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


PROXY_DynamicRateLimitHandlerV3: Final = _PROXY_DynamicRateLimitHandlerV3
