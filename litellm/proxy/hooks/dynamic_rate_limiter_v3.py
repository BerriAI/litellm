"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting
"""

import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

from fastapi import HTTPException

import litellm
from litellm import ModelResponse, Router
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.hooks.rate_limiter_utils import convert_priority_to_percent
from litellm.proxy.utils import InternalUsageCache
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import CallTypesLiteral


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
        time_provider: Optional[Callable[[], datetime]] = None,
    ):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self.v3_limiter = _PROXY_MaxParallelRequestsHandler_v3(
            self.internal_usage_cache, time_provider=time_provider
        )

    def update_variables(self, llm_router: Router):
        self.llm_router = llm_router

    def _get_saturation_check_cache_ttl(self) -> int:
        """Get the configurable TTL for local cache when reading saturation values."""
        return litellm.priority_reservation_settings.saturation_check_cache_ttl

    async def _get_saturation_value_from_cache(
        self,
        counter_key: str,
    ) -> Optional[str]:
        """
        Get saturation value with configurable local cache TTL.

        Uses DualCache with configurable TTL for local cache storage.
        TTL is configurable via litellm.priority_reservation_settings.saturation_check_cache_ttl

        Args:
            counter_key: The cache key for the saturation counter

        Returns:
            Counter value as string, or None if not found
        """
        local_cache_ttl = self._get_saturation_check_cache_ttl()

        return await self.internal_usage_cache.async_get_cache(
            key=counter_key,
            litellm_parent_otel_span=None,
            local_only=False,
            ttl=local_cache_ttl,
        )

    def _get_priority_weight(
        self, priority: Optional[str], model_info: Optional[ModelGroupInfo] = None
    ) -> float:
        """Get the weight for a given priority from litellm.priority_reservation"""
        weight: float = litellm.priority_reservation_settings.default_priority
        if (
            litellm.priority_reservation is None
            or priority not in litellm.priority_reservation
        ):
            verbose_proxy_logger.debug(
                "Priority Reservation not set for the given priority."
            )
        elif priority is not None and litellm.priority_reservation is not None:
            if os.getenv("LITELLM_LICENSE", None) is None:
                verbose_proxy_logger.error(
                    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\nGet a license: https://docs.litellm.ai/docs/proxy/enterprise."
                )
            else:
                value = litellm.priority_reservation[priority]
                weight = convert_priority_to_percent(value, model_info)
        return weight

    def _get_priority_from_user_api_key_dict(
        self, user_api_key_dict: UserAPIKeyAuth
    ) -> Optional[str]:
        """
        Get priority from user_api_key_dict.
        
        Checks team metadata first (takes precedence), then falls back to key metadata.
        
        Args:
            user_api_key_dict: User authentication info
            
        Returns:
            Priority string if found, None otherwise
        """
        priority: Optional[str] = None
        
        # Check team metadata first (takes precedence)
        if user_api_key_dict.team_metadata is not None:
            priority = user_api_key_dict.team_metadata.get("priority", None)
        
        # Fall back to key metadata
        if priority is None:
            priority = user_api_key_dict.metadata.get("priority", None)
            
        return priority

    def _normalize_priority_weights(
        self, model_info: ModelGroupInfo
    ) -> Dict[str, float]:
        """
        Normalize priority weights if they sum to > 1.0

        Handles over-allocation: {key_a: 0.60, key_b: 0.80} -> {key_a: 0.43, key_b: 0.57}
        Converts absolute rpm/tpm values to percentages based on model capacity.
        """
        if litellm.priority_reservation is None:
            return {}

        # Convert all values to percentages first
        weights: Dict[str, float] = {}
        for k, v in litellm.priority_reservation.items():
            weights[k] = convert_priority_to_percent(v, model_info)

        total_weight = sum(weights.values())

        if total_weight > 1.0:
            normalized = {k: v / total_weight for k, v in weights.items()}
            verbose_proxy_logger.debug(
                f"Normalized over-allocated priorities: {weights} -> {normalized}"
            )
            return normalized

        return weights

    def _get_priority_allocation(
        self,
        model: str,
        priority: Optional[str],
        normalized_weights: Dict[str, float],
        model_info: Optional[ModelGroupInfo] = None,
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
        has_explicit_priority = (
            priority is not None
            and litellm.priority_reservation is not None
            and priority in litellm.priority_reservation
        )

        if has_explicit_priority and priority is not None:
            # Explicit priority: get its specific allocation
            priority_weight = normalized_weights.get(
                priority, self._get_priority_weight(priority, model_info)
            )
            # Use unique key per priority level
            priority_key = f"{model}:{priority}"
        else:
            # No explicit priority: share the default_priority pool with ALL other default keys
            priority_weight = litellm.priority_reservation_settings.default_priority
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
                counter_value = await self._get_saturation_value_from_cache(
                    counter_key=counter_key
                )

                if counter_value is not None:
                    current_requests = int(counter_value)
                    rpm_saturation = current_requests / model_group_info.rpm
                    max_saturation = max(max_saturation, rpm_saturation)

                    verbose_proxy_logger.debug(
                        f"Model {model} RPM: {current_requests}/{model_group_info.rpm} "
                        f"({rpm_saturation:.1%})"
                    )

            # Query TPM saturation
            if model_group_info.tpm is not None and model_group_info.tpm > 0:
                counter_key = self.v3_limiter.create_rate_limit_keys(
                    key="model_saturation_check",
                    value=model,
                    rate_limit_type="tokens",
                )

                counter_value = await self._get_saturation_value_from_cache(
                    counter_key=counter_key
                )

                if counter_value is not None:
                    current_tokens = float(counter_value)
                    tpm_saturation = current_tokens / model_group_info.tpm
                    max_saturation = max(max_saturation, tpm_saturation)

                    verbose_proxy_logger.debug(
                        f"Model {model} TPM: {current_tokens}/{model_group_info.tpm} "
                        f"({tpm_saturation:.1%})"
                    )

            verbose_proxy_logger.debug(
                f"Model {model} overall saturation: {max_saturation:.1%}"
            )

            return max_saturation

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error checking saturation for {model}: {str(e)}"
            )
            # Fail open: assume not saturated on error
            return 0.0

    def _create_priority_based_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: Optional[str],
    ) -> List[RateLimitDescriptor]:
        """
        Create rate limit descriptors with normalized priority weights.

        Uses normalized weights to handle over-allocation scenarios.

        For explicit priorities: each priority gets its own pool (e.g., prod gets 75%)
        For default priority: ALL keys without explicit priority share ONE pool (e.g., all share 25%)
        """
        descriptors: List[RateLimitDescriptor] = []

        # Get model group info
        model_group_info: Optional[ModelGroupInfo] = (
            self.llm_router.get_model_group_info(model_group=model)
        )
        if model_group_info is None:
            return descriptors

        # Get normalized priority weight and pool key
        normalized_weights = self._normalize_priority_weights(model_group_info)
        priority_weight, priority_key = self._get_priority_allocation(
            model=model,
            priority=priority,
            normalized_weights=normalized_weights,
            model_info=model_group_info,
        )

        rate_limit_config: RateLimitDescriptorRateLimitObject = {}

        # Apply priority weight to model limits
        if model_group_info.tpm is not None:
            reserved_tpm = int(model_group_info.tpm * priority_weight)
            rate_limit_config["tokens_per_unit"] = reserved_tpm

        if model_group_info.rpm is not None:
            reserved_rpm = int(model_group_info.rpm * priority_weight)
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
                "requests_per_unit": (
                    model_group_info.rpm * high_limit_multiplier
                    if model_group_info.rpm
                    else None
                ),
                "tokens_per_unit": (
                    model_group_info.tpm * high_limit_multiplier
                    if model_group_info.tpm
                    else None
                ),
                "window_size": self.v3_limiter.window_size,
            },
        )

    async def _check_rate_limits(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        priority: Optional[str],
        saturation: float,
        data: dict,
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
            data: Request data dictionary

        Raises:
            HTTPException: If any limit is exceeded
        """
        import json

        saturation_threshold = (
            litellm.priority_reservation_settings.saturation_threshold
        )
        should_enforce_priority = saturation >= saturation_threshold

        # Build ALL descriptors upfront
        descriptors_to_check: List[RateLimitDescriptor] = []

        # Model-wide descriptor (always enforce)
        model_wide_descriptor = self._create_model_tracking_descriptor(
            model=model,
            model_group_info=model_group_info,
            high_limit_multiplier=1,
        )
        descriptors_to_check.append(model_wide_descriptor)

        # Priority descriptors (always track, conditionally enforce)
        priority_descriptors = self._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user_api_key_dict,
            priority=priority,
        )
        if priority_descriptors:
            descriptors_to_check.extend(priority_descriptors)

        # PHASE 1: Read-only check of ALL limits (no increments)
        check_response = await self.v3_limiter.should_rate_limit(
            descriptors=descriptors_to_check,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            read_only=True,  # CRITICAL: Don't increment counters yet
        )

        verbose_proxy_logger.debug(
            f"Read-only check: {json.dumps(check_response, indent=2)}"
        )

        # PHASE 2: Decide which limits to enforce
        if check_response["overall_code"] == "OVER_LIMIT":
            for status in check_response["statuses"]:
                if status["code"] == "OVER_LIMIT":
                    descriptor_key = status["descriptor_key"]

                    # Model-wide limit exceeded (ALWAYS enforce)
                    if descriptor_key == "model_saturation_check":
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": f"Model capacity reached for {model}. "
                                f"Priority: {priority}, "
                                f"Rate limit type: {status['rate_limit_type']}, "
                                f"Remaining: {status['limit_remaining']}"
                            },
                            headers={
                                "retry-after": str(self.v3_limiter.window_size),
                                "rate_limit_type": str(status["rate_limit_type"]),
                                "x-litellm-priority": priority or "default",
                            },
                        )

                    # Priority limit exceeded (ONLY enforce when saturated)
                    elif descriptor_key == "priority_model" and should_enforce_priority:
                        verbose_proxy_logger.debug(
                            f"Enforcing priority limits for {model}, saturation: {saturation:.1%}, "
                            f"priority: {priority}"
                        )
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": f"Priority-based rate limit exceeded. "
                                f"Priority: {priority}, "
                                f"Rate limit type: {status['rate_limit_type']}, "
                                f"Remaining: {status['limit_remaining']}, "
                                f"Model saturation: {saturation:.1%}"
                            },
                            headers={
                                "retry-after": str(self.v3_limiter.window_size),
                                "rate_limit_type": str(status["rate_limit_type"]),
                                "x-litellm-priority": priority or "default",
                                "x-litellm-saturation": f"{saturation:.2%}",
                            },
                        )

        # PHASE 3: Increment counters separately to avoid early-exit issues
        # Model counter must ALWAYS increment, but priority counter might be over limit
        # If we increment them together, v3_limiter's in-memory check will exit early
        # and skip incrementing the model counter

        # Step 3a: Increment model-wide counter (always)
        model_increment_response = await self.v3_limiter.should_rate_limit(
            descriptors=[model_wide_descriptor],
            parent_otel_span=user_api_key_dict.parent_otel_span,
            read_only=False,
        )

        # Step 3b: Increment priority counter (may be over limit, but we still track it)
        if priority_descriptors:
            priority_increment_response = await self.v3_limiter.should_rate_limit(
                descriptors=priority_descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                read_only=False,
            )

            # Combine responses for post-call hook
            combined_response = {
                "overall_code": model_increment_response["overall_code"],
                "statuses": model_increment_response["statuses"]
                + priority_increment_response["statuses"],
            }
            data["litellm_proxy_rate_limit_response"] = combined_response
        else:
            data["litellm_proxy_rate_limit_response"] = model_increment_response

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
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

        model = data["model"]
        priority = self._get_priority_from_user_api_key_dict(
            user_api_key_dict=user_api_key_dict
        )

        # Get model configuration
        model_group_info: Optional[ModelGroupInfo] = (
            self.llm_router.get_model_group_info(model_group=model)
        )
        if model_group_info is None:
            verbose_proxy_logger.debug(
                f"No model group info for {model}, allowing request"
            )
            return None

        try:
            # STEP 1: Check current saturation level
            saturation = await self._check_model_saturation(model, model_group_info)

            saturation_threshold = (
                litellm.priority_reservation_settings.saturation_threshold
            )

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
            )

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in dynamic rate limiter: {str(e)}, allowing request"
            )
            # Fail open on unexpected errors
            return None

        return None

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        """
        Post-call hook to add rate limit headers to response.
        Leverages v3 limiter's post-call hook functionality.
        """
        try:
            # Call v3 limiter's post-call hook to add standard rate limit headers
            await self.v3_limiter.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )

            # Add additional priority-specific headers
            if isinstance(response, ModelResponse):
                priority = self._get_priority_from_user_api_key_dict(
                    user_api_key_dict=user_api_key_dict
                )

                # Get existing additional headers
                additional_headers = (
                    getattr(response, "_hidden_params", {}).get(
                        "additional_headers", {}
                    )
                    or {}
                )

                # Add priority information
                additional_headers["x-litellm-priority"] = priority or "default"
                additional_headers["x-litellm-rate-limiter-version"] = "v3"

                # Update response
                if not hasattr(response, "_hidden_params"):
                    response._hidden_params = {}
                response._hidden_params["additional_headers"] = additional_headers

            return response

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in dynamic rate limiter v3 post-call hook: {str(e)}"
            )
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
        from litellm.types.caching import RedisPipelineIncrementOperation
        from litellm.types.utils import Usage

        try:
            verbose_proxy_logger.debug(
                "INSIDE dynamic rate limiter ASYNC SUCCESS LOGGING"
            )

            litellm_parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)

            # Get metadata from standard_logging_object
            standard_logging_object = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata = standard_logging_object.get("metadata") or {}

            # Get model and priority
            model_group = get_model_group_from_litellm_kwargs(kwargs)
            if not model_group:
                return

            # Get priority from user_api_key_auth_metadata in standard_logging_metadata
            # This is where user_api_key_dict.metadata is stored during pre-call
            user_api_key_auth_metadata = standard_logging_metadata.get("user_api_key_auth_metadata") or {}
            key_priority: Optional[str] = user_api_key_auth_metadata.get("priority")

            # Get total tokens from response
            total_tokens = 0
            rate_limit_type = self.v3_limiter.get_rate_limit_type()

            if isinstance(response_obj, ModelResponse):
                _usage = getattr(response_obj, "usage", None)
                if _usage and isinstance(_usage, Usage):
                    if rate_limit_type == "output":
                        total_tokens = _usage.completion_tokens
                    elif rate_limit_type == "input":
                        total_tokens = _usage.prompt_tokens
                    elif rate_limit_type == "total":
                        total_tokens = _usage.total_tokens

            if total_tokens == 0:
                return

            # Create pipeline operations for token increments
            pipeline_operations: List[RedisPipelineIncrementOperation] = []

            # Model-wide token tracking (model_saturation_check)
            model_token_key = self.v3_limiter.create_rate_limit_keys(
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
            has_explicit_priority = (
                key_priority is not None
                and litellm.priority_reservation is not None
                and key_priority in litellm.priority_reservation
            )

            if has_explicit_priority and key_priority is not None:
                priority_key = f"{model_group}:{key_priority}"
            else:
                priority_key = f"{model_group}:default_pool"

            priority_token_key = self.v3_limiter.create_rate_limit_keys(
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
                SAFE_PRIORITIES = {"low", "medium", "high", "default"}
                logged_priority = key_priority if key_priority in SAFE_PRIORITIES else "REDACTED"
                verbose_proxy_logger.debug(
                    f"[Dynamic Rate Limiter] Incremented tokens by {total_tokens} for "
                    f"model={model_group}, priority={logged_priority}"
                )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in dynamic rate limiter success event: {str(e)}"
            )
