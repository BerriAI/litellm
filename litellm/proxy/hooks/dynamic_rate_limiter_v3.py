"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting
"""

import os
from typing import Dict, List, Literal, Optional, Union

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
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
)
from litellm.types.router import ModelGroupInfo


class _PROXY_DynamicRateLimitHandlerV3(CustomLogger):
    """
    Adaptive saturation-aware priority-based rate limiter using v3 infrastructure.
    
    AUTOMATICALLY DETECTS AND SWITCHES BETWEEN TWO MODES:
    
    MODE 1: ABSOLUTE LIMITS (when model has rpm/tpm configured)
    - Enforces actual TPM/RPM limits
    - Saturation calculated from usage (e.g., 800/1000 RPM = 80%)
    - Use case: Public APIs with known limits (OpenAI, Anthropic)
    
    MODE 2: PERCENTAGE SPLITTING (when model has NO rpm/tpm)
    - Enforces traffic percentage splits
    - Saturation calculated from error counts (e.g., 5 x 429 errors)
    - Use case: Self-hosted models, Vertex AI dynamic quotas, unknown limits
    
    Key features:
    1. Automatic mode detection based on model configuration
    2. Model capacity enforcement (absolute mode) or traffic splitting (percentage mode)
    3. Priority usage tracked from first request (accurate accounting)
    4. Priority limits enforced when saturated >= threshold
    5. Three-phase checking prevents partial counter increments
    6. Reuses v3 limiter's Redis-based tracking (multi-instance safe)
    
    How it works:
    - Phase 1: Read-only check of ALL limits (no increments)
    - Phase 2: Decide enforcement based on saturation
    - Phase 3: Increment counters only if request allowed
    - When under-saturated: priorities can borrow unused capacity (generous)
    - When saturated: strict priority-based limits enforced (fair)
    - Uses v3 limiter's atomic Lua scripts for race-free increments
    """
    def __init__(self, internal_usage_cache: DualCache):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self.v3_limiter = _PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache)

    def update_variables(self, llm_router: Router):
        self.llm_router = llm_router

    def _get_priority_weight(self, priority: Optional[str]) -> float:
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
                weight = litellm.priority_reservation[priority]
        return weight

    def _normalize_priority_weights(self) -> Dict[str, float]:
        """
        Normalize priority weights if they sum to > 1.0
        
        Handles over-allocation: {key_a: 0.60, key_b: 0.80} -> {key_a: 0.43, key_b: 0.57}
        """
        if litellm.priority_reservation is None:
            return {}
        
        weights = dict(litellm.priority_reservation)
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
    ) -> tuple[float, str]:
        """
        Get priority weight and pool key for a given priority.
        
        For explicit priorities: returns specific allocation and unique pool key
        For default priority: returns default allocation and shared pool key
        
        Args:
            model: Model name
            priority: Priority level (None for default)
            normalized_weights: Pre-computed normalized weights
            
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
            priority_weight = normalized_weights.get(priority, self._get_priority_weight(priority))
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
            
            # Query RPM saturation
            if model_group_info.rpm is not None and model_group_info.rpm > 0:
                # Use v3 limiter's key format: {key:value}:rate_limit_type
                counter_key = self.v3_limiter.create_rate_limit_keys(
                    key="model_saturation_check",
                    value=model,
                    rate_limit_type="requests",
                )
                
                # Query cache for current counter value
                counter_value = await self.internal_usage_cache.async_get_cache(
                    key=counter_key,
                    litellm_parent_otel_span=None,
                    local_only=False,  # Check Redis too
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
                
                counter_value = await self.internal_usage_cache.async_get_cache(
                    key=counter_key,
                    litellm_parent_otel_span=None,
                    local_only=False,
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

    ###############################################################################
    # MODE DETECTION & ERROR-BASED SATURATION (for models without rpm/tpm limits)
    ###############################################################################

    def _has_explicit_limits(self, model_group_info: ModelGroupInfo) -> bool:
        """
        Check if model has explicit rpm/tpm limits configured.
        
        Used to automatically detect which rate limiting mode to use:
        - Has limits → Use absolute TPM/RPM enforcement
        - No limits → Use percentage-based traffic splitting
        
        Args:
            model_group_info: Model configuration
            
        Returns:
            True if model has rpm or tpm configured, False otherwise
        """
        return (
            (model_group_info.rpm is not None and model_group_info.rpm > 0)
            or (model_group_info.tpm is not None and model_group_info.tpm > 0)
        )

    def _get_model_group_failure_count(
        self,
        model: str,
    ) -> int:
        """
        Get the total number of failures across all deployments in a model group.
        
        Reuses the router's existing failure tracking infrastructure (60s window).
        
        Args:
            model: Model group name
            
        Returns:
            int: Total failure count across all deployments
        """
        try:
            # Get all deployment IDs for this model group
            deployment_ids = self.llm_router.get_model_ids(model_name=model)
            
            if not deployment_ids:
                return 0
            
            # Sum up failures across all deployments
            total_failures = 0
            for deployment_id in deployment_ids:
                failures = get_deployment_failures_for_current_minute(
                    litellm_router_instance=self.llm_router,
                    deployment_id=deployment_id,
                )
                total_failures += failures
            
            verbose_proxy_logger.debug(
                f"Model {model} total failures: {total_failures} "
                f"(across {len(deployment_ids)} deployments)"
            )
            
            return total_failures
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error getting failure count for {model}: {str(e)}"
            )
            return 0

    def _check_error_saturation(
        self,
        model: str,
    ) -> float:
        """
        Check if model is saturated based on failure counts from router's tracking.
        
        Reuses the router's existing failure tracking (60s TTL) instead of creating 
        separate counters. Used in percentage mode when rpm/tpm limits are unknown.
        
        Args:
            model: Model group name
        
        Returns:
            float: Saturation ratio (0.0 = no saturation, 1.0 = at/above threshold)
        """
        try:
            saturation_policy = litellm.priority_reservation_settings.saturation_policy  # type: ignore
            
            # Get failure threshold from saturation_policy
            failure_threshold = None
            
            if saturation_policy is not None:
                # Use RateLimitErrorSaturationThreshold as the primary threshold
                failure_threshold = saturation_policy.RateLimitErrorSaturationThreshold
                
                # If not set, try other thresholds
                if failure_threshold is None:
                    thresholds = [
                        saturation_policy.TimeoutErrorSaturationThreshold,
                        saturation_policy.InternalServerErrorSaturationThreshold,
                        saturation_policy.ServiceUnavailableErrorSaturationThreshold,
                        saturation_policy.BadRequestErrorSaturationThreshold,
                    ]
                    # Use the first non-None threshold
                    for threshold in thresholds:
                        if threshold is not None and threshold > 0:
                            failure_threshold = threshold
                            break

            if failure_threshold is None or failure_threshold <= 0:
                return 0.0
            
            # Get total failures from router's existing tracking
            total_failures = self._get_model_group_failure_count(model)
            
            if total_failures == 0:
                return 0.0
            
            # Calculate saturation: 1.0 if at/above threshold, proportional below
            error_saturation = min(1.0, total_failures / failure_threshold)
            
            verbose_proxy_logger.debug(
                f"Model {model} error-based saturation: {total_failures}/{failure_threshold} "
                f"({error_saturation:.1%})"
            )
            
            return error_saturation
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error checking error saturation for {model}: {str(e)}"
            )
            return 0.0

    ###############################################################################
    # DESCRIPTOR CREATION (for both modes)
    ###############################################################################

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
        model_group_info: Optional[ModelGroupInfo] = self.llm_router.get_model_group_info(
            model_group=model
        )
        if model_group_info is None:
            return descriptors

        # Get normalized priority weight and pool key
        normalized_weights = self._normalize_priority_weights()
        priority_weight, priority_key = self._get_priority_allocation(
            model=model,
            priority=priority,
            normalized_weights=normalized_weights,
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
                    if model_group_info.rpm else None
                ),
                "tokens_per_unit": (
                    model_group_info.tpm * high_limit_multiplier 
                    if model_group_info.tpm else None
                ),
                "window_size": self.v3_limiter.window_size,
            },
        )

    def _create_aggregate_traffic_descriptor(
        self,
        model: str,
    ) -> RateLimitDescriptor:
        """
        Create descriptor for tracking total aggregate traffic across all priorities.
        
        Used in percentage mode to track ALL requests regardless of priority.
        This counter serves as the denominator for percentage calculations.
        
        Uses very high limit so it never actually blocks, but gets tracked.
        
        Args:
            model: Model name
            
        Returns:
            Descriptor that tracks total traffic without enforcing limits
        """
        return RateLimitDescriptor(
            key="aggregate_traffic",
            value=model,
            rate_limit={
                "requests_per_unit": 999999999,  # Very high limit - tracking only
                "window_size": self.v3_limiter.window_size,
            },
        )

    def _create_priority_traffic_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: Optional[str],
    ) -> List[RateLimitDescriptor]:
        """
        Create tracking-only descriptors for percentage mode.
        
        In percentage mode, we track per-priority counters but don't enforce
        absolute limits on them. Enforcement is based on percentage calculations.
        
        Args:
            model: Model name
            user_api_key_dict: User authentication
            priority: Priority level
            
        Returns:
            List with single descriptor for tracking this priority's traffic
        """
        descriptors: List[RateLimitDescriptor] = []
        
        # Get normalized priority weight and pool key
        normalized_weights = self._normalize_priority_weights()
        _, priority_key = self._get_priority_allocation(
            model=model,
            priority=priority,
            normalized_weights=normalized_weights,
        )
        
        # Create tracking-only descriptor (uses high limit so it gets tracked but never blocks)
        descriptors.append(
            RateLimitDescriptor(
                key="priority_traffic",
                value=priority_key,
                rate_limit={
                    "requests_per_unit": 999999999,  # Very high limit - tracking only
                    "window_size": self.v3_limiter.window_size,
                },
            )
        )

        return descriptors

    ###############################################################################
    # RATE LIMIT CHECKING (mode-aware)
    ###############################################################################

    async def _check_rate_limits(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        key_priority: Optional[str],
        saturation: float,
        data: dict,
        mode: Literal["absolute", "percentage"] = "absolute",
    ) -> None:
        """
        Check rate limits using THREE-PHASE approach to prevent partial increments.
        
        ADAPTIVE: Automatically uses absolute or percentage mode based on model config.
        
        MODE: ABSOLUTE (when model has rpm/tpm)
        - Phase 1: Read-only check of model + priority limits
        - Phase 2: Enforce model capacity (always) + priority limits (when saturated)
        - Phase 3: Increment model + priority counters
        
        MODE: PERCENTAGE (when model has NO rpm/tpm)
        - Phase 1: Read aggregate + priority traffic counts
        - Phase 2: Enforce percentage splits (when saturated by errors)
        - Phase 3: Increment aggregate + priority counters
        
        This prevents the bug where:
        - Counter increments in phase 1
        - Limit check fails in phase 2
        - Request blocked but counter already incremented
        
        Args:
            model: Model name
            model_group_info: Model configuration
            user_api_key_dict: User authentication info
            key_priority: User's priority level
            saturation: Current saturation level
            data: Request data dictionary
            mode: "absolute" (rpm/tpm enforcement) or "percentage" (traffic split)
            
        Raises:
            HTTPException: If any limit is exceeded
        """
        import json

        ###############################################################################
        # MODE BRANCHING: Absolute vs Percentage
        ###############################################################################
        
        if mode == "percentage":
            # Percentage mode uses error saturation (>= 1.0 = saturated)
            should_enforce_priority = saturation >= 1.0
            
            # Delegate to percentage-specific logic
            await self._check_percentage_rate_limits(
                model=model,
                user_api_key_dict=user_api_key_dict,
                key_priority=key_priority,
                saturation=saturation,
            )
            return
        
        ###############################################################################
        # ABSOLUTE MODE (traditional rpm/tpm enforcement)
        ###############################################################################
        
        saturation_threshold = litellm.priority_reservation_settings.saturation_threshold
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
            priority=key_priority,
        )
        if priority_descriptors:
            descriptors_to_check.extend(priority_descriptors)
        
        # PHASE 1: Read-only check of ALL limits (no increments)
        check_response = await self.v3_limiter.should_rate_limit(
            descriptors=descriptors_to_check,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            read_only=True,  # CRITICAL: Don't increment counters yet
        )
        
        verbose_proxy_logger.debug(f"Read-only check: {json.dumps(check_response, indent=2)}")
        
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
                                        f"Priority: {key_priority}, "
                                        f"Rate limit type: {status['rate_limit_type']}, "
                                        f"Remaining: {status['limit_remaining']}"
                            },
                            headers={
                                "retry-after": str(self.v3_limiter.window_size),
                                "rate_limit_type": str(status["rate_limit_type"]),
                                "x-litellm-priority": key_priority or "default",
                            },
                        )
                    
                    # Priority limit exceeded (ONLY enforce when saturated)
                    elif descriptor_key == "priority_model" and should_enforce_priority:
                        verbose_proxy_logger.debug(
                            f"Enforcing priority limits for {model}, saturation: {saturation:.1%}, "
                            f"priority: {key_priority}"
                        )
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": f"Priority-based rate limit exceeded. "
                                        f"Priority: {key_priority}, "
                                        f"Rate limit type: {status['rate_limit_type']}, "
                                        f"Remaining: {status['limit_remaining']}, "
                                        f"Model saturation: {saturation:.1%}"
                            },
                            headers={
                                "retry-after": str(self.v3_limiter.window_size),
                                "rate_limit_type": str(status["rate_limit_type"]),
                                "x-litellm-priority": key_priority or "default",
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
                "statuses": model_increment_response["statuses"] + priority_increment_response["statuses"]
            }
            data["litellm_proxy_rate_limit_response"] = combined_response
        else:
            data["litellm_proxy_rate_limit_response"] = model_increment_response

    async def _check_percentage_rate_limits(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        key_priority: Optional[str],
        saturation: float,
    ) -> None:
        """
        Percentage-based rate limiting for models without rpm/tpm limits.
        
        Flow:
        1. Read current counters (aggregate + priority)
        2. If saturated, check if priority would exceed percentage
        3. If within limits, increment counters
        
        Args:
            model: Model name
            user_api_key_dict: User authentication
            key_priority: User's priority level
            saturation: Current saturation level (from error tracking)
            
        Raises:
            HTTPException: If percentage limit exceeded when saturated
        """
        import json

        # For percentage-based limiting, enforce when error saturation hits 100% (>= 1.0)
        should_enforce_priority = saturation >= 1.0
        
        # Build descriptors for tracking
        descriptors_to_check: List[RateLimitDescriptor] = []
        
        # Aggregate traffic counter (always)
        agg_descriptor = self._create_aggregate_traffic_descriptor(model=model)
        descriptors_to_check.append(agg_descriptor)
        
        # Priority traffic counter (always)
        priority_descriptors = self._create_priority_traffic_descriptors(
            model=model,
            user_api_key_dict=user_api_key_dict,
            priority=key_priority,
        )
        descriptors_to_check.extend(priority_descriptors)
        
        # STEP 1: Read current counts (read-only)
        read_response = await self.v3_limiter.should_rate_limit(
            descriptors=descriptors_to_check,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            read_only=True,  # Don't increment yet
        )
        
        verbose_proxy_logger.debug(f"Read response: {json.dumps(read_response, indent=2)}")
        
        # Extract counts from response
        aggregate_count = 0
        priority_count = 0
        
        for status in read_response.get("statuses", []):
            descriptor_key = status.get("descriptor_key")
            # Calculate current usage from limit
            current_limit = status.get("current_limit", 0)
            limit_remaining = status.get("limit_remaining", 0)
            current_usage = current_limit - limit_remaining
            
            # Match on descriptor key from the RateLimitDescriptor
            if descriptor_key == "aggregate_traffic":
                aggregate_count = int(current_usage)
            elif descriptor_key == "priority_traffic":
                priority_count = int(current_usage)
        
        # STEP 2: Check percentage limits if saturated (BEFORE incrementing)
        if should_enforce_priority and aggregate_count > 0:
            normalized_weights = self._normalize_priority_weights()
            
            # Check with the NEXT request included
            is_within_limits, debug_msg = self._check_percentage_based_limits(
                priority=key_priority,
                normalized_weights=normalized_weights,
                aggregate_count=aggregate_count + 1,  # Include this request
                priority_count=priority_count + 1,    # Include this request
            )
            
            verbose_proxy_logger.info(
                f"[Percentage Rate Limiter] Model={model}, Saturated={saturation:.1%}, {debug_msg}"
            )
            
            if not is_within_limits:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": f"Priority-based percentage limit exceeded. {debug_msg}",
                        "model": model,
                        "priority": key_priority,
                        "saturation": f"{saturation:.1%}",
                    },
                    headers={
                        "retry-after": str(self.v3_limiter.window_size),
                        "x-litellm-priority": key_priority or "default",
                        "x-litellm-saturation": f"{saturation:.2%}",
                        "x-litellm-limiter-type": "percentage-based",
                    },
                )
        
        # STEP 3: Increment counters (only if request allowed)
        await self.v3_limiter.should_rate_limit(
            descriptors=descriptors_to_check,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            read_only=False,
        )

    def _check_percentage_based_limits(
        self,
        priority: Optional[str],
        normalized_weights: Dict[str, float],
        aggregate_count: int,
        priority_count: int,
    ) -> tuple[bool, str]:
        """
        Check if this priority is within their percentage allocation.
        
        Args:
            priority: Priority level
            normalized_weights: Pre-computed normalized weights
            aggregate_count: Total requests across all priorities
            priority_count: Requests for this specific priority
            
        Returns:
            tuple: (is_within_limits: bool, debug_message: str)
        """
        # Get priority allocation
        priority_weight = (
            normalized_weights.get(priority, self._get_priority_weight(priority))
            if priority
            else self._get_priority_weight(None)
        )
        
        # Calculate current and allowed shares
        current_share = priority_count / aggregate_count if aggregate_count > 0 else 0.0
        allowed_share = priority_weight
        
        # Check if within limits (with small buffer to avoid edge cases)
        buffer = 0.01
        is_within_limits = current_share <= (allowed_share + buffer)
        
        debug_msg = (
            f"Priority {priority}: {priority_count}/{aggregate_count} requests "
            f"({current_share:.1%} of total, allowed: {allowed_share:.1%})"
        )
        
        return is_within_limits, debug_msg

    ###############################################################################
    # PRE-CALL HOOK (adaptive mode detection)
    ###############################################################################

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion", 
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
        ],
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
        key_priority: Optional[str] = user_api_key_dict.metadata.get("priority", None)
        
        # Get model configuration
        model_group_info: Optional[ModelGroupInfo] = self.llm_router.get_model_group_info(
            model_group=model
        )
        if model_group_info is None:
            verbose_proxy_logger.debug(f"No model group info for {model}, allowing request")
            return None

        try:
            ###################################################################
            # STEP 1: AUTOMATIC MODE DETECTION
            ###################################################################
            
            has_explicit_limits = self._has_explicit_limits(model_group_info)
            
            if has_explicit_limits:
                # MODE: ABSOLUTE (traditional rpm/tpm enforcement)
                saturation = await self._check_model_saturation(model, model_group_info)
                mode: Literal["absolute", "percentage"] = "absolute"
                saturation_threshold = litellm.priority_reservation_settings.saturation_threshold
                
                verbose_proxy_logger.debug(
                    f"[Dynamic Rate Limiter V3] Mode=ABSOLUTE, Model={model}, "
                    f"Saturation={saturation:.1%}, Threshold={saturation_threshold:.1%}, "
                    f"Priority={key_priority}"
                )
            else:
                # MODE: PERCENTAGE (error-based traffic splitting)
                saturation = self._check_error_saturation(model)
                mode = "percentage"
                
                verbose_proxy_logger.debug(
                    f"[Dynamic Rate Limiter V3] Mode=PERCENTAGE, Model={model}, "
                    f"Error Saturation={saturation:.1%}, Priority={key_priority}"
                )
            
            ###################################################################
            # STEP 2: CHECK RATE LIMITS (mode-aware)
            ###################################################################
            
            # Three-phase checking prevents partial increments:
            # Phase 1: Read-only check of ALL limits (no increments)
            # Phase 2: Decide which limits to enforce (based on saturation & mode)
            # Phase 3: Increment ALL counters only if request will be allowed
            await self._check_rate_limits(
                model=model,
                model_group_info=model_group_info,
                user_api_key_dict=user_api_key_dict,
                key_priority=key_priority,
                saturation=saturation,
                data=data,
                mode=mode,  # Pass detected mode
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
                key_priority: Optional[str] = user_api_key_dict.metadata.get("priority", None)
                
                # Get existing additional headers
                additional_headers = getattr(response, "_hidden_params", {}).get("additional_headers", {}) or {}
                
                # Add priority information
                additional_headers["x-litellm-priority"] = key_priority or "default"
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
