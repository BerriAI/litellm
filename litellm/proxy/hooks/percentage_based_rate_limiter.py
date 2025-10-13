"""
Percentage-based rate limiter for models without known rpm/tpm limits.

Tracks actual traffic and enforces percentage splits when saturated.
Inherits from DynamicRateLimitHandlerV3 to reuse infrastructure.
"""

import os
from typing import Dict, List, Literal, Optional, Union

from fastapi import HTTPException

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitDescriptor
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
)
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import PriorityReservationSettings


class _PROXY_PercentageBasedRateLimitHandler(_PROXY_DynamicRateLimitHandlerV3):
    """
    Percentage-based rate limiter for models WITHOUT known rpm/tpm limits.
    
    Use this when:
    - Model's rate limits are unknown
    - Want to split traffic by priority percentages when saturated
    - Saturation detected via error counts (429s, timeouts, etc.)
    
    How it works:
    1. Tracks total aggregate traffic (all priorities combined)
    2. Tracks per-priority traffic
    3. When NOT saturated: Allow all traffic through
    4. When saturated: Enforce percentage splits based on actual traffic
    
    Example:
    - priority_reservation = {"prod": 0.9, "dev": 0.1}
    - 100 requests in last 60s: prod can use 90, dev can use 10
    - 1000 requests in last 60s: prod can use 900, dev can use 100
    - Self-adjusting based on observed traffic levels
    
    Key difference from parent class:
    - Parent: Uses absolute limits (e.g., "prod gets 90 RPM")
    - This: Uses relative percentages (e.g., "prod gets 90% of traffic")
    """

    def _get_model_group_failure_count(
        self,
        model: str,
    ) -> int:
        """
        Get the total number of failures across all deployments in a model group.
        
        Reuses the router's existing failure tracking infrastructure.
        
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
        
        Reuses the router's existing failure tracking instead of creating separate counters.
        The router already tracks failures per deployment with 60s TTL.
        
        Returns:
            float: Saturation ratio (0.0 = no saturation, 1.0 = at/above threshold)
        """
        try:
            saturation_policy = litellm.priority_reservation_settings.saturation_policy  # type: ignore
            
            # Get failure threshold - use any configured threshold from saturation_policy
            # Since router tracks all failures together, we use the most permissive threshold
            failure_threshold = None
            
            if saturation_policy is not None:
                # Use RateLimitErrorSaturationThreshold as the primary threshold
                # (this is the most common use case)
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
                f"Model {model} failure-based saturation: {total_failures}/{failure_threshold} "
                f"({error_saturation:.1%})"
            )
            
            return error_saturation
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error checking error saturation for {model}: {str(e)}"
            )
            return 0.0

    def _create_aggregate_traffic_descriptor(
        self,
        model: str,
    ) -> RateLimitDescriptor:
        """
        Create descriptor for tracking total aggregate traffic across all priorities.
        
        This counter tracks ALL requests regardless of priority.
        Used as the denominator for percentage calculations.
        
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

    def _create_priority_based_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: Optional[str],
    ) -> List[RateLimitDescriptor]:
        """
        Override: Create tracking-only descriptors (no absolute limits).
        
        In percentage mode, we track per-priority counters but don't enforce
        absolute limits on them. Enforcement is based on percentage calculations
        done in _check_percentage_based_limits.
        
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

    def _check_percentage_based_limits(
        self,
        priority: Optional[str],
        normalized_weights: Dict[str, float],
        aggregate_count: int,
        priority_count: int,
    ) -> tuple[bool, str]:
        """
        Check if this priority is within their percentage allocation.
        
        Simplified: Just does the math on already-incremented counters.
        
        Args:
            priority: Priority level
            normalized_weights: Pre-computed normalized weights
            aggregate_count: Total requests across all priorities
            priority_count: Requests for this specific priority
            
        Returns:
            tuple: (is_within_limits: bool, debug_message: str)
        """
        # Get priority allocation
        priority_weight = normalized_weights.get(priority, self._get_priority_weight(priority)) if priority else self._get_priority_weight(None)
        
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

    async def _check_rate_limits(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        key_priority: Optional[str],
        saturation: float,
        data: dict,
    ) -> None:
        """
        Override: Simplified percentage-based rate limiting.
        
        Flow:
        1. Read current counters (aggregate + priority)
        2. If saturated, check if priority would exceed percentage
        3. If within limits, increment counters
        
        Args:
            model: Model name
            model_group_info: Model configuration (may have no rpm/tpm)
            user_api_key_dict: User authentication
            key_priority: User's priority level
            saturation: Current saturation level (from error tracking)
            data: Request data
            
        Raises:
            HTTPException: If percentage limit exceeded when saturated
        """
        import json

        # For percentage-based limiting, enforce when error saturation hits 100% (>= 1.0)
        # This is simpler than the parent class which uses saturation_threshold for TPM/RPM
        should_enforce_priority = saturation >= 1.0
        
        # Build descriptors for tracking
        descriptors_to_check: List[RateLimitDescriptor] = []
        
        # Aggregate traffic counter (always)
        agg_descriptor = self._create_aggregate_traffic_descriptor(model=model)
        descriptors_to_check.append(agg_descriptor)
        
        # Priority traffic counter (always)
        priority_descriptors = self._create_priority_based_descriptors(
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
        Override: Percentage-based rate limiting for models without rpm/tpm limits.
        
        Flow:
        1. Check saturation (via error counts)
        2. If not saturated: Allow all traffic
        3. If saturated: Check percentage-based limits
        4. Increment traffic counters
        
        Args:
            user_api_key_dict: User authentication and metadata
            cache: Dual cache instance
            data: Request data containing model name
            call_type: Type of API call
            
        Returns:
            None if allowed, raises HTTPException if over limit
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
            # STEP 1: Check saturation (error-based)
            saturation = self._check_error_saturation(model)
            
            saturation_threshold = litellm.priority_reservation_settings.saturation_threshold
            
            verbose_proxy_logger.debug(
                f"[Percentage Rate Limiter] Model={model}, Saturation={saturation:.1%}, "
                f"Threshold={saturation_threshold:.1%}, Priority={key_priority}"
            )
            
            # STEP 2: Check and enforce percentage-based rate limits
            await self._check_rate_limits(
                model=model,
                model_group_info=model_group_info,
                user_api_key_dict=user_api_key_dict,
                key_priority=key_priority,
                saturation=saturation,
                data=data,
            )
                
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in percentage-based rate limiter: {str(e)}, allowing request"
            )
            # Fail open on unexpected errors
            return None

        return None


# Export handler
DynamicRateLimitHandler = _PROXY_PercentageBasedRateLimitHandler

