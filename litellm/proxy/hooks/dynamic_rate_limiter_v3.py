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
from litellm.types.router import ModelGroupInfo


class _PROXY_DynamicRateLimitHandlerV3(CustomLogger):
    """
    Saturation-aware priority-based rate limiter using v3 infrastructure.
    
    Key features:
    1. Reuses v3 limiter's Redis-based tracking (works across multiple instances)
    2. Only enforces priority limits when model is saturated (>80% usage)
    3. When under capacity, allows all requests (generous behavior)
    4. When saturated, enforces strict priority-based limits (fairness)
    
    How it works:
    - Uses v3 limiter's counter keys to check model-wide saturation
    - Saturation check reads existing counters without incrementing
    - Priority enforcement reuses v3 limiter's atomic Lua scripts
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

    def _create_priority_based_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: Optional[str],
    ) -> List[RateLimitDescriptor]:
        """
        Create rate limit descriptors with normalized priority weights.
        
        Uses normalized weights to handle over-allocation scenarios.
        Only called when system is saturated.
        """
        descriptors: List[RateLimitDescriptor] = []
        
        # Get model group info
        model_group_info: Optional[ModelGroupInfo] = self.llm_router.get_model_group_info(
            model_group=model
        )
        if model_group_info is None:
            return descriptors

        # Get normalized priority weight (handles over-allocation)
        normalized_weights = self._normalize_priority_weights()
        priority_weight = normalized_weights.get(priority, None) if priority else None
        if priority_weight is None:
            # Fallback to non-normalized weight
            priority_weight = self._get_priority_weight(priority)
        
        
        # Create priority-specific rate limits
        # Use model:priority as the key to separate different priority levels
        priority_key = f"{model}:{priority or 'default'}"
        
        rate_limit_config: RateLimitDescriptorRateLimitObject = {}
        
        # Apply normalized priority weight to model limits
        if model_group_info.tpm is not None:
            # Reserve portion of TPM based on normalized priority
            reserved_tpm = int(model_group_info.tpm * priority_weight)
            rate_limit_config["tokens_per_unit"] = reserved_tpm
            
        if model_group_info.rpm is not None:
            # Reserve portion of RPM based on normalized priority  
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

    async def _handle_generous_mode(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        key_priority: Optional[str],
    ) -> None:
        """
        Handle rate limiting in generous mode (under saturation threshold).
        
        In this mode, we enforce model-wide capacity but NOT priority-specific limits.
        This allows lower-priority users to borrow unused capacity from higher-priority users.
        
        Args:
            model: Model name
            model_group_info: Model configuration
            user_api_key_dict: User authentication info
            key_priority: User's priority level
            
        Raises:
            HTTPException: If model capacity is reached
        """
        descriptor = self._create_model_tracking_descriptor(
            model=model,
            model_group_info=model_group_info,
            high_limit_multiplier=1,  # Enforce actual limits in generous mode
        )
        
        response = await self.v3_limiter.should_rate_limit(
            descriptors=[descriptor],
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        
        if response["overall_code"] == "OVER_LIMIT":
            for status in response["statuses"]:
                if status["code"] == "OVER_LIMIT":
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

    async def _handle_strict_mode(
        self,
        model: str,
        model_group_info: ModelGroupInfo,
        user_api_key_dict: UserAPIKeyAuth,
        key_priority: Optional[str],
        saturation: float,
        data: dict,
    ) -> None:
        """
        Handle rate limiting in strict mode (above saturation threshold).
        
        In this mode, we enforce priority-specific limits using normalized weights.
        
        Args:
            model: Model name
            model_group_info: Model configuration
            user_api_key_dict: User authentication info
            key_priority: User's priority level
            saturation: Current saturation level
            data: Request data dictionary
            
        Raises:
            HTTPException: If priority-specific limit is exceeded
        """
        # Create priority-based descriptors
        descriptors = self._create_priority_based_descriptors(
            model=model,
            user_api_key_dict=user_api_key_dict,
            priority=key_priority,
        )

        if not descriptors:
            verbose_proxy_logger.debug("No rate limit descriptors created, allowing request")
            return

        # Track model-wide usage for future saturation checks
        # Why tracking_multiplier: v3_limiter.should_rate_limit() both increments AND checks limits.
        # We need the increment (for saturation detection) but NOT the limit check (priority limits handle enforcement).
        # Setting limit to 10x capacity ensures tracking never blocks while keeping accurate counters.
        tracking_multiplier = litellm.priority_reservation_settings.tracking_multiplier
        tracking_descriptor = self._create_model_tracking_descriptor(
            model=model,
            model_group_info=model_group_info,
            high_limit_multiplier=tracking_multiplier,
        )
        
        await self.v3_limiter.should_rate_limit(
            descriptors=[tracking_descriptor],
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        
        # Enforce priority-specific limits
        response = await self.v3_limiter.should_rate_limit(
            descriptors=descriptors,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        if response["overall_code"] == "OVER_LIMIT":
            for status in response["statuses"]:
                if status["code"] == "OVER_LIMIT":
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": f"Priority-based rate limit exceeded for {status['descriptor_key']}. "
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
        else:
            # Store response for post-call hook
            data["litellm_proxy_rate_limit_response"] = response

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
        
        This hook implements a two-mode rate limiting strategy:
        - Generous mode (< 80% saturation): Enforces model capacity, allows priority borrowing
        - Strict mode (>= 80% saturation): Enforces normalized priority-based limits
        
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

        # Check current saturation level
        try:
            saturation = await self._check_model_saturation(model, model_group_info)
            
            saturation_threshold = litellm.priority_reservation_settings.saturation_threshold
            
            verbose_proxy_logger.debug(
                f"[Dynamic Rate Limiter] Model={model}, Saturation={saturation:.1%}, "
                f"Threshold={saturation_threshold:.1%}, Priority={key_priority}"
            )
            
            data["litellm_model_saturation"] = saturation
            
            # Route to appropriate mode based on saturation
            if saturation < saturation_threshold:
                await self._handle_generous_mode(
                    model=model,
                    model_group_info=model_group_info,
                    user_api_key_dict=user_api_key_dict,
                    key_priority=key_priority,
                )
            else:
                await self._handle_strict_mode(
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
