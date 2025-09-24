"""
Dynamic rate limiter v3
"""

import os
from typing import List, Literal, Optional, Union

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
    Simple validation version that uses v3 parallel request limiter for priority-based rate limiting.
    
    Key differences from original:
    1. Uses v3 limiter's sliding window approach instead of per-minute cache buckets
    2. Leverages Redis Lua scripts for atomic operations under high traffic
    3. Creates priority-specific rate limit descriptors
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

    def _create_priority_based_descriptors(
        self,
        model: str,
        user_api_key_dict: UserAPIKeyAuth,
        priority: Optional[str],
    ) -> List[RateLimitDescriptor]:
        """
        Create rate limit descriptors based on priority and model group limits.
        
        This is the key change: instead of calculating dynamic quotas based on active projects,
        we create descriptors with priority-adjusted limits and let the v3 limiter handle
        the actual rate limiting with its sliding window approach.
        """
        descriptors: List[RateLimitDescriptor] = []
        
        # Get model group info
        model_group_info: Optional[ModelGroupInfo] = self.llm_router.get_model_group_info(
            model_group=model
        )
        if model_group_info is None:
            return descriptors

        # Get priority weight
        priority_weight = self._get_priority_weight(priority)
        
        # Create priority-specific rate limits
        # Use model:priority as the key to separate different priority levels
        priority_key = f"{model}:{priority or 'default'}"
        
        rate_limit_config: RateLimitDescriptorRateLimitObject = {}
        
        # Apply priority weight to model limits
        if model_group_info.tpm is not None:
            # Reserve portion of TPM based on priority
            reserved_tpm = int(model_group_info.tpm * priority_weight)
            rate_limit_config["tokens_per_unit"] = reserved_tpm
            
        if model_group_info.rpm is not None:
            # Reserve portion of RPM based on priority  
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
        Pre-call hook using v3 limiter for priority-based rate limiting.
        """
        if "model" not in data:
            return None

        key_priority: Optional[str] = user_api_key_dict.metadata.get("priority", None)
        
        # Create priority-based descriptors
        descriptors = self._create_priority_based_descriptors(
            model=data["model"],
            user_api_key_dict=user_api_key_dict,
            priority=key_priority,
        )

        if not descriptors:
            verbose_proxy_logger.debug("No rate limit descriptors created, allowing request")
            return None

        try:
            # Use v3 limiter to check rate limits
            response = await self.v3_limiter.should_rate_limit(
                descriptors=descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )

            if response["overall_code"] == "OVER_LIMIT":
                # Find which descriptor hit the limit
                for status in response["statuses"]:
                    if status["code"] == "OVER_LIMIT":
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": f"Priority-based rate limit exceeded for {status['descriptor_key']}. "
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
            else:
                # Store response for post-call hook
                data["litellm_proxy_rate_limit_response"] = response

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in dynamic rate limiter v3 pre-call hook: {str(e)}"
            )
            # Allow request to proceed on unexpected errors
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
