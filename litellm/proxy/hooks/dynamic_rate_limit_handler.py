"""
Dynamic Rate Limit Handler

Handles provider and error-type specific rate limit policies.
Tracks failures by provider and error type to enable fine-grained dynamic rate limiting.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.constants import DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE
from litellm.types.router import DynamicRateLimitPolicy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router as _Router

    LitellmRouter = _Router
    Span = _Span
else:
    LitellmRouter = Any
    Span = Any


class DynamicRateLimitHandler:
    """
    Handles dynamic rate limiting based on provider-specific and error-type-specific thresholds.
    
    Supports any exception type - users can configure thresholds for any error by using
    the pattern: {ExceptionClassName}Threshold
    
    Example:
        CustomTimeoutError -> CustomTimeoutErrorThreshold
        MySpecialError -> MySpecialErrorThreshold
    """

    def __init__(self):
        pass

    def _get_cache_key_for_deployment_error(
        self,
        deployment_id: str,
        provider: str,
        error_type: str,
    ) -> str:
        """
        Generate cache key for tracking provider and error type specific failures.

        Args:
            deployment_id: Deployment ID
            provider: Provider name (e.g., "openai", "bedrock")
            error_type: Exception class name (e.g., "BadRequestError")

        Returns:
            str: Cache key
        """
        return f"{deployment_id}:{provider}:{error_type}:fails"

    def increment_deployment_failure_for_error_type(
        self,
        litellm_router_instance: LitellmRouter,
        deployment_id: str,
        provider: str,
        error_type: str,
    ) -> None:
        """
        Increment the failure count for a specific deployment, provider, and error type.
        
        Uses Redis cache (not local-only) to ensure failure counts are shared across
        all proxy instances in a distributed setup.

        Args:
            litellm_router_instance: Router instance
            deployment_id: Deployment ID
            provider: Provider name
            error_type: Exception class name
        """
        key = self._get_cache_key_for_deployment_error(
            deployment_id=deployment_id,
            provider=provider,
            error_type=error_type,
        )
        litellm_router_instance.cache.increment_cache(
            local_only=False,  # Use Redis to share failure counts across proxy instances
            key=key,
            value=1,
            ttl=60,
        )

    def get_deployment_failures_by_error_type(
        self,
        litellm_router_instance: LitellmRouter,
        deployment_id: str,
        provider: str,
        error_type: str,
    ) -> int:
        """
        Get failure count for a specific deployment, provider, and error type.
        
        Uses Redis cache (not local-only) to get failure counts that are shared across
        all proxy instances in a distributed setup.

        Args:
            litellm_router_instance: Router instance
            deployment_id: Deployment ID
            provider: Provider name
            error_type: Exception class name

        Returns:
            int: Number of failures (0 if not found)
        """
        key = self._get_cache_key_for_deployment_error(
            deployment_id=deployment_id,
            provider=provider,
            error_type=error_type,
        )
        return (
            litellm_router_instance.cache.get_cache(
                local_only=False,  # Use Redis to get shared failure counts across proxy instances
                key=key,
            )
            or 0
        )

    def get_threshold_for_provider_and_error(
        self,
        provider: Optional[str],
        error_type: str,
        dynamic_rate_limit_policy: Optional[DynamicRateLimitPolicy],
        default_threshold: int = DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE,
    ) -> int:
        """
        Get the error threshold for a specific provider and error type.
        
        Dynamically constructs the threshold field name from the exception class name
        by appending "Threshold". This allows users to configure thresholds for ANY
        exception type without code changes.
        
        Example:
            BadRequestError -> BadRequestErrorThreshold
            CustomTimeoutError -> CustomTimeoutErrorThreshold

        Args:
            provider: Provider name (e.g., "openai", "bedrock")
            error_type: Exception class name (e.g., "BadRequestError", "CustomError")
            dynamic_rate_limit_policy: The configured dynamic rate limit policy
            default_threshold: Default threshold if no policy configured

        Returns:
            int: The threshold to use for this provider and error type
        """
        if dynamic_rate_limit_policy is None or provider is None:
            return default_threshold

        # Dynamically construct threshold field name from exception class name
        threshold_field = f"{error_type}Threshold"

        # Use the policy's built-in method to get threshold
        return dynamic_rate_limit_policy.get_threshold_for_provider(
            provider=provider,
            error_type=threshold_field,
            default_threshold=default_threshold,
        )

    def _get_error_types_to_check(
        self,
        provider: str,
        dynamic_rate_limit_policy: Optional[DynamicRateLimitPolicy],
    ) -> list:
        """
        Get list of error types to check based on what's configured in the policy.
        
        Only checks error types that are explicitly configured for this provider.
        If no policy is configured or provider not in policy, returns empty list
        (no point checking if there are no thresholds configured).
        
        Args:
            provider: Provider name
            dynamic_rate_limit_policy: Optional policy configuration
            
        Returns:
            list: List of error type names (without "Threshold" suffix) configured for this provider
        """
        if dynamic_rate_limit_policy is None:
            return []
        
        # Get provider-specific config
        provider_config = getattr(dynamic_rate_limit_policy, provider, None)
        if provider_config is None:
            return []
        
        # Extract error types from config keys
        error_types = []
        if isinstance(provider_config, dict):
            for key in provider_config.keys():
                if key.endswith("Threshold"):
                    # Remove "Threshold" suffix to get error type name
                    error_type = key[:-9]  # len("Threshold") = 9
                    error_types.append(error_type)
            
        return error_types

    async def check_model_has_failures_exceeding_threshold(
        self,
        litellm_router_instance: Optional[LitellmRouter],
        model: str,
        dynamic_rate_limit_policy: Optional[DynamicRateLimitPolicy] = None,
    ) -> bool:
        """
        Check if any deployment for this model has failures exceeding the configured threshold.

        This checks failures by provider and error type, allowing fine-grained control
        over when rate limits should be enforced.
        
        Dynamically determines which error types to check based on the user's policy configuration.

        Args:
            litellm_router_instance: Router instance
            model: Model name to check
            dynamic_rate_limit_policy: Optional provider and error-type specific thresholds

        Returns:
            bool: True if any deployment has failures exceeding threshold
        """
        if litellm_router_instance is None:
            return False

        try:
            # Get all deployments for this model
            model_list = litellm_router_instance.get_model_list(model_name=model)
            if not model_list:
                return False

            # Check each deployment's failure counts by error type
            for deployment in model_list:
                deployment_id = deployment.get("model_info", {}).get("id")
                litellm_params = deployment.get("litellm_params", {})
                provider = litellm_params.get("custom_llm_provider")

                if not deployment_id or not provider:
                    continue

                # Get list of error types to check based on policy configuration
                error_types = self._get_error_types_to_check(
                    provider=provider,
                    dynamic_rate_limit_policy=dynamic_rate_limit_policy,
                )

                # Check each error type
                for error_type in error_types:
                    failure_count = self.get_deployment_failures_by_error_type(
                        litellm_router_instance=litellm_router_instance,
                        deployment_id=deployment_id,
                        provider=provider,
                        error_type=error_type,
                    )

                    if failure_count == 0:
                        continue

                    # Get threshold for this provider and error type
                    threshold = self.get_threshold_for_provider_and_error(
                        provider=provider,
                        error_type=error_type,
                        dynamic_rate_limit_policy=dynamic_rate_limit_policy,
                    )

                    if failure_count > threshold:
                        verbose_proxy_logger.debug(
                            f"[Dynamic Rate Limit] Deployment {deployment_id} (provider: {provider}) "
                            f"has {failure_count} {error_type} failures (threshold: {threshold}) "
                            f"in current minute - enforcing rate limits for model {model}"
                        )
                        return True

            verbose_proxy_logger.debug(
                f"[Dynamic Rate Limit] No failures exceeding threshold for model {model} - allowing dynamic exceeding"
            )
            return False

        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error checking model failure status: {str(e)}, defaulting to enforce limits"
            )
            # Fail safe: enforce limits if we can't check
            return True
