"""
Local-First Routing Strategy for LiteLLM

Routes requests to local models first, then falls back to cloud models.
This strategy is ideal for users who want to:
- Minimize latency (local models are faster)
- Reduce costs (local models are free)
- Maintain privacy (data stays local)

Usage:
    router_settings:
      routing_strategy: local-first-routing
      routing_strategy_args:
        local_provider: ollama
        local_fallback_order:
          - local
          - domestic_free
          - openrouter_free
          - openrouter_paid
"""

from typing import Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy


class LocalFirstRoutingStrategy(BaseRoutingStrategy):
    """
    Routes requests to local models first, then falls back to cloud models.

    This strategy prioritizes local models (e.g., Ollama) for:
    - Lower latency
    - Zero cost
    - Better privacy

    When local models are unavailable, it falls back to cloud models
    in the specified order.
    """

    def __init__(
        self,
        dual_cache: DualCache,
        local_provider: str = "ollama",
        local_fallback_order: Optional[List[str]] = None,
        should_batch_redis_writes: bool = False,
        default_sync_interval: Optional[Union[int, float]] = None,
    ):
        super().__init__(
            dual_cache=dual_cache,
            should_batch_redis_writes=should_batch_redis_writes,
            default_sync_interval=default_sync_interval,
        )
        self.local_provider = local_provider
        self.local_fallback_order = local_fallback_order or [
            "local",
            "domestic_free",
            "openrouter_free",
            "openrouter_paid",
        ]

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Select the best deployment using local-first strategy.

        Args:
            model_group: The requested model group name
            healthy_deployments: List of healthy deployments to choose from
            messages: Optional messages for context
            input: Optional input for context
            request_kwargs: Optional request kwargs

        Returns:
            The selected deployment dict, or None if no suitable deployment found
        """
        if not healthy_deployments:
            return None

        # Try local models first
        for deployment in healthy_deployments:
            model_name = deployment.get("model_name") or deployment.get("litellm_params", {}).get("model", "")
            if self._is_local_model(model_name):
                verbose_router_logger.debug(
                    f"Local-first routing: using local model '{model_name}'"
                )
                return deployment

        # No local model available, return first healthy deployment
        verbose_router_logger.debug(
            f"Local-first routing: no local model available, using first healthy deployment"
        )
        return healthy_deployments[0]

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Async version of get_available_deployments"""
        return self.get_available_deployments(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            messages=messages,
            input=input,
            request_kwargs=request_kwargs,
        )

    def _is_local_model(self, model: str) -> bool:
        """Check if a model is a local model"""
        local_indicators = [
            "ollama/",
            "localhost:",
            "127.0.0.1:",
            "local/",
        ]
        return any(indicator in model.lower() for indicator in local_indicators)

    def log_success(self, kwargs, response_obj, start_time, end_time):
        """Log successful request for analytics"""
        verbose_router_logger.debug(f"Local-first routing success")

    def log_failure(self, kwargs, response_obj, start_time, end_time):
        """Log failed request for analytics"""
        verbose_router_logger.warning(f"Local-first routing failure")