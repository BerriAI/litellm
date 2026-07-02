"""
Task-Aware Routing Strategy for LiteLLM

Routes requests based on task type (coding, chinese, general, english, fast).

Usage:
    router_settings:
      routing_strategy: task-aware-routing
      routing_strategy_args:
        task_mapping:
          coding: [devstral, qwen3-coder, claude-sonnet]
          chinese: [qwen3, sf-qwen2.5-72b, claude-sonnet]
          general: [gemma4, nemotron-ultra-free, claude-sonnet]
          english: [ornith-35b, nemotron-ultra-free, claude-opus]
          fast: [deepseek-r1-14b, sf-deepseek-r1, glm-4-flash]
"""

from typing import Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy


class TaskAwareRoutingStrategy(BaseRoutingStrategy):
    """
    Routes requests based on task type.

    This strategy allows users to define task-specific model preferences.
    When a request comes in with a task type, it selects the best model
    for that task based on the configured task_mapping.

    Example:
        task_mapping = {
            "coding": ["devstral", "qwen3-coder", "claude-sonnet"],
            "chinese": ["qwen3", "sf-qwen2.5-72b", "claude-sonnet"],
            "general": ["gemma4", "nemotron-ultra-free", "claude-sonnet"],
        }

        # Request with task="coding" will prefer "devstral"
        # If "devstral" is unavailable, falls back to "qwen3-coder", then "claude-sonnet"
    """

    def __init__(
        self,
        dual_cache: DualCache,
        task_mapping: Optional[Dict[str, List[str]]] = None,
        default_task: str = "general",
        should_batch_redis_writes: bool = False,
        default_sync_interval: Optional[Union[int, float]] = None,
    ):
        super().__init__(
            dual_cache=dual_cache,
            should_batch_redis_writes=should_batch_redis_writes,
            default_sync_interval=default_sync_interval,
        )
        self.task_mapping = task_mapping or {}
        self.default_task = default_task

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Select the best deployment based on task type.

        Args:
            model_group: The requested model group name
            healthy_deployments: List of healthy deployments to choose from
            messages: Optional messages for context
            input: Optional input for context
            request_kwargs: Optional request kwargs containing task info

        Returns:
            The selected deployment dict, or None if no suitable deployment found
        """
        # Extract task from request_kwargs or metadata
        task = self.default_task
        if request_kwargs:
            # Check metadata first, then direct kwargs
            metadata = request_kwargs.get("metadata", {})
            task = metadata.get("task", request_kwargs.get("task", self.default_task))

        # Get task-specific model list
        task_models = self.task_mapping.get(task, [])

        if not task_models:
            verbose_router_logger.debug(
                f"No task mapping found for task '{task}', using default model group '{model_group}'"
            )
            # Fall back to first healthy deployment
            if healthy_deployments:
                return healthy_deployments[0]
            return None

        # Find matching healthy deployment (use forward match only)
        for model_name in task_models:
            for deployment in healthy_deployments:
                dep_model = deployment.get("model_name") or deployment.get("litellm_params", {}).get("model", "")
                if model_name in dep_model:
                    verbose_router_logger.debug(
                        f"Task-aware routing: task='{task}', selected='{dep_model}'"
                    )
                    return deployment

        # No exact match, return first healthy deployment
        verbose_router_logger.warning(
            f"No exact match for task '{task}', using first healthy deployment"
        )
        if healthy_deployments:
            return healthy_deployments[0]
        return None

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

    def log_success(self, kwargs, response_obj, start_time, end_time):
        """Log successful request for analytics"""
        verbose_router_logger.debug(f"Task-aware routing success")

    def log_failure(self, kwargs, response_obj, start_time, end_time):
        """Log failed request for analytics"""
        verbose_router_logger.warning(f"Task-aware routing failure")