"""
Provider budget limiting

Use this if you want to set $ budget limits for each provider.

Note: This is a filter, like tag-routing. Meaning it will accept healthy deployments and then filter out deployments that have exceeded their budget limit.

This means you can use this with weighted-pick, lowest-latency, simple-shuffle, routing etc

Example:
```
openai:
	budget_limit: 0.000000000001
	time_period: 1d
anthropic:
	budget_limit: 100
	time_period: 7d
```
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.router_utils.cooldown_callbacks import (
    _get_prometheus_logger_from_callbacks,
)
from litellm.types.router import (
    LiteLLM_Params,
    ProviderBudgetConfigType,
    ProviderBudgetInfo,
    RouterErrors,
)
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any

DEFAULT_REDIS_SYNC_INTERVAL = 1


class ProviderBudgetLimiting(CustomLogger):
    def __init__(self, router_cache: DualCache, provider_budget_config: dict):
        self.router_cache = router_cache
        self.redis_increment_operation_queue: List[RedisPipelineIncrementOperation] = []
        asyncio.create_task(self.periodic_sync_in_memory_spend_with_redis())

        # cast elements of provider_budget_config to ProviderBudgetInfo
        for provider, config in provider_budget_config.items():
            if config is None:
                raise ValueError(
                    f"No budget config found for provider {provider}, provider_budget_config: {provider_budget_config}"
                )

            if not isinstance(config, ProviderBudgetInfo):
                provider_budget_config[provider] = ProviderBudgetInfo(
                    budget_limit=config.get("budget_limit"),
                    time_period=config.get("time_period"),
                )
            asyncio.create_task(
                self._init_provider_budget_in_cache(
                    provider=provider,
                    budget_config=provider_budget_config[provider],
                )
            )

        self.provider_budget_config: ProviderBudgetConfigType = provider_budget_config
        verbose_router_logger.debug(
            f"Initalized Provider budget config: {self.provider_budget_config}"
        )

        # Add self to litellm callbacks if it's a list
        if isinstance(litellm.callbacks, list):
            litellm.callbacks.append(self)  # type: ignore

    async def async_filter_deployments(
        self,
        healthy_deployments: Union[List[Dict[str, Any]], Dict[str, Any]],
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Filter out deployments that have exceeded their provider budget limit.


        Example:
        if deployment = openai/gpt-3.5-turbo
            and openai spend > openai budget limit
                then skip this deployment
        """

        # If a single deployment is passed, convert it to a list
        if isinstance(healthy_deployments, dict):
            healthy_deployments = [healthy_deployments]

        # Don't do any filtering if there are no healthy deployments
        if len(healthy_deployments) == 0:
            return healthy_deployments

        potential_deployments: List[Dict] = []

        # Extract the parent OpenTelemetry span for tracing
        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(
            request_kwargs
        )

        # Collect all providers and their budget configs
        # {"openai": ProviderBudgetInfo, "anthropic": ProviderBudgetInfo, "azure": None}
        _provider_configs: Dict[str, Optional[ProviderBudgetInfo]] = {}
        for deployment in healthy_deployments:
            provider = self._get_llm_provider_for_deployment(deployment)
            if provider is None:
                continue
            budget_config = self._get_budget_config_for_provider(provider)
            _provider_configs[provider] = budget_config

        # Filter out providers without budget config
        provider_configs: Dict[str, ProviderBudgetInfo] = {
            provider: config
            for provider, config in _provider_configs.items()
            if config is not None
        }

        # Build cache keys for batch retrieval
        cache_keys = []
        for provider, config in provider_configs.items():
            cache_keys.append(f"provider_spend:{provider}:{config.time_period}")

        # Fetch current spend for all providers using batch cache
        _current_spends = await self.router_cache.async_batch_get_cache(
            keys=cache_keys,
            parent_otel_span=parent_otel_span,
        )
        current_spends: List = _current_spends or [0.0] * len(provider_configs)

        # Map providers to their current spend values
        provider_spend_map: Dict[str, float] = {}
        for idx, provider in enumerate(provider_configs.keys()):
            provider_spend_map[provider] = float(current_spends[idx] or 0.0)

        # Filter healthy deployments based on budget constraints
        deployment_above_budget_info: str = ""  # used to return in error message
        for deployment in healthy_deployments:
            provider = self._get_llm_provider_for_deployment(deployment)
            if provider is None:
                continue
            budget_config = provider_configs.get(provider)

            if not budget_config:
                continue

            current_spend = provider_spend_map.get(provider, 0.0)
            budget_limit = budget_config.budget_limit

            verbose_router_logger.debug(
                f"Current spend for {provider}: {current_spend}, budget limit: {budget_limit}"
            )
            self._track_provider_remaining_budget_prometheus(
                provider=provider,
                spend=current_spend,
                budget_limit=budget_limit,
            )

            if current_spend >= budget_limit:
                debug_msg = f"Exceeded budget for provider {provider}: {current_spend} >= {budget_limit}"
                verbose_router_logger.debug(debug_msg)
                deployment_above_budget_info += f"{debug_msg}\n"
                continue

            potential_deployments.append(deployment)

        if len(potential_deployments) == 0:
            raise ValueError(
                f"{RouterErrors.no_deployments_with_provider_budget_routing.value}: {deployment_above_budget_info}"
            )

        return potential_deployments

    async def _get_or_set_budget_start_time(
        self, start_time_key: str, current_time: float, ttl_seconds: int
    ) -> float:
        """
        Checks if the key = `provider_budget_start_time:{provider}` exists in cache.

        If it does, return the value.
        If it does not, set the key to `current_time` and return the value.
        """
        budget_start = await self.router_cache.async_get_cache(start_time_key)
        if budget_start is None:
            await self.router_cache.async_set_cache(
                key=start_time_key, value=current_time, ttl=ttl_seconds
            )
            return current_time
        return float(budget_start)

    async def _handle_new_budget_window(
        self,
        spend_key: str,
        start_time_key: str,
        current_time: float,
        response_cost: float,
        ttl_seconds: int,
    ) -> float:
        """
        Handle start of new budget window by resetting spend and start time

        Enters this when:
        - The budget does not exist in cache, so we need to set it
        - The budget window has expired, so we need to reset everything

        Does 2 things:
        - stores key: `provider_spend:{provider}:1d`, value: response_cost
        - stores key: `provider_budget_start_time:{provider}`, value: current_time.
            This stores the start time of the new budget window
        """
        await self.router_cache.async_set_cache(
            key=spend_key, value=response_cost, ttl=ttl_seconds
        )
        await self.router_cache.async_set_cache(
            key=start_time_key, value=current_time, ttl=ttl_seconds
        )
        return current_time

    async def _increment_spend_in_current_window(
        self, spend_key: str, response_cost: float, ttl: int
    ):
        """
        Increment spend within existing budget window

        Runs once the budget start time exists in Redis Cache (on the 2nd and subsequent requests to the same provider)

        - Increments the spend in memory cache (so spend instantly updated in memory)
        - Queues the increment operation to Redis Pipeline (using batched pipeline to optimize performance. Using Redis for multi instance environment of LiteLLM)
        """
        await self.router_cache.in_memory_cache.async_increment(
            key=spend_key,
            value=response_cost,
            ttl=ttl,
        )
        increment_op = RedisPipelineIncrementOperation(
            key=spend_key,
            increment_value=response_cost,
            ttl=ttl,
        )
        self.redis_increment_operation_queue.append(increment_op)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Original method now uses helper functions"""
        verbose_router_logger.debug("in ProviderBudgetLimiting.async_log_success_event")
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_payload is required")

        response_cost: float = standard_logging_payload.get("response_cost", 0)

        custom_llm_provider: str = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", None
        )
        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required")

        budget_config = self._get_budget_config_for_provider(custom_llm_provider)
        if budget_config is None:
            raise ValueError(
                f"No budget config found for provider {custom_llm_provider}, self.provider_budget_config: {self.provider_budget_config}"
            )

        spend_key = f"provider_spend:{custom_llm_provider}:{budget_config.time_period}"
        start_time_key = f"provider_budget_start_time:{custom_llm_provider}"

        current_time = datetime.now(timezone.utc).timestamp()
        ttl_seconds = duration_in_seconds(budget_config.time_period)

        budget_start = await self._get_or_set_budget_start_time(
            start_time_key=start_time_key,
            current_time=current_time,
            ttl_seconds=ttl_seconds,
        )

        if budget_start is None:
            # First spend for this provider
            budget_start = await self._handle_new_budget_window(
                spend_key=spend_key,
                start_time_key=start_time_key,
                current_time=current_time,
                response_cost=response_cost,
                ttl_seconds=ttl_seconds,
            )
        elif (current_time - budget_start) > ttl_seconds:
            # Budget window expired - reset everything
            verbose_router_logger.debug("Budget window expired - resetting everything")
            budget_start = await self._handle_new_budget_window(
                spend_key=spend_key,
                start_time_key=start_time_key,
                current_time=current_time,
                response_cost=response_cost,
                ttl_seconds=ttl_seconds,
            )
        else:
            # Within existing window - increment spend
            remaining_time = ttl_seconds - (current_time - budget_start)
            ttl_for_increment = int(remaining_time)

            await self._increment_spend_in_current_window(
                spend_key=spend_key, response_cost=response_cost, ttl=ttl_for_increment
            )

        verbose_router_logger.debug(
            f"Incremented spend for {spend_key} by {response_cost}"
        )

    async def periodic_sync_in_memory_spend_with_redis(self):
        """
        Handler that triggers sync_in_memory_spend_with_redis every DEFAULT_REDIS_SYNC_INTERVAL seconds

        Required for multi-instance environment usage of provider budgets
        """
        while True:
            try:
                await self._sync_in_memory_spend_with_redis()
                await asyncio.sleep(
                    DEFAULT_REDIS_SYNC_INTERVAL
                )  # Wait for DEFAULT_REDIS_SYNC_INTERVAL seconds before next sync
            except Exception as e:
                verbose_router_logger.error(f"Error in periodic sync task: {str(e)}")
                await asyncio.sleep(
                    DEFAULT_REDIS_SYNC_INTERVAL
                )  # Still wait DEFAULT_REDIS_SYNC_INTERVAL seconds on error before retrying

    async def _push_in_memory_increments_to_redis(self):
        """
        How this works:
        - async_log_success_event collects all provider spend increments in `redis_increment_operation_queue`
        - This function pushes all increments to Redis in a batched pipeline to optimize performance

        Only runs if Redis is initialized
        """
        try:
            if not self.router_cache.redis_cache:
                return  # Redis is not initialized

            verbose_router_logger.debug(
                "Pushing Redis Increment Pipeline for queue: %s",
                self.redis_increment_operation_queue,
            )
            if len(self.redis_increment_operation_queue) > 0:
                asyncio.create_task(
                    self.router_cache.redis_cache.async_increment_pipeline(
                        increment_list=self.redis_increment_operation_queue,
                    )
                )

            self.redis_increment_operation_queue = []

        except Exception as e:
            verbose_router_logger.error(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )

    async def _sync_in_memory_spend_with_redis(self):
        """
        Ensures in-memory cache is updated with latest Redis values for all provider spends.

        Why Do we need this?
        - Optimization to hit sub 100ms latency. Performance was impacted when redis was used for read/write per request
        - Use provider budgets in multi-instance environment, we use Redis to sync spend across all instances

        What this does:
        1. Push all provider spend increments to Redis
        2. Fetch all current provider spend from Redis to update in-memory cache
        """

        try:
            # No need to sync if Redis cache is not initialized
            if self.router_cache.redis_cache is None:
                return

            # 1. Push all provider spend increments to Redis
            await self._push_in_memory_increments_to_redis()

            # 2. Fetch all current provider spend from Redis to update in-memory cache
            cache_keys = []
            for provider, config in self.provider_budget_config.items():
                if config is None:
                    continue
                cache_keys.append(f"provider_spend:{provider}:{config.time_period}")

            # Batch fetch current spend values from Redis
            redis_values = await self.router_cache.redis_cache.async_batch_get_cache(
                key_list=cache_keys
            )

            # Update in-memory cache with Redis values
            if isinstance(redis_values, dict):  # Check if redis_values is a dictionary
                for key, value in redis_values.items():
                    if value is not None:
                        await self.router_cache.in_memory_cache.async_set_cache(
                            key=key, value=float(value)
                        )
                        verbose_router_logger.debug(
                            f"Updated in-memory cache for {key}: {value}"
                        )

        except Exception as e:
            verbose_router_logger.error(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )

    def _get_budget_config_for_provider(
        self, provider: str
    ) -> Optional[ProviderBudgetInfo]:
        return self.provider_budget_config.get(provider, None)

    def _get_llm_provider_for_deployment(self, deployment: Dict) -> Optional[str]:
        try:
            _litellm_params: LiteLLM_Params = LiteLLM_Params(
                **deployment.get("litellm_params", {"model": ""})
            )
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=_litellm_params.model,
                litellm_params=_litellm_params,
            )
        except Exception:
            verbose_router_logger.error(
                f"Error getting LLM provider for deployment: {deployment}"
            )
            return None
        return custom_llm_provider

    def _track_provider_remaining_budget_prometheus(
        self, provider: str, spend: float, budget_limit: float
    ):
        """
        Optional helper - emit provider remaining budget metric to Prometheus

        This is helpful for debugging and monitoring provider budget limits.
        """
        from litellm.integrations.prometheus import PrometheusLogger

        prometheus_logger = _get_prometheus_logger_from_callbacks()
        if prometheus_logger:
            prometheus_logger.track_provider_remaining_budget(
                provider=provider,
                spend=spend,
                budget_limit=budget_limit,
            )

    async def _get_current_provider_spend(self, provider: str) -> Optional[float]:
        """
        GET the current spend for a provider from cache

        used for GET /provider/budgets endpoint in spend_management_endpoints.py

        Args:
            provider (str): The provider to get spend for (e.g., "openai", "anthropic")

        Returns:
            Optional[float]: The current spend for the provider, or None if not found
        """
        budget_config = self._get_budget_config_for_provider(provider)
        if budget_config is None:
            return None

        spend_key = f"provider_spend:{provider}:{budget_config.time_period}"

        if self.router_cache.redis_cache:
            # use Redis as source of truth since that has spend across all instances
            current_spend = await self.router_cache.redis_cache.async_get_cache(
                spend_key
            )
        else:
            # use in-memory cache if Redis is not initialized
            current_spend = await self.router_cache.async_get_cache(spend_key)
        return float(current_spend) if current_spend is not None else 0.0

    async def _get_current_provider_budget_reset_at(
        self, provider: str
    ) -> Optional[str]:
        budget_config = self._get_budget_config_for_provider(provider)
        if budget_config is None:
            return None

        spend_key = f"provider_spend:{provider}:{budget_config.time_period}"
        if self.router_cache.redis_cache:
            ttl_seconds = await self.router_cache.redis_cache.async_get_ttl(spend_key)
        else:
            ttl_seconds = await self.router_cache.async_get_ttl(spend_key)

        if ttl_seconds is None:
            return None

        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

    async def _init_provider_budget_in_cache(
        self, provider: str, budget_config: ProviderBudgetInfo
    ):
        """
        Initialize provider budget in cache by storing the following keys if they don't exist:
        - provider_spend:{provider}:{budget_config.time_period} - stores the current spend
        - provider_budget_start_time:{provider} - stores the start time of the budget window

        """
        spend_key = f"provider_spend:{provider}:{budget_config.time_period}"
        start_time_key = f"provider_budget_start_time:{provider}"
        ttl_seconds = duration_in_seconds(budget_config.time_period)
        budget_start = await self.router_cache.async_get_cache(start_time_key)
        if budget_start is None:
            budget_start = datetime.now(timezone.utc).timestamp()
            await self.router_cache.async_set_cache(
                key=start_time_key, value=budget_start, ttl=ttl_seconds
            )

        _spend_key = await self.router_cache.async_get_cache(spend_key)
        if _spend_key is None:
            await self.router_cache.async_set_cache(
                key=spend_key, value=0.0, ttl=ttl_seconds
            )
