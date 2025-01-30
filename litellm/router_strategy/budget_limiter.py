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
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs
from litellm.router_utils.cooldown_callbacks import (
    _get_prometheus_logger_from_callbacks,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import DeploymentTypedDict, LiteLLM_Params, RouterErrors
from litellm.types.utils import BudgetConfig
from litellm.types.utils import BudgetConfig as GenericBudgetInfo
from litellm.types.utils import GenericBudgetConfigType, StandardLoggingPayload

DEFAULT_REDIS_SYNC_INTERVAL = 1


class RouterBudgetLimiting(CustomLogger):
    def __init__(
        self,
        dual_cache: DualCache,
        provider_budget_config: Optional[dict],
        model_list: Optional[
            Union[List[DeploymentTypedDict], List[Dict[str, Any]]]
        ] = None,
    ):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue: List[RedisPipelineIncrementOperation] = []
        asyncio.create_task(self.periodic_sync_in_memory_spend_with_redis())
        self.provider_budget_config: Optional[GenericBudgetConfigType] = (
            provider_budget_config
        )
        self.deployment_budget_config: Optional[GenericBudgetConfigType] = None
        self.tag_budget_config: Optional[GenericBudgetConfigType] = None
        self._init_provider_budgets()
        self._init_deployment_budgets(model_list=model_list)
        self._init_tag_budgets()

        # Add self to litellm callbacks if it's a list
        if isinstance(litellm.callbacks, list):
            litellm.callbacks.append(self)  # type: ignore

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,  # type: ignore
    ) -> List[dict]:
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

        cache_keys, provider_configs, deployment_configs = (
            await self._async_get_cache_keys_for_router_budget_limiting(
                healthy_deployments=healthy_deployments,
                request_kwargs=request_kwargs,
            )
        )

        # Single cache read for all spend values
        if len(cache_keys) > 0:
            _current_spends = await self.dual_cache.async_batch_get_cache(
                keys=cache_keys,
                parent_otel_span=parent_otel_span,
            )
            current_spends: List = _current_spends or [0.0] * len(cache_keys)

            # Map spends to their respective keys
            spend_map: Dict[str, float] = {}
            for idx, key in enumerate(cache_keys):
                spend_map[key] = float(current_spends[idx] or 0.0)

            potential_deployments, deployment_above_budget_info = (
                self._filter_out_deployments_above_budget(
                    healthy_deployments=healthy_deployments,
                    provider_configs=provider_configs,
                    deployment_configs=deployment_configs,
                    spend_map=spend_map,
                    potential_deployments=potential_deployments,
                    request_tags=_get_tags_from_request_kwargs(
                        request_kwargs=request_kwargs
                    ),
                )
            )

            if len(potential_deployments) == 0:
                raise ValueError(
                    f"{RouterErrors.no_deployments_with_provider_budget_routing.value}: {deployment_above_budget_info}"
                )

            return potential_deployments
        else:
            return healthy_deployments

    def _filter_out_deployments_above_budget(
        self,
        potential_deployments: List[Dict[str, Any]],
        healthy_deployments: List[Dict[str, Any]],
        provider_configs: Dict[str, GenericBudgetInfo],
        deployment_configs: Dict[str, GenericBudgetInfo],
        spend_map: Dict[str, float],
        request_tags: List[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Filter out deployments that have exceeded their budget limit.
        Follow budget checks are run here:
            - Provider budget
            - Deployment budget
            - Request tags budget
        Returns:
            Tuple[List[Dict[str, Any]], str]:
                - A tuple containing the filtered deployments
                - A string containing debug information about deployments that exceeded their budget limit.
        """
        # Filter deployments based on both provider and deployment budgets
        deployment_above_budget_info: str = ""
        for deployment in healthy_deployments:
            is_within_budget = True

            # Check provider budget
            if self.provider_budget_config:
                provider = self._get_llm_provider_for_deployment(deployment)
                if provider in provider_configs:
                    config = provider_configs[provider]
                    if config.max_budget is None:
                        continue
                    current_spend = spend_map.get(
                        f"provider_spend:{provider}:{config.budget_duration}", 0.0
                    )
                    self._track_provider_remaining_budget_prometheus(
                        provider=provider,
                        spend=current_spend,
                        budget_limit=config.max_budget,
                    )

                    if config.max_budget and current_spend >= config.max_budget:
                        debug_msg = f"Exceeded budget for provider {provider}: {current_spend} >= {config.max_budget}"
                        deployment_above_budget_info += f"{debug_msg}\n"
                        is_within_budget = False
                        continue

            # Check deployment budget
            if self.deployment_budget_config and is_within_budget:
                _model_name = deployment.get("model_name")
                _litellm_params = deployment.get("litellm_params") or {}
                _litellm_model_name = _litellm_params.get("model")
                model_id = deployment.get("model_info", {}).get("id")
                if model_id in deployment_configs:
                    config = deployment_configs[model_id]
                    current_spend = spend_map.get(
                        f"deployment_spend:{model_id}:{config.budget_duration}", 0.0
                    )
                    if config.max_budget and current_spend >= config.max_budget:
                        debug_msg = f"Exceeded budget for deployment model_name: {_model_name}, litellm_params.model: {_litellm_model_name}, model_id: {model_id}: {current_spend} >= {config.budget_duration}"
                        verbose_router_logger.debug(debug_msg)
                        deployment_above_budget_info += f"{debug_msg}\n"
                        is_within_budget = False
                        continue
            # Check tag budget
            if self.tag_budget_config and is_within_budget:
                for _tag in request_tags:
                    _tag_budget_config = self._get_budget_config_for_tag(_tag)
                    if _tag_budget_config:
                        _tag_spend = spend_map.get(
                            f"tag_spend:{_tag}:{_tag_budget_config.budget_duration}",
                            0.0,
                        )
                        if (
                            _tag_budget_config.max_budget
                            and _tag_spend >= _tag_budget_config.max_budget
                        ):
                            debug_msg = f"Exceeded budget for tag='{_tag}', tag_spend={_tag_spend}, tag_budget_limit={_tag_budget_config.max_budget}"
                            verbose_router_logger.debug(debug_msg)
                            deployment_above_budget_info += f"{debug_msg}\n"
                            is_within_budget = False
                            continue
            if is_within_budget:
                potential_deployments.append(deployment)

        return potential_deployments, deployment_above_budget_info

    async def _async_get_cache_keys_for_router_budget_limiting(
        self,
        healthy_deployments: List[Dict[str, Any]],
        request_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[str], Dict[str, GenericBudgetInfo], Dict[str, GenericBudgetInfo]]:
        """
        Returns list of cache keys to fetch from router cache for budget limiting and provider and deployment configs

        Returns:
            Tuple[List[str], Dict[str, GenericBudgetInfo], Dict[str, GenericBudgetInfo]]:
                - List of cache keys to fetch from router cache for budget limiting
                - Dict of provider budget configs `provider_configs`
                - Dict of deployment budget configs `deployment_configs`
        """
        cache_keys: List[str] = []
        provider_configs: Dict[str, GenericBudgetInfo] = {}
        deployment_configs: Dict[str, GenericBudgetInfo] = {}

        for deployment in healthy_deployments:
            # Check provider budgets
            if self.provider_budget_config:
                provider = self._get_llm_provider_for_deployment(deployment)
                if provider is not None:
                    budget_config = self._get_budget_config_for_provider(provider)
                    if (
                        budget_config is not None
                        and budget_config.budget_duration is not None
                    ):
                        provider_configs[provider] = budget_config
                        cache_keys.append(
                            f"provider_spend:{provider}:{budget_config.budget_duration}"
                        )

            # Check deployment budgets
            if self.deployment_budget_config:
                model_id = deployment.get("model_info", {}).get("id")
                if model_id is not None:
                    budget_config = self._get_budget_config_for_deployment(model_id)
                    if budget_config is not None:
                        deployment_configs[model_id] = budget_config
                        cache_keys.append(
                            f"deployment_spend:{model_id}:{budget_config.budget_duration}"
                        )
            # Check tag budgets
            if self.tag_budget_config:
                request_tags = _get_tags_from_request_kwargs(
                    request_kwargs=request_kwargs
                )
                for _tag in request_tags:
                    _tag_budget_config = self._get_budget_config_for_tag(_tag)
                    if _tag_budget_config:
                        cache_keys.append(
                            f"tag_spend:{_tag}:{_tag_budget_config.budget_duration}"
                        )
        return cache_keys, provider_configs, deployment_configs

    async def _get_or_set_budget_start_time(
        self, start_time_key: str, current_time: float, ttl_seconds: int
    ) -> float:
        """
        Checks if the key = `provider_budget_start_time:{provider}` exists in cache.

        If it does, return the value.
        If it does not, set the key to `current_time` and return the value.
        """
        budget_start = await self.dual_cache.async_get_cache(start_time_key)
        if budget_start is None:
            await self.dual_cache.async_set_cache(
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
        await self.dual_cache.async_set_cache(
            key=spend_key, value=response_cost, ttl=ttl_seconds
        )
        await self.dual_cache.async_set_cache(
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
        await self.dual_cache.in_memory_cache.async_increment(
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
        verbose_router_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_payload is required")

        response_cost: float = standard_logging_payload.get("response_cost", 0)
        model_id: str = str(standard_logging_payload.get("model_id", ""))
        custom_llm_provider: str = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", None
        )
        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required")

        budget_config = self._get_budget_config_for_provider(custom_llm_provider)
        if budget_config:
            # increment spend for provider
            spend_key = (
                f"provider_spend:{custom_llm_provider}:{budget_config.budget_duration}"
            )
            start_time_key = f"provider_budget_start_time:{custom_llm_provider}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )

        deployment_budget_config = self._get_budget_config_for_deployment(model_id)
        if deployment_budget_config:
            # increment spend for specific deployment id
            deployment_spend_key = f"deployment_spend:{model_id}:{deployment_budget_config.budget_duration}"
            deployment_start_time_key = f"deployment_budget_start_time:{model_id}"
            await self._increment_spend_for_key(
                budget_config=deployment_budget_config,
                spend_key=deployment_spend_key,
                start_time_key=deployment_start_time_key,
                response_cost=response_cost,
            )

        request_tags = _get_tags_from_request_kwargs(kwargs)
        if len(request_tags) > 0:
            for _tag in request_tags:
                _tag_budget_config = self._get_budget_config_for_tag(_tag)
                if _tag_budget_config:
                    _tag_spend_key = (
                        f"tag_spend:{_tag}:{_tag_budget_config.budget_duration}"
                    )
                    _tag_start_time_key = f"tag_budget_start_time:{_tag}"
                    await self._increment_spend_for_key(
                        budget_config=_tag_budget_config,
                        spend_key=_tag_spend_key,
                        start_time_key=_tag_start_time_key,
                        response_cost=response_cost,
                    )

    async def _increment_spend_for_key(
        self,
        budget_config: GenericBudgetInfo,
        spend_key: str,
        start_time_key: str,
        response_cost: float,
    ):
        if budget_config.budget_duration is None:
            return

        current_time = datetime.now(timezone.utc).timestamp()
        ttl_seconds = duration_in_seconds(budget_config.budget_duration)

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
            if not self.dual_cache.redis_cache:
                return  # Redis is not initialized

            verbose_router_logger.debug(
                "Pushing Redis Increment Pipeline for queue: %s",
                self.redis_increment_operation_queue,
            )
            if len(self.redis_increment_operation_queue) > 0:
                asyncio.create_task(
                    self.dual_cache.redis_cache.async_increment_pipeline(
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
            if self.dual_cache.redis_cache is None:
                return

            # 1. Push all provider spend increments to Redis
            await self._push_in_memory_increments_to_redis()

            # 2. Fetch all current provider spend from Redis to update in-memory cache
            cache_keys = []

            if self.provider_budget_config is not None:
                for provider, config in self.provider_budget_config.items():
                    if config is None:
                        continue
                    cache_keys.append(
                        f"provider_spend:{provider}:{config.budget_duration}"
                    )

            if self.deployment_budget_config is not None:
                for model_id, config in self.deployment_budget_config.items():
                    if config is None:
                        continue
                    cache_keys.append(
                        f"deployment_spend:{model_id}:{config.budget_duration}"
                    )

            if self.tag_budget_config is not None:
                for tag, config in self.tag_budget_config.items():
                    if config is None:
                        continue
                    cache_keys.append(f"tag_spend:{tag}:{config.budget_duration}")

            # Batch fetch current spend values from Redis
            redis_values = await self.dual_cache.redis_cache.async_batch_get_cache(
                key_list=cache_keys
            )

            # Update in-memory cache with Redis values
            if isinstance(redis_values, dict):  # Check if redis_values is a dictionary
                for key, value in redis_values.items():
                    if value is not None:
                        await self.dual_cache.in_memory_cache.async_set_cache(
                            key=key, value=float(value)
                        )
                        verbose_router_logger.debug(
                            f"Updated in-memory cache for {key}: {value}"
                        )

        except Exception as e:
            verbose_router_logger.error(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )

    def _get_budget_config_for_deployment(
        self,
        model_id: str,
    ) -> Optional[GenericBudgetInfo]:
        if self.deployment_budget_config is None:
            return None
        return self.deployment_budget_config.get(model_id, None)

    def _get_budget_config_for_provider(
        self, provider: str
    ) -> Optional[GenericBudgetInfo]:
        if self.provider_budget_config is None:
            return None
        return self.provider_budget_config.get(provider, None)

    def _get_budget_config_for_tag(self, tag: str) -> Optional[GenericBudgetInfo]:
        if self.tag_budget_config is None:
            return None
        return self.tag_budget_config.get(tag, None)

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

        spend_key = f"provider_spend:{provider}:{budget_config.budget_duration}"

        if self.dual_cache.redis_cache:
            # use Redis as source of truth since that has spend across all instances
            current_spend = await self.dual_cache.redis_cache.async_get_cache(spend_key)
        else:
            # use in-memory cache if Redis is not initialized
            current_spend = await self.dual_cache.async_get_cache(spend_key)
        return float(current_spend) if current_spend is not None else 0.0

    async def _get_current_provider_budget_reset_at(
        self, provider: str
    ) -> Optional[str]:
        budget_config = self._get_budget_config_for_provider(provider)
        if budget_config is None:
            return None

        spend_key = f"provider_spend:{provider}:{budget_config.budget_duration}"
        if self.dual_cache.redis_cache:
            ttl_seconds = await self.dual_cache.redis_cache.async_get_ttl(spend_key)
        else:
            ttl_seconds = await self.dual_cache.async_get_ttl(spend_key)

        if ttl_seconds is None:
            return None

        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

    async def _init_provider_budget_in_cache(
        self, provider: str, budget_config: GenericBudgetInfo
    ):
        """
        Initialize provider budget in cache by storing the following keys if they don't exist:
        - provider_spend:{provider}:{budget_config.time_period} - stores the current spend
        - provider_budget_start_time:{provider} - stores the start time of the budget window

        """

        spend_key = f"provider_spend:{provider}:{budget_config.budget_duration}"
        start_time_key = f"provider_budget_start_time:{provider}"
        ttl_seconds: Optional[int] = None
        if budget_config.budget_duration is not None:
            ttl_seconds = duration_in_seconds(budget_config.budget_duration)

        budget_start = await self.dual_cache.async_get_cache(start_time_key)
        if budget_start is None:
            budget_start = datetime.now(timezone.utc).timestamp()
            await self.dual_cache.async_set_cache(
                key=start_time_key, value=budget_start, ttl=ttl_seconds
            )

        _spend_key = await self.dual_cache.async_get_cache(spend_key)
        if _spend_key is None:
            await self.dual_cache.async_set_cache(
                key=spend_key, value=0.0, ttl=ttl_seconds
            )

    @staticmethod
    def should_init_router_budget_limiter(
        provider_budget_config: Optional[dict],
        model_list: Optional[
            Union[List[DeploymentTypedDict], List[Dict[str, Any]]]
        ] = None,
    ):
        """
        Returns `True` if the router budget routing settings are set and RouterBudgetLimiting should be initialized

        Either:
         - provider_budget_config is set
         - budgets are set for deployments in the model_list
         - tag_budget_config is set
        """
        if provider_budget_config is not None:
            return True

        if litellm.tag_budget_config is not None:
            return True

        if model_list is None:
            return False

        for _model in model_list:
            _litellm_params = _model.get("litellm_params", {})
            if (
                _litellm_params.get("max_budget")
                or _litellm_params.get("budget_duration") is not None
            ):
                return True
        return False

    def _init_provider_budgets(self):
        if self.provider_budget_config is not None:
            # cast elements of provider_budget_config to GenericBudgetInfo
            for provider, config in self.provider_budget_config.items():
                if config is None:
                    raise ValueError(
                        f"No budget config found for provider {provider}, provider_budget_config: {self.provider_budget_config}"
                    )

                if not isinstance(config, GenericBudgetInfo):
                    self.provider_budget_config[provider] = GenericBudgetInfo(
                        budget_limit=config.get("budget_limit"),
                        time_period=config.get("time_period"),
                    )
                asyncio.create_task(
                    self._init_provider_budget_in_cache(
                        provider=provider,
                        budget_config=self.provider_budget_config[provider],
                    )
                )

            verbose_router_logger.debug(
                f"Initalized Provider budget config: {self.provider_budget_config}"
            )

    def _init_deployment_budgets(
        self,
        model_list: Optional[
            Union[List[DeploymentTypedDict], List[Dict[str, Any]]]
        ] = None,
    ):
        if model_list is None:
            return
        for _model in model_list:
            _litellm_params = _model.get("litellm_params", {})
            _model_info = _model.get("model_info", {})
            _model_id = _model_info.get("id")
            _max_budget = _litellm_params.get("max_budget")
            _budget_duration = _litellm_params.get("budget_duration")

            verbose_router_logger.debug(
                f"Init Deployment Budget: max_budget: {_max_budget}, budget_duration: {_budget_duration}, model_id: {_model_id}"
            )
            if (
                _max_budget is not None
                and _budget_duration is not None
                and _model_id is not None
            ):
                _budget_config = GenericBudgetInfo(
                    time_period=_budget_duration,
                    budget_limit=_max_budget,
                )
                if self.deployment_budget_config is None:
                    self.deployment_budget_config = {}
                self.deployment_budget_config[_model_id] = _budget_config

        verbose_router_logger.debug(
            f"Initialized Deployment Budget Config: {self.deployment_budget_config}"
        )

    def _init_tag_budgets(self):
        if litellm.tag_budget_config is None:
            return
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            raise ValueError(
                f"Tag budgets are an Enterprise only feature, {CommonProxyErrors.not_premium_user}"
            )

        if self.tag_budget_config is None:
            self.tag_budget_config = {}

        for _tag, _tag_budget_config in litellm.tag_budget_config.items():
            if isinstance(_tag_budget_config, dict):
                _tag_budget_config = BudgetConfig(**_tag_budget_config)
            _generic_budget_config = GenericBudgetInfo(
                time_period=_tag_budget_config.budget_duration,
                budget_limit=_tag_budget_config.max_budget,
            )
            self.tag_budget_config[_tag] = _generic_budget_config

        verbose_router_logger.debug(
            f"Initialized Tag Budget Config: {self.tag_budget_config}"
        )
