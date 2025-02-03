import json
from typing import List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import Span
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    BudgetConfig,
    GenericBudgetConfigType,
    StandardLoggingPayload,
)

VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX = "virtual_key_spend"


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue = []

    async def is_key_within_model_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> bool:
        """
        Check if the user_api_key_dict is within the model budget

        Raises:
            BudgetExceededError: If the user_api_key_dict has exceeded the model budget
        """
        _model_max_budget = user_api_key_dict.model_max_budget
        internal_model_max_budget: GenericBudgetConfigType = {}

        for _model, _budget_info in _model_max_budget.items():
            internal_model_max_budget[_model] = BudgetConfig(**_budget_info)

        verbose_proxy_logger.debug(
            "internal_model_max_budget %s",
            json.dumps(internal_model_max_budget, indent=4, default=str),
        )

        # check if current model is in internal_model_max_budget
        _current_model_budget_info = self._get_request_model_budget_config(
            model=model, internal_model_max_budget=internal_model_max_budget
        )
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(
                f"Model {model} not found in internal_model_max_budget"
            )
            return True

        # check if current model is within budget
        if (
            _current_model_budget_info.max_budget
            and _current_model_budget_info.max_budget > 0
        ):
            _current_spend = await self._get_virtual_key_spend_for_model(
                user_api_key_hash=user_api_key_dict.token,
                model=model,
                key_budget_config=_current_model_budget_info,
            )
            if (
                _current_spend is not None
                and _current_model_budget_info.max_budget is not None
                and _current_spend > _current_model_budget_info.max_budget
            ):
                raise litellm.BudgetExceededError(
                    message=f"LiteLLM Virtual Key: {user_api_key_dict.token}, key_alias: {user_api_key_dict.key_alias}, exceeded budget for model={model}",
                    current_cost=_current_spend,
                    max_budget=_current_model_budget_info.max_budget,
                )

        return True

    async def _get_virtual_key_spend_for_model(
        self,
        user_api_key_hash: Optional[str],
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        """
        Get the current spend for a virtual key for a model

        Lookup model in this order:
            1. model: directly look up `model`
            2. If 1, does not exist, check if passed as {custom_llm_provider}/model
        """

        # 1. model: directly look up `model`
        virtual_key_model_spend_cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{model}:{key_budget_config.budget_duration}"
        _current_spend = await self.dual_cache.async_get_cache(
            key=virtual_key_model_spend_cache_key,
        )

        if _current_spend is None:
            # 2. If 1, does not exist, check if passed as {custom_llm_provider}/model
            # if "/" in model, remove first part before "/" - eg. openai/o1-preview -> o1-preview
            virtual_key_model_spend_cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            _current_spend = await self.dual_cache.async_get_cache(
                key=virtual_key_model_spend_cache_key,
            )
        return _current_spend

    def _get_request_model_budget_config(
        self, model: str, internal_model_max_budget: GenericBudgetConfigType
    ) -> Optional[BudgetConfig]:
        """
        Get the budget config for the request model

        1. Check if `model` is in `internal_model_max_budget`
        2. If not, check if `model` without custom llm provider is in `internal_model_max_budget`
        """
        return internal_model_max_budget.get(
            model, None
        ) or internal_model_max_budget.get(
            self._get_model_without_custom_llm_provider(model), None
        )

    def _get_model_without_custom_llm_provider(self, model: str) -> str:
        if "/" in model:
            return model.split("/")[-1]
        return model

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,  # type: ignore
    ) -> List[dict]:
        return healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Track spend for virtual key + model in DualCache

        Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
        """
        verbose_proxy_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_payload is required")

        _litellm_params: dict = kwargs.get("litellm_params", {}) or {}
        _metadata: dict = _litellm_params.get("metadata", {}) or {}
        user_api_key_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_model_max_budget", None
        )
        if (
            user_api_key_model_max_budget is None
            or len(user_api_key_model_max_budget) == 0
        ):
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event because user_api_key_model_max_budget is None or empty. `user_api_key_model_max_budget`=%s",
                user_api_key_model_max_budget,
            )
            return
        response_cost: float = standard_logging_payload.get("response_cost", 0)
        model = standard_logging_payload.get("model")

        virtual_key = standard_logging_payload.get("metadata").get("user_api_key_hash")
        model = standard_logging_payload.get("model")
        if virtual_key is not None:
            budget_config = BudgetConfig(time_period="1d", budget_limit=0.1)
            virtual_spend_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_config.budget_duration}"
            virtual_start_time_key = f"virtual_key_budget_start_time:{virtual_key}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=virtual_spend_key,
                start_time_key=virtual_start_time_key,
                response_cost=response_cost,
            )
        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(
                self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str
            ),
        )
