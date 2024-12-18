import traceback
from typing import Optional

from fastapi import HTTPException

import litellm
from litellm import verbose_logger
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import GenericBudgetInfo, StandardLoggingPayload


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(self):
        # Override parent's __init__ to do nothing
        pass

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

        response_cost: float = standard_logging_payload.get("response_cost", 0)
        model = standard_logging_payload.get("model")

        virtual_key = standard_logging_payload.get("metadata").get("user_api_key_hash")
        model = standard_logging_payload.get("model")
        if virtual_key is not None:
            budget_config = GenericBudgetInfo(time_period="1d", budget_limit=0.1)
            virtual_spend_key = (
                f"virtual_key_spend:{virtual_key}:{model}:{budget_config.time_period}"
            )
            virtual_start_time_key = f"virtual_key_budget_start_time:{virtual_key}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=virtual_spend_key,
                start_time_key=virtual_start_time_key,
                response_cost=response_cost,
            )
        # import json

        # print(
        #     "current state of in memory cache",
        #     json.dumps(
        #         self.router_cache.in_memory_cache.cache_dict, indent=4, default=str
        #     ),
        # )
