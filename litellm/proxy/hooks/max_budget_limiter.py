from fastapi import HTTPException

from litellm import verbose_logger
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.cost_calculator import convert_budget_to_askii_coins
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class _PROXY_MaxBudgetLimiter(CustomLogger):
    # Class variables or attributes
    def __init__(self):
        pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        try:
            verbose_proxy_logger.debug("Inside Max Budget Limiter Pre-Call Hook")
            cache_key = f"{user_api_key_dict.user_id}_user_api_key_user_id"
            user_row = await cache.async_get_cache(
                cache_key, parent_otel_span=user_api_key_dict.parent_otel_span
            )
            if user_row is None:  # value not yet cached
                return
            max_budget = user_row["max_budget"]
            curr_spend = user_row["spend"]

            if max_budget is None:
                return

            if curr_spend is None:
                return

            # CHECK IF REQUEST ALLOWED
            # Convert USD budget to Askii Coins for comparison with spend (which is now in Askii Coins)
            max_budget_askii_coins = convert_budget_to_askii_coins(max_budget)
            if max_budget_askii_coins is not None and curr_spend >= max_budget_askii_coins:
                raise HTTPException(
                    status_code=429,
                    detail=f"Max budget limit reached. Current spend: {curr_spend} Askii Coins, Budget: {max_budget_askii_coins} Askii Coins (${max_budget} USD)"
                )
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.max_budget_limiter.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
