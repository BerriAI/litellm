from fastapi import HTTPException

from litellm import verbose_logger
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
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

            # Use the budget information directly from the validated user_api_key_dict
            max_budget = user_api_key_dict.max_budget
            curr_spend = user_api_key_dict.spend

            if max_budget is None:
                # No budget limit set for this key/user/team
                return

            if curr_spend is None:
                # If spend tracking hasn't started, assume 0
                curr_spend = 0.0

            # CHECK IF REQUEST ALLOWED
            if curr_spend >= max_budget:
                verbose_proxy_logger.info(
                    f"Budget Limit Reached for {user_api_key_dict.user_id or user_api_key_dict.team_id or user_api_key_dict.api_key}. Current Spend: {curr_spend}, Max Budget: {max_budget}"
                )
                raise HTTPException(status_code=429, detail="Max budget limit reached.")
            else:
                verbose_proxy_logger.debug(
                    f"Budget Check Passed for {user_api_key_dict.user_id or user_api_key_dict.team_id or user_api_key_dict.api_key}. Current Spend: {curr_spend}, Max Budget: {max_budget}"
                )

        except HTTPException as e:
            # Re-raise HTTPException to ensure FastAPI handles it correctly
            raise e
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.max_budget_limiter.py::async_pre_call_hook(): Exception occurred - {}".format(
                    str(e)
                )
            )
            pass
