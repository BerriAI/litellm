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
            max_budget = user_api_key_dict.user_max_budget
            user_id = user_api_key_dict.user_id

            if max_budget is None or user_id is None:
                return

            # Personal budget applies only to non-team requests, matching
            # the explicit team-key exemption in common_checks section 4.1.
            if user_api_key_dict.team_id is not None:
                return

            # The reservation path admits at the strict-`<` boundary and
            # atomically pre-fills the same counter we'd read here. Re-checking
            # with `>=` would reject a request the reservation already admitted
            # when the reservation fills the counter to exactly max_budget.
            # Imported lazily to avoid a circular import via proxy.utils.
            from litellm.proxy.spend_tracking.budget_reservation import (
                get_reserved_counter_keys,
            )

            user_counter_key = f"spend:user:{user_id}"
            if user_counter_key in get_reserved_counter_keys(
                user_api_key_dict.budget_reservation
            ):
                return

            from litellm.proxy.proxy_server import get_current_spend

            curr_spend = await get_current_spend(
                counter_key=user_counter_key,
                fallback_spend=user_api_key_dict.user_spend or 0.0,
            )

            verbose_proxy_logger.debug(
                "MaxBudgetLimiter: user_id=%s, spend=%.6f, max=%.6f",
                user_id,
                curr_spend,
                max_budget,
            )

            # CHECK IF REQUEST ALLOWED
            if curr_spend >= max_budget:
                raise HTTPException(status_code=429, detail="Max budget limit reached.")
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.max_budget_limiter.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
