from typing import Optional
import litellm
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
import json, traceback

class MaxBudgetLimiter(CustomLogger): 
    # Class variables or attributes
    def __init__(self):
        pass

    def print_verbose(self, print_statement):
        if litellm.set_verbose is True: 
            print(print_statement) # noqa
    
    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: str): 
        try: 
            self.print_verbose(f"Inside Max Budget Limiter Pre-Call Hook")
            cache_key = f"{user_api_key_dict.user_id}_user_api_key_user_id"
            user_row = cache.get_cache(cache_key)
            if user_row is None: # value not yet cached
                return 
            max_budget = user_row["max_budget"]
            curr_spend = user_row["spend"]

            if max_budget is None:
                return
            
            if curr_spend is None: 
                return 
            
            # CHECK IF REQUEST ALLOWED
            if curr_spend >= max_budget:
                raise HTTPException(status_code=429, detail="Max budget limit reached.")
        except HTTPException as e: 
            raise e
        except Exception as e:
            traceback.print_exc()