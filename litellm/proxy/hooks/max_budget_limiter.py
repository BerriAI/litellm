from typing import Optional
import litellm
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException

class MaxBudgetLimiter(CustomLogger): 
    # Class variables or attributes
    def __init__(self):
        pass

    def print_verbose(self, print_statement):
        if litellm.set_verbose is True: 
            print(print_statement) # noqa

    
    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: str): 
        self.print_verbose(f"Inside Max Budget Limiter Pre-Call Hook")
        api_key = user_api_key_dict.api_key
        max_budget = user_api_key_dict.max_budget
        curr_spend = user_api_key_dict.spend

        if api_key is None:
            return

        if max_budget is None:
            return
        
        if curr_spend is None: 
            return 
        
        # CHECK IF REQUEST ALLOWED
        if curr_spend >= max_budget:
            raise HTTPException(status_code=429, detail="Max budget limit reached.")