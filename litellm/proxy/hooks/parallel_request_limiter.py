from typing import Optional
import litellm
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException

class MaxParallelRequestsHandler(CustomLogger): 
    # Class variables or attributes
    def __init__(self):
        pass

    def print_verbose(self, print_statement):
        if litellm.set_verbose is True: 
            print(print_statement) # noqa

    
    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: str): 
        self.print_verbose(f"Inside Max Parallel Request Pre-Call Hook")
        api_key = user_api_key_dict.api_key
        max_parallel_requests = user_api_key_dict.max_parallel_requests

        if api_key is None:
            return

        if max_parallel_requests is None:
            return
        
        self.user_api_key_cache = cache # save the api key cache for updating the value

        # CHECK IF REQUEST ALLOWED
        request_count_api_key = f"{api_key}_request_count"
        current = cache.get_cache(key=request_count_api_key)
        self.print_verbose(f"current: {current}")
        if current is None:
            cache.set_cache(request_count_api_key, 1)
        elif int(current) <  max_parallel_requests:
            # Increase count for this token
            cache.set_cache(request_count_api_key, int(current) + 1)
        else: 
            raise HTTPException(status_code=429, detail="Max parallel request limit reached.")


    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try: 
            self.print_verbose(f"INSIDE ASYNC SUCCESS LOGGING")
            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
            if user_api_key is None:
                return
            
            if self.user_api_key_cache is None: 
                return
            
            request_count_api_key = f"{user_api_key}_request_count"
            # check if it has collected an entire stream response
            self.print_verbose(f"'complete_streaming_response' is in kwargs: {'complete_streaming_response' in kwargs}")
            if "complete_streaming_response" in kwargs or kwargs["stream"] != True:
                # Decrease count for this token
                current = self.user_api_key_cache.get_cache(key=request_count_api_key) or 1
                new_val = current - 1
                self.print_verbose(f"updated_value in success call: {new_val}")
                self.user_api_key_cache.set_cache(request_count_api_key, new_val)
        except Exception as e: 
            self.print_verbose(e) # noqa

    async def async_log_failure_call(self, user_api_key_dict: UserAPIKeyAuth, original_exception: Exception):
        try:
            self.print_verbose(f"Inside Max Parallel Request Failure Hook")
            api_key = user_api_key_dict.api_key
            if api_key is None:
                return
            
            if self.user_api_key_cache is None: 
                return
            
            ## decrement call count if call failed
            if (hasattr(original_exception, "status_code") 
                and original_exception.status_code == 429 
                and "Max parallel request limit reached" in str(original_exception)):
                pass # ignore failed calls due to max limit being reached
            else:  
                request_count_api_key = f"{api_key}_request_count"
                # Decrease count for this token
                current = self.user_api_key_cache.get_cache(key=request_count_api_key) or 1
                new_val = current - 1
                self.print_verbose(f"updated_value in failure call: {new_val}")
                self.user_api_key_cache.set_cache(request_count_api_key, new_val)
        except Exception as e:
            self.print_verbose(f"An exception occurred - {str(e)}") # noqa