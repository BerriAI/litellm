from typing import Optional
import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException

class MaxParallelRequestsHandler(CustomLogger): 
    # Class variables or attributes
    def __init__(self):
        pass

    def print_verbose(self, print_statement):
        if litellm.set_verbose is True: 
            print(print_statement) # noqa

    
    async def max_parallel_request_allow_request(self, max_parallel_requests: Optional[int], api_key: Optional[str], user_api_key_cache: DualCache): 
        if api_key is None:
            return

        if max_parallel_requests is None:
            return
        
        self.user_api_key_cache = user_api_key_cache # save the api key cache for updating the value

        # CHECK IF REQUEST ALLOWED
        request_count_api_key = f"{api_key}_request_count"
        current = user_api_key_cache.get_cache(key=request_count_api_key)
        self.print_verbose(f"current: {current}")
        if current is None:
            user_api_key_cache.set_cache(request_count_api_key, 1)
        elif int(current) <  max_parallel_requests:
            # Increase count for this token
            user_api_key_cache.set_cache(request_count_api_key, int(current) + 1)
        else: 
            raise HTTPException(status_code=429, detail="Max parallel request limit reached.")


    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try: 
            self.print_verbose(f"INSIDE ASYNC SUCCESS LOGGING")
            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
            if user_api_key is None:
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

    async def async_log_failure_call(self, api_key, user_api_key_cache):
        try:
            if api_key is None:
                return
            
            request_count_api_key = f"{api_key}_request_count"
            # Decrease count for this token
            current = self.user_api_key_cache.get_cache(key=request_count_api_key) or 1
            new_val = current - 1
            self.print_verbose(f"updated_value in failure call: {new_val}")
            self.user_api_key_cache.set_cache(request_count_api_key, new_val)
        except Exception as e:
            self.print_verbose(f"An exception occurred - {str(e)}") # noqa