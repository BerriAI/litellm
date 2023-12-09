from typing import Optional
from litellm.caching import DualCache
from fastapi import HTTPException

async def max_parallel_request_allow_request(max_parallel_requests: Optional[int], api_key: Optional[str], user_api_key_cache: DualCache): 
    if api_key is None:
        return

    if max_parallel_requests is None:
        return
    
    # CHECK IF REQUEST ALLOWED
    request_count_api_key = f"{api_key}_request_count"
    current = user_api_key_cache.get_cache(key=request_count_api_key)
    if current is None:
        user_api_key_cache.set_cache(request_count_api_key, 1)
    elif int(current) <  max_parallel_requests:
        # Increase count for this token
        user_api_key_cache.set_cache(request_count_api_key, int(current) + 1)
    else: 
        raise HTTPException(status_code=429, detail="Max parallel request limit reached.")


async def max_parallel_request_update_count(api_key: Optional[str], user_api_key_cache: DualCache): 
    if api_key is None:
        return
    
    request_count_api_key = f"{api_key}_request_count"
    # Decrease count for this token
    current = user_api_key_cache.get_cache(key=request_count_api_key) or 1
    user_api_key_cache.set_cache(request_count_api_key, int(current) - 1)

    return 