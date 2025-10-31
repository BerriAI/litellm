"""
Decorator for profiling endpoint memory usage.

This module provides the @profile_endpoint decorator that automatically:
- Tracks request timing
- Captures memory usage
- Records profile data
- Handles both async and sync endpoints
"""

import asyncio
import functools
import time
from typing import Callable, Optional

from litellm._logging import verbose_proxy_logger

from .profiler_core import EndpointProfiler


def _extract_request_from_args(args, kwargs):
    """Extract request object from function arguments."""
    request = None
    for arg in args:
        if hasattr(arg, 'url') and hasattr(arg, 'method'):
            request = arg
            break
    
    method = kwargs.get('method', 'UNKNOWN')
    if request and hasattr(request, 'method'):
        method = request.method
    
    return request, method


def _record_profile_data(profiler, endpoint, method, request_counter, start_time, end_time, 
                         status_code, error_message, request, response):
    """Record profile data with error handling."""
    try:
        profiler.add_profile(
            endpoint=endpoint,
            method=method,
            request_counter=request_counter,
            start_time=start_time,
            end_time=end_time,
            status_code=status_code,
            error_message=error_message,
            request=request,
            response=response,
        )
    except Exception as profile_error:
        verbose_proxy_logger.error(f"Error recording profile: {profile_error}")


def profile_endpoint(
    profiler: Optional[EndpointProfiler] = None,
    endpoint: Optional[str] = None,
) -> Callable:
    """
    Decorator to profile endpoint requests.
    
    This decorator:
    1. Checks if profiling is enabled and request should be sampled
    2. Captures start time
    3. Executes the endpoint function
    4. Captures end time and response status
    5. Records profile data to buffer
    
    Args:
        profiler: EndpointProfiler instance (uses singleton if None)
        endpoint: Endpoint path override (auto-detected if None)
        
    Returns:
        Decorator function
        
    Example:
        >>> @profile_endpoint()
        ... async def chat_completions(request: Request):
        ...     return {"response": "data"}
    """
    def decorator(func: Callable) -> Callable:
        # Get profiler instance
        _profiler = profiler or EndpointProfiler.get_instance()
        
        # Determine endpoint name
        _endpoint = endpoint or getattr(func, '__name__', 'unknown')
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                request_counter = _profiler.increment_counter()
                
                if not _profiler.should_profile_request(request_counter):
                    return await func(*args, **kwargs)
                
                request, method = _extract_request_from_args(args, kwargs)
                start_time = time.time()
                error_message = None
                status_code = 200
                response = None
                
                try:
                    response = await func(*args, **kwargs)
                    if hasattr(response, 'status_code'):
                        status_code = response.status_code
                    return response
                except Exception as e:
                    error_message = str(e)
                    status_code = 500
                    raise
                finally:
                    end_time = time.time()
                    _record_profile_data(
                        _profiler, _endpoint, method, request_counter, 
                        start_time, end_time, status_code, error_message, request, response
                    )
                    
            return async_wrapper
            
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                request_counter = _profiler.increment_counter()
                
                if not _profiler.should_profile_request(request_counter):
                    return func(*args, **kwargs)
                
                request, method = _extract_request_from_args(args, kwargs)
                start_time = time.time()
                error_message = None
                status_code = 200
                response = None
                
                try:
                    response = func(*args, **kwargs)
                    if hasattr(response, 'status_code'):
                        status_code = response.status_code
                    return response
                except Exception as e:
                    error_message = str(e)
                    status_code = 500
                    raise
                finally:
                    end_time = time.time()
                    _record_profile_data(
                        _profiler, _endpoint, method, request_counter,
                        start_time, end_time, status_code, error_message, request, response
                    )
            
            return sync_wrapper
    
    return decorator

