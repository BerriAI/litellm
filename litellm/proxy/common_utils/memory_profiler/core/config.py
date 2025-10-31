"""
Configuration builders for memory leak testing.

Provides functions for:
- Building completion request parameters (SDK and Router)
- Selecting appropriate completion functions (sync/async, streaming/non-streaming)
- Creating FastAPI Request objects for authentication testing
- Building test configuration dictionaries
"""

from typing import Dict, Any, Callable, Optional

from ..constants import (
    DEFAULT_SDK_MODEL,
    DEFAULT_TEST_MESSAGE_CONTENT,
    TEST_API_KEY,
    DEFAULT_TEST_USER,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_REQUEST_PATH,
    DEFAULT_REQUEST_SCHEME,
    DEFAULT_REQUEST_SERVER,
    DEFAULT_REQUEST_CLIENT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_BATCHES,
    DEFAULT_WARMUP_BATCHES,
    DEFAULT_ROLLING_AVERAGE_WINDOW,
    DEFAULT_TEST_MAX_GROWTH_PERCENT,
    DEFAULT_TEST_STABILIZATION_TOLERANCE_MB,
    MAX_NUM_BATCHES,
    DEFAULT_SKIP_ON_HIGH_NOISE,
    DEFAULT_CLEANUP_SNAPSHOTS_AFTER_TEST,
)


def get_completion_kwargs(
    model: str = DEFAULT_SDK_MODEL,
    content: str = DEFAULT_TEST_MESSAGE_CONTENT,
    api_base: str = "http://0.0.0.0:4000",
    api_key: str = TEST_API_KEY,
    user: str = DEFAULT_TEST_USER
) -> Dict[str, Any]:
    """
    Create standardized completion kwargs for memory leak tests.
    
    Args:
        model: The model to use for completion
        content: The message content
        api_base: The API base URL
        api_key: The API key
        user: The user identifier
    
    Returns:
        dict: Completion parameters for test requests
        
    Example:
        >>> kwargs = get_completion_kwargs(model="gpt-3.5-turbo")
        >>> # Use with litellm.completion(**kwargs)
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "api_base": api_base,
        "api_key": api_key,
        "user": user,
    }


def get_completion_function(litellm_module, use_async: bool, streaming: bool) -> Callable:
    """
    Get the appropriate completion function based on async and streaming parameters.
    
    Args:
        litellm_module: The litellm module instance
        use_async: Whether to use async completion (acompletion) or sync (completion)
        streaming: Whether to use streaming mode
    
    Returns:
        Callable: The appropriate completion function or wrapper
        
    Example:
        >>> import litellm
        >>> func = get_completion_function(litellm, use_async=True, streaming=False)
        >>> # func is now litellm.acompletion
    """
    # Define streaming wrapper functions
    async def async_streaming_wrapper(**kwargs):
        response = await litellm_module.acompletion(**kwargs, stream=True)
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    def sync_streaming_wrapper(**kwargs):
        response = litellm_module.completion(**kwargs, stream=True)
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    # Select appropriate completion function
    completion_funcs = {
        (True, True): async_streaming_wrapper,
        (True, False): litellm_module.acompletion,
        (False, True): sync_streaming_wrapper,
        (False, False): litellm_module.completion,
    }
    return completion_funcs[(use_async, streaming)]


def get_router_completion_kwargs(
    model: str = DEFAULT_ROUTER_MODEL,
    content: str = DEFAULT_TEST_MESSAGE_CONTENT,
    user: str = DEFAULT_TEST_USER
) -> Dict[str, Any]:
    """
    Create standardized completion kwargs for Router memory leak tests.
    
    Router tests don't need api_base or api_key since the Router instance
    handles those internally based on its model_list configuration.
    
    Args:
        model: The model name (as configured in Router's model_list)
        content: The message content
        user: The user identifier
    
    Returns:
        dict: Completion parameters for Router test requests
        
    Example:
        >>> kwargs = get_router_completion_kwargs(model="gpt-3.5-turbo")
        >>> # Use with router.completion(**kwargs)
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "user": user,
    }


def get_router_completion_function(router, use_async: bool, streaming: bool) -> Callable:
    """
    Get the appropriate router completion function based on async and streaming parameters.
    
    Args:
        router: The Router instance
        use_async: Whether to use async completion (acompletion) or sync (completion)
        streaming: Whether to use streaming mode
    
    Returns:
        Callable: The appropriate completion function or wrapper
        
    Example:
        >>> from litellm import Router
        >>> router = Router(...)
        >>> func = get_router_completion_function(router, use_async=True, streaming=False)
        >>> # func is now router.acompletion
    """
    # Define streaming wrapper functions
    async def async_streaming_wrapper(**kwargs):
        response = await router.acompletion(**kwargs, stream=True)
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    def sync_streaming_wrapper(**kwargs):
        response = router.completion(**kwargs, stream=True)
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    # Select appropriate completion function
    completion_funcs = {
        (True, True): async_streaming_wrapper,
        (True, False): router.acompletion,
        (False, True): sync_streaming_wrapper,
        (False, False): router.completion,
    }
    return completion_funcs[(use_async, streaming)]


def create_fastapi_request(url_path: str = DEFAULT_REQUEST_PATH, auth_header: str = f"Bearer {TEST_API_KEY}"):
    """
    Create a real FastAPI Request object for testing.
    
    Useful for testing authentication and authorization flows that require
    a FastAPI Request object.
    
    Args:
        url_path: The URL path for the request
        auth_header: The authorization header value
        
    Returns:
        A FastAPI Request object
        
    Example:
        >>> request = create_fastapi_request("/chat/completions")
        >>> # Use in authentication testing
    """
    from fastapi import Request
    
    scope = {
        "type": "http",
        "method": "POST",
        "path": url_path,
        "query_string": b"",
        "headers": [
            (b"authorization", auth_header.encode()),
            (b"content-type", b"application/json"),
        ],
        "scheme": DEFAULT_REQUEST_SCHEME,
        "server": DEFAULT_REQUEST_SERVER,
        "client": DEFAULT_REQUEST_CLIENT,
    }
    
    # Create request with a receive callable
    async def receive():
        return {"type": "http.request", "body": b'{}'}
    
    request = Request(scope, receive=receive)
    return request


def get_memory_test_config(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_batches: int = DEFAULT_NUM_BATCHES,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    sample_window: int = DEFAULT_ROLLING_AVERAGE_WINDOW,
    max_growth_percent: float = DEFAULT_TEST_MAX_GROWTH_PERCENT,
    stabilization_tolerance_mb: float = DEFAULT_TEST_STABILIZATION_TOLERANCE_MB,
    skip_on_high_noise: bool = DEFAULT_SKIP_ON_HIGH_NOISE,
    cleanup_snapshots_after_test: bool = DEFAULT_CLEANUP_SNAPSHOTS_AFTER_TEST
) -> Dict[str, Any]:
    """
    Get standardized memory test configuration parameters.
    
    Centralized configuration ensures consistent testing across all memory leak tests.
    All defaults are imported from constants module.
    
    Args:
        batch_size: Number of requests per batch
        num_batches: Number of batches to measure (max: MAX_NUM_BATCHES)
        warmup_batches: Number of warmup batches to run
        sample_window: Rolling average window size
        max_growth_percent: Maximum allowed growth percentage
        stabilization_tolerance_mb: Minimum growth to consider significant in MB
        skip_on_high_noise: Whether to skip test if measurements are too noisy
        cleanup_snapshots_after_test: Whether to delete snapshot JSON files after test completes
    
    Returns:
        dict: Configuration parameters for memory leak tests
        
    Example:
        >>> config = get_memory_test_config(batch_size=50, num_batches=5)
        >>> # Use in test execution
        >>> # To run test even with high noise:
        >>> config = get_memory_test_config(skip_on_high_noise=False)
        >>> # To cleanup snapshots after test:
        >>> config = get_memory_test_config(cleanup_snapshots_after_test=True)
    """
    # Enforce maximum num_batches to prevent overly long test runs
    if num_batches > MAX_NUM_BATCHES:
        num_batches = MAX_NUM_BATCHES
    
    return {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'warmup_batches': warmup_batches,
        'sample_window': sample_window,
        'max_growth_percent': max_growth_percent,
        'stabilization_tolerance_mb': stabilization_tolerance_mb,
        'skip_on_high_noise': skip_on_high_noise,
        'cleanup_snapshots_after_test': cleanup_snapshots_after_test
    }

