"""
Module-level HTTP clients and caches with lazy loading.

This module provides lazy-loaded HTTP clients and caches that are initialized only when first accessed,
improving startup performance.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from litellm.caching.llm_caching_handler import LLMClientCache

    from .http_handler import AsyncHTTPHandler, HTTPHandler

# Global variables to store the clients and caches once initialized
_module_level_aclient: Optional["AsyncHTTPHandler"] = None
_module_level_client: Optional["HTTPHandler"] = None
_in_memory_llm_clients_cache: Optional["LLMClientCache"] = None


def get_module_level_aclient() -> "AsyncHTTPHandler":
    """
    Get the module-level async HTTP client, creating it if it doesn't exist.
    
    Returns:
        AsyncHTTPHandler: The lazy-loaded async HTTP client instance
    """
    global _module_level_aclient
    if _module_level_aclient is None:
        from litellm.constants import request_timeout

        from .http_handler import AsyncHTTPHandler
        
        _module_level_aclient = AsyncHTTPHandler(
            timeout=request_timeout, client_alias="module level aclient"
        )
    return _module_level_aclient


def get_module_level_client() -> "HTTPHandler":
    """
    Get the module-level sync HTTP client, creating it if it doesn't exist.
    
    Returns:
        HTTPHandler: The lazy-loaded sync HTTP client instance
    """
    global _module_level_client
    if _module_level_client is None:
        from litellm.constants import request_timeout

        from .http_handler import HTTPHandler
        
        _module_level_client = HTTPHandler(timeout=request_timeout)
    return _module_level_client


def get_in_memory_llm_clients_cache() -> "LLMClientCache":
    """
    Get the module-level LLM clients cache, creating it if it doesn't exist.
    
    Returns:
        LLMClientCache: The lazy-loaded LLM clients cache instance
    """
    global _in_memory_llm_clients_cache
    if _in_memory_llm_clients_cache is None:
        from litellm.caching.llm_caching_handler import LLMClientCache
        
        _in_memory_llm_clients_cache = LLMClientCache()
    return _in_memory_llm_clients_cache


def reset_module_level_clients() -> None:
    """
    Reset the module-level clients and caches to None, forcing re-initialization on next access.
    Useful for testing or when configuration changes.
    """
    global _module_level_aclient, _module_level_client, _in_memory_llm_clients_cache
    _module_level_aclient = None
    _module_level_client = None
    _in_memory_llm_clients_cache = None