from typing import Optional
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
import httpx


def _get_async_httpx_client(params: Optional[dict] = None) -> AsyncHTTPHandler:
    """
    Retrieves the async HTTP client from the cache
    If not present, creates a new client

    Caches the new client and returns it.
    """
    _params_key_name = ""
    if params is not None:
        for key, value in params.items():
            try:
                _params_key_name += f"{key}_{value}"
            except Exception:
                pass

    _cache_key_name = "async_httpx_client" + _params_key_name
    if _cache_key_name in litellm.in_memory_llm_clients_cache:
        return litellm.in_memory_llm_clients_cache[_cache_key_name]

    if params is not None:
        _new_client = AsyncHTTPHandler(**params)
    else:
        _new_client = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
    litellm.in_memory_llm_clients_cache[_cache_key_name] = _new_client
    return _new_client


def _get_httpx_client(params: Optional[dict] = None) -> HTTPHandler:
    """
    Retrieves the HTTP client from the cache
    If not present, creates a new client

    Caches the new client and returns it.
    """
    _params_key_name = ""
    if params is not None:
        for key, value in params.items():
            try:
                _params_key_name += f"{key}_{value}"
            except Exception:
                pass

    _cache_key_name = "httpx_client" + _params_key_name
    if _cache_key_name in litellm.in_memory_llm_clients_cache:
        return litellm.in_memory_llm_clients_cache[_cache_key_name]

    if params is not None:
        _new_client = HTTPHandler(**params)
    else:
        _new_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

    litellm.in_memory_llm_clients_cache[_cache_key_name] = _new_client
    return _new_client
