"""
Main Search function for LiteLLM.
"""
import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager, client, filter_out_litellm_params

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


def _build_search_optional_params(
    max_results: Optional[int] = None,
    search_domain_filter: Optional[List[str]] = None,
    max_tokens_per_page: Optional[int] = None,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper function to build optional_params dict from Perplexity Search API parameters.
    
    Args:
        max_results: Maximum number of results (1-20)
        search_domain_filter: List of domains to filter (max 20)
        max_tokens_per_page: Max tokens per page
        country: Country code filter
        
    Returns:
        Dict with non-None optional parameters
    """
    optional_params: Dict[str, Any] = {}
    
    if max_results is not None:
        optional_params["max_results"] = max_results
    if search_domain_filter is not None:
        optional_params["search_domain_filter"] = search_domain_filter
    if max_tokens_per_page is not None:
        optional_params["max_tokens_per_page"] = max_tokens_per_page
    if country is not None:
        optional_params["country"] = country
    
    return optional_params


@client
async def asearch(
    query: Union[str, List[str]],
    search_provider: str,
    max_results: Optional[int] = None,
    search_domain_filter: Optional[List[str]] = None,
    max_tokens_per_page: Optional[int] = None,
    country: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> SearchResponse:
    """
    Async Search function.
    
    Args:
        query: Search query (string or list of strings)
        search_provider: Provider name (e.g., "perplexity")
        max_results: Optional maximum number of results (1-20), default 10
        search_domain_filter: Optional list of domains to filter (max 20)
        max_tokens_per_page: Optional max tokens per page, default 1024
        country: Optional country code filter (e.g., 'US', 'GB', 'DE')
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        extra_headers: Optional extra headers
        **kwargs: Additional parameters
        
    Returns:
        SearchResponse with results list following Perplexity format
        
    Example:
        ```python
        import litellm
        
        # Basic search
        response = await litellm.asearch(
            query="latest AI developments 2024",
            search_provider="perplexity"
        )
        
        # Search with options
        response = await litellm.asearch(
            query="AI developments",
            search_provider="perplexity",
            max_results=10,
            search_domain_filter=["arxiv.org", "nature.com"],
            max_tokens_per_page=1024,
            country="US"
        )
        
        # Access results
        for result in response.results:
            print(f"{result.title}: {result.url}")
            print(f"Snippet: {result.snippet}")
        ```
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["asearch"] = True

        func = partial(
            search,
            query=query,
            search_provider=search_provider,
            max_results=max_results,
            search_domain_filter=search_domain_filter,
            max_tokens_per_page=max_tokens_per_page,
            country=country,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            extra_headers=extra_headers,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        if response is None:
            raise ValueError(
                f"Got an unexpected None response from the Search API: {response}"
            )

        return response
    except Exception as e:
        model_name = f"{search_provider}/search"
        raise litellm.exception_type(
            model=model_name,
            custom_llm_provider=search_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def search(
    query: Union[str, List[str]],
    search_provider: str,
    max_results: Optional[int] = None,
    search_domain_filter: Optional[List[str]] = None,
    max_tokens_per_page: Optional[int] = None,
    country: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[SearchResponse, Coroutine[Any, Any, SearchResponse]]:
    """
    Synchronous Search function.
    
    Args:
        query: Search query (string or list of strings)
        search_provider: Provider name (e.g., "perplexity")
        max_results: Optional maximum number of results (1-20), default 10
        search_domain_filter: Optional list of domains to filter (max 20)
        max_tokens_per_page: Optional max tokens per page, default 1024
        country: Optional country code filter (e.g., 'US', 'GB', 'DE')
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        extra_headers: Optional extra headers
        **kwargs: Additional parameters
        
    Returns:
        SearchResponse with results list following Perplexity format
        
    Example:
        ```python
        import litellm
        
        # Basic search
        response = litellm.search(
            query="latest AI developments 2024",
            search_provider="perplexity"
        )
        
        # Search with options
        response = litellm.search(
            query="AI developments",
            search_provider="perplexity",
            max_results=10,
            search_domain_filter=["arxiv.org", "nature.com"],
            max_tokens_per_page=1024,
            country="US"
        )
        
        # Multi-query search
        response = litellm.search(
            query=["AI developments", "machine learning trends"],
            search_provider="perplexity"
        )
        
        # Access results
        for result in response.results:
            print(f"{result.title}: {result.url}")
            print(f"Snippet: {result.snippet}")
            if result.date:
                print(f"Date: {result.date}")
        ```
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("asearch", False) is True
        
        # Validate query parameter
        if not isinstance(query, (str, list)):
            raise ValueError(f"query must be a string or list of strings, got {type(query)}")
        
        if isinstance(query, list) and not all(isinstance(q, str) for q in query):
            raise ValueError("All items in query list must be strings")

        # Get provider config
        search_provider_config: Optional[BaseSearchConfig] = (
            ProviderConfigManager.get_provider_search_config(
                provider=SearchProviders(search_provider),
            )
        )

        if search_provider_config is None:
            raise ValueError(
                f"Search is not supported for provider: {search_provider}"
            )

        verbose_logger.debug(
            f"Search call - provider: {search_provider}"
        )

        # Build optional_params from explicit parameters
        optional_params = _build_search_optional_params(
            max_results=max_results,
            search_domain_filter=search_domain_filter,
            max_tokens_per_page=max_tokens_per_page,
            country=country,
        )
        
        # Filter out internal LiteLLM parameters from kwargs
        filtered_kwargs = filter_out_litellm_params(kwargs=kwargs)
        
        # Add remaining kwargs to optional_params (for provider-specific params)
        for key, value in filtered_kwargs.items():
            if key not in optional_params:
                optional_params[key] = value
        
        verbose_logger.debug(f"Search optional_params: {optional_params}")

        # Validate environment and get headers
        headers = search_provider_config.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers=extra_headers or {},
        )

        # Get complete URL
        complete_url = search_provider_config.get_complete_url(
            api_base=api_base,
            optional_params=optional_params,
        )

        # Pre Call logging
        model_name = f"{search_provider}/search"
        litellm_logging_obj.update_environment_variables(
            model=model_name,
            optional_params=optional_params,
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "api_base": complete_url,
            },
            custom_llm_provider=search_provider,
        )

        # Call the handler
        response = base_llm_http_handler.search(
            query=query,
            optional_params=optional_params,
            timeout=timeout or request_timeout,
            logging_obj=litellm_logging_obj,
            api_key=api_key,
            api_base=complete_url,
            custom_llm_provider=search_provider,
            asearch=_is_async,
            headers=headers,
            provider_config=search_provider_config,
        )

        return response
    except Exception as e:
        model_name = f"{search_provider}/search"
        raise litellm.exception_type(
            model=model_name,
            custom_llm_provider=search_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )

