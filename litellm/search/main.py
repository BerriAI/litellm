"""
Main Search function for LiteLLM.
"""

import asyncio
import contextvars
from collections.abc import Coroutine, Mapping
from functools import partial
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    FetchResponse,
    SearchResponse,
    to_fetch_response,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager, client, filter_out_litellm_params

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


_SEARCH_KWARGS: Final = TypeAdapter(Mapping[str, object])


def _search_pricing_params(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Always carries the key so an unpriced fallback tool clears the price a failed primary left in the
    shared logging object (`Logging.update_environment_variables` merges litellm_params)."""
    return MappingProxyType({"input_cost_per_query": kwargs.get("input_cost_per_query")})


def _build_search_optional_params(
    max_results: int | None = None,
    search_domain_filter: list[str] | None = None,
    max_tokens_per_page: int | None = None,
    country: str | None = None,
) -> dict[str, object]:
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
    optional_params: Final[dict[str, object]] = {}

    if max_results is not None:
        optional_params["max_results"] = max_results
    if search_domain_filter is not None:
        optional_params["search_domain_filter"] = search_domain_filter
    if max_tokens_per_page is not None:
        optional_params["max_tokens_per_page"] = max_tokens_per_page
    if country is not None:
        optional_params["country"] = country

    return optional_params


def _tool_model(search_provider: str, fetch_url: str | None) -> str:
    is_fetch: Final = search_provider == SearchProviders.DEEPSEEK.value and isinstance(fetch_url, str)
    kind: Final = "fetch" if is_fetch else "search"
    return f"{search_provider}/{kind}"


def _present_search_response(response: SearchResponse) -> SearchResponse:
    if response.object != "fetch":
        return response
    return to_fetch_response(response)  # pyright: ignore[reportReturnType]  # fetch pages are a FetchResponse at runtime


def _require_deepseek_fetch(fetch_provider: str) -> None:
    if fetch_provider == SearchProviders.DEEPSEEK.value:
        return
    raise BadRequestError(
        message=f"Fetch is only supported for the deepseek provider, got {fetch_provider}.",
        model=f"{fetch_provider}/fetch",
        llm_provider=fetch_provider,
    )


@client
async def asearch(
    query: str | list[str],
    search_provider: str,
    max_results: int | None = None,
    search_domain_filter: list[str] | None = None,
    max_tokens_per_page: int | None = None,
    country: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    extra_headers: dict[str, object] | None = None,
    *,
    fetch_url: str | None = None,
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
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["asearch"] = True

        func: Final = partial(
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
            fetch_url=fetch_url,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        if response is None:
            raise ValueError(f"Got an unexpected None response from the Search API: {response}")

        return _present_search_response(response)
    except Exception as e:
        raise litellm.exception_type(
            model=_tool_model(search_provider, fetch_url),
            custom_llm_provider=search_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def search(
    query: str | list[str],
    search_provider: str,
    max_results: int | None = None,
    search_domain_filter: list[str] | None = None,
    max_tokens_per_page: int | None = None,
    country: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    extra_headers: dict[str, object] | None = None,
    *,
    fetch_url: str | None = None,
    **kwargs,
) -> SearchResponse | Coroutine[object, object, SearchResponse]:
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
    local_vars: Final = locals()
    tool_model: Final = _tool_model(search_provider, fetch_url)
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.pop("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("asearch", False) is True

        # Validate query parameter
        if not isinstance(query, (str, list)):
            raise ValueError(f"query must be a string or list of strings, got {type(query)}")

        if isinstance(query, list) and not all(isinstance(q, str) for q in query):
            raise ValueError("All items in query list must be strings")

        # Get provider config
        search_provider_config: Final[BaseSearchConfig | None] = ProviderConfigManager.get_provider_search_config(
            provider=SearchProviders(search_provider),
        )

        if search_provider_config is None:
            raise ValueError(f"Search is not supported for provider: {search_provider}")

        verbose_logger.debug("Search call - provider: %s", search_provider)

        # Build optional_params from explicit parameters
        base_optional: Final = _build_search_optional_params(
            max_results=max_results,
            search_domain_filter=search_domain_filter,
            max_tokens_per_page=max_tokens_per_page,
            country=country,
        )

        # Filter out internal LiteLLM parameters from kwargs
        search_kwargs: Final = _SEARCH_KWARGS.validate_python(kwargs)
        filtered_kwargs: Final = filter_out_litellm_params(kwargs=search_kwargs)
        forwarded: Final = {key: value for key, value in filtered_kwargs.items() if key not in base_optional}
        fetch_param: Final[Mapping[str, object]] = (
            {"fetch_url": fetch_url} if _tool_model(search_provider, fetch_url).endswith("/fetch") else {}
        )
        optional_params: Final = {**base_optional, **forwarded, **fetch_param}

        verbose_logger.debug("Search optional_params: %s", optional_params)

        # Validate environment and get headers
        headers: Final = search_provider_config.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers=extra_headers or {},
        )

        # Get complete URL
        complete_url: Final = search_provider_config.get_complete_url(
            api_base=api_base,
            optional_params=optional_params,
            api_key=api_key,
        )

        # Pre Call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=tool_model,
            optional_params=optional_params,
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "api_base": complete_url,
                **_search_pricing_params(search_kwargs),
            },
            custom_llm_provider=search_provider,
        )

        # Call the handler
        response: Final = base_llm_http_handler.search(
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

        if _is_async or not isinstance(response, SearchResponse):
            return response
        return _present_search_response(response)
    except Exception as e:
        raise litellm.exception_type(
            model=tool_model,
            custom_llm_provider=search_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


def fetch(
    url: str,
    fetch_provider: str = SearchProviders.DEEPSEEK.value,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> FetchResponse:
    """Fetch one page through a provider tool. DeepSeek is the supported provider."""
    _require_deepseek_fetch(fetch_provider)
    response: Final = search(
        query=url,
        search_provider=fetch_provider,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        extra_headers=dict(extra_headers) if extra_headers is not None else None,
        fetch_url=url,
    )
    if isinstance(response, FetchResponse):
        return response
    raise BadRequestError(
        message="DeepSeek Fetch did not return a page.",
        model="deepseek/fetch",
        llm_provider=SearchProviders.DEEPSEEK.value,
    )


async def afetch(
    url: str,
    fetch_provider: str = SearchProviders.DEEPSEEK.value,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> FetchResponse:
    """Async fetch of one page through a provider tool. DeepSeek is the supported provider."""
    _require_deepseek_fetch(fetch_provider)
    response: Final = await asearch(
        query=url,
        search_provider=fetch_provider,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        extra_headers=dict(extra_headers) if extra_headers is not None else None,
        fetch_url=url,
    )
    if isinstance(response, FetchResponse):
        return response
    raise BadRequestError(
        message="DeepSeek Fetch did not return a page.",
        model="deepseek/fetch",
        llm_provider=SearchProviders.DEEPSEEK.value,
    )
