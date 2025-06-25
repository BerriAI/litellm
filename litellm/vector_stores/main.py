"""
LiteLLM SDK Functions for Creating and Searching Vector Stores
"""
import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Optional, Union

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreFileCounts,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)
from litellm.utils import ProviderConfigManager, client
from litellm.vector_stores.utils import VectorStoreRequestUtils

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


def mock_vector_store_search_response(
    mock_results: Optional[List[VectorStoreSearchResult]] = None,
):
    """Mock response for vector store search"""
    if mock_results is None:
        mock_results = [
            VectorStoreSearchResult(
                score=0.95,
                content=[
                    VectorStoreResultContent(
                        text="This is a sample search result from the vector store.",
                        type="text"
                    )
                ]
            )
        ]
    
    return VectorStoreSearchResponse(
        object="vector_store.search_results.page",
        search_query="sample query",
        data=mock_results,
    )


def mock_vector_store_create_response(
    mock_response: Optional[VectorStoreCreateResponse] = None,
):
    """Mock response for vector store create"""
    if mock_response is None:
        mock_response = VectorStoreCreateResponse(
            id="vs_mock123",
            object="vector_store",
            created_at=1699061776,
            name="Mock Vector Store",
            bytes=0,
            file_counts=VectorStoreFileCounts(
                in_progress=0,
                completed=0,
                failed=0,
                cancelled=0,
                total=0,
            ),
            status="completed",
            expires_after=None,
            expires_at=None,
            last_active_at=None,
            metadata=None,
        )
    
    return mock_response


@client
async def acreate(
    name: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    expires_after: Optional[Dict] = None,
    chunking_strategy: Optional[Dict] = None,
    metadata: Optional[Dict[str, str]] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreCreateResponse:
    """
    Async: Create a vector store.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            custom_llm_provider = "openai"  # Default to OpenAI for vector stores

        func = partial(
            create,
            name=name,
            file_ids=file_ids,
            expires_after=expires_after,
            chunking_strategy=chunking_strategy,
            metadata=metadata,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create(
    name: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    expires_after: Optional[Dict] = None,
    chunking_strategy: Optional[Dict] = None,
    metadata: Optional[Dict[str, str]] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[VectorStoreCreateResponse, Coroutine[Any, Any, VectorStoreCreateResponse]]:
    """
    Create a vector store.
    
    Args:
        name: The name of the vector store.
        file_ids: A list of File IDs that the vector store should use.
        expires_after: The expiration policy for the vector store.
        chunking_strategy: The chunking strategy used to chunk the file(s).
        metadata: Set of 16 key-value pairs that can be attached to an object.
        
    Returns:
        VectorStoreCreateResponse containing the created vector store details.
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acreate", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        ## MOCK RESPONSE LOGIC
        if litellm_params.mock_response and isinstance(
            litellm_params.mock_response, dict
        ):
            return mock_vector_store_create_response(
                mock_response=VectorStoreCreateResponse(**litellm_params.mock_response)
            )

        # Default to OpenAI for vector stores
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # get provider config - using vector store custom logger for now
        vector_store_provider_config = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if vector_store_provider_config is None:
            raise ValueError(
                f"Vector store create is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)
        
        # Get VectorStoreCreateOptionalRequestParams with only valid parameters
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams = (
            VectorStoreRequestUtils.get_requested_vector_store_create_optional_param(
                local_vars
            )
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "name": name,
                **vector_store_create_optional_params,
            },
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_create_handler(
            vector_store_create_optional_params=vector_store_create_optional_params,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )
        
        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def asearch(
    vector_store_id: str,
    query: Union[str, List[str]],
    filters: Optional[Dict] = None,
    max_num_results: Optional[int] = None,
    ranking_options: Optional[Dict] = None,
    rewrite_query: Optional[bool] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VectorStoreSearchResponse:
    """
    Async: Search a vector store for relevant chunks based on a query and file attributes filter.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["asearch"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            custom_llm_provider = "openai"  # Default to OpenAI for vector stores

        func = partial(
            search,
            vector_store_id=vector_store_id,
            query=query,
            filters=filters,
            max_num_results=max_num_results,
            ranking_options=ranking_options,
            rewrite_query=rewrite_query,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def search(
    vector_store_id: str,
    query: Union[str, List[str]],
    filters: Optional[Dict] = None,
    max_num_results: Optional[int] = None,
    ranking_options: Optional[Dict] = None,
    rewrite_query: Optional[bool] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[VectorStoreSearchResponse, Coroutine[Any, Any, VectorStoreSearchResponse]]:
    """
    Search a vector store for relevant chunks based on a query and file attributes filter.
    
    Args:
        vector_store_id: The ID of the vector store to search.
        query: A query string or array for the search.
        filters: Optional filter to apply based on file attributes.
        max_num_results: Maximum number of results to return (1-50, default 10).
        ranking_options: Optional ranking options for search.
        rewrite_query: Whether to rewrite the natural language query for vector search.
        
    Returns:
        VectorStoreSearchResponse containing the search results.
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("asearch", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        ## MOCK RESPONSE LOGIC
        if litellm_params.mock_response and isinstance(
            litellm_params.mock_response, (str, list)
        ):
            mock_results = None
            if isinstance(litellm_params.mock_response, list):
                mock_results = litellm_params.mock_response
            return mock_vector_store_search_response(mock_results=mock_results)

        # Default to OpenAI for vector stores
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # get provider config - using vector store custom logger for now
        vector_store_provider_config = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if vector_store_provider_config is None:
            raise ValueError(
                f"Vector store search is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)
        
        # Get VectorStoreSearchOptionalRequestParams with only valid parameters
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams = (
            VectorStoreRequestUtils.get_requested_vector_store_search_optional_param(
                local_vars
            )
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "query": query,
                **vector_store_search_optional_params,
            },
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "vector_store_id": vector_store_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.vector_store_search_handler(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )
        
        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )