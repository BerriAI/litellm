"""
LiteLLM SDK Functions for Creating and Searching Vector Stores
"""

import asyncio
import builtins
import contextvars
from collections.abc import Coroutine, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.hidden_params import set_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    VectorStoreEmbeddingExecutor,
    vector_store_request_metadata,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreFileCounts,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchQueryImage,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)
from litellm.utils import ProviderConfigManager, acredential_updates, client, load_credentials_from_list
from litellm.vector_stores.fallbacks import (
    Attempt,
    Failed,
    Served,
    VectorStoreFallbackPlan,
    VectorStoreTarget,
    classify_error,
    current_fallback_plan,
    execute_with_vector_store_fallbacks,
    native_store_id,
    served_api_base,
    served_headers,
)
from litellm.vector_stores.search_query import searched_query, sent_query_image, store_search_params
from litellm.vector_stores.utils import VectorStoreRequestUtils

if TYPE_CHECKING:
    from litellm.router import Router

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


def _direct_vector_store_embedding_executor(
    value: object, router: "Router | None", request_kwargs: Mapping[str, object]
) -> VectorStoreEmbeddingExecutor:
    if value is not None and not isinstance(value, VectorStoreEmbeddingExecutor):
        raise TypeError("Invalid direct vector store embedding executor")
    return BaseQueryEmbeddingVectorStoreConfig.query_embedding_executor(
        value, router, vector_store_request_metadata(request_kwargs)
    )


def _passed_http_client(value: object) -> HTTPHandler | AsyncHTTPHandler | None:
    """Return the caller's client when it is one the HTTP handler would reuse, else None."""
    return value if isinstance(value, (HTTPHandler, AsyncHTTPHandler)) else None


def _items_without_key(values: Mapping[str, object], excluded_key: str) -> Iterator[tuple[str, object]]:
    return ((key, value) for key, value in values.items() if key != excluded_key)


def mock_vector_store_search_response(
    mock_results: builtins.list[VectorStoreSearchResult] | None = None,
):
    """Mock response for vector store search"""
    if mock_results is None:
        mock_results = [
            VectorStoreSearchResult(
                score=0.95,
                content=[
                    VectorStoreResultContent(
                        text="This is a sample search result from the vector store.",
                        type="text",
                    )
                ],
            )
        ]

    return VectorStoreSearchResponse(
        object="vector_store.search_results.page",
        search_query="sample query",
        data=mock_results,
    )


def mock_vector_store_create_response(
    mock_response: VectorStoreCreateResponse | None = None,
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
    name: str | None = None,
    file_ids: builtins.list[str] | None = None,
    expires_after: Mapping[str, object] | None = None,
    chunking_strategy: Mapping[str, object] | None = None,
    metadata: dict[str, str] | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> VectorStoreCreateResponse:
    """
    Async: Create a vector store.
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            custom_llm_provider = "openai"  # Default to OpenAI for vector stores

        func: Final = partial(
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

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

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
    name: str | None = None,
    file_ids: builtins.list[str] | None = None,
    expires_after: Mapping[str, object] | None = None,
    chunking_strategy: Mapping[str, object] | None = None,
    metadata: dict[str, str] | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> VectorStoreCreateResponse | Coroutine[object, object, VectorStoreCreateResponse]:
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
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acreate", False) is True

        # get llm provider logic
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        ## MOCK RESPONSE LOGIC
        if litellm_params.mock_response and isinstance(litellm_params.mock_response, dict):
            return mock_vector_store_create_response(
                mock_response=VectorStoreCreateResponse(**litellm_params.mock_response)
            )

        # Default to OpenAI for vector stores
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        if "/" in custom_llm_provider:
            api_type, custom_llm_provider, _, _ = get_llm_provider(
                model=custom_llm_provider,
                custom_llm_provider=None,
                litellm_params=None,
            )
        else:
            api_type = None
            custom_llm_provider = custom_llm_provider

        # get provider config - using vector store custom logger for now
        vector_store_provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
            api_type=api_type,
        )

        if vector_store_provider_config is None:
            raise ValueError(f"Vector store create is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)

        # Get VectorStoreCreateOptionalRequestParams with only valid parameters
        vector_store_create_optional_params: Final[VectorStoreCreateOptionalRequestParams] = (
            VectorStoreRequestUtils.get_requested_vector_store_create_optional_param(local_vars)
        )

        # Pre Call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
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

        response: Final = base_llm_http_handler.vector_store_create_handler(
            vector_store_create_optional_params=vector_store_create_optional_params,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=_passed_http_client(kwargs.get("client")),
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


_SEARCH_FIELDS: Final = TypeAdapter(dict[str, object])
_SEARCH_METADATA: Final = frozenset(
    {
        "metadata",
        "litellm_metadata",
        "litellm_call_id",
        "proxy_server_request",
        "litellm_trace_id",
        "userContext",
        "user_context",
        "drop_params",
    }
)


class _SearchEmbeddingExecutor:
    def __init__(self, inner: VectorStoreEmbeddingExecutor) -> None:
        self.inner: Final = inner
        self.failed: bool = False

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        try:
            return self.inner.embed(model, query, configuration)
        except Exception:  # noqa: BLE001  # failure attribution must distinguish embedding from vector storage
            self.failed = True
            raise

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        try:
            return await self.inner.aembed(model, query, configuration)
        except Exception:  # noqa: BLE001  # failure attribution must distinguish embedding from vector storage
            self.failed = True
            raise


@dataclass(frozen=True, slots=True)
class _FallbackSearch:
    plan: VectorStoreFallbackPlan
    query: str | tuple[str, ...] | None
    options: Mapping[str, object]
    query_image: VectorStoreSearchQueryImage | None
    timeout: float | httpx.Timeout | None
    logger: LiteLLMLoggingObj
    embedding: VectorStoreEmbeddingExecutor
    metadata: Mapping[str, object]
    http_client: AsyncHTTPHandler | None


def _optional_search_mapping(value: object) -> Mapping[str, object] | None:
    return None if value is None else _SEARCH_FIELDS.validate_python(value)


async def _fallback_search_attempt(
    request: _FallbackSearch, target: VectorStoreTarget
) -> Attempt[VectorStoreSearchResponse]:
    registered: Final = store_search_params(target.params)
    connection: Final = {**registered, **await acredential_updates(registered)}
    provider_name, separator, api_type = target.provider.partition("/")
    resolved: Final = VectorStoreTarget(target.store_id, provider_name, MappingProxyType(connection))
    physical_id: Final = native_store_id(target)
    provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
        provider=litellm.LlmProviders(provider_name), api_type=api_type if separator else None
    )
    if provider_config is None:
        return Failed(target, ValueError(f"Vector store search is not supported for {target.provider}"), False)
    litellm_params: Final = GenericLiteLLMParams(**{**connection, **request.metadata, "vector_store_id": physical_id})
    searched: Final = searched_query(
        query=builtins.list(request.query) if isinstance(request.query, tuple) else request.query,
        query_image=request.query_image,
        custom_llm_provider=provider_name,
        supports_query_image="query_image" in provider_config.get_supported_openai_params(model=""),
        drop_params=drops_unsupported_params(request.metadata.get("drop_params")),
    )
    options: Final = VectorStoreRequestUtils.get_requested_vector_store_search_optional_param(
        {
            **connection,
            **{key: value for key, value in request.options.items() if value is not None},
            "filters": provider_config.combine_search_filters(
                connection.get("filters"), request.options.get("filters")
            ),
            "query_image": searched.image,
            "drop_params": request.metadata.get("drop_params"),
        },
        provider_config,
    )
    mapped_options: Final = _SEARCH_FIELDS.validate_python(options)
    if connection.get("filters") is not None and mapped_options.get("filters") is None:
        raise litellm.BadRequestError(
            message="Search parameter mapping cannot drop a registered vector store filter",
            model=physical_id,
            llm_provider=provider_name,
        )
    embedding: Final = _SearchEmbeddingExecutor(request.embedding)
    extra_headers: Final = _optional_search_mapping(connection.get("extra_headers"))
    extra_body: Final = _optional_search_mapping(connection.get("extra_body"))
    request.logger.update_from_kwargs(  # pyright: ignore[reportUnknownMemberType]  # legacy logging boundary
        kwargs={**connection, **request.metadata},
        model=None,
        optional_params={"vector_store_id": physical_id, "query": searched.query, **options},
        litellm_params={**request.metadata, **litellm_params.model_dump(exclude_none=True)},
        custom_llm_provider=provider_name,
    )
    try:
        response: Final = await base_llm_http_handler.async_vector_store_search_handler(
            vector_store_id=physical_id,
            query=searched.query,
            vector_store_search_optional_params=options,
            vector_store_provider_config=provider_config,
            custom_llm_provider=provider_name,
            litellm_params=litellm_params,
            logging_obj=request.logger,
            embedding_executor=embedding,
            extra_headers=None if extra_headers is None else dict(extra_headers),
            extra_body=None if extra_body is None else dict(extra_body),
            timeout=request.timeout or request_timeout,
            client=request.http_client,
            _is_async=True,
        )
    except Exception as error:  # noqa: BLE001  # classify only the provider attempt, never preparation or cancellation
        return Failed(resolved, error, False) if embedding.failed else classify_error(resolved, error)
    return Served(resolved, response)


async def _search_with_fallbacks(request: _FallbackSearch) -> VectorStoreSearchResponse:
    outcome: Final = await execute_with_vector_store_fallbacks(request.plan, partial(_fallback_search_attempt, request))
    if isinstance(outcome, Failed):
        raise litellm.exception_type(
            model=outcome.target.store_id,
            custom_llm_provider=outcome.target.provider,
            original_exception=outcome.error or ValueError("No configured vector store served the search"),
            completion_kwargs={},
            extra_kwargs={},
        )
    set_hidden_params(
        outcome.value,
        {
            "api_base": served_api_base(outcome.target),
            "additional_headers": dict(served_headers(outcome)),
        },
    )
    return outcome.value


@client
async def asearch(
    vector_store_id: str,
    query: str | builtins.list[str] | None = None,
    filters: Mapping[str, object] | None = None,
    max_num_results: int | None = None,
    ranking_options: Mapping[str, object] | None = None,
    rewrite_query: bool | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    router: "Router | None" = None,
    *,
    query_image: VectorStoreSearchQueryImage | None = None,
    **kwargs: object,
) -> VectorStoreSearchResponse:
    """
    Async: Search a vector store for relevant chunks based on a query and file attributes filter.
    """
    embedding_executor: Final = _direct_vector_store_embedding_executor(
        kwargs.pop("_direct_vector_store_embedding_executor", None), router, kwargs
    )
    local_vars: Final = dict(_items_without_key(locals(), "embedding_executor"))
    if "fallbacks" in kwargs:
        raise ValueError("Configure fallbacks on a registered vector store, not on an SDK request")
    fallback_plan: Final = current_fallback_plan(vector_store_id)
    if fallback_plan is not None:
        request_fields: Final = _SEARCH_FIELDS.validate_python(kwargs)
        logger: Final = request_fields.get("litellm_logging_obj")
        if not isinstance(logger, LiteLLMLoggingObj):
            raise TypeError("Vector store fallback requires the search logging context")
        supplied_client: Final = request_fields.get("client")
        return await _search_with_fallbacks(
            _FallbackSearch(
                plan=fallback_plan,
                query=tuple(query) if isinstance(query, builtins.list) else query,
                options=MappingProxyType(
                    {
                        "filters": _optional_search_mapping(filters),
                        "max_num_results": max_num_results,
                        "ranking_options": _optional_search_mapping(ranking_options),
                        "rewrite_query": rewrite_query,
                    }
                ),
                query_image=query_image,
                timeout=timeout,
                logger=logger,
                embedding=embedding_executor,
                metadata=MappingProxyType(
                    {key: value for key, value in request_fields.items() if key in _SEARCH_METADATA}
                ),
                http_client=supplied_client if isinstance(supplied_client, AsyncHTTPHandler) else None,
            )
        )

    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["asearch"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            custom_llm_provider = "openai"  # Default to OpenAI for vector stores

        func: Final = partial(
            search,
            vector_store_id=vector_store_id,
            query=query,
            filters=filters,
            max_num_results=max_num_results,
            ranking_options=ranking_options,
            rewrite_query=rewrite_query,
            query_image=query_image,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            _direct_vector_store_embedding_executor=embedding_executor,
            router=router,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

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
    query: str | builtins.list[str] | None = None,
    filters: Mapping[str, object] | None = None,
    max_num_results: int | None = None,
    ranking_options: Mapping[str, object] | None = None,
    rewrite_query: bool | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    router: "Router | None" = None,
    *,
    query_image: VectorStoreSearchQueryImage | None = None,
    **kwargs: object,
) -> VectorStoreSearchResponse | Coroutine[object, object, VectorStoreSearchResponse]:
    """
    Search a vector store for relevant chunks based on a query and file attributes filter.

    Args:
        vector_store_id: The ID of the vector store to search.
        query: A query string or array for the search. Optional when query_image is sent.
        filters: Optional filter to apply based on file attributes.
        max_num_results: Maximum number of results to return (1-50, default 10).
        ranking_options: Optional ranking options for search.
        rewrite_query: Whether to rewrite the natural language query for vector search.
        query_image: A base64 image to search with in place of query, on providers that support it.

    Returns:
        VectorStoreSearchResponse containing the search results.
    """
    embedding_executor: Final = _direct_vector_store_embedding_executor(
        kwargs.pop("_direct_vector_store_embedding_executor", None), router, kwargs
    )
    if "fallbacks" in kwargs:
        raise ValueError("Configure fallbacks on a registered vector store, not on an SDK request")
    local_vars: Final = dict(_items_without_key(locals(), "embedding_executor"))
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("asearch", False) is True
        # pull credentials from registry if available
        if litellm.vector_store_registry is not None and vector_store_id is not None:
            try:
                registry_credentials = litellm.vector_store_registry.get_credentials_for_vector_store(vector_store_id)
                kwargs.update(registry_credentials)
            except Exception:
                pass
            load_credentials_from_list(kwargs)

        # get llm provider logic
        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        ## MOCK RESPONSE LOGIC
        if litellm_params.mock_response and isinstance(litellm_params.mock_response, (str, builtins.list)):
            mock_results = None
            if isinstance(litellm_params.mock_response, builtins.list):
                mock_results = litellm_params.mock_response
            return mock_vector_store_search_response(mock_results=mock_results)

        # Default to OpenAI for vector stores
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        if "/" in custom_llm_provider:
            api_type, custom_llm_provider, _, _ = get_llm_provider(
                model=custom_llm_provider,
                custom_llm_provider=None,
                litellm_params=None,
            )
        else:
            api_type = None
            custom_llm_provider = custom_llm_provider

        # get provider config - using vector store custom logger for now
        vector_store_provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
            api_type=api_type,
        )

        if vector_store_provider_config is None:
            raise ValueError(f"Vector store search is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)

        search_query: Final = searched_query(
            query=query,
            query_image=sent_query_image(query_image, extra_body),
            custom_llm_provider=custom_llm_provider,
            supports_query_image="query_image" in vector_store_provider_config.get_supported_openai_params(model=""),
            drop_params=drops_unsupported_params(local_vars.get("drop_params")),
        )

        # Get VectorStoreSearchOptionalRequestParams with only valid parameters
        vector_store_search_optional_params: Final[VectorStoreSearchOptionalRequestParams] = (
            VectorStoreRequestUtils.get_requested_vector_store_search_optional_param(
                {**local_vars, "query_image": search_query.image},
                vector_store_provider_config=vector_store_provider_config,
            )
        )

        sent_extra_body: Final = as_str_mapping(extra_body)

        # Pre Call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=api_type,
            optional_params={
                "vector_store_id": vector_store_id,
                "query": search_query.query,
                **vector_store_search_optional_params,
            },
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "vector_store_id": vector_store_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_search_handler(
            vector_store_id=vector_store_id,
            query=search_query.query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            embedding_executor=embedding_executor,
            extra_headers=extra_headers,
            extra_body=extra_body
            if sent_extra_body is None
            else dict(_items_without_key(sent_extra_body, "query_image")),
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=_passed_http_client(kwargs.get("client")),
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
async def aretrieve(
    vector_store_id: str,
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> VectorStoreCreateResponse:
    """
    Async: Retrieve a vector store.
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aretrieve"] = True

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        func: Final = partial(
            retrieve,
            vector_store_id=vector_store_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

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
def retrieve(
    vector_store_id: str,
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> VectorStoreCreateResponse | Coroutine[object, object, VectorStoreCreateResponse]:
    """
    Retrieve a vector store.

    Args:
        vector_store_id: The ID of the vector store to retrieve.

    Returns:
        VectorStoreCreateResponse containing the vector store details.
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aretrieve", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        if "/" in custom_llm_provider:
            api_type, custom_llm_provider, _, _ = get_llm_provider(
                model=custom_llm_provider,
                custom_llm_provider=None,
                litellm_params=None,
            )
        else:
            api_type = None
            custom_llm_provider = custom_llm_provider

        vector_store_provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
            api_type=api_type,
        )

        if vector_store_provider_config is None:
            raise ValueError(f"Vector store retrieve is not supported for {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"vector_store_id": vector_store_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_retrieve_handler(
            vector_store_id=vector_store_id,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=_passed_http_client(kwargs.get("client")),
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
async def alist(
    after: str | None = None,
    before: str | None = None,
    limit: int | None = 20,
    order: str | None = "desc",
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> object:
    """
    Async: List vector stores.
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist"] = True

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        func: Final = partial(
            list,
            after=after,
            before=before,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response: object = await init_response
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
def list(
    after: str | None = None,
    before: str | None = None,
    limit: int | None = 20,
    order: str | None = "desc",
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> object | Coroutine[object, object, object]:
    """
    List vector stores.

    Args:
        after: A cursor for use in pagination.
        before: A cursor for use in pagination.
        limit: A limit on the number of objects to be returned.
        order: Sort order by the created_at timestamp.

    Returns:
        List of vector stores.
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("alist", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        if "/" in custom_llm_provider:
            api_type, custom_llm_provider, _, _ = get_llm_provider(
                model=custom_llm_provider,
                custom_llm_provider=None,
                litellm_params=None,
            )
        else:
            api_type = None
            custom_llm_provider = custom_llm_provider

        vector_store_provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
            api_type=api_type,
        )

        if vector_store_provider_config is None:
            raise ValueError(f"Vector store list is not supported for {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "after": after,
                "before": before,
                "limit": limit,
                "order": order,
            },
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_list_handler(
            after=after,
            before=before,
            limit=limit,
            order=order,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=_passed_http_client(kwargs.get("client")),
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
async def aupdate(
    vector_store_id: str,
    name: str | None = None,
    expires_after: Mapping[str, object] | None = None,
    metadata: dict[str, str] | None = None,
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> VectorStoreCreateResponse:
    """
    Async: Update a vector store.
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aupdate"] = True

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        func: Final = partial(
            update,
            vector_store_id=vector_store_id,
            name=name,
            expires_after=expires_after,
            metadata=metadata,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

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
def update(
    vector_store_id: str,
    name: str | None = None,
    expires_after: Mapping[str, object] | None = None,
    metadata: dict[str, str] | None = None,
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> VectorStoreCreateResponse | Coroutine[object, object, VectorStoreCreateResponse]:
    """
    Update a vector store.

    Args:
        vector_store_id: The ID of the vector store to update.
        name: The name of the vector store.
        expires_after: The expiration policy for the vector store.
        metadata: Set of 16 key-value pairs that can be attached to an object.

    Returns:
        VectorStoreCreateResponse containing the updated vector store details.
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aupdate", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        if "/" in custom_llm_provider:
            api_type, custom_llm_provider, _, _ = get_llm_provider(
                model=custom_llm_provider,
                custom_llm_provider=None,
                litellm_params=None,
            )
        else:
            api_type = None
            custom_llm_provider = custom_llm_provider

        vector_store_provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
            api_type=api_type,
        )

        if vector_store_provider_config is None:
            raise ValueError(f"Vector store update is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)

        vector_store_update_optional_params: Final[VectorStoreCreateOptionalRequestParams] = (
            VectorStoreRequestUtils.get_requested_vector_store_create_optional_param(local_vars)
        )

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "name": name,
                **vector_store_update_optional_params,
            },
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_update_handler(
            vector_store_id=vector_store_id,
            vector_store_update_optional_params=vector_store_update_optional_params,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=_passed_http_client(kwargs.get("client")),
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
async def adelete(
    vector_store_id: str,
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> object:
    """
    Async: Delete a vector store.
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete"] = True

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        func: Final = partial(
            delete,
            vector_store_id=vector_store_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response: object = await init_response
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
def delete(
    vector_store_id: str,
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: object,
) -> object | Coroutine[object, object, object]:
    """
    Delete a vector store.

    Args:
        vector_store_id: The ID of the vector store to delete.

    Returns:
        Deletion confirmation response.
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("adelete", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        if "/" in custom_llm_provider:
            api_type, custom_llm_provider, _, _ = get_llm_provider(
                model=custom_llm_provider,
                custom_llm_provider=None,
                litellm_params=None,
            )
        else:
            api_type = None
            custom_llm_provider = custom_llm_provider

        vector_store_provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders(custom_llm_provider),
            api_type=api_type,
        )

        if vector_store_provider_config is None:
            raise ValueError(f"Vector store delete is not supported for {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"vector_store_id": vector_store_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_delete_handler(
            vector_store_id=vector_store_id,
            vector_store_provider_config=vector_store_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=_passed_http_client(kwargs.get("client")),
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
