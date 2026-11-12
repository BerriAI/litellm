#### Search Endpoints #####

from collections.abc import Callable, Mapping
from typing import Final

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.search.transformation import FetchResponse, SearchResponse
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError

router: Final = APIRouter()


class FetchPageRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: StrictStr
    search_tool_name: str | None = None


class ToolSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    query: StrictStr | tuple[StrictStr, ...]
    count: StrictInt | None = None
    max_results: StrictInt | None = None
    search_domain_filter: tuple[str, ...] | None = None
    country: str | None = None
    max_tokens_per_page: StrictInt | None = None
    search_tool_name: str | None = None
    need_content: bool | None = None
    ori_question: str | None = None
    localization: str | None = None


def _request_mapping(body: BaseModel) -> Mapping[str, object]:
    extra: Final = body.__pydantic_extra__ or {}
    dumped: Final = body.model_dump(exclude_none=True, mode="json")
    return {**dumped, **extra}


def _tool_search_data(body: ToolSearchRequest, search_tool_name: str | None) -> Mapping[str, object]:
    fields: Final = _request_mapping(body)
    count: Final = fields.get("count")
    without_count: Final = {key: value for key, value in fields.items() if key != "count"}
    counted: Final = (
        without_count
        if "max_results" in without_count or isinstance(count, bool) or not isinstance(count, int)
        else {**without_count, "max_results": count}
    )
    tool_name: Final = search_tool_name if search_tool_name is not None else body.search_tool_name
    if tool_name is None:
        return counted
    return {**counted, "search_tool_name": tool_name}


def _fetch_data(body: FetchPageRequest, search_tool_name: str | None) -> Mapping[str, object]:
    tool_name: Final = search_tool_name if search_tool_name is not None else body.search_tool_name
    payload: Final[Mapping[str, object]] = {"query": body.url, "fetch_url": body.url}
    if tool_name is None:
        return payload
    return {**payload, "search_tool_name": tool_name}


_STRING_KEY_DICT: Final = TypeAdapter(dict[str, object])


def _metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    return _parsed_object(data.get("metadata"))


def _parsed_object(raw: object) -> Mapping[str, object]:
    try:
        return _STRING_KEY_DICT.validate_python(raw)
    except ValidationError:
        return {}


@router.post(
    "/v1/search/{search_tool_name}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["search"],
)
@router.post(
    "/search/{search_tool_name}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["search"],
)
@router.post(
    "/v1/search",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["search"],
)
@router.post(
    "/search",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["search"],
)
async def search(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    search_tool_name: str | None = None,
):
    """
    Search endpoint for performing web searches.
    
    Follows the Perplexity Search API spec:
    https://docs.perplexity.ai/api-reference/search-post
    
    The search_tool_name can be passed either:
    1. In the URL path: /v1/search/{search_tool_name}
    2. In the request body: {"search_tool_name": "..."}
    
    Example with search_tool_name in URL (recommended - keeps body Perplexity-compatible):
    ```bash
    curl -X POST "http://localhost:4000/v1/search/litellm-search" \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "latest AI developments 2024",
            "max_results": 5,
            "search_domain_filter": ["arxiv.org", "nature.com"],
            "country": "US"
        }'
    ```
    
    Example with search_tool_name in body:
    ```bash
    curl -X POST "http://localhost:4000/v1/search" \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "search_tool_name": "litellm-search",
            "query": "latest AI developments 2024",
            "max_results": 5,
            "search_domain_filter": ["arxiv.org", "nature.com"],
            "country": "US"
        }'
    ```
    
    Request Body Parameters (when search_tool_name not in URL):
    - search_tool_name (str, required if not in URL): Name of the search tool configured in router
    - query (str or list[str], required): Search query
    - max_results (int, optional): Maximum number of results (1-20), default 10
    - search_domain_filter (list[str], optional): List of domains to filter (max 20)
    - max_tokens_per_page (int, optional): Max tokens per page, default 1024
    - country (str, optional): Country code filter (e.g., 'US', 'GB', 'DE')
    
    When using URL path parameter, only Perplexity-compatible parameters are needed in body:
    - query (str or list[str], required): Search query
    - max_results (int, optional): Maximum number of results (1-20), default 10
    - search_domain_filter (list[str], optional): List of domains to filter (max 20)
    - max_tokens_per_page (int, optional): Max tokens per page, default 1024
    - country (str, optional): Country code filter (e.g., 'US', 'GB', 'DE')
    
    Response follows Perplexity Search API format:
    ```json
    {
        "object": "search",
        "results": [
            {
                "title": "Result title",
                "url": "https://example.com",
                "snippet": "Result snippet...",
                "date": "2024-01-01",
                "last_updated": "2024-01-01"
            }
        ]
    }
    ```
    """
    body: Final = await request.body()
    incoming: Final = _parsed_object(orjson.loads(body))
    named: Final[Mapping[str, object]] = {"search_tool_name": search_tool_name} if search_tool_name is not None else {}
    return await _dispatch_search(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        data={**incoming, **named},
    )


def _routed_tool_fields(
    data: Mapping[str, object],
    llm_router: object,
    tool_name: str,
) -> Mapping[str, object]:
    from litellm import Router

    model_field: Final[Mapping[str, object]] = {"model": tool_name}
    if not isinstance(llm_router, Router) or not llm_router.settings().is_set("search_tools"):
        return model_field
    router_search_tools: Final = llm_router.settings().search_tools
    verbose_proxy_logger.debug(
        "Search endpoint - Looking for search_tool_name: %s. Available search tools in router: %s. Total search tools: %s",
        tool_name,
        [tool.get("search_tool_name") for tool in router_search_tools],
        len(router_search_tools),
    )
    matching_tools: Final = [tool for tool in router_search_tools if tool.get("search_tool_name") == tool_name]
    if not matching_tools:
        return model_field
    search_provider: Final = matching_tools[0].get("litellm_params", {}).get("search_provider")
    return {
        **model_field,
        **({"custom_llm_provider": search_provider} if search_provider else {}),
        "metadata": {**_metadata(data), "model_group": tool_name},
    }


async def _dispatch_search(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    data: Mapping[str, object],
) -> object:
    from litellm.proxy.auth.auth_checks import can_token_call_search_tool
    from litellm.proxy.common_utils.http_parsing_utils import resolve_inference_model
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    routed_search_tool_name: Final = data.get("search_tool_name") or resolve_inference_model(
        data.get("model"), general_settings, user_model
    )
    if not isinstance(routed_search_tool_name, str) or not routed_search_tool_name:
        raise ProxyMissingRequiredParamError(route="/search", param="search_tool_name")
    try:
        await can_token_call_search_tool(search_tool_name=routed_search_tool_name, valid_token=user_api_key_dict)
    except ProxyException as e:
        verbose_proxy_logger.debug("Search tool authorization denied: %s", e.type)
        raise

    search_tool_name_value: Final = data.get("search_tool_name")
    routed_tool: Final[Mapping[str, object]] = (
        _routed_tool_fields(data, llm_router, search_tool_name_value)
        if isinstance(search_tool_name_value, str) and search_tool_name_value
        else {}
    )
    provider: Final = routed_tool.get("custom_llm_provider", data.get("custom_llm_provider"))
    if data.get("fetch_url") is not None and provider != "deepseek":
        raise ProxyException(
            message=(
                f"Fetch is only available for search provider deepseek, got {provider}."
                if isinstance(provider, str)
                else "Fetch needs a search tool whose provider is deepseek."
            ),
            type="invalid_request_error",
            param="search_tool_name",
            code=400,
        )

    # Team context is also injected later, and is set here so direct router paths still see it.
    caller_metadata: Final[Mapping[str, object]] = {
        **(
            {"user_api_key_team_metadata": user_api_key_dict.team_metadata}
            if getattr(user_api_key_dict, "team_metadata", None) is not None
            else {}
        ),
        **(
            {"user_api_key_team_id": user_api_key_dict.team_id}
            if getattr(user_api_key_dict, "team_id", None) is not None
            else {}
        ),
    }
    request_data: Final[Mapping[str, object]] = {
        **data,
        **routed_tool,
        "metadata": {**_metadata({**data, **routed_tool}), **caller_metadata},
    }

    processor: Final = ProxyBaseLLMRequestProcessing(data=request_data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="asearch",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


async def fetch_page(
    request: Request,
    fastapi_response: Response,
    body: FetchPageRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    search_tool_name: str | None = None,
) -> object:
    """
    Fetch one web page through a configured search tool whose provider is deepseek.

    The body is DeepSeek's fetch request, `{"url": "https://example.com/article"}`.
    `search_tool_name` can be the path (`/v1/tools/fetch/{search_tool_name}`) or a body field.
    A successful call returns the page title and markdown content.
    """
    return await _dispatch_search(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        data=_fetch_data(body, search_tool_name),
    )


async def tool_search(
    request: Request,
    fastapi_response: Response,
    body: ToolSearchRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    search_tool_name: str | None = None,
) -> object:
    """
    Search through a configured search tool using the DeepSeek search body.

    `count` is sent as the unified `max_results` and clamped to the provider range.
    `search_tool_name` can be the path (`/v1/tools/search/{search_tool_name}`) or a body field.
    Provider fields such as `need_content` are forwarded.
    """
    return await _dispatch_search(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        data=_tool_search_data(body, search_tool_name),
    )


_FETCH_PATHS: Final = (
    "/v1/fetch/{search_tool_name}",
    "/fetch/{search_tool_name}",
    "/v1/fetch",
    "/fetch",
    "/v1/tools/fetch/{search_tool_name}",
    "/tools/fetch/{search_tool_name}",
    "/v1/tools/fetch",
    "/tools/fetch",
)
_TOOL_SEARCH_PATHS: Final = (
    "/v1/tools/search/{search_tool_name}",
    "/tools/search/{search_tool_name}",
    "/v1/tools/search",
    "/tools/search",
)


def _mount_tool_posts(paths: tuple[str, ...], endpoint: Callable[..., object], model: type[BaseModel]) -> None:
    for path in paths:
        router.add_api_route(
            path,
            endpoint,
            methods=["POST"],
            dependencies=[Depends(user_api_key_auth)],
            response_model=None,
            responses={200: {"model": model}},  # pyright: ignore[reportArgumentType] # FastAPI types the responses dict as Any
            tags=["search"],
        )


_mount_tool_posts(_FETCH_PATHS, fetch_page, FetchResponse)
_mount_tool_posts(_TOOL_SEARCH_PATHS, tool_search, SearchResponse)


@router.get(
    "/v1/search/tools",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["search"],
)
@router.get(
    "/search/tools",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["search"],
)
async def list_search_tools(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all available search tools configured in the router.
    
    This endpoint returns the search tools that are currently loaded and available
    for use with the /v1/search endpoint.
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/search/tools" \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    
    Response:
    ```json
    {
        "object": "list",
        "data": [
            {
                "search_tool_name": "litellm-search",
                "search_provider": "perplexity",
                "description": "Perplexity search tool"
            }
        ]
    }
    ```
    """
    from litellm.proxy.proxy_server import llm_router

    try:
        search_tools_list: Final = []

        if llm_router is not None and llm_router.settings().is_set("search_tools"):
            for tool in llm_router.settings().search_tools:
                tool_info = {
                    "search_tool_name": tool.get("search_tool_name"),
                    "search_provider": tool.get("litellm_params", {}).get("search_provider"),
                }

                # Add description if available
                if "search_tool_info" in tool and tool["search_tool_info"]:
                    description = tool["search_tool_info"].get("description")
                    if description:
                        tool_info["description"] = description

                search_tools_list.append(tool_info)

        return {"object": "list", "data": search_tools_list}
    except Exception as e:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.exception("Error listing search tools: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
