#### Search Endpoints #####

import orjson
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


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
    search_tool_name: Optional[str] = None,
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
        -H "Authorization: Bearer sk-1234" \
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
        -H "Authorization: Bearer sk-1234" \
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

    # Read request body
    body = await request.body()
    data = orjson.loads(body)
    
    # If search_tool_name is provided in URL path, use it (takes precedence over body)
    if search_tool_name is not None:
        data["search_tool_name"] = search_tool_name

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
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
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )

