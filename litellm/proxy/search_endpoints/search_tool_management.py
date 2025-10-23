"""
CRUD ENDPOINTS FOR SEARCH TOOLS
"""
from datetime import datetime
from typing import Any, Dict, List, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.search_endpoints.search_tool_registry import (
    IN_MEMORY_SEARCH_TOOL_HANDLER,
    SearchToolRegistry,
)
from litellm.types.search import (
    AvailableSearchProvider,
    ListSearchToolsResponse,
    SearchTool,
    SearchToolInfoResponse,
)
from litellm.types.utils import SearchProviders

#### SEARCH TOOLS ENDPOINTS ####

router = APIRouter()
SEARCH_TOOL_REGISTRY = SearchToolRegistry()


def _convert_datetime_to_str(value: Union[datetime, str, None]) -> Union[str, None]:
    """
    Convert datetime object to ISO format string.
    
    Args:
        value: datetime object, string, or None
        
    Returns:
        ISO format string or original value if already string or None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@router.get(
    "/search_tools/list",
    tags=["Search Tools"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListSearchToolsResponse,
)
async def list_search_tools():
    """
    List all search tools that are available in the database.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/search_tools/list" -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "search_tools": [
            {
                "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
                "search_tool_name": "litellm-search",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "sk-***",
                    "api_base": "https://api.perplexity.ai"
                },
                "search_tool_info": {
                    "description": "Perplexity search tool"
                },
                "created_at": "2023-11-09T12:34:56.789Z",
                "updated_at": "2023-11-09T12:34:56.789Z"
            }
        ]
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        search_tools = await SEARCH_TOOL_REGISTRY.get_all_search_tools_from_db(
            prisma_client=prisma_client
        )

        search_tool_configs: List[SearchToolInfoResponse] = []
        for search_tool in search_tools:
            search_tool_configs.append(
                SearchToolInfoResponse(
                    search_tool_id=search_tool.get("search_tool_id"),
                    search_tool_name=search_tool.get("search_tool_name", ""),
                    litellm_params=dict(search_tool.get("litellm_params", {})),
                    search_tool_info=search_tool.get("search_tool_info"),
                    created_at=_convert_datetime_to_str(search_tool.get("created_at")),
                    updated_at=_convert_datetime_to_str(search_tool.get("updated_at")),
                )
            )

        return ListSearchToolsResponse(search_tools=search_tool_configs)
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting search tools from db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CreateSearchToolRequest(BaseModel):
    search_tool: SearchTool


@router.post(
    "/search_tools",
    tags=["Search Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_search_tool(request: CreateSearchToolRequest):
    """
    Create a new search tool.

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/search_tools" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "search_tool": {
                "search_tool_name": "litellm-search",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "sk-..."
                },
                "search_tool_info": {
                    "description": "Perplexity search tool"
                }
            }
        }'
    ```

    Example Response:
    ```json
    {
        "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
        "search_tool_name": "litellm-search",
        "litellm_params": {
            "search_provider": "perplexity",
            "api_key": "sk-..."
        },
        "search_tool_info": {
            "description": "Perplexity search tool"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T12:34:56.789Z"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await SEARCH_TOOL_REGISTRY.add_search_tool_to_db(
            search_tool=request.search_tool, prisma_client=prisma_client
        )

        # Add to in-memory cache
        try:
            IN_MEMORY_SEARCH_TOOL_HANDLER.add_search_tool(search_tool=cast(SearchTool, result))
            verbose_proxy_logger.info(
                f"Successfully added search tool '{result.get('search_tool_name')}' to in-memory cache"
            )
        except Exception as cache_error:
            verbose_proxy_logger.warning(
                f"Failed to add search tool to in-memory cache: {cache_error}"
            )

        return result
    except Exception as e:
        verbose_proxy_logger.exception(f"Error adding search tool to db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdateSearchToolRequest(BaseModel):
    search_tool: SearchTool


@router.put(
    "/search_tools/{search_tool_id}",
    tags=["Search Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_search_tool(search_tool_id: str, request: UpdateSearchToolRequest):
    """
    Update an existing search tool.

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/search_tools/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "search_tool": {
                "search_tool_name": "updated-search",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "sk-new-key"
                },
                "search_tool_info": {
                    "description": "Updated search tool"
                }
            }
        }'
    ```

    Example Response:
    ```json
    {
        "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
        "search_tool_name": "updated-search",
        "litellm_params": {
            "search_provider": "perplexity",
            "api_key": "sk-new-key"
        },
        "search_tool_info": {
            "description": "Updated search tool"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T13:45:12.345Z"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Check if search tool exists
        existing_tool = await SEARCH_TOOL_REGISTRY.get_search_tool_by_id_from_db(
            search_tool_id=search_tool_id, prisma_client=prisma_client
        )

        if existing_tool is None:
            raise HTTPException(
                status_code=404,
                detail=f"Search tool with ID {search_tool_id} not found",
            )

        result = await SEARCH_TOOL_REGISTRY.update_search_tool_in_db(
            search_tool_id=search_tool_id,
            search_tool=request.search_tool,
            prisma_client=prisma_client,
        )

        # Update in-memory cache
        try:
            IN_MEMORY_SEARCH_TOOL_HANDLER.update_search_tool(
                search_tool_id=search_tool_id, search_tool=cast(SearchTool, result)
            )
            verbose_proxy_logger.info(
                f"Successfully updated search tool '{result.get('search_tool_name')}' in in-memory cache"
            )
        except Exception as cache_error:
            verbose_proxy_logger.warning(
                f"Failed to update search tool in in-memory cache: {cache_error}"
            )

        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating search tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/search_tools/{search_tool_id}",
    tags=["Search Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_search_tool(search_tool_id: str):
    """
    Delete a search tool.

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/search_tools/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "message": "Search tool 123e4567-e89b-12d3-a456-426614174000 deleted successfully",
        "search_tool_name": "litellm-search"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Check if search tool exists
        existing_tool = await SEARCH_TOOL_REGISTRY.get_search_tool_by_id_from_db(
            search_tool_id=search_tool_id, prisma_client=prisma_client
        )

        if existing_tool is None:
            raise HTTPException(
                status_code=404,
                detail=f"Search tool with ID {search_tool_id} not found",
            )

        result = await SEARCH_TOOL_REGISTRY.delete_search_tool_from_db(
            search_tool_id=search_tool_id, prisma_client=prisma_client
        )

        # Delete from in-memory cache
        try:
            IN_MEMORY_SEARCH_TOOL_HANDLER.delete_search_tool(
                search_tool_id=search_tool_id
            )
            verbose_proxy_logger.info(
                f"Successfully removed search tool from in-memory cache"
            )
        except Exception as cache_error:
            verbose_proxy_logger.warning(
                f"Failed to remove search tool from in-memory cache: {cache_error}"
            )

        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting search tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/search_tools/{search_tool_id}",
    tags=["Search Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_search_tool_info(search_tool_id: str):
    """
    Get detailed information about a specific search tool by ID.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/search_tools/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "search_tool_id": "123e4567-e89b-12d3-a456-426614174000",
        "search_tool_name": "litellm-search",
        "litellm_params": {
            "search_provider": "perplexity",
            "api_key": "sk-***"
        },
        "search_tool_info": {
            "description": "Perplexity search tool"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T12:34:56.789Z"
    }
    ```
    """
    from litellm.litellm_core_utils.litellm_logging import _get_masked_values
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await SEARCH_TOOL_REGISTRY.get_search_tool_by_id_from_db(
            search_tool_id=search_tool_id, prisma_client=prisma_client
        )

        if result is None:
            # Try in-memory cache
            result = IN_MEMORY_SEARCH_TOOL_HANDLER.get_search_tool_by_id(
                search_tool_id=search_tool_id
            )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Search tool with ID {search_tool_id} not found",
            )

        # Mask sensitive data
        litellm_params_dict = dict(result.get("litellm_params", {}))
        masked_litellm_params_dict = _get_masked_values(
            litellm_params_dict,
            unmasked_length=4,
            number_of_asterisks=4,
        )

        return SearchToolInfoResponse(
            search_tool_id=result.get("search_tool_id"),
            search_tool_name=result.get("search_tool_name", ""),
            litellm_params=masked_litellm_params_dict,
            search_tool_info=result.get("search_tool_info"),
            created_at=_convert_datetime_to_str(result.get("created_at")),
            updated_at=_convert_datetime_to_str(result.get("updated_at")),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting search tool info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/search_tools/ui/available_providers",
    tags=["Search Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_available_search_providers():
    """
    Get the list of available search providers with their configuration fields.
    
    Auto-discovers search providers and their UI-friendly names from transformation configs.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/search_tools/ui/available_providers" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "providers": [
            {
                "provider_name": "perplexity",
                "ui_friendly_name": "Perplexity"
            },
            {
                "provider_name": "tavily",
                "ui_friendly_name": "Tavily"
            }
        ]
    }
    ```
    """
    try:
        from litellm.utils import ProviderConfigManager
        
        available_providers = []
        
        # Auto-discover providers from SearchProviders enum
        for provider in SearchProviders:
            try:
                # Get the config class for this provider
                config = ProviderConfigManager.get_provider_search_config(provider=provider)
                
                if config is not None:
                    # Get the UI-friendly name from the config class
                    ui_name = config.ui_friendly_name()
                    
                    available_providers.append({
                        "provider_name": provider.value,
                        "ui_friendly_name": ui_name,
                    })
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Could not get config for search provider {provider.value}: {e}"
                )
                continue
        
        return {"providers": available_providers}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting available search providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

