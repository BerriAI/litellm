"""
CRUD ENDPOINTS FOR FETCH TOOLS
"""

from datetime import datetime
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.fetch_endpoints.fetch_tool_registry import FetchToolRegistry
from litellm.types.fetch import (
    FetchProviders,
    FetchTool,
    FetchToolInfoResponse,
    ListFetchToolsResponse,
)

#### FETCH TOOLS ENDPOINTS ####

router = APIRouter()
FETCH_TOOL_REGISTRY = FetchToolRegistry()


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
    "/fetch_tools/list",
    tags=["Fetch Tools"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListFetchToolsResponse,
)
async def list_fetch_tools():
    """
    List all fetch tools that are available in the database and config file.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/fetch_tools/list" -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "fetch_tools": [
            {
                "fetch_tool_id": "123e4567-e89b-12d3-a456-426614174000",
                "fetch_tool_name": "litellm-fetch",
                "litellm_params": {
                    "fetch_provider": "firecrawl",
                    "api_key": "fc-***",
                    "api_base": "https://api.firecrawl.dev"
                },
                "fetch_tool_info": {
                    "description": "Firecrawl fetch tool"
                },
                "created_at": "2023-11-09T12:34:56.789Z",
                "updated_at": "2023-11-09T12:34:56.789Z",
                "is_from_config": false
            },
            {
                "fetch_tool_name": "config-fetch-tool",
                "litellm_params": {
                    "fetch_provider": "firecrawl",
                    "api_key": "fc-***"
                },
                "is_from_config": true
            }
        ]
    }
    ```
    """
    from litellm.litellm_core_utils.litellm_logging import _get_masked_values
    from litellm.proxy.proxy_server import prisma_client, proxy_config

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        fetch_tools_from_db = await FETCH_TOOL_REGISTRY.get_all_fetch_tools_from_db(
            prisma_client=prisma_client
        )

        # Mask API keys
        for tool in fetch_tools_from_db:
            litellm_params = tool.get("litellm_params", {})
            if isinstance(litellm_params, dict):
                for key, value in litellm_params.items():
                    if isinstance(value, str):
                        litellm_params[key] = _get_masked_values(key, value)
            tool["litellm_params"] = litellm_params

        db_tool_names = {tool.get("fetch_tool_name") for tool in fetch_tools_from_db}

        # Get fetch tools from config
        config_fetch_tools = []
        config = getattr(proxy_config, "config", None)
        if config:
            parsed_tools = proxy_config.parse_fetch_tools(config)
            if parsed_tools:
                config_fetch_tools = parsed_tools

        for fetch_tool in config_fetch_tools:
            tool_name = fetch_tool.get("fetch_tool_name")
            if tool_name not in db_tool_names:
                # Add config tool to response
                fetch_tool_info = fetch_tool.get("fetch_tool_info", {})
                litellm_params = fetch_tool.get("litellm_params", {})

                # Mask API keys
                if isinstance(litellm_params, dict):
                    for key, value in litellm_params.items():
                        if isinstance(value, str):
                            litellm_params[key] = _get_masked_values(key, value)

                fetch_tools_from_db.append(
                    FetchToolInfoResponse(
                        fetch_tool_name=tool_name,
                        litellm_params=litellm_params,
                        fetch_tool_info=fetch_tool_info if fetch_tool_info else None,
                        is_from_config=True,
                    )
                )

        return ListFetchToolsResponse(fetch_tools=fetch_tools_from_db)
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing fetch tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/fetch_tools",
    tags=["Fetch Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_fetch_tool(request: Dict[str, Any]):
    """
    Create a new fetch tool in the database.

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/fetch_tools" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "fetch_tool": {
                "fetch_tool_name": "litellm-fetch",
                "litellm_params": {
                    "fetch_provider": "firecrawl",
                    "api_key": "fc-1234",
                    "api_base": "https://api.firecrawl.dev"
                },
                "fetch_tool_info": {
                    "description": "Firecrawl fetch tool"
                }
            }
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        fetch_tool_data = request.get("fetch_tool", {})
        fetch_tool = FetchTool(**fetch_tool_data)

        created_tool = await FETCH_TOOL_REGISTRY.add_fetch_tool_to_db(
            fetch_tool=fetch_tool,
            prisma_client=prisma_client,
        )

        return created_tool
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating fetch tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/fetch_tools/{fetch_tool_id}",
    tags=["Fetch Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_fetch_tool(fetch_tool_id: str, request: Dict[str, Any]):
    """
    Update an existing fetch tool in the database.

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/fetch_tools/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "fetch_tool": {
                "fetch_tool_name": "litellm-fetch",
                "litellm_params": {
                    "fetch_provider": "firecrawl",
                    "api_key": "fc-5678",
                    "api_base": "https://api.firecrawl.dev"
                },
                "fetch_tool_info": {
                    "description": "Updated Firecrawl fetch tool"
                }
            }
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        fetch_tool_data = request.get("fetch_tool", {})
        fetch_tool = FetchTool(**fetch_tool_data)

        updated_tool = await FETCH_TOOL_REGISTRY.update_fetch_tool_in_db(
            fetch_tool_id=fetch_tool_id,
            fetch_tool=fetch_tool,
            prisma_client=prisma_client,
        )

        return updated_tool
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating fetch tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/fetch_tools/{fetch_tool_id}",
    tags=["Fetch Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_fetch_tool(fetch_tool_id: str):
    """
    Delete a fetch tool from the database.

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/fetch_tools/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await FETCH_TOOL_REGISTRY.delete_fetch_tool_from_db(
            fetch_tool_id=fetch_tool_id,
            prisma_client=prisma_client,
        )

        return result
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting fetch tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/fetch_tools/{fetch_tool_id}",
    tags=["Fetch Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_fetch_tool(fetch_tool_id: str):
    """
    Get a specific fetch tool by ID.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/fetch_tools/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        fetch_tool = await FETCH_TOOL_REGISTRY.get_fetch_tool_from_db(
            fetch_tool_id=fetch_tool_id,
            prisma_client=prisma_client,
        )

        if fetch_tool is None:
            raise HTTPException(status_code=404, detail="Fetch tool not found")

        return fetch_tool
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting fetch tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/fetch_tools/ui/available_providers",
    tags=["Fetch Tools"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_available_fetch_providers():
    """
    List all available fetch providers for the UI.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/fetch_tools/ui/available_providers" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "providers": [
            {"provider_name": "firecrawl", "ui_friendly_name": "Firecrawl"}
        ]
    }
    ```
    """
    try:
        providers = [
            {
                "provider_name": provider.value,
                "ui_friendly_name": provider.value.capitalize(),
            }
            for provider in FetchProviders
        ]
        return {"providers": providers}
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error listing available fetch providers: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
