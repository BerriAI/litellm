"""
FALLBACK MANAGEMENT ENDPOINTS

Dedicated endpoints for managing model fallbacks separately from general config.

POST /fallback - Create or update fallbacks for a specific model
GET /fallback/{model} - Get fallbacks for a specific model
DELETE /fallback/{model} - Delete fallbacks for a specific model
"""
# pyright: reportMissingImports=false

import json
from typing import TYPE_CHECKING, Dict, List, Literal

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_checks import get_all_fallbacks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException, status
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException, status
    except ImportError:
        # fastapi is only required for proxy, not for SDK usage
        pass

from litellm.types.management_endpoints.router_settings_endpoints import (
    FallbackCreateRequest,
    FallbackDeleteResponse,
    FallbackGetResponse,
    FallbackResponse,
)

router = APIRouter()


@router.post(
    "/fallback",
    tags=["Fallback Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FallbackResponse,
    status_code=status.HTTP_200_OK,
)
async def create_fallback(
    data: FallbackCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create or update fallbacks for a specific model.

    This endpoint allows you to configure fallback models separately from the general config.
    Fallbacks are triggered when a model call fails after retries.

    **Example Request:**
    ```json
    {
        "model": "gpt-3.5-turbo",
        "fallback_models": ["gpt-4", "claude-3-haiku"],
        "fallback_type": "general"
    }
    ```

    **Fallback Types:**
    - `general`: Standard fallbacks for any error (default)
    - `context_window`: Fallbacks specifically for context window exceeded errors
    - `content_policy`: Fallbacks specifically for content policy violations
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

    try:
        # Validate that we have a router
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        # Validate that the model exists in the router
        model_names = llm_router.model_names
        if data.model not in model_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Model '{data.model}' not found in router",
                    "available_models": list(model_names),
                },
            )

        # Validate that all fallback models exist in the router
        invalid_fallback_models = [
            m for m in data.fallback_models if m not in model_names
        ]
        if invalid_fallback_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"Invalid fallback models: {invalid_fallback_models}",
                    "available_models": list(model_names),
                },
            )

        # Check if fallback model is the same as the primary model
        if data.model in data.fallback_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"Model '{data.model}' cannot be its own fallback"
                },
            )

        # Check if we need to store in DB
        if store_model_in_db is not True or prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Database storage not enabled. Set 'STORE_MODEL_IN_DB=True' in your environment to use this feature."
                },
            )

        # Load existing config
        config = await proxy_config.get_config()
        router_settings = config.get("router_settings", {})

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if data.fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif data.fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        # Get existing fallbacks
        existing_fallbacks: List[Dict[str, List[str]]] = router_settings.get(
            fallback_key, []
        )

        # Update or add the fallback configuration
        fallback_updated = False
        for i, fallback_dict in enumerate(existing_fallbacks):
            if data.model in fallback_dict:
                # Update existing fallback
                existing_fallbacks[i] = {data.model: data.fallback_models}
                fallback_updated = True
                break

        if not fallback_updated:
            # Add new fallback
            existing_fallbacks.append({data.model: data.fallback_models})

        # Update router settings
        router_settings[fallback_key] = existing_fallbacks

        # Save to database - convert router_settings to JSON string
        router_settings_json = json.dumps(router_settings)
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "router_settings"},
            data={
                "create": {
                    "param_name": "router_settings",
                    "param_value": router_settings_json,
                },
                "update": {
                    "param_value": router_settings_json
                },
            },
        )

        # Update the in-memory router configuration
        setattr(llm_router, fallback_key, existing_fallbacks)

        verbose_proxy_logger.info(
            f"Fallback configured: {data.model} -> {data.fallback_models} (type: {data.fallback_type})"
        )

        return FallbackResponse(
            model=data.model,
            fallback_models=data.fallback_models,
            fallback_type=data.fallback_type,
            message=f"Fallback configuration {'updated' if fallback_updated else 'created'} successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error creating fallback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to create fallback: {str(e)}"},
        )


@router.get(
    "/fallback/{model}",
    tags=["Fallback Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FallbackGetResponse,
)
async def get_fallback(
    model: str,
    fallback_type: Literal["general", "context_window", "content_policy"] = "general",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get fallback configuration for a specific model.

    **Parameters:**
    - `model`: The model name to get fallbacks for
    - `fallback_type`: Type of fallback to retrieve (query parameter)

    **Example:**
    ```
    GET /fallback/gpt-3.5-turbo?fallback_type=general
    ```
    """
    from litellm.proxy.proxy_server import llm_router

    try:
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        # Get fallbacks using the existing utility function
        fallback_models = get_all_fallbacks(
            model=model, llm_router=llm_router, fallback_type=fallback_type
        )

        if not fallback_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"No {fallback_type} fallbacks configured for model '{model}'"
                },
            )

        return FallbackGetResponse(
            model=model,
            fallback_models=fallback_models,
            fallback_type=fallback_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting fallback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to get fallback: {str(e)}"},
        )


@router.delete(
    "/fallback/{model}",
    tags=["Fallback Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FallbackDeleteResponse,
)
async def delete_fallback(
    model: str,
    fallback_type: Literal["general", "context_window", "content_policy"] = "general",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete fallback configuration for a specific model.

    **Parameters:**
    - `model`: The model name to delete fallbacks for
    - `fallback_type`: Type of fallback to delete (query parameter)

    **Example:**
    ```
    DELETE /fallback/gpt-3.5-turbo?fallback_type=general
    ```
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

    try:
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        if store_model_in_db is not True or prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Database storage not enabled. Set 'STORE_MODEL_IN_DB=True' in your environment to use this feature."
                },
            )

        # Load existing config
        config = await proxy_config.get_config()
        router_settings = config.get("router_settings", {})

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        # Get existing fallbacks
        existing_fallbacks: List[Dict[str, List[str]]] = router_settings.get(
            fallback_key, []
        )

        # Find and remove the fallback configuration
        fallback_found = False
        updated_fallbacks = []
        for fallback_dict in existing_fallbacks:
            if model not in fallback_dict:
                updated_fallbacks.append(fallback_dict)
            else:
                fallback_found = True

        if not fallback_found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"No {fallback_type} fallbacks configured for model '{model}'"
                },
            )

        # Update router settings
        router_settings[fallback_key] = updated_fallbacks

        # Save to database - convert router_settings to JSON string
        router_settings_json = json.dumps(router_settings)
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "router_settings"},
            data={
                "create": {
                    "param_name": "router_settings",
                    "param_value": router_settings_json,
                },
                "update": {
                    "param_value": router_settings_json
                },
            },
        )

        # Update the in-memory router configuration
        setattr(llm_router, fallback_key, updated_fallbacks)

        verbose_proxy_logger.info(
            f"Fallback deleted: {model} (type: {fallback_type})"
        )

        return FallbackDeleteResponse(
            model=model,
            fallback_type=fallback_type,
            message="Fallback configuration deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error deleting fallback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to delete fallback: {str(e)}"},
        )
