"""
FALLBACK MANAGEMENT ENDPOINTS

Dedicated endpoints for managing model fallbacks separately from general config.

POST /fallback - Create or update fallbacks for a specific model
GET /fallback/{model} - Get fallbacks for a specific model
DELETE /fallback/{model} - Delete fallbacks for a specific model
"""

# pyright: reportMissingImports=false

import json
from typing import TYPE_CHECKING, Final, Literal

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_checks import get_all_fallbacks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException, status

    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException, status
    except ImportError:
        # fastapi is only required for proxy, not for SDK usage
        pass

from litellm.repositories.config_repository import ConfigRepository
from litellm.types.management_endpoints.router_settings_endpoints import (
    FallbackCreateRequest,
    FallbackDeleteResponse,
    FallbackGetResponse,
    FallbackResponse,
)

router: Final = APIRouter()

FALLBACK_SETTING_KEYS: Final[tuple[str, ...]] = (
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
)


def _as_target_list(targets: object) -> list[str]:
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, list):
        return [target for target in targets if isinstance(target, str)]
    return []


def _as_fallback_entry(entry: object) -> dict[str, list[str]]:
    if not isinstance(entry, dict):
        return {}
    return {
        primary: kept
        for primary, targets in entry.items()
        if isinstance(primary, str)
        for kept in (_as_target_list(targets),)
        if kept
    }


def _fallback_entries(value: object) -> list[dict[str, list[str]]]:
    if not isinstance(value, list):
        return []
    return [cleaned for entry in value if (cleaned := _as_fallback_entry(entry))]


def scrub_model_from_fallback_entries(
    existing: object,
    model_name: str,
) -> list[dict[str, list[str]]]:
    """Drop mappings from ``model_name`` and leftover references to it as a target."""
    return [
        remaining
        for remaining in (
            {
                primary: kept
                for primary, targets in entry.items()
                if primary != model_name
                for kept in ([target for target in targets if target != model_name],)
                if kept
            }
            for entry in _fallback_entries(existing)
        )
        if remaining
    ]


def router_lost_last_deployment_for_model(llm_router: object, model_name: str) -> bool:
    model_names: Final = getattr(llm_router, "model_names", None)
    if not isinstance(model_names, (set, frozenset, list, tuple)):
        return False
    return model_name not in model_names


async def remove_deleted_model_from_router_fallbacks(
    *,
    model_name: str,
    prisma_client: "PrismaClient",
    llm_router: "Router | None",
) -> None:
    """Persist fallback lists with ``model_name`` removed after its last deployment is deleted."""
    from litellm.proxy.proxy_server import proxy_config

    config: Final = await proxy_config.get_config()
    raw_settings: Final = config.get("router_settings", {}) if isinstance(config, dict) else {}
    router_settings: Final[dict[str, object]] = dict(raw_settings) if isinstance(raw_settings, dict) else {}
    original_by_key: Final = {key: _fallback_entries(router_settings.get(key)) for key in FALLBACK_SETTING_KEYS}
    cleaned_by_key: Final = {
        key: scrub_model_from_fallback_entries(router_settings.get(key), model_name) for key in FALLBACK_SETTING_KEYS
    }
    if cleaned_by_key == original_by_key:
        return

    updated_settings: Final = {**router_settings, **cleaned_by_key}
    router_settings_json: Final = json.dumps(updated_settings)
    await ConfigRepository(prisma_client).table.upsert(
        where={"param_name": "router_settings"},
        data={
            "create": {
                "param_name": "router_settings",
                "param_value": router_settings_json,
            },
            "update": {"param_value": router_settings_json},
        },
    )
    if llm_router is not None:
        for key in FALLBACK_SETTING_KEYS:
            setattr(llm_router, key, cleaned_by_key[key])
    verbose_proxy_logger.info("Removed deleted model %s from router fallbacks", model_name)


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
        model_names: Final = llm_router.model_names
        if data.model not in model_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Model '{data.model}' not found in router",
                    "available_models": list(model_names),
                },
            )

        # Validate that all fallback models exist in the router
        invalid_fallback_models: Final = [m for m in data.fallback_models if m not in model_names]
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
                detail={"error": f"Model '{data.model}' cannot be its own fallback"},
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
        config: Final = await proxy_config.get_config()
        router_settings: Final = config.get("router_settings", {})

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if data.fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif data.fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        # Get existing fallbacks
        existing_fallbacks: Final[list[dict[str, list[str]]]] = router_settings.get(fallback_key, [])

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
        router_settings_json: Final = json.dumps(router_settings)
        await ConfigRepository(prisma_client).table.upsert(
            where={"param_name": "router_settings"},
            data={
                "create": {
                    "param_name": "router_settings",
                    "param_value": router_settings_json,
                },
                "update": {"param_value": router_settings_json},
            },
        )

        # Update the in-memory router configuration
        setattr(llm_router, fallback_key, existing_fallbacks)

        verbose_proxy_logger.info(
            "Fallback configured: %s -> %s (type: %s)", data.model, data.fallback_models, data.fallback_type
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
        verbose_proxy_logger.error("Error creating fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to create fallback: {e}"},
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
        fallback_models: Final = get_all_fallbacks(model=model, llm_router=llm_router, fallback_type=fallback_type)

        if not fallback_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"No {fallback_type} fallbacks configured for model '{model}'"},
            )

        return FallbackGetResponse(
            model=model,
            fallback_models=fallback_models,
            fallback_type=fallback_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("Error getting fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to get fallback: {e}"},
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
        config: Final = await proxy_config.get_config()
        router_settings: Final = config.get("router_settings", {})

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        # Get existing fallbacks
        existing_fallbacks: Final[list[dict[str, list[str]]]] = router_settings.get(fallback_key, [])

        # Find and remove the fallback configuration
        fallback_found = False
        updated_fallbacks: Final = []
        for fallback_dict in existing_fallbacks:
            if model not in fallback_dict:
                updated_fallbacks.append(fallback_dict)
            else:
                fallback_found = True

        if not fallback_found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"No {fallback_type} fallbacks configured for model '{model}'"},
            )

        # Update router settings
        router_settings[fallback_key] = updated_fallbacks

        # Save to database - convert router_settings to JSON string
        router_settings_json: Final = json.dumps(router_settings)
        await ConfigRepository(prisma_client).table.upsert(
            where={"param_name": "router_settings"},
            data={
                "create": {
                    "param_name": "router_settings",
                    "param_value": router_settings_json,
                },
                "update": {"param_value": router_settings_json},
            },
        )

        # Update the in-memory router configuration
        setattr(llm_router, fallback_key, updated_fallbacks)

        verbose_proxy_logger.info("Fallback deleted: %s (type: %s)", model, fallback_type)

        return FallbackDeleteResponse(
            model=model,
            fallback_type=fallback_type,
            message="Fallback configuration deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("Error deleting fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to delete fallback: {e}"},
        )
