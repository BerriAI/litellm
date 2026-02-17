"""
CRUD ENDPOINTS FOR GUARDRAILS
"""

import inspect
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.guardrail_registry import GuardrailRegistry
from litellm.types.guardrails import (PII_ENTITY_CATEGORIES_MAP,
                                      ApplyGuardrailRequest,
                                      ApplyGuardrailResponse,
                                      BaseLitellmParams,
                                      BedrockGuardrailConfigModel, Guardrail,
                                      GuardrailEventHooks,
                                      GuardrailInfoResponse,
                                      GuardrailUIAddGuardrailSettings,
                                      LakeraV2GuardrailConfigModel,
                                      ListGuardrailsResponse, LitellmParams,
                                      PatchGuardrailRequest, PiiAction,
                                      PiiEntityType,
                                      PresidioPresidioConfigModelUserInterface,
                                      SupportedGuardrailIntegrations,
                                      ToolPermissionGuardrailConfigModel)

#### GUARDRAILS ENDPOINTS ####

router = APIRouter()
GUARDRAIL_REGISTRY = GuardrailRegistry()


def _get_guardrails_list_response(
    guardrails_config: List[Dict],
) -> ListGuardrailsResponse:
    """
    Helper function to get the guardrails list response
    """
    guardrail_configs: List[GuardrailInfoResponse] = []
    for guardrail in guardrails_config:
        guardrail_configs.append(
            GuardrailInfoResponse(
                guardrail_name=guardrail.get("guardrail_name"),
                litellm_params=guardrail.get("litellm_params"),
                guardrail_info=guardrail.get("guardrail_info"),
            )
        )
    return ListGuardrailsResponse(guardrails=guardrail_configs)


@router.get(
    "/guardrails/list",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListGuardrailsResponse,
)
async def list_guardrails():
    """
    List the guardrails that are available on the proxy server

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/guardrails/list" -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "guardrails": [
            {
            "guardrail_name": "bedrock-pre-guard",
            "guardrail_info": {
                "params": [
                {
                    "name": "toxicity_score",
                    "type": "float",
                    "description": "Score between 0-1 indicating content toxicity level"
                },
                {
                    "name": "pii_detection",
                    "type": "boolean"
                }
                ]
            }
            }
        ]
    }
    ```
    """
    from litellm.proxy.proxy_server import proxy_config

    config = proxy_config.config

    _guardrails_config = cast(Optional[list[dict]], config.get("guardrails"))

    if _guardrails_config is None:
        return _get_guardrails_list_response([])

    return _get_guardrails_list_response(_guardrails_config)


@router.get(
    "/v2/guardrails/list",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListGuardrailsResponse,
)
async def list_guardrails_v2():
    """
    List the guardrails that are available in the database using GuardrailRegistry

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/v2/guardrails/list" -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "guardrails": [
            {
                "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
                "guardrail_name": "my-bedrock-guard",
                "litellm_params": {
                    "guardrail": "bedrock",
                    "mode": "pre_call",
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "DRAFT",
                    "default_on": true
                },
                "guardrail_info": {
                    "description": "Bedrock content moderation guardrail"
                }
            }
        ]
    }
    ```
    """
    from litellm.litellm_core_utils.litellm_logging import _get_masked_values
    from litellm.proxy.guardrails.guardrail_registry import \
        IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        guardrails = await GUARDRAIL_REGISTRY.get_all_guardrails_from_db(
            prisma_client=prisma_client
        )

        guardrail_configs: List[GuardrailInfoResponse] = []
        seen_guardrail_ids = set()
        for guardrail in guardrails:
            litellm_params: Optional[Union[LitellmParams, dict]] = guardrail.get(
                "litellm_params"
            )
            litellm_params_dict = (
                litellm_params.model_dump(exclude_none=True)
                if isinstance(litellm_params, LitellmParams)
                else litellm_params
            ) or {}
            masked_litellm_params_dict = _get_masked_values(
                litellm_params_dict,
                unmasked_length=4,
                number_of_asterisks=4,
            )
            masked_litellm_params = (
                BaseLitellmParams(**masked_litellm_params_dict)
                if masked_litellm_params_dict
                else None
            )
            guardrail_configs.append(
                GuardrailInfoResponse(
                    guardrail_id=guardrail.get("guardrail_id"),
                    guardrail_name=guardrail.get("guardrail_name"),
                    litellm_params=masked_litellm_params,
                    guardrail_info=guardrail.get("guardrail_info"),
                    created_at=guardrail.get("created_at"),
                    updated_at=guardrail.get("updated_at"),
                    guardrail_definition_location="db",
                )
            )
            seen_guardrail_ids.add(guardrail.get("guardrail_id"))

        # get guardrails initialized on litellm config.yaml
        in_memory_guardrails = IN_MEMORY_GUARDRAIL_HANDLER.list_in_memory_guardrails()
        for guardrail in in_memory_guardrails:
            # only add guardrails that are not in DB guardrail list already
            if guardrail.get("guardrail_id") not in seen_guardrail_ids:
                in_memory_litellm_params_raw = guardrail.get("litellm_params")
                in_memory_litellm_params_dict = (
                    in_memory_litellm_params_raw.model_dump(exclude_none=True)
                    if isinstance(in_memory_litellm_params_raw, LitellmParams)
                    else in_memory_litellm_params_raw
                ) or {}
                masked_in_memory_litellm_params = _get_masked_values(
                    in_memory_litellm_params_dict,
                    unmasked_length=4,
                    number_of_asterisks=4,
                )
                masked_in_memory_litellm_params_typed = (
                    BaseLitellmParams(**masked_in_memory_litellm_params)
                    if masked_in_memory_litellm_params
                    else None
                )
                guardrail_configs.append(
                    GuardrailInfoResponse(
                        guardrail_id=guardrail.get("guardrail_id"),
                        guardrail_name=guardrail.get("guardrail_name"),
                        litellm_params=masked_in_memory_litellm_params_typed,
                        guardrail_info=dict(guardrail.get("guardrail_info") or {}),
                        guardrail_definition_location="config",
                    )
                )
                seen_guardrail_ids.add(guardrail.get("guardrail_id"))

        return ListGuardrailsResponse(guardrails=guardrail_configs)
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting guardrails from db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CreateGuardrailRequest(BaseModel):
    guardrail: Guardrail


@router.post(
    "/guardrails",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_guardrail(request: CreateGuardrailRequest):
    """
    Create a new guardrail

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/guardrails" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "guardrail": {
                "guardrail_name": "my-bedrock-guard",
                "litellm_params": {
                    "guardrail": "bedrock",
                    "mode": "pre_call",
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "DRAFT",
                    "default_on": true
                },
                "guardrail_info": {
                    "description": "Bedrock content moderation guardrail"
                }
            }
        }'
    ```

    Example Response:
    ```json
    {
        "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
        "guardrail_name": "my-bedrock-guard",
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "pre_call",
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "default_on": true
        },
        "guardrail_info": {
            "description": "Bedrock content moderation guardrail"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T12:34:56.789Z"
    }
    ```
    """
    from litellm.proxy.guardrails.guardrail_registry import \
        IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await GUARDRAIL_REGISTRY.add_guardrail_to_db(
            guardrail=request.guardrail, prisma_client=prisma_client
        )

        guardrail_name = result.get("guardrail_name", "Unknown")
        guardrail_id = result.get("guardrail_id", "Unknown")

        try:
            IN_MEMORY_GUARDRAIL_HANDLER.initialize_guardrail(
                guardrail=cast(Guardrail, result)
            )
            verbose_proxy_logger.info(
                f"Immediate sync: Successfully initialized guardrail '{guardrail_name}' (ID: {guardrail_id})"
            )
        except Exception as init_error:
            verbose_proxy_logger.warning(
                f"Immediate sync: Failed to initialize guardrail '{guardrail_name}' (ID: {guardrail_id}) in memory: {init_error}"
            )

        return result
    except Exception as e:
        verbose_proxy_logger.exception(f"Error adding guardrail to db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdateGuardrailRequest(BaseModel):
    guardrail: Guardrail


@router.put(
    "/guardrails/{guardrail_id}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_guardrail(guardrail_id: str, request: UpdateGuardrailRequest):
    """
    Update an existing guardrail

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "guardrail": {
                "guardrail_name": "updated-bedrock-guard",
                "litellm_params": {
                    "guardrail": "bedrock",
                    "mode": "pre_call",
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "1.0",
                    "default_on": true
                },
                "guardrail_info": {
                    "description": "Updated Bedrock content moderation guardrail"
                }
            }
        }'
    ```

    Example Response:
    ```json
    {
        "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
        "guardrail_name": "updated-bedrock-guard",
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "pre_call",
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "1.0",
            "default_on": true
        },
        "guardrail_info": {
            "description": "Updated Bedrock content moderation guardrail"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T13:45:12.345Z"
    }
    ```
    """
    from litellm.proxy.guardrails.guardrail_registry import \
        IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Check if guardrail exists
        existing_guardrail = await GUARDRAIL_REGISTRY.get_guardrail_by_id_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )

        if existing_guardrail is None:
            raise HTTPException(
                status_code=404, detail=f"Guardrail with ID {guardrail_id} not found"
            )

        result = await GUARDRAIL_REGISTRY.update_guardrail_in_db(
            guardrail_id=guardrail_id,
            guardrail=request.guardrail,
            prisma_client=prisma_client,
        )

        guardrail_name = result.get("guardrail_name", "Unknown")

        try:
            IN_MEMORY_GUARDRAIL_HANDLER.update_in_memory_guardrail(
                guardrail_id=guardrail_id, guardrail=cast(Guardrail, result)
            )
            verbose_proxy_logger.info(
                f"Immediate sync: Successfully updated guardrail '{guardrail_name}' (ID: {guardrail_id})"
            )
        except Exception as update_error:
            verbose_proxy_logger.warning(
                f"Immediate sync: Failed to update '{guardrail_name}' (ID: {guardrail_id}) in memory: {update_error}"
            )

        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/guardrails/{guardrail_id}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_guardrail(guardrail_id: str):
    """
    Delete a guardrail

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "message": "Guardrail 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
    }
    ```
    """
    from litellm.proxy.guardrails.guardrail_registry import \
        IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Check if guardrail exists
        existing_guardrail = await GUARDRAIL_REGISTRY.get_guardrail_by_id_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )

        if existing_guardrail is None:
            raise HTTPException(
                status_code=404, detail=f"Guardrail with ID {guardrail_id} not found"
            )

        result = await GUARDRAIL_REGISTRY.delete_guardrail_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )

        guardrail_name = result.get("guardrail_name", "Unknown")

        try:
            IN_MEMORY_GUARDRAIL_HANDLER.delete_in_memory_guardrail(
                guardrail_id=guardrail_id,
            )
            verbose_proxy_logger.info(
                f"Immediate sync: Successfully removed guardrail '{guardrail_name}' (ID: {guardrail_id}) from memory"
            )
        except Exception as delete_error:
            verbose_proxy_logger.warning(
                f"Immediate sync: Failed to remove guardrail '{guardrail_name}' (ID: {guardrail_id}) from memory: {delete_error}"
            )

        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/guardrails/{guardrail_id}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def patch_guardrail(guardrail_id: str, request: PatchGuardrailRequest):
    """
    Partially update an existing guardrail

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    This endpoint allows updating specific fields of a guardrail without sending the entire object.
    Only the following fields can be updated:
    - guardrail_name: The name of the guardrail
    - default_on: Whether the guardrail is enabled by default
    - guardrail_info: Additional information about the guardrail

    Example Request:
    ```bash
    curl -X PATCH "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "guardrail_name": "updated-name",
            "default_on": true,
            "guardrail_info": {
                "description": "Updated description"
            }
        }'
    ```

    Example Response:
    ```json
    {
        "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
        "guardrail_name": "updated-name",
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "pre_call",
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "default_on": true
        },
        "guardrail_info": {
            "description": "Updated description"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T14:22:33.456Z"
    }
    ```
    """
    from litellm.proxy.guardrails.guardrail_registry import \
        IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Check if guardrail exists and get current data
        existing_guardrail = await GUARDRAIL_REGISTRY.get_guardrail_by_id_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )

        if existing_guardrail is None:
            raise HTTPException(
                status_code=404, detail=f"Guardrail with ID {guardrail_id} not found"
            )

        # Create updated guardrail object
        guardrail_name = (
            request.guardrail_name
            if request.guardrail_name is not None
            else existing_guardrail.get("guardrail_name")
        )

        # Update litellm_params if default_on is provided or pii_entities_config is provided
        litellm_params = LitellmParams(
            **dict(existing_guardrail.get("litellm_params", {}))
        )
        if request.litellm_params is not None:
            requested_litellm_params = request.litellm_params.model_dump(
                exclude_unset=True
            )
            litellm_params_dict = litellm_params.model_dump(exclude_unset=True)
            litellm_params_dict.update(requested_litellm_params)
            litellm_params = LitellmParams(**litellm_params_dict)

        # Update guardrail_info if provided
        guardrail_info = (
            request.guardrail_info
            if request.guardrail_info is not None
            else existing_guardrail.get("guardrail_info", {})
        )

        # Create the guardrail object
        guardrail = Guardrail(
            guardrail_id=guardrail_id,
            guardrail_name=guardrail_name or "",
            litellm_params=litellm_params,
            guardrail_info=guardrail_info,
        )
        result = await GUARDRAIL_REGISTRY.update_guardrail_in_db(
            guardrail_id=guardrail_id,
            guardrail=guardrail,
            prisma_client=prisma_client,
        )

        guardrail_name = result.get("guardrail_name", "Unknown")

        try:
            IN_MEMORY_GUARDRAIL_HANDLER.sync_guardrail_from_db(
                guardrail=guardrail,
            )
            verbose_proxy_logger.info(
                f"Immediate sync: Successfully updated guardrail '{guardrail_name}' (ID: {guardrail_id})"
            )
        except Exception as update_error:
            verbose_proxy_logger.warning(
                f"Immediate sync: Failed to update '{guardrail_name}' (ID: {guardrail_id}) in memory: {update_error}"
            )

        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating guardrail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/guardrails/{guardrail_id}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/guardrails/{guardrail_id}/info",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_guardrail_info(guardrail_id: str):
    """
    Get detailed information about a specific guardrail by ID

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000/info" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "guardrail_id": "123e4567-e89b-12d3-a456-426614174000",
        "guardrail_name": "my-bedrock-guard",
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "pre_call",
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "default_on": true
        },
        "guardrail_info": {
            "description": "Bedrock content moderation guardrail"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T12:34:56.789Z"
    }
    ```
    """

    from litellm.litellm_core_utils.litellm_logging import _get_masked_values
    from litellm.proxy.guardrails.guardrail_registry import \
        IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client
    from litellm.types.guardrails import GUARDRAIL_DEFINITION_LOCATION

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        guardrail_definition_location: GUARDRAIL_DEFINITION_LOCATION = (
            GUARDRAIL_DEFINITION_LOCATION.DB
        )
        result = await GUARDRAIL_REGISTRY.get_guardrail_by_id_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )
        if result is None:
            result = IN_MEMORY_GUARDRAIL_HANDLER.get_guardrail_by_id(
                guardrail_id=guardrail_id
            )
            guardrail_definition_location = GUARDRAIL_DEFINITION_LOCATION.CONFIG

        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Guardrail with ID {guardrail_id} not found"
            )

        litellm_params: Optional[Union[LitellmParams, dict]] = result.get(
            "litellm_params"
        )
        result_litellm_params_dict = (
            litellm_params.model_dump(exclude_none=True)
            if isinstance(litellm_params, LitellmParams)
            else litellm_params
        ) or {}
        masked_litellm_params_dict = _get_masked_values(
            result_litellm_params_dict,
            unmasked_length=4,
            number_of_asterisks=4,
        )
        masked_litellm_params = (
            BaseLitellmParams(**masked_litellm_params_dict)
            if masked_litellm_params_dict
            else None
        )

        return GuardrailInfoResponse(
            guardrail_id=result.get("guardrail_id"),
            guardrail_name=result.get("guardrail_name"),
            litellm_params=masked_litellm_params,
            guardrail_info=dict(result.get("guardrail_info") or {}),
            created_at=result.get("created_at"),
            updated_at=result.get("updated_at"),
            guardrail_definition_location=guardrail_definition_location,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/guardrails/ui/add_guardrail_settings",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_guardrail_ui_settings():
    """
    Get the UI settings for the guardrails

    Returns:
    - Supported entities for guardrails
    - Supported modes for guardrails
    - PII entity categories for UI organization
    - Content filter settings (patterns and categories)
    """
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
        PATTERN_CATEGORIES, get_available_content_categories,
        get_pattern_metadata)

    # Convert the PII_ENTITY_CATEGORIES_MAP to the format expected by the UI
    category_maps = []
    for category, entities in PII_ENTITY_CATEGORIES_MAP.items():
        category_maps.append(
            {
                "category": category.value,
                "entities": [entity.value for entity in entities],
            }
        )

    return GuardrailUIAddGuardrailSettings(
        supported_entities=[entity.value for entity in PiiEntityType],
        supported_actions=[action.value for action in PiiAction],
        supported_modes=[mode.value for mode in GuardrailEventHooks],
        pii_entity_categories=category_maps,
        content_filter_settings={
            "prebuilt_patterns": get_pattern_metadata(),
            "pattern_categories": list(PATTERN_CATEGORIES.keys()),
            "supported_actions": ["BLOCK", "MASK"],
            "content_categories": get_available_content_categories(),
        },
    )


@router.get(
    "/guardrails/ui/category_yaml/{category_name}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_category_yaml(category_name: str):
    """
    Get the YAML or JSON content for a specific content filter category.

    Args:
        category_name: The name of the category (e.g., "bias_gender", "harmful_self_harm")

    Returns:
        The raw YAML or JSON content of the category file with file type indicator
    """
    import os

    # Get the categories directory path
    categories_dir = os.path.join(
        os.path.dirname(__file__),
        "guardrail_hooks",
        "litellm_content_filter",
        "categories",
    )

    # Try to find the file with either .yaml or .json extension
    yaml_path = os.path.join(categories_dir, f"{category_name}.yaml")
    json_path = os.path.join(categories_dir, f"{category_name}.json")

    category_file_path = None
    file_type = None

    if os.path.exists(yaml_path):
        category_file_path = yaml_path
        file_type = "yaml"
    elif os.path.exists(json_path):
        category_file_path = json_path
        file_type = "json"
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Category file not found: {category_name} (tried .yaml and .json)",
        )

    try:
        # Read and return the raw content
        with open(category_file_path, "r") as f:
            content = f.read()

        return {
            "category_name": category_name,
            "yaml_content": content,  # Keep key name for backwards compatibility
            "file_type": file_type,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading category file: {str(e)}"
        )


@router.post(
    "/guardrails/validate_blocked_words_file",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def validate_blocked_words_file(request: Dict[str, str]):
    """
    Validate a blocked_words YAML file content.

    Args:
        request: Dictionary with 'file_content' key containing the YAML string

    Returns:
        Dictionary with 'valid' boolean and either 'message'/'errors' depending on result

    Example Request:
    ```json
    {
        "file_content": "blocked_words:\\n  - keyword: \\"test\\"\\n    action: \\"BLOCK\\""
    }
    ```

    Example Success Response:
    ```json
    {
        "valid": true,
        "message": "Valid YAML file with 2 blocked words"
    }
    ```

    Example Error Response:
    ```json
    {
        "valid": false,
        "errors": ["Entry 0: missing 'action' field"]
    }
    ```
    """
    import yaml

    try:
        file_content = request.get("file_content", "")
        if not file_content:
            return {"valid": False, "error": "No file content provided"}

        data = yaml.safe_load(file_content)

        if not isinstance(data, dict) or "blocked_words" not in data:
            return {
                "valid": False,
                "error": "Invalid format: file must contain 'blocked_words' key with a list",
            }

        blocked_words_list = data["blocked_words"]
        if not isinstance(blocked_words_list, list):
            return {"valid": False, "error": "'blocked_words' must be a list"}

        # Validate each entry
        errors = []
        for idx, word_data in enumerate(blocked_words_list):
            if not isinstance(word_data, dict):
                errors.append(f"Entry {idx}: must be an object")
                continue

            if "keyword" not in word_data:
                errors.append(f"Entry {idx}: missing 'keyword' field")
            elif not isinstance(word_data["keyword"], str):
                errors.append(f"Entry {idx}: 'keyword' must be a string")

            if "action" not in word_data:
                errors.append(f"Entry {idx}: missing 'action' field")
            elif word_data["action"] not in ["BLOCK", "MASK"]:
                errors.append(
                    f"Entry {idx}: action must be 'BLOCK' or 'MASK', got '{word_data['action']}'"
                )

            if "description" in word_data and not isinstance(
                word_data["description"], str
            ):
                errors.append(f"Entry {idx}: 'description' must be a string")

        if errors:
            return {"valid": False, "errors": errors}

        return {
            "valid": True,
            "message": f"Valid YAML file with {len(blocked_words_list)} blocked word(s)",
        }
    except yaml.YAMLError as e:
        return {"valid": False, "error": f"Invalid YAML syntax: {str(e)}"}
    except Exception as e:
        verbose_proxy_logger.exception("Error validating blocked words file")
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def _get_field_type_from_annotation(field_annotation: Any) -> str:
    """
    Convert a Python type annotation to a UI-friendly type string
    """
    # Handle Union types (like Optional[T])
    if (
        hasattr(field_annotation, "__origin__")
        and field_annotation.__origin__ is Union
        and hasattr(field_annotation, "__args__")
    ):
        # For Optional[T], get the non-None type
        args = field_annotation.__args__
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            field_annotation = non_none_args[0]

    # Handle List types
    if hasattr(field_annotation, "__origin__") and field_annotation.__origin__ is list:
        return "array"

    # Handle Dict types
    if hasattr(field_annotation, "__origin__") and field_annotation.__origin__ is dict:
        return "dict"

    # Handle Literal types
    if hasattr(field_annotation, "__origin__") and hasattr(
        field_annotation, "__args__"
    ):
        # Check for Literal types (Python 3.8+)
        origin = field_annotation.__origin__
        if hasattr(origin, "__name__") and origin.__name__ == "Literal":
            return "select"  # For dropdown/select inputs

    # Handle basic types
    if field_annotation is str:
        return "string"
    elif field_annotation is int:
        return "number"
    elif field_annotation is float:
        return "number"
    elif field_annotation is bool:
        return "boolean"
    elif field_annotation is dict:
        return "object"
    elif field_annotation is list:
        return "array"

    # Default to string for unknown types
    return "string"


def _extract_literal_values(annotation: Any) -> List[str]:
    """
    Extract literal values from a Literal type annotation
    """
    if hasattr(annotation, "__origin__") and hasattr(annotation, "__args__"):
        origin = annotation.__origin__
        if hasattr(origin, "__name__") and origin.__name__ == "Literal":
            return list(annotation.__args__)
    return []


def _get_dict_key_options(field_annotation: Any) -> Optional[List[str]]:
    """
    Extract key options from Dict[Literal[...], T] types
    """
    if (
        hasattr(field_annotation, "__origin__")
        and field_annotation.__origin__ is dict
        and hasattr(field_annotation, "__args__")
    ):
        args = field_annotation.__args__
        if len(args) >= 2:
            key_type = args[0]
            return _extract_literal_values(key_type)
    return None


def _get_dict_value_type(field_annotation: Any) -> str:
    """
    Get the value type from Dict[K, V] types
    """
    if (
        hasattr(field_annotation, "__origin__")
        and field_annotation.__origin__ is dict
        and hasattr(field_annotation, "__args__")
    ):
        args = field_annotation.__args__
        if len(args) >= 2:
            value_type = args[1]
            return _get_field_type_from_annotation(value_type)
    return "string"


def _get_list_element_options(field_annotation: Any) -> Optional[List[str]]:
    """
    Extract element options from List[Literal[...]] types
    """
    if (
        hasattr(field_annotation, "__origin__")
        and field_annotation.__origin__ is list
        and hasattr(field_annotation, "__args__")
    ):
        args = field_annotation.__args__
        if len(args) >= 1:
            element_type = args[0]
            return _extract_literal_values(element_type)
    return None


def _should_skip_optional_params(field_name: str, field_annotation: Any) -> bool:
    """Check if optional_params field should be skipped (not meaningfully overridden)."""
    if field_name != "optional_params":
        return False

    if field_annotation is None:
        return True

    # Check if the annotation is still a generic TypeVar (not specialized)
    if isinstance(field_annotation, TypeVar) or (
        hasattr(field_annotation, "__origin__")
        and field_annotation.__origin__ is TypeVar
    ):
        return True

    # Also skip if it's a generic type that wasn't specialized
    if hasattr(field_annotation, "__name__") and field_annotation.__name__ in (
        "T",
        "TypeVar",
    ):
        return True

    # Handle Optional[T] where T is still a TypeVar
    if hasattr(field_annotation, "__args__"):
        non_none_args = [
            arg for arg in field_annotation.__args__ if arg is not type(None)
        ]
        if non_none_args and isinstance(non_none_args[0], TypeVar):
            return True

    return False


def _unwrap_optional_type(field_annotation: Any) -> Any:
    """Unwrap Optional types to get the actual type."""
    if (
        hasattr(field_annotation, "__origin__")
        and field_annotation.__origin__ is Union
        and hasattr(field_annotation, "__args__")
    ):
        # For Optional[BaseModel], get the non-None type
        args = field_annotation.__args__
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return non_none_args[0]
    return field_annotation


def _build_field_dict(
    field: Any,
    field_annotation: Any,
    description: str,
    required: bool,
) -> Dict[str, Any]:
    """Build field dictionary for non-nested fields."""
    # Determine the field type from annotation
    field_type = _get_field_type_from_annotation(field_annotation)

    # Check for custom UI type override
    field_json_schema_extra = getattr(field, "json_schema_extra", {})
    if field_json_schema_extra and "ui_type" in field_json_schema_extra:
        field_type = field_json_schema_extra["ui_type"].value
    elif field_json_schema_extra and "type" in field_json_schema_extra:
        field_type = field_json_schema_extra["type"]

    # Add the field to the dictionary
    field_dict = {
        "description": description,
        "required": required,
        "type": field_type,
    }

    # Extract options from type annotations
    if field_type == "dict":
        # For Dict[Literal[...], T] types, extract key options
        dict_key_options = _get_dict_key_options(field_annotation)
        if dict_key_options:
            field_dict["dict_key_options"] = dict_key_options

        # Extract value type for the dict values
        dict_value_type = _get_dict_value_type(field_annotation)
        field_dict["dict_value_type"] = dict_value_type

    elif field_type == "array":
        # For List[Literal[...]] types, extract element options
        list_element_options = _get_list_element_options(field_annotation)
        if list_element_options:
            field_dict["options"] = list_element_options
            field_dict["type"] = "multiselect"

    # Add options if they exist in json_schema_extra (this takes precedence)
    if field_json_schema_extra and "options" in field_json_schema_extra:
        field_dict["options"] = field_json_schema_extra["options"]

    # Add default value if it exists
    if field.default is not None and field.default is not ...:
        field_dict["default_value"] = field.default

    return field_dict


def _extract_fields_recursive(
    model: Type[BaseModel],
    depth: int = 0,
) -> Dict[str, Any]:
    # Check if we've exceeded the maximum recursion depth
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise HTTPException(
            status_code=400,
            detail=f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing model fields. Please check the model structure for excessive nesting.",
        )

    fields = {}

    for field_name, field in model.model_fields.items():
        field_annotation = field.annotation

        # Skip optional_params if it's not meaningfully overridden
        if _should_skip_optional_params(
            field_name=field_name, field_annotation=field_annotation
        ):
            continue

        # Handle Optional types and get the actual type
        if field_annotation is None:
            continue

        field_annotation = _unwrap_optional_type(field_annotation=field_annotation)

        # Get field metadata
        description = field.description or field_name
        required = field.is_required()

        # Check if this is a BaseModel subclass
        is_basemodel_subclass = (
            inspect.isclass(field_annotation)
            and issubclass(field_annotation, BaseModel)
            and field_annotation is not BaseModel
        )

        if is_basemodel_subclass:
            # Recursively get fields from the nested model
            nested_fields = _extract_fields_recursive(
                cast(Type[BaseModel], field_annotation), depth + 1
            )
            fields[field_name] = {
                "description": description,
                "required": required,
                "type": "nested",
                "fields": nested_fields,
            }
        else:
            fields[field_name] = _build_field_dict(
                field=field,
                field_annotation=field_annotation,
                description=description,
                required=required,
            )

    return fields


def _get_fields_from_model(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Get the fields from a Pydantic model as a nested dictionary structure
    """

    return _extract_fields_recursive(model_class, depth=0)


@router.get(
    "/guardrails/ui/provider_specific_params",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_provider_specific_params():
    """
    Get provider-specific parameters for different guardrail types.

    Returns a dictionary mapping guardrail providers to their specific parameters,
    including parameter names, descriptions, and whether they are required.

    Example Response:
    ```json
    {
        "bedrock": {
            "guardrailIdentifier": {
                "description": "The ID of your guardrail on Bedrock",
                "required": true,
                "type": null
            },
            "guardrailVersion": {
                "description": "The version of your Bedrock guardrail (e.g., DRAFT or version number)",
                "required": true,
                "type": null
            }
        },
        "azure_content_safety_text_moderation": {
            "api_key": {
                "description": "API key for the Azure Content Safety Text Moderation guardrail",
                "required": false,
                "type": null
            },
            "optional_params": {
                "description": "Optional parameters for the Azure Content Safety Text Moderation guardrail",
                "required": true,
                "type": "nested",
                "fields": {
                    "severity_threshold": {
                        "description": "Severity threshold for the Azure Content Safety Text Moderation guardrail across all categories",
                        "required": false,
                        "type": null
                    },
                    "categories": {
                        "description": "Categories to scan for the Azure Content Safety Text Moderation guardrail",
                        "required": false,
                        "type": "multiselect",
                        "options": ["Hate", "SelfHarm", "Sexual", "Violence"],
                        "default_value": None
                    }
                }
            }
        }
    }
    ```
    """
    # Get fields from the models
    bedrock_fields = _get_fields_from_model(BedrockGuardrailConfigModel)
    presidio_fields = _get_fields_from_model(PresidioPresidioConfigModelUserInterface)
    lakera_v2_fields = _get_fields_from_model(LakeraV2GuardrailConfigModel)
    tool_permission_fields = _get_fields_from_model(ToolPermissionGuardrailConfigModel)

    tool_permission_fields["ui_friendly_name"] = (
        ToolPermissionGuardrailConfigModel.ui_friendly_name()
    )

    # Return the provider-specific parameters
    provider_params = {
        SupportedGuardrailIntegrations.BEDROCK.value: bedrock_fields,
        SupportedGuardrailIntegrations.PRESIDIO.value: presidio_fields,
        SupportedGuardrailIntegrations.LAKERA_V2.value: lakera_v2_fields,
        SupportedGuardrailIntegrations.TOOL_PERMISSION.value: tool_permission_fields,
    }

    ### get the config model for the guardrail - go through the registry and get the config model for the guardrail
    from litellm.proxy.guardrails.guardrail_registry import \
        guardrail_class_registry

    for guardrail_name, guardrail_class in guardrail_class_registry.items():
        guardrail_config_model = guardrail_class.get_config_model()

        if guardrail_config_model:
            fields = _get_fields_from_model(guardrail_config_model)
            ui_friendly_name = guardrail_config_model.ui_friendly_name()
            fields["ui_friendly_name"] = ui_friendly_name
            provider_params[guardrail_name] = fields

    return provider_params


class TestCustomCodeGuardrailRequest(BaseModel):
    """Request model for testing custom code guardrails."""

    custom_code: str
    """The Python-like code containing the apply_guardrail function."""

    test_input: Dict[str, Any]
    """The test input to pass to the guardrail. Should contain 'texts', optionally 'images', 'tools', etc."""

    input_type: str = "request"
    """Whether this is a 'request' or 'response' input type."""

    request_data: Optional[Dict[str, Any]] = None
    """Optional mock request_data (model, user_id, team_id, metadata, etc.)."""


class TestCustomCodeGuardrailResponse(BaseModel):
    """Response model for testing custom code guardrails."""

    success: bool
    """Whether the test executed successfully (no errors)."""

    result: Optional[Dict[str, Any]] = None
    """The guardrail result: action (allow/block/modify), reason, modified_texts, etc."""

    error: Optional[str] = None
    """Error message if execution failed."""

    error_type: Optional[str] = None
    """Type of error: 'compilation' or 'execution'."""


@router.post(
    "/guardrails/test_custom_code",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TestCustomCodeGuardrailResponse,
)
async def test_custom_code_guardrail(request: TestCustomCodeGuardrailRequest):
    """
    Test custom code guardrail logic without creating a guardrail.

    This endpoint allows admins to experiment with custom code guardrails by:
    1. Compiling the provided code in a sandbox
    2. Executing the apply_guardrail function with test input
    3. Returning the result (allow/block/modify)

    👉 [Custom Code Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/custom_code_guardrail)

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/guardrails/test_custom_code" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "custom_code": "def apply_guardrail(inputs, request_data, input_type):\\n    for text in inputs[\\"texts\\"]:\\n        if regex_match(text, r\\"\\\\d{3}-\\\\d{2}-\\\\d{4}\\"):\\n            return block(\\"SSN detected\\")\\n    return allow()",
            "test_input": {
                "texts": ["My SSN is 123-45-6789"]
            },
            "input_type": "request"
        }'
    ```

    Example Success Response (blocked):
    ```json
    {
        "success": true,
        "result": {
            "action": "block",
            "reason": "SSN detected"
        },
        "error": null,
        "error_type": null
    }
    ```

    Example Success Response (allowed):
    ```json
    {
        "success": true,
        "result": {
            "action": "allow"
        },
        "error": null,
        "error_type": null
    }
    ```

    Example Success Response (modified):
    ```json
    {
        "success": true,
        "result": {
            "action": "modify",
            "texts": ["My SSN is [REDACTED]"]
        },
        "error": null,
        "error_type": null
    }
    ```

    Example Error Response (compilation error):
    ```json
    {
        "success": false,
        "result": null,
        "error": "Syntax error in custom code: invalid syntax (<guardrail>, line 1)",
        "error_type": "compilation"
    }
    ```
    """
    import concurrent.futures
    import re

    from litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives import \
        get_custom_code_primitives

    # Security validation patterns
    FORBIDDEN_PATTERNS = [
        # Import statements
        (r"\bimport\s+", "import statements are not allowed"),
        (r"\bfrom\s+\w+\s+import\b", "from...import statements are not allowed"),
        (r"__import__\s*\(", "__import__() is not allowed"),
        # Dangerous builtins
        (r"\bexec\s*\(", "exec() is not allowed"),
        (r"\beval\s*\(", "eval() is not allowed"),
        (r"\bcompile\s*\(", "compile() is not allowed"),
        (r"\bopen\s*\(", "open() is not allowed"),
        (r"\bgetattr\s*\(", "getattr() is not allowed"),
        (r"\bsetattr\s*\(", "setattr() is not allowed"),
        (r"\bdelattr\s*\(", "delattr() is not allowed"),
        (r"\bglobals\s*\(", "globals() is not allowed"),
        (r"\blocals\s*\(", "locals() is not allowed"),
        (r"\bvars\s*\(", "vars() is not allowed"),
        (r"\bdir\s*\(", "dir() is not allowed"),
        (r"\bbreakpoint\s*\(", "breakpoint() is not allowed"),
        (r"\binput\s*\(", "input() is not allowed"),
        # Dangerous dunder access
        (r"__builtins__", "__builtins__ access is not allowed"),
        (r"__globals__", "__globals__ access is not allowed"),
        (r"__code__", "__code__ access is not allowed"),
        (r"__subclasses__", "__subclasses__ access is not allowed"),
        (r"__bases__", "__bases__ access is not allowed"),
        (r"__mro__", "__mro__ access is not allowed"),
        (r"__class__", "__class__ access is not allowed"),
        (r"__dict__", "__dict__ access is not allowed"),
        (r"__getattribute__", "__getattribute__ access is not allowed"),
        (r"__reduce__", "__reduce__ access is not allowed"),
        (r"__reduce_ex__", "__reduce_ex__ access is not allowed"),
        # OS/system access
        (r"\bos\.", "os module access is not allowed"),
        (r"\bsys\.", "sys module access is not allowed"),
        (r"\bsubprocess\.", "subprocess module access is not allowed"),
    ]

    EXECUTION_TIMEOUT_SECONDS = 5

    try:
        # Step 0: Security validation - check for forbidden patterns
        code = request.custom_code
        for pattern, error_msg in FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                return TestCustomCodeGuardrailResponse(
                    success=False,
                    error=f"Security violation: {error_msg}",
                    error_type="compilation",
                )

        # Step 1: Compile the custom code with restricted environment
        exec_globals = get_custom_code_primitives().copy()

        # Remove access to builtins to prevent escape
        exec_globals["__builtins__"] = {}

        try:
            exec(compile(request.custom_code, "<guardrail>", "exec"), exec_globals)
        except SyntaxError as e:
            return TestCustomCodeGuardrailResponse(
                success=False,
                error=f"Syntax error in custom code: {e}",
                error_type="compilation",
            )
        except Exception as e:
            return TestCustomCodeGuardrailResponse(
                success=False,
                error=f"Failed to compile custom code: {e}",
                error_type="compilation",
            )

        # Step 2: Verify apply_guardrail function exists
        if "apply_guardrail" not in exec_globals:
            return TestCustomCodeGuardrailResponse(
                success=False,
                error="Custom code must define an 'apply_guardrail' function. "
                "Expected signature: apply_guardrail(inputs, request_data, input_type)",
                error_type="compilation",
            )

        apply_fn = exec_globals["apply_guardrail"]
        if not callable(apply_fn):
            return TestCustomCodeGuardrailResponse(
                success=False,
                error="'apply_guardrail' must be a callable function",
                error_type="compilation",
            )

        # Step 3: Prepare test inputs
        test_inputs = request.test_input
        if "texts" not in test_inputs:
            test_inputs["texts"] = []

        # Prepare mock request_data
        mock_request_data = request.request_data or {}
        safe_request_data = {
            "model": mock_request_data.get("model", "test-model"),
            "user_id": mock_request_data.get("user_id"),
            "team_id": mock_request_data.get("team_id"),
            "end_user_id": mock_request_data.get("end_user_id"),
            "metadata": mock_request_data.get("metadata", {}),
        }

        # Step 4: Execute the function with timeout protection

        def execute_guardrail():
            return apply_fn(test_inputs, safe_request_data, request.input_type)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_guardrail)
                try:
                    result = future.result(timeout=EXECUTION_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    return TestCustomCodeGuardrailResponse(
                        success=False,
                        error=f"Execution timeout: code took longer than {EXECUTION_TIMEOUT_SECONDS} seconds",
                        error_type="execution",
                    )
        except Exception as e:
            return TestCustomCodeGuardrailResponse(
                success=False,
                error=f"Execution error: {e}",
                error_type="execution",
            )

        # Step 5: Validate and return result
        if not isinstance(result, dict):
            return TestCustomCodeGuardrailResponse(
                success=True,
                result={
                    "action": "allow",
                    "warning": f"Expected dict result, got {type(result).__name__}. Treating as allow.",
                },
            )

        return TestCustomCodeGuardrailResponse(
            success=True,
            result=result,
        )

    except Exception as e:
        verbose_proxy_logger.exception(f"Error testing custom code guardrail: {e}")
        return TestCustomCodeGuardrailResponse(
            success=False,
            error=f"Unexpected error: {e}",
            error_type="execution",
        )


@router.post("/guardrails/apply_guardrail", response_model=ApplyGuardrailResponse)
@router.post("/apply_guardrail", response_model=ApplyGuardrailResponse)
async def apply_guardrail(
    request: ApplyGuardrailRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Apply a guardrail to text input and return the processed result.

    This endpoint allows testing guardrails by applying them to custom text inputs.
    """
    from litellm.proxy.utils import handle_exception_on_proxy

    try:
        active_guardrail: Optional[CustomGuardrail] = (
            GUARDRAIL_REGISTRY.get_initialized_guardrail_callback(
                guardrail_name=request.guardrail_name
            )
        )
        if active_guardrail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Guardrail '{request.guardrail_name}' not found. Please ensure the guardrail is configured in your LiteLLM proxy.",
            )

        guardrailed_inputs = await active_guardrail.apply_guardrail(
            inputs={"texts": [request.text]},
            request_data={},
            input_type="request",
        )
        response_text = guardrailed_inputs.get("texts", [])

        return ApplyGuardrailResponse(
            response_text=response_text[0] if response_text else request.text
        )
    except Exception as e:
        raise handle_exception_on_proxy(e)
