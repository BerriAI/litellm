"""
CRUD ENDPOINTS FOR GUARDRAILS
"""

from typing import Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.guardrail_registry import GuardrailRegistry
from litellm.types.guardrails import (
    PII_ENTITY_CATEGORIES_MAP,
    Guardrail,
    GuardrailEventHooks,
    GuardrailInfoResponse,
    GuardrailUIAddGuardrailSettings,
    ListGuardrailsResponse,
    PiiAction,
    PiiEntityType,
)

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
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        guardrails = await GUARDRAIL_REGISTRY.get_all_guardrails_from_db(
            prisma_client=prisma_client
        )

        guardrail_configs: List[GuardrailInfoResponse] = []
        for guardrail in guardrails:
            guardrail_configs.append(
                GuardrailInfoResponse(
                    guardrail_name=guardrail.get("guardrail_name"),
                    litellm_params=guardrail.get("litellm_params"),
                    guardrail_info=guardrail.get("guardrail_info"),
                    created_at=guardrail.get("created_at"),
                    updated_at=guardrail.get("updated_at"),
                )
            )

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
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await GUARDRAIL_REGISTRY.add_guardrail_to_db(
            guardrail=request.guardrail, prisma_client=prisma_client
        )
        return result
    except Exception as e:
        verbose_proxy_logger.exception(f"Error adding guardrail to db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/guardrails/{guardrail_id}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_guardrail(guardrail_id: str):
    """
    Get a guardrail by ID

    👉 [Guardrail docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/guardrails/123e4567-e89b-12d3-a456-426614174000" \\
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
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await GUARDRAIL_REGISTRY.get_guardrail_by_id_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Guardrail with ID {guardrail_id} not found"
            )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
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
        return result
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
    """
    # Convert the PII_ENTITY_CATEGORIES_MAP to the format expected by the UI
    category_maps = []
    for category, entities in PII_ENTITY_CATEGORIES_MAP.items():
        category_maps.append({"category": category, "entities": entities})

    return GuardrailUIAddGuardrailSettings(
        supported_entities=list(PiiEntityType),
        supported_actions=list(PiiAction),
        supported_modes=list(GuardrailEventHooks),
        pii_entity_categories=category_maps,
    )
