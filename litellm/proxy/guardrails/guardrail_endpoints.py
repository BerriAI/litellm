"""
CRUD ENDPOINTS FOR GUARDRAILS
"""

import inspect
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.guardrail_registry import GuardrailRegistry
from litellm.types.guardrails import (
    PII_ENTITY_CATEGORIES_MAP,
    BedrockGuardrailConfigModel,
    Guardrail,
    GuardrailEventHooks,
    GuardrailInfoResponse,
    GuardrailUIAddGuardrailSettings,
    LakeraV2GuardrailConfigModel,
    ListGuardrailsResponse,
    LitellmParams,
    PatchGuardrailRequest,
    PiiAction,
    PiiEntityType,
    PresidioPresidioConfigModelUserInterface,
    SupportedGuardrailIntegrations,
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
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER
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
            guardrail_configs.append(
                GuardrailInfoResponse(
                    guardrail_id=guardrail.get("guardrail_id"),
                    guardrail_name=guardrail.get("guardrail_name"),
                    litellm_params=guardrail.get("litellm_params"),
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
                guardrail_configs.append(
                    GuardrailInfoResponse(
                        guardrail_id=guardrail.get("guardrail_id"),
                        guardrail_name=guardrail.get("guardrail_name"),
                        litellm_params=dict(guardrail.get("litellm_params") or {}),
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
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER
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

        # delete in memory guardrail
        IN_MEMORY_GUARDRAIL_HANDLER.delete_in_memory_guardrail(
            guardrail_id=guardrail_id,
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
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER
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

        # update in memory guardrail
        IN_MEMORY_GUARDRAIL_HANDLER.update_in_memory_guardrail(
            guardrail_id=guardrail_id,
            guardrail=guardrail,
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
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        result = await GUARDRAIL_REGISTRY.get_guardrail_by_id_from_db(
            guardrail_id=guardrail_id, prisma_client=prisma_client
        )
        if result is None:
            result = IN_MEMORY_GUARDRAIL_HANDLER.get_guardrail_by_id(
                guardrail_id=guardrail_id
            )

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

        return GuardrailInfoResponse(
            guardrail_id=result.get("guardrail_id"),
            guardrail_name=result.get("guardrail_name"),
            litellm_params=masked_litellm_params_dict,
            guardrail_info=dict(result.get("guardrail_info") or {}),
            created_at=result.get("created_at"),
            updated_at=result.get("updated_at"),
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
        # Skip optional_params if it's not meaningfully overridden
        if field_name == "optional_params":
            field_annotation = field.annotation
            if field_annotation is None:
                continue
            # Check if the annotation is still a generic TypeVar (not specialized)
            if isinstance(field_annotation, TypeVar) or (
                hasattr(field_annotation, "__origin__")
                and field_annotation.__origin__ is TypeVar
            ):
                # Skip this field as it's not meaningfully overridden
                continue
            # Also skip if it's a generic type that wasn't specialized
            if hasattr(field_annotation, "__name__") and field_annotation.__name__ in (
                "T",
                "TypeVar",
            ):
                continue

        # Get field metadata
        description = field.description or field_name

        # Check if this field is required
        required = field.is_required()

        # Check if the field annotation is a BaseModel subclass
        field_annotation = field.annotation

        # Handle Optional types and get the actual type
        if field_annotation is None:
            continue

        if (
            hasattr(field_annotation, "__origin__")
            and field_annotation.__origin__ is Union
            and hasattr(field_annotation, "__args__")
        ):
            # For Optional[BaseModel], get the non-None type
            args = field_annotation.__args__
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                field_annotation = non_none_args[0]

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

            fields[field_name] = field_dict

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

    # Return the provider-specific parameters
    provider_params = {
        SupportedGuardrailIntegrations.BEDROCK.value: bedrock_fields,
        SupportedGuardrailIntegrations.PRESIDIO.value: presidio_fields,
        SupportedGuardrailIntegrations.LAKERA_V2.value: lakera_v2_fields,
    }

    ### get the config model for the guardrail - go through the registry and get the config model for the guardrail
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    for guardrail_name, guardrail_class in guardrail_class_registry.items():
        guardrail_config_model = guardrail_class.get_config_model()

        if guardrail_config_model:
            fields = _get_fields_from_model(guardrail_config_model)
            ui_friendly_name = guardrail_config_model.ui_friendly_name()
            fields["ui_friendly_name"] = ui_friendly_name
            provider_params[guardrail_name] = fields

    return provider_params
