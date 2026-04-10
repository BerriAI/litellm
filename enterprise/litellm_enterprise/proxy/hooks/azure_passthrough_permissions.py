"""
Permission management for Azure OpenAI pass-through routes.

Intercepts /batches, /files, and /responses operations on Azure pass-through
to enforce per-user ownership:

- Create: After Azure returns, store ownership in ManagedObjectTable/ManagedFileTable
- Read/Delete/Cancel: Check ownership before forwarding, rewrite managed IDs to real IDs
- List: Filter results by user_id from the managed tables
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from fastapi import HTTPException, Response

from litellm import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import (
    encode_file_id_with_model,
    get_original_file_id,
    is_model_embedded_id,
)

# Regex to match Azure OpenAI deployment-based resource endpoints
# Captures: resource_type (batches|files|responses), resource_id (optional), sub_action (cancel|content, optional)
_AZURE_RESOURCE_PATTERN = re.compile(
    r"openai/deployments/[^/]+/(batches|files|responses)(?:/([^/?]+))?(?:/(cancel|content))?$"
)


ResourceType = Literal["batch", "file", "response"]
OperationType = Literal["create", "retrieve", "delete", "list", "cancel", "content"]


@dataclass
class AzurePassthroughOp:
    """Classified Azure pass-through operation."""

    resource_type: ResourceType
    operation: OperationType
    resource_id: Optional[str]  # Raw ID from URL path, if present


def classify_azure_passthrough_request(
    endpoint: str, method: str
) -> Optional[AzurePassthroughOp]:
    """
    Classify an Azure pass-through request by resource type and operation.

    Args:
        endpoint: The path portion after /azure/ (e.g. "openai/deployments/gpt-4/batches/batch_abc123")
        method: HTTP method (GET, POST, DELETE, etc.)

    Returns:
        AzurePassthroughOp if matched, None otherwise
    """
    match = _AZURE_RESOURCE_PATTERN.search(endpoint)
    if not match:
        return None

    resource_type_raw = match.group(1)  # batches, files, responses
    resource_id = match.group(2)  # e.g. batch_abc123, file-xyz, resp_123
    sub_action = match.group(3)  # cancel, content

    # Map plural URL segment to singular resource type
    resource_type_map: Dict[str, ResourceType] = {
        "batches": "batch",
        "files": "file",
        "responses": "response",
    }
    resource_type = resource_type_map[resource_type_raw]

    method_upper = method.upper()

    # Determine operation
    if sub_action == "cancel":
        operation: OperationType = "cancel"
    elif sub_action == "content":
        operation = "content"
    elif resource_id is not None:
        # Has resource ID in path
        if method_upper == "GET":
            operation = "retrieve"
        elif method_upper == "DELETE":
            operation = "delete"
        elif method_upper == "POST" and resource_type == "batch":
            # POST to /batches/{id} is not standard; treat as retrieve for safety
            operation = "retrieve"
        else:
            operation = "retrieve"
    else:
        # No resource ID
        if method_upper == "POST":
            operation = "create"
        elif method_upper == "GET":
            operation = "list"
        else:
            return None

    return AzurePassthroughOp(
        resource_type=resource_type,
        operation=operation,
        resource_id=resource_id,
    )


async def passthrough_pre_request(
    endpoint: str,
    request_method: str,
    user_api_key_dict: UserAPIKeyAuth,
    managed_files_obj: Any,
) -> Tuple[str, Optional[AzurePassthroughOp]]:
    """
    Pre-request hook for Azure pass-through permission management.

    For read/delete/cancel/content operations:
    - Checks if the resource ID is a managed (encoded) ID
    - Verifies ownership
    - Rewrites the endpoint to use the real provider ID

    Args:
        endpoint: URL path after /azure/
        request_method: HTTP method
        user_api_key_dict: Authenticated user info
        managed_files_obj: Enterprise managed files hook instance

    Returns:
        Tuple of (potentially rewritten endpoint, classified operation)
    """
    op = classify_azure_passthrough_request(endpoint, request_method)
    if op is None:
        return endpoint, None

    # Only intercept operations that target a specific resource
    if op.operation not in ("retrieve", "delete", "cancel", "content"):
        return endpoint, op

    if op.resource_id is None:
        return endpoint, op

    if managed_files_obj is None:
        return endpoint, op

    # Check if the resource ID is a managed (encoded) ID
    resource_id = op.resource_id

    # Check if the resource ID is a model-embedded (managed) ID
    if is_model_embedded_id(resource_id):
        if op.resource_type == "file":
            can_access = await managed_files_obj.can_user_call_unified_file_id(
                unified_file_id=resource_id,
                user_api_key_dict=user_api_key_dict,
            )
        else:
            can_access = await managed_files_obj.can_user_call_unified_object_id(
                unified_object_id=resource_id,
                user_api_key_dict=user_api_key_dict,
            )
        if not can_access:
            raise HTTPException(
                status_code=403,
                detail=f"You don't have permission to access this {op.resource_type}.",
            )
        # Decode real provider ID and rewrite endpoint
        real_id = get_original_file_id(resource_id)
        endpoint = endpoint.replace(resource_id, real_id)
        op.resource_id = real_id
    elif op.resource_type in ("batch", "response"):
        # Raw provider ID - check if it exists in the managed object table
        try:
            db_obj = await managed_files_obj.prisma_client.db.litellm_managedobjecttable.find_first(
                where={"model_object_id": resource_id}
            )
            if db_obj and db_obj.created_by != user_api_key_dict.user_id:
                raise HTTPException(
                    status_code=403,
                    detail=f"You don't have permission to access this {op.resource_type}.",
                )
        except HTTPException:
            raise
        except Exception:
            # DB lookup failed - pass through (backward compat)
            pass
    elif op.resource_type == "file":
        # Raw file ID - check if it exists in the managed file table
        try:
            db_obj = await managed_files_obj.prisma_client.db.litellm_managedfiletable.find_first(
                where={"flat_model_file_ids": {"has": resource_id}}
            )
            if db_obj and db_obj.created_by != user_api_key_dict.user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to access this file.",
                )
        except HTTPException:
            raise
        except Exception:
            # DB lookup failed - pass through (backward compat)
            pass

    return endpoint, op


async def passthrough_post_response(
    response_body: bytes,
    op: AzurePassthroughOp,
    user_api_key_dict: UserAPIKeyAuth,
    managed_files_obj: Any,
    deployment_name: Optional[str] = None,
) -> bytes:
    """
    Post-response hook for Azure pass-through permission management.

    For create operations:
    - Parses the response body
    - Encodes provider IDs into managed IDs
    - Stores ownership in the database

    Args:
        response_body: Raw response body bytes from Azure
        op: Classified operation
        user_api_key_dict: Authenticated user info
        managed_files_obj: Enterprise managed files hook instance
        deployment_name: Azure deployment name (used as model for encoding)

    Returns:
        Potentially modified response body bytes
    """
    if op.operation != "create":
        return response_body

    if managed_files_obj is None:
        return response_body

    try:
        response_json = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return response_body

    provider_id = response_json.get("id")
    if not provider_id:
        return response_body

    # Use deployment_name as the model for encoding, fallback to "azure-passthrough"
    model = deployment_name or "azure-passthrough"

    if op.resource_type == "file":
        # Encode file ID
        managed_id = encode_file_id_with_model(
            file_id=provider_id, model=model, id_type="file"
        )
        response_json["id"] = managed_id

        # Store ownership in DB
        try:
            from litellm.types.llms.openai import OpenAIFileObject

            file_object = OpenAIFileObject(**response_json)
            file_object.id = provider_id  # Store with original ID
            await managed_files_obj.store_unified_file_id(
                file_id=managed_id,
                file_object=file_object,
                litellm_parent_otel_span=None,
                model_mappings={model: provider_id},
                user_api_key_dict=user_api_key_dict,
            )
        except Exception as e:
            verbose_logger.error(
                f"Failed to store managed file ID for passthrough: {e}"
            )

    elif op.resource_type == "batch":
        # Encode batch ID
        managed_id = encode_file_id_with_model(
            file_id=provider_id, model=model, id_type="batch"
        )
        response_json["id"] = managed_id

        # Also encode input_file_id, output_file_id, error_file_id if present
        for file_field in ("input_file_id", "output_file_id", "error_file_id"):
            raw_file_id = response_json.get(file_field)
            if raw_file_id:
                response_json[file_field] = encode_file_id_with_model(
                    file_id=raw_file_id, model=model, id_type="file"
                )

        # Store ownership in DB
        try:
            from litellm.types.utils import LiteLLMBatch

            batch_object = LiteLLMBatch(**response_json)
            batch_object.id = provider_id  # Store with original ID
            await managed_files_obj.store_unified_object_id(
                unified_object_id=managed_id,
                file_object=batch_object,
                litellm_parent_otel_span=None,
                model_object_id=provider_id,
                file_purpose="batch",
                user_api_key_dict=user_api_key_dict,
            )
        except Exception as e:
            verbose_logger.error(
                f"Failed to store managed batch ID for passthrough: {e}"
            )

    elif op.resource_type == "response":
        # Encode response ID
        managed_id = encode_file_id_with_model(
            file_id=provider_id, model=model, id_type="batch"
        )
        response_json["id"] = managed_id

        # Store ownership in DB
        try:
            from litellm.types.llms.openai import ResponsesAPIResponse

            response_obj = ResponsesAPIResponse(**response_json)
            response_obj.id = provider_id  # Store with original ID
            await managed_files_obj.store_unified_object_id(
                unified_object_id=managed_id,
                file_object=response_obj,
                litellm_parent_otel_span=None,
                model_object_id=provider_id,
                file_purpose="response",
                user_api_key_dict=user_api_key_dict,
            )
        except Exception as e:
            verbose_logger.error(
                f"Failed to store managed response ID for passthrough: {e}"
            )

    return json.dumps(response_json).encode()


async def passthrough_list_filter(
    op: AzurePassthroughOp,
    user_api_key_dict: UserAPIKeyAuth,
    managed_files_obj: Any,
) -> Optional[Response]:
    """
    Handle list operations by filtering from the managed tables.

    Returns a Response if handled, None to fall through to Azure.
    """
    if op.operation != "list":
        return None

    if managed_files_obj is None:
        return None

    if op.resource_type == "batch":
        try:
            result = await managed_files_obj.list_user_batches(
                user_api_key_dict=user_api_key_dict,
            )
            return Response(
                content=json.dumps(result),
                media_type="application/json",
                status_code=200,
            )
        except Exception as e:
            verbose_logger.warning(
                f"Failed to list managed batches, falling through to Azure: {e}"
            )
            return None

    elif op.resource_type == "file":
        try:
            where_clause: Dict[str, Any] = {}
            if user_api_key_dict.user_id:
                where_clause["created_by"] = user_api_key_dict.user_id

            files = await managed_files_obj.prisma_client.db.litellm_managedfiletable.find_many(
                where=where_clause,
                order={"created_at": "desc"},
            )

            file_objects = []
            for f in files:
                try:
                    file_data = (
                        json.loads(f.file_object)
                        if isinstance(f.file_object, str)
                        else f.file_object
                    )
                    if isinstance(file_data, dict):
                        file_data["id"] = f.unified_file_id
                        file_objects.append(file_data)
                except Exception:
                    continue

            result = {
                "object": "list",
                "data": file_objects,
                "has_more": False,
            }
            return Response(
                content=json.dumps(result),
                media_type="application/json",
                status_code=200,
            )
        except Exception as e:
            verbose_logger.warning(
                f"Failed to list managed files, falling through to Azure: {e}"
            )
            return None

    return None
