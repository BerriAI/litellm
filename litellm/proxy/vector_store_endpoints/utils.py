from typing import Any, Dict, Literal, Optional

from fastapi import HTTPException, Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _does_endpoint_match(endpoint_path: str, request_path: str) -> bool:
    if endpoint_path in request_path:
        return True
    if "{" in endpoint_path:
        prefix = endpoint_path.split("{", 1)[0]
        if prefix and prefix in request_path:
            return True
    return False


def check_vector_store_permission(
    index_name: str,
    permission: str,
    key_metadata: Optional[Dict[str, Any]],
    team_metadata: Optional[Dict[str, Any]],
) -> bool:
    """
    Check if a specific permission is allowed for a given vector store index.

    Args:
        index_name: The name of the vector store index
        permission: The permission to check (e.g., "read", "write")
        key_metadata: Metadata from the API key
        team_metadata: Metadata from the team

    Returns:
        True if the permission is allowed, False otherwise

    Example metadata format:
        "metadata": {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "dall-e-3",
                    "index_permissions": ["write"]
                }
            ]
        }
    """
    # Check both key_metadata and team_metadata
    for metadata in [key_metadata, team_metadata]:
        if metadata is None:
            continue

        allowed_indexes = metadata.get("allowed_vector_store_indexes")
        if not allowed_indexes or not isinstance(allowed_indexes, list):
            continue

        # Look for matching index
        for index_config in allowed_indexes:
            if not isinstance(index_config, dict):
                continue

            if index_config.get("index_name") == index_name:
                index_permissions = index_config.get("index_permissions", [])
                if (
                    isinstance(index_permissions, list)
                    and permission in index_permissions
                ):
                    return True

    return False


def is_allowed_to_call_vector_store_endpoint(
    provider: LlmProviders,
    index_name: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Literal[True]]:
    """
    Check if the user is allowed to call the vector store endpoint.

    Cover:
    1. Creating a vector store index
    2. Reading a vector store index (Search / List / Get)
    """
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True
    # check what allowed permissions are for the key
    key_metadata = user_api_key_dict.metadata
    team_metadata = user_api_key_dict.team_metadata

    provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=provider
    )
    if provider_config is None:
        return None

    provider_vector_store_endpoints = (
        provider_config.get_vector_store_endpoints_by_type()
    )

    # Determine the permission type based on the request
    permission_type = None
    for endpoint in provider_vector_store_endpoints["read"]:
        if request.method == endpoint[0] and _does_endpoint_match(
            endpoint[1], request.url.path
        ):
            permission_type = "read"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints["write"]:
            if request.method == endpoint[0] and _does_endpoint_match(
                endpoint[1], request.url.path
            ):
                permission_type = "write"
                break

    if permission_type is None:
        return None

    # Check if key has specific permission for allowed_vector_store_indexes
    has_permission = check_vector_store_permission(
        index_name=index_name,
        permission=permission_type,
        key_metadata=key_metadata,
        team_metadata=team_metadata,
    )

    if not has_permission:
        raise HTTPException(
            status_code=403,
            detail=f"User does not have permission to call vector store endpoint {index_name}. Ask your administrator to add the necessary permissions to your API key/Team.",
        )

    return has_permission


def is_allowed_to_call_vector_store_files_endpoint(
    provider: LlmProviders,
    vector_store_id: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Literal[True]]:
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    key_metadata = user_api_key_dict.metadata
    team_metadata = user_api_key_dict.team_metadata

    provider_config = ProviderConfigManager.get_provider_vector_store_files_config(
        provider=provider
    )
    if provider_config is None:
        return None

    provider_vector_store_endpoints = (
        provider_config.get_vector_store_file_endpoints_by_type()
    )

    permission_type: Optional[str] = None
    for endpoint in provider_vector_store_endpoints.get("read", ()):
        if request.method == endpoint[0] and _does_endpoint_match(
            endpoint[1], request.url.path
        ):
            permission_type = "read"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints.get("write", ()):
            if request.method == endpoint[0] and _does_endpoint_match(
                endpoint[1], request.url.path
            ):
                permission_type = "write"
                break

    if permission_type is None:
        return None

    has_permission = check_vector_store_permission(
        index_name=vector_store_id,
        permission=permission_type,
        key_metadata=key_metadata,
        team_metadata=team_metadata,
    )

    if not has_permission:
        raise HTTPException(
            status_code=403,
            detail=f"User does not have permission to call vector store file endpoint {vector_store_id}. Ask your administrator to add the necessary permissions to your API key/Team.",
        )

    return has_permission
