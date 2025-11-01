from typing import Any, Dict, Optional

from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


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
) -> Optional[bool]:
    """
    Check if the user is allowed to call the vector store endpoint.

    Cover:
    1. Creating a vector store index
    2. Reading a vector store index (Search / List / Get)
    """

    # check what allowed permissions are for the key
    key_metadata = user_api_key_dict.metadata
    team_metadata = user_api_key_dict.team_metadata

    provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=provider
    )
    if provider_config is None:
        return False

    provider_vector_store_endpoints = (
        provider_config.get_vector_store_endpoints_by_type()
    )

    # Determine the permission type based on the request
    permission_type = None
    for endpoint in provider_vector_store_endpoints["read"]:
        if request.method == endpoint[0] and endpoint[1] in request.url.path:
            permission_type = "read"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints["write"]:
            if request.method == endpoint[0] and endpoint[1] in request.url.path:
                permission_type = "write"
                break

    if permission_type is None:
        return False

    # Check if key has specific permission for allowed_vector_store_indexes
    has_permission = check_vector_store_permission(
        index_name=index_name,
        permission=permission_type,
        key_metadata=key_metadata,
        team_metadata=team_metadata,
    )

    return has_permission
