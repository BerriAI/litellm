from typing import Any, Dict, Literal, Optional

from fastapi import HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.types.utils import LlmProviders
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.utils import ProviderConfigManager


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )


def _object_permission_allows_vector_store(
    object_permission: Optional[LiteLLM_ObjectPermissionTable],
    vector_store_id: str,
) -> bool:
    """Returns True if an object permission explicitly allowlists the vector store."""
    if object_permission is None:
        return False
    allowed = object_permission.vector_stores
    if not allowed:
        return False
    return vector_store_id in allowed


async def _get_object_permission_for_id(
    object_permission_id: Optional[str],
) -> Optional[LiteLLM_ObjectPermissionTable]:
    """Load an object permission record by id, using the shared cache/DB helper."""
    if not object_permission_id:
        return None

    from litellm.proxy.auth.auth_checks import get_object_permission
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        return None

    try:
        return await get_object_permission(
            object_permission_id=object_permission_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            "Failed to load object_permission id=%s: %s",
            object_permission_id,
            e,
        )
        return None


async def can_user_access_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """
    Returns True if the caller is allowed to access this managed vector store.

    Access is granted (first match wins) when any of the following is true:
    1. The caller's role is PROXY_ADMIN.
    2. The vector store has no team_id (legacy behavior - accessible to all).
    3. The caller's key-level object_permission.vector_stores explicitly lists
       this vector store id.
    4. The caller's team-level object_permission.vector_stores explicitly lists
       this vector store id.
    5. The caller's team_id matches the vector store's team_id.

    Otherwise access is denied.
    """
    if _is_proxy_admin(user_api_key_dict):
        return True

    vector_store_team_id = vector_store.get("team_id")
    if vector_store_team_id is None:
        return True

    vector_store_id = vector_store.get("vector_store_id") or ""

    key_object_permission = user_api_key_dict.object_permission
    if key_object_permission is None:
        key_object_permission = await _get_object_permission_for_id(
            user_api_key_dict.object_permission_id
        )
    if _object_permission_allows_vector_store(key_object_permission, vector_store_id):
        return True

    team_object_permission: Optional[LiteLLM_ObjectPermissionTable] = (
        user_api_key_dict.team_object_permission
    )
    if team_object_permission is None:
        team_object_permission = await _get_object_permission_for_id(
            user_api_key_dict.team_object_permission_id
        )
    if _object_permission_allows_vector_store(team_object_permission, vector_store_id):
        return True

    if (
        user_api_key_dict.team_id is not None
        and user_api_key_dict.team_id == vector_store_team_id
    ):
        return True

    return False


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
