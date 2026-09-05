import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final, Literal

from fastapi import HTTPException, Request
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.ui_session_utils import (
    is_ui_session_credential,
    resolve_ui_session_team_ids,
)
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.types.utils import LlmProviders
from litellm.types.vector_stores import (
    MILVUS_ADMIN_CONFIGURED_CONNECTION,
    LiteLLM_ManagedVectorStore,
    VectorStoreIndexEndpoints,
)
from litellm.utils import ProviderConfigManager
from litellm.vector_stores.vector_store_registry import deserialize_litellm_params

MILVUS_MANAGED_CONFIGURATION_FIELDS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "custom_llm_provider",
        "litellm_credential_name",
        "milvus_transport",
        "milvus_db_name",
        "milvus_partition_names",
        "litellm_embedding_config",
        "litellm_embedding_model",
        "milvus_text_field",
    }
)
_MANAGED_VECTOR_STORE_ADAPTER: Final = TypeAdapter(LiteLLM_ManagedVectorStore)


def _normalize_litellm_params(
    vector_store: LiteLLM_ManagedVectorStore,
) -> LiteLLM_ManagedVectorStore:
    litellm_params: Final = vector_store.get("litellm_params")
    if isinstance(litellm_params, str):
        return _MANAGED_VECTOR_STORE_ADAPTER.validate_python(
            {**vector_store, "litellm_params": deserialize_litellm_params(litellm_params)}
        )
    return vector_store


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )


def normalize_vector_store_provider(custom_llm_provider: object) -> str | None:
    if not isinstance(custom_llm_provider, str) or not custom_llm_provider:
        return None
    if "/" not in custom_llm_provider:
        return custom_llm_provider
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=custom_llm_provider)
        return provider
    except Exception:  # noqa: BLE001  # provider parsing failures fall back to the explicit prefix
        return custom_llm_provider.split("/", 1)[0]


def is_milvus_grpc_connection(custom_llm_provider: object, litellm_params: object) -> bool:
    return (
        isinstance(litellm_params, dict)
        and (
            normalize_vector_store_provider(custom_llm_provider) == "milvus"
            or normalize_vector_store_provider(litellm_params.get("custom_llm_provider")) == "milvus"
        )
        and litellm_params.get("milvus_transport") == "grpc"
    )


def assert_proxy_admin_for_vector_store_index_management(
    user_api_key_dict: UserAPIKeyAuth,
    *,
    operation: Literal["create", "delete", "update", "list"] = "create",
) -> None:
    """Raise 403 unless the caller is a proxy admin."""
    if _is_proxy_admin(user_api_key_dict):
        return
    raise HTTPException(
        status_code=403,
        detail=(f"Only proxy admins can {operation} vector store indexes. Contact your LiteLLM administrator."),
    )


def assert_proxy_admin_for_user_supplied_vector_store_connection(
    custom_llm_provider: object,
    litellm_params: object,
    user_api_key_dict: UserAPIKeyAuth | None = None,
    *,
    managed: bool = False,
) -> None:
    if not is_milvus_grpc_connection(custom_llm_provider, litellm_params):
        return
    if managed:
        if isinstance(litellm_params, dict) and litellm_params.get(MILVUS_ADMIN_CONFIGURED_CONNECTION) is True:
            return
        raise HTTPException(
            status_code=403,
            detail="This managed Milvus gRPC connection must be re-saved by a proxy admin before it can be used.",
        )
    if user_api_key_dict is not None and _is_proxy_admin(user_api_key_dict):
        return
    raise HTTPException(
        status_code=403,
        detail="Only proxy admins can configure vector store connections. Contact your LiteLLM administrator.",
    )


def prepare_milvus_connection_for_persistence(
    *,
    custom_llm_provider: object,
    litellm_params: object,
    user_api_key_dict: UserAPIKeyAuth,
    existing_custom_llm_provider: object | None = None,
    existing_litellm_params: object | None = None,
    litellm_credential_name: object | None = None,
    existing_litellm_credential_name: object | None = None,
    litellm_credential_name_supplied: bool = False,
) -> dict[str, object]:  # mutable-ok: persistence requires a serializable effective-connection dict
    existing: Final = existing_litellm_params if isinstance(existing_litellm_params, dict) else MappingProxyType({})
    supplied: Final = litellm_params if isinstance(litellm_params, dict) else MappingProxyType({})
    effective: Final = {  # mutable-ok: the validated connection must be JSON-serializable for database persistence
        key: value
        for params in (existing, supplied)
        for key, value in params.items()
        if key != MILVUS_ADMIN_CONFIGURED_CONNECTION
    }
    previous_is_grpc: Final = is_milvus_grpc_connection(existing_custom_llm_provider, existing)
    effective_is_grpc: Final = is_milvus_grpc_connection(custom_llm_provider, effective)
    if not previous_is_grpc and not effective_is_grpc:
        return (  # mutable-ok: persistence requires an isolated JSON-serializable dict
            dict(supplied) if isinstance(litellm_params, dict) else dict(existing)
        )

    is_create: Final = existing_custom_llm_provider is None
    provider_changed: Final = not is_create and custom_llm_provider != existing_custom_llm_provider
    managed_configuration_changed: Final = any(
        existing.get(field) != effective.get(field) for field in MILVUS_MANAGED_CONFIGURATION_FIELDS
    )
    credential_changed: Final = litellm_credential_name_supplied and (
        litellm_credential_name != existing_litellm_credential_name
    )
    missing_marker: Final = effective_is_grpc and existing.get(MILVUS_ADMIN_CONFIGURED_CONNECTION) is not True

    if (
        is_create or provider_changed or managed_configuration_changed or credential_changed or missing_marker
    ) and not _is_proxy_admin(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins can configure vector store connections. Contact your LiteLLM administrator.",
        )

    return (
        {**effective, MILVUS_ADMIN_CONFIGURED_CONNECTION: True}  # mutable-ok: persisted JSON carries the server marker
        if effective_is_grpc
        else effective
    )


def _suffix_after_index_name(request_path: str, index_name: str) -> str | None:
    """Return the path suffix after ``/indexes/{index_name}``, or None if absent."""
    match: Final = re.search(rf"/indexes/{re.escape(index_name)}(?=$|[/?])", request_path)
    if match is None:
        return None
    return request_path[match.end() :]


def _is_vector_store_index_lifecycle_request(
    request_method: str,
    request_path: str,
    index_name: str,
) -> bool:
    """
    True when the request creates or deletes a search index itself (not documents).

    Examples (admin-only):
    - DELETE /azure_ai/indexes/my-index
    - PUT /azure_ai/indexes/my-index
    - POST /azure_ai/indexes
    """
    if request_method not in ("POST", "PUT", "DELETE", "PATCH"):
        return False

    suffix: Final = _suffix_after_index_name(request_path, index_name)
    if suffix is not None:
        # Document operations live under /indexes/{name}/docs/...
        if suffix.startswith("/docs"):
            return False
        # DELETE/PUT/PATCH on /indexes/{name} itself is index lifecycle.
        if suffix == "" or suffix.startswith("?"):
            return True

    # POST /indexes (create index at service level; no index name in path).
    normalized: Final = request_path.split("?", 1)[0].rstrip("/")
    if request_method == "POST" and normalized.endswith("/indexes"):
        return True

    return False


def _object_permission_allows_vector_store(
    object_permission: LiteLLM_ObjectPermissionTable | None,
    vector_store_id: str,
) -> bool:
    """Returns True if an object permission explicitly allowlists the vector store."""
    if object_permission is None:
        return False
    allowed: Final = object_permission.vector_stores
    if not allowed:
        return False
    return vector_store_id in allowed


async def _get_object_permission_for_id(
    object_permission_id: str | None,
) -> LiteLLM_ObjectPermissionTable | None:
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

    if vector_store.get("team_id") is None:
        return True

    return await _is_vector_store_granted(vector_store, user_api_key_dict)


async def _is_vector_store_granted(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    vector_store_id: Final = vector_store.get("vector_store_id") or ""

    key_object_permission = user_api_key_dict.object_permission
    if key_object_permission is None:
        key_object_permission = await _get_object_permission_for_id(user_api_key_dict.object_permission_id)
    if _object_permission_allows_vector_store(key_object_permission, vector_store_id):
        return True

    team_object_permission: LiteLLM_ObjectPermissionTable | None = user_api_key_dict.team_object_permission
    if team_object_permission is None:
        team_object_permission = await _get_object_permission_for_id(user_api_key_dict.team_object_permission_id)
    if _object_permission_allows_vector_store(team_object_permission, vector_store_id):
        return True

    return user_api_key_dict.team_id is not None and user_api_key_dict.team_id == vector_store.get("team_id")


async def _team_auth_context(team_id: str, user_api_key_dict: UserAPIKeyAuth) -> UserAPIKeyAuth:
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    team: Final = await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    return user_api_key_dict.model_copy(
        update=MappingProxyType(
            {
                "team_id": team_id,
                "team_object_permission": team.object_permission,
                "team_object_permission_id": team.object_permission_id,
            }
        )
    )


async def _vector_store_listing_auth_contexts(
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[UserAPIKeyAuth, ...]:
    if not is_ui_session_credential(user_api_key_dict):
        return (user_api_key_dict,)
    session_key_context: Final = user_api_key_dict.model_copy(
        update=MappingProxyType({"team_id": None, "team_object_permission": None, "team_object_permission_id": None})
    )
    team_ids: Final = await resolve_ui_session_team_ids(user_api_key_dict)
    team_contexts: Final = tuple([await _team_auth_context(team_id, user_api_key_dict) for team_id in team_ids])
    return (session_key_context, *team_contexts)


async def _is_vector_store_granted_to_any(
    vector_store: LiteLLM_ManagedVectorStore,
    auth_contexts: tuple[UserAPIKeyAuth, ...],
) -> bool:
    for auth_context in auth_contexts:
        if await _is_vector_store_granted(vector_store, auth_context):
            return True
    return False


async def filter_listable_vector_stores(
    vector_stores: Iterable[LiteLLM_ManagedVectorStore],
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[LiteLLM_ManagedVectorStore, ...]:
    """Non-admins only see stores their key, one of their teams' object_permission, or team ownership grants."""
    if _is_proxy_admin(user_api_key_dict):
        return tuple(vector_stores)

    auth_contexts: Final = await _vector_store_listing_auth_contexts(user_api_key_dict)
    return tuple([vs for vs in vector_stores if await _is_vector_store_granted_to_any(vs, auth_contexts)])


async def get_litellm_managed_vector_store(
    vector_store_id: str,
) -> LiteLLM_ManagedVectorStore | None:
    """
    Resolve a LiteLLM-managed vector store from the registry or shared cache.

    Provider-native vector store IDs will not be present in either location and
    return None, preserving direct provider behavior while still protecting
    LiteLLM-managed multi-tenant stores.
    """
    if not vector_store_id:
        return None

    if litellm.vector_store_registry is not None:
        try:
            vector_store: Final = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=vector_store_id
            )
            if vector_store is not None:
                return _normalize_litellm_params(vector_store)
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to resolve vector store id=%s from registry: %s",
                vector_store_id,
                e,
            )
            raise HTTPException(
                status_code=500,
                detail="Unable to validate vector store access",
            ) from e

    try:
        from litellm.proxy.auth.auth_checks import (
            get_managed_vector_store_rows_by_uuids,
        )
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            return None
        rows: Final = await get_managed_vector_store_rows_by_uuids(
            uuids=[vector_store_id],
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if not rows:
            return None
        return _normalize_litellm_params(LiteLLM_ManagedVectorStore(**rows[0].model_dump()))
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to resolve vector store id=%s from shared cache: %s",
            vector_store_id,
            e,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to validate vector store access",
        ) from e


async def assert_user_can_access_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
    detail: str = "Access denied: You do not have permission to access this vector store",
) -> None:
    """Raise 403 unless the caller can access the resolved vector store."""
    if not await can_user_access_vector_store(vector_store, user_api_key_dict):
        raise HTTPException(status_code=403, detail=detail)


async def assert_user_can_access_vector_store_id(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    detail: str = "Access denied: You do not have permission to access this vector store",
) -> LiteLLM_ManagedVectorStore | None:
    """
    Resolve a managed vector store id and enforce ownership if it exists.

    Unknown ids are treated as provider-native ids and are not rejected here.
    """
    vector_store: Final = await get_litellm_managed_vector_store(vector_store_id=vector_store_id)
    if vector_store is not None:
        await assert_user_can_access_vector_store(
            vector_store=vector_store,
            user_api_key_dict=user_api_key_dict,
            detail=detail,
        )
    return vector_store


def _does_endpoint_match(endpoint_path: str, request_path: str) -> bool:
    if endpoint_path in request_path:
        return True
    if "{" in endpoint_path:
        prefix: Final = endpoint_path.split("{", 1)[0]
        if prefix and prefix in request_path:
            return True
    return False


def check_vector_store_permission(
    index_name: str,
    permission: str,
    key_metadata: Mapping[str, object] | None,
    team_metadata: Mapping[str, object] | None,
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
                if isinstance(index_permissions, list) and permission in index_permissions:
                    return True

    return False


def _index_lifecycle_operation(request_method: str) -> Literal["create", "delete", "update"]:
    if request_method == "DELETE":
        return "delete"
    if request_method in ("PUT", "PATCH"):
        return "update"
    return "create"


def _matching_index_permission(
    endpoints: VectorStoreIndexEndpoints,
    request_method: str,
    request_path: str,
) -> Literal["read", "write"] | None:
    if any(
        request_method == method and _does_endpoint_match(path, request_path) for method, path in endpoints["write"]
    ):
        return "write"
    if any(request_method == method and _does_endpoint_match(path, request_path) for method, path in endpoints["read"]):
        return "read"
    return None


def is_allowed_to_call_vector_store_endpoint(
    provider: LlmProviders,
    index_name: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Literal[True] | None:
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
    key_metadata: Final = user_api_key_dict.metadata
    team_metadata: Final = user_api_key_dict.team_metadata

    provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(provider=provider)
    if provider_config is None:
        return None

    provider_vector_store_endpoints: Final = provider_config.get_vector_store_endpoints_by_type()

    # Inline import — auth_utils participates in a proxy import cycle.
    from litellm.proxy.auth.auth_utils import get_request_route  # noqa: PLC0415

    request_route: Final = get_request_route(request)

    if _is_vector_store_index_lifecycle_request(
        request_method=request.method,
        request_path=request_route,
        index_name=index_name,
    ):
        assert_proxy_admin_for_vector_store_index_management(
            user_api_key_dict,
            operation=_index_lifecycle_operation(request.method),
        )
        return True

    permission_type: Final = _matching_index_permission(
        provider_vector_store_endpoints,
        request.method,
        request_route,
    )
    if permission_type is None:
        raise HTTPException(
            status_code=403,
            detail=(
                f"User does not have permission to call vector store endpoint "
                f"{index_name}. Ask your administrator to add the necessary "
                "permissions to your API key/Team."
            ),
        )

    # Check if key has specific permission for allowed_vector_store_indexes
    has_permission: Final = check_vector_store_permission(
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
) -> Literal[True] | None:
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    key_metadata: Final = user_api_key_dict.metadata
    team_metadata: Final = user_api_key_dict.team_metadata

    provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(provider=provider)
    if provider_config is None:
        return None

    provider_vector_store_endpoints: Final = provider_config.get_vector_store_file_endpoints_by_type()

    # Inline import — auth_utils participates in a proxy import cycle.
    from litellm.proxy.auth.auth_utils import get_request_route  # noqa: PLC0415

    request_route: Final = get_request_route(request)

    permission_type: str | None = None
    for endpoint in provider_vector_store_endpoints.get("write", ()):
        if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
            permission_type = "write"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints.get("read", ()):
            if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
                permission_type = "read"
                break

    if permission_type is None:
        return None

    has_permission: Final = check_vector_store_permission(
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
