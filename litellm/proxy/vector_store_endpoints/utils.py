import json
import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate
from types import MappingProxyType
from typing import Final, Literal

import httpx
from fastapi import HTTPException, Request
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.mcp_server.ui_session_utils import (
    is_ui_session_credential,
    resolve_ui_session_team_ids,
)
from litellm.proxy.openai_files_endpoints.common_utils import managed_file_ids_for_provider_file_ids
from litellm.types.utils import LlmProviders
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore, VectorStoreIndexEndpoints
from litellm.utils import ProviderConfigManager

_PLAIN_PROVIDER_PATH: Final = re.compile(r"[\w\-/:.$]+")
_JSON_OBJECT: Final = TypeAdapter(dict[str, object], config=ConfigDict(hide_input_in_errors=True))


def normalize_litellm_params(
    vector_store: LiteLLM_ManagedVectorStore,
) -> LiteLLM_ManagedVectorStore:
    litellm_params: Final = vector_store.get("litellm_params")
    if isinstance(litellm_params, str):
        normalized: Final = LiteLLM_ManagedVectorStore(**dict(vector_store))
        try:
            normalized["litellm_params"] = _JSON_OBJECT.validate_python(json.loads(litellm_params))
        except (TypeError, ValueError):
            normalized["litellm_params"] = {}
        return normalized
    return vector_store


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )


def _key_metadata(user_api_key_dict: UserAPIKeyAuth) -> Mapping[str, object]:
    return user_api_key_dict.metadata


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

    A dashboard session credential is evaluated against the same effective
    contexts as listing (its own grants plus each real team of the user).
    Otherwise access is denied.
    """
    if _is_proxy_admin(user_api_key_dict):
        return True

    if vector_store.get("team_id") is None:
        return True

    auth_contexts: Final = await _vector_store_auth_contexts(user_api_key_dict)
    return await _is_vector_store_granted_to_any(vector_store, auth_contexts)


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


async def _vector_store_auth_contexts(
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


async def _group_store_ids_for_context(user_api_key_dict: UserAPIKeyAuth) -> frozenset[str]:
    """Reads only add access here, so a group or team that cannot be read grants nothing."""
    from litellm.proxy.auth.auth_checks import get_team_object, vector_stores_opened_by_groups
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    team_id: Final = user_api_key_dict.team_id
    try:
        team: Final = (
            await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_id is not None
            else None
        )
        return await vector_stores_opened_by_groups(user_api_key_dict, team)
    except Exception as e:  # noqa: BLE001  # grants nothing, per the docstring
        verbose_proxy_logger.debug("Could not read access groups for vector store access: %s", e)
        return frozenset()


async def _group_granted_vector_store_ids(auth_contexts: tuple[UserAPIKeyAuth, ...]) -> frozenset[str]:
    return frozenset[str]().union(*[await _group_store_ids_for_context(context) for context in auth_contexts])


def _is_group_granted(vector_store: LiteLLM_ManagedVectorStore, group_store_ids: frozenset[str]) -> bool:
    """Exact ids only: a provider id can belong to several registered stores, on different backends."""
    return (vector_store.get("vector_store_id") or "") in group_store_ids


async def can_user_read_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """Using or reading a store also counts access groups. Managing one does not."""
    if await can_user_access_vector_store(vector_store, user_api_key_dict):
        return True
    auth_contexts: Final = await _vector_store_auth_contexts(user_api_key_dict)
    return _is_group_granted(vector_store, await _group_granted_vector_store_ids(auth_contexts))


async def filter_listable_vector_stores(
    vector_stores: Iterable[LiteLLM_ManagedVectorStore],
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[LiteLLM_ManagedVectorStore, ...]:
    """Non-admins only see stores their key, a team's object_permission, team ownership or access groups grant."""
    if _is_proxy_admin(user_api_key_dict):
        return tuple(vector_stores)

    auth_contexts: Final = await _vector_store_auth_contexts(user_api_key_dict)
    group_store_ids: Final = await _group_granted_vector_store_ids(auth_contexts)
    return tuple(
        [
            vs
            for vs in vector_stores
            if _is_group_granted(vs, group_store_ids) or await _is_vector_store_granted_to_any(vs, auth_contexts)
        ]
    )


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
                return normalize_litellm_params(vector_store)
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
        return normalize_litellm_params(LiteLLM_ManagedVectorStore(**rows[0].model_dump()))
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


async def assert_user_can_change_vector_store_id(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> LiteLLM_ManagedVectorStore | None:
    """Group grants never change a store, managed or not, so the caller's own grants decide first."""
    from litellm.proxy.auth.auth_checks import assert_own_grants_allow_vector_store

    await assert_own_grants_allow_vector_store(user_api_key_dict, vector_store_id)
    return await assert_user_can_access_vector_store_id(vector_store_id, user_api_key_dict)


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


async def assert_user_can_read_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    if not await can_user_read_vector_store(vector_store, user_api_key_dict):
        raise HTTPException(
            status_code=403, detail="Access denied: You do not have permission to access this vector store"
        )


async def assert_user_can_read_vector_store_id(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> LiteLLM_ManagedVectorStore | None:
    vector_store: Final = await get_litellm_managed_vector_store(vector_store_id=vector_store_id)
    if vector_store is not None:
        await assert_user_can_read_vector_store(vector_store, user_api_key_dict)
    return vector_store


def _store_marker_count(segments: Sequence[str], marker: str) -> int:
    """How many stores the path names: a segment ending in the marker, as an unanchored extractor matches it,
    followed by an id, which may itself contain the marker."""

    def opens_a_store(after_marker: bool, segment_and_next: tuple[str, str | None]) -> bool:
        segment, following = segment_and_next
        return segment.endswith(marker) and following is not None and not after_marker

    return sum(accumulate(zip(segments, (*segments[1:], None)), opens_a_store, initial=False))


def _is_provider_read_request(
    endpoints: VectorStoreIndexEndpoints, method: str, endpoint: str, store_segment: str | None
) -> bool:
    """A substring match could hide a write, so a read needs a plain path the forwarder keeps as is, ending with a
    read entry. A GET only has to contain one."""
    path: Final = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if _PLAIN_PROVIDER_PATH.fullmatch(path) is None or not {"", ".", ".."}.isdisjoint(path.split("/")[1:]):
        return False
    if httpx.URL(endpoint).path != endpoint:
        return False
    if store_segment is not None and _store_marker_count(path.split("/"), store_segment) > 1:
        return False
    if any(write_method == method and write_path in path for write_method, write_path in endpoints["write"]):
        return False
    return any(
        read_method == method and (path.endswith(read_path) or (method == "GET" and read_path in path))
        for read_method, read_path in endpoints["read"]
    )


async def assert_user_can_call_vector_store_passthrough(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
    provider_config: BaseVectorStoreConfig | None,
    method: str,
    endpoint: str,
    store_segment: str | None = None,
) -> None:
    """Pass-through routes also carry writes, so groups count only on a read the provider's table names.
    A path naming a second store through `store_segment` is never a read, since the gate checked only the first."""
    if provider_config is not None and _is_provider_read_request(
        provider_config.get_vector_store_endpoints_by_type(), method, endpoint, store_segment
    ):
        await assert_user_can_read_vector_store(vector_store, user_api_key_dict)
        return
    await assert_user_can_access_vector_store(vector_store, user_api_key_dict)


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


def vector_store_endpoint_forbidden(index_name: str) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail=(
            f"User does not have permission to call vector store endpoint {index_name}. "
            "Ask your administrator to add the necessary permissions to your API key/Team."
        ),
    )


def has_vector_store_index_access(
    user_api_key_dict: UserAPIKeyAuth, index_name: str, permission: Literal["read", "write"] | None
) -> bool:
    return _is_proxy_admin(user_api_key_dict) or (
        permission is not None
        and check_vector_store_permission(
            index_name, permission, _key_metadata(user_api_key_dict), user_api_key_dict.team_metadata
        )
    )


def assert_vector_store_index_access(
    user_api_key_dict: UserAPIKeyAuth, index_name: str, permission: Literal["read", "write"] | None
) -> None:
    if not has_vector_store_index_access(user_api_key_dict, index_name, permission):
        raise vector_store_endpoint_forbidden(index_name)


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
    key_metadata: Final = _key_metadata(user_api_key_dict)
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
        operation_label: Literal["create", "delete", "update"] = "create"
        if request.method == "DELETE":
            operation_label = "delete"
        elif request.method in ("PUT", "PATCH"):
            operation_label = "update"
        assert_proxy_admin_for_vector_store_index_management(
            user_api_key_dict,
            operation=operation_label,
        )
        return True

    # Writes are classified before reads so a path matching both patterns
    # requires the stronger grant (e.g. the azure batch write on an index
    # named "analyze*" also contains the "/analyze" read fragment)
    permission_type = None
    for endpoint in provider_vector_store_endpoints["write"]:
        if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
            permission_type = "write"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints["read"]:
            if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
                permission_type = "read"
                break

    if permission_type is None:
        raise vector_store_endpoint_forbidden(index_name)

    # Check if key has specific permission for allowed_vector_store_indexes
    has_permission: Final = check_vector_store_permission(
        index_name=index_name,
        permission=permission_type,
        key_metadata=key_metadata,
        team_metadata=team_metadata,
    )

    if not has_permission:
        raise vector_store_endpoint_forbidden(index_name)

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

    key_metadata: Final = _key_metadata(user_api_key_dict)
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


def _search_hit_file_id(hit: object) -> str | None:
    fields: Final = as_str_mapping(hit)
    file_id: Final = None if fields is None else fields.get("file_id")
    return file_id if isinstance(file_id, str) else None


def _with_managed_search_file_id(hit: object, id_map: Mapping[str, str]) -> object:
    fields: Final = as_str_mapping(hit)
    file_id: Final = _search_hit_file_id(hit)
    if fields is None or file_id is None or file_id not in id_map:
        return hit
    return {**fields, "file_id": id_map[file_id]}


async def with_managed_search_file_ids(
    search_page: Mapping[str, object],
    managed_files_obj: object | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> Mapping[str, object]:
    hits: Final = as_object_list(search_page.get("data"))
    if hits is None:
        return search_page
    id_map: Final = await managed_file_ids_for_provider_file_ids(
        tuple(file_id for hit in hits if (file_id := _search_hit_file_id(hit)) is not None),
        managed_files_obj,
        user_api_key_dict,
    )
    if not id_map:
        return search_page
    return {**search_page, "data": [_with_managed_search_file_id(hit, id_map) for hit in hits]}
