import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
    user_can_access_resource_owner,
)
from litellm.proxy.container_endpoints.ownership_store import (
    CONTAINER_OBJECT_PURPOSE,
    ContainerOwnershipStore,
)
from litellm.responses.utils import ResponsesAPIRequestUtils

MAX_IN_MEMORY_CONTAINER_OWNERS = 10000
_IN_MEMORY_CONTAINER_OWNERS: "OrderedDict[str, str]" = OrderedDict()

# Short-lived cache keeps every container access check from hitting the DB
# (`_get_container_owner` is invoked on retrieve / delete / list / file-content
# paths). Mirrors the `_byok_cred_cache` pattern in mcp_server/server.py:
# (value, monotonic_timestamp) tuples, TTL'd, LRU-evicted at capacity,
# invalidated by writes. A ``None`` value caches "untracked" so repeated
# negative lookups also avoid DB.
_CONTAINER_OWNER_CACHE: "OrderedDict[str, Tuple[Optional[str], float]]" = OrderedDict()
_CONTAINER_OWNER_CACHE_TTL = 60  # seconds
_CONTAINER_OWNER_CACHE_MAX_SIZE = 10000


def _read_container_owner_cache(model_object_id: str) -> Tuple[bool, Optional[str]]:
    """Return (hit, value). hit=False means caller must consult the DB."""
    entry = _CONTAINER_OWNER_CACHE.get(model_object_id)
    if entry is None:
        return False, None
    value, timestamp = entry
    if time.monotonic() - timestamp > _CONTAINER_OWNER_CACHE_TTL:
        _CONTAINER_OWNER_CACHE.pop(model_object_id, None)
        return False, None
    _CONTAINER_OWNER_CACHE.move_to_end(model_object_id)
    return True, value


def _write_container_owner_cache(model_object_id: str, owner: Optional[str]) -> None:
    # LRU eviction (popitem(last=False)) instead of full ``clear()`` — a
    # full clear at capacity converts a steady-state cached workload into
    # a periodic full-DB-load oscillation as the cache repopulates from
    # zero and clears again.
    if model_object_id in _CONTAINER_OWNER_CACHE:
        _CONTAINER_OWNER_CACHE.move_to_end(model_object_id)
    _CONTAINER_OWNER_CACHE[model_object_id] = (owner, time.monotonic())
    while len(_CONTAINER_OWNER_CACHE) > _CONTAINER_OWNER_CACHE_MAX_SIZE:
        _CONTAINER_OWNER_CACHE.popitem(last=False)


def _invalidate_container_owner_cache(model_object_id: str) -> None:
    """Drop a cache entry after a write so the next read sees the new owner."""
    _CONTAINER_OWNER_CACHE.pop(model_object_id, None)


def _remember_container_owner(model_object_id: str, owner: str) -> None:
    existing_owner = _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)
    if existing_owner is not None:
        _IN_MEMORY_CONTAINER_OWNERS.move_to_end(model_object_id)
    _IN_MEMORY_CONTAINER_OWNERS[model_object_id] = owner
    while len(_IN_MEMORY_CONTAINER_OWNERS) > MAX_IN_MEMORY_CONTAINER_OWNERS:
        _IN_MEMORY_CONTAINER_OWNERS.popitem(last=False)


def _container_model_object_id(
    original_container_id: str,
    custom_llm_provider: str,
) -> str:
    return f"{CONTAINER_OBJECT_PURPOSE}:{custom_llm_provider}:{original_container_id}"


def decode_container_id_for_ownership(
    container_id: str,
    custom_llm_provider: str,
) -> Tuple[str, str]:
    decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
    original_container_id = decoded.get("response_id", container_id)
    decoded_provider = decoded.get("custom_llm_provider")
    if decoded_provider and custom_llm_provider == "openai":
        custom_llm_provider = decoded_provider
    return original_container_id, custom_llm_provider


def get_container_forwarding_params(
    container_id: str,
    original_container_id: str,
    custom_llm_provider: str,
) -> Dict[str, str]:
    params = {
        "container_id": original_container_id,
        "custom_llm_provider": custom_llm_provider,
    }
    decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
    model_id = decoded.get("model_id")
    if isinstance(model_id, str) and model_id:
        params["model_id"] = model_id
    return params


def _get_response_id(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        value = response.get("id")
    else:
        value = getattr(response, "id", None)
    return value if isinstance(value, str) else None


def _dump_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return {"id": _get_response_id(response)}


async def _get_prisma_client():
    from litellm.proxy.proxy_server import prisma_client

    return prisma_client


async def record_container_owner(
    response: Any,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Any:
    container_id = _get_response_id(response)
    owner = get_primary_resource_owner_scope(user_api_key_dict)
    if is_proxy_admin(user_api_key_dict) and (container_id is None or owner is None):
        return response
    if container_id is None:
        verbose_proxy_logger.warning(
            "Skipping container ownership tracking because provider response has no id"
        )
        return response
    if owner is None:
        # Caller has no identity (no user_id / team_id / org_id / api_key /
        # token) we can stamp on the row. Recording with a placeholder
        # would collapse every such caller into a single shared owner —
        # the cross-tenant data-access primitive we explicitly avoid.
        # Reject with 403 rather than fall back to a sentinel.
        raise HTTPException(
            status_code=403,
            detail="Unable to record container ownership: caller has no identity scope.",
        )

    original_container_id, resolved_provider = decode_container_id_for_ownership(
        container_id,
        custom_llm_provider,
    )
    model_object_id = _container_model_object_id(
        original_container_id,
        resolved_provider,
    )
    file_object = _dump_response(response)
    file_object["custom_llm_provider"] = resolved_provider
    file_object["provider_container_id"] = original_container_id

    try:
        prisma_client = await _get_prisma_client()
        if prisma_client is None:
            existing_owner = _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)
            if existing_owner is not None and not user_can_access_resource_owner(
                existing_owner, user_api_key_dict
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
            _remember_container_owner(model_object_id, owner)
            _invalidate_container_owner_cache(model_object_id)
            return response

        store = ContainerOwnershipStore(prisma_client)
        existing = await store.find_by_model_object_id(model_object_id)
        if existing is not None:
            if getattr(existing, "file_purpose", None) != CONTAINER_OBJECT_PURPOSE:
                raise HTTPException(status_code=500, detail="Unable to track container")
            if not user_can_access_resource_owner(
                getattr(existing, "created_by", None), user_api_key_dict
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
            await store.update_owner_record(
                model_object_id=model_object_id,
                data={
                    "unified_object_id": container_id,
                    "file_object": file_object,
                    "updated_by": owner,
                },
            )
        else:
            await store.create_owner_record(
                data={
                    "unified_object_id": container_id,
                    "model_object_id": model_object_id,
                    "file_object": file_object,
                    "file_purpose": CONTAINER_OBJECT_PURPOSE,
                    "created_by": owner,
                    "updated_by": owner,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to persist container ownership for container_id=%s; "
            "falling back to in-process tracking: %s",
            model_object_id,
            e,
        )
        existing_owner = _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)
        if existing_owner is not None and not user_can_access_resource_owner(
            existing_owner, user_api_key_dict
        ):
            raise HTTPException(status_code=403, detail="Forbidden")
        _remember_container_owner(model_object_id, owner)

    _invalidate_container_owner_cache(model_object_id)
    return response


async def _get_container_owner(
    original_container_id: str,
    custom_llm_provider: str,
) -> Optional[str]:
    model_object_id = _container_model_object_id(
        original_container_id,
        custom_llm_provider,
    )

    cached_hit, cached_value = _read_container_owner_cache(model_object_id)
    if cached_hit:
        return cached_value

    try:
        prisma_client = await _get_prisma_client()
        if prisma_client is None:
            owner = _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)
            _write_container_owner_cache(model_object_id, owner)
            return owner

        owner = await ContainerOwnershipStore(prisma_client).get_owner(model_object_id)
        if owner is None:
            owner = _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)
        _write_container_owner_cache(model_object_id, owner)
        return owner
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to load container ownership for container_id=%s; "
            "falling back to in-process tracking: %s",
            model_object_id,
            e,
        )
        # Don't cache transient DB errors — let the next request retry.
        return _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)


async def assert_user_can_access_container(
    container_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Tuple[str, str]:
    original_container_id, resolved_provider = decode_container_id_for_ownership(
        container_id,
        custom_llm_provider,
    )

    if is_proxy_admin(user_api_key_dict):
        return original_container_id, resolved_provider

    # Untracked containers (no ownership row) are admin-only. Pre-isolation
    # rows that pre-date this enforcement need the admin to either re-create
    # via the now-tracked flow or explicitly assign ``created_by`` on the
    # ``litellm_managedobjecttable`` row.
    owner = await _get_container_owner(original_container_id, resolved_provider)
    if not user_can_access_resource_owner(owner, user_api_key_dict):
        raise HTTPException(status_code=403, detail="Forbidden")

    return original_container_id, resolved_provider


def _get_container_list_data(response: Any) -> Optional[List[Any]]:
    if response is None:
        return None
    if isinstance(response, dict):
        data = response.get("data")
    else:
        data = getattr(response, "data", None)
    return data if isinstance(data, list) else None


def _set_container_list_data(
    response: Any, data: List[Any], removed_filtered_items: bool = False
) -> Any:
    if isinstance(response, dict):
        response["data"] = data
        if data:
            response["first_id"] = _get_response_id(data[0])
            response["last_id"] = _get_response_id(data[-1])
        else:
            response["first_id"] = None
            response["last_id"] = None
            response["has_more"] = False
        if removed_filtered_items:
            response["has_more"] = False
        return response

    response.data = data
    response.first_id = _get_response_id(data[0]) if data else None
    response.last_id = _get_response_id(data[-1]) if data else None
    if not data and hasattr(response, "has_more"):
        response.has_more = False
    if removed_filtered_items and hasattr(response, "has_more"):
        response.has_more = False
    return response


async def _get_allowed_container_ids(
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Set[str]:
    owner_scopes = get_resource_owner_scopes(user_api_key_dict)
    if not owner_scopes:
        return set()

    in_memory_allowed_ids = {
        model_object_id
        for model_object_id, owner in _IN_MEMORY_CONTAINER_OWNERS.items()
        if owner in owner_scopes
    }
    try:
        prisma_client = await _get_prisma_client()
        if prisma_client is None:
            return in_memory_allowed_ids

        db_allowed_ids = await ContainerOwnershipStore(
            prisma_client
        ).list_model_object_ids_for_owners(
            owner_scopes=owner_scopes,
        )
        return in_memory_allowed_ids | db_allowed_ids
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to load allowed container ids; falling back to in-process "
            "tracking: %s",
            e,
        )
        return in_memory_allowed_ids


async def filter_container_list_response(
    response: Any,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Any:
    if is_proxy_admin(user_api_key_dict):
        return response

    data = _get_container_list_data(response)
    if data is None:
        return response

    allowed_container_ids = await _get_allowed_container_ids(
        user_api_key_dict,
        custom_llm_provider,
    )
    filtered: List[Any] = []
    for item in data:
        container_id = _get_response_id(item)
        if container_id is None:
            continue
        original_container_id, resolved_provider = decode_container_id_for_ownership(
            container_id,
            custom_llm_provider,
        )
        if (
            _container_model_object_id(original_container_id, resolved_provider)
            in allowed_container_ids
        ):
            filtered.append(item)

    return _set_container_list_data(
        response,
        filtered,
        removed_filtered_items=len(filtered) != len(data),
    )
