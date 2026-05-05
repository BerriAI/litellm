import json
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
    user_can_access_resource_owner,
)
from litellm.responses.utils import ResponsesAPIRequestUtils

CONTAINER_OBJECT_PURPOSE = "container"

# 60s LRU/TTL cache absorbs every container access check before it reaches
# Prisma. ``_NEGATIVE_OWNER_SENTINEL`` lets us cache a true "untracked"
# answer so repeated misses also avoid the DB — ``InMemoryCache`` returns
# ``None`` indistinguishably for "miss" and "cached as None".
_NEGATIVE_OWNER_SENTINEL = "__litellm_container_no_owner__"
_CONTAINER_OWNER_CACHE = InMemoryCache(max_size_in_memory=10000, default_ttl=60)

# Per-caller-scope cache for ``GET /v1/containers`` list filtering. Without
# this, every list call issues a fresh ``find_many`` against
# ``litellm_managedobjecttable``. The cache key is the sorted owner-scope
# tuple — different keys for the same user share the same allow-set, but
# different users with different scopes get disjoint cache entries.
_ALLOWED_CONTAINER_IDS_CACHE = InMemoryCache(max_size_in_memory=2048, default_ttl=60)


def _allowed_container_ids_cache_key(owner_scopes: List[str]) -> str:
    """JSON-encode the sorted scope list — using a separator like ``|``
    would collide for any tenant whose user_id / team_id / org_id /
    api_key happens to contain the separator. JSON quoting escapes
    every separator that matters."""
    return json.dumps(sorted(owner_scopes))


def _container_model_object_id(
    original_container_id: str, custom_llm_provider: str
) -> str:
    return f"{CONTAINER_OBJECT_PURPOSE}:{custom_llm_provider}:{original_container_id}"


def decode_container_id_for_ownership(
    container_id: str, custom_llm_provider: str
) -> Tuple[str, str]:
    decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
    original_container_id = decoded.get("response_id", container_id)
    decoded_provider = decoded.get("custom_llm_provider")
    if decoded_provider and custom_llm_provider == "openai":
        custom_llm_provider = decoded_provider
    return original_container_id, custom_llm_provider


def get_container_forwarding_params(
    container_id: str, original_container_id: str, custom_llm_provider: str
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
    if container_id is None:
        verbose_proxy_logger.warning(
            "Skipping container ownership tracking because provider response has no id"
        )
        return response
    owner = get_primary_resource_owner_scope(user_api_key_dict)
    if owner is None:
        # Admins with identity (the common path: master-key auth populates
        # ``user_id`` + ``api_key``) flow through the normal record path
        # below so admin-created containers are still tracked. Truly
        # identity-less admins (no user_id / team_id / org_id / api_key /
        # token) can't be uniquely stamped on the row — stamping a
        # placeholder would collapse every such caller into a shared
        # owner, the cross-tenant primitive we explicitly avoid.
        raise HTTPException(
            status_code=403,
            detail="Unable to record container ownership: caller has no identity scope.",
        )

    original_container_id, resolved_provider = decode_container_id_for_ownership(
        container_id, custom_llm_provider
    )
    model_object_id = _container_model_object_id(
        original_container_id, resolved_provider
    )
    file_object = _dump_response(response)
    file_object["custom_llm_provider"] = resolved_provider
    file_object["provider_container_id"] = original_container_id

    prisma_client = await _get_prisma_client()
    if prisma_client is None:
        verbose_proxy_logger.warning(
            "Skipping container ownership tracking because prisma_client is None"
        )
        return response

    table = prisma_client.db.litellm_managedobjecttable
    existing = await table.find_unique(where={"model_object_id": model_object_id})
    if existing is not None:
        if getattr(existing, "file_purpose", None) != CONTAINER_OBJECT_PURPOSE:
            raise HTTPException(status_code=500, detail="Unable to track container")
        if not user_can_access_resource_owner(
            getattr(existing, "created_by", None), user_api_key_dict
        ):
            raise HTTPException(status_code=403, detail="Forbidden")
        await table.update(
            where={"model_object_id": model_object_id},
            data={
                "unified_object_id": container_id,
                "file_object": file_object,
                "updated_by": owner,
            },
        )
    else:
        await table.create(
            data={
                "unified_object_id": container_id,
                "model_object_id": model_object_id,
                "file_object": file_object,
                "file_purpose": CONTAINER_OBJECT_PURPOSE,
                "created_by": owner,
                "updated_by": owner,
            }
        )

    _CONTAINER_OWNER_CACHE.set_cache(model_object_id, owner)
    # Drop the caller's own list-cache entry so the just-created container
    # shows up on their next ``GET /v1/containers``. Other callers with
    # disjoint scope tuples have their own entries; intersecting-scope
    # tuples self-correct on the 60s TTL.
    caller_scopes = get_resource_owner_scopes(user_api_key_dict)
    if caller_scopes:
        _ALLOWED_CONTAINER_IDS_CACHE.delete_cache(
            _allowed_container_ids_cache_key(caller_scopes)
        )
    return response


async def _get_container_owner(
    original_container_id: str, custom_llm_provider: str
) -> Optional[str]:
    model_object_id = _container_model_object_id(
        original_container_id, custom_llm_provider
    )

    cached = _CONTAINER_OWNER_CACHE.get_cache(model_object_id)
    if cached == _NEGATIVE_OWNER_SENTINEL:
        return None
    if cached is not None:
        return cached

    prisma_client = await _get_prisma_client()
    if prisma_client is None:
        return None

    row = await prisma_client.db.litellm_managedobjecttable.find_first(
        where={
            "model_object_id": model_object_id,
            "file_purpose": CONTAINER_OBJECT_PURPOSE,
        }
    )
    owner = getattr(row, "created_by", None) if row is not None else None
    _CONTAINER_OWNER_CACHE.set_cache(
        model_object_id, owner if owner is not None else _NEGATIVE_OWNER_SENTINEL
    )
    return owner


async def assert_user_can_access_container(
    container_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Tuple[str, str]:
    original_container_id, resolved_provider = decode_container_id_for_ownership(
        container_id, custom_llm_provider
    )

    if is_proxy_admin(user_api_key_dict):
        return original_container_id, resolved_provider

    # Untracked rows (no ownership) are admin-only. Pre-isolation rows
    # that pre-date this enforcement need an admin to either re-create
    # via the now-tracked flow or assign ``created_by`` on the row.
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
) -> Set[str]:
    owner_scopes = get_resource_owner_scopes(user_api_key_dict)
    if not owner_scopes:
        return set()

    cache_key = _allowed_container_ids_cache_key(owner_scopes)
    cached = _ALLOWED_CONTAINER_IDS_CACHE.get_cache(cache_key)
    if cached is not None:
        return set(cached)

    prisma_client = await _get_prisma_client()
    if prisma_client is None:
        return set()

    rows = await prisma_client.db.litellm_managedobjecttable.find_many(
        where={
            "file_purpose": CONTAINER_OBJECT_PURPOSE,
            "created_by": {"in": owner_scopes},
        }
    )
    allowed_ids = {
        row.model_object_id
        for row in rows
        if getattr(row, "model_object_id", None) is not None
    }
    # ``InMemoryCache.get_cache`` attempts ``json.loads`` on the stored
    # value; passing a set would round-trip through that path
    # unnecessarily. Store as a list and rehydrate above.
    _ALLOWED_CONTAINER_IDS_CACHE.set_cache(cache_key, list(allowed_ids))
    return allowed_ids


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

    allowed_container_ids = await _get_allowed_container_ids(user_api_key_dict)
    filtered: List[Any] = []
    for item in data:
        container_id = _get_response_id(item)
        if container_id is None:
            continue
        original_container_id, resolved_provider = decode_container_id_for_ownership(
            container_id, custom_llm_provider
        )
        if (
            _container_model_object_id(original_container_id, resolved_provider)
            in allowed_container_ids
        ):
            filtered.append(item)

    return _set_container_list_data(
        response, filtered, removed_filtered_items=len(filtered) != len(data)
    )
