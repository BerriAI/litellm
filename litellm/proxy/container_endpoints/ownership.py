import json
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any, Final, Protocol

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
from litellm.repositories.table_repositories import ManagedObjectRepository
from litellm.responses.utils import ResponsesAPIRequestUtils

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _ManagedObjectRow(Protocol):
    model_object_id: str
    unified_object_id: str | None
    file_purpose: str | None
    created_by: str | None


class _ManagedObjectTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, str]) -> _ManagedObjectRow | None: ...

    async def find_first(self, *, where: Mapping[str, str]) -> _ManagedObjectRow | None: ...

    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_ManagedObjectRow]: ...

    async def create(self, *, data: Mapping[str, str]) -> _ManagedObjectRow: ...

    async def update(self, *, where: Mapping[str, str], data: Mapping[str, str]) -> _ManagedObjectRow | None: ...


CONTAINER_OBJECT_PURPOSE: Final = "container"

# 60s LRU/TTL cache absorbs every container access check before it reaches
# Prisma. ``_NEGATIVE_OWNER_SENTINEL`` lets us cache a true "untracked"
# answer so repeated misses also avoid the DB — ``InMemoryCache`` returns
# ``None`` indistinguishably for "miss" and "cached as None".
_NEGATIVE_OWNER_SENTINEL: Final = "__litellm_container_no_owner__"
_CONTAINER_OWNER_CACHE: Final = InMemoryCache(max_size_in_memory=10000, default_ttl=60)

# Caches the stored ``unified_object_id`` (the encoded container ID
# captured at create time) so ``get_container_forwarding_params`` can
# recover the deployment ``model_id`` for native upstream IDs without
# re-hitting Prisma on every retrieve/delete.
_NEGATIVE_STORED_ID_SENTINEL: Final = "__litellm_container_no_stored_id__"
_CONTAINER_STORED_ID_CACHE: Final = InMemoryCache(max_size_in_memory=10000, default_ttl=60)

# Per-caller-scope cache for ``GET /v1/containers`` list filtering. Without
# this, every list call issues a fresh ``find_many`` against
# ``litellm_managedobjecttable``. The cache key is the sorted owner-scope
# tuple — different keys for the same user share the same allow-set, but
# different users with different scopes get disjoint cache entries.
_ALLOWED_CONTAINER_IDS_CACHE: Final = InMemoryCache(max_size_in_memory=2048, default_ttl=60)


def _allowed_container_ids_cache_key(owner_scopes: Sequence[str]) -> str:
    """JSON-encode the sorted scope list — using a separator like ``|``
    would collide for any tenant whose user_id / team_id / org_id /
    api_key happens to contain the separator. JSON quoting escapes
    every separator that matters."""
    return json.dumps(sorted(owner_scopes))


def _container_model_object_id(original_container_id: str, custom_llm_provider: str) -> str:
    return f"{CONTAINER_OBJECT_PURPOSE}:{custom_llm_provider}:{original_container_id}"


def decode_container_id_for_ownership(container_id: str, custom_llm_provider: str) -> tuple[str, str]:
    decoded: Final = ResponsesAPIRequestUtils._decode_container_id(container_id)
    original_container_id: Final = decoded.get("response_id", container_id)
    decoded_provider: Final = decoded.get("custom_llm_provider")
    if decoded_provider and custom_llm_provider == "openai":
        custom_llm_provider = decoded_provider
    return original_container_id, custom_llm_provider


async def get_container_forwarding_params(
    container_id: str, original_container_id: str, custom_llm_provider: str
) -> dict[str, str]:
    params: Final = {
        "container_id": original_container_id,
        "custom_llm_provider": custom_llm_provider,
    }
    decoded: Final = ResponsesAPIRequestUtils._decode_container_id(container_id)
    model_id = decoded.get("model_id")
    if not (isinstance(model_id, str) and model_id):
        # Native upstream IDs (e.g. Azure ``cntr_<hex>``) carry no LiteLLM
        # routing payload, so decoding the user-supplied id yields no
        # ``model_id``. Recover it from the encoded ``unified_object_id``
        # captured on the ownership row at create time — when the router
        # selected a specific deployment that ID embeds the model_id.
        stored_id: Final = await _get_stored_container_id(original_container_id, custom_llm_provider)
        if stored_id and stored_id != container_id:
            stored_decoded: Final = ResponsesAPIRequestUtils._decode_container_id(stored_id)
            stored_model_id: Final = stored_decoded.get("model_id")
            if isinstance(stored_model_id, str) and stored_model_id:
                model_id = stored_model_id
    if isinstance(model_id, str) and model_id:
        params["model_id"] = model_id
    return params


def _get_response_id(response: object) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        value = response.get("id")
    else:
        value = getattr(response, "id", None)
    return value if isinstance(value, str) else None


def _dump_response(response: Any) -> dict[str, object]:
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return {"id": _get_response_id(response)}


async def _get_prisma_client() -> "PrismaClient | None":
    from litellm.proxy.proxy_server import prisma_client

    return prisma_client


def _custom_llm_provider_from_responses_response(
    response: object,
    default: str = "openai",
) -> str:
    hidden_params: Mapping[str, object] = {}
    if isinstance(response, dict):
        hidden_params = response.get("_hidden_params") or {}
    else:
        hidden_params = getattr(response, "_hidden_params", None) or {}

    provider: Final = hidden_params.get("custom_llm_provider")
    if isinstance(provider, str) and provider:
        return provider
    return default


async def record_container_owners_from_responses_response(
    response: object,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str | None = None,
) -> None:
    """Track containers created implicitly by code interpreter in /v1/responses."""
    container_ids: Final = ResponsesAPIRequestUtils.collect_container_ids_from_responses_response(response)
    if not container_ids:
        return

    resolved_provider: Final = custom_llm_provider or _custom_llm_provider_from_responses_response(response)

    for container_id in container_ids:
        try:
            await record_container_owner(
                response={"id": container_id, "object": "container"},
                user_api_key_dict=user_api_key_dict,
                custom_llm_provider=resolved_provider,
            )
        except Exception as e:
            # Per-container errors (including ``HTTPException`` from
            # conflicting/forbidden ownership rows) must not abort the
            # batch — other containers in the same response should still
            # get recorded so their follow-up file API calls don't 403.
            verbose_proxy_logger.exception(
                "Failed to record container ownership from responses output for container_id=%s: %s",
                container_id,
                e,
            )


async def record_container_owner(
    response: object,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> object:
    container_id: Final = _get_response_id(response)
    if container_id is None:
        verbose_proxy_logger.warning("Skipping container ownership tracking because provider response has no id")
        return response
    owner: Final = get_primary_resource_owner_scope(user_api_key_dict)
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

    original_container_id, resolved_provider = decode_container_id_for_ownership(container_id, custom_llm_provider)
    model_object_id: Final = _container_model_object_id(original_container_id, resolved_provider)
    file_object: Final = _dump_response(response)
    file_object["custom_llm_provider"] = resolved_provider
    file_object["provider_container_id"] = original_container_id
    # Prisma Python requires Json fields to be serialized as a JSON string.
    file_object_json: Final[str] = json.dumps(file_object)

    prisma_client: Final = await _get_prisma_client()
    if prisma_client is None:
        verbose_proxy_logger.warning("Skipping container ownership tracking because prisma_client is None")
        return response

    table: Final[_ManagedObjectTable] = ManagedObjectRepository(prisma_client).table
    existing: Final = await table.find_unique(where={"model_object_id": model_object_id})
    if existing is not None:
        if getattr(existing, "file_purpose", None) != CONTAINER_OBJECT_PURPOSE:
            raise HTTPException(status_code=500, detail="Unable to track container")
        if not user_can_access_resource_owner(getattr(existing, "created_by", None), user_api_key_dict):
            raise HTTPException(status_code=403, detail="Forbidden")
        await table.update(
            where={"model_object_id": model_object_id},
            data={
                "unified_object_id": container_id,
                "file_object": file_object_json,
                "updated_by": owner,
            },
        )
    else:
        await table.create(
            data={
                "unified_object_id": container_id,
                "model_object_id": model_object_id,
                "file_object": file_object_json,
                "file_purpose": CONTAINER_OBJECT_PURPOSE,
                "created_by": owner,
                "updated_by": owner,
            }
        )

    _CONTAINER_OWNER_CACHE.set_cache(model_object_id, owner)
    _CONTAINER_STORED_ID_CACHE.set_cache(model_object_id, container_id)
    # Drop the caller's own list-cache entry so the just-created container
    # shows up on their next ``GET /v1/containers``. Other callers with
    # disjoint scope tuples have their own entries; intersecting-scope
    # tuples self-correct on the 60s TTL.
    caller_scopes: Final = get_resource_owner_scopes(user_api_key_dict)
    if caller_scopes:
        _ALLOWED_CONTAINER_IDS_CACHE.delete_cache(_allowed_container_ids_cache_key(caller_scopes))
    return response


async def _get_container_owner(original_container_id: str, custom_llm_provider: str) -> str | None:
    model_object_id: Final = _container_model_object_id(original_container_id, custom_llm_provider)

    cached: Final = _CONTAINER_OWNER_CACHE.get_cache(model_object_id)
    if cached == _NEGATIVE_OWNER_SENTINEL:
        return None
    if cached is not None:
        return cached

    prisma_client: Final = await _get_prisma_client()
    if prisma_client is None:
        return None

    table: Final[_ManagedObjectTable] = ManagedObjectRepository(prisma_client).table
    row: Final[_ManagedObjectRow | None] = await table.find_first(
        where={
            "model_object_id": model_object_id,
            "file_purpose": CONTAINER_OBJECT_PURPOSE,
        }
    )
    owner: Final[str | None] = getattr(row, "created_by", None) if row is not None else None
    _CONTAINER_OWNER_CACHE.set_cache(model_object_id, owner if owner is not None else _NEGATIVE_OWNER_SENTINEL)
    stored_id: Final[str | None] = getattr(row, "unified_object_id", None) if row is not None else None
    _CONTAINER_STORED_ID_CACHE.set_cache(
        model_object_id,
        (stored_id if isinstance(stored_id, str) and stored_id else _NEGATIVE_STORED_ID_SENTINEL),
    )
    return owner


async def _get_stored_container_id(original_container_id: str, custom_llm_provider: str) -> str | None:
    """Return the ``unified_object_id`` stored at create time, if any.

    Used by :func:`get_container_forwarding_params` to recover the
    deployment ``model_id`` for native upstream container IDs: the stored
    value is the encoded form produced by ``encode_container_id_in_response``
    when the router selected a specific deployment.
    """
    model_object_id: Final = _container_model_object_id(original_container_id, custom_llm_provider)

    cached: Final = _CONTAINER_STORED_ID_CACHE.get_cache(model_object_id)
    if cached == _NEGATIVE_STORED_ID_SENTINEL:
        return None
    if isinstance(cached, str) and cached:
        return cached

    prisma_client: Final = await _get_prisma_client()
    if prisma_client is None:
        return None

    table: Final[_ManagedObjectTable] = ManagedObjectRepository(prisma_client).table
    row: Final[_ManagedObjectRow | None] = await table.find_first(
        where={
            "model_object_id": model_object_id,
            "file_purpose": CONTAINER_OBJECT_PURPOSE,
        }
    )
    stored_id: Final[str | None] = getattr(row, "unified_object_id", None) if row is not None else None
    _CONTAINER_STORED_ID_CACHE.set_cache(
        model_object_id,
        (stored_id if isinstance(stored_id, str) and stored_id else _NEGATIVE_STORED_ID_SENTINEL),
    )
    return stored_id if isinstance(stored_id, str) and stored_id else None


async def assert_user_can_access_container(
    container_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> tuple[str, str]:
    original_container_id, resolved_provider = decode_container_id_for_ownership(container_id, custom_llm_provider)

    if is_proxy_admin(user_api_key_dict):
        return original_container_id, resolved_provider

    # Untracked rows (no ownership) are admin-only. Pre-isolation rows
    # that pre-date this enforcement need an admin to either re-create
    # via the now-tracked flow or assign ``created_by`` on the row.
    owner: Final = await _get_container_owner(original_container_id, resolved_provider)
    if not user_can_access_resource_owner(owner, user_api_key_dict):
        raise HTTPException(status_code=403, detail="Forbidden")

    return original_container_id, resolved_provider


def _get_container_list_data(response: object) -> Sequence[object] | None:
    if response is None:
        return None
    if isinstance(response, dict):
        data = response.get("data")
    else:
        data = getattr(response, "data", None)
    return data if isinstance(data, list) else None


def _set_container_list_data(response: Any, data: list[object], removed_filtered_items: bool = False) -> object:
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
) -> AbstractSet[str]:
    owner_scopes: Final = get_resource_owner_scopes(user_api_key_dict)
    if not owner_scopes:
        return set()

    cache_key: Final = _allowed_container_ids_cache_key(owner_scopes)
    cached: Final = _ALLOWED_CONTAINER_IDS_CACHE.get_cache(cache_key)
    if cached is not None:
        return set(cached)

    prisma_client: Final = await _get_prisma_client()
    if prisma_client is None:
        return set()

    table: Final[_ManagedObjectTable] = ManagedObjectRepository(prisma_client).table
    rows: Final[Sequence[_ManagedObjectRow]] = await table.find_many(
        where={
            "file_purpose": CONTAINER_OBJECT_PURPOSE,
            "created_by": {"in": owner_scopes},
        }
    )
    allowed_ids: Final = {row.model_object_id for row in rows if getattr(row, "model_object_id", None) is not None}
    # ``InMemoryCache.get_cache`` attempts ``json.loads`` on the stored
    # value; passing a set would round-trip through that path
    # unnecessarily. Store as a list and rehydrate above.
    _ALLOWED_CONTAINER_IDS_CACHE.set_cache(cache_key, list(allowed_ids))
    return allowed_ids


async def filter_container_list_response(
    response: object,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> object:
    if is_proxy_admin(user_api_key_dict):
        return response

    data: Final = _get_container_list_data(response)
    if data is None:
        return response

    allowed_container_ids: Final = await _get_allowed_container_ids(user_api_key_dict)
    filtered: Final[list[object]] = []
    for item in data:
        container_id = _get_response_id(item)
        if container_id is None:
            continue
        original_container_id, resolved_provider = decode_container_id_for_ownership(container_id, custom_llm_provider)
        if _container_model_object_id(original_container_id, resolved_provider) in allowed_container_ids:
            filtered.append(item)

    return _set_container_list_data(response, filtered, removed_filtered_items=len(filtered) != len(data))
