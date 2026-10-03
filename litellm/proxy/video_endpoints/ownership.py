"""Per-caller ownership of videos created through the proxy.

Ownership rows live in ``LiteLLM_ManagedObjectTable`` with ``file_purpose="video"``
(the same table and owner scopes container ownership uses). A row is keyed by the
provider-native video id only, so re-wrapping an id with a different provider or
model_id cannot dodge the lookup. Proxy admins bypass the check; for everyone else,
a video without a row is treated as admin-only.

Without a connected database there is nowhere to record ownership: recording and
checks are skipped and videos remain reachable by any caller allowed on the route.
"""

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from typing_extensions import TypeIs  # noqa: TID251  # narrows untyped wire payloads without a runtime conversion

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
    user_can_access_resource_owner,
)
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.table_repositories import ManagedObjectRepository
from litellm.types.videos.utils import extract_original_video_id

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

VIDEO_OBJECT_PURPOSE: Final = "video"

# Status is polled; a short-lived cache of known owners keeps polling off the DB.
# Only positive answers are cached, so a missing row is always re-checked.
_VIDEO_OWNER_CACHE: Final = InMemoryCache(max_size_in_memory=10000, default_ttl=60)


def _video_model_object_id(video_id: str) -> str:
    return f"{VIDEO_OBJECT_PURPOSE}:{extract_original_video_id(video_id)}"


def _video_id_of(item: object) -> str | None:
    match item:
        case {"id": str(video_id)} if video_id:
            return video_id
        case object(id=str(video_id)) if video_id:
            return video_id
        case _:
            return None


def _is_object_mapping(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: provider JSON objects have str keys
    return isinstance(value, Mapping)


def _is_object_sequence(value: object) -> TypeIs[Sequence[object]]:  # guard-ok: a list is a Sequence of anything
    return isinstance(value, list)


def _get_prisma_client() -> "PrismaClient | None":
    from litellm.proxy.proxy_server import prisma_client

    return prisma_client


async def record_video_owner(response: object, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Stamp the caller as owner of a video the provider just created.

    Failures are logged rather than raised: the provider job already exists and is
    billed, so the caller still gets its id; the untracked video is admin-only.
    """
    prisma_client: Final = _get_prisma_client()
    if prisma_client is None:
        return
    video_id: Final = _video_id_of(response)
    owner: Final = get_primary_resource_owner_scope(user_api_key_dict)
    if video_id is None or owner is None:
        verbose_proxy_logger.warning(
            "Skipping video ownership tracking: response has no id or caller has no identity scope"
        )
        return
    model_object_id: Final = _video_model_object_id(video_id)
    row: Final = {
        "unified_object_id": video_id,
        "model_object_id": model_object_id,
        "file_object": json.dumps({"id": video_id, "object": "video"}),
        "file_purpose": VIDEO_OBJECT_PURPOSE,
        "created_by": owner,
        "updated_by": owner,
    }
    try:
        await ManagedObjectRepository(prisma_client).table.upsert(
            where={"model_object_id": model_object_id},
            data={"create": row, "update": {"updated_by": owner}},
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "Video ownership recording failed; video_id=%s is untracked and admin-only: %s", video_id, e
        )


async def _get_video_owner(prisma_client: "PrismaClient", model_object_id: str) -> str | None:
    cached: Final = _VIDEO_OWNER_CACHE.get_cache(model_object_id)
    if isinstance(cached, str):
        return cached
    row: Final = await ManagedObjectRepository(prisma_client).table.find_first(
        where={
            "model_object_id": model_object_id,
            "file_purpose": VIDEO_OBJECT_PURPOSE,
        }
    )
    owner: Final = getattr(row, "created_by", None) if row is not None else None
    if not isinstance(owner, str):
        return None
    _VIDEO_OWNER_CACHE.set_cache(model_object_id, owner)
    return owner


async def assert_user_can_access_video(video_id: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Raise 403 unless the caller owns ``video_id`` (or is a proxy admin)."""
    if not video_id or is_proxy_admin(user_api_key_dict):
        return
    prisma_client: Final = _get_prisma_client()
    if prisma_client is None:
        return
    owner: Final = await _get_video_owner(prisma_client, _video_model_object_id(video_id))
    if not user_can_access_resource_owner(owner, user_api_key_dict):
        raise HTTPException(status_code=403, detail="Forbidden")


async def filter_video_list_for_caller(listed: object, user_api_key_dict: UserAPIKeyAuth) -> object:
    """Drop videos the caller does not own from a provider list page.

    Pagination cursors are left as the provider returned them so ``after`` keeps
    walking the provider's list even when a page has no videos owned by the caller.
    """
    if is_proxy_admin(user_api_key_dict) or not _is_object_mapping(listed):
        return listed
    prisma_client: Final = _get_prisma_client()
    items: Final = listed.get("data")
    if prisma_client is None or not _is_object_sequence(items):
        return listed
    ids_by_item: Final = tuple((item, _video_id_of(item)) for item in items)
    candidate_ids: Final = tuple(
        _video_model_object_id(video_id) for _, video_id in ids_by_item if video_id is not None
    )
    owner_scopes: Final = get_resource_owner_scopes(user_api_key_dict)
    rows: Final = (
        await find_many_in(
            ManagedObjectRepository(prisma_client).table,
            "model_object_id",
            candidate_ids,
            where={
                "file_purpose": VIDEO_OBJECT_PURPOSE,
                # bounded-ok: owner scopes are the caller's user, team, org and key identities
                "created_by": {"in": owner_scopes},
            },
        )
        if candidate_ids and owner_scopes
        else ()
    )
    owned: Final = frozenset(row.model_object_id for row in rows)
    kept: Final = tuple(
        item for item, video_id in ids_by_item if video_id is not None and _video_model_object_id(video_id) in owned
    )
    return {**listed, "data": kept}
