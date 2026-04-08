from typing import Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_AccessGroupTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _cache_access_object,
    _cache_key_object,
    _cache_team_object,
    _delete_cache_access_object,
    _get_team_object_from_cache,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.types.access_group import (
    AccessGroupCreateRequest,
    AccessGroupResponse,
    AccessGroupUpdateRequest,
)

router = APIRouter(
    tags=["access group management"],
)


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


async def _merge_access_group_resources_into_data_json(
    data_json: dict,
    access_group_ids: List[str],
    prisma_client,
) -> dict:
    """
    Batch-fetch access groups and merge their models/mcp_servers/agents into data_json.

    - Models are merged into data_json["models"]
    - MCP server IDs are merged into data_json["object_permission"]["mcp_servers"]
    - Agent IDs are merged into data_json["object_permission"]["agents"]

    This ensures that resources defined on an access group are propagated directly
    to any team or key that references those access groups.
    """
    if not access_group_ids:
        return data_json

    records = await prisma_client.db.litellm_accessgrouptable.find_many(
        where={"access_group_id": {"in": access_group_ids}}
    )

    ag_models: List[str] = list(
        {m for r in records for m in (r.access_model_names or [])}
    )
    ag_mcp_servers: List[str] = list(
        {s for r in records for s in (r.access_mcp_server_ids or [])}
    )
    ag_agents: List[str] = list(
        {a for r in records for a in (r.access_agent_ids or [])}
    )

    if ag_models:
        existing_models: List[str] = list(data_json.get("models") or [])
        data_json["models"] = list(set(existing_models + ag_models))

    if ag_mcp_servers or ag_agents:
        obj_perm: Dict = data_json.get("object_permission") or {}
        if not isinstance(obj_perm, dict):
            obj_perm = {}
        if ag_mcp_servers:
            existing_mcp: List[str] = list(obj_perm.get("mcp_servers") or [])
            obj_perm["mcp_servers"] = list(set(existing_mcp + ag_mcp_servers))
        if ag_agents:
            existing_agents: List[str] = list(obj_perm.get("agents") or [])
            obj_perm["agents"] = list(set(existing_agents + ag_agents))
        data_json["object_permission"] = obj_perm

    return data_json


def _record_to_response(record) -> AccessGroupResponse:
    return AccessGroupResponse(
        access_group_id=record.access_group_id,
        access_group_name=record.access_group_name,
        description=record.description,
        access_model_names=record.access_model_names,
        access_mcp_server_ids=record.access_mcp_server_ids,
        access_agent_ids=record.access_agent_ids,
        assigned_team_ids=record.assigned_team_ids,
        assigned_key_ids=record.assigned_key_ids,
        created_at=record.created_at,
        created_by=record.created_by,
        updated_at=record.updated_at,
        updated_by=record.updated_by,
    )


def _record_to_access_group_table(record) -> LiteLLM_AccessGroupTable:
    """Convert a Prisma record to a LiteLLM_AccessGroupTable pydantic object for caching."""
    return LiteLLM_AccessGroupTable(**record.dict())


async def _cache_access_group_record(record) -> None:
    """
    Cache an access group Prisma record in the user_api_key_cache.

    Uses a lazy import of user_api_key_cache and proxy_logging_obj from proxy_server
    to avoid circular imports, following the same pattern as key_management_endpoints.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    access_group_table = _record_to_access_group_table(record)
    await _cache_access_object(
        access_group_id=record.access_group_id,
        access_group_table=access_group_table,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def _invalidate_cache_access_group(access_group_id: str) -> None:
    """
    Invalidate (delete) an access group entry from both in-memory and Redis caches.

    Uses a lazy import of user_api_key_cache and proxy_logging_obj from proxy_server
    to avoid circular imports, following the same pattern as key_management_endpoints.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _delete_cache_access_object(
        access_group_id=access_group_id,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


# ---------------------------------------------------------------------------
# Object-permission helpers (called inside a Prisma transaction)
# ---------------------------------------------------------------------------


async def _upsert_mcp_agents_in_object_permission(
    tx,
    existing_op_id: Optional[str],
    ag_mcp_servers: List[str],
    ag_agents: List[str],
    existing_op=None,
) -> Optional[str]:
    """
    Upsert LiteLLM_ObjectPermissionTable to add MCP servers and agents.

    Merges ``ag_mcp_servers`` / ``ag_agents`` into the existing record (if any),
    creating a new record when ``existing_op_id`` is None.

    ``existing_op``: optional pre-fetched record to avoid an extra DB round-trip.

    Returns the ``object_permission_id`` of the upserted row, or ``None`` when
    both lists are empty (nothing to do).
    """
    if not ag_mcp_servers and not ag_agents:
        return None

    existing_mcp: List[str] = []
    existing_agents: List[str] = []
    existing_data: Dict = {}

    if existing_op_id:
        if existing_op is None:
            existing_op = await tx.litellm_objectpermissiontable.find_unique(
                where={"object_permission_id": existing_op_id}
            )
        if existing_op is not None:
            try:
                existing_data = existing_op.model_dump(exclude_none=True)
            except Exception:
                existing_data = existing_op.dict(exclude_none=True)
            existing_mcp = list(existing_data.get("mcp_servers") or [])
            existing_agents = list(existing_data.get("agents") or [])

    upsert_data: Dict = {
        k: v for k, v in existing_data.items() if k != "object_permission_id"
    }
    if ag_mcp_servers:
        upsert_data["mcp_servers"] = list(set(existing_mcp + ag_mcp_servers))
    if ag_agents:
        upsert_data["agents"] = list(set(existing_agents + ag_agents))

    op_id_to_use: str = existing_op_id or str(uuid.uuid4())
    create_data: Dict = {**upsert_data, "object_permission_id": op_id_to_use}
    created_row = await tx.litellm_objectpermissiontable.upsert(
        where={"object_permission_id": op_id_to_use},
        data={"create": create_data, "update": upsert_data},
    )
    return created_row.object_permission_id


async def _remove_mcp_agents_from_object_permission(
    tx,
    existing_op_id: Optional[str],
    mcp_servers_to_remove: List[str],
    agents_to_remove: List[str],
    existing_op=None,
) -> None:
    """
    Remove specific MCP server IDs and agent IDs from an existing
    LiteLLM_ObjectPermissionTable row.  No-ops when the record does not exist
    or the removal sets are empty.

    ``existing_op``: optional pre-fetched record to avoid an extra DB round-trip.
    """
    if not existing_op_id or (not mcp_servers_to_remove and not agents_to_remove):
        return

    if existing_op is None:
        existing_op = await tx.litellm_objectpermissiontable.find_unique(
            where={"object_permission_id": existing_op_id}
        )
    if existing_op is None:
        return

    op_update: Dict = {}
    if mcp_servers_to_remove:
        remove_set: Set[str] = set(mcp_servers_to_remove)
        op_update["mcp_servers"] = [
            s for s in (existing_op.mcp_servers or []) if s not in remove_set
        ]
    if agents_to_remove:
        remove_set = set(agents_to_remove)
        op_update["agents"] = [
            a for a in (existing_op.agents or []) if a not in remove_set
        ]

    if op_update:
        await tx.litellm_objectpermissiontable.update(
            where={"object_permission_id": existing_op_id},
            data=op_update,
        )


# ---------------------------------------------------------------------------
# DB sync helpers (called inside a Prisma transaction)
# ---------------------------------------------------------------------------


async def _sync_add_access_group_to_teams(
    tx,
    team_ids: List[str],
    access_group_id: str,
    access_group_record=None,
) -> None:
    """Add access_group_id to each team's access_group_ids and merge the group's
    models/mcp_servers/agents into the team's direct resource lists (idempotent).

    access_group_record: the Prisma record for the access group being added (optional).
        When provided, its resources are merged directly into the team rather than
        making an extra DB round-trip.
    """
    ag_models: List[str] = list(
        getattr(access_group_record, "access_model_names", None) or []
    )
    ag_mcp_servers: List[str] = list(
        getattr(access_group_record, "access_mcp_server_ids", None) or []
    )
    ag_agents: List[str] = list(
        getattr(access_group_record, "access_agent_ids", None) or []
    )

    if not team_ids:
        return

    # Batch-fetch all teams to avoid N+1 queries.
    teams = await tx.litellm_teamtable.find_many(
        where={"team_id": {"in": team_ids}}
    )
    team_map: Dict = {t.team_id: t for t in teams}

    # Batch-fetch object permissions for teams that need MCP/agent merging.
    op_map: Dict = {}
    if ag_mcp_servers or ag_agents:
        op_ids = [
            t.object_permission_id
            for t in teams
            if getattr(t, "object_permission_id", None)
            and access_group_id not in (t.access_group_ids or [])
        ]
        if op_ids:
            op_records = await tx.litellm_objectpermissiontable.find_many(
                where={"object_permission_id": {"in": op_ids}}
            )
            op_map = {r.object_permission_id: r for r in op_records}

    for team_id in team_ids:
        team = team_map.get(team_id)
        if team is None or access_group_id in (team.access_group_ids or []):
            continue

        update_data: Dict = {
            "access_group_ids": list(team.access_group_ids or []) + [access_group_id]
        }

        # Merge models from the access group into the team's model list
        if ag_models:
            merged_models = list(set(list(team.models or []) + ag_models))
            update_data["models"] = merged_models

        # Merge MCP servers and agents into the team's object_permission
        if ag_mcp_servers or ag_agents:
            existing_op_id: Optional[str] = getattr(
                team, "object_permission_id", None
            )
            new_op_id = await _upsert_mcp_agents_in_object_permission(
                tx,
                existing_op_id=existing_op_id,
                ag_mcp_servers=ag_mcp_servers,
                ag_agents=ag_agents,
                existing_op=op_map.get(existing_op_id) if existing_op_id else None,
            )
            # Link the (possibly newly created) object_permission row to the team
            if new_op_id is not None and new_op_id != existing_op_id:
                update_data["object_permission_id"] = new_op_id

        await tx.litellm_teamtable.update(
            where={"team_id": team_id},
            data=update_data,
        )


async def _sync_remove_access_group_from_teams(
    tx,
    team_ids: List[str],
    access_group_id: str,
    removed_access_group_record=None,
) -> None:
    """Remove access_group_id from each team's access_group_ids and clean up
    models / object_permission resources that were exclusively contributed by
    the removed access group (idempotent).

    removed_access_group_record: the Prisma record for the access group being
        removed (optional).  Pass the *pre-update* snapshot when calling from
        ``update_access_group`` so that stale post-update data is not used to
        compute the removal delta.  When ``None`` the record is fetched from
        the DB (safe for the delete path where the row still exists).
    """
    # Resolve removed AG's resources once outside the per-team loop.
    ag_record = removed_access_group_record
    if ag_record is None:
        ag_record = await tx.litellm_accessgrouptable.find_unique(
            where={"access_group_id": access_group_id}
        )

    removed_ag_models: Set[str] = set(
        getattr(ag_record, "access_model_names", None) or []
    )
    removed_ag_mcp: Set[str] = set(
        getattr(ag_record, "access_mcp_server_ids", None) or []
    )
    removed_ag_agents: Set[str] = set(
        getattr(ag_record, "access_agent_ids", None) or []
    )

    if not team_ids:
        return

    # Batch-fetch all teams to avoid N+1 queries.
    teams = await tx.litellm_teamtable.find_many(
        where={"team_id": {"in": team_ids}}
    )
    relevant_teams = [
        t for t in teams if access_group_id in (t.access_group_ids or [])
    ]

    # Collect all unique remaining AG IDs across affected teams so we can
    # batch-fetch their records in one query instead of one per team.
    all_remaining_ag_ids: Set[str] = set()
    for team in relevant_teams:
        for ag in (team.access_group_ids or []):
            if ag != access_group_id:
                all_remaining_ag_ids.add(ag)

    remaining_ag_map: Dict = {}
    if all_remaining_ag_ids:
        remaining_ag_records = await tx.litellm_accessgrouptable.find_many(
            where={"access_group_id": {"in": list(all_remaining_ag_ids)}}
        )
        remaining_ag_map = {r.access_group_id: r for r in remaining_ag_records}

    # Batch-fetch object permissions for affected teams.
    all_op_ids = [
        t.object_permission_id
        for t in relevant_teams
        if getattr(t, "object_permission_id", None)
    ]
    op_map: Dict = {}
    if all_op_ids:
        op_records = await tx.litellm_objectpermissiontable.find_many(
            where={"object_permission_id": {"in": all_op_ids}}
        )
        op_map = {r.object_permission_id: r for r in op_records}

    for team in relevant_teams:
        remaining_ag_ids = [
            ag for ag in (team.access_group_ids or []) if ag != access_group_id
        ]
        remaining_records = [
            remaining_ag_map[ag] for ag in remaining_ag_ids if ag in remaining_ag_map
        ]

        models_in_remaining: Set[str] = {
            m for r in remaining_records for m in (r.access_model_names or [])
        }
        mcp_in_remaining: Set[str] = {
            s for r in remaining_records for s in (r.access_mcp_server_ids or [])
        }
        agents_in_remaining: Set[str] = {
            a for r in remaining_records for a in (r.access_agent_ids or [])
        }

        # Only remove models/MCP/agents that were *exclusively* from the removed AG
        models_to_remove = removed_ag_models - models_in_remaining
        current_models = set(team.models or [])
        updated_models = list(current_models - models_to_remove)

        update_data: Dict = {
            "access_group_ids": remaining_ag_ids,
            "models": updated_models,
        }

        # Clean up MCP servers / agents in object_permission
        mcp_to_remove = list(removed_ag_mcp - mcp_in_remaining)
        agents_to_remove = list(removed_ag_agents - agents_in_remaining)
        existing_op_id: Optional[str] = getattr(team, "object_permission_id", None)
        await _remove_mcp_agents_from_object_permission(
            tx,
            existing_op_id=existing_op_id,
            mcp_servers_to_remove=mcp_to_remove,
            agents_to_remove=agents_to_remove,
            existing_op=op_map.get(existing_op_id) if existing_op_id else None,
        )

        await tx.litellm_teamtable.update(
            where={"team_id": team.team_id},
            data=update_data,
        )


async def _sync_add_access_group_to_keys(
    tx, key_tokens: List[str], access_group_id: str, access_group_record=None
) -> None:
    """Add access_group_id to each key's access_group_ids and merge the group's
    models/mcp_servers/agents into the key's direct resource lists (idempotent).
    """
    ag_models: List[str] = list(
        getattr(access_group_record, "access_model_names", None) or []
    )
    ag_mcp_servers: List[str] = list(
        getattr(access_group_record, "access_mcp_server_ids", None) or []
    )
    ag_agents: List[str] = list(
        getattr(access_group_record, "access_agent_ids", None) or []
    )

    if not key_tokens:
        return

    # Batch-fetch all keys to avoid N+1 queries.
    keys = await tx.litellm_verificationtoken.find_many(
        where={"token": {"in": key_tokens}}
    )
    key_map: Dict = {k.token: k for k in keys}

    # Batch-fetch object permissions for keys that need MCP/agent merging.
    op_map: Dict = {}
    if ag_mcp_servers or ag_agents:
        op_ids = [
            k.object_permission_id
            for k in keys
            if getattr(k, "object_permission_id", None)
            and access_group_id not in (k.access_group_ids or [])
        ]
        if op_ids:
            op_records = await tx.litellm_objectpermissiontable.find_many(
                where={"object_permission_id": {"in": op_ids}}
            )
            op_map = {r.object_permission_id: r for r in op_records}

    for token in key_tokens:
        key = key_map.get(token)
        if key is None or access_group_id in (key.access_group_ids or []):
            continue

        update_data: Dict = {
            "access_group_ids": list(key.access_group_ids or []) + [access_group_id]
        }

        if ag_models:
            merged_models = list(set(list(key.models or []) + ag_models))
            update_data["models"] = merged_models

        # Merge MCP servers and agents into the key's object_permission
        if ag_mcp_servers or ag_agents:
            existing_op_id: Optional[str] = getattr(
                key, "object_permission_id", None
            )
            new_op_id = await _upsert_mcp_agents_in_object_permission(
                tx,
                existing_op_id=existing_op_id,
                ag_mcp_servers=ag_mcp_servers,
                ag_agents=ag_agents,
                existing_op=op_map.get(existing_op_id) if existing_op_id else None,
            )
            # Link the (possibly newly created) object_permission row to the key
            if new_op_id is not None and new_op_id != existing_op_id:
                update_data["object_permission_id"] = new_op_id

        await tx.litellm_verificationtoken.update(
            where={"token": token},
            data=update_data,
        )


async def _sync_remove_access_group_from_keys(
    tx,
    key_tokens: List[str],
    access_group_id: str,
    removed_access_group_record=None,
) -> None:
    """Remove access_group_id from each key's access_group_ids and clean up
    models / object_permission resources that were exclusively contributed by
    the removed access group (idempotent).

    removed_access_group_record: the Prisma record for the access group being
        removed (optional).  Pass the *pre-update* snapshot when calling from
        ``update_access_group`` so that stale post-update data is not used to
        compute the removal delta.  When ``None`` the record is fetched from
        the DB (safe for the delete path where the row still exists).
    """
    # Resolve removed AG's resources once outside the per-key loop.
    ag_record = removed_access_group_record
    if ag_record is None:
        ag_record = await tx.litellm_accessgrouptable.find_unique(
            where={"access_group_id": access_group_id}
        )

    removed_ag_models: Set[str] = set(
        getattr(ag_record, "access_model_names", None) or []
    )
    removed_ag_mcp: Set[str] = set(
        getattr(ag_record, "access_mcp_server_ids", None) or []
    )
    removed_ag_agents: Set[str] = set(
        getattr(ag_record, "access_agent_ids", None) or []
    )

    if not key_tokens:
        return

    # Batch-fetch all keys to avoid N+1 queries.
    keys = await tx.litellm_verificationtoken.find_many(
        where={"token": {"in": key_tokens}}
    )
    relevant_keys = [
        k for k in keys if access_group_id in (k.access_group_ids or [])
    ]

    # Collect all unique remaining AG IDs across affected keys so we can
    # batch-fetch their records in one query instead of one per key.
    all_remaining_ag_ids: Set[str] = set()
    for key in relevant_keys:
        for ag in (key.access_group_ids or []):
            if ag != access_group_id:
                all_remaining_ag_ids.add(ag)

    remaining_ag_map: Dict = {}
    if all_remaining_ag_ids:
        remaining_ag_records = await tx.litellm_accessgrouptable.find_many(
            where={"access_group_id": {"in": list(all_remaining_ag_ids)}}
        )
        remaining_ag_map = {r.access_group_id: r for r in remaining_ag_records}

    # Batch-fetch object permissions for affected keys.
    all_op_ids = [
        k.object_permission_id
        for k in relevant_keys
        if getattr(k, "object_permission_id", None)
    ]
    op_map: Dict = {}
    if all_op_ids:
        op_records = await tx.litellm_objectpermissiontable.find_many(
            where={"object_permission_id": {"in": all_op_ids}}
        )
        op_map = {r.object_permission_id: r for r in op_records}

    for key in relevant_keys:
        remaining_ag_ids = [
            ag for ag in (key.access_group_ids or []) if ag != access_group_id
        ]
        remaining_records = [
            remaining_ag_map[ag] for ag in remaining_ag_ids if ag in remaining_ag_map
        ]

        models_in_remaining: Set[str] = {
            m for r in remaining_records for m in (r.access_model_names or [])
        }
        mcp_in_remaining: Set[str] = {
            s for r in remaining_records for s in (r.access_mcp_server_ids or [])
        }
        agents_in_remaining: Set[str] = {
            a for r in remaining_records for a in (r.access_agent_ids or [])
        }

        # Only remove models/MCP/agents that were *exclusively* from the removed AG
        models_to_remove = removed_ag_models - models_in_remaining
        current_models = set(key.models or [])
        updated_models = list(current_models - models_to_remove)

        # Clean up MCP servers / agents in object_permission
        mcp_to_remove = list(removed_ag_mcp - mcp_in_remaining)
        agents_to_remove = list(removed_ag_agents - agents_in_remaining)
        existing_op_id: Optional[str] = getattr(key, "object_permission_id", None)
        await _remove_mcp_agents_from_object_permission(
            tx,
            existing_op_id=existing_op_id,
            mcp_servers_to_remove=mcp_to_remove,
            agents_to_remove=agents_to_remove,
            existing_op=op_map.get(existing_op_id) if existing_op_id else None,
        )

        await tx.litellm_verificationtoken.update(
            where={"token": key.token},
            data={
                "access_group_ids": remaining_ag_ids,
                "models": updated_models,
            },
        )


# ---------------------------------------------------------------------------
# Cache patch helpers
# ---------------------------------------------------------------------------


async def _patch_team_caches_add_access_group(
    team_ids: List[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached team objects to include access_group_id."""
    for team_id in team_ids:
        cached_team = await _get_team_object_from_cache(
            key="team_id:{}".format(team_id),
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
        )
        if cached_team is None:
            continue
        if cached_team.access_group_ids is None:
            cached_team.access_group_ids = [access_group_id]
        elif access_group_id not in cached_team.access_group_ids:
            cached_team.access_group_ids = list(cached_team.access_group_ids) + [
                access_group_id
            ]
        else:
            continue
        await _cache_team_object(
            team_id=team_id,
            team_table=cached_team,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )


async def _patch_team_caches_remove_access_group(
    team_ids: List[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached team objects to remove access_group_id."""
    for team_id in team_ids:
        cached_team = await _get_team_object_from_cache(
            key="team_id:{}".format(team_id),
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
        )
        if cached_team is not None and cached_team.access_group_ids:
            cached_team.access_group_ids = [
                ag for ag in cached_team.access_group_ids if ag != access_group_id
            ]
            await _cache_team_object(
                team_id=team_id,
                team_table=cached_team,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )


async def _patch_key_caches_add_access_group(
    key_tokens: List[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached key objects to include access_group_id."""
    for token in key_tokens:
        cached_key = await user_api_key_cache.async_get_cache(key=token)
        if cached_key is None:
            continue
        if isinstance(cached_key, dict):
            cached_key = UserAPIKeyAuth(**cached_key)
        if not isinstance(cached_key, UserAPIKeyAuth):
            continue
        if cached_key.access_group_ids is None:
            cached_key.access_group_ids = [access_group_id]
        elif access_group_id not in cached_key.access_group_ids:
            cached_key.access_group_ids = list(cached_key.access_group_ids) + [
                access_group_id
            ]
        else:
            continue
        await _cache_key_object(
            hashed_token=token,
            user_api_key_obj=cached_key,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )


async def _patch_key_caches_remove_access_group(
    key_tokens: List[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached key objects to remove access_group_id."""
    for token in key_tokens:
        cached_key = await user_api_key_cache.async_get_cache(key=token)
        if cached_key is None:
            continue
        if isinstance(cached_key, dict):
            cached_key = UserAPIKeyAuth(**cached_key)
        if isinstance(cached_key, UserAPIKeyAuth) and cached_key.access_group_ids:
            cached_key.access_group_ids = [
                ag for ag in cached_key.access_group_ids if ag != access_group_id
            ]
            await _cache_key_object(
                hashed_token=token,
                user_api_key_obj=cached_key,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/v1/access_group",
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_access_group(
    data: AccessGroupCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    try:
        async with prisma_client.db.tx() as tx:
            existing = await tx.litellm_accessgrouptable.find_unique(
                where={"access_group_name": data.access_group_name}
            )
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Access group '{data.access_group_name}' already exists",
                )

            record = await tx.litellm_accessgrouptable.create(
                data={
                    "access_group_name": data.access_group_name,
                    "description": data.description,
                    "access_model_names": data.access_model_names or [],
                    "access_mcp_server_ids": data.access_mcp_server_ids or [],
                    "access_agent_ids": data.access_agent_ids or [],
                    "assigned_team_ids": data.assigned_team_ids or [],
                    "assigned_key_ids": data.assigned_key_ids or [],
                    "created_by": user_api_key_dict.user_id,
                    "updated_by": user_api_key_dict.user_id,
                }
            )

            # Sync team and key tables to reference the new access group
            # Pass the created record so sync can merge models/MCP/agents directly.
            await _sync_add_access_group_to_teams(
                tx,
                data.assigned_team_ids or [],
                record.access_group_id,
                access_group_record=record,
            )
            await _sync_add_access_group_to_keys(
                tx,
                data.assigned_key_ids or [],
                record.access_group_id,
                access_group_record=record,
            )
    except HTTPException:
        raise
    except Exception as e:
        # Race condition: another request created the same name between find_unique and create.
        if "unique constraint" in str(e).lower() or "P2002" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Access group '{data.access_group_name}' already exists",
            )
        raise

    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _cache_access_group_record(record)
    await _patch_team_caches_add_access_group(
        data.assigned_team_ids or [],
        record.access_group_id,
        user_api_key_cache,
        proxy_logging_obj,
    )
    await _patch_key_caches_add_access_group(
        data.assigned_key_ids or [],
        record.access_group_id,
        user_api_key_cache,
        proxy_logging_obj,
    )

    return _record_to_response(record)


@router.get(
    "/v1/access_group",
    response_model=List[AccessGroupResponse],
)
async def list_access_groups(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[AccessGroupResponse]:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    records = await prisma_client.db.litellm_accessgrouptable.find_many(
        order={"created_at": "desc"}
    )
    return [_record_to_response(r) for r in records]


@router.get(
    "/v1/access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def get_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    record = await prisma_client.db.litellm_accessgrouptable.find_unique(
        where={"access_group_id": access_group_id}
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access group '{access_group_id}' not found",
        )
    return _record_to_response(record)


@router.put(
    "/v1/access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def update_access_group(
    access_group_id: str,
    data: AccessGroupUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    update_fields = data.model_dump(exclude_unset=True)
    update_data: dict = {"updated_by": user_api_key_dict.user_id}
    for field, value in update_fields.items():
        if (
            field
            in (
                "assigned_team_ids",
                "assigned_key_ids",
                "access_model_names",
                "access_mcp_server_ids",
                "access_agent_ids",
            )
            and value is None
        ):
            value = []
        update_data[field] = value

    # Initialize delta lists before the try block so they remain accessible
    # for cache updates after the transaction, even if an error path is added later.
    teams_to_add: List[str] = []
    teams_to_remove: List[str] = []
    keys_to_add: List[str] = []
    keys_to_remove: List[str] = []

    try:
        async with prisma_client.db.tx() as tx:
            # Read inside the transaction so delta computation is consistent with the write,
            # avoiding a TOCTOU race where a concurrent update could make deltas stale.
            existing = await tx.litellm_accessgrouptable.find_unique(
                where={"access_group_id": access_group_id}
            )
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Access group '{access_group_id}' not found",
                )

            old_team_ids: Set[str] = set(existing.assigned_team_ids or [])
            old_key_ids: Set[str] = set(existing.assigned_key_ids or [])
            new_team_ids: Set[str] = (
                set(update_fields["assigned_team_ids"] or [])
                if "assigned_team_ids" in update_fields
                else old_team_ids
            )
            new_key_ids: Set[str] = (
                set(update_fields["assigned_key_ids"] or [])
                if "assigned_key_ids" in update_fields
                else old_key_ids
            )

            teams_to_add = list(new_team_ids - old_team_ids)
            teams_to_remove = list(old_team_ids - new_team_ids)
            keys_to_add = list(new_key_ids - old_key_ids)
            keys_to_remove = list(old_key_ids - new_key_ids)

            record = await tx.litellm_accessgrouptable.update(
                where={"access_group_id": access_group_id},
                data=update_data,
            )

            await _sync_add_access_group_to_teams(
                tx, teams_to_add, access_group_id, access_group_record=record
            )
            # Pass `existing` (pre-update snapshot) so remove logic uses the
            # OLD resource lists when computing the removal delta, not the
            # post-update ones.
            await _sync_remove_access_group_from_teams(
                tx,
                teams_to_remove,
                access_group_id,
                removed_access_group_record=existing,
            )
            await _sync_add_access_group_to_keys(
                tx, keys_to_add, access_group_id, access_group_record=record
            )
            await _sync_remove_access_group_from_keys(
                tx,
                keys_to_remove,
                access_group_id,
                removed_access_group_record=existing,
            )
    except HTTPException:
        raise
    except Exception as e:
        # Unique constraint violation (e.g. access_group_name already exists).
        if "unique constraint" in str(e).lower() or "P2002" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Access group '{update_data.get('access_group_name', '')}' already exists",
            )
        raise

    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _cache_access_group_record(record)
    await _patch_team_caches_add_access_group(
        teams_to_add, access_group_id, user_api_key_cache, proxy_logging_obj
    )
    await _patch_team_caches_remove_access_group(
        teams_to_remove, access_group_id, user_api_key_cache, proxy_logging_obj
    )
    await _patch_key_caches_add_access_group(
        keys_to_add, access_group_id, user_api_key_cache, proxy_logging_obj
    )
    await _patch_key_caches_remove_access_group(
        keys_to_remove, access_group_id, user_api_key_cache, proxy_logging_obj
    )

    return _record_to_response(record)


@router.delete(
    "/v1/access_group/{access_group_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> None:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    try:
        affected_team_ids: List[str] = []
        affected_key_tokens: List[str] = []

        async with prisma_client.db.tx() as tx:
            existing = await tx.litellm_accessgrouptable.find_unique(
                where={"access_group_id": access_group_id}
            )
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Access group '{access_group_id}' not found",
                )

            # Union of: teams that have this access_group_id in their own access_group_ids
            # AND teams listed in assigned_team_ids (handles out-of-sync data from before this sync was added)
            teams_with_group = await tx.litellm_teamtable.find_many(
                where={"access_group_ids": {"hasSome": [access_group_id]}}
            )
            all_affected_team_ids: Set[str] = {
                team.team_id for team in teams_with_group
            } | set(existing.assigned_team_ids or [])
            affected_team_ids = list(all_affected_team_ids)

            # Union of: keys that have this access_group_id in their own access_group_ids
            # AND keys listed in assigned_key_ids (handles out-of-sync data)
            keys_with_group = await tx.litellm_verificationtoken.find_many(
                where={"access_group_ids": {"hasSome": [access_group_id]}}
            )
            all_affected_key_tokens: Set[str] = {
                key.token for key in keys_with_group
            } | set(existing.assigned_key_ids or [])
            affected_key_tokens = list(all_affected_key_tokens)

            # Use _sync_remove for ALL affected teams — it correctly handles
            # model/MCP/agent cleanup and is idempotent.  The `existing` record
            # is passed as the pre-delete snapshot so that removal deltas are
            # computed from the right resource lists before the row is deleted.
            await _sync_remove_access_group_from_teams(
                tx,
                affected_team_ids,
                access_group_id,
                removed_access_group_record=existing,
            )
            await _sync_remove_access_group_from_keys(
                tx,
                affected_key_tokens,
                access_group_id,
                removed_access_group_record=existing,
            )

            await tx.litellm_accessgrouptable.delete(
                where={"access_group_id": access_group_id}
            )

        from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

        await _invalidate_cache_access_group(access_group_id)
        await _patch_team_caches_remove_access_group(
            affected_team_ids, access_group_id, user_api_key_cache, proxy_logging_obj
        )
        await _patch_key_caches_remove_access_group(
            affected_key_tokens, access_group_id, user_api_key_cache, proxy_logging_obj
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "delete_access_group failed: access_group_id=%s error=%s",
            access_group_id,
            e,
        )
        if PrismaDBExceptionHandler.is_database_connection_error(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )
        if "P2025" in str(e) or (
            "record" in str(e).lower() and "not found" in str(e).lower()
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Access group '{access_group_id}' not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete access group. Please try again.",
        )


# Alias routes for /v1/unified_access_group
router.add_api_route(
    "/v1/unified_access_group",
    create_access_group,
    methods=["POST"],
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
router.add_api_route(
    "/v1/unified_access_group",
    list_access_groups,
    methods=["GET"],
    response_model=List[AccessGroupResponse],
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    get_access_group,
    methods=["GET"],
    response_model=AccessGroupResponse,
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    update_access_group,
    methods=["PUT"],
    response_model=AccessGroupResponse,
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    delete_access_group,
    methods=["DELETE"],
    status_code=status.HTTP_204_NO_CONTENT,
)
