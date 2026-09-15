from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AwareDatetime

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.memory.memory_endpoints import require_memory_prisma
from litellm.proxy.memory.policy import (
    MEMORY_CONFIG_PARAM,
    MemoryIdentity,
    invalidate_memory_configuration,
    memory_primary_client,
    memory_settings,
    resolve_memory_access,
)
from litellm.proxy.memory.store import MemoryStore
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.memory_v2 import (
    MemoryCapture,
    MemoryEnrollment,
    MemoryEntry,
    MemoryQuery,
    MemorySearch,
    MemorySettings,
    MemorySettingsView,
    MemoryStatus,
)

_AUTH: Final = Depends(user_api_key_auth)
router: Final = APIRouter(prefix="/memory/v2", tags=["memory management"])  # mutable-ok: FastAPI requires native tags.


def require_memory_admin(auth: UserAPIKeyAuth, *, write: bool = False) -> None:
    if not user_api_key_has_admin_view(auth) or write and auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy administrators can configure gateway memory")


async def settings_view(settings: MemorySettings) -> MemorySettingsView:
    selected: Final = tuple(frozenset((*settings.user_ids, *settings.read.user_ids)))
    users: Final = (
        await UserRepository(memory_primary_client(require_memory_prisma())).table.find_many(
            where={"user_id": {"in": list(selected)}},  # mutable-ok: Prisma requires native JSON.
            take=len(selected),
        )
        if selected
        else ()
    )
    return MemorySettingsView(
        **settings.model_dump(),
        user_names=MappingProxyType(
            {user.user_id: user.user_alias or user.user_email or user.user_id for user in users}
        ),
    )


@router.get("/settings", response_model=MemorySettingsView)
async def get_settings(auth: UserAPIKeyAuth = _AUTH) -> MemorySettingsView:
    require_memory_admin(auth)
    return await settings_view(await memory_settings(require_memory_prisma()))


@router.put("/settings", response_model=MemorySettingsView)
async def set_settings(settings: MemorySettings, auth: UserAPIKeyAuth = _AUTH) -> MemorySettingsView:
    require_memory_admin(auth, write=True)
    prisma: Final = memory_primary_client(require_memory_prisma())
    enrollments: Final = tuple(
        MemoryEnrollment(
            enabled=item.enabled,
            everyone=item.everyone,
            user_ids=tuple(sorted(frozenset(item.user_ids))) if not item.everyone else (),
        )
        for item in (settings, settings.read)
    )
    if any(item.enabled and not item.everyone and not item.user_ids for item in enrollments):
        raise HTTPException(status_code=422, detail="Select at least one user or enable memory for everyone")
    selected: Final = tuple(frozenset(user_id for item in enrollments for user_id in item.user_ids))
    if selected:
        users: Final = await UserRepository(prisma).table.find_many(
            where={"user_id": {"in": list(selected)}},  # mutable-ok: Prisma requires native query JSON.
            take=len(selected),
        )
        if frozenset(user.user_id for user in users) != frozenset(selected):
            raise HTTPException(status_code=422, detail="One or more selected users no longer exist")
    saved: Final = MemorySettings(
        **enrollments[0].model_dump(), read=enrollments[1], capture_instructions=settings.capture_instructions
    )
    await ConfigRepository(prisma).set_param(MEMORY_CONFIG_PARAM, saved.model_dump(mode="json"))
    await invalidate_memory_configuration()
    return await settings_view(saved)


async def memory_store(auth: UserAPIKeyAuth) -> MemoryStore:
    prisma: Final = memory_primary_client(require_memory_prisma())
    return MemoryStore(prisma, await resolve_memory_access(prisma, MemoryIdentity.from_auth(auth)))


@router.get("/status", response_model=MemoryStatus)
async def get_status(auth: UserAPIKeyAuth = _AUTH) -> MemoryStatus:
    store: Final = await memory_store(auth)
    user: Final = (
        await UserRepository(store.prisma_client).find_by_id(store.access.identity.user_id)
        if store.access.identity.user_id
        else None
    )
    return store.access.status.model_copy(
        update=MappingProxyType({"user_name": user.user_alias or user.user_email or user.user_id if user else None})
    )


async def named_entries(store: MemoryStore, entries: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
    actors: Final = tuple(frozenset(entry.actor for entry in entries if entry.actor))
    teams: Final = tuple(frozenset(entry.team_id for entry in entries if entry.team_id))
    users: Final = (
        await UserRepository(store.prisma_client).table.find_many(
            where={"user_id": {"in": list(actors)}},  # mutable-ok: Prisma requires native JSON.
            take=len(actors),  # mutable-ok: Prisma requires native query JSON.
        )
        if actors
        else ()
    )
    team_rows: Final = (
        await TeamRepository(store.prisma_client).table.find_many(
            where={"team_id": {"in": list(teams)}},  # mutable-ok: Prisma requires native JSON.
            take=len(teams),  # mutable-ok: Prisma requires native query JSON.
        )
        if teams
        else ()
    )
    names: Final = MappingProxyType(
        {user.user_id: user.user_alias or user.user_email or user.user_id for user in users}
    )
    team_names: Final = MappingProxyType({team.team_id: team.team_alias or team.team_id for team in team_rows})
    return tuple(
        entry.model_copy(
            update=MappingProxyType(
                {
                    "actor_name": names.get(entry.actor or ""),
                    "team_name": team_names.get(entry.team_id or ""),
                }
            )
        )
        for entry in entries
    )


@router.get("/entries", response_model=list[MemoryEntry])
async def list_entries(
    query: Annotated[MemoryQuery, Query(max_length=500)] = "",
    limit: int = Query(20, ge=1, le=20),
    offset: int = Query(0, ge=0, le=10000),
    before_updated_at: Annotated[AwareDatetime | None, Query()] = None,
    before_memory_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    team_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
    user_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
    auth: UserAPIKeyAuth = _AUTH,
) -> tuple[MemoryEntry, ...]:
    if (before_updated_at is None) != (before_memory_id is None):
        raise HTTPException(status_code=422, detail="Provide both memory cursor fields")
    store: Final = await memory_store(auth)
    entries: Final = await store.search(
        MemorySearch(query=query, limit=limit, offset=offset),
        require_active=False,
        recent_first=True,
        before=(before_updated_at, before_memory_id) if before_updated_at and before_memory_id else None,
        team_id=team_id,
        user_id=user_id,
    )
    return await named_entries(store, entries)


@router.get("/entries/{memory_id}", response_model=MemoryEntry)
async def read_entry(memory_id: str, auth: UserAPIKeyAuth = _AUTH) -> MemoryEntry:
    store: Final = await memory_store(auth)
    return (await named_entries(store, (await store.read(memory_id, require_active=False),)))[0]


@router.post("/entries", response_model=MemoryEntry)
async def capture_entry(capture: MemoryCapture, auth: UserAPIKeyAuth = _AUTH) -> MemoryEntry:
    return await (await memory_store(auth)).capture(capture)


@router.put("/entries/{memory_id}", response_model=MemoryEntry)
async def update_entry(memory_id: str, capture: MemoryCapture, auth: UserAPIKeyAuth = _AUTH) -> MemoryEntry:
    return await (await memory_store(auth)).update(memory_id, capture)


@router.delete("/entries/{memory_id}", status_code=204)
async def delete_entry(memory_id: str, auth: UserAPIKeyAuth = _AUTH) -> Response:
    if not await (await memory_store(auth)).delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)
