import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from litellm.caching.caching import DualCache
from litellm.proxy._types import UI_TEAM_ID, KeyManagementRoutes, LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.config_sync_pubsub import coordination_redis_cache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper, WriterPinnedClient
from litellm.proxy.management_helpers.record_permissions import permitted_record_teams
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.memory_v2 import MemorySettings, MemoryStatus

_SETTINGS_CACHE_KEY: Final = "litellm:memory_v2:settings"
MEMORY_CONFIG_PARAM: Final = "memory_v2"


async def memory_settings(prisma_client: object) -> MemorySettings:
    row: Final = await ConfigRepository(memory_primary_client(prisma_client)).get_param(MEMORY_CONFIG_PARAM)
    return MemorySettings.model_validate(row.param_value) if row is not None else MemorySettings()


async def gateway_memory_is_enabled(prisma_client: object, cache: DualCache, identity: "MemoryIdentity") -> bool:
    cached: Final = await cache.async_get_cache(key=_SETTINGS_CACHE_KEY, ttl=30)
    if cached is not None and MemoryAccess(identity, MemorySettings.model_validate(cached)).active:
        return True
    redis_cache: Final = cache.redis_cache or coordination_redis_cache()
    shared_cache: Final = DualCache(redis_cache=redis_cache) if redis_cache is not None else None
    if cached is not None:
        if shared_cache is None:
            return False
        shared: Final = await shared_cache.async_get_cache(key=_SETTINGS_CACHE_KEY, ttl=30)
        if shared is not None:
            return MemoryAccess(identity, MemorySettings.model_validate(shared)).active
    settings: Final = await memory_settings(prisma_client)
    await cache.async_set_cache(key=_SETTINGS_CACHE_KEY, value=settings.model_dump(mode="json"), ttl=30)
    if shared_cache is not None and cache.redis_cache is None:
        await shared_cache.async_set_cache(key=_SETTINGS_CACHE_KEY, value=settings.model_dump(mode="json"), ttl=30)
    return MemoryAccess(identity, settings).active


async def invalidate_memory_configuration() -> None:
    from litellm.proxy.proxy_server import user_api_key_cache

    cache: Final = UserApiKeyCache(
        in_memory_cache=user_api_key_cache.in_memory_cache,
        redis_cache=user_api_key_cache.redis_cache or coordination_redis_cache(),
    )
    await evict_and_broadcast(cache_keys=(_SETTINGS_CACHE_KEY,), user_api_key_cache=cache)


def memory_primary_client(prisma_client: object) -> WriterPinnedClient:
    db: Final = getattr(prisma_client, "db", None)
    if db is None:
        raise RuntimeError("Memory requires a connected Prisma database")
    return WriterPinnedClient(db.writer if isinstance(db, RoutingPrismaWrapper) else db)


def memory_digest(*parts: str | None) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MemoryIdentity:
    key_id: str | None
    user_id: str | None
    team_id: str | None
    organization_id: str | None
    read_only: bool
    role: str | None = None
    dashboard: bool = False

    @classmethod
    def from_auth(cls, auth: UserAPIKeyAuth) -> "MemoryIdentity":
        token: Final = auth.token or auth.api_key
        dashboard: Final = auth.is_session_token or auth.team_id == UI_TEAM_ID
        key_id: Final = (
            token
            if token and len(token) == 64 and all(c in "0123456789abcdef" for c in token) and not dashboard
            else None
        )
        return cls(
            key_id=key_id,
            user_id=auth.user_id,
            team_id=None if dashboard else auth.team_id,
            organization_id=auth.org_id,
            read_only=auth.user_role
            in (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY),
            role=auth.user_role,
            dashboard=dashboard,
        )

    @property
    def namespace(self) -> str:
        return "v2:" + memory_digest(
            self.organization_id, self.team_id, self.user_id, None if self.user_id else self.key_id
        )


@dataclass(frozen=True)
class MemoryAccess:
    identity: MemoryIdentity
    settings: MemorySettings
    team_ids: tuple[str, ...] = ()
    admin_view: bool = False
    permission_revision: str = ""

    @property
    def namespace(self) -> str:
        return self.identity.namespace

    @property
    def active(self) -> bool:
        return self.save_enabled or self.read_enabled

    @property
    def save_enabled(self) -> bool:
        return (
            bool(self.identity.user_id or self.identity.key_id)
            and not self.identity.read_only
            and self.settings.allows(self.identity.user_id)
        )

    @property
    def read_enabled(self) -> bool:
        return bool(self.identity.user_id or self.identity.key_id) and self.settings.read.allows(self.identity.user_id)

    @property
    def continuation_revision(self) -> str:
        return memory_digest(self.permission_revision, str(self.read_enabled))

    @property
    def status(self) -> MemoryStatus:
        return MemoryStatus(
            active=self.active,
            save_enabled=self.save_enabled,
            read_enabled=self.read_enabled,
            user_id=self.identity.user_id,
            enabled=self.settings.enabled or self.settings.read.enabled,
            team_ids=self.team_ids,
            admin_view=self.admin_view,
        )

    def visible_rows(self, *, write: bool = False) -> Mapping[str, object]:
        if self.admin_view:
            return {"namespace": {"not": None}}  # mutable-ok: Prisma requires native query JSON.
        owner: Final = (
            {"user_id": self.identity.user_id}  # mutable-ok: Prisma requires native query JSON.
            if self.identity.user_id
            else {  # mutable-ok: Prisma requires native JSON.
                "owner_key_id": self.identity.key_id,
                "user_id": None,
            }  # mutable-ok: Prisma requires native query JSON.
            if self.identity.key_id
            else {"memory_id": "__no_match__"}  # mutable-ok: Prisma requires native query JSON.
        )
        return {  # mutable-ok: Prisma requires native query JSON.
            "namespace": {"startswith": "v2:"},  # mutable-ok: Prisma requires native query JSON.
            **(
                {"organization_id": self.identity.organization_id}  # mutable-ok: Prisma requires native query JSON.
                if not self.identity.dashboard
                else {}  # mutable-ok: Prisma requires native JSON.
            ),
            "OR": [  # mutable-ok: Prisma requires native JSON.
                owner,
                *(
                    [{"team_id": {"in": list(self.team_ids)}}] if self.team_ids and not write else []
                ),  # mutable-ok: Prisma requires native JSON.
            ],  # mutable-ok: Prisma requires native JSON.
        }


async def resolve_memory_access(prisma_client: object, identity: MemoryIdentity) -> MemoryAccess:
    primary: Final = memory_primary_client(prisma_client)
    settings: Final = await memory_settings(primary)
    user: Final = await UserRepository(primary).find_by_id(identity.user_id) if identity.user_id else None
    # Global roles come from normal authentication, including JWT and master-key grants.
    role: Final = identity.role
    admin_view: Final = role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    ids: Final = tuple(user.teams or ()) if user else ()
    team_ids: Final = tuple(dict.fromkeys((*ids, *((identity.team_id,) if identity.team_id else ()))))
    rows: Final = (
        await TeamRepository(primary).table.find_many(
            where={"team_id": {"in": list(team_ids)}},  # mutable-ok: Prisma requires native query JSON.
            take=len(team_ids),
        )
        if team_ids
        else ()
    )
    teams: Final = tuple(LiteLLM_TeamTable.model_validate(row.model_dump()) for row in rows)
    context_team: Final = next((team for team in teams if team.team_id == identity.team_id), None)
    current: Final = replace(
        identity,
        organization_id=identity.organization_id or (context_team.organization_id if context_team else None),
        read_only=identity.read_only
        or role in (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY),
    )
    eligible: Final = tuple(
        team for team in teams if current.dashboard or admin_view or team.organization_id == current.organization_id
    )
    auth: Final = UserAPIKeyAuth(user_id=current.user_id, user_role=role)
    permitted: Final = permitted_record_teams(auth, eligible, KeyManagementRoutes.MEMORY_READ)
    return MemoryAccess(
        identity=current,
        settings=settings,
        team_ids=permitted,
        admin_view=admin_view,
        permission_revision=memory_digest(
            current.namespace,
            role,
            str(admin_view),
            *(
                f"{team.team_id}:{team.organization_id}"
                for team in sorted(eligible, key=lambda item: item.team_id)
                if team.team_id in permitted
            ),
        ),
    )
