"""Warm the user, team, membership, org and project cache entries auth reads: one MGET, one DB query, one
pipeline write instead of one Redis GET (and one DB query when cold) per object. The per-object getters stay
the readers and the fallback, so enforcement never depends on this running."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_IN_MEMORY_TTL
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LiteLLM_ProjectTableCachedObj, UserAPIKeyAuth
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    get_management_object_ttl,
    team_membership_auth_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.utils import PrismaClient

_RowKind: TypeAlias = Literal["user_row", "team_row", "membership_row", "organization_row", "project_row"]

_TEAM_MEMBERSHIP_AUTH_TTL: Final = 5
_RowValues: Final = TypeAdapter(dict[str, object])
_NO_ROWS: Final[Mapping[str, object]] = MappingProxyType({})
_TEAM_BOUND_ROWS: Final = frozenset({"team_row", "membership_row"})
_REFRESH_STAMPED_ROWS: Final = frozenset({"team_row", "project_row"})


def _lists_as_json(alias: str, columns: Sequence[str]) -> str:
    """Prisma reads a NULL scalar list as ``[]``; ``to_jsonb`` reads it as ``null``, which the models reject."""
    return ", ".join(f"'{column}', COALESCE(to_jsonb({alias}.{column}), '[]'::jsonb)" for column in columns)


_USER_LISTS: Final = _lists_as_json("u", ("teams", "models", "allowed_cache_controls", "policies"))
_TEAM_LISTS: Final = _lists_as_json(
    "t",
    (
        "admins",
        "members",
        "models",
        "team_member_permissions",
        "access_group_ids",
        "policies",
        "default_team_member_models",
    ),
)
_ORG_LISTS: Final = _lists_as_json("o", ("models",))
_PROJECT_LISTS: Final = _lists_as_json("p", ("models",))
_PERMISSION_LISTS: Final = _lists_as_json(
    "op",
    (
        "mcp_servers",
        "mcp_access_groups",
        "mcp_toolsets",
        "blocked_tools",
        "vector_stores",
        "agents",
        "agent_access_groups",
        "models",
        "search_tools",
        "skills",
    ),
)
_BUDGET_LISTS: Final = _lists_as_json("b", ("allowed_models",))


def _budget_json(owner_alias: str) -> str:
    return (
        f"(SELECT to_jsonb(b) || jsonb_build_object({_BUDGET_LISTS}) "
        f'FROM "LiteLLM_BudgetTable" b WHERE b.budget_id = {owner_alias}.budget_id)'
    )


def _permission_json(owner_alias: str) -> str:
    return (
        f"(SELECT to_jsonb(op) || jsonb_build_object({_PERMISSION_LISTS}) "
        f'FROM "LiteLLM_ObjectPermissionTable" op WHERE op.object_permission_id = {owner_alias}.object_permission_id)'
    )


_SQL: Final = f"""
SELECT
  (
    SELECT to_jsonb(u) || jsonb_build_object(
      {_USER_LISTS},
      'organization_memberships',
      COALESCE((
        SELECT jsonb_agg(to_jsonb(om)) FROM "LiteLLM_OrganizationMembership" om WHERE om.user_id = u.user_id
      ), '[]'::jsonb)
    )
    FROM "LiteLLM_UserTable" u WHERE u.user_id = $1
  ) AS user_row,
  (
    SELECT to_jsonb(t) || jsonb_build_object(
      {_TEAM_LISTS},
      'litellm_model_table', (
        SELECT (to_jsonb(m) - 'aliases') || jsonb_build_object('model_aliases', m.aliases)
        FROM "LiteLLM_ModelTable" m WHERE m.id = t.model_id
      ),
      'object_permission', {_permission_json("t")}
    )
    FROM "LiteLLM_TeamTable" t WHERE t.team_id = $2
  ) AS team_row,
  (
    SELECT to_jsonb(tm) || jsonb_build_object('litellm_budget_table', {_budget_json("tm")})
    FROM "LiteLLM_TeamMembership" tm WHERE tm.user_id = $3 AND tm.team_id = $2
  ) AS membership_row,
  (
    SELECT to_jsonb(o) || jsonb_build_object(
      {_ORG_LISTS},
      'litellm_budget_table', {_budget_json("o")},
      'object_permission', {_permission_json("o")}
    )
    FROM "LiteLLM_OrganizationTable" o WHERE o.organization_id = $4
  ) AS organization_row,
  (
    SELECT to_jsonb(p) || jsonb_build_object(
      {_PROJECT_LISTS},
      'litellm_budget_table', {_budget_json("p")},
      'object_permission', {_permission_json("p")}
    )
    FROM "LiteLLM_ProjectTable" p WHERE p.project_id = $5
  ) AS project_row
"""


@dataclass(frozen=True, slots=True)
class AuthObjectRefs:
    """Ids of the objects a request's auth checks will read. ``None`` means not referenced."""

    user_id: str | None = None
    team_id: str | None = None
    membership_user_id: str | None = None
    organization_id: str | None = None
    project_id: str | None = None

    @classmethod
    def from_token(cls, token: UserAPIKeyAuth) -> AuthObjectRefs:
        has_membership: Final = token.team_id is not None and token.user_id is not None
        return cls(
            user_id=token.user_id,
            team_id=token.team_id,
            membership_user_id=token.user_id if has_membership else None,
            organization_id=token.org_id,
            project_id=token.project_id,
        )


class _InMemoryCache(Protocol):
    def get_cache(self, key: str) -> object: ...
    def set_cache(self, key: str, value: object, *, ttl: float | None = ...) -> None: ...


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    cache_key: str
    row: _RowKind
    model_type: type[BaseModel]
    ttl: float | None


def _iter_entries(refs: AuthObjectRefs, management_ttl: float) -> Iterator[_CacheEntry]:
    if refs.user_id is not None:
        yield _CacheEntry(refs.user_id, "user_row", LiteLLM_UserTable, management_ttl)
    if refs.team_id is not None:
        yield _CacheEntry(f"team_id:{refs.team_id}", "team_row", LiteLLM_TeamTableCachedObj, management_ttl)
    if refs.team_id is not None and refs.membership_user_id is not None:
        yield _CacheEntry(
            team_membership_auth_cache_key(team_id=refs.team_id, user_id=refs.membership_user_id),
            "membership_row",
            LiteLLM_TeamMembership,
            _TEAM_MEMBERSHIP_AUTH_TTL,
        )
        yield _CacheEntry(
            team_membership_reservation_cache_key(user_id=refs.membership_user_id, team_id=refs.team_id),
            "membership_row",
            LiteLLM_TeamMembership,
            None,
        )
    if refs.organization_id is not None:
        yield _CacheEntry(
            f"org_id:{refs.organization_id}", "organization_row", LiteLLM_OrganizationTable, DEFAULT_IN_MEMORY_TTL
        )
        yield _CacheEntry(
            f"org_id:{refs.organization_id}:with_budget",
            "organization_row",
            LiteLLM_OrganizationTable,
            DEFAULT_IN_MEMORY_TTL,
        )
    if refs.project_id is not None:
        yield _CacheEntry(f"project_id:{refs.project_id}", "project_row", LiteLLM_ProjectTableCachedObj, management_ttl)


def _entries(refs: AuthObjectRefs, cache: UserApiKeyCache) -> tuple[_CacheEntry, ...]:
    return tuple(_iter_entries(refs, get_management_object_ttl(cache)))


def _missing_in_memory(entries: Sequence[_CacheEntry], memory: _InMemoryCache) -> tuple[_CacheEntry, ...]:
    return tuple(entry for entry in entries if memory.get_cache(key=entry.cache_key) is None)


def _set_in_memory(memory: _InMemoryCache, cache_key: str, value: object, ttl: float | None) -> None:
    if ttl is None:
        memory.set_cache(key=cache_key, value=value)
    else:
        memory.set_cache(key=cache_key, value=value, ttl=ttl)


async def _fill_from_redis(entries: Sequence[_CacheEntry], redis_cache: RedisCache, memory: _InMemoryCache) -> None:
    if not entries:
        return
    found: Final = _RowValues.validate_python(
        await redis_cache.async_batch_get_cache(key_list=sorted(entry.cache_key for entry in entries))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped cache API
    )
    for entry, value in ((entry, found.get(entry.cache_key)) for entry in entries):
        if value is not None:
            _set_in_memory(memory, entry.cache_key, value, entry.ttl)


def _validate_row(
    row_value: object, model_type: type[BaseModel], row: _RowKind, refreshed_at: float
) -> BaseModel | None:
    if row_value is None:
        return None
    try:
        columns: Final = _RowValues.validate_python(row_value)
        if row in _REFRESH_STAMPED_ROWS:
            stamped: Final = {**columns, "last_refreshed_at": refreshed_at}  # mutable-ok: validators write into it
            return model_type.model_validate(stamped)
        return model_type.model_validate(columns)
    except ValidationError as e:
        verbose_proxy_logger.warning("auth prefetch: %s did not validate as %s: %s", row, model_type.__name__, e)
        return None


async def _fetch_rows(
    refs: AuthObjectRefs, kinds: frozenset[_RowKind], prisma_client: PrismaClient
) -> Mapping[str, object]:
    row: Final[object] = await prisma_client.db.query_first(  # pyright: ignore[reportAny]  # prisma types query_first as Any
        _SQL,
        refs.user_id if "user_row" in kinds else None,
        refs.team_id if kinds & _TEAM_BOUND_ROWS else None,
        refs.membership_user_id if "membership_row" in kinds else None,
        refs.organization_id if "organization_row" in kinds else None,
        refs.project_id if "project_row" in kinds else None,
    )
    return _RowValues.validate_python(row) if row is not None else _NO_ROWS


async def _write_back(entries: Sequence[tuple[_CacheEntry, BaseModel]], cache: UserApiKeyCache) -> None:
    payloads: Final = tuple(
        (entry.cache_key, CacheCodec.serialize(value, model_type=entry.model_type), entry.ttl)
        for entry, value in entries
    )
    memory: Final[_InMemoryCache] = cache.in_memory_cache
    for cache_key, payload, ttl in payloads:
        _set_in_memory(memory, cache_key, payload, cache.default_in_memory_ttl if ttl is None else ttl)
    if cache.redis_cache is not None:
        await cache.redis_cache.async_set_cache_pipeline_with_ttls(payloads)


async def _fill_from_db(
    refs: AuthObjectRefs, entries: Sequence[_CacheEntry], cache: UserApiKeyCache, prisma_client: PrismaClient
) -> None:
    if not entries:
        return
    model_for: Final[Mapping[_RowKind, type[BaseModel]]] = MappingProxyType(
        {entry.row: entry.model_type for entry in entries}
    )
    rows: Final = await _fetch_rows(refs, frozenset(model_for), prisma_client)
    refreshed_at: Final = time.time()
    objects: Final[Mapping[_RowKind, BaseModel | None]] = MappingProxyType(
        {row: _validate_row(rows.get(row), model_type, row, refreshed_at) for row, model_type in model_for.items()}
    )
    writes: Final = tuple((entry, value) for entry in entries if (value := objects[entry.row]) is not None)
    if writes:
        await _write_back(writes, cache)


async def prefetch_auth_objects(
    refs: AuthObjectRefs,
    user_api_key_cache: UserApiKeyCache,
    prisma_client: PrismaClient | None,
) -> None:
    """Best effort: any failure leaves the per-object getters to fetch as before."""
    try:
        memory: Final[_InMemoryCache] = user_api_key_cache.in_memory_cache
        missing: Final = _missing_in_memory(_entries(refs, user_api_key_cache), memory)
        if user_api_key_cache.redis_cache is not None:
            await _fill_from_redis(missing, user_api_key_cache.redis_cache, memory)
        if prisma_client is None:
            return
        await _fill_from_db(refs, _missing_in_memory(missing, memory), user_api_key_cache, prisma_client)
    except Exception as e:  # noqa: BLE001  # warm-up only; the getters enforce and fail closed on their own
        verbose_proxy_logger.warning("auth prefetch skipped, falling back to per-object lookups: %s", e)
