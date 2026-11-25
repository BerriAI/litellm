"""Warm the user, team, membership, org and project cache entries auth reads: one MGET, one DB query, one
pipeline write instead of one Redis GET (and one DB query when cold) per object. The per-object getters stay
the readers and the fallback, so enforcement never depends on this running."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._internal_context import service_target
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_IN_MEMORY_TTL, REGISTRY_ERROR_NEGATIVE_CACHE_TTL
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LiteLLM_ProjectTableCachedObj, UserAPIKeyAuth
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import (
    AUTH_OBJECTS_TARGET,
    UserApiKeyCache,
    get_management_object_ttl,
)
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.utils import PrismaClient
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories.key_owners_repository import KeyOwnersRepository

_RowKind: TypeAlias = Literal["user_row", "team_row", "membership_row", "organization_row", "project_row"]

_TEAM_MEMBERSHIP_AUTH_TTL: Final = 5
_RowValues: Final = TypeAdapter(dict[str, object])
_TEAM_BOUND_ROWS: Final = frozenset({"team_row", "membership_row"})
_REFRESH_STAMPED_ROWS: Final = frozenset({"team_row", "project_row"})


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


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    cache_key: AuthObject
    row: _RowKind
    model_type: type[BaseModel]
    ttl: float | None


def _iter_entries(refs: AuthObjectRefs, management_ttl: float) -> Iterator[_CacheEntry]:
    if refs.user_id is not None:
        yield _CacheEntry(AuthObject.user(refs.user_id), "user_row", LiteLLM_UserTable, management_ttl)
    if refs.team_id is not None:
        yield _CacheEntry(AuthObject.team(refs.team_id), "team_row", LiteLLM_TeamTableCachedObj, management_ttl)
    if refs.team_id is not None and refs.membership_user_id is not None:
        yield _CacheEntry(
            AuthObject.team_membership(team_id=refs.team_id, user_id=refs.membership_user_id),
            "membership_row",
            LiteLLM_TeamMembership,
            _TEAM_MEMBERSHIP_AUTH_TTL,
        )
        yield _CacheEntry(
            AuthObject.team_membership_reservation(user_id=refs.membership_user_id, team_id=refs.team_id),
            "membership_row",
            LiteLLM_TeamMembership,
            None,
        )
    if refs.organization_id is not None:
        yield _CacheEntry(
            AuthObject.org(refs.organization_id), "organization_row", LiteLLM_OrganizationTable, DEFAULT_IN_MEMORY_TTL
        )
        yield _CacheEntry(
            AuthObject.org_with_budget(refs.organization_id),
            "organization_row",
            LiteLLM_OrganizationTable,
            DEFAULT_IN_MEMORY_TTL,
        )
    if refs.project_id is not None:
        yield _CacheEntry(
            AuthObject.project(refs.project_id), "project_row", LiteLLM_ProjectTableCachedObj, management_ttl
        )


def _entries(refs: AuthObjectRefs, cache: UserApiKeyCache) -> tuple[_CacheEntry, ...]:
    return tuple(_iter_entries(refs, get_management_object_ttl(cache)))


def _missing_in_memory(entries: Sequence[_CacheEntry], cache: UserApiKeyCache) -> tuple[_CacheEntry, ...]:
    return tuple(
        entry
        for entry in entries
        if (
            not cache.has_local_team(entry.cache_key)
            if entry.row == "team_row"
            else cache.get_local(entry.cache_key) is None
        )
    )


def _nothing_to_arm() -> None:
    return None


def _sorted_by_wire_key(keys: Iterable[AuthObject]) -> tuple[AuthObject, ...]:
    return tuple(sorted(keys, key=lambda auth_object: auth_object.key))


async def _fill_from_redis(
    entries: Sequence[_CacheEntry], cache: UserApiKeyCache, before_read: Callable[[], None]
) -> None:
    """On the request's native pipeline when one is open; a failed read is a miss, as a failed MGET was."""
    if not entries:
        return
    user_entry: Final = next((entry for entry in entries if entry.row == "user_row"), None)
    generation: Final = cache.store.local_generation(user_entry.cache_key) if user_entry is not None else None
    before_read()
    try:
        with service_target(AUTH_OBJECTS_TARGET):
            found: Final = await cache.store.read_missing(_sorted_by_wire_key(entry.cache_key for entry in entries))
    except Exception as e:  # noqa: BLE001  # the DB fill below takes over, as it does after a failed MGET
        verbose_proxy_logger.debug("auth prefetch Redis read failed, filling from the database: %s", e)
        return
    for entry, value in ((entry, found.get(entry.cache_key)) for entry in entries):
        if value is not None:
            ttl: Final = cache.default_in_memory_ttl if entry.ttl is None else entry.ttl
            if entry is user_entry and generation is not None:
                _ = cache.store.set_local_if_current(generation, entry.cache_key, value, ttl)
            else:
                cache.store.set_local(entry.cache_key, value, ttl)


def _validate_row(
    row_value: object, model_type: type[BaseModel], row: _RowKind, refreshed_at: float
) -> BaseModel | None:
    if row_value is None:
        return None
    try:
        columns: Final = _RowValues.validate_python(row_value)
        if row in _REFRESH_STAMPED_ROWS:
            stamped: Final = {**columns, "last_refreshed_at": refreshed_at}
            return model_type.model_validate(stamped)
        return model_type.model_validate(columns)
    except ValidationError as e:
        verbose_proxy_logger.warning("auth prefetch: %s did not validate as %s: %s", row, model_type.__name__, e)
        return None


async def _fetch_rows(refs: AuthObjectRefs, kinds: frozenset[_RowKind]) -> Mapping[str, object]:
    with service_target(AUTH_OBJECTS_TARGET):
        return await KeyOwnersRepository().find(
            user_id=refs.user_id if "user_row" in kinds else None,
            team_id=refs.team_id if kinds & _TEAM_BOUND_ROWS else None,
            membership_user_id=refs.membership_user_id if "membership_row" in kinds else None,
            organization_id=refs.organization_id if "organization_row" in kinds else None,
            project_id=refs.project_id if "project_row" in kinds else None,
        )


async def _write_back(entries: Sequence[tuple[_CacheEntry, BaseModel]], cache: UserApiKeyCache) -> None:
    payloads: Final = tuple(
        (entry.cache_key, CacheCodec.serialize(value, model_type=entry.model_type), entry.ttl)
        for entry, value in entries
    )
    with service_target(AUTH_OBJECTS_TARGET):
        await cache.store.write_back(payloads)


async def _fill_from_db(refs: AuthObjectRefs, entries: Sequence[_CacheEntry], cache: UserApiKeyCache) -> None:
    if not entries:
        return
    model_for: Final[Mapping[_RowKind, type[BaseModel]]] = MappingProxyType(
        {entry.row: entry.model_type for entry in entries}
    )
    user_entry: Final = next((entry for entry in entries if entry.row == "user_row"), None)
    user_lease: Final = await cache.lease(user_entry.cache_key) if user_entry is not None else None
    rows: Final = await bounded_db_lookup(_fetch_rows(refs, frozenset(model_for)), name="key owners", tracker=None)
    refreshed_at: Final = time.time()
    objects: Final[Mapping[_RowKind, BaseModel | None]] = MappingProxyType(
        {row: _validate_row(rows.get(row), model_type, row, refreshed_at) for row, model_type in model_for.items()}
    )
    user: Final = objects.get("user_row")
    if user_lease is not None and user_entry is not None and user is not None:
        _ = await cache.async_set_leased(user_lease, user, user_entry.ttl)
    writes: Final = tuple(
        (entry, value) for entry in entries if entry.row != "user_row" and (value := objects[entry.row]) is not None
    )
    if writes:
        await _write_back(writes, cache)


async def prefetch_auth_objects(
    refs: AuthObjectRefs,
    user_api_key_cache: UserApiKeyCache,
    prisma_client: PrismaClient | None,
    before_read: Callable[[], None] = _nothing_to_arm,
) -> None:
    """Best effort: any failure leaves the per-object getters to fetch as before. ``before_read`` runs right before
    a Redis read goes out, so a caller can declare other reads into the same round trip."""
    try:
        missing: Final = _missing_in_memory(_entries(refs, user_api_key_cache), user_api_key_cache)
        if user_api_key_cache.has_redis:
            await _fill_from_redis(missing, user_api_key_cache, before_read)
        if prisma_client is None:
            return
        reader: Final = attribute_or_none(prisma_client, "read_db")
        has_replica: Final = reader is not None and reader is not prisma_client.writer_db
        # With a replica, leave a cold user to the writer-first lookup so its fallback has one writer deadline.
        db_entries: Final = tuple(entry for entry in missing if entry.row != "user_row" or not has_replica)
        await _fill_from_db(refs, _missing_in_memory(db_entries, user_api_key_cache), user_api_key_cache)
    except Exception as e:  # noqa: BLE001  # warm-up only; the getters enforce and fail closed on their own
        verbose_proxy_logger.warning("auth prefetch skipped, falling back to per-object lookups: %s", e)


def _identity_memory_ttl(value: object, management_ttl: float) -> float:
    """A registry stored as a string is a sentinel, written with the shorter of the two registry TTLs."""
    return min(REGISTRY_ERROR_NEGATIVE_CACHE_TTL, management_ttl) if isinstance(value, str) else management_ttl


async def prefetch_identity_keys(
    cache_keys: Sequence[AuthObject],
    user_api_key_cache: UserApiKeyCache,
    before_read: Callable[[], None] = _nothing_to_arm,
) -> None:
    """Warm the entries auth reads before it knows the key's owners (the key object, the end user and the two
    registries) in one read on the request's native pipeline. Keys it finds absent stay noted on the request, so the
    per-key getters that follow go to the database without a GET of their own. Best effort, like the
    owner prefetch: the getters read and enforce on their own."""
    try:
        if not user_api_key_cache.has_redis:
            return
        shared: Final = user_api_key_cache.store
        missing: Final = tuple(key for key in dict.fromkeys(cache_keys) if shared.get_local(key) is None)
        if not missing:
            return
        before_read()
        with service_target(AUTH_OBJECTS_TARGET):
            found: Final = await shared.read_missing(_sorted_by_wire_key(missing))
        management_ttl: Final = get_management_object_ttl(user_api_key_cache)
    except Exception as e:  # noqa: BLE001  # warm-up only; the getters read Redis and the database on their own
        verbose_proxy_logger.warning("auth identity prefetch skipped, falling back to per-key lookups: %s", e)
        return
    for key, value in ((key, found.get(key)) for key in missing):
        if value is not None:
            shared.set_local(key, value, _identity_memory_ttl(value, management_ttl))
