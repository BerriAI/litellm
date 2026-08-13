"""
Reverse sync for the team side of the team <-> access group relationship.

`litellm_accessgrouptable.assigned_team_ids` and `litellm_teamtable.access_group_ids`
are two copies of the same relationship, and both are read: the access group's
attached-teams view reads the former, and so does the key-side grant check in
`auth_checks.get_authorized_resources_from_key_access_groups`. The access-group
endpoints maintain both copies already; this module is what the team write paths
call so an edit from that side is mirrored back.

It deliberately lives outside `access_group_endpoints`, which is a lazily
registered feature router (see `_lazy_features.LAZY_FEATURES`). Importing that
module eagerly from `team_endpoints` would put it in `sys.modules` without its
router ever being included, which drops its routes from the OpenAPI schema.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.auth.auth_checks import _delete_cache_access_object

# hashtext collisions only cost two unrelated teams a little serialization, and the
# lock is never taken by the access-group endpoints, so it cannot join their
# access-group-then-team lock order to form a cycle.
_LOCK_TEAM_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"

_READ_TEAM_SQL: Final = 'SELECT access_group_ids FROM "LiteLLM_TeamTable" WHERE team_id = $1'

_ATTACH_SQL: Final = """
UPDATE "LiteLLM_AccessGroupTable"
SET assigned_team_ids = array_append(COALESCE(assigned_team_ids, ARRAY[]::TEXT[]), $1)
WHERE access_group_id = ANY($2::TEXT[])
  AND NOT ($1 = ANY(COALESCE(assigned_team_ids, ARRAY[]::TEXT[])))
RETURNING access_group_id
"""

_DETACH_SQL: Final = """
UPDATE "LiteLLM_AccessGroupTable"
SET assigned_team_ids = array_remove(assigned_team_ids, $1)
WHERE $1 = ANY(COALESCE(assigned_team_ids, ARRAY[]::TEXT[]))
  AND NOT (access_group_id = ANY($2::TEXT[]))
RETURNING access_group_id
"""


class _ChangedGroup(BaseModel):
    access_group_id: str


class _TeamGroups(BaseModel):
    access_group_ids: tuple[str, ...] | None = None


_ChangedGroups: Final = TypeAdapter(tuple[_ChangedGroup, ...])
_TeamRows: Final = TypeAdapter(tuple[_TeamGroups, ...])


class _RawQueryExecutor(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class _Transaction(Protocol):
    async def __aenter__(self) -> _RawQueryExecutor: ...

    async def __aexit__(self, *exc_info: object) -> None: ...


class _PrismaDb(Protocol):
    def tx(self) -> _Transaction: ...


class _PrismaClient(Protocol):
    @property
    def db(self) -> _PrismaDb: ...


async def invalidate_access_group_cache(access_group_id: str) -> None:
    """
    Drop an access group entry from both the in-memory and Redis caches.

    Uses a lazy import of user_api_key_cache and proxy_logging_obj from proxy_server
    to avoid circular imports, following the same pattern as key_management_endpoints.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _delete_cache_access_object(
        access_group_id=access_group_id,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def sync_team_access_group_membership(prisma_client: _PrismaClient, team_id: str) -> None:
    """
    Reconcile every access group's `assigned_team_ids` against the team's own committed
    `access_group_ids`, so a team-side edit is visible on the access-group side.

    Call this after the team row is written, or after it is deleted: a team with no row
    reconciles to an empty set, which detaches it from every group.

    The team row is read here rather than passed in, under an advisory lock held for the
    rest of the transaction. That is what makes concurrent writes to the same team
    converge, since each mirror reconciles against the row as committed instead of
    against the snapshot its own caller happened to see. It also means a retry heals a
    sync that failed partway, where a before/after delta would compute nothing to do.

    Both mirror statements are set-based and mutate the array inside the statement, so a
    concurrent write for a different team cannot be lost the way a read-modify-write of
    the whole array can, and the pair commits together or not at all.
    """
    async with prisma_client.db.tx() as tx:
        await tx.query_raw(_LOCK_TEAM_SQL, team_id)
        team_rows: Final = _TeamRows.validate_python(await tx.query_raw(_READ_TEAM_SQL, team_id))
        desired: Final = list(team_rows[0].access_group_ids or ()) if team_rows else []
        attached: Final = _ChangedGroups.validate_python(await tx.query_raw(_ATTACH_SQL, team_id, desired))
        detached: Final = _ChangedGroups.validate_python(await tx.query_raw(_DETACH_SQL, team_id, desired))

    for group in (*attached, *detached):
        await invalidate_access_group_cache(group.access_group_id)
