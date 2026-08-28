"""
Reverse sync for the key side of the key <-> access group relationship.

`litellm_accessgrouptable.assigned_key_ids` and `litellm_verificationtoken.access_group_ids`
are the two halves of one relationship and BOTH are read: the access group's
attached-keys view reads the former, and so does the grant check in
`auth_checks.get_authorized_resources_from_key_access_groups`, which authorizes a
key only when the group lists the key's token (or the key's team). The access-group
endpoints maintain both halves already; this module is what the key write paths call
so an edit from that side is mirrored back.

Every write is a single guarded statement rather than a read-modify-write. Prisma has no
atomic scalar-list removal (see `TeamRepository.remove_member`), and the read-modify-write
it otherwise forces is not safe here: a lost update would put an already revoked token back
into a group and restore its grants, or drop a grant an admin just made. The guards also
make each statement idempotent, so a retry cannot duplicate an entry. Each statement covers
every group the request touches at once, so the size of the caller's id list does not turn
into a matching number of round trips, and returns the ids it actually moved so only those
groups are dropped from cache.

Each statement is an `UPDATE ... RETURNING` run inside a transaction so it lands on the
writer. `RoutingPrismaWrapper` routes a bare `query_raw` to the read replica, where the
write fails with `cannot execute UPDATE in a read-only transaction`; it routes `tx()` to
the writer, so wrapping the statement keeps the write off the read-only reader while still
yielding its RETURNING rows. The cache is dropped only after the transaction commits, so a
rolled-back write never evicts a still-valid entry. This mirrors `access_group_team_sync`.

It deliberately lives outside `access_group_endpoints`, which is a lazily
registered feature router (see `_lazy_features.LAZY_FEATURES`). Importing that
module eagerly from `key_management_endpoints` would put it in `sys.modules`
without its router ever being included, which drops its routes from the OpenAPI
schema.
"""

from collections.abc import Sequence
from typing import Final, Protocol

from pydantic import BaseModel

from litellm.proxy._types import (
    LiteLLM_VerificationToken,
    RegenerateKeyRequest,
    UpdateKeyRequest,
)
from litellm.proxy.auth.auth_checks import (
    _delete_cache_access_object,  # pyright: ignore[reportPrivateUsage]  # the access-group endpoints reach for this same cache primitive
)
from litellm.repositories.table_repositories import AccessGroupRepository


class _MovedGroupRow(BaseModel):
    access_group_id: str


class _AccessGroupSyncTx(Protocol):
    async def query_raw(self, query: str, *args: str | Sequence[str]) -> Sequence[object]: ...


class _Transaction(Protocol):
    async def __aenter__(self) -> _AccessGroupSyncTx: ...

    async def __aexit__(self, *exc_info: object) -> None: ...


class _PrismaDb(Protocol):
    def tx(self) -> _Transaction: ...


_ATTACH_KEY_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "assigned_key_ids" = array_append("assigned_key_ids", $1) '
    'WHERE "access_group_id" = ANY($2::text[]) AND NOT ($1 = ANY("assigned_key_ids")) '
    'RETURNING "access_group_id"'
)

_DETACH_KEY_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "assigned_key_ids" = array_remove("assigned_key_ids", $1) '
    'WHERE "access_group_id" = ANY($2::text[]) AND $1 = ANY("assigned_key_ids") '
    'RETURNING "access_group_id"'
)

_REPOINT_KEY_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "assigned_key_ids" = array_append(array_remove(array_remove("assigned_key_ids", $1), $2), $2) '
    'WHERE $1 = ANY("assigned_key_ids") '
    'RETURNING "access_group_id"'
)


def _prisma_db(prisma_client: object) -> _PrismaDb:
    """Narrow the untyped Prisma client down to the transaction accessor this module needs."""
    return AccessGroupRepository(prisma_client).prisma_client.db  # pyright: ignore[reportAny]  # untyped Prisma client


async def _invalidate_access_group_cache(access_group_id: str) -> None:
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


async def _invalidate_moved_groups(moved_rows: Sequence[object]) -> None:
    for row in moved_rows:
        await _invalidate_access_group_cache(_MovedGroupRow.model_validate(row).access_group_id)


async def _run_membership_statement(
    prisma_client: object, sql: str, *args: str | Sequence[str]
) -> Sequence[object]:
    """Run one guarded membership `UPDATE ... RETURNING` on the writer and return the rows it moved.

    Wrapped in a transaction because `RoutingPrismaWrapper` routes a bare `query_raw` to the
    read replica, where the write fails with `cannot execute UPDATE in a read-only transaction`,
    whereas it routes `tx()` to the writer.
    """
    async with _prisma_db(prisma_client).tx() as tx:
        return await tx.query_raw(sql, *args)


async def _write_membership(prisma_client: object, sql: str, access_group_ids: frozenset[str], key_token: str) -> None:
    """Run one guarded membership statement for every listed group, dropping the cache of those it moved."""
    if not access_group_ids:
        return
    moved: Final = await _run_membership_statement(prisma_client, sql, key_token, sorted(access_group_ids))
    await _invalidate_moved_groups(moved)


async def sync_key_access_group_membership(
    prisma_client: object,
    key_token: str,
    previous_access_group_ids: Sequence[str] | None,
    updated_access_group_ids: Sequence[str] | None,
) -> None:
    """Mirror a key-side change to `access_group_ids` onto each access group's `assigned_key_ids`."""
    previous: Final = frozenset(previous_access_group_ids or ())
    updated: Final = frozenset(updated_access_group_ids or ())

    await _write_membership(prisma_client, _ATTACH_KEY_SQL, updated - previous, key_token)
    await _write_membership(prisma_client, _DETACH_KEY_SQL, previous - updated, key_token)


async def sync_key_update_access_group_membership(
    prisma_client: object,
    key_token: str,
    data: UpdateKeyRequest | RegenerateKeyRequest,
    existing_key_row: LiteLLM_VerificationToken,
) -> None:
    """
    Mirror a key UPDATE onto the group side, honouring `exclude_unset` semantics.

    The key row is written from `model_dump(exclude_unset=True)`, so a request that never
    mentions `access_group_ids` leaves the key's own list alone and must leave the group's
    copy alone too. Reading the attribute instead of `model_fields_set` would see None on
    every unrelated edit and withdraw the token from every group it belongs to.
    """
    if "access_group_ids" not in data.model_fields_set:
        return
    await sync_key_access_group_membership(
        prisma_client=prisma_client,
        key_token=key_token,
        previous_access_group_ids=existing_key_row.access_group_ids,
        updated_access_group_ids=data.access_group_ids,
    )


async def sync_key_regeneration_access_group_membership(
    prisma_client: object,
    previous_key_token: str,
    new_key_token: str,
    data: RegenerateKeyRequest | None,
    existing_key_row: LiteLLM_VerificationToken,
) -> None:
    """
    Re-point every group's copy from the old token to the regenerated one.

    Regeneration replaces the token, which is the identity `assigned_key_ids` stores, so
    leaving the old hash behind both points the group at a row that no longer exists and
    denies the regenerated key the group's grants. The swap is driven by the groups that
    hold the old token when the statement runs, not by the key row read earlier, so a group
    edited in between is neither resurrected nor skipped. Removing the new token before
    appending it keeps a re-run from duplicating it.
    """
    moved: Final = await _run_membership_statement(prisma_client, _REPOINT_KEY_SQL, previous_key_token, new_key_token)
    await _invalidate_moved_groups(moved)
    if data is not None:
        await sync_key_update_access_group_membership(
            prisma_client=prisma_client,
            key_token=new_key_token,
            data=data,
            existing_key_row=existing_key_row,
        )
