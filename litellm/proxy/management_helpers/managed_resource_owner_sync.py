"""Carry key-owned managed resources over to a regenerated key's new hash."""

from __future__ import annotations

from typing import Final, Protocol

from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper
from litellm.repositories.table_repositories import ManagedObjectRepository

_KEY_OWNED_TABLES: Final = (
    "LiteLLM_ManagedFileTable",
    "LiteLLM_ManagedObjectTable",
    "LiteLLM_ManagedVectorStoreTable",
)


class _RawWriter(Protocol):
    async def execute_raw(self, query: str, *args: str) -> int: ...


def _raw_writer(prisma_client: object) -> _RawWriter:
    db: Final = ManagedObjectRepository(prisma_client).prisma_client.db  # pyright: ignore[reportAny]  # untyped Prisma client
    return writer_wrapper(db)  # pyright: ignore[reportAny, reportReturnType]  # untyped Prisma client behind the pin


async def repoint_key_owned_managed_resources(
    prisma_client: object,
    existing_key_row: LiteLLM_VerificationToken,
    previous_key_token: str,
    new_key_token: str,
) -> None:
    """A key with neither a user_id nor a team_id owns its managed files, batches and
    vector stores as ``key:<hashed token>`` (``resolve_resource_owner_id``), so a
    regenerated key would otherwise be denied its own resources."""
    if existing_key_row.user_id is not None or existing_key_row.team_id is not None:
        return
    writer: Final = _raw_writer(prisma_client)
    for table in _KEY_OWNED_TABLES:
        await writer.execute_raw(
            f'UPDATE "{table}" SET "created_by" = $2 WHERE "created_by" = $1',
            f"key:{previous_key_token}",
            f"key:{new_key_token}",
        )
