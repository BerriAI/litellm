from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from litellm.proxy.memory.policy import memory_digest, memory_primary_client
from litellm.proxy.memory.store import MemoryStore
from litellm.repositories.table_repositories import MemoryContinuationRepository
from litellm.repositories.unit_of_work import prisma_transaction

_MAX_PATCH_BYTES: Final = 1024 * 1024
_MAX_PATCHES: Final = 256
_MAX_NAMESPACE_BYTES: Final = 32 * 1024 * 1024


async def cleanup_memory_continuations(prisma_client: object) -> None:
    async with prisma_transaction(memory_primary_client(prisma_client)) as transaction:
        await transaction.execute_raw(
            'DELETE FROM "LiteLLM_MemoryContinuation" WHERE id IN '
            '(SELECT id FROM "LiteLLM_MemoryContinuation" WHERE expires_at <= $1::timestamp '
            "ORDER BY expires_at LIMIT 1000 FOR UPDATE SKIP LOCKED)",
            datetime.now(timezone.utc),
        )


class MemoryContinuation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    response: Mapping[str, object] | None = None
    upstream_ids: tuple[str, ...] = ()
    pending_results: tuple[Mapping[str, object], ...] = ()
    permission_revision: str | None = None


class MemoryContinuations:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.table = MemoryContinuationRepository(store.prisma_client).table

    def identifier(self, response_id: str) -> str:
        return memory_digest(
            self.store.access.namespace,
            self.store.access.identity.key_id or self.store.access.identity.user_id,
            "aresponses",
            response_id,
        )

    async def load_response(self, response_id: str) -> MemoryContinuation | None:
        namespace: Final = await self.store.authorize_namespace()
        row: Final = await self.table.find_first(
            where={  # mutable-ok: Prisma requires native query and write JSON.
                "id": self.identifier(response_id),
                "namespace": namespace,
                "key_id": self.store.access.identity.key_id or self.store.access.identity.user_id or "",
                "expires_at": {
                    "gt": datetime.now(timezone.utc)
                },  # mutable-ok: Prisma requires native query and write JSON.
            }
        )
        if row is None:
            return None
        patch: Final = MemoryContinuation.model_validate(row.payload)
        if patch.permission_revision != self.store.access.permission_revision:
            raise HTTPException(status_code=403, detail="Memory permissions changed; start a new conversation")
        return patch

    async def save(self, response_id: str, patch: MemoryContinuation) -> None:
        namespace: Final = await self.store.authorize_namespace()
        payload: Final = patch.model_copy(
            update=MappingProxyType({"permission_revision": self.store.access.permission_revision})
        ).model_dump_json()
        if len(payload.encode()) > _MAX_PATCH_BYTES:
            raise HTTPException(status_code=413, detail="Memory response exceeds one megabyte")
        key_id: Final = self.store.access.identity.key_id or self.store.access.identity.user_id or ""
        now: Final = datetime.now(timezone.utc)
        async with prisma_transaction(self.store.prisma_client) as transaction:
            lock_key: Final = int(memory_digest("memory-continuation-quota", namespace)[:16], 16) - (1 << 63)
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
            table: Final = MemoryContinuationRepository(SimpleNamespace(db=transaction)).table
            await table.delete_many(
                where={"namespace": namespace, "expires_at": {"lte": now}}
            )  # mutable-ok: Prisma requires native query and write JSON.
            await table.upsert(
                where={"id": self.identifier(response_id)},  # mutable-ok: Prisma requires native query and write JSON.
                data={  # mutable-ok: Prisma requires native query and write JSON.
                    "create": {  # mutable-ok: Prisma requires native query and write JSON.
                        "id": self.identifier(response_id),
                        "namespace": namespace,
                        "key_id": key_id,
                        "payload": payload,
                        "expires_at": now + timedelta(hours=24),
                    },
                    "update": {
                        "payload": payload,
                        "expires_at": now + timedelta(hours=24),
                    },  # mutable-ok: Prisma requires native query and write JSON.
                },
            )
            await transaction.execute_raw(
                'DELETE FROM "LiteLLM_MemoryContinuation" WHERE id IN ('
                "SELECT id FROM (SELECT id, "
                "ROW_NUMBER() OVER (PARTITION BY key_id ORDER BY (id = $4) DESC, expires_at DESC, id) AS position, "
                "SUM(octet_length(payload::text)) OVER (ORDER BY (id = $4) DESC, expires_at DESC, id) AS bytes "
                'FROM "LiteLLM_MemoryContinuation" WHERE namespace = $1) retained '
                "WHERE position > $2 OR bytes > $3)",
                namespace,
                _MAX_PATCHES,
                _MAX_NAMESPACE_BYTES,
                self.identifier(response_id),
            )

    async def delete_response(self, response_id: str) -> None:
        namespace: Final = await self.store.authorize_namespace()
        await self.table.delete_many(
            where={  # mutable-ok: Prisma requires native query and write JSON.
                "namespace": namespace,
                "key_id": self.store.access.identity.key_id or self.store.access.identity.user_id or "",
                "id": self.identifier(response_id),
            }
        )
