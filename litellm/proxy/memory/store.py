import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm.proxy.memory.content import fuzzy_memories, redact_memory
from litellm.proxy.memory.policy import MemoryAccess, memory_digest, memory_primary_client, resolve_memory_access
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import MemoryRepository
from litellm.repositories.unit_of_work import prisma_transaction
from litellm.types.memory_v2 import MemoryCapture, MemoryEntry, MemorySearch

if TYPE_CHECKING:
    from prisma.models import LiteLLM_MemoryTable

_METADATA: Final = TypeAdapter(dict[str, object])
_MAX_NAMESPACE_ENTRIES: Final = 1000


def memory_entry(row: "LiteLLM_MemoryTable") -> MemoryEntry:
    metadata: Final = (
        _METADATA.validate_python(row.metadata)
        if isinstance(row.metadata, dict)
        else {  # mutable-ok: Prisma serializes these as native JSON containers.
        }
    )
    title: Final = metadata.get("title")
    evidence: Final = metadata.get("evidence")
    return MemoryEntry.model_validate(
        {  # mutable-ok: Pydantic validates stored JSON metadata and database fields together.
            "memory_id": row.memory_id,
            "key": row.key.rsplit(":", 1)[-1],
            "title": title if isinstance(title, str) else row.key,
            "content": row.value,
            "evidence": evidence if isinstance(evidence, str) else "",
            "updated_at": row.updated_at,
            "created_at": row.created_at,
            "actor": row.created_by,
            **{  # mutable-ok: Pydantic validates these stored JSON metadata fields.
                name: value
                for name in ("when_to_use", "scope", "kind", "certainty", "source")
                if isinstance(value := metadata.get(name), str)
            },
        }
    )


class MemoryStore:
    def __init__(self, prisma_client: object, access: MemoryAccess) -> None:
        self.prisma_client = memory_primary_client(prisma_client)
        self.access = access
        self.table = MemoryRepository(self.prisma_client).table

    async def authorize_namespace(self, *, write: bool = False, require_active: bool = True) -> str:
        current: Final = await resolve_memory_access(self.prisma_client, self.access.identity)
        if (
            current.namespace is None
            or current.namespace != self.access.namespace
            or require_active
            and not current.active
            or write
            and current.identity.read_only
        ):
            raise HTTPException(status_code=403, detail="Memory is not available under the current policy")
        return current.namespace

    async def search(self, search: MemorySearch, *, require_active: bool = True) -> tuple[MemoryEntry, ...]:
        entries: Final = await self.entries(require_active=require_active)
        ranked: Final = await asyncio.to_thread(fuzzy_memories, search.query, entries)
        return tuple(entry for entry, _, _ in ranked[search.offset : search.offset + search.limit])

    async def entries(self, *, require_active: bool = True) -> tuple[MemoryEntry, ...]:
        namespace: Final = await self.authorize_namespace(require_active=require_active)
        rows: Final = await self.table.find_many(
            where={  # mutable-ok: Prisma serializes these as native JSON containers.
                "namespace": namespace,
            },
            order=[  # mutable-ok: Prisma serializes these as native JSON containers.
                {  # mutable-ok: Prisma serializes these as native JSON containers.
                    "updated_at": "desc"
                },
                {  # mutable-ok: Prisma serializes these as native JSON containers.
                    "memory_id": "asc"
                },
            ],
            take=_MAX_NAMESPACE_ENTRIES,
        )
        return tuple(memory_entry(row) for row in rows)

    async def read(self, memory_id: str) -> MemoryEntry:
        namespace: Final = await self.authorize_namespace()
        row: Final = await self.table.find_first(
            where={  # mutable-ok: Prisma serializes these as native JSON containers.
                "memory_id": memory_id,
                "namespace": namespace,
            }
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory_entry(row)

    async def capture(self, capture: MemoryCapture) -> MemoryEntry:
        return (await self.capture_many((capture,)))[0]

    async def capture_many(self, captures: tuple[MemoryCapture, ...]) -> tuple[MemoryEntry, ...]:
        namespace: Final = await self.authorize_namespace(write=True)
        if not captures:
            return ()
        async with prisma_transaction(self.prisma_client) as transaction:
            lock_key: Final = int(memory_digest("memory-quota", namespace)[:16], 16) - (1 << 63)
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
            table: Final = MemoryRepository(SimpleNamespace(db=transaction)).table
            saved: Final = tuple([await self._capture(capture, namespace, table) for capture in captures])
            await self.authorize_namespace(write=True)
            return saved

    async def _capture(
        self, capture: MemoryCapture, namespace: str, table: TableActions["LiteLLM_MemoryTable"]
    ) -> MemoryEntry:
        key: Final = f"memory-v2:{namespace}:{capture.key}"
        metadata: Final = {  # mutable-ok: Prisma query and write JSON.
            name: redact_memory(value)
            for name, value in (
                ("title", capture.title),
                ("evidence", capture.evidence),
                ("when_to_use", capture.when_to_use),
                ("scope", capture.scope),
                ("kind", capture.kind),
                ("certainty", capture.certainty),
                ("source", capture.source),
            )
        }
        content: Final = redact_memory(capture.content)
        data: Final = {  # mutable-ok: Prisma query and write JSON.
            "value": content,
            "metadata": json.dumps(metadata),
            "updated_by": self.access.identity.user_id or self.access.identity.key_id,
        }
        existing: Final = await table.find_unique(
            where={  # mutable-ok: Prisma query and write JSON.
                "key": key
            }
        )
        if existing is not None:
            if existing.namespace != namespace:
                raise HTTPException(status_code=409, detail="Memory key conflict")
            entry: Final = memory_entry(existing)
            if existing.value == content and all(getattr(entry, name) == value for name, value in metadata.items()):
                return entry
            if capture.expected_revision != existing.updated_at:
                raise HTTPException(status_code=409, detail="Read the current memory before replacing it")
            count: Final = await table.update_many(
                where={  # mutable-ok: Prisma query and write JSON.
                    "key": key,
                    "namespace": namespace,
                    "updated_at": capture.expected_revision,
                    "value": existing.value,
                    "metadata": {  # mutable-ok: Prisma query and write JSON.
                        "equals": json.dumps(existing.metadata)
                    },
                },
                data=data,
            )
            if count != 1:
                raise HTTPException(status_code=409, detail="Memory changed; read it again before replacing it")
            updated: Final = await table.find_unique(
                where={  # mutable-ok: Prisma query and write JSON.
                    "key": key
                }
            )
            if updated is None:
                raise HTTPException(status_code=409, detail="Memory no longer exists")
            return memory_entry(updated)
        if capture.expected_revision is not None:
            raise HTTPException(status_code=409, detail="Memory no longer exists")
        entries: Final = await table.count(
            where={  # mutable-ok: Prisma query and write JSON.
                "namespace": namespace
            }
        )
        if entries >= _MAX_NAMESPACE_ENTRIES:
            raise HTTPException(
                status_code=429, detail="Memory scope has reached 1000 entries; delete unused memories first"
            )
        created: Final = await table.create(
            data={  # mutable-ok: Prisma query and write JSON.
                **data,
                "memory_id": memory_digest(namespace, capture.key),
                "key": key,
                "namespace": namespace,
                "user_id": self.access.identity.user_id,
                "team_id": self.access.identity.team_id,
                "created_by": self.access.identity.user_id or self.access.identity.key_id,
            }
        )
        return memory_entry(created)

    async def delete(self, memory_id: str) -> bool:
        namespace: Final = await self.authorize_namespace(write=True, require_active=False)
        return bool(
            await self.table.delete_many(
                where={  # mutable-ok: Prisma serializes these as native JSON containers.
                    "memory_id": memory_id,
                    "namespace": namespace,
                }
            )
        )
