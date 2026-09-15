import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm.proxy.memory.content import fuzzy_memories, redact_memory
from litellm.proxy.memory.policy import MemoryAccess, memory_digest, memory_primary_client, resolve_memory_access
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import MemoryRepository
from litellm.repositories.unit_of_work import prisma_transaction
from litellm.types.memory_v2 import MemoryCapture, MemoryCatalogRequest, MemoryEntry, MemoryRecallRequest, MemorySearch

if TYPE_CHECKING:
    from prisma.models import LiteLLM_MemoryTable

_METADATA: Final = TypeAdapter(dict[str, object])
_MAX_OWNER_ENTRIES: Final = 1000
_PAGE_SIZE: Final = 128
RankedMemory = tuple[MemoryEntry, float, tuple[str, ...]]


def memory_entry(row: "LiteLLM_MemoryTable") -> MemoryEntry:
    metadata: Final = (
        _METADATA.validate_python(row.metadata) if isinstance(row.metadata, dict) else MappingProxyType({})
    )  # mutable-ok: Prisma requires native JSON.
    title: Final = metadata.get("title")
    evidence: Final = metadata.get("evidence")
    return MemoryEntry.model_validate(
        {  # mutable-ok: Pydantic validates stored JSON and database fields together.
            "memory_id": row.memory_id,
            "key": row.key.rsplit(":", 1)[-1],
            "title": title if isinstance(title, str) else row.key,
            "content": row.value,
            "evidence": evidence if isinstance(evidence, str) else "",
            "updated_at": row.updated_at,
            "created_at": row.created_at,
            "actor": row.created_by,
            "user_id": row.user_id,
            "team_id": row.team_id,
            **{  # mutable-ok: Prisma requires native JSON.
                name: value
                for name in ("when_to_use", "scope", "kind", "certainty", "source")
                if isinstance(value := metadata.get(name), str)
            },
        }
    )


def _where(*conditions: Mapping[str, object]) -> Mapping[str, object]:
    return {"AND": list(conditions)}  # mutable-ok: Prisma requires native query JSON.


def _before(cursor: tuple[datetime, str] | None) -> Mapping[str, object]:
    if cursor is None:
        return {}  # mutable-ok: Prisma requires native JSON.
    return {  # mutable-ok: Prisma requires native query JSON.
        "OR": [  # mutable-ok: Prisma requires native JSON.
            {"updated_at": {"lt": cursor[0]}},  # mutable-ok: Prisma requires native JSON.
            {  # mutable-ok: Prisma requires native JSON.
                "updated_at": cursor[0],
                "memory_id": {"gt": cursor[1]},  # mutable-ok: Prisma requires native JSON.
            },  # mutable-ok: Prisma requires native JSON.
        ]
    }


class MemoryStore:
    def __init__(self, prisma_client: object, access: MemoryAccess, *, actor: str | None = None) -> None:
        self.prisma_client = memory_primary_client(prisma_client)
        self.access = access
        self.actor = actor or access.identity.user_id or access.identity.key_id
        self.table = MemoryRepository(self.prisma_client).table

    async def authorize(self, *, write: bool = False, require_active: bool = True) -> MemoryAccess:
        current: Final = await resolve_memory_access(self.prisma_client, self.access.identity)
        if (
            not (current.identity.user_id or current.identity.key_id)
            or current.permission_revision != self.access.permission_revision
            or require_active
            and not current.active
            or write
            and current.identity.read_only
        ):
            raise HTTPException(status_code=403, detail="Memory access changed or is disabled")
        return current

    async def authorize_namespace(self, *, write: bool = False, require_active: bool = True) -> str:
        return (await self.authorize(write=write, require_active=require_active)).namespace

    def entry(self, row: "LiteLLM_MemoryTable") -> MemoryEntry:
        owned: Final = (
            row.user_id == self.access.identity.user_id
            if self.access.identity.user_id
            else bool(
                row.user_id is None and self.access.identity.key_id and row.owner_key_id == self.access.identity.key_id
            )
        )
        return memory_entry(row).model_copy(
            update={  # mutable-ok: Prisma requires native JSON.
                "can_edit": not self.access.identity.read_only and (owned or self.access.admin_view)
            }  # mutable-ok: Prisma requires native JSON.
        )

    async def _page(
        self, where: Mapping[str, object], *, limit: int, offset: int = 0, before: tuple[datetime, str] | None = None
    ) -> tuple[MemoryEntry, ...]:
        rows: Final = await self.table.find_many(
            where=_where(where, _before(before)),
            order=[{"updated_at": "desc"}, {"memory_id": "asc"}],  # mutable-ok: Prisma requires native query JSON.
            take=limit,
            skip=offset,
        )
        return tuple(self.entry(row) for row in rows)

    async def catalog(self, request: MemoryCatalogRequest) -> tuple[tuple[MemoryEntry, ...], int, str]:
        access: Final = await self.authorize()
        where: Final = access.visible_rows()
        total: Final = await self.table.count(where=where)
        entries: Final = await self._page(where, limit=request.limit, offset=request.offset)
        latest: Final = await self._page(where, limit=1) if request.offset else entries[:1]
        await self.authorize()
        return (
            entries,
            total,
            memory_digest(str(total), *(entry.memory_id + entry.updated_at.isoformat() for entry in latest)),
        )

    async def _ranked(
        self,
        query: str,
        where: Mapping[str, object],
        keep: int,
        *,
        scope: str | None = None,
        recent_first: bool = False,
    ) -> tuple[tuple[RankedMemory, ...], int]:
        return await asyncio.wait_for(
            self._ranked_pages(query, where, keep, scope=scope, recent_first=recent_first), timeout=15
        )

    async def _ranked_pages(
        self, query: str, where: Mapping[str, object], keep: int, *, scope: str | None, recent_first: bool
    ) -> tuple[tuple[RankedMemory, ...], int]:
        best: tuple[RankedMemory, ...] = ()  # rebind-ok: Bounded top results are replaced after each database page.
        total = 0  # rebind-ok: Count matches across bounded database pages.
        cursor: tuple[datetime, str] | None = None  # rebind-ok: Advance the database cursor after each page.
        while True:
            page = await self._page(where, limit=_PAGE_SIZE, before=cursor)
            candidates = tuple(entry for entry in page if scope is None or scope.casefold() in entry.scope.casefold())
            matches = await asyncio.to_thread(fuzzy_memories, query, candidates)
            total = total + len(matches)
            combined = (*best, *matches)
            best = tuple(
                sorted(combined, key=lambda item: (-item[0].updated_at.timestamp(), item[0].memory_id))
                if recent_first
                else sorted(combined, key=lambda item: (-item[1], item[0].memory_id))
            )[:keep]
            if len(page) < _PAGE_SIZE:
                return best, total
            cursor = (page[-1].updated_at, page[-1].memory_id)

    async def recall(self, request: MemoryRecallRequest) -> tuple[tuple[RankedMemory, ...], int]:
        access: Final = await self.authorize()
        ranked: Final = await self._ranked(request.query, access.visible_rows(), request.limit, scope=request.scope)
        await self.authorize()
        return ranked

    async def search(
        self,
        search: MemorySearch,
        *,
        require_active: bool = True,
        recent_first: bool = False,
        before: tuple[datetime, str] | None = None,
        team_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        access: Final = await self.authorize(require_active=require_active)
        where: Final = _where(
            access.visible_rows(),
            {"team_id": team_id} if team_id else {},  # mutable-ok: Prisma requires native JSON.
            {"user_id": user_id} if user_id else {},  # mutable-ok: Prisma requires native JSON.
            _before(before),
        )
        result: Final[tuple[MemoryEntry, ...]]
        if not search.query.strip():
            result = await self._page(where, limit=search.limit, offset=search.offset)
        else:
            ranked, _ = await self._ranked(search.query, where, search.offset + search.limit, recent_first=recent_first)
            result = tuple(entry for entry, _, _ in ranked[search.offset :])
        await self.authorize(require_active=require_active)
        return result

    async def read(self, memory_id: str, *, require_active: bool = True) -> MemoryEntry:
        access: Final = await self.authorize(require_active=require_active)
        row: Final = await self.table.find_first(
            where=_where(access.visible_rows(), {"memory_id": memory_id})  # mutable-ok: Prisma requires native JSON.
        )  # mutable-ok: Prisma requires native JSON.
        if row is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        await self.authorize(require_active=require_active)
        return self.entry(row)

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
            await MemoryStore(SimpleNamespace(db=transaction), self.access).authorize(write=True)
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
            "updated_by": self.actor,
        }
        existing: Final = await table.find_unique(
            where={  # mutable-ok: Prisma query and write JSON.
                "key": key
            }
        )
        if existing is not None:
            if existing.namespace != namespace:
                raise HTTPException(status_code=409, detail="Memory key conflict")
            entry: Final = self.entry(existing)
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
            return self.entry(updated)
        if capture.expected_revision is not None:
            raise HTTPException(status_code=409, detail="Memory no longer exists")
        entries: Final = await table.count(
            where={  # mutable-ok: Prisma query and write JSON.
                "namespace": namespace
            }
        )
        if entries >= _MAX_OWNER_ENTRIES:
            raise HTTPException(
                status_code=429, detail="Memory storage limit reached for this owner; contact your administrator"
            )
        created: Final = await table.create(
            data={  # mutable-ok: Prisma query and write JSON.
                **data,
                "memory_id": memory_digest(namespace, capture.key),
                "key": key,
                "namespace": namespace,
                "user_id": self.access.identity.user_id,
                "team_id": self.access.identity.team_id,
                "organization_id": self.access.identity.organization_id,
                "owner_key_id": self.access.identity.key_id,
                "created_by": self.actor,
            }
        )
        return self.entry(created)

    async def update(self, memory_id: str, capture: MemoryCapture) -> MemoryEntry:
        access: Final = await self.authorize(write=True, require_active=False)
        async with prisma_transaction(self.prisma_client) as transaction:
            table: Final = MemoryRepository(SimpleNamespace(db=transaction)).table
            row: Final = await table.find_first(
                where=_where(
                    access.visible_rows(write=True), {"memory_id": memory_id}
                )  # mutable-ok: Prisma requires native JSON.
            )  # mutable-ok: Prisma requires native JSON.
            if row is None or row.namespace is None:
                raise HTTPException(status_code=404, detail="Memory not found")
            if row.key.rsplit(":", 1)[-1] != capture.key:
                raise HTTPException(status_code=422, detail="The memory key cannot be changed")
            result: Final = await self._capture(capture, row.namespace, table)
            await MemoryStore(SimpleNamespace(db=transaction), self.access).authorize(write=True, require_active=False)
            return result

    async def delete(self, memory_id: str) -> bool:
        access: Final = await self.authorize(write=True, require_active=False)
        async with prisma_transaction(self.prisma_client) as transaction:
            table: Final = MemoryRepository(SimpleNamespace(db=transaction)).table
            deleted: Final = await table.delete_many(
                where=_where(
                    access.visible_rows(write=True),
                    {"memory_id": memory_id},  # mutable-ok: Prisma requires native JSON.
                )  # mutable-ok: Prisma requires native JSON.
            )
            await MemoryStore(SimpleNamespace(db=transaction), self.access).authorize(write=True, require_active=False)
            return bool(deleted)
