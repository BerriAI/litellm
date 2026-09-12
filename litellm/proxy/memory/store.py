import json
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from prisma.errors import UniqueViolationError
from pydantic import TypeAdapter

from litellm.proxy.memory.policy import MemoryAccess, memory_digest, resolve_memory_access
from litellm.repositories.table_repositories import MemoryRepository
from litellm.types.memory_v2 import MemoryCapture, MemoryEntry, MemorySearch

if TYPE_CHECKING:
    from prisma.models import LiteLLM_MemoryTable

_METADATA: Final = TypeAdapter(dict[str, object])


def memory_entry(row: "LiteLLM_MemoryTable") -> MemoryEntry:
    metadata: Final = (
        _METADATA.validate_python(row.metadata)
        if isinstance(row.metadata, dict)
        else {  # mutable-ok: Prisma serializes these as native JSON containers.
        }
    )
    title: Final = metadata.get("title")
    evidence: Final = metadata.get("evidence")
    return MemoryEntry(
        memory_id=row.memory_id,
        key=row.key.rsplit(":", 1)[-1],
        title=title if isinstance(title, str) else row.key,
        content=row.value,
        evidence=evidence if isinstance(evidence, str) else "",
        updated_at=row.updated_at,
    )


class MemoryStore:
    def __init__(self, prisma_client: object, access: MemoryAccess) -> None:
        self.prisma_client = prisma_client
        self.access = access
        self.table = MemoryRepository(prisma_client).table

    async def _namespace(self, *, write: bool = False, require_active: bool = True) -> str:
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

    async def search(self, search: MemorySearch, *, require_active: bool = True) -> list[MemoryEntry]:
        namespace: Final = await self._namespace(require_active=require_active)
        words: Final = tuple(dict.fromkeys(search.query.split()))[:12]
        filters: Final = [  # mutable-ok: Prisma serializes these as native JSON containers.
            {  # mutable-ok: Prisma serializes these as native JSON containers.
                "OR": [  # mutable-ok: Prisma serializes these as native JSON containers.
                    {  # mutable-ok: Prisma serializes these as native JSON containers.
                        "value": {  # mutable-ok: Prisma serializes these as native JSON containers.
                            "contains": word,
                            "mode": "insensitive",
                        }
                    },
                    {  # mutable-ok: Prisma serializes these as native JSON containers.
                        "key": {  # mutable-ok: Prisma serializes these as native JSON containers.
                            "contains": word,
                            "mode": "insensitive",
                        }
                    },
                ]
            }
            for word in words
        ]
        rows: Final = await self.table.find_many(
            where={  # mutable-ok: Prisma serializes these as native JSON containers.
                "namespace": namespace,
                **(
                    {  # mutable-ok: Prisma serializes these as native JSON containers.
                        "AND": filters
                    }
                    if filters
                    else {  # mutable-ok: Prisma serializes these as native JSON containers.
                    }
                ),
            },
            order=[  # mutable-ok: Prisma serializes these as native JSON containers.
                {  # mutable-ok: Prisma serializes these as native JSON containers.
                    "updated_at": "desc"
                },
                {  # mutable-ok: Prisma serializes these as native JSON containers.
                    "memory_id": "asc"
                },
            ],
            take=search.limit,
            skip=search.offset,
        )
        return [  # mutable-ok: Prisma serializes these as native JSON containers.
            memory_entry(row) for row in rows
        ]

    async def read(self, memory_id: str) -> MemoryEntry:
        namespace: Final = await self._namespace()
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
        namespace: Final = await self._namespace(write=True)
        key: Final = f"memory-v2:{namespace}:{capture.key}"
        metadata: Final = {  # mutable-ok: Prisma serializes these as native JSON containers.
            "title": capture.title,
            "evidence": capture.evidence,
        }
        data: Final = {  # mutable-ok: Prisma serializes these as native JSON containers.
            "value": capture.content,
            "metadata": json.dumps(metadata),
            "updated_by": self.access.identity.user_id or self.access.identity.key_id,
        }
        existing: Final = await self.table.find_unique(
            where={  # mutable-ok: Prisma serializes these as native JSON containers.
                "key": key
            }
        )
        if existing is not None:
            if existing.namespace != namespace:
                raise HTTPException(status_code=409, detail="Memory key conflict")
            if existing.value == capture.content and existing.metadata == metadata:
                return memory_entry(existing)
            if capture.expected_revision != existing.updated_at:
                raise HTTPException(status_code=409, detail="Read the current memory before replacing it")
            count: Final = await self.table.update_many(
                where={  # mutable-ok: Prisma serializes these as native JSON containers.
                    "key": key,
                    "namespace": namespace,
                    "updated_at": capture.expected_revision,
                    "value": existing.value,
                    "metadata": {  # mutable-ok: Prisma serializes these as native JSON containers.
                        "equals": json.dumps(existing.metadata)
                    },
                },
                data=data,
            )
            if count != 1:
                raise HTTPException(status_code=409, detail="Memory changed; read it again before replacing it")
            return await self.read(existing.memory_id)
        if capture.expected_revision is not None:
            raise HTTPException(status_code=409, detail="Memory no longer exists")
        try:
            created: Final = await self.table.create(
                data={  # mutable-ok: Prisma serializes these as native JSON containers.
                    **data,
                    "memory_id": memory_digest(namespace, capture.key),
                    "key": key,
                    "namespace": namespace,
                    "user_id": self.access.identity.user_id,
                    "team_id": self.access.identity.team_id,
                    "created_by": self.access.identity.user_id or self.access.identity.key_id,
                }
            )
        except UniqueViolationError as exc:
            raise HTTPException(status_code=409, detail="Memory changed; read it again before replacing it") from exc
        return memory_entry(created)

    async def delete(self, memory_id: str) -> bool:
        namespace: Final = await self._namespace(write=True, require_active=False)
        return bool(
            await self.table.delete_many(
                where={  # mutable-ok: Prisma serializes these as native JSON containers.
                    "memory_id": memory_id,
                    "namespace": namespace,
                }
            )
        )
