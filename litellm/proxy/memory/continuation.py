import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import accumulate, islice
from types import MappingProxyType, SimpleNamespace
from typing import Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.litellm_core_utils.prompt_templates.server_tool_responses import object_items
from litellm.litellm_core_utils.prompt_templates.server_tools import ServerToolRoute
from litellm.proxy.memory.policy import memory_digest, memory_primary_client
from litellm.proxy.memory.store import MemoryStore
from litellm.repositories.table_repositories import MemoryContinuationRepository
from litellm.repositories.unit_of_work import prisma_transaction

_ITEMS: Final = TypeAdapter(tuple[object, ...])
_OBJECT: Final = TypeAdapter(dict[str, object])
_MAX_PATCH_BYTES: Final = 1024 * 1024
_MAX_PATCHES: Final = 1000


async def cleanup_memory_continuations(prisma_client: object) -> None:
    async with prisma_transaction(memory_primary_client(prisma_client)) as transaction:
        await transaction.execute_raw(
            'DELETE FROM "LiteLLM_MemoryContinuation" WHERE id IN '
            '(SELECT id FROM "LiteLLM_MemoryContinuation" WHERE expires_at <= $1::timestamp '
            "ORDER BY expires_at LIMIT 1000 FOR UPDATE SKIP LOCKED)",
            datetime.now(timezone.utc),
        )


class MemoryContinuation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    replaces: int = Field(ge=0)
    replacement: tuple[Mapping[str, object], ...] = ()
    response: Mapping[str, object] | None = None
    upstream_ids: tuple[str, ...] = ()
    pending_results: tuple[Mapping[str, object], ...] = ()
    transcript_anchor: str | None = None


def _empty_array(value: object) -> bool:
    return isinstance(value, list) and not value


def _canonical(value: object, depth: int = 0) -> object:
    if depth > 64:
        raise HTTPException(status_code=400, detail="Memory conversation nesting exceeds 64 levels")
    if isinstance(value, dict):
        return {  # mutable-ok: Native provider JSON containers.
            key: _canonical(item, depth + 1)
            for key, item in _OBJECT.validate_python(value).items()
            if key not in ("cache_control",) and item is not None and not _empty_array(item)
        }
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item, depth + 1) for item in _ITEMS.validate_python(value))
    return value


def transcript_items(data: Mapping[str, object], route: ServerToolRoute) -> tuple[Mapping[str, object], ...]:
    content: Final = data.get("input" if route == "aresponses" else "messages")
    return (
        (
            {  # mutable-ok: Native provider JSON containers.
                "role": "user",
                "content": content,
            },
        )
        if isinstance(content, str)
        else object_items(content)
    )


def prefix_hashes(items: tuple[Mapping[str, object], ...], route: ServerToolRoute) -> tuple[str, ...]:
    def canonical_item(item: Mapping[str, object]) -> str:
        content: Final = item.get("content")
        normalized: Final = (
            {  # mutable-ok: Native provider JSON containers.
                **item,
                "content": [  # mutable-ok: Native provider JSON containers.
                    {  # mutable-ok: Native provider JSON containers.
                        "type": "text",
                        "text": content,
                    }
                ],
            }
            if route == "anthropic_messages" and isinstance(content, str)
            else item
        )
        return json.dumps(_canonical(normalized), sort_keys=True, separators=(",", ":"))

    return tuple(islice(accumulate((canonical_item(item) for item in items), memory_digest, initial=route), 1, None))


def _append_items(
    previous: tuple[Mapping[str, object], ...], added: tuple[Mapping[str, object], ...], route: ServerToolRoute
) -> tuple[Mapping[str, object], ...]:
    if not previous or not added or route != "anthropic_messages":
        return (*previous, *added)
    last: Final = previous[-1]
    first: Final = added[0]
    blocks: Final = object_items(last.get("content"))
    if (
        last.get("role") != "user"
        or first.get("role") != "user"
        or not blocks
        or any(block.get("type") != "tool_result" for block in blocks)
    ):
        return (*previous, *added)
    content: Final = first.get("content")
    following: Final = (
        (
            {  # mutable-ok: Prisma query and write JSON.
                "type": "text",
                "text": content,
            },
        )
        if isinstance(content, str)
        else object_items(content)
    )
    return (
        *previous[:-1],
        {  # mutable-ok: Prisma query and write JSON.
            **first,
            "content": [  # mutable-ok: Prisma query and write JSON.
                *blocks,
                *following,
            ],
        },
        *added[1:],
    )


class MemoryContinuations:
    def __init__(self, store: MemoryStore, route: ServerToolRoute) -> None:
        self.store = store
        self.route: Final[ServerToolRoute] = route
        self.table = MemoryContinuationRepository(store.prisma_client).table

    def identifier(self, anchor: str) -> str:
        return memory_digest(
            self.store.access.namespace,
            self.store.access.identity.key_id or self.store.access.identity.user_id,
            self.route,
            anchor,
        )

    async def restore(self, items: tuple[Mapping[str, object], ...]) -> tuple[Mapping[str, object], ...]:
        namespace: Final = await self.store.authorize_namespace()
        anchors: Final = prefix_hashes(items, self.route)
        rows: Final = await self.table.find_many(
            where={  # mutable-ok: Prisma query and write JSON.
                "namespace": namespace,
                "key_id": self.store.access.identity.key_id or self.store.access.identity.user_id or "",
                "id": {  # mutable-ok: Prisma query and write JSON.
                    "in": [  # mutable-ok: Prisma query and write JSON.
                        self.identifier(anchor) for anchor in anchors
                    ]
                },
                "expires_at": {  # mutable-ok: Prisma query and write JSON.
                    "gt": datetime.now(timezone.utc)
                },
            }
        )
        patches: Final = MappingProxyType({row.id: MemoryContinuation.model_validate(row.payload) for row in rows})

        def apply(result: tuple[Mapping[str, object], ...], index: int) -> tuple[Mapping[str, object], ...]:
            patch: Final = patches.get(self.identifier(anchors[index]))
            if patch is None:
                return _append_items(result, (items[index],), self.route)
            if patch.replaces > index + 1 or patch.replaces < 1:
                raise HTTPException(status_code=409, detail="Invalid memory continuation")
            prefix: Final = result[: -(patch.replaces - 1)] if patch.replaces > 1 else result
            return _append_items(prefix, patch.replacement, self.route)

        return reduce(apply, range(len(items)), ())

    async def load_response(self, response_id: str) -> MemoryContinuation | None:
        namespace: Final = await self.store.authorize_namespace()
        row: Final = await self.table.find_first(
            where={  # mutable-ok: Prisma query and write JSON.
                "id": self.identifier(response_id),
                "namespace": namespace,
                "key_id": self.store.access.identity.key_id or self.store.access.identity.user_id or "",
                "expires_at": {  # mutable-ok: Prisma query and write JSON.
                    "gt": datetime.now(timezone.utc)
                },
            }
        )
        return MemoryContinuation.model_validate(row.payload) if row is not None else None

    async def save(self, anchor: str, patch: MemoryContinuation) -> None:
        await self.save_many(((anchor, patch),))

    async def save_many(self, patches: tuple[tuple[str, MemoryContinuation], ...]) -> None:
        namespace: Final = await self.store.authorize_namespace()
        payloads: Final = tuple((self.identifier(anchor), patch.model_dump_json()) for anchor, patch in patches)
        if any(len(payload.encode()) > _MAX_PATCH_BYTES for _, payload in payloads):
            raise HTTPException(status_code=413, detail="Memory continuation exceeds one megabyte")
        key_id: Final = self.store.access.identity.key_id or self.store.access.identity.user_id or ""
        now: Final = datetime.now(timezone.utc)
        async with prisma_transaction(self.store.prisma_client) as transaction:
            lock_key: Final = int(memory_digest("memory-continuation-quota", namespace, key_id)[:16], 16) - (1 << 63)
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
            table: Final = MemoryContinuationRepository(SimpleNamespace(db=transaction)).table
            await table.delete_many(
                where={  # mutable-ok: Prisma query and write JSON.
                    "namespace": namespace,
                    "key_id": key_id,
                    "expires_at": {  # mutable-ok: Prisma query and write JSON.
                        "lte": now
                    },
                }
            )
            count: Final = await table.count(
                where={  # mutable-ok: Prisma query and write JSON.
                    "namespace": namespace,
                    "key_id": key_id,
                    "id": {  # mutable-ok: Prisma query and write JSON.
                        "not_in": [  # mutable-ok: Prisma query and write JSON.
                            identifier for identifier, _ in payloads
                        ]
                    },
                }
            )
            if count + len(payloads) > _MAX_PATCHES:
                raise HTTPException(status_code=429, detail="Too many active memory continuations for this key")
            for identifier, payload in payloads:
                await table.upsert(
                    where={  # mutable-ok: Prisma query and write JSON.
                        "id": identifier
                    },
                    data={  # mutable-ok: Prisma query and write JSON.
                        "create": {  # mutable-ok: Prisma query and write JSON.
                            "id": identifier,
                            "namespace": namespace,
                            "key_id": key_id,
                            "payload": payload,
                            "expires_at": now + timedelta(hours=24),
                        },
                        "update": {  # mutable-ok: Prisma query and write JSON.
                            "payload": payload,
                            "expires_at": now + timedelta(hours=24),
                        },
                    },
                )

    async def delete_response(self, response_id: str, patch: MemoryContinuation) -> None:
        namespace: Final = await self.store.authorize_namespace()
        anchors: Final = (response_id, patch.transcript_anchor) if patch.transcript_anchor else (response_id,)
        await self.table.delete_many(
            where={  # mutable-ok: Prisma query and write JSON.
                "namespace": namespace,
                "key_id": self.store.access.identity.key_id or self.store.access.identity.user_id or "",
                "id": {  # mutable-ok: Prisma query and write JSON.
                    "in": [  # mutable-ok: Prisma query and write JSON.
                        self.identifier(anchor) for anchor in anchors
                    ]
                },
            }
        )
