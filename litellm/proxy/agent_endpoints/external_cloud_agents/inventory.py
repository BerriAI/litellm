import hmac
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from typing import Final

from pydantic import Field, TypeAdapter, ValidationError

from litellm.proxy.agent_endpoints.external_cloud_agents.database import Database
from litellm.proxy.agent_endpoints.external_cloud_agents.models import Failure, FrozenModel, SourceRecord
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper


class ToolInfo(FrozenModel):
    toolset_name: str | None = None


class Block(FrozenModel):
    type: str = ""
    name: str | None = None
    tool_name: str | None = None
    tool_info: ToolInfo | None = None


class Message(FrozenModel):
    content: tuple[Block, ...] | str = ()


class Actor(FrozenModel):
    id: str | None = None


class Application(FrozenModel):
    application: str = ""


class Frame(FrozenModel):
    request_id: str | None = Field(default=None, max_length=256)
    session_id: str = ""
    tenant_id: str | None = None
    type: str = ""
    actor: Actor | None = None
    model: str | None = None
    source: Application = Application()
    messages: tuple[Message, ...] = ()


def parse_frame(body: bytes) -> Frame | None:
    try:
        return Frame.model_validate_json(body)
    except ValidationError:
        return None


def message_tools(message: Message) -> tuple[str, ...]:
    if isinstance(message.content, str):
        return ()
    names: Final = (tool_names(block) for block in message.content if block.type == "tool_use")
    return ToolCollector().collect(chain.from_iterable(names))


class ToolCollector:
    def __init__(self) -> None:
        self.names: frozenset[str] = frozenset()

    def collect(self, names: Iterable[str]) -> tuple[str, ...]:
        for name in names:
            self.names = self.names | {name}
            if len(self.names) >= 50:
                break
        return tuple(sorted(self.names))


def tool_names(block: Block) -> tuple[str, ...]:
    names: Final = (block.tool_name or block.name, block.tool_info.toolset_name if block.tool_info else None)
    return tuple(name for name in names if name)


def authenticated(source: SourceRecord, authorization: str) -> bool:
    key: Final = decrypt_value_helper(source.ingest_key, "ingest_key")
    return bool(key) and hmac.compare_digest(authorization.encode(), f"Bearer {key}".encode())


@dataclass(frozen=True, slots=True)
class CacheEntry:
    source: SourceRecord
    expires_at: float


class SourceCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.entries: tuple[CacheEntry, ...] = ()
        self.missing: tuple[tuple[str, float], ...] = ()

    def discard(self, source_id: str) -> None:
        self.entries = tuple(entry for entry in self.entries if entry.source.source_id != source_id)
        self.missing = tuple(entry for entry in self.missing if entry[0] != source_id)

    def put(self, source: SourceRecord) -> None:
        self.discard(source.source_id)
        self.entries = (*self.entries[-255:], CacheEntry(source, self.clock() + 20))

    async def source(self, db: Database, source_id: str) -> SourceRecord | None:
        if any(key == source_id and expiry > self.clock() for key, expiry in self.missing):
            return None
        cached: Final = next(
            (
                entry.source
                for entry in self.entries
                if entry.source.source_id == source_id and entry.expires_at > self.clock()
            ),
            None,
        )
        if cached is not None:
            return cached
        source: Final = await get_source(db, source_id)
        if source is not None:
            self.put(source)
        else:
            self.missing = (*self.missing[-255:], (source_id, self.clock() + 5))
        return source


async def get_source(db: Database, source_id: str) -> SourceRecord | None:
    rows: Final = TypeAdapter(tuple[SourceRecord, ...]).validate_python(
        await db.query_raw('SELECT * FROM "LiteLLM_ExternalCloudAgentSource" WHERE id=$1', source_id)
    )
    return rows[0] if rows else None


class Inventory:
    def __init__(self, db: Database, cache: SourceCache) -> None:
        self.db = db
        self.cache = cache

    async def record(self, source: SourceRecord, frame: Frame | None) -> Failure | None:
        if frame is not None and frame.tenant_id:
            tenant: Final = await self.bind(source, frame.tenant_id)
            if isinstance(tenant, Failure):
                return tenant
        await self.db.execute_raw(
            'UPDATE "LiteLLM_ExternalCloudAgentSource" SET last_event_at=NOW() WHERE id=$1', source.source_id
        )
        if frame is None or frame.source.application == "config-test" or frame.type == "config-test":
            return None
        if not frame.session_id or not frame.tenant_id:
            return None
        tools: Final = ToolCollector().collect(chain.from_iterable(map(message_tools, frame.messages)))
        await self.db.execute_raw(
            'INSERT INTO "LiteLLM_ExternalCloudAgent" AS a '
            "(source_id,external_id,last_actor_id,last_model,recent_request_ids,tools) "
            "VALUES ($1,$2,$3,$4,array_remove(ARRAY[$5::text],NULL),$6::text[]) "
            "ON CONFLICT (source_id,external_id) DO UPDATE SET request_count=a.request_count+1,last_seen_at=NOW(),"
            "last_actor_id=EXCLUDED.last_actor_id,last_model=EXCLUDED.last_model,"
            "recent_request_ids=(a.recent_request_ids || EXCLUDED.recent_request_ids)"
            "[GREATEST(1,cardinality(a.recent_request_ids || EXCLUDED.recent_request_ids)-255):],"
            "tools=ARRAY(SELECT DISTINCT t FROM unnest(a.tools || EXCLUDED.tools) AS t ORDER BY t LIMIT 50) "
            "WHERE $5::text IS NULL OR NOT ($5=ANY(a.recent_request_ids))",
            source.source_id,
            frame.session_id,
            frame.actor.id if frame.actor else None,
            frame.model,
            frame.request_id or None,
            list(tools),
        )
        return None

    async def bind(self, source: SourceRecord, tenant_id: str) -> str | Failure:
        if source.tenant_id is not None:
            return tenant_id if source.tenant_id == tenant_id else Failure(status=401, message="Tenant mismatch")
        await self.db.execute_raw(
            'UPDATE "LiteLLM_ExternalCloudAgentSource" SET tenant_id=$2 WHERE id=$1 AND tenant_id IS NULL',
            source.source_id,
            tenant_id,
        )
        bound: Final = await get_source(self.db, source.source_id)
        if bound is None or bound.tenant_id != tenant_id:
            return Failure(status=401, message="Tenant mismatch")
        self.cache.put(bound)
        return tenant_id
