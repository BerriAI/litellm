import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from prisma import Prisma
from prisma.errors import PrismaError
from prisma.types import DatasourceOverride
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

from litellm._logging import verbose_router_logger
from litellm.router_strategy.adept_router.store.store_template import (
    AdeptTemplateStore,
    StoredTemplate,
)

_JSON_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_DRAIN_TIMEOUT_SECONDS: Final = 30.0


class _IdRow(BaseModel):
    id: str


class _CountRow(BaseModel):
    c: int


class _TemplateRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    template: str
    template_hash: str | None = None
    router_id: str
    target_model: str | None = None
    additional_information: Mapping[str, object] | None = None
    created_at: datetime | None = None

    @field_validator("additional_information", mode="before")
    @classmethod
    def _coerce_json(cls, value: object) -> object:
        if isinstance(value, str):
            return _JSON_ADAPTER.validate_json(value)
        return value


def _redact_url(url: str) -> str:
    """Return the URL with any embedded password replaced by '***'."""
    try:
        split: Final = urlsplit(url)
    except ValueError:
        return "<redacted>"
    if split.password is None:
        return url
    user: Final = split.username or ""
    host: Final = split.hostname or ""
    port_suffix: Final = f":{split.port}" if split.port is not None else ""
    netloc: Final = f"{user}:***@{host}{port_suffix}" if user else f":***@{host}{port_suffix}"
    return urlunsplit((split.scheme, netloc, split.path, split.query, split.fragment))


async def _create_tables(client: Prisma) -> None:
    await client.execute_raw(
        "CREATE TABLE IF NOT EXISTS templates ("
        "id TEXT PRIMARY KEY, template TEXT NOT NULL, template_hash VARCHAR(64) NOT NULL, "
        "router_id TEXT NOT NULL, target_model TEXT, additional_information JSONB, "
        "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    await client.execute_raw(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_templates_router_hash ON templates (router_id, template_hash)"
    )
    await client.execute_raw(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "id SERIAL PRIMARY KEY, template_id TEXT NOT NULL REFERENCES templates(id), "
        "prompt TEXT NOT NULL, response TEXT NOT NULL, additional_information JSONB, "
        "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    await client.execute_raw("CREATE INDEX IF NOT EXISTS ix_conversations_template_id ON conversations (template_id)")


def _json_or_none(payload: Mapping[str, object] | None) -> str | None:
    return json.dumps(payload) if payload is not None else None


class _ClientHandle:
    """Wraps a shared Prisma client with an in-flight refcount so `disconnect_client` can drain
    active operations before tearing the socket down."""

    __slots__ = ("_closed", "_drained", "_inflight", "_lock", "client")

    def __init__(self, client: Prisma) -> None:
        self.client: Final = client
        self._lock: Final = asyncio.Lock()
        self._inflight = 0  # mutable-ok: refcount for graceful drain
        self._drained: Final = asyncio.Event()
        self._drained.set()
        self._closed = False  # mutable-ok: one-way close flag rejecting new borrows

    @asynccontextmanager
    async def borrow(self) -> AsyncGenerator[Prisma]:
        async with self._lock:
            if self._closed:
                raise PrismaError("Prisma client has been disconnected")
            self._inflight += 1  # rebind-ok: refcount increment guarded by _lock
            self._drained.clear()
        try:
            yield self.client
        finally:
            async with self._lock:
                self._inflight -= 1  # rebind-ok: refcount decrement guarded by _lock
                if self._inflight == 0:
                    self._drained.set()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True  # rebind-ok: one-way transition guarded by _lock
            already_drained: Final = self._inflight == 0
        if already_drained:
            return
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=_DRAIN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            verbose_router_logger.warning(
                "AdeptPrismaRepo: drain timed out with %s in-flight operations; disconnecting anyway.",
                self._inflight,
            )


_CLIENTS: Final[dict[str, _ClientHandle]] = {}  # mutable-ok: connection registry keyed by database URL
_REGISTRY_LOCK: Final = asyncio.Lock()


async def _get_handle(db_url: str) -> _ClientHandle:
    cached: Final = _CLIENTS.get(db_url)
    if cached is not None:
        return cached
    async with _REGISTRY_LOCK:
        existing: Final = _CLIENTS.get(db_url)
        if existing is not None:
            return existing
        client: Final = Prisma(datasource=DatasourceOverride(url=db_url))
        await client.connect()
        await _create_tables(client)
        handle: Final = _ClientHandle(client)
        _CLIENTS[db_url] = handle
        return handle


async def disconnect_client(db_url: str) -> None:
    async with _REGISTRY_LOCK:
        existing: Final = _CLIENTS.pop(db_url, None)
    if existing is None:
        return
    await existing.close()
    try:
        await existing.client.disconnect()
    except Exception as e:  # noqa: BLE001  # best-effort: a hung PG socket must not block the rebuild path
        verbose_router_logger.warning("AdeptPrismaRepo: disconnect failed for %s: %s", _redact_url(db_url), e)


def schedule_disconnect(db_url: str) -> None:
    try:
        loop: Final = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(disconnect_client(db_url))


class AdeptPrismaRepo(AdeptTemplateStore):
    def __init__(self, db_url: str) -> None:
        if not db_url:
            raise ValueError(
                "A PostgreSQL connection URL is required. Example: postgresql://user:password@host:5432/dbname"
            )
        self._db_url = db_url

    async def match_by_hash(self, template_hash: str, router_id: str) -> str | None:
        try:
            handle: Final = await _get_handle(self._db_url)
            async with handle.borrow() as client:
                rows: Final = await client.query_raw(
                    "SELECT id FROM templates WHERE router_id = $1 AND template_hash = $2 LIMIT 1",
                    router_id,
                    template_hash,
                    model=_IdRow,
                )
        except PrismaError as e:
            verbose_router_logger.error("Error matching template by hash: %s", e)
            return None
        else:
            return rows[0].id if rows else None

    async def load_all_for_router(self, router_id: str, limit: int) -> Sequence[StoredTemplate]:
        try:
            handle: Final = await _get_handle(self._db_url)
            async with handle.borrow() as client:
                rows: Final = await client.query_raw(
                    "SELECT id, template, template_hash, router_id, target_model, additional_information, created_at "
                    "FROM templates WHERE router_id = $1 ORDER BY created_at DESC LIMIT $2",
                    router_id,
                    limit,
                    model=_TemplateRow,
                )
        except PrismaError as e:
            verbose_router_logger.error("Error loading templates for router %s: %s", router_id, e)
            return ()
        return tuple(
            StoredTemplate(
                id=row.id,
                template=row.template,
                template_hash=row.template_hash,
                router_id=row.router_id,
                target_model=row.target_model,
                additional_information=row.additional_information,
                created_at=row.created_at,
            )
            for row in rows
        )

    async def store_conversation(
        self,
        prompt: str,
        response: str,
        template_id: str | None = None,
        additional_information: Mapping[str, object] | None = None,
    ) -> bool:
        if not template_id:
            verbose_router_logger.error("template_id is required to store a conversation.")
            return False
        try:
            handle: Final = await _get_handle(self._db_url)
            async with handle.borrow() as client:
                await client.execute_raw(
                    "INSERT INTO conversations (template_id, prompt, response, additional_information) "
                    "VALUES ($1, $2, $3, $4::jsonb)",
                    template_id,
                    prompt,
                    response,
                    _json_or_none(additional_information),
                )
        except PrismaError as e:
            verbose_router_logger.error("Error storing conversation: %s", e)
            return False
        return True

    async def store_template(
        self,
        template_id: str,
        template: str,
        template_hash: str,
        target_model: str,
        router_id: str,
        additional_information: Mapping[str, object] | None = None,
    ) -> str | None:
        """Insert a new template row, returning the surviving id (ours or a concurrent insert's).

        ON CONFLICT DO NOTHING on the (router_id, template_hash) unique index makes concurrent
        inserts safe: the loser no-ops and we re-read the winner.
        """
        try:
            handle: Final = await _get_handle(self._db_url)
            async with handle.borrow() as client:
                await client.execute_raw(
                    "INSERT INTO templates (id, template, template_hash, target_model, router_id, additional_information) "
                    "VALUES ($1, $2, $3, $4, $5, $6::jsonb) ON CONFLICT (router_id, template_hash) DO NOTHING",
                    template_id,
                    template,
                    template_hash,
                    target_model,
                    router_id,
                    _json_or_none(additional_information),
                )
                rows: Final = await client.query_raw(
                    "SELECT id FROM templates WHERE router_id = $1 AND template_hash = $2 LIMIT 1",
                    router_id,
                    template_hash,
                    model=_IdRow,
                )
        except PrismaError as e:
            verbose_router_logger.error("AdeptRouter: error storing template: %s", e)
            return None
        return rows[0].id if rows else template_id

    async def get_template(self, template_id: str) -> StoredTemplate | None:
        try:
            handle: Final = await _get_handle(self._db_url)
            async with handle.borrow() as client:
                rows: Final = await client.query_raw(
                    "SELECT id, template, template_hash, router_id, target_model, additional_information, created_at "
                    "FROM templates WHERE id = $1 LIMIT 1",
                    template_id,
                    model=_TemplateRow,
                )
        except PrismaError as e:
            verbose_router_logger.error("Error retrieving template: %s", e)
            return None
        if not rows:
            return None
        row: Final = rows[0]
        return StoredTemplate(
            id=row.id,
            template=row.template,
            template_hash=row.template_hash,
            router_id=row.router_id,
            target_model=row.target_model,
            additional_information=row.additional_information,
            created_at=row.created_at,
        )

    async def count_conversation_by_template_id(self, template_id: str) -> int | None:
        try:
            handle: Final = await _get_handle(self._db_url)
            async with handle.borrow() as client:
                rows: Final = await client.query_raw(
                    "SELECT count(*)::int AS c FROM conversations WHERE template_id = $1",
                    template_id,
                    model=_CountRow,
                )
        except PrismaError as e:
            verbose_router_logger.error("Error counting conversations: %s", e)
            return None
        return rows[0].c if rows else 0
