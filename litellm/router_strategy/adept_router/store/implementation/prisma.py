import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Final

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
        # Prisma's raw-query model path returns JSONB as a JSON string; parse it back to a mapping.
        if isinstance(value, str):
            return _JSON_ADAPTER.validate_json(value)
        return value


async def _create_tables(client: Prisma) -> None:
    """Create ADEPT's tables in the user's database if they are absent. Column names match the
    prior SQLAlchemy schema so existing ADEPT databases stay compatible."""
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


# One long-lived Prisma client per database URL, connected once and reused. Keying on the URL
# (rather than per repo instance) means a router rebuild — which drops the old repo and builds a
# new one for the same database — reuses the existing client instead of connecting a second one
# and orphaning the first, so no connection or engine process leaks across rebuilds.
_CLIENTS: Final[dict[str, Prisma]] = {}  # mutable-ok: connection registry keyed by database URL
_REGISTRY_LOCK: Final = asyncio.Lock()


async def _get_client(db_url: str) -> Prisma:
    """Return the connected client for db_url (one per URL, shared), creating tables once."""
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
        _CLIENTS[db_url] = client
        return client


class AdeptPrismaRepo(AdeptTemplateStore):
    """ADEPT template/conversation store backed by the user's own PostgreSQL.

    Reaches the user's tables through litellm's Prisma client pointed at the user's database via a
    datasource override, using parameterized raw SQL. One client is kept per database URL (see
    `_CLIENTS`), connected once and reused for the app's lifetime, so a router rebuild does not
    leak a connection. Nothing is added to litellm's own Prisma schema and litellm's database is
    never opened; `auto_register` stays off so litellm's global client is never affected. Tables
    are created on first use.
    """

    def __init__(self, db_url: str) -> None:
        if not db_url:
            raise ValueError(
                "A PostgreSQL connection URL is required. Example: postgresql://user:password@host:5432/dbname"
            )
        self._db_url = db_url

    async def match_by_hash(self, template_hash: str, router_id: str) -> str | None:
        try:
            client: Final = await _get_client(self._db_url)
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
            client: Final = await _get_client(self._db_url)
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
        else:
            verbose_router_logger.debug("Stored conversation for template %s", template_id)
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
        """
        Insert a new template row, returning the surviving template_id (ours or a concurrent
        insert's). ON CONFLICT DO NOTHING on the (router_id, template_hash) unique index makes
        two concurrent requests with the same hash safe: the loser no-ops and we re-read the winner.
        """
        try:
            client: Final = await _get_client(self._db_url)
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
        else:
            surviving_id: Final = rows[0].id if rows else template_id
            verbose_router_logger.debug("AdeptRouter: stored template %s", surviving_id)
            return surviving_id

    async def get_template(self, template_id: str) -> StoredTemplate | None:
        try:
            client: Final = await _get_client(self._db_url)
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
            client: Final = await _get_client(self._db_url)
            rows: Final = await client.query_raw(
                "SELECT count(*)::int AS c FROM conversations WHERE template_id = $1",
                template_id,
                model=_CountRow,
            )
        except PrismaError as e:
            verbose_router_logger.error("Error counting conversations: %s", e)
            return None
        else:
            return rows[0].c if rows else 0
