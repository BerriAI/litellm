from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.engine.models import Engine, Worker


class Database(Protocol):
    def query_raw(self, query: str, *args: object) -> Awaitable[object]: ...
    def execute_raw(self, query: str, *args: object) -> Awaitable[int]: ...


class Row(BaseModel):
    data: JsonValue


_ROWS: Final = TypeAdapter(tuple[Row, ...])


class EngineRepository:
    def __init__(self, db: Database) -> None:
        self.db: Final = db

    async def engines(self) -> tuple[Engine, ...]:
        rows: Final = _ROWS.validate_python(await self.db.query_raw('SELECT data FROM "LiteLLM_Engine" ORDER BY id'))
        return tuple(Engine.model_validate(row.data) for row in rows)

    async def get(self, engine_id: str) -> Engine | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                'SELECT data FROM "LiteLLM_Engine" WHERE id=$1',
                engine_id,
            )
        )
        return Engine.model_validate(rows[0].data) if rows else None

    async def create(self, engine: Engine) -> Engine:
        await self.db.execute_raw(
            'INSERT INTO "LiteLLM_Engine" (id, version, data) VALUES ($1,0,$2::jsonb)',
            engine.id,
            engine.model_dump_json(),
        )
        return engine

    async def update(self, engine_id: str, transform: Callable[[Engine], Engine], attempts: int = 8) -> Engine | None:
        previous: Final = await self.get(engine_id)
        if previous is None or attempts == 0:
            return None
        candidate: Final = transform(previous)
        if candidate == previous:
            return previous
        updated: Final = candidate.model_copy(update=MappingProxyType({"version": previous.version + 1}))
        count: Final = await self.db.execute_raw(
            'UPDATE "LiteLLM_Engine" SET data=$1::jsonb, version=version+1 WHERE id=$2 AND version=$3',
            updated.model_dump_json(),
            engine_id,
            previous.version,
        )
        return updated if count else await self.update(engine_id, transform, attempts - 1)

    async def workers(self) -> tuple[Worker, ...]:
        rows: Final = _ROWS.validate_python(await self.db.query_raw('SELECT data FROM "LiteLLM_EngineWorker"'))
        return tuple(Worker.model_validate(row.data) for row in rows)

    async def worker(self, token_hash: str) -> Worker | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                'SELECT data FROM "LiteLLM_EngineWorker" WHERE token_hash=$1',
                token_hash,
            )
        )
        return Worker.model_validate(rows[0].data) if rows else None

    async def save_worker(self, worker: Worker, token_hash: str | None = None) -> None:
        if token_hash is not None:
            await self.db.execute_raw(
                'INSERT INTO "LiteLLM_EngineWorker" (id,token_hash,data) VALUES ($1,$2,$3::jsonb)',
                worker.id,
                token_hash,
                worker.model_dump_json(),
            )
            return
        await self.db.execute_raw(
            'UPDATE "LiteLLM_EngineWorker" SET data=$1::jsonb WHERE id=$2', worker.model_dump_json(), worker.id
        )

    async def heartbeat(self, worker_id: str, now: str) -> None:
        await self.db.execute_raw(
            """UPDATE "LiteLLM_EngineWorker" SET data=jsonb_set(data, '{last_seen}', to_jsonb($1::text)) WHERE id=$2""",
            now,
            worker_id,
        )


class WriterDatabase:
    def __init__(self, writer: PrismaWrapper) -> None:
        self.writer: Final = writer

    async def query_raw(self, query: str, *args: object) -> object:
        return _ROWS.validate_python(await self.writer.query_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate rows here.

    async def execute_raw(self, query: str, *args: object) -> int:
        return TypeAdapter(int).validate_python(await self.writer.execute_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate the count here.
