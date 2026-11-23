from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import SecretStr
from sqlalchemy import inspect, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from litellm.proxy.control_plane.store.migrate import ADVISORY_LOCK, VERSION_TABLE, engine_for, head_revision
from litellm.proxy.control_plane.store.models import Entry, Kind
from litellm.proxy.control_plane.store.schema import dataplanes, roles


class Transaction(Protocol):
    async def load(self) -> tuple[Entry, ...]: ...
    async def write(self, entry: Entry) -> None: ...


class ConfigStore(Protocol):
    async def current(self) -> bool: ...
    def transaction(self) -> AbstractAsyncContextManager[Transaction]: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PostgresTransaction:
    connection: AsyncConnection

    async def _load_kind(self, kind: Kind) -> tuple[Entry, ...]:
        table: Final = dataplanes if kind == "dataplane" else roles
        result: Final = await self.connection.execute(select(table))
        return tuple(
            Entry.model_validate(
                {
                    "revision": row["revision"],
                    "change": {
                        "op": "upsert",
                        "kind": kind,
                        "name": row["id" if kind == "dataplane" else "name"],
                        "entity": row["spec"]
                        if kind == "dataplane"
                        else {"scopes": row["scopes"], "inherits": row["inherits"]},
                        "actor": row["updated_by"],
                    },
                }
            )
            for row in result.mappings()
        )

    async def load(self) -> tuple[Entry, ...]:
        dp: Final = await self._load_kind("dataplane")
        role: Final = await self._load_kind("role")
        return (*dp, *role)

    async def write(self, entry: Entry) -> None:
        change: Final = entry.change
        table: Final = dataplanes if change.kind == "dataplane" else roles
        name_column: Final = table.c.id if change.kind == "dataplane" else table.c.name
        if change.op == "delete":
            await self.connection.execute(table.delete().where(name_column == change.name))
            return
        values: Final = {
            name_column.name: change.name,
            **(
                {"spec": dict(change.entity)}
                if change.kind == "dataplane"
                else {
                    "scopes": change.entity.get("scopes", []),
                    "inherits": change.entity.get("inherits", []),
                }
            ),
            "revision": entry.revision,
            "updated_by": change.actor,
        }
        statement: Final = insert(table).values(**values)
        await self.connection.execute(
            statement.on_conflict_do_update(
                index_elements=[name_column],
                set_={
                    **{name: value for name, value in values.items() if name != name_column.name},
                    "updated_at": text("CURRENT_TIMESTAMP"),
                },
            )
        )


@dataclass(frozen=True, slots=True)
class PostgresStore:
    engine: AsyncEngine

    @classmethod
    def build(cls, url: SecretStr) -> "PostgresStore":
        return cls(engine_for(url))

    async def current(self) -> bool:
        async with self.engine.connect() as connection:
            exists: Final = await connection.run_sync(lambda sync: inspect(sync).has_table(VERSION_TABLE))
            if not exists:
                return False
            result: Final = await connection.execute(text("SELECT version_num FROM control_plane_alembic_version"))
            return tuple(result.scalars()) == (head_revision(),)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Transaction]:
        async with self.engine.begin() as connection:
            await connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADVISORY_LOCK})
            yield PostgresTransaction(connection)

    async def close(self) -> None:
        await self.engine.dispose()
