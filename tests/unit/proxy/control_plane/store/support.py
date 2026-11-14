import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Final

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from litellm.proxy.control_plane.container import Container, build_container
from litellm.proxy.control_plane.dataplanes.client import HttpTransport
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.proxy.control_plane.store.models import Change, Entry, revision_key
from litellm.proxy.control_plane.store.postgres import Transaction
from tests.unit.proxy.control_plane.support import MASTER_KEY, no_exchange


class MemoryStore:
    def __init__(self, rows: tuple[Entry, ...] = (), migrated: bool = True) -> None:
        self.rows = rows
        self.migrated = migrated
        self.lock = asyncio.Lock()
        self.fail_write = False
        self.fail_load = False
        self.closed = False
        self.loads = 0
        self.writes: tuple[Entry, ...] = ()

    async def current(self) -> bool:
        return self.migrated

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        async with self.lock:
            original: Final = self.rows
            try:
                yield self
            except BaseException:
                self.rows = original
                raise

    async def load(self) -> tuple[Entry, ...]:
        self.loads += 1
        if self.fail_load:
            raise SQLAlchemyError("snapshot unavailable")
        return self.rows

    async def write(self, entry: Entry) -> None:
        if self.fail_write:
            raise SQLAlchemyError("database write failed")
        self.writes = (*self.writes, entry)
        remaining: Final = tuple(
            row for row in self.rows if (row.change.kind, row.change.name) != (entry.change.kind, entry.change.name)
        )
        self.rows = (*remaining, entry) if entry.change.op == "upsert" else remaining

    async def close(self) -> None:
        self.closed = True


class MemoryStream:
    def __init__(self) -> None:
        self.entries: tuple[Entry, ...] = ()
        self.changed = asyncio.Event()
        self.reads: asyncio.Queue[str] = asyncio.Queue()
        self.failure: RedisError | None = None
        self.closed = False

    async def append(self, change: Change) -> str:
        revision: Final = f"{100 + len(self.entries)}-0"
        self.entries = (*self.entries, Entry(revision=revision, change=change))
        self.changed.set()
        return revision

    async def read(self, after: str) -> tuple[Entry, ...]:
        self.reads.put_nowait(after)
        if self.failure is not None:
            error: Final = self.failure
            self.failure = None
            raise error
        pending: Final = tuple(entry for entry in self.entries if revision_key(entry.revision) > revision_key(after))
        if pending:
            return pending
        self.changed.clear()
        await self.changed.wait()
        return tuple(entry for entry in self.entries if revision_key(entry.revision) > revision_key(after))

    async def consumed(self, revision: str) -> None:
        while True:
            cursor: Final = await asyncio.wait_for(self.reads.get(), timeout=30)
            if revision_key(cursor) >= revision_key(revision):
                return

    async def close(self) -> None:
        self.closed = True


def container(
    store: MemoryStore | None = None,
    stream: MemoryStream | None = None,
    transport: HttpTransport | None = None,
    **overrides: object,
) -> Container:
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "CONTROL_PLANE_SALT_KEY": "runtime-test-salt",
            "audit_sinks": (),
            **overrides,
        }
    )
    return build_container(
        settings, exchanger=no_exchange, transport=transport, config_store=store, config_stream=stream
    )


def role_entry(revision: str, name: str = "runtime", *scopes: str) -> Entry:
    return Entry(
        revision=revision,
        change=Change(
            op="upsert", kind="role", name=name, entity={"scopes": list(scopes), "inherits": []}, actor="operator"
        ),
    )
