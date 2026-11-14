import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Final

from nacl.exceptions import CryptoError
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import (
    ErrorBody,
    bad_request,
    error_body,
    forbidden,
    not_found,
    service_unavailable,
)
from litellm.proxy.control_plane.store.migrate import MIGRATE_COMMAND
from litellm.proxy.control_plane.store.models import Change, Entry, maximum_revision, revision_key
from litellm.proxy.control_plane.store.postgres import ConfigStore
from litellm.proxy.control_plane.store.state import InvalidSnapshot, LiveState, Snapshot
from litellm.proxy.control_plane.store.stream import ConfigStream

LOGGER: Final = logging.getLogger("litellm.control_plane.config")
NOT_CONFIGURED: Final = (
    "Runtime config store is not configured or not migrated. Set CONTROL_PLANE_DATABASE_URL and CONTROL_PLANE_REDIS_URL, then run "
    + MIGRATE_COMMAND
    + "."
)


async def retry_pause() -> None:
    await asyncio.sleep(0.2)


@dataclass(frozen=True, slots=True)
class Written:
    entry: Entry
    snapshot: Snapshot


class RuntimeConfig:
    def __init__(
        self,
        state: LiveState,
        store: ConfigStore | None,
        stream: ConfigStream | None,
        pause: Callable[[], Awaitable[None]] = retry_pause,
    ) -> None:
        self.state = state
        self.store = store
        self.stream = stream
        self.pause = pause
        self.ready = False
        self.cursor = "0-0"
        self.task: asyncio.Task[None] | None = None

    async def reload(self) -> bool:
        if self.store is None:
            return False
        async with self.store.transaction() as transaction:
            entries: Final = await transaction.load()
        if not self.state.reload(entries):
            LOGGER.error("Invalid runtime config snapshot. Reads retain the last valid configuration.")
            return False
        self.cursor = max((self.cursor, maximum_revision(entries)), key=revision_key)
        return True

    async def start(self) -> None:
        if self.store is None or self.stream is None:
            return
        try:
            if not await self.store.current():
                LOGGER.error(NOT_CONFIGURED)
                return
            if not await self.reload():
                return
        except (SQLAlchemyError, OSError, ValueError, CryptoError):
            LOGGER.error("Runtime config unavailable. Run %s before serving runtime writes.", MIGRATE_COMMAND)
            return
        self.ready = True
        self.task = asyncio.create_task(self.consume(), name="control-plane-config")

    async def close(self) -> None:
        self.ready = False
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        try:
            if self.stream is not None:
                await self.stream.close()
        finally:
            if self.store is not None:
                await self.store.close()

    async def process(self, entry: Entry) -> bool:
        if entry.change.op == "resync":
            return await self.reload()
        if self.store is None:
            return False
        async with self.store.transaction() as transaction:
            entries: Final = await transaction.load()
        committed: Final = next(
            (row for row in entries if (row.change.kind, row.change.name) == (entry.change.kind, entry.change.name)),
            None,
        )
        if (
            entry.change.op == "upsert"
            and committed is not None
            and revision_key(committed.revision) >= revision_key(entry.revision)
        ):
            return self.state.apply(committed) or await self.reload()
        elif entry.change.op == "delete" and committed is None:
            return self.state.apply(entry) or await self.reload()
        return True

    async def consume(self) -> None:
        stream: Final = self.stream
        if stream is None:
            return
        while True:
            if not self.ready:
                await self.pause()
            try:
                if not self.ready:
                    self.ready = await self.reload()
                    continue
                batch = await stream.read(self.cursor)
                for entry in batch:
                    if not await self.process(entry):
                        self.ready = False
                        break
                    self.cursor = max((self.cursor, entry.revision), key=revision_key)
                if batch and self.ready:
                    # A trimmed stream can omit older changes, including role revocations.
                    self.ready = await self.reload()
            except (RedisError, SQLAlchemyError, OSError, ValueError, CryptoError):
                self.ready = False
                LOGGER.warning("Runtime config consumer reconnecting.")

    def validate(self, change: Change, principal: Principal) -> ErrorBody | Snapshot:
        if self.state.protected(change.kind, change.name):
            return error_body("conflict", "Conflict", 409, "Config-defined entries and default roles are read-only.")
        key: Final = (change.kind, change.name)
        if change.op == "delete" and key not in self.state.view.entries:
            return not_found("Runtime entry not found.")
        if (
            change.kind == "role"
            and change.op == "delete"
            and any(change.name in spec.inherits for name, spec in self.state.view.roles.items() if name != change.name)
        ):
            return error_body("conflict", "Conflict", 409, "Another role inherits this role.")
        desired: Final = tuple(row for row_key, row in self.state.view.entries.items() if row_key != key)
        candidate: Final = Entry(revision="0-0", change=change)
        view: Final = self.state.build((*desired, candidate) if change.op == "upsert" else desired, {})
        if isinstance(view, InvalidSnapshot):
            return bad_request(view.detail)
        granted: Final = (
            view.policy.scopes_for(change.name)
            if change.op == "upsert"
            else self.state.view.policy.scopes_for(change.name)
        )
        if change.kind == "role" and change.op == "upsert" and not granted.issubset(principal.scopes):
            return forbidden("A role cannot grant scopes beyond the caller's own scopes.")
        return view

    async def write(self, desired: Change, principal: Principal) -> Written | ErrorBody:
        change: Final = desired.model_copy(update={"actor": principal.subject})
        store: Final = self.store
        stream: Final = self.stream
        if not self.ready or store is None or stream is None:
            return service_unavailable(NOT_CONFIGURED)
        published: str | None = None  # rebind-ok: compensate only after XADD succeeded
        try:
            async with store.transaction() as transaction:
                current: Final = await transaction.load()
                if not self.state.reload(current):
                    return service_unavailable("Runtime config snapshot is invalid.")
                writer: Final = (
                    principal
                    if principal.token_kind == "master_key"
                    else replace(principal, scopes=principal.scopes & self.state.view.policy.grant(principal.roles))
                )
                required: Final = Scope.DATAPLANES_WRITE if change.kind == "dataplane" else Scope.ROLES_WRITE
                if not writer.has_scope(required):
                    return forbidden("Missing runtime write scope in the committed policy.")
                invalid: Final = self.validate(change, writer)
                if isinstance(invalid, ErrorBody):
                    return invalid
                published = await stream.append(change)  # rebind-ok: compensate only after XADD succeeded
                entry: Final = Entry(revision=published, change=change)
                await transaction.write(entry)
            return Written(entry, invalid)
        except (SQLAlchemyError, RedisError, OSError, asyncio.CancelledError) as error:
            if published is not None:
                with suppress(RedisError, OSError):
                    await asyncio.shield(stream.append(Change(op="resync", actor=principal.subject)))
            if isinstance(error, asyncio.CancelledError):
                raise
            return service_unavailable(
                "Runtime config persistence failed. Reload from the authoritative database is required."
            )
