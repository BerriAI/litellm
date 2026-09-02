# pyright: reportDeprecated=false
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.python_extension.generated.v1 import extension_host_pb2 as pb

from .loader import ExtensionLoadError, config_from_proto, load_extension
from .models import LoadedExtension, operation_error, operation_ok


@dataclass(frozen=True, slots=True)
class Revision:
    extensions: MappingProxyType[str, LoadedExtension]


class RevisionStore:
    def __init__(self) -> None:
        self._prepared: dict[str, Revision] = {}  # mutable-ok: LiteLLM compatibility payload
        self._committed: set[str] = set()  # mutable-ok: LiteLLM compatibility payload
        self._active_revision: str | None = None
        self._in_flight: dict[str, int] = {}  # mutable-ok: LiteLLM compatibility payload
        self._condition: Final = asyncio.Condition()

    @property
    def active_revision(self) -> str | None:
        return self._active_revision

    async def prepare(
        self, revision_id: str, specs: tuple[pb.ExtensionSpec, ...]
    ) -> tuple[pb.OperationResult, tuple[LoadedExtension, ...]]:
        async with self._condition:
            validation_error: Final = self._validate_prepare(revision_id, specs)
            if validation_error is not None:
                return validation_error, ()
            prepared: Final = self._prepared.get(revision_id)
            if prepared is not None:
                existing_extensions: Final = tuple(prepared.extensions.values())
                requested_configs: Final = tuple(config_from_proto(spec) for spec in specs)
                if requested_configs != tuple(extension.config for extension in existing_extensions):
                    return (
                        operation_error(
                            pb.ERROR_CODE_INVALID_ARGUMENT,
                            f"revision {revision_id!r} already exists with a different manifest",
                        ),
                        (),
                    )
                return (
                    operation_error(pb.ERROR_CODE_ALREADY_EXISTS, f"revision {revision_id!r} already exists"),
                    existing_extensions,
                )
            active: Final = self._prepared.get(self._active_revision or "")
            loaded: list[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
                LoadedExtension
            ] = []  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
            try:
                for spec in specs:
                    config = config_from_proto(spec)
                    existing = active.extensions.get(config.id) if active is not None else None
                    loaded.append(
                        existing if existing is not None and existing.config == config else load_extension(config)
                    )
            except ExtensionLoadError as error:
                return operation_error(pb.ERROR_CODE_LOAD_FAILED, str(error)), ()
            self._prepared[revision_id] = Revision(
                MappingProxyType({extension.config.id: extension for extension in loaded})
            )
            return operation_ok(), tuple(loaded)

    async def commit(self, revision_id: str) -> pb.OperationResult:
        async with self._condition:
            if revision_id not in self._prepared:
                return operation_error(pb.ERROR_CODE_NOT_FOUND, f"revision {revision_id!r} was not prepared")
            self._active_revision = revision_id
            self._committed.add(revision_id)
            return operation_ok()

    async def retire(self, revision_id: str) -> pb.OperationResult:
        async with self._condition:
            if revision_id == self._active_revision:
                return operation_error(
                    pb.ERROR_CODE_INVALID_ARGUMENT, f"active revision {revision_id!r} cannot be retired"
                )
            if revision_id not in self._prepared:
                return operation_error(pb.ERROR_CODE_NOT_FOUND, f"revision {revision_id!r} was not prepared")
            await self._condition.wait_for(lambda: self._in_flight.get(revision_id, 0) == 0)
            self._prepared.pop(revision_id, None)
            self._committed.discard(revision_id)
            self._in_flight.pop(revision_id, None)
            return operation_ok()

    @asynccontextmanager
    async def acquire(self, revision_id: str, extension_id: str) -> AsyncIterator[LoadedExtension]:
        async with self._condition:
            if revision_id not in self._committed:
                raise LookupError(f"revision {revision_id!r} is not committed")
            revision: Final = self._prepared.get(revision_id)
            extension: Final = revision.extensions.get(extension_id) if revision is not None else None
            if extension is None:
                raise LookupError(f"extension {extension_id!r} was not found in revision {revision_id!r}")
            self._in_flight[revision_id] = self._in_flight.get(revision_id, 0) + 1
        try:
            yield extension
        finally:
            async with self._condition:
                self._in_flight[revision_id] -= 1
                self._condition.notify_all()

    def _validate_prepare(self, revision_id: str, specs: tuple[pb.ExtensionSpec, ...]) -> pb.OperationResult | None:
        if not revision_id:
            return operation_error(pb.ERROR_CODE_INVALID_ARGUMENT, "revision_id is required")
        if any(not spec.id or not spec.entrypoint for spec in specs):
            return operation_error(pb.ERROR_CODE_INVALID_ARGUMENT, "extension id and entrypoint are required")
        ids: Final = tuple(spec.id for spec in specs)
        if len(ids) != len(set(ids)):  # mutable-ok: LiteLLM compatibility payload
            return operation_error(pb.ERROR_CODE_INVALID_ARGUMENT, "extension ids must be unique")
        return None
