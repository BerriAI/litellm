"""
Typed boundary for ``managed_id_rewriter``'s untyped dependencies.

The managed file/object Prisma tables are reached through
``PrismaTableRepository.table`` (untyped) and the enterprise managed-files
hook arrives as a bare ``CustomLogger``. This module narrows both to
structural protocols once, at the boundary, so the rewriter itself works
with fully typed values.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import JsonValue

from litellm.repositories.table_repositories import (
    ManagedFileRepository,
    ManagedObjectRepository,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.models.managed_files import LiteLLM_ManagedFileTable
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient
    from litellm.types.llms.openai import OpenAIFileObject


@runtime_checkable
class ManagedFileRowLike(Protocol):
    unified_file_id: str
    created_by: str | None
    team_id: str | None
    created_at: datetime | None
    file_object: object


@runtime_checkable
class ManagedObjectRowLike(Protocol):
    unified_object_id: str
    created_by: str | None
    team_id: str | None
    created_at: datetime | None
    file_object: object


@runtime_checkable
class ManagedTableActions(Protocol):
    def find_first(self, where: Mapping[str, object]) -> Awaitable[object]: ...

    def find_many(
        self,
        where: Mapping[str, object],
        order: Sequence[Mapping[str, str]] | Mapping[str, str] | None = None,
        take: int | None = None,
    ) -> Awaitable[Sequence[object]]: ...

    def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Awaitable[object]: ...

    def upsert(self, where: Mapping[str, object], data: Mapping[str, Mapping[str, object]]) -> Awaitable[object]: ...


@runtime_checkable
class ManagedFileReaderLike(Protocol):
    def get_unified_file_id(
        self,
        file_id: str,
        litellm_parent_otel_span: object = None,
    ) -> Awaitable[LiteLLM_ManagedFileTable | None]: ...


@runtime_checkable
class ManagedFileWriterLike(Protocol):
    def store_unified_file_id(
        self,
        file_id: str,
        file_object: OpenAIFileObject | None,
        litellm_parent_otel_span: object,
        model_mappings: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Awaitable[None]: ...


class PassthroughListResponse(TypedDict):
    object: Literal["list"]
    data: list[dict[str, JsonValue]]
    first_id: str | None
    last_id: str | None
    has_more: bool


class _TableHolder(Protocol):
    @property
    def table(self) -> object: ...


def _table_actions(holder: _TableHolder) -> ManagedTableActions:
    table = holder.table
    if isinstance(table, ManagedTableActions):
        return table
    raise TypeError("Prisma table object does not expose find/update/upsert actions")


def file_table_actions(prisma_client: PrismaClient) -> ManagedTableActions:
    return _table_actions(ManagedFileRepository(prisma_client))


def object_table_actions(prisma_client: PrismaClient) -> ManagedTableActions:
    return _table_actions(ManagedObjectRepository(prisma_client))


def as_managed_file_reader(hook: CustomLogger) -> ManagedFileReaderLike:
    if isinstance(hook, ManagedFileReaderLike):
        return hook
    raise TypeError("managed_files hook does not expose get_unified_file_id")


def as_managed_file_writer(hook: CustomLogger) -> ManagedFileWriterLike:
    if isinstance(hook, ManagedFileWriterLike):
        return hook
    raise TypeError("managed_files hook does not expose store_unified_file_id")


def as_file_row(record: object) -> ManagedFileRowLike | None:
    if isinstance(record, ManagedFileRowLike):
        return record
    return None


def as_object_row(record: object) -> ManagedObjectRowLike | None:
    if isinstance(record, ManagedObjectRowLike):
        return record
    return None


def as_file_rows(records: Sequence[object] | None) -> list[ManagedFileRowLike]:
    if records is None:
        return []
    return [record for record in records if isinstance(record, ManagedFileRowLike)]


def as_object_rows(records: Sequence[object] | None) -> list[ManagedObjectRowLike]:
    if records is None:
        return []
    return [record for record in records if isinstance(record, ManagedObjectRowLike)]
