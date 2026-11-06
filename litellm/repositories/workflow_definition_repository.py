from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import Final, Protocol, TypeVar
from uuid import uuid4

from liteflow import bundle_digest  # pyright: ignore[reportMissingTypeStubs]  # pinned SDK omits its py.typed marker
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.models.team import LiteLLM_TeamTable
from litellm.models.workflow_definition import (
    DefinitionError,
    DefinitionPage,
    WorkflowBundle,
    WorkflowDefinition,
    WorkflowDraft,
    WorkflowSnapshot,
)
from litellm.repositories.base_repository import DbRecord, record_to_dict
from litellm.repositories.prisma_protocols import TableActions

_Model: Final = TypeVar("_Model", bound=BaseModel)
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_STRING: Final = TypeAdapter(str)


class SnapshotTransaction(Protocol):
    @property
    def litellm_workflowsnapshot(self) -> TableActions[DbRecord]: ...


class DefinitionTransaction(SnapshotTransaction, Protocol):
    @property
    def litellm_workflowdefinition(self) -> TableActions[DbRecord]: ...

    @property
    def litellm_teamtable(self) -> TableActions[DbRecord]: ...

    async def query_raw(self, query: str, *args: object) -> object: ...


class DefinitionDatabase(DefinitionTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[DefinitionTransaction]: ...


class _DefinitionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    draft: WorkflowDraft
    contributors: tuple[str, ...]
    published_etag: str | None


def _decoded(row: DbRecord, model: type[_Model]) -> _Model:
    values: Final = record_to_dict(row)
    bundle: Final = _JSON.validate_json(_STRING.validate_python(values["bundle_json"]))
    return model.model_validate({field: bundle if field == "bundle" else values[field] for field in model.model_fields})


def _definition(row: DbRecord) -> _DefinitionRow:
    return _DefinitionRow.model_validate({**record_to_dict(row), "draft": _decoded(row, WorkflowDraft)})


def _content(bundle: WorkflowBundle) -> Mapping[str, object]:
    serialized: Final = bundle.model_dump_json(by_alias=True, exclude_unset=True)
    return {
        "etag": f'"{uuid4()}"',
        "digest": bundle_digest(_JSON.validate_json(serialized)),
        "bundle_json": serialized,
    }


async def _locked_definition(
    transaction: DefinitionTransaction, workflow_id: str, team_id: str, etag: str
) -> _DefinitionRow | DefinitionError:
    await transaction.query_raw(
        'SELECT workflow_id FROM "LiteLLM_WorkflowDefinition" WHERE workflow_id = $1 AND team_id = $2 FOR UPDATE',
        workflow_id,
        team_id,
    )
    stored: Final = await transaction.litellm_workflowdefinition.find_first(
        where={"workflow_id": workflow_id, "team_id": team_id}
    )
    if stored is None:
        return DefinitionError("not_found", "Workflow not found")
    row: Final = _definition(stored)
    if row.draft.etag != etag:
        return DefinitionError("stale_draft", "The draft has changed. Read it again before retrying", "If-Match")
    return row


def _prepared_snapshot(row: _DefinitionRow, published_by: str | None) -> WorkflowSnapshot:
    return WorkflowSnapshot.model_validate(
        {
            **row.draft.model_dump(exclude={"etag", "published_snapshot_id"}),
            "snapshot_id": str(uuid4()),
            "contributors": row.contributors,
            "published_by": published_by,
            "created_at": datetime.now(timezone.utc),
        }
    )


async def persist_snapshot(transaction: SnapshotTransaction, snapshot: WorkflowSnapshot) -> WorkflowSnapshot:
    stored: Final = await transaction.litellm_workflowsnapshot.create(
        data={
            **snapshot.model_dump(exclude={"bundle", "contributors"}),
            "bundle_json": snapshot.bundle.model_dump_json(by_alias=True, exclude_unset=True),
            "contributors": list(snapshot.contributors),
        }
    )
    return _decoded(stored, WorkflowSnapshot)


class WorkflowDefinitionRepository:
    def __init__(self, database: DefinitionDatabase) -> None:
        self._database: Final = database

    async def get_team(self, team_id: str) -> LiteLLM_TeamTable | None:
        row: Final = await self._database.litellm_teamtable.find_unique(where={"team_id": team_id})
        return None if row is None else LiteLLM_TeamTable.model_validate(record_to_dict(row))

    async def get_draft(self, workflow_id: str, team_id: str | None = None) -> WorkflowDefinition | None:
        stored: Final = await self._database.litellm_workflowdefinition.find_first(
            where={"workflow_id": workflow_id, **({"team_id": team_id} if team_id is not None else {})}
        )
        if stored is None:
            return None
        row: Final = _definition(stored)
        return WorkflowDefinition(draft=row.draft, contributors=row.contributors)

    async def list_drafts(self, team_id: str, limit: int, cursor: str | None) -> DefinitionPage:
        rows: Final = await self._database.litellm_workflowdefinition.find_many(
            where={"team_id": team_id, **({"workflow_id": {"gt": cursor}} if cursor is not None else {})},
            order={"workflow_id": "asc"},
            take=limit + 1,
        )
        drafts: Final = tuple(_decoded(row, WorkflowDraft) for row in rows[:limit])
        return DefinitionPage(workflows=drafts, next_cursor=drafts[-1].workflow_id if len(rows) > limit else None)

    async def get_snapshot(self, snapshot_id: str, team_id: str | None = None) -> WorkflowSnapshot | None:
        row: Final = await self._database.litellm_workflowsnapshot.find_first(
            where={"snapshot_id": snapshot_id, **({"team_id": team_id} if team_id is not None else {})}
        )
        return None if row is None else _decoded(row, WorkflowSnapshot)

    async def get_published(self, name: str, team_id: str) -> WorkflowSnapshot | None:
        row: Final = await self._database.litellm_workflowdefinition.find_first(
            where={"name": name, "team_id": team_id}
        )
        if row is None or (snapshot_id := _decoded(row, WorkflowDraft).published_snapshot_id) is None:
            return None
        return await self.get_snapshot(snapshot_id, team_id)

    async def create(self, bundle: WorkflowBundle, team_id: str, contributor: str) -> WorkflowDraft | DefinitionError:
        from prisma.errors import UniqueViolationError

        try:
            stored: Final = await self._database.litellm_workflowdefinition.create(
                data={
                    "workflow_id": str(uuid4()),
                    "name": bundle.name,
                    "team_id": team_id,
                    "contributors": [contributor],
                    **_content(bundle),
                }
            )
        except UniqueViolationError:
            return DefinitionError("conflict", "A workflow with this name already exists in the team", "bundle.name")
        return _decoded(stored, WorkflowDraft)

    async def replace(
        self, workflow_id: str, team_id: str, etag: str, bundle: WorkflowBundle, contributor: str
    ) -> WorkflowDraft | DefinitionError:
        async with self._database.tx() as transaction:
            row: Final = await _locked_definition(transaction, workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            if row.draft.name != bundle.name:
                return DefinitionError("conflict", "A workflow name cannot change", "bundle.name")
            contributors: Final = (
                (contributor,)
                if etag == row.published_etag
                else tuple(sorted(frozenset(row.contributors) | {contributor}))
            )
            stored: Final = await transaction.litellm_workflowdefinition.update(
                where={"workflow_id": workflow_id}, data={**_content(bundle), "contributors": list(contributors)}
            )
            assert stored is not None
            return _decoded(stored, WorkflowDraft)

    async def publish(
        self, workflow_id: str, team_id: str, etag: str, digest: str, publisher: str
    ) -> WorkflowSnapshot | DefinitionError:
        async with self._database.tx() as transaction:
            row: Final = await _locked_definition(transaction, workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            if row.draft.digest != digest:
                return DefinitionError("stale_draft", "The reviewed digest does not match the draft", "digest")
            if publisher in row.contributors:
                return DefinitionError("forbidden", "Draft contributors cannot publish their own changes")
            if row.published_etag == etag and row.draft.published_snapshot_id is not None:
                existing: Final = await transaction.litellm_workflowsnapshot.find_unique(
                    where={"snapshot_id": row.draft.published_snapshot_id}
                )
                assert existing is not None
                return _decoded(existing, WorkflowSnapshot)
            snapshot: Final = await persist_snapshot(transaction, _prepared_snapshot(row, publisher))
            await transaction.litellm_workflowdefinition.update(
                where={"workflow_id": workflow_id},
                data={"published_snapshot_id": snapshot.snapshot_id, "published_etag": etag},
            )
            return snapshot

    async def prepare_draft(self, workflow_id: str, team_id: str, etag: str) -> WorkflowSnapshot | DefinitionError:
        async with self._database.tx() as transaction:
            row: Final = await _locked_definition(transaction, workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            return _prepared_snapshot(row, None)
