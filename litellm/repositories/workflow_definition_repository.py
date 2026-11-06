import hashlib
import json
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Final, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from litellm.models.team import LiteLLM_TeamTable
from litellm.models.workflow_definition import (
    DefinitionError,
    DefinitionPage,
    WorkflowBundle,
    WorkflowDefinition,
    WorkflowDraft,
    WorkflowSnapshot,
)
from litellm.repositories.prisma_protocols import TableActions


class DefinitionTransaction(Protocol):
    @property
    def litellm_workflowdefinition(self) -> TableActions[object]: ...

    @property
    def litellm_workflowsnapshot(self) -> TableActions[object]: ...

    @property
    def litellm_teamtable(self) -> TableActions[object]: ...

    async def query_raw(self, query: str, *args: object) -> object: ...


class DefinitionDatabase(DefinitionTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[DefinitionTransaction]: ...


class _DefinitionRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    workflow_id: str
    name: str
    team_id: str
    etag: str
    digest: str
    bundle_json: str
    contributors: tuple[str, ...]
    published_snapshot_id: str | None
    published_etag: str | None

    def definition(self) -> WorkflowDefinition:
        return WorkflowDefinition(
            draft=WorkflowDraft(
                workflow_id=self.workflow_id,
                name=self.name,
                team_id=self.team_id,
                etag=self.etag,
                digest=self.digest,
                bundle=WorkflowBundle.model_validate_json(self.bundle_json),
                published_snapshot_id=self.published_snapshot_id,
            ),
            contributors=self.contributors,
        )


class _SnapshotRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    snapshot_id: str
    workflow_id: str
    team_id: str
    name: str
    digest: str
    bundle_json: str
    contributors: tuple[str, ...]
    published_by: str | None
    created_at: datetime

    def snapshot(self) -> WorkflowSnapshot:
        return WorkflowSnapshot(
            snapshot_id=self.snapshot_id,
            workflow_id=self.workflow_id,
            team_id=self.team_id,
            name=self.name,
            digest=self.digest,
            bundle=WorkflowBundle.model_validate_json(self.bundle_json),
            contributors=self.contributors,
            published_by=self.published_by,
            created_at=self.created_at,
        )


def bundle_digest(bundle: WorkflowBundle) -> str:
    return hashlib.sha256(
        json.dumps(
            bundle.model_dump(mode="json", by_alias=True, exclude_unset=True),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _bundle_json(bundle: WorkflowBundle) -> str:
    return bundle.model_dump_json(by_alias=True, exclude_unset=True)


def _etag() -> str:
    return f'"{uuid4()}"'


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
    row: Final = _DefinitionRow.model_validate(stored)
    if row.etag != etag:
        return DefinitionError("stale_draft", "The draft has changed. Read it again before retrying", "If-Match")
    return row


async def _snapshot(
    transaction: DefinitionTransaction, row: _DefinitionRow, published_by: str | None
) -> WorkflowSnapshot:
    stored: Final = await transaction.litellm_workflowsnapshot.create(
        data={
            "snapshot_id": str(uuid4()),
            "workflow_id": row.workflow_id,
            "team_id": row.team_id,
            "name": row.name,
            "digest": row.digest,
            "bundle_json": row.bundle_json,
            "contributors": list(row.contributors),
            "published_by": published_by,
        }
    )
    return _SnapshotRow.model_validate(stored).snapshot()


class WorkflowDefinitionRepository:
    def __init__(self, database: DefinitionDatabase) -> None:
        self._database: Final = database

    async def get_team(self, team_id: str) -> LiteLLM_TeamTable | None:
        row: Final = await self._database.litellm_teamtable.find_unique(where={"team_id": team_id})
        return None if row is None else LiteLLM_TeamTable.model_validate(row, from_attributes=True)

    async def get_draft(self, workflow_id: str, team_id: str | None = None) -> WorkflowDefinition | None:
        row: Final = await self._database.litellm_workflowdefinition.find_first(
            where={"workflow_id": workflow_id, **({"team_id": team_id} if team_id is not None else {})}
        )
        return None if row is None else _DefinitionRow.model_validate(row).definition()

    async def list_drafts(self, team_id: str, limit: int, cursor: str | None) -> DefinitionPage:
        rows: Final = await self._database.litellm_workflowdefinition.find_many(
            where={"team_id": team_id, **({"workflow_id": {"gt": cursor}} if cursor is not None else {})},
            order={"workflow_id": "asc"},
            take=limit + 1,
        )
        drafts: Final = tuple(_DefinitionRow.model_validate(row).definition().draft for row in rows[:limit])
        return DefinitionPage(workflows=drafts, next_cursor=drafts[-1].workflow_id if len(rows) > limit else None)

    async def get_snapshot(self, snapshot_id: str, team_id: str | None = None) -> WorkflowSnapshot | None:
        row: Final = await self._database.litellm_workflowsnapshot.find_first(
            where={"snapshot_id": snapshot_id, **({"team_id": team_id} if team_id is not None else {})}
        )
        return None if row is None else _SnapshotRow.model_validate(row).snapshot()

    async def get_published(self, name: str, team_id: str) -> WorkflowSnapshot | None:
        row: Final = await self._database.litellm_workflowdefinition.find_first(
            where={"name": name, "team_id": team_id}
        )
        if row is None:
            return None
        definition: Final = _DefinitionRow.model_validate(row)
        if definition.published_snapshot_id is None:
            return None
        return await self.get_snapshot(definition.published_snapshot_id, team_id)

    async def create(self, bundle: WorkflowBundle, team_id: str, contributor: str) -> WorkflowDraft | DefinitionError:
        from prisma.errors import UniqueViolationError

        try:
            stored: Final = await self._database.litellm_workflowdefinition.create(
                data={
                    "workflow_id": str(uuid4()),
                    "name": bundle.name,
                    "team_id": team_id,
                    "etag": _etag(),
                    "digest": bundle_digest(bundle),
                    "bundle_json": _bundle_json(bundle),
                    "contributors": [contributor],
                }
            )
            return _DefinitionRow.model_validate(stored).definition().draft
        except UniqueViolationError:
            return DefinitionError("conflict", "A workflow with this name already exists in the team", "bundle.name")

    async def replace(
        self, workflow_id: str, team_id: str, etag: str, bundle: WorkflowBundle, contributor: str
    ) -> WorkflowDraft | DefinitionError:
        async with self._database.tx() as transaction:
            row: Final = await _locked_definition(transaction, workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            if row.name != bundle.name:
                return DefinitionError("conflict", "A workflow name cannot change", "bundle.name")
            contributors: Final = (
                (contributor,)
                if row.etag == row.published_etag
                else tuple(sorted(frozenset(row.contributors) | {contributor}))
            )
            new_etag: Final = _etag()
            digest: Final = bundle_digest(bundle)
            await transaction.litellm_workflowdefinition.update(
                where={"workflow_id": workflow_id},
                data={
                    "etag": new_etag,
                    "digest": digest,
                    "bundle_json": _bundle_json(bundle),
                    "contributors": list(contributors),
                },
            )
            return WorkflowDraft(
                name=row.name,
                workflow_id=workflow_id,
                team_id=team_id,
                etag=new_etag,
                digest=digest,
                bundle=bundle,
                published_snapshot_id=row.published_snapshot_id,
            )

    async def publish(
        self, workflow_id: str, team_id: str, etag: str, digest: str, publisher: str
    ) -> WorkflowSnapshot | DefinitionError:
        async with self._database.tx() as transaction:
            row: Final = await _locked_definition(transaction, workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            if row.digest != digest:
                return DefinitionError("stale_draft", "The reviewed digest does not match the draft", "digest")
            if publisher in row.contributors:
                return DefinitionError("forbidden", "Draft contributors cannot publish their own changes")
            if row.published_etag == etag and row.published_snapshot_id is not None:
                existing: Final = await transaction.litellm_workflowsnapshot.find_unique(
                    where={"snapshot_id": row.published_snapshot_id}
                )
                return _SnapshotRow.model_validate(existing).snapshot()
            snapshot: Final = await _snapshot(transaction, row, publisher)
            await transaction.litellm_workflowdefinition.update(
                where={"workflow_id": workflow_id},
                data={"published_snapshot_id": snapshot.snapshot_id, "published_etag": etag},
            )
            return snapshot

    async def freeze_draft(self, workflow_id: str, team_id: str, etag: str) -> WorkflowSnapshot | DefinitionError:
        async with self._database.tx() as transaction:
            row: Final = await _locked_definition(transaction, workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            return await _snapshot(transaction, row, None)
