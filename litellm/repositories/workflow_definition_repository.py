import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final
from uuid import uuid4

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
from litellm.repositories.table_repositories import PrismaTableRepository, TeamRepository, WorkflowSnapshotRepository

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models


def _definition(row: "prisma_models.LiteLLM_WorkflowDefinition") -> WorkflowDefinition:
    return WorkflowDefinition(
        draft=WorkflowDraft(
            workflow_id=row.workflow_id,
            name=row.name,
            team_id=row.team_id,
            etag=row.etag,
            digest=row.digest,
            bundle=WorkflowBundle.model_validate_json(row.bundle_json),
            published_snapshot_id=row.published_snapshot_id,
        ),
        contributors=tuple(row.contributors),
    )


def _snapshot(row: "prisma_models.LiteLLM_WorkflowSnapshot") -> WorkflowSnapshot:
    return WorkflowSnapshot(
        snapshot_id=row.snapshot_id,
        workflow_id=row.workflow_id,
        team_id=row.team_id,
        name=row.name,
        digest=row.digest,
        bundle=WorkflowBundle.model_validate_json(row.bundle_json),
        contributors=tuple(row.contributors),
        published_by=row.published_by,
        created_at=row.created_at,
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


def _prepared_snapshot(row: "prisma_models.LiteLLM_WorkflowDefinition", published_by: str | None) -> WorkflowSnapshot:
    return WorkflowSnapshot(
        snapshot_id=str(uuid4()),
        workflow_id=row.workflow_id,
        team_id=row.team_id,
        name=row.name,
        digest=row.digest,
        bundle=WorkflowBundle.model_validate_json(row.bundle_json),
        contributors=tuple(row.contributors),
        published_by=published_by,
        created_at=datetime.now(timezone.utc),
    )


async def persist_snapshot(transaction: "Prisma", snapshot: WorkflowSnapshot) -> WorkflowSnapshot:
    """Persist a prepared definition in the caller's publication/admission transaction."""
    data: Final = {
        "snapshot_id": snapshot.snapshot_id,
        "workflow_id": snapshot.workflow_id,
        "team_id": snapshot.team_id,
        "name": snapshot.name,
        "digest": snapshot.digest,
        "bundle_json": _bundle_json(snapshot.bundle),
        "contributors": list(snapshot.contributors),
        "published_by": snapshot.published_by,
        "created_at": snapshot.created_at,
    }
    stored: Final = await WorkflowSnapshotRepository.transaction_table(transaction).create(data=data)
    return _snapshot(stored)


async def _locked_definition(
    table: TableActions["prisma_models.LiteLLM_WorkflowDefinition"], workflow_id: str, team_id: str, etag: str
) -> "prisma_models.LiteLLM_WorkflowDefinition | DefinitionError":
    # Preserve audit timestamps while taking the row lock. If a concurrent
    # publication changed the row, re-read its contributor boundary before editing.
    for _ in range(3):
        row = await table.find_first(where={"workflow_id": workflow_id, "team_id": team_id})
        if row is None:
            return DefinitionError("not_found", "Workflow not found")
        if row.etag != etag:
            break
        if await table.update_many(
            where={"workflow_id": workflow_id, "team_id": team_id, "etag": etag, "updated_at": row.updated_at},
            data={"updated_at": row.updated_at},
        ):
            locked = await table.find_unique(where={"workflow_id": workflow_id})
            if locked is not None:
                return locked
            return DefinitionError("not_found", "Workflow not found")
    return DefinitionError("stale_draft", "The draft has changed. Read it again before retrying", "If-Match")


class WorkflowDefinitionRepository(PrismaTableRepository["prisma_models.LiteLLM_WorkflowDefinition"]):
    table_name = "litellm_workflowdefinition"

    async def get_team(self, team_id: str) -> LiteLLM_TeamTable | None:
        row: Final = await TeamRepository(self._prisma_client, use_writer=True).table.find_unique(
            where={"team_id": team_id}
        )
        return None if row is None else LiteLLM_TeamTable.model_validate(row, from_attributes=True)

    async def get_draft(self, workflow_id: str, team_id: str | None = None) -> WorkflowDefinition | None:
        row: Final = await self.table.find_first(
            where={"workflow_id": workflow_id, **({"team_id": team_id} if team_id is not None else {})}
        )
        return None if row is None else _definition(row)

    async def list_drafts(self, team_id: str, limit: int, cursor: str | None) -> DefinitionPage:
        rows: Final = await self.table.find_many(
            where={"team_id": team_id, **({"workflow_id": {"gt": cursor}} if cursor is not None else {})},
            order={"workflow_id": "asc"},
            take=limit + 1,
        )
        drafts: Final = tuple(_definition(row).draft for row in rows[:limit])
        return DefinitionPage(workflows=drafts, next_cursor=drafts[-1].workflow_id if len(rows) > limit else None)

    async def get_snapshot(self, snapshot_id: str, team_id: str | None = None) -> WorkflowSnapshot | None:
        row: Final = await WorkflowSnapshotRepository(self._prisma_client, use_writer=True).table.find_first(
            where={"snapshot_id": snapshot_id, **({"team_id": team_id} if team_id is not None else {})}
        )
        return None if row is None else _snapshot(row)

    async def get_published(self, name: str, team_id: str) -> WorkflowSnapshot | None:
        row: Final = await self.table.find_first(where={"name": name, "team_id": team_id})
        if row is None or row.published_snapshot_id is None:
            return None
        return await self.get_snapshot(row.published_snapshot_id, team_id)

    async def create(self, bundle: WorkflowBundle, team_id: str, contributor: str) -> WorkflowDraft | DefinitionError:
        from prisma.errors import UniqueViolationError

        try:
            stored: Final = await self.table.create(
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
            return _definition(stored).draft
        except UniqueViolationError:
            return DefinitionError("conflict", "A workflow with this name already exists in the team", "bundle.name")

    async def replace(
        self, workflow_id: str, team_id: str, etag: str, bundle: WorkflowBundle, contributor: str
    ) -> WorkflowDraft | DefinitionError:
        database: Final[Prisma] = self.writer_db
        async with database.tx() as transaction:
            row: Final = await _locked_definition(self.transaction_table(transaction), workflow_id, team_id, etag)
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
            await self.transaction_table(transaction).update(
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
        database: Final[Prisma] = self.writer_db
        async with database.tx() as transaction:
            row: Final = await _locked_definition(self.transaction_table(transaction), workflow_id, team_id, etag)
            if isinstance(row, DefinitionError):
                return row
            if row.digest != digest:
                return DefinitionError("stale_draft", "The reviewed digest does not match the draft", "digest")
            if publisher in row.contributors:
                return DefinitionError("forbidden", "Draft contributors cannot publish their own changes")
            if row.published_etag == etag and row.published_snapshot_id is not None:
                existing: Final = await WorkflowSnapshotRepository.transaction_table(transaction).find_unique(
                    where={"snapshot_id": row.published_snapshot_id}
                )
                if existing is not None:
                    return _snapshot(existing)
            snapshot: Final = await persist_snapshot(transaction, _prepared_snapshot(row, publisher))
            await self.transaction_table(transaction).update(
                where={"workflow_id": workflow_id},
                data={"published_snapshot_id": snapshot.snapshot_id, "published_etag": etag},
            )
            return snapshot

    async def prepare_draft(self, workflow_id: str, team_id: str, etag: str) -> WorkflowSnapshot | DefinitionError:
        row: Final = await self.table.find_first(where={"workflow_id": workflow_id, "team_id": team_id})
        if row is None:
            return DefinitionError("not_found", "Workflow not found")
        if row.etag != etag:
            return DefinitionError("stale_draft", "The draft has changed. Read it again before retrying", "If-Match")
        return _prepared_snapshot(row, None)
