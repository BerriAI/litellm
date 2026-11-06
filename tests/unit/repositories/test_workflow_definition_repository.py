import asyncio
from typing import Final

import pytest

from litellm.models.workflow_definition import DefinitionError, WorkflowDraft, WorkflowSnapshot
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository, persist_snapshot
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle


@pytest.mark.asyncio
async def test_edit_and_republish_leave_old_snapshot_and_frozen_draft_unchanged() -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    published: Final = await store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(published, WorkflowSnapshot)
    prepared: Final = await store.prepare_draft(draft.workflow_id, "team", draft.etag)
    assert isinstance(prepared, WorkflowSnapshot)
    assert await store.get_snapshot(prepared.snapshot_id) is None
    frozen: Final = await persist_snapshot(store.prisma_client.writer_db, prepared)
    assert frozen.published_by is None
    updated: Final = await store.replace(draft.workflow_id, "team", draft.etag, bundle(model="second"), "other")
    assert isinstance(updated, WorkflowDraft)
    assert updated.published_snapshot_id == published.snapshot_id
    assert await store.get_published("example", "team") == published
    second: Final = await store.publish(updated.workflow_id, "team", updated.etag, updated.digest, "reviewer")
    assert isinstance(second, WorkflowSnapshot)
    assert second.snapshot_id != published.snapshot_id
    assert second.contributors == ("other",)
    assert await store.get_snapshot(published.snapshot_id) == published
    assert await store.get_snapshot(frozen.snapshot_id) == frozen
    assert await store.get_published("example", "team") == second
    assert await store.get_snapshot(second.snapshot_id, "other-team") is None


@pytest.mark.asyncio
async def test_two_edits_of_same_revision_accept_only_one_and_preserve_contributors() -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    edits: Final = await asyncio.gather(
        store.replace(draft.workflow_id, "team", draft.etag, bundle(model="one"), "editor-one"),
        store.replace(draft.workflow_id, "team", draft.etag, bundle(model="two"), "editor-two"),
    )
    accepted: Final = tuple(edit for edit in edits if isinstance(edit, WorkflowDraft))
    rejected: Final = tuple(edit for edit in edits if isinstance(edit, DefinitionError))
    assert len(accepted) == 1
    assert rejected == (
        DefinitionError("stale_draft", "The draft has changed. Read it again before retrying", "If-Match"),
    )
    stored: Final = await store.get_draft(draft.workflow_id)
    assert stored is not None
    assert stored.draft == accepted[0]
    assert "author" in stored.contributors
    assert len(stored.contributors) == 2


@pytest.mark.asyncio
async def test_publish_and_edit_race_never_snapshots_unreviewed_content() -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    published, edited = await asyncio.gather(
        store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer"),
        store.replace(draft.workflow_id, "team", draft.etag, bundle(model="changed"), "other"),
    )
    assert isinstance(published, WorkflowSnapshot)
    assert isinstance(edited, WorkflowDraft)
    assert published.bundle == draft.bundle
    assert published.digest == draft.digest
    assert edited.digest != published.digest
    assert await store.get_published("example", "team") == published


@pytest.mark.asyncio
async def test_edit_rereads_publication_after_lock_even_when_audit_timestamp_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database: Final = DefinitionRows()
    store: Final = WorkflowDefinitionRepository(database.database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    table: Final = database.litellm_workflowdefinition
    stale: Final = await table.find_first(where={"workflow_id": draft.workflow_id})
    published: Final = await store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(published, WorkflowSnapshot)
    original: Final = table.find_first
    pending = True

    async def read_once_before_publication(*args, **kwargs):
        nonlocal pending
        if pending:
            pending = False
            return stale
        return await original(*args, **kwargs)

    monkeypatch.setattr(table, "find_first", read_once_before_publication)
    edited: Final = await store.replace(draft.workflow_id, "team", draft.etag, bundle(model="edited"), "other")
    assert isinstance(edited, WorkflowDraft)
    assert edited.published_snapshot_id == published.snapshot_id
    stored: Final = await store.get_draft(draft.workflow_id)
    assert stored is not None and stored.contributors == ("other",)
