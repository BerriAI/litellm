from typing import Final

import pytest
from liteflow import Draft, bundle_digest

from litellm.models.workflow_definition import DefinitionError, WorkflowSnapshot, parsed_bundle
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle


@pytest.mark.asyncio
async def test_publication_is_immutable_and_preparing_draft_does_not_persist() -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, Draft) and draft.digest == bundle_digest(draft.bundle)
    published: Final = await store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    frozen: Final = await store.prepare_draft(draft.workflow_id, "team", draft.etag)
    assert isinstance(published, WorkflowSnapshot) and isinstance(frozen, WorkflowSnapshot)
    assert published.bundle == frozen.bundle == parsed_bundle(draft.bundle)
    assert frozen.published_by is None
    assert await store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer") == published
    updated: Final = await store.replace(draft.workflow_id, "team", draft.etag, bundle(model="second"), "other")
    assert isinstance(updated, Draft) and updated.published_snapshot_id == published.snapshot_id
    assert await store.get_published("example", "team") == published
    stale: Final = await store.prepare_draft(draft.workflow_id, "team", draft.etag)
    assert isinstance(stale, DefinitionError) and stale.code == "stale_draft"
    second: Final = await store.publish(updated.workflow_id, "team", updated.etag, updated.digest, "reviewer")
    assert isinstance(second, WorkflowSnapshot) and second.contributors == ("other",)
    assert second.snapshot_id != published.snapshot_id and second.digest != published.digest
    assert await store.get_snapshot(published.snapshot_id) == published
    assert await store.get_snapshot(frozen.snapshot_id) is None
    assert await store.get_snapshot(second.snapshot_id, "other-team") is None
    assert await store.get_published("example", "team") == second
