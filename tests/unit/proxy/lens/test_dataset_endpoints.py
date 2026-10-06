from datetime import datetime
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.dataset_endpoints import create_dataset, eval_cases, read_dataset, save_revision
from litellm.proxy.lens.dataset_repository import StoredSummary
from litellm.proxy.lens.datasets import case_id
from litellm.proxy.lens.models import CaseSource, Dataset, DatasetCase, DatasetCreate, DatasetMessage, RevisionSave

ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin", team_id="alpha")


class MemoryStore:
    def __init__(self) -> None:
        self.rows: Final[dict[tuple[str, int], Dataset]] = {}  # mutable-ok: stands in for the table

    async def summaries(self) -> tuple[StoredSummary, ...]:
        return ()

    async def get(self, dataset_id: str, revision: int | None = None) -> Dataset | None:
        revisions: Final = sorted(r for i, r in self.rows if i == dataset_id)
        wanted: Final = revision if revision is not None else (revisions[-1] if revisions else None)
        return self.rows.get((dataset_id, wanted)) if wanted is not None else None

    async def insert(self, dataset: Dataset, saved_at: datetime) -> bool:
        key: Final = (dataset.id, dataset.revision)
        if key in self.rows:
            return False
        self.rows[key] = dataset
        return True


def case(text: str, included: bool = True) -> DatasetCase:
    message: Final = (DatasetMessage(role="user", content=text),)
    return DatasetCase(id=case_id(message, "", ()), messages=message, included=included, source=CaseSource())


@pytest.mark.asyncio
async def test_saving_inserts_a_new_revision_and_leaves_the_older_one_unchanged() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    first: Final = await save_revision(created.id, RevisionSave(base_revision=0, cases=(case("a"),)), ADMIN, store)
    second: Final = await save_revision(
        created.id, RevisionSave(base_revision=1, cases=(case("a"), case("b"))), ADMIN, store
    )

    assert (first.revision, second.revision) == (1, 2)
    assert await read_dataset(created.id, ADMIN, store, revision=1) == first
    assert (await read_dataset(created.id, ADMIN, store, revision=0)).cases == ()
    assert await read_dataset(created.id, ADMIN, store) == second


@pytest.mark.asyncio
async def test_saving_on_a_stale_revision_is_a_conflict_and_writes_nothing() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    await save_revision(created.id, RevisionSave(base_revision=0, cases=(case("a"),)), ADMIN, store)

    with pytest.raises(HTTPException) as error:
        await save_revision(created.id, RevisionSave(base_revision=0, cases=(case("b"),)), ADMIN, store)
    assert error.value.status_code == 409
    assert sorted(store.rows) == [(created.id, 0), (created.id, 1)]


@pytest.mark.asyncio
async def test_saving_dedupes_cases_and_restores_content_hash_ids_but_keeps_edits() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    edited: Final = case("a").model_copy(update={"id": "forged", "expected": "say hi"})
    saved: Final = await save_revision(
        created.id, RevisionSave(base_revision=0, cases=(edited, case("a"))), ADMIN, store
    )

    assert len(saved.cases) == 1
    assert saved.cases[0].id == case("a").id
    assert saved.cases[0].expected == "say hi"


@pytest.mark.asyncio
async def test_eval_cases_return_only_included_cases_of_the_requested_revision() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    await save_revision(
        created.id, RevisionSave(base_revision=0, cases=(case("a"), case("b", included=False))), ADMIN, store
    )
    await save_revision(created.id, RevisionSave(base_revision=1, cases=(case("c"),)), ADMIN, store)

    cases: Final = await eval_cases(created.id, 1, ADMIN, store)
    assert (cases.dataset_id, cases.revision) == (created.id, 1)
    assert tuple(c.messages[0].content for c in cases.cases) == ("a",)


@pytest.mark.asyncio
async def test_read_only_admin_can_read_but_cannot_save_a_revision() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    viewer: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

    assert (await read_dataset(created.id, viewer, store)).id == created.id
    with pytest.raises(HTTPException) as error:
        await save_revision(created.id, RevisionSave(base_revision=0, cases=(case("a"),)), viewer, store)
    assert error.value.status_code == 403


async def create_dataset_named(store: MemoryStore) -> Dataset:
    return await create_dataset(DatasetCreate(name="Refunds", agent_name="support"), ADMIN, store)
