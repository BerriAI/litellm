import json
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.constants import LENS_DATASET_MAX_CASE_CHARS, LENS_DATASET_MAX_CASES
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.dataset_endpoints import (
    ProxyDatasetReader,
    build_dataset_cases,
    create_dataset,
    eval_cases,
    export_dataset,
    list_datasets,
    read_dataset,
    save_revision,
)
from litellm.proxy.lens.dataset_repository import StoredSummary
from litellm.proxy.lens.datasets import case_id
from litellm.proxy.lens.models import (
    BuildRequest,
    CaseSource,
    Dataset,
    DatasetCase,
    DatasetCreate,
    DatasetMessage,
    DatasetSummary,
    Evidence,
    Lens,
    RevisionSave,
    Scope,
    TextSource,
)
from litellm.proxy.lens.repository import LensRepository, Row
from litellm.rust_bridge.trace.errors import TraceChanged
from litellm.rust_bridge.trace.generated.types import SpanDetail, Trace, TraceScope
from litellm.rust_bridge.trace.storage import ClickHouseStorage
from litellm.tracing import TraceReceiver
from tests.unit.proxy.lens.test_datasets import detail, span, stored_finding, trace
from tests.unit.proxy.lens.test_state import lens

ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin", team_id="alpha")


class MemoryStore:
    def __init__(self) -> None:
        self.rows: Final[dict[tuple[str, int], Dataset]] = {}

    async def summaries(self) -> tuple[StoredSummary, ...]:
        latest: Final = {i: d for (i, _), d in sorted(self.rows.items(), key=lambda item: item[0][1])}
        return tuple(StoredSummary(team_id=d.team_id, summary=summary(d)) for d in latest.values())

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


def summary(dataset: Dataset) -> DatasetSummary:
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        agent_name=dataset.agent_name,
        revision=dataset.revision,
        case_count=len(dataset.cases),
        updated_at=dataset.created_at,
    )


class RejectingStore:
    def __init__(self, stored: Dataset | None = None) -> None:
        self.stored: Final = stored

    async def summaries(self) -> tuple[StoredSummary, ...]:
        return ()

    async def get(self, dataset_id: str, revision: int | None = None) -> Dataset | None:
        return self.stored

    async def insert(self, dataset: Dataset, saved_at: datetime) -> bool:
        return False


class TraceStorage(ClickHouseStorage):
    def __init__(
        self,
        pages: Mapping[str | None, Trace],
        spans: Mapping[str, SpanDetail] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.pages: Final = pages
        self.spans: Final = spans or {}
        self.error: Final = error

    async def get_trace(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        if self.error:
            raise self.error
        return self.pages.get(cursor)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        if self.error:
            raise self.error
        return self.spans.get(span_id)


class LensTable:
    def __init__(self, stored: Lens) -> None:
        self.stored: Final = stored

    async def query_raw(self, query: str, *args: object) -> tuple[Row, ...]:
        return (Row(data=self.stored.model_dump(mode="json")),) if args == (self.stored.id,) else ()

    async def execute_raw(self, query: str, *args: object) -> int:
        return 0


def page(next_cursor: str | None, *span_ids: str) -> Trace:
    return {**trace(*(span(s, "llm", i) for i, s in enumerate(span_ids))), "next_cursor": next_cursor}


def lenses(stored: Lens | None = None) -> LensRepository:
    return LensRepository(LensTable(stored or lens()))


def reader(
    storage: TraceStorage | None = None, stored: Lens | None = None, scope: Scope | None = None
) -> ProxyDatasetReader:
    return ProxyDatasetReader(
        TraceReceiver(storage) if storage else None,
        lenses(stored),
        scope or Scope(all_teams=True),
    )


@pytest.mark.asyncio
async def test_reading_a_trace_follows_every_cursor_page_and_joins_the_spans() -> None:
    storage: Final = TraceStorage({None: page("c1", "a", "b"), "c1": page("c2", "c"), "c2": page(None, "d")})
    read: Final = await reader(storage).trace("t1", "ref")

    assert read is not None
    assert tuple(s["span_id"] for s in read["spans"]) == ("a", "b", "c", "d")
    assert read.get("next_cursor") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("pages", ({}, {None: page("c1", "a")}), ids=("missing-trace", "page-disappears"))
async def test_reading_a_trace_is_none_when_the_trace_or_a_later_page_is_gone(
    pages: Mapping[str | None, Trace],
) -> None:
    assert await reader(TraceStorage(pages)).trace("t1", "ref") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"), ((TraceChanged("moved"), 409), (ValueError("bad cursor"), 400), (RuntimeError("down"), 503))
)
async def test_trace_and_span_read_failures_become_the_matching_http_error(error: Exception, status: int) -> None:
    failing: Final = reader(TraceStorage({}, error=error))

    with pytest.raises(HTTPException) as traced:
        await failing.trace("t1", "ref")
    with pytest.raises(HTTPException) as spanned:
        await failing.span("t1", "s1", "ref")
    assert (traced.value.status_code, spanned.value.status_code) == (status, status)


@pytest.mark.asyncio
async def test_reading_traces_without_tracing_enabled_is_not_implemented() -> None:
    disabled: Final = reader()

    with pytest.raises(HTTPException) as traced:
        await disabled.trace("t1", "ref")
    with pytest.raises(HTTPException) as spanned:
        await disabled.span("t1", "s1", "ref")
    assert (traced.value.status_code, spanned.value.status_code) == (501, 501)


@pytest.mark.asyncio
async def test_reading_a_span_returns_its_detail() -> None:
    stored: Final = detail("s1", "refund?")
    assert await reader(TraceStorage({}, {"s1": stored})).span("t1", "s1", "") == stored


def lens_with_findings() -> Lens:
    evidence: Final = Evidence(execution_id="run", span_id="s1", quote="q")
    return lens().model_copy(
        update={
            "findings": (stored_finding("f1", evidence), stored_finding("f2", evidence), stored_finding("f3", evidence))
        }
    )


@pytest.mark.asyncio
async def test_findings_returns_only_the_requested_ids() -> None:
    found: Final = await reader(stored=lens_with_findings()).findings("lens", ("f3", "f1", "unknown"))
    assert tuple(f.id for f in found) == ("f1", "f3")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lens_id", "scope"),
    (("missing", Scope(all_teams=True)), ("lens", Scope(team_id="beta"))),
    ids=("unknown-lens", "lens-outside-scope"),
)
async def test_findings_of_an_unknown_or_inaccessible_lens_are_not_found(lens_id: str, scope: Scope) -> None:
    with pytest.raises(HTTPException) as error:
        await reader(stored=lens_with_findings(), scope=scope).findings(lens_id, ("f1",))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_list_returns_a_summary_of_the_latest_revision_of_each_dataset() -> None:
    store: Final = MemoryStore()
    first: Final = await create_dataset_named(store)
    second: Final = await create_dataset_named(store)
    await save_revision(first.id, RevisionSave(base_revision=0, cases=(case("a"), case("b"))), ADMIN, store)

    listed: Final = await list_datasets(ADMIN, store)
    assert sorted((s.id, s.revision, s.case_count) for s in listed) == sorted(((first.id, 1, 2), (second.id, 0, 0)))


@pytest.mark.asyncio
async def test_listing_requires_proxy_admin_access() -> None:
    with pytest.raises(HTTPException) as error:
        await list_datasets(UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER), MemoryStore())
    assert error.value.status_code == 403


def text_source(*texts: str) -> TextSource:
    return TextSource(text="\n".join(json.dumps({"messages": [{"role": "user", "content": t}]}) for t in texts))


@pytest.mark.asyncio
async def test_building_into_a_dataset_skips_cases_it_already_holds() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    await save_revision(created.id, RevisionSave(base_revision=0, cases=(case("a"),)), ADMIN, store)
    request: Final = BuildRequest(sources=(text_source("a", "b"),), dataset_id=created.id)

    built: Final = await build_dataset_cases(request, ADMIN, store, lenses(), None)
    fresh: Final = await build_dataset_cases(
        request.model_copy(update={"dataset_id": ""}), ADMIN, store, lenses(), None
    )

    assert tuple(c.messages[0].content for c in built.cases) == ("b",)
    assert tuple(s.reason for s in built.skipped) == ("duplicate",)
    assert tuple(c.messages[0].content for c in fresh.cases) == ("a", "b")


@pytest.mark.asyncio
async def test_building_into_an_unknown_dataset_is_not_found() -> None:
    request: Final = BuildRequest(sources=(text_source("a"),), dataset_id="missing")
    with pytest.raises(HTTPException) as error:
        await build_dataset_cases(request, ADMIN, MemoryStore(), lenses(), None)
    assert error.value.status_code == 404


def exported_texts(body: bytes) -> tuple[str, ...]:
    return tuple(DatasetCase.model_validate_json(line).messages[0].content for line in body.splitlines())


@pytest.mark.asyncio
async def test_export_downloads_included_cases_of_the_requested_revision_as_ndjson() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    await save_revision(
        created.id, RevisionSave(base_revision=0, cases=(case("a"), case("b", included=False))), ADMIN, store
    )
    await save_revision(created.id, RevisionSave(base_revision=1, cases=(case("c"),)), ADMIN, store)

    exported: Final = await export_dataset(created.id, ADMIN, store, revision=1)
    latest: Final = await export_dataset(created.id, ADMIN, store)

    assert exported.media_type == "application/x-ndjson"
    assert exported.headers["content-disposition"] == f'attachment; filename="dataset-{created.id}-r1.jsonl"'
    assert exported_texts(bytes(exported.body)) == ("a",)
    assert latest.headers["content-disposition"] == f'attachment; filename="dataset-{created.id}-r2.jsonl"'
    assert exported_texts(bytes(latest.body)) == ("c",)


@pytest.mark.asyncio
async def test_creating_a_dataset_whose_insert_is_rejected_is_a_conflict() -> None:
    with pytest.raises(HTTPException) as error:
        await create_dataset(DatasetCreate(name="Refunds"), ADMIN, RejectingStore())
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_saving_loses_to_a_concurrent_save_of_the_same_revision() -> None:
    created: Final = await create_dataset_named(MemoryStore())

    with pytest.raises(HTTPException) as error:
        await save_revision(
            created.id, RevisionSave(base_revision=0, cases=(case("a"),)), ADMIN, RejectingStore(created)
        )
    assert error.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cases",
    (
        tuple(case(str(i)) for i in range(LENS_DATASET_MAX_CASES + 1)),
        (case("x" * (LENS_DATASET_MAX_CASE_CHARS + 1)),),
    ),
    ids=("too-many-cases", "case-too-large"),
)
async def test_saving_an_oversized_revision_is_rejected_and_writes_nothing(cases: tuple[DatasetCase, ...]) -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)

    with pytest.raises(HTTPException) as error:
        await save_revision(created.id, RevisionSave(base_revision=0, cases=cases), ADMIN, store)
    assert error.value.status_code == 422
    assert sorted(store.rows) == [(created.id, 0)]
