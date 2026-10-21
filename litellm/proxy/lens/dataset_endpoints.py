from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from litellm.constants import LENS_DATASET_TRACE_PAGE_SIZE
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper
from litellm.proxy.lens.dataset_repository import DatasetRepository, DatasetStore
from litellm.proxy.lens.datasets import build_cases, export_jsonl, included_cases, revision_cases, revision_problem
from litellm.proxy.lens.endpoints import Auth, repository, user_scope
from litellm.proxy.lens.models import (
    BuildRequest,
    BuildResult,
    Dataset,
    DatasetCase,
    DatasetCreate,
    DatasetSummary,
    EvalCases,
    Finding,
    RevisionSave,
    Scope,
)
from litellm.proxy.lens.repository import LensRepository, WriterDatabase
from litellm.proxy.lens.state import can_access
from litellm.proxy.tracing_endpoints import read_failure
from litellm.proxy.tracing_runtime import provide_receiver, require_receiver
from litellm.rust_bridge.trace.errors import TraceChanged
from litellm.rust_bridge.trace.generated.types import SpanDetail, Trace, TraceScope
from litellm.tracing import TraceReceiver

router: Final = APIRouter(prefix="/lens/datasets", tags=["Lens"])
ALL_TRACES: Final = TraceScope(all_teams=1, user_id="", team_ids=())


def dataset_store() -> DatasetStore:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Lens needs a connected Postgres database")
    return DatasetRepository(WriterDatabase(writer_wrapper(prisma_client.db)))


Datasets: TypeAlias = Annotated[DatasetStore, Depends(dataset_store)]
Lenses: TypeAlias = Annotated[LensRepository, Depends(repository)]
Receiver: TypeAlias = Annotated[TraceReceiver | None, Depends(provide_receiver)]
RevisionQuery: TypeAlias = Annotated[int | None, Query(ge=0)]


class ProxyDatasetReader:
    def __init__(self, receiver: TraceReceiver | None, lenses: LensRepository, scope: Scope) -> None:
        self.receiver: Final = receiver
        self.lenses: Final = lenses
        self.scope: Final = scope

    async def trace(self, trace_id: str, trace_ref: str) -> Trace | None:
        try:
            return await self._all_pages(trace_id, trace_ref)
        except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
            raise read_failure(error) from error

    async def _all_pages(self, trace_id: str, trace_ref: str) -> Trace | None:
        receiver: Final = require_receiver(self.receiver)
        first: Final = await receiver.get_trace(trace_id, ALL_TRACES, trace_ref, page_size=LENS_DATASET_TRACE_PAGE_SIZE)
        if first is None:
            return None
        spans = first["spans"]  # rebind-ok: accumulate spans across cursor pages
        cursor = first.get("next_cursor")  # rebind-ok: advance the trace cursor
        while cursor:
            page = await receiver.get_trace(trace_id, ALL_TRACES, trace_ref, cursor, LENS_DATASET_TRACE_PAGE_SIZE)
            if page is None:
                return None
            spans = (*spans, *page["spans"])
            cursor = page.get("next_cursor")
        return Trace(summary=first["summary"], agents=first["agents"], spans=spans, next_cursor=None)

    async def span(self, trace_id: str, span_id: str, trace_ref: str) -> SpanDetail | None:
        try:
            return await require_receiver(self.receiver).get_span(trace_id, span_id, ALL_TRACES, trace_ref)
        except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
            raise read_failure(error) from error

    async def findings(self, lens_id: str, ids: tuple[str, ...]) -> tuple[Finding, ...]:
        lens: Final = await self.lenses.get(lens_id)
        if lens is None or not can_access(self.scope, lens.scope):
            raise HTTPException(404, "Lens not found")
        wanted: Final = frozenset(ids)
        return tuple(f for f in lens.findings if f.id in wanted)


async def get_dataset(datasets: DatasetStore, dataset_id: str, scope: Scope, revision: int | None = None) -> Dataset:
    dataset: Final = await datasets.get(dataset_id, revision)
    if dataset is None or not can_access(scope, Scope(team_id=dataset.team_id)):
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.get("", response_model=tuple[DatasetSummary, ...])
async def list_datasets(auth: Auth, datasets: Datasets) -> tuple[DatasetSummary, ...]:
    scope: Final = user_scope(auth)
    return tuple(s.summary for s in await datasets.summaries() if can_access(scope, Scope(team_id=s.team_id)))


@router.post("", response_model=Dataset)
async def create_dataset(body: DatasetCreate, auth: Auth, datasets: Datasets) -> Dataset:
    user_scope(auth, write=True)
    now: Final = datetime.now(timezone.utc)
    dataset: Final = Dataset(
        id=str(uuid4()),
        name=body.name,
        agent_name=body.agent_name,
        team_id=auth.team_id or "",
        created_at=now,
        revision=0,
        created_by=auth.user_id or "",
        cases=(),
    )
    if not await datasets.insert(dataset, now):
        raise HTTPException(409, "Dataset already exists")
    return dataset


@router.post("/build", response_model=BuildResult)
async def build_dataset_cases(
    body: BuildRequest, auth: Auth, datasets: Datasets, lenses: Lenses, receiver: Receiver
) -> BuildResult:
    scope: Final = user_scope(auth)
    existing: Final[tuple[DatasetCase, ...]] = (
        (await get_dataset(datasets, body.dataset_id, scope)).cases if body.dataset_id else ()
    )
    return await build_cases(body, ProxyDatasetReader(receiver, lenses, scope), existing)


@router.get("/{dataset_id}", response_model=Dataset)
async def read_dataset(dataset_id: str, auth: Auth, datasets: Datasets, revision: RevisionQuery = None) -> Dataset:
    return await get_dataset(datasets, dataset_id, user_scope(auth), revision)


@router.post("/{dataset_id}/revisions", response_model=Dataset)
async def save_revision(dataset_id: str, body: RevisionSave, auth: Auth, datasets: Datasets) -> Dataset:
    latest: Final = await get_dataset(datasets, dataset_id, user_scope(auth, write=True))
    if body.base_revision != latest.revision:
        raise HTTPException(409, "Dataset changed, reload")
    cases: Final = revision_cases(body.cases)
    if problem := revision_problem(cases):
        raise HTTPException(422, problem)
    saved: Final = latest.model_copy(
        update=MappingProxyType({"revision": latest.revision + 1, "cases": cases, "created_by": auth.user_id or ""})
    )
    if not await datasets.insert(saved, datetime.now(timezone.utc)):
        raise HTTPException(409, "Dataset changed, reload")
    return saved


@router.get(
    "/{dataset_id}/export",
    response_class=Response,
    responses={200: {"content": {"application/x-ndjson": {}}}},
)
async def export_dataset(dataset_id: str, auth: Auth, datasets: Datasets, revision: RevisionQuery = None) -> Response:
    dataset: Final = await get_dataset(datasets, dataset_id, user_scope(auth), revision)
    return Response(
        content=export_jsonl(dataset.cases),
        media_type="application/x-ndjson",
        headers=MappingProxyType(
            {"Content-Disposition": f'attachment; filename="dataset-{dataset.id}-r{dataset.revision}.jsonl"'}
        ),
    )


@router.get("/{dataset_id}/revisions/{revision}/cases", response_model=EvalCases)
async def eval_cases(dataset_id: str, revision: int, auth: Auth, datasets: Datasets) -> EvalCases:
    dataset: Final = await get_dataset(datasets, dataset_id, user_scope(auth), revision)
    return EvalCases(dataset_id=dataset.id, revision=dataset.revision, cases=included_cases(dataset.cases))
